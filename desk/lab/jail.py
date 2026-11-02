"""Host side of the strategy jail: run untrusted strategy / factor code in a fresh guarded child process per call.

    mod = load_jailed(code)            # metadata only (DESK, PARAMS, ...); the code never runs in this process
    mod.signal(df, params, model)      # -> np.ndarray float64, computed in a jail on a copy of df
    m = mod.fit(train_df, params)      # -> JailedModel: an opaque pickle that is only ever unpickled inside a jail
    mod.factor(panel, params)          # factor modules (register_factor)

Everything that comes back from a jail is parsed as data (JSON + raw float64 bytes). The host never unpickles
anything a jail produced. See desk/lab/jail_worker.py for the guards inside the child.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from ..config import ROOT

WORKER = Path(__file__).resolve().parent / 'jail_worker.py'
DEFAULT_TIMEOUT_S = int(os.getenv('DESK_JAIL_TIMEOUT_S', '900'))
_ENV_KEEP = ('SYSTEMROOT', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'NUMBER_OF_PROCESSORS',
             'PROCESSOR_ARCHITECTURE', 'LANG', 'LC_ALL')


class StrategyError(ValueError):
    pass


class JailError(StrategyError):
    """The strategy failed, broke a jail rule, or the jail itself failed. The message is safe to show agents."""


class JailedModel:
    """A fitted model as produced inside a jail: opaque bytes, never unpickled in a trusted process."""
    __slots__ = ('blob',)

    def __init__(self, blob: bytes):
        self.blob = bytes(blob)

    def __reduce__(self):                      # bundles pickle the wrapper; unpickling it only restores the bytes
        return (JailedModel, (self.blob,))

    def __repr__(self):
        return f'JailedModel({len(self.blob)} bytes, sha256={hashlib.sha256(self.blob).hexdigest()[:12]})'


def _python_roots():
    from .guard import python_roots
    return python_roots(exclude=[str(ROOT)] + _deny_roots())


def _deny_roots():
    out = {str(ROOT / 'results'), str(ROOT / 'data'), str(ROOT / '.git'), str(ROOT / '.env')}
    try:
        from ..config import SETTINGS
        for attr in ('home', 'store', 'artifacts', 'data_dir'):
            v = getattr(SETTINGS, attr, None)
            if v:
                out.add(str(v))
    except Exception:
        pass
    for k in ('DESK_STORE_ROOT', 'DESK_CARRY_ROOT', 'RESULTS_DIR', 'OKX_DATA_DIR'):
        if os.getenv(k):
            out.add(os.getenv(k))
    return sorted(out)


def _env(home=None):
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP}
    if home:                                   # libraries call Path.home(); give them the empty jail directory
        env.update({'HOME': str(home), 'USERPROFILE': str(home)})
    env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'POLARS_MAX_THREADS': os.getenv('DESK_JAIL_THREADS', '2'), 'OMP_NUM_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'NUMBA_DISABLE_JIT': '1'})
    return env


_BOOT = ('import sys; sys.path[:] = {path!r}; import runpy; runpy.run_path({worker!r}, run_name="__main__")')


def _frame(b: bytes) -> bytes:
    return struct.pack('<Q', len(b)) + b


def _unframe(buf: bytes, pos: int):
    if pos + 8 > len(buf):
        raise JailError('strategy process returned a truncated response')
    n = struct.unpack('<Q', buf[pos:pos + 8])[0]
    if pos + 8 + n > len(buf):
        raise JailError('strategy process returned a truncated response')
    return buf[pos + 8:pos + 8 + n], pos + 8 + n


def _ipc(df: pl.DataFrame | None) -> bytes:
    if df is None:
        return b''
    buf = io.BytesIO()
    df.write_ipc(buf, compression='uncompressed')
    return buf.getvalue()


def call(op: str, code: str, df: pl.DataFrame | None = None, params=None, model: JailedModel | None = None,
         fitted: bool = False, timeout_s: int | None = None):
    """One jailed call. Returns (header: dict, payload: bytes). Raises JailError on any failure."""
    header = {'op': op, 'code': code, 'params': params, 'fitted': bool(fitted), 'root': str(ROOT),
              'python_roots': _python_roots(), 'deny_roots': _deny_roots(),
              'require_kernel_guard': os.getenv('DESK_JAIL_REQUIRE_KERNEL') == '1'}
    try:
        hb = json.dumps(header).encode()
    except TypeError as e:
        raise JailError(f'params must be JSON-serializable: {e}')
    req = _frame(hb) + _frame(_ipc(df)) + _frame(model.blob if model is not None else b'')
    worker_path = [str(ROOT)] + [p for p in sys.path if p and os.path.realpath(p) != os.path.realpath(ROOT)]
    boot = _BOOT.format(path=worker_path, worker=str(WORKER))
    with tempfile.TemporaryDirectory(prefix='jail_') as cwd:     # empty working directory
        try:
            proc = subprocess.run([sys.executable, '-E', '-B', '-c', boot], input=req, capture_output=True,
                                  cwd=cwd, env=_env(cwd), timeout=timeout_s or DEFAULT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise JailError(f'strategy {op} timed out after {timeout_s or DEFAULT_TIMEOUT_S}s')
    out = proc.stdout
    try:
        hdr_b, pos = _unframe(out, 0)
        payload, _ = _unframe(out, pos)
        resp = json.loads(hdr_b)
    except (JailError, ValueError):
        tail = proc.stderr.decode('utf-8', 'replace')[-1500:]
        raise JailError(f'strategy process exited without a valid response (code {proc.returncode}). {tail}'.strip())
    if not isinstance(resp, dict) or not resp.get('ok'):
        raise JailError(str(resp.get('error') if isinstance(resp, dict) else resp)[:2000])
    return resp, payload


def _array(resp, payload, n_rows, what):
    if not isinstance(resp.get('n'), int) or len(payload) != 8 * resp['n']:
        raise JailError(f'{what} returned a malformed array')
    arr = np.frombuffer(payload, dtype='<f8').astype(np.float64)
    if n_rows is not None and len(arr) != n_rows:
        raise JailError(f'{what} length {len(arr)} != rows {n_rows}')
    return arr


class JailedStrategy:
    """Looks like the old exec'd module to callers (attributes + signal/fit/factor), but holds no strategy code
    objects: every call runs in a fresh jail."""

    def __init__(self, code: str, meta: dict):
        self._code = code
        self._meta = dict(meta)
        for k, v in meta.items():
            if k.isupper():
                setattr(self, k, v)
        self.signal = self._signal if meta.get('has_signal') else None
        self.fit = self._fit if meta.get('has_fit') else None
        self.factor = self._factor if meta.get('has_factor') else None

    def _signal(self, df, params=None, model=None):
        if model is not None and not isinstance(model, JailedModel):
            raise JailError('model must come from fit() of a jailed strategy')
        resp, payload = call('signal', self._code, df, params if params is not None else dict(self.PARAMS),
                             model, fitted=bool(getattr(self, 'FITTED', False)))
        return _array(resp, payload, len(df), 'signal')

    def _fit(self, df, params=None):
        resp, payload = call('fit', self._code, df, params if params is not None else dict(self.PARAMS))
        return JailedModel(payload)

    def _factor(self, panel, params=None):
        resp, payload = call('factor', self._code, panel, params if params is not None else dict(self.PARAMS))
        return _array(resp, payload, len(panel), 'factor')


@functools.lru_cache(maxsize=256)
def _meta_cached(code: str) -> str:
    resp, _ = call('meta', code, timeout_s=120)
    return json.dumps(resp.get('meta') or {})


def load_jailed(code: str) -> JailedStrategy:
    """Execute the module body in a jail once (cached by code) and return a proxy carrying its metadata."""
    return JailedStrategy(code, json.loads(_meta_cached(code)))
