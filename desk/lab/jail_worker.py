"""Strategy jail: the child process that runs untrusted strategy code (spawned by desk/lab/jail.py, one per call).

Why a separate process: the evaluator (backtest, statistics, gates, ledger) lives in the parent. Code running here
cannot reach the parent's memory, so it cannot patch the scorer (desk.lab.stats.dsr etc.) or rewrite its own report.
Why one process per call: nothing survives between calls, so a strategy cannot remember the full sample from one call
and replay it in a later prefix call of the causality test.

Before any strategy code runs (see desk/lab/guard.py for the layers and their limits):
  * the input frame is decoded, then polars' file/cloud I/O and plugin entry points are deleted (they run in Rust)
  * Landlock + seccomp where the kernel supports them: reads limited to the Python installation and desk/lib,
    no writes, no exec/fork/sockets
  * an audit hook with the same file policy, plus no sockets, subprocesses, ctypes, or evaluator modules
  * the environment is empty except for thread caps: no API keys, no data paths

This file imports only the standard library (and desk.lab.guard, also stdlib-only) before the guards are up.

Wire protocol (binary, stdin -> stdout), each frame = 8-byte little-endian length + bytes:
  request : JSON header, payload (polars IPC of the input frame or b''), model blob (pickle made by a jail, or b'')
  response: JSON header, payload (float64 little-endian array, a model pickle, or b'')
"""
from __future__ import annotations

import json
import os
import struct
import sys

MAX_MODEL_BYTES = 64 * 1024 * 1024

# beyond guard.BLOCKED_MODULES: libraries with their own native file/network/JIT paths, and the evaluator itself
JAIL_EXTRA_BLOCKED = {
    'pyarrow', 'duckdb', 'fsspec', 'sqlite3', '_sqlite3', 'numba', 'llvmlite', 'h5py', 'tables', 'pandas',
    'pickletools', 'desk.lab', 'desk.pipeline', 'desk.company', 'desk.data', 'desk.live', 'desk.web', 'desk.config',
    'desk.research',
}


def _read_frame(stream) -> bytes:
    head = stream.read(8)
    if len(head) != 8:
        raise EOFError('truncated request')
    n = struct.unpack('<Q', head)[0]
    buf = stream.read(n)
    if len(buf) != n:
        raise EOFError('truncated request')
    return buf


def _write_frame(stream, data: bytes):
    stream.write(struct.pack('<Q', len(data)))
    stream.write(data)


def _neuter_polars(pl):
    """Delete polars' file/cloud I/O and plugin entry points: they run in Rust, where the audit hook cannot see them
    (Landlock still does on Linux). Constructors and in-memory operations are untouched."""
    import re
    rules = [
        (sys.modules.get('polars._plr') or sys.modules.get('polars.polars'),
         r'^(read_|write_|register_plugin|prepare_cloud|.*clipboard)'),
        (type(pl.DataFrame({'a': [0]})._df), r'^(read_|write_|serialize_|deserialize_)'),
        (type(pl.LazyFrame({'a': [0]})._ldf),
         r'^(new_from_(csv|ipc|parquet|ndjson|scan_lines|expand_paths|dataset_object)|scan_from_python_function|'
         r'sink_|serialize_|deserialize_)'),
        (type(pl.col('a')._pyexpr), r'^(serialize_|deserialize_)'),
    ]
    for target, pattern in rules:
        if target is None:
            continue
        for attr in [a for a in list(vars(target)) if re.match(pattern, a)]:
            try:
                delattr(target, attr)
            except Exception:
                pass

    def _blocked(*a, **k):
        raise PermissionError('file and network I/O are not available to strategy code')
    for mod in [m for n, m in list(sys.modules.items()) if n == 'polars' or n.startswith('polars.io')]:
        for attr in list(vars(mod)):
            if attr.startswith(('read_', 'scan_')):
                try:
                    setattr(mod, attr, _blocked)
                except Exception:
                    pass
    for cls in (pl.DataFrame, pl.LazyFrame, pl.Series):
        for attr in list(vars(cls)):
            if attr.startswith(('write_', 'sink_', 'serialize', 'deserialize')):
                try:
                    setattr(cls, attr, _blocked)
                except Exception:
                    pass


def _jsonable(x):
    import numpy as np
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f'value of type {type(x).__name__} is not JSON-serializable')


META_ATTRS = ('DESK', 'RESOLUTION', 'PARAMS', 'LOOKBACK', 'EXECUTION', 'TRAIN_DAYS', 'REFIT_DAYS', 'WARMUP_DAYS',
              'PERPS_EXECUTION', 'CARRY_EXECUTION')


def _as_array(np, pl, out, what):
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    elif isinstance(out, pl.DataFrame):
        raise ValueError(f'{what} must return a Series/array, not a DataFrame')
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(arr, dtype='<f8')



def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                         # strategy prints go to stderr, never into the protocol stream
    header = json.loads(_read_frame(stdin))
    payload = _read_frame(stdin)
    model_blob = _read_frame(stdin)

    import io
    import pickle
    import numpy as np
    import polars as pl
    df = pl.read_ipc(io.BytesIO(payload)) if payload else None
    del payload

    root = header['root']
    from desk.lab import guard                      # stdlib-only module, imported before any guard is up
    read_roots = list(header['python_roots']) + [os.path.join(root, 'desk', 'lib'),
                                                  os.path.join(root, 'desk', '__init__.py')]
    _neuter_polars(pl)
    import ctypes                                   # noqa: F401  (loads pythonapi now; lightgbm/xgboost need ctypes)
    import subprocess                               # noqa: F401  (scipy imports it at import time)
    import mmap                                     # noqa: F401  (joblib/sklearn need it; files stay guarded at open)
    if os.name == 'nt':
        try:
            import colorama                         # noqa: F401  (binds kernel32 by bare name at import on Windows)
        except Exception:
            pass
    try:
        import _posixsubprocess                     # its fork_exec starts programs without an audit event
        del _posixsubprocess.fork_exec
    except (ImportError, AttributeError):
        pass
    landlock = guard.install_landlock(read_roots, [])
    seccomp = guard.install_seccomp()
    kernel = '+'.join(x for x, on in (('landlock', landlock), ('seccomp', seccomp)) if on) or 'none'
    if header.get('require_kernel_guard') and not (landlock and seccomp):
        _write_frame(stdout, json.dumps({'ok': False, 'error': f'kernel guard required, but this machine only '
                                                               f'provides {kernel!r}'}).encode())
        _write_frame(stdout, b'')
        stdout.flush()
        return
    guard.install_audit_hook(read_roots, deny_roots=header.get('deny_roots', []),
                             blocked_modules=guard.BLOCKED_MODULES | JAIL_EXTRA_BLOCKED, what='strategy code')

    resp, out_bytes = {'ok': True, 'kernel_guard': kernel}, b''
    try:
        import types
        mod = types.ModuleType('jailed_strategy')
        mod.__dict__.update({'np': np, 'pl': pl})
        sys.modules['jailed_strategy'] = mod                # so fitted models defined in the module can be pickled
        exec(compile(header['code'], '<strategy>', 'exec'), mod.__dict__)
        op = header['op']
        if op == 'meta':
            meta = {k: getattr(mod, k) for k in META_ATTRS if hasattr(mod, k)}
            meta.update({'has_signal': callable(getattr(mod, 'signal', None)),
                         'has_fit': callable(getattr(mod, 'fit', None)),
                         'has_factor': callable(getattr(mod, 'factor', None))})
            resp['meta'] = json.loads(json.dumps(meta, default=_jsonable))
        elif op in ('signal', 'factor'):
            params = header.get('params') or {}
            model = pickle.loads(model_blob) if model_blob else None
            if op == 'factor':
                out = mod.factor(df, params)
            elif model_blob or header.get('fitted'):
                out = mod.signal(df, params, model)
            else:
                try:
                    out = mod.signal(df, params)
                except TypeError:
                    out = mod.signal(df, params, None)
            arr = _as_array(np, pl, out, op)
            resp['n'] = int(arr.shape[0])
            out_bytes = arr.tobytes()
        elif op == 'fit':
            blob = pickle.dumps(mod.fit(df, header.get('params') or {}))
            if len(blob) > MAX_MODEL_BYTES:
                raise ValueError(f'fitted model is {len(blob) >> 20} MB (limit {MAX_MODEL_BYTES >> 20} MB)')
            out_bytes = blob
        else:
            raise ValueError(f'unknown op {op!r}')
    except BaseException as e:                              # report, never crash silently
        resp = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:2000], 'kernel_guard': kernel}
        out_bytes = b''
    _write_frame(stdout, json.dumps(resp).encode())
    _write_frame(stdout, out_bytes)
    stdout.flush()


if __name__ == '__main__':
    main()
