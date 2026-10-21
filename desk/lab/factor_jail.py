"""Run agent-authored factor modules (`def factor(panel, params)`) inside the OS jail instead of exec() in-process.

Used by panel_factor.load_factor and combo.factor_matrix. The host writes the code and each panel to a private
temp dir; the jailed worker (_factor_worker.py) execs the module with {'np', 'pl'} preloaded, calls factor() and
writes the values back as .npy. Only that temp dir is writable inside the jail.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from . import jail

WORKER = Path(__file__).resolve().parent / '_factor_worker.py'
TIMEOUT_S = int(os.getenv('DESK_FACTOR_TIMEOUT_S', '600'))
MEM_MB = int(os.getenv('DESK_FACTOR_MB', '4000'))
SAFE = ('PATH', 'LANG', 'LC_ALL')


def _env():
    env = {k: v for k, v in os.environ.items() if k in SAFE}
    env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'POLARS_MAX_THREADS': '3', 'OMP_NUM_THREADS': '2', 'NUMBA_NUM_THREADS': '2'})
    return env


def _inproc(code):                                        # DESK_SANDBOX_UNSAFE=1 local-dev path only
    import polars as pl
    ns = {'np': np, 'pl': pl}
    exec(compile(code, '<factor>', 'exec'), ns)
    return ns


def run(code: str, panels=(), params=None, timeout_s=None):
    """Exec `code` in the jail; return (meta, [values per panel]). meta = {'has_factor', 'PARAMS', 'DESK'}.
    params=None uses the module's own PARAMS."""
    # already jailed (e.g. inside a jailed run_job harness): exec here, the outer jail is the boundary
    if jail.inside() or (not jail.available() and jail.unsafe_opt_in()):
        ns = _inproc(code)
        f = ns.get('factor')
        meta = {'has_factor': callable(f), 'PARAMS': dict(ns.get('PARAMS', {})), 'DESK': ns.get('DESK')}
        p = meta['PARAMS'] if params is None else params
        return meta, [_arr(f(pn, dict(p))) for pn in panels]
    jail.require()
    with tempfile.TemporaryDirectory(prefix='desk_factor_') as td:
        io = Path(td)
        (io / 'factor.py').write_text(code, encoding='utf-8')
        (io / 'params.json').write_text(json.dumps(params))
        for i, pn in enumerate(panels):
            pn.write_parquet(io / f'panel_{i}.parquet')
        lim = jail.limits(timeout_s=timeout_s or TIMEOUT_S, mem_mb=MEM_MB)
        argv, pre = jail.python_cmd([str(WORKER), str(io), str(len(panels))], rw=[io], cwd=io, lim=lim)
        try:
            r = subprocess.run(argv, env=_env(), preexec_fn=pre, capture_output=True, timeout=timeout_s or TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise TimeoutError('factor module timed out in the sandbox') from None
        meta_p = io / 'meta.json'
        if r.returncode != 0 or not meta_p.exists():
            err = r.stderr.decode('utf-8', 'replace')[-2000:]
            raise RuntimeError(f'factor module failed in the sandbox (exit {r.returncode}): {err}')
        meta = json.loads(meta_p.read_text())
        if meta.get('error'):
            raise ValueError(meta['error'])
        return meta, [np.load(io / f'values_{i}.npy') for i in range(len(panels))]


def _arr(v):
    import polars as pl
    if isinstance(v, pl.Series):
        v = v.to_numpy()
    return np.asarray(v, dtype=float).reshape(-1)


class JailedFactor:
    """Callable stand-in for an agent factor() that executes in the jail on every call."""

    def __init__(self, code):
        self.code = code

    def __call__(self, panel, params):
        return run(self.code, [panel], params=dict(params))[1][0]


def load(code):
    meta, _ = run(code)
    if not meta.get('has_factor'):
        raise ValueError('factor module must define factor(panel, params)')
    return types.SimpleNamespace(factor=JailedFactor(code), PARAMS=dict(meta.get('PARAMS') or {}), DESK=meta.get('DESK'))
