"""Jailed worker: exec a factor module and evaluate it on parquet panels (see factor_jail.py)."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import polars as pl

io, n = Path(sys.argv[1]), int(sys.argv[2])
meta = {}
try:
    ns = {'np': np, 'pl': pl}
    exec(compile((io / 'factor.py').read_text(encoding='utf-8'), '<factor>', 'exec'), ns)
    f = ns.get('factor')
    meta = {'has_factor': callable(f), 'PARAMS': dict(ns.get('PARAMS', {})), 'DESK': ns.get('DESK')}
    params = json.loads((io / 'params.json').read_text())
    params = meta['PARAMS'] if params is None else params
    for i in range(n):
        v = f(pl.read_parquet(io / f'panel_{i}.parquet'), dict(params))
        if isinstance(v, pl.Series):
            v = v.to_numpy()
        np.save(io / f'values_{i}.npy', np.asarray(v, dtype=float).reshape(-1))
except Exception as e:
    meta['error'] = f'{type(e).__name__}: {e}'
    traceback.print_exc()
(io / 'meta.json').write_text(json.dumps(meta, default=str))
