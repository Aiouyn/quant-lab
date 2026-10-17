"""Frozen, hashed strategy bundles for Hyperliquid paper trading."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def export_bundle(settings, strategy: dict, cost: dict) -> Path:
    from ..lab.strategy import load_strategy
    from ..data.store import FeatureStore
    d = settings.home / 'bundles' / strategy['id']
    d.mkdir(parents=True, exist_ok=True)
    code = strategy['code']
    (d / 'strategy.py').write_text(code, encoding='utf-8')
    mod = load_strategy(code)
    model_hash = None
    if mod.FITTED:
        store = FeatureStore(settings)
        days = [x for k in ('dev', 'val', 'holdout') for x in store.days(k)]
        model = mod.fit(store.load(days[-mod.TRAIN_DAYS:], mod.RESOLUTION), dict(mod.PARAMS))
        blob = pickle.dumps(model)
        (d / 'model.pkl').write_bytes(blob)
        model_hash = sha(blob)
    v, h = strategy.get('val_report') or {}, strategy.get('holdout_report') or {}
    manifest = {
        'id': strategy['id'], 'name': strategy['name'], 'author': strategy['author'], 'pod': strategy['pod'],
        'resolution': mod.RESOLUTION, 'lookback': mod.LOOKBACK, 'params': mod.PARAMS, 'fitted': mod.FITTED,
        'code_sha256': sha(code.encode()), 'model_sha256': model_hash, 'cost_model': cost,
        'expected': {'val': v.get('summary'), 'val_daily': v.get('daily'), 'holdout': h.get('summary'),
                     'holdout_daily': h.get('daily')},
        'exported': time.time(), 'mode': 'PAPER_ONLY',
    }
    (d / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    return d


def load_bundle(path: Path):
    from ..lab.strategy import load_strategy
    path = Path(path)
    m = json.loads((path / 'manifest.json').read_text())
    code = (path / 'strategy.py').read_text(encoding='utf-8')
    if sha(code.encode()) != m['code_sha256']:
        raise ValueError('bundle code hash mismatch (edited after export)')
    model = None
    if m.get('fitted'):
        blob = (path / 'model.pkl').read_bytes()
        if sha(blob) != m['model_sha256']:
            raise ValueError('bundle model hash mismatch')
        model = pickle.loads(blob)
    return m, load_strategy(code), model
