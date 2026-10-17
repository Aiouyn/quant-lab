"""Paper-session supervision: status, live-vs-backtest drift bands, process launch."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..config import ROOT


def _expected(manifest_expected):
    d = (manifest_expected or {})
    daily = d.get('val_daily') or {}
    r = np.array(daily.get('ret') or [])
    if len(r) < 5:
        return None
    return float(r.mean()), float(r.std(ddof=1))


def paper_status(settings):
    out = []
    root = settings.home / 'paper'
    if not root.exists():
        return {'sessions': [], 'note': 'no paper sessions yet'}
    for d in sorted(root.iterdir(), reverse=True)[:30]:
        st = d / 'status.json'
        if not st.exists():
            continue
        s = json.loads(st.read_text())
        hourly = s.get('kind') in ('carry', 'spot_carry', 'funding_flip', 'hl_carry')     # these engines write status once an hour
        alive = (not s.get('killed')) and (time.time() - s['updated'] < (3900 if hourly else 180))   # killed flag set by operator kill (K-013458)
        days = s['hours'] / 24
        drift = []
        for sl in s.get('sleeves', []):
            bpath = settings.home / 'bundles' / sl['id'] / 'manifest.json'
            exp = _expected(json.loads(bpath.read_text()).get('expected')) if bpath.exists() else None
            if exp and days >= 0.5:
                mu, sd = exp
                realized = sl['equity'] / sl['unit']
                z = (realized - mu * days) / (sd * math.sqrt(days) + 1e-12)
                drift.append({'sleeve': sl['id'], 'realized_units': round(realized, 4), 'expected_units': round(mu * days, 4),
                              'z': round(z, 2), 'flag': 'DRIFT' if z < -2 else 'ok'})
        out.append({'session': s['id'], 'kind': s['kind'], 'target': s['target'], 'alive': alive, 'hours': round(s['hours'], 2),
                    'pnl_usd': round(s['pnl'], 2), 'funding_usd': round(s.get('funding', 0.0), 2), 'fees_usd': round(s.get('fees', 0.0), 2),
                    'n_positions': len(s.get('positions') or s.get('open') or {}),
                    'sleeves': [{k: v for k, v in sl.items() if k in ('id', 'status', 'target', 'pos_btc', 'equity', 'fills')} for sl in s.get('sleeves', [])],
                    'drift': drift})
    return {'sessions': out}


def start_paper(settings, target, hours=168):
    logs = settings.home / 'paper' / 'launch'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    flags = 0
    if sys.platform == 'win32':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with open(logs / f'{target}_{stamp}.log', 'w') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'desk', 'paper', target, '--hours', str(hours)], cwd=str(ROOT),
                             stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return {'started': True, 'pid': p.pid, 'target': target, 'hours': hours, 'log': str(logs / f'{target}_{stamp}.log')}
