"""Portfolio construction from approved strategies: de-duplicate, shrink covariance, risk-balance, cap."""
from __future__ import annotations

import json
import time
import uuid

import numpy as np


def _returns(strategy):
    """Aligned daily returns from sealed validation + holdout (out-of-sample streams only)."""
    out = {}
    for key in ('val_report', 'holdout_report'):
        r = strategy.get(key) or {}
        d = r.get('daily') or {}
        for day, ret in zip(d.get('day', []), d.get('ret', [])):
            out[int(day)] = float(ret)
    return out


def allocate(R, names, cap=0.35, corr_limit=0.85, sharpe=None):
    """R: T x N daily returns. Returns weights dict (sum <= 1; remainder is cash)."""
    from sklearn.covariance import LedoitWolf
    T, N = R.shape
    keep = list(range(N))
    if N > 1:
        C = np.corrcoef(R.T)
        order = np.argsort(-(np.asarray(sharpe) if sharpe is not None else R.mean(0) / (R.std(0) + 1e-12)))
        keep = []
        for i in order:
            if all(abs(C[i, j]) < corr_limit for j in keep):
                keep.append(int(i))
    Rk = R[:, keep]
    if len(keep) == 1:
        w = np.array([min(1.0, cap * 2)])
    else:
        cov = LedoitWolf().fit(Rk).covariance_
        # equal risk contribution via simple fixed-point iterations
        w = np.ones(len(keep)) / len(keep)
        for _ in range(500):
            mrc = cov @ w
            rc = w * mrc
            w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
            w = np.clip(w / w.sum(), 0, cap)
            w = w / w.sum()
        w = np.minimum(w, cap)
    weights = {names[k]: round(float(x), 4) for k, x in zip(keep, w)}
    return weights


def build_portfolio(settings, ids=None, author='cli', rationale='', ledger=None, max_members=8):
    from ..company.ledger import Ledger
    L = ledger or Ledger(settings.ledger)
    if not ids:
        ids = [r['id'] for r in L.q("SELECT id FROM strategies WHERE status IN ('approved','paper') ORDER BY updated DESC LIMIT ?", (max_members,))]
    strats = [L.strategy(i) for i in ids]
    strats = [s for s in strats if s and s['status'] in ('approved', 'paper')]
    if not strats:
        return {'error': 'no approved strategies among the given ids'}
    streams = [_returns(s) for s in strats]
    days = sorted(set().union(*[set(x) for x in streams]))
    R = np.array([[st.get(d, 0.0) for st in streams] for d in days])
    sh = [np.mean(R[:, i]) / (np.std(R[:, i]) + 1e-12) for i in range(R.shape[1])]
    w = allocate(R, [s['id'] for s in strats], sharpe=sh)
    port = R[:, [ [s['id'] for s in strats].index(k) for k in w ]] @ np.array(list(w.values()))
    from ..lab.stats import sharpe, max_drawdown
    report = {'days': len(days), 'sharpe': round(sharpe(port), 3), 'mean_daily_bps': round(float(port.mean() * 1e4), 3),
              'max_drawdown': round(max_drawdown(np.cumsum(port)), 4), 'members': len(w),
              'avg_pair_corr': round(float(np.mean(np.corrcoef(R.T)[np.triu_indices(R.shape[1], 1)])), 3) if R.shape[1] > 1 else None}
    pid = 'P-' + uuid.uuid4().hex[:6]
    d = settings.home / 'portfolios' / pid
    d.mkdir(parents=True, exist_ok=True)
    manifest = {'id': pid, 'weights': w, 'bundles': {k: str(settings.home / 'bundles' / k) for k in w}, 'report': report,
                'rationale': rationale, 'author': author, 'created': time.time(), 'mode': 'PAPER_ONLY'}
    (d / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    L.x('INSERT INTO portfolios VALUES(?,?,?,?,?,?,?)', (pid, author, json.dumps(list(w)), json.dumps(w), json.dumps(report), 'built', time.time()))
    L.event(author, 'PORTFOLIO', {'id': pid, 'weights': w, 'report': report})
    return {'id': pid, 'weights': w, 'report': report, 'path': str(d)}
