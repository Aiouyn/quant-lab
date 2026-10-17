from __future__ import annotations
"""Embargoed day-level CV generator (fixes the classic leak: random/row-level splits across day edges where rolling windows and labels cross). purged_day_folds(days, n_splits, mode='expanding'|'blocked', embargo_days, min_train_days) -> [{'train': day_ids, 'test': day_ids}]. expanding = walk-forward, train strictly before test minus embargo (use in fit()-on-train ML: volregime T-01d22f, midfreq T-89c9ce); blocked = K contiguous test blocks with train purged within embargo_days of ANY test day both sides (set embargo >= label horizon in days). fold_report() audits non-overlap/causality/coverage and flags leaky splits. Row-level stats inflate t and flip signs (L-487548) — this is the tool that enforces episode/day-level discipline in CV."""
import numpy as np
import polars as pl
"""Purged, embargoed day-level cross-validation folds for strategy fit()/research ML.

Two modes (day arrays are sorted unique day ids, epoch days):
  expanding : walk-forward — train = days strictly BEFORE the test block, minus `embargo_days` gap.
              Causal by construction: a model fitted this way never sees days after its test block.
              Use for honest time-series validation of fit() models (default, safest).
  blocked   : blocked K-fold WITH purging — each fold takes a contiguous test block; train = all other
              days excluding any day within `embargo_days` of ANY test day (both sides). Use when you
              need K test blocks across the whole sample (more test coverage) and the label horizon
              leaks across day edges (set embargo_days >= label horizon in days).

Row selection: folds return day-id arrays; select rows with np.isin(row_day_ids, fold['train']).
NEVER split rows randomly — labels/features span day edges and rolling windows cross them.
"""

import numpy as np


def purged_day_folds(days, n_splits=5, mode='expanding', embargo_days=1, min_train_days=10):
    """days: 1d array of UNIQUE sorted day ids. Returns list of {'train': days, 'test': days} arrays."""
    d = np.unique(np.asarray(days))
    if len(d) < 3:
        raise ValueError('need >=3 unique days')
    folds = []

    if mode == 'expanding':
        n = len(d)
        start = max(min_train_days, n // (n_splits + 1))
        blocks = np.array_split(np.arange(start, n), n_splits)
        for b in blocks:
            if len(b) == 0:
                continue
            te = d[b]
            tr = d[d < te[0] - embargo_days]
            if len(tr) < min_train_days or len(tr) == 0 or len(np.intersect1d(tr, te)):
                continue
            folds.append({'train': tr, 'test': te})
    elif mode == 'blocked':
        n = len(d)
        blocks = np.array_split(np.arange(n), n_splits)
        for b in blocks:
            if len(b) == 0:
                continue
            te = d[b]
            lo, hi = te[0] - embargo_days, te[-1] + embargo_days
            tr = d[(d < lo) | (d > hi)]
            if len(tr) < min_train_days or len(np.intersect1d(tr, te)):
                continue
            folds.append({'train': tr, 'test': te})
    else:
        raise ValueError("mode must be 'expanding' or 'blocked'")
    if not folds:
        raise ValueError('no valid folds — reduce n_splits/embargo or min_train_days')
    return folds


def fold_report(days, folds):
    """Sanity/leakage audit of folds -> dict (no_overlap must be True; expanding_causal True for walk-forward)."""
    d = np.unique(np.asarray(days))
    no_overlap = all(len(np.intersect1d(f['train'], f['test'])) == 0 for f in folds)
    direction = bool(folds) and all(np.all(f['train'] < f['test'][0]) for f in folds)
    causal = bool(direction and all(int(f['test'][0]) - int(f['train'][-1]) >= 1 for f in folds))
    tests = np.unique(np.concatenate([f['test'] for f in folds])) if folds else np.array([])
    return {'no_overlap': bool(no_overlap),
            'train_before_test': direction,       # True: every fold is walk-forward (train strictly earlier)
            'expanding_causal': causal,           # True: also respects the embargo gap
            'test_days_covered': int(len(tests)),
            'test_frac': round(len(tests) / max(len(d), 1), 3),
            'n_folds': len(folds)}
