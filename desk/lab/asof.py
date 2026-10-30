"""Structural look-ahead guard (cycle 1b).

Statistics alone pass a leaky strategy (arXiv 2608.27734: a next-bar leak sailed through Sharpe/DSR/bootstrap), so
look-ahead is enforced by the ENGINE, as a hard error, on every evaluation (base run, every parameter neighbour):

- the strategy's weights on the full panel are compared with its weights computed on AS-OF VIEWS: the panel
  physically truncated to rows with time <= cut, so the code cannot see anything after the cut;
- cuts are evenly spaced plus fresh random ones (unpredictable, so a leak cannot be tuned around fixed cuts) plus
  the second-to-last timestamp (catches next-bar leaks on the final decision);
- any weight at or before a cut that changes when later data is removed means it used data from after its
  decision time -> LookAheadError. There is no threshold, no gate to fail softly, no flag to skip it.

Full per-timestamp replay (one as-of view per decision time) is exact but O(T^2); the randomized cuts are exhaustive
at every row on or before each cut, and the l2 desk keeps its own prefix + future-perturbation test (also hard).
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl


class LookAheadError(RuntimeError):
    """A strategy's decision depended on data at or after its decision time."""


def _cuts(n_unique, n_cuts, seed):
    if n_unique < 3:
        return []
    rng = np.random.default_rng(seed if seed is not None else int.from_bytes(os.urandom(8), 'little'))
    even = np.linspace(1, n_unique - 2, max(n_cuts // 2, 1)).astype(int)
    rand = rng.integers(1, n_unique - 1, size=max(n_cuts - len(even), 1))
    return sorted(set(even.tolist()) | set(rand.tolist()) | {n_unique - 2})


def asof_guard(signal_fn, panel: pl.DataFrame, time_col: str, n_cuts=8, seed=None, tol=1e-6, label='strategy'):
    """Raise LookAheadError unless signal_fn(panel) restricted to rows <= cut equals signal_fn(as-of view at cut)."""
    if time_col not in panel.columns:
        raise LookAheadError(f'{label}: panel has no {time_col!r} column; cannot establish decision times')
    if len(panel) == 0:
        return {'ok': True, 'issues': [], 'rows': 0, 'cuts': 0, 'guard': 'asof'}
    if panel[time_col].null_count():
        raise LookAheadError(f'{label}: {panel[time_col].null_count()} rows without a timestamp')
    ts = panel[time_col].to_numpy()
    full = np.asarray(signal_fn(panel), dtype=float).reshape(-1)
    if len(full) != len(panel):
        raise LookAheadError(f'{label}: signal returned {len(full)} values for {len(panel)} rows')
    ut = np.unique(ts)
    cuts = _cuts(len(ut), n_cuts, seed)
    issues = []
    for i in cuts:
        cut = ut[i]
        mask = ts <= cut
        pre = np.asarray(signal_fn(panel.filter(pl.Series(mask))), dtype=float).reshape(-1)
        ref = full[mask]
        if len(pre) != len(ref):
            issues.append(f'as-of view at {cut}: signal length {len(pre)} != {len(ref)}')
            continue
        bad = ~(np.isclose(pre, ref, rtol=0.0, atol=tol) | (np.isnan(pre) & np.isnan(ref)))
        if bad.any():
            first = ts[mask][int(np.argmax(bad))]
            issues.append(f'as-of view at {cut}: {int(bad.sum())} weights at/before the cut changed when later data was '
                          f'removed (first at {first}); the strategy reads data from after its decision time')
    if issues:
        raise LookAheadError(f'{label}: look-ahead detected. ' + ' | '.join(issues[:3]))
    return {'ok': True, 'issues': [], 'rows': len(panel), 'cuts': len(cuts), 'guard': 'asof'}
