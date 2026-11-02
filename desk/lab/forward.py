"""Forward verdicts: the firm's only claim to profit.

The forward period is market data that did not exist when a strategy was approved, so nobody could have tuned on it.
The question for each approved strategy is whether its forward daily returns show an edge, and the earlier rule
(Sharpe >= 0.5 over 60 days) could not answer it: over 60 days the standard error of an annualised Sharpe ratio is
about 2, so a strategy with no edge at all passed that bar about 40% of the time.

This module replaces it with a pre-registered test.

Per strategy (forward_verdict):
- Looks happen only at fixed forward lengths LOOKS = (60, 120, 250) days, and each look uses exactly the first `look`
  days, so checking every day between looks changes nothing.
- CONFIRMED at a look if the one-sided t-test of mean daily return > 0 has p <= ALPHA / (len(LOOKS) * m), where m is
  the number of strategies the firm has ever sent to the forward test (Bonferroni over looks and strategies), and the
  mean without the 5 best days is still positive.
- FAILED at the 120-day look if the mean is not positive (futility), or at the last look if it never confirmed.
- RUNNING otherwise. Every report states the Sharpe the test can detect at the next look, so a long RUNNING streak
  is expected rather than a surprise.

For the whole firm (firm_forward): one equal-weight book of every strategy sent to the forward test, including ones
that later failed (dropping losers after the fact would be selection), tested once with the same looks and m = 1. This
is the single number that says whether the firm makes money.

Part of the sealed kernel.
"""
from __future__ import annotations

import math

import numpy as np

LOOKS = (60, 120, 250)
FUTILITY_LOOK = 120
ALPHA = 0.05
DAYS_PER_YEAR = {'l2': 365, 'perps': 365, 'carry': 365, 'pred': 365, 'macro': 252}


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _norm_isf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _norm_sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _t_stat(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if not sd:
        return 0.0 if float(np.mean(x)) <= 0 else float('inf')
    return float(np.mean(x) / sd * math.sqrt(len(x)))


def _ex_top(x: np.ndarray, k: int = 5) -> float:
    if len(x) <= k:
        return float('nan')
    return float(np.mean(np.sort(x)[:-k]))


def forward_verdict(daily_ret, n_strategies: int = 1, desk: str = 'perps') -> dict:
    """daily_ret: forward daily returns in time order. n_strategies: how many strategies the firm has forward-tested."""
    x = np.asarray([r for r in ([] if daily_ret is None else daily_ret) if r is not None and math.isfinite(r)], float)
    n, m = len(x), max(1, int(n_strategies))
    ann = DAYS_PER_YEAR.get(desk, 365)
    p_crit = ALPHA / (len(LOOKS) * m)
    z_crit = _norm_isf(p_crit)
    done = [L for L in LOOKS if n >= L]
    nxt = next((L for L in LOOKS if n < L), None)
    out = {'days': n, 'tests_counted': m, 'p_critical': round(p_crit, 6), 'looks': list(LOOKS),
           'detectable_sharpe_next_look': round(z_crit * math.sqrt(ann / nxt), 2) if nxt else None,
           'sharpe_to_date': round(float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(ann)), 3)
           if n > 1 and np.std(x, ddof=1) > 0 else None}
    if not done:
        return {**out, 'verdict': 'RUNNING', 'reason': f'first look at {LOOKS[0]} forward days'}
    for look in done:
        seg = x[:look]
        t = _t_stat(seg)
        p = _norm_sf(t) if math.isfinite(t) else 0.0
        out.update({'look': look, 't': round(t, 3) if math.isfinite(t) else None, 'p': round(p, 6),
                    'mean_ex_top5_bps': round(_ex_top(seg) * 1e4, 3)})
        if p <= p_crit and _ex_top(seg) > 0:
            return {**out, 'verdict': 'FORWARD_CONFIRMED',
                    'reason': f'p {p:.2g} <= {p_crit:.2g} on the first {look} forward days'}
        if look >= FUTILITY_LOOK and float(np.mean(seg)) <= 0:
            return {**out, 'verdict': 'FORWARD_FAILED', 'reason': f'mean forward return not positive after {look} days'}
    if n >= LOOKS[-1]:
        return {**out, 'verdict': 'FORWARD_FAILED', 'reason': f'not confirmed by the last look ({LOOKS[-1]} days)'}
    return {**out, 'verdict': 'RUNNING', 'reason': f'next look at {nxt} days'}


def firm_book(reports: list[dict]) -> tuple[list[int], list[float]]:
    """Equal-weight daily book over every forward report: on each day, the mean return of the strategies trading."""
    by_day: dict[int, list[float]] = {}
    for rep in reports:
        d = (rep or {}).get('daily') or {}
        for day, r in zip(d.get('day') or [], d.get('ret') or []):
            if r is not None and math.isfinite(r):
                by_day.setdefault(int(day), []).append(float(r))
    days = sorted(by_day)
    return days, [float(np.mean(by_day[d])) for d in days]


def firm_forward(reports: list[dict]) -> dict:
    days, ret = firm_book(reports)
    desks = {(r or {}).get('desk') for r in reports}
    desk = 'macro' if desks == {'macro'} else 'perps'
    v = forward_verdict(ret, n_strategies=1, desk=desk)
    return {**v, 'strategies': len(reports), 'pnl_units': round(float(np.sum(ret)), 4) if ret else 0.0}
