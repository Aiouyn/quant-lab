"""Standardized factor evaluation: the common currency of the research organization.

Every factor is a causal function of past data. We measure how it predicts forward mid returns
(starting one row AFTER the decision, i.e. respecting latency) across horizons, days and regimes, and
compare its tradable spread with the round-trip cost hurdle.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as st

from .backtest import RES_SECONDS

DEFAULT_HORIZONS = (10, 60, 300, 900, 3600, 14400)
ROUND_TRIP_COST_BPS = 2 * (4.5 + 0.5) + 0.01          # taker in/out
MAKER_ROUND_TRIP_BPS = 2 * 1.5                         # maker in/out (before adverse selection / missed fills)
TRADABLE_MIN_H = 300                                   # horizons shorter than 5 min cannot pay for a round trip


def forward_returns(df: pl.DataFrame, horizons_s, resolution):
    """bps return from the row after t to the row after t+h (research labels; never strategy input)."""
    res = RES_SECONDS[resolution]
    mid = df['mid'].to_numpy().astype(np.float64)
    ts = df['ts'].to_numpy()
    n = len(mid)
    out = {}
    for h in horizons_s:
        k = max(1, int(round(h / res)))
        r = np.full(n, np.nan)
        if n > k + 1:
            a = mid[1:n - k]; b = mid[1 + k:]
            r[:n - k - 1] = (b / a - 1) * 1e4
            # invalidate windows that span gaps
            span = np.full(n, np.inf); span[:n - k - 1] = ts[1 + k:] - ts[1:n - k]
            r[span > (k * res * 1000 + 10_000)] = np.nan
        out[h] = r
    return out


def evaluate_factor(df: pl.DataFrame, values, resolution='1m', horizons_s=DEFAULT_HORIZONS):
    x = np.asarray(values, dtype=np.float64)
    if len(x) != len(df):
        raise ValueError('factor length mismatch')
    res = RES_SECONDS[resolution]
    fwd = forward_returns(df, horizons_s, resolution)
    day = df['ts'].to_numpy() // 86_400_000
    mid = df['mid'].to_numpy()
    # regime: trailing 1h realized vol tercile (causal)
    lr = np.r_[0.0, np.diff(np.log(mid))]
    w = max(2, int(3600 / res))
    rv = pl.Series(lr).rolling_std(w).to_numpy()
    finite = np.isfinite(rv)
    q1, q2 = (np.nanquantile(rv[finite], [1 / 3, 2 / 3]) if finite.sum() > 10 else (np.nan, np.nan))
    regime = np.where(rv <= q1, 0, np.where(rv <= q2, 1, 2))
    report = {'rows': int(len(x)), 'days': int(len(np.unique(day))), 'coverage': float(np.mean(np.isfinite(x) & (x != 0))),
              'horizons': {}}
    for h, y in fwd.items():
        if h < res:
            continue                                        # horizon shorter than the bar: not measurable
        k = max(1, int(round(h / res)))
        sel = np.arange(0, len(x), k)                       # non-overlapping samples
        xs, ys, ds, rs = x[sel], y[sel], day[sel], regime[sel]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys, ds, rs = xs[ok], ys[ok], ds[ok], rs[ok]
        if len(xs) < 30 or np.std(xs) == 0:
            report['horizons'][h] = {'n': int(len(xs)), 'ic': None}
            continue
        ic = float(np.corrcoef(xs, ys)[0, 1])
        ric = float(st.spearmanr(xs, ys).statistic)
        daily = []
        per_day = max(1, len(xs) // max(1, len(np.unique(ds))))
        group = ds // max(1, int(np.ceil(20 / per_day)))     # pool days so each IC uses >= ~20 samples
        for d in np.unique(group):
            m = group == d
            if m.sum() >= 8 and np.std(xs[m]) > 0 and np.std(ys[m]) > 0:
                daily.append(st.spearmanr(xs[m], ys[m]).statistic)
        daily = np.array([v for v in daily if np.isfinite(v)])
        t = float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))) if len(daily) > 2 and daily.std(ddof=1) > 0 else None
        qs = np.quantile(xs, [0.2, 0.8])
        top = ys[xs >= qs[1]].mean() if np.any(xs >= qs[1]) else np.nan
        bot = ys[xs <= qs[0]].mean() if np.any(xs <= qs[0]) else np.nan
        reg = {}
        for g in (0, 1, 2):
            m = rs == g
            if m.sum() > 30 and np.std(xs[m]) > 0:
                reg[('low_vol', 'mid_vol', 'high_vol')[g]] = round(float(st.spearmanr(xs[m], ys[m]).statistic), 4)
        spread = float(top - bot)
        report['horizons'][h] = {
            'n': int(len(xs)), 'ic': round(ic, 4), 'rank_ic': round(ric, 4),
            'daily_rank_ic_mean': round(float(daily.mean()), 4) if len(daily) else None,
            'daily_rank_ic_t': round(t, 2) if t is not None else None,
            'pct_days_positive_ic': round(float(np.mean(daily > 0)), 3) if len(daily) else None,
            'q5_minus_q1_bps': round(spread, 3),
            'half_spread_vs_cost': round(abs(spread) / 2 / ROUND_TRIP_COST_BPS, 3),
            'half_spread_vs_maker_cost': round(abs(spread) / 2 / MAKER_ROUND_TRIP_BPS, 3),
            'by_vol_regime': reg,
        }
    xa = x[np.isfinite(x)]
    if len(xa) > 10:
        report['autocorr_1'] = round(float(np.corrcoef(xa[:-1], xa[1:])[0, 1]), 4) if np.std(xa) > 0 else None
    # economic value at TRADABLE horizons: significance x cost coverage (maker coverage counts half: fills uncertain)
    def value(v):
        econ = min(1.5, max(v['half_spread_vs_cost'], 0.5 * v.get('half_spread_vs_maker_cost', 0.0)))
        return abs(v['daily_rank_ic_t']) * (0.25 + econ)
    cands = [(h, v) for h, v in report['horizons'].items() if v.get('daily_rank_ic_t') is not None and h >= TRADABLE_MIN_H]
    best = max(cands, key=lambda hv: value(hv[1]), default=(None, None))
    report['best_horizon_s'] = best[0]
    report['best_t'] = best[1]['daily_rank_ic_t'] if best[1] else None
    report['best_cost_coverage'] = best[1]['half_spread_vs_cost'] if best[1] else None
    report['best_maker_cost_coverage'] = best[1].get('half_spread_vs_maker_cost') if best[1] else None
    report['economic_value'] = round(value(best[1]), 3) if best[1] else 0.0
    report['interpretation'] = ('rank IC on non-overlapping samples; t uses per-day rank ICs; best horizon = max of '
                                '|t| x cost coverage over horizons >= 5 min; half_spread_vs_cost >= 1 means a top/bottom-'
                                'quintile trade covers taker round-trip costs (maker variant: 3 bps, before adverse selection)')
    return report


def novelty(values, library_values: dict, sample=None):
    """Max |Spearman| correlation of a factor with existing library factors on the same rows."""
    x = np.asarray(values, float)
    best, who = 0.0, None
    for fid, v in library_values.items():
        v = np.asarray(v, float)
        if len(v) != len(x):
            continue
        ok = np.isfinite(x) & np.isfinite(v)
        if ok.sum() < 50 or np.std(x[ok]) == 0 or np.std(v[ok]) == 0:
            continue
        c = abs(st.spearmanr(x[ok], v[ok]).statistic)
        if c > best:
            best, who = float(c), fid
    return {'max_abs_corr': round(best, 4), 'closest': who, 'novel': best < 0.7}


def rescore(report):
    """Re-derive best horizon / economic value for a stored report under the current scoring rule."""
    hs = {}
    for h, v in (report.get('horizons') or {}).items():
        v = dict(v)
        if v.get('q5_minus_q1_bps') is not None:
            v.setdefault('half_spread_vs_cost', round(abs(v['q5_minus_q1_bps']) / 2 / ROUND_TRIP_COST_BPS, 3))
            v['half_spread_vs_maker_cost'] = round(abs(v['q5_minus_q1_bps']) / 2 / MAKER_ROUND_TRIP_BPS, 3)
        hs[int(h)] = v

    def value(v):
        econ = min(1.5, max(v.get('half_spread_vs_cost') or 0.0, 0.5 * (v.get('half_spread_vs_maker_cost') or 0.0)))
        return abs(v['daily_rank_ic_t']) * (0.25 + econ)
    cands = [(h, v) for h, v in hs.items() if v.get('daily_rank_ic_t') is not None and h >= TRADABLE_MIN_H]
    best = max(cands, key=lambda hv: value(hv[1]), default=(None, None))
    out = dict(report, horizons={str(h): v for h, v in hs.items()})
    out['best_horizon_s'] = best[0]
    out['best_t'] = best[1]['daily_rank_ic_t'] if best[1] else None
    out['best_cost_coverage'] = best[1].get('half_spread_vs_cost') if best[1] else None
    out['best_maker_cost_coverage'] = best[1].get('half_spread_vs_maker_cost') if best[1] else None
    out['economic_value'] = round(value(best[1]), 3) if best[1] else 0.0
    return out
