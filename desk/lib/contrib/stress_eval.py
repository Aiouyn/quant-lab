"""Pre-submission robustness + stability slices for strategies (pairs with cost_eval).
Runs the truth-machine stress matrix (fees x1.5, impact x2, +1-row latency, combined)
and the halves/months/long-short slices in two calls:
    from desk.lib.contrib.stress_eval import stress_eval, stability_report
    st = stress_eval(df, sig)   # configs matrix, worst_net_bps_per_day, degrade_frac, stress_pass
    sr = stability_report(df, sig)  # by month, halves, long vs short book
    assert st['stress_pass'] and st['degrade_frac'] > 0.5   # else do NOT submit
House rule: only submit_strategy when cost_eval.funnel_pass AND stress_eval.stress_pass
AND degrade_frac >= 0.5 - validation slots are scarce and each burn raises the bar."""
import numpy as np
import polars as pl
import numpy as np

try:
    from desk.lib.contrib.cost_eval import cost_eval, bar_returns
except ImportError:  # standalone use
    raise ImportError('stress_eval requires desk.lib.contrib.cost_eval (registered 2026-10-01)')

__all__ = ['stress_eval', 'stability_report']


def _shift_signal(s, k):
    if k <= 0:
        return s
    out = np.zeros_like(s); out[k:] = s[:-k]
    return out


def stress_eval(df, signal, deadband=0.05, funding=True):
    """Pre-submission robustness matrix (truth-machine robustness, cheap version):
    cost_eval under fees x1.5 (side 7.35bps), impact/slippage x2 (5.7), both (7.95),
    +1 extra fill row of latency (signal shifted: fill k rows later), and worst
    combined. Returns dict: configs {name: {net_bps_per_day, sharpe, maxdd_bps,
    top5_day_share, n_fills, boot_lb_mean_daily_bps, funnel_pass}}, worst_net_bps_per_day,
    degrade_frac (worst/base), stress_pass (base>0 AND worst>0).
    House rule: do NOT submit strategies with stress_pass=False or degrade_frac<0.5."""
    s = np.asarray(signal, float)
    configs = {
        'base':          dict(side_bps=5.10, lag=0),
        'fees_x1.5':     dict(side_bps=7.35, lag=0),
        'impact_x2':     dict(side_bps=5.70, lag=0),
        'fees+impact':   dict(side_bps=7.95, lag=0),
        'latency_+1row': dict(side_bps=5.10, lag=1),
        'worst':         dict(side_bps=7.95, lag=1),
    }
    res = {}
    for name, c in configs.items():
        ev = cost_eval(df, _shift_signal(s, c['lag']), side_bps=c['side_bps'],
                       deadband=deadband, funding=funding, bootstrap=2000, seed=0)
        res[name] = {k: ev[k] for k in ('net_bps_per_day', 'sharpe', 'maxdd_bps',
                                        'top5_day_share', 'n_fills',
                                        'boot_lb_mean_daily_bps', 'funnel_pass')}
    base = res['base']['net_bps_per_day']
    worst = min(v['net_bps_per_day'] for v in res.values())
    return {'configs': res, 'worst_net_bps_per_day': worst,
            'stress_pass': bool(base > 0 and worst > 0),
            'degrade_frac': float(worst / base) if base > 0 else float('nan')}


def stability_report(df, signal, side_bps=5.1, deadband=0.05, funding=True):
    """Robustness slices the truth machine asks about (halves / months / long-short):
    net bps/day by calendar month, first vs second half of the sample, and long-book
    vs short-book contribution (gross+fund by position sign, fill costs by trade
    direction; a flip fill is attributed to the side being opened). Invariant:
    long_net + short_net == total_net. Use together with stress_eval pre-submission."""
    import pandas as pd
    ts = df['ts'].to_numpy(); mid = df['mid'].to_numpy(); n = len(ts)
    s = np.asarray(signal, float).copy()
    s = np.where(np.isfinite(s), s, 0.0); s = np.clip(s, -1.0, 1.0)
    s = np.where(np.abs(s) < deadband, 0.0, s)
    ret = bar_returns(ts, mid)
    gapidx = np.where(np.diff(ts) > 600000)[0] + 1
    for gi in gapidx:
        s[max(0, gi - 2):gi] = 0.0
    pos = np.zeros(n); pos[2:] = s[:-2]
    gross = pos * ret
    d = np.diff(s, prepend=0.0)
    dprev = np.concatenate([[0.0], d[:-1]])          # fill executed at this bar
    cost = np.zeros(n); cost[1:] = np.abs(d[:-1]) * side_bps
    fund = np.zeros(n)
    if funding and 'funding_rate' in df.columns:
        fr = df['funding_rate'].to_numpy().copy()
        fr = np.where(np.isfinite(fr), fr, 0.0)
        fund = -pos * fr * 1e4 * (np.median(np.diff(ts)) / 3.6e6)
    net = gross - cost + fund
    long_net = np.where(pos > 0, gross + fund, 0.0) - np.where(dprev > 0, cost, 0.0)
    short_net = np.where(pos < 0, gross + fund, 0.0) - np.where(dprev < 0, cost, 0.0)
    day = (ts // 86400000).astype(np.int64)
    month = pd.to_datetime(ts, unit='ms').strftime('%Y-%m')
    dfb = pd.DataFrame({'day': day, 'net': net, 'long_net': long_net,
                        'short_net': short_net, 'month': month})
    daily = dfb.groupby('day').sum()
    ndays = max(len(daily), 1)
    mdays = dfb.groupby('month')['day'].nunique()
    by_month = dfb.groupby('month')[['net', 'long_net', 'short_net']].sum().div(mdays, axis=0)
    dpnl = daily['net'].to_numpy()
    k = len(dpnl) // 2
    halves = [float(dpnl[:k].mean()) if k else float('nan'),
              float(dpnl[k:].mean()) if len(dpnl) - k else float('nan')]
    return {'by_month_net_bps_per_day': {kk: float(vv) for kk, vv in by_month['net'].items()},
            'halves_net_bps_per_day': halves,
            'long_net_bps_per_day': float(daily['long_net'].sum()) / ndays,
            'short_net_bps_per_day': float(daily['short_net'].sum()) / ndays,
            'total_net_bps_per_day': float(daily['net'].sum()) / ndays,
            'worst_month_net_bps_per_day': float(by_month['net'].min()),
            'n_months': int(len(by_month))}
