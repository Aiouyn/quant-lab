"""Cost-aware evaluation + truth-machine funnel report for 1m-bar strategies. Usage (research sandbox):
    from desk.lib.contrib import cost_eval, horizon_edges, causal_threshold
    df = load(contiguous_days(30), resolution='1m', columns=['ts','mid','funding_rate'])
    sig = my_causal_signal(df)                       # in [-1,1], filled 1 row later
    thr = causal_threshold(sig, rate=0.2)            # fires >=1-in-5 days, causal
    ev = cost_eval(df, sig * (np.abs(sig) >= thr))   # 5.1bps/side + hourly funding
    print(ev['sharpe'], ev['net_bps_per_day'], ev['n_fills'], ev['funnel'], ev['funnel_pass'])
    print(horizon_edges(df, sig, horizons_min=(15,60,240), entry_mask=np.abs(sig)>=thr))
funnel keys map 1:1 to the validation gates (>=40 trades, >=55% positive weeks, top-5
days <50% of PnL, bootstrap LB of mean daily >0, >=20% days active). Report cost_cov
from horizon_edges + n_fills with every factor/strategy idea (DESK memo 2026-01-10)."""
import numpy as np
import polars as pl
import numpy as np

__all__ = ['bar_returns', 'causal_threshold', 'cost_eval', 'horizon_edges']


def _prep(signal, n, deadband=0.05):
    s = np.asarray(signal, dtype=float).copy()
    assert len(s) == n, "signal length must match df"
    s = np.where(np.isfinite(s), s, 0.0)
    s = np.clip(s, -1.0, 1.0)
    s = np.where(np.abs(s) < deadband, 0.0, s)   # never open tiny positions
    return s


def bar_returns(ts, mid, gap_ms=600000):
    """ret[t] = bps return earned during (t-1 -> t]. Zeroed across data gaps
    (non-contiguous day loads), so a gap can never create phantom PnL."""
    mid = np.asarray(mid, float); ts = np.asarray(ts, np.int64)
    r = np.zeros(len(mid))
    dr = np.diff(mid) / mid[:-1] * 1e4
    dr[np.diff(ts) > gap_ms] = 0.0
    r[1:] = dr
    return r


def causal_threshold(x, rate=0.2, window=10080, min_samples=1440):
    """Trailing rolling quantile threshold on |x| such that |x| >= thr fires ~`rate`
    of the time. STRICTLY causal: threshold at t uses x[<=t-1] only (shift(1)).
    Use to calibrate selectivity ('signals must fire >= 1-in-5 days' -> rate=0.2)
    without leaking the future. Returns np.ndarray (nan where undefined)."""
    import polars as pl
    s = pl.Series(np.abs(np.asarray(x, float)))
    return s.shift(1).rolling_quantile(1.0 - rate, window_size=window,
                                       min_samples=min_samples).to_numpy()


def cost_eval(df, signal, side_bps=5.1, deadband=0.05, funding=True, gap_ms=600000,
              bootstrap=10000, seed=0):
    """Cost-aware backtest-lite + funnel report for a strategy signal in [-1,1]
    (already causal). Conventions match the harness: signal[t] fills 1 row later
    (earns from bar t+2), costs side_bps on |dpos| (5.1/side = taker 4.5 + slip 0.6
    -> 10.2 RT), hourly funding charged on held position (funding_rate col is
    decimal/hour; longs pay when > 0), positions flat across data gaps, deadband
    suppresses sub-threshold position changes.

    Returns dict with: net_bps_per_day, net_daily_frac (= daily PnL in fractions of
    the $100k unit), sharpe (annualized, daily), hit_days, active_days_frac,
    pos_weeks_frac, top5_day_share, maxdd_bps, n_fills, n_entries, fills_per_day,
    boot_lb_mean_daily_bps (5th pct bootstrap of mean daily PnL), gross/cost/fund
    _bps_per_day, daily_net_bps, daily, funnel (truth-machine checklist dict),
    funnel_pass."""
    import pandas as pd
    ts = df['ts'].to_numpy(); mid = df['mid'].to_numpy(); n = len(ts)
    s = _prep(signal, n, deadband)
    ret = bar_returns(ts, mid, gap_ms)
    gapidx = np.where(np.diff(ts) > gap_ms)[0] + 1
    for gi in gapidx:
        s[max(0, gi - 2):gi] = 0.0            # flat going into every gap
    pos = np.zeros(n); pos[2:] = s[:-2]       # 1-row fill latency
    gross = pos * ret
    d = np.diff(s, prepend=0.0)
    fill = np.abs(d) >= deadband * 0.999
    cost = np.zeros(n); cost[1:] = np.abs(d[:-1]) * side_bps
    fund = np.zeros(n)
    if funding and 'funding_rate' in df.columns:
        fr = df['funding_rate'].to_numpy().copy()
        fr = np.where(np.isfinite(fr), fr, 0.0)
        dt_h = np.median(np.diff(ts)) / 3.6e6
        fund = -pos * fr * 1e4 * dt_h         # bps; longs pay when funding > 0
    net = gross - cost + fund
    day = (ts // 86400000).astype(np.int64)
    dd = pd.DataFrame({'day': day, 'net': net, 'gross': gross, 'cost': cost,
                       'fund': fund, 'pos': np.abs(pos)})
    daily = dd.groupby('day').sum()
    dpnl = daily['net'].to_numpy()
    days_u = daily.index.to_numpy()
    wk = pd.Series(dpnl).groupby(days_u // 7).sum().to_numpy()
    tot = float(dpnl.sum()); ndays = len(dpnl)
    pos_days = int((dpnl > 0).sum()); act_days = int((daily['pos'].to_numpy() > 0).sum())
    sd = float(dpnl.std(ddof=1)) if ndays > 1 else float('nan')
    sharpe = float(dpnl.mean() / sd * np.sqrt(365.0)) if ndays > 1 and sd > 0 else float('nan')
    csum = np.cumsum(dpnl)
    maxdd = float((csum - np.maximum.accumulate(csum)).min())
    rng = np.random.default_rng(seed)
    bs = dpnl[rng.integers(0, ndays, size=(bootstrap, ndays))].mean(axis=1)
    top5 = float(np.sort(dpnl)[-5:].sum() / (abs(tot) + 1e-12)) if tot > 0 else float('nan')
    n_fills = int(fill.sum())
    active = np.abs(s) > 0
    n_entries = int(active[0]) + int((active[1:] & ~active[:-1]).sum())
    out = dict(
        n_days=ndays, net_bps_per_day=tot / max(ndays, 1),
        net_daily_frac=tot / 1e4 / max(ndays, 1), sharpe=sharpe,
        hit_days=pos_days / max(ndays, 1), active_days_frac=act_days / max(ndays, 1),
        pos_weeks_frac=float((wk > 0).mean()), top5_day_share=top5, maxdd_bps=maxdd,
        n_fills=n_fills, n_entries=n_entries, fills_per_day=n_fills / max(ndays, 1),
        boot_lb_mean_daily_bps=float(np.percentile(bs, 5)),
        gross_bps_per_day=float(daily['gross'].sum() / max(ndays, 1)),
        cost_bps_per_day=float(daily['cost'].sum() / max(ndays, 1)),
        fund_bps_per_day=float(daily['fund'].sum() / max(ndays, 1)),
        daily_net_bps=dpnl, daily=daily)
    out['funnel'] = dict(
        trades_ge40=n_fills >= 40, pos_weeks_ge55=float((wk > 0).mean()) >= 0.55,
        top5_lt50=bool(top5 < 0.5) if tot > 0 else False,
        boot_lb_pos=out['boot_lb_mean_daily_bps'] > 0,
        days_active_ge20=act_days / max(ndays, 1) >= 0.2)
    out['funnel_pass'] = all(out['funnel'].values())
    return out


def horizon_edges(df, signal, horizons_min=(15, 60, 240), entry_mask=None, cost_bps=10.2):
    """Mean signed forward edge (bps) of sign(signal) at entry bars per horizon, with
    hit rate and cost coverage (mean_edge / cost_bps). entry_mask lets you test
    selective entries (e.g. |signal| > causal_threshold(sig)). Forward returns are
    RESEARCH-ONLY labels: never use them inside a strategy signal (causality test)."""
    ts = df['ts'].to_numpy(); mid = df['mid'].to_numpy(); n = len(ts)
    s = np.asarray(signal, float)
    if entry_mask is None:
        entry_mask = np.abs(s) > 0
    entry_mask = np.asarray(entry_mask, bool) & (s != 0)
    rows = []
    for h in horizons_min:
        k = int(h)
        fwd = np.full(n, np.nan)
        if k < n:
            dr = (mid[k:] - mid[:-k]) / mid[:-k] * 1e4
            dr[(ts[k:] - ts[:-k]) != k * 60000] = np.nan
            fwd[:-k] = dr
        m = entry_mask & np.isfinite(fwd)
        e = np.sign(s[m]) * fwd[m]
        rows.append(dict(h_min=h, n=int(m.sum()),
                         edge_bps=float(e.mean()) if m.sum() else float('nan'),
                         hit=float((e > 0).mean()) if m.sum() else float('nan'),
                         e_abs=float(np.abs(fwd[m]).mean()) if m.sum() else float('nan'),
                         cost_cov=float(e.mean() / cost_bps) if m.sum() else float('nan')))
    return rows
