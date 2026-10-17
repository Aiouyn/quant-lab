"""Oracle snapback: fade Hyperliquid mid dislocations from the oracle (CEX spot index) on the minute HL ctx store.

x[t] = dev[t] - median(dev over the previous 60 min),  dev = mid / oracle - 1 (bps)     (persistent premium removed)
Entry when |x| >= K and the coin's trailing impact spread is below a liquidity cap; side = -sign(x).
Execution (conservative): fill at the NEXT minute snapshot's mid (~60 s after the signal; live latency is ~2 s),
taker fee + half the impact spread on entry and on exit; exit at mid after H minutes. One position per coin.
"""
from __future__ import annotations

import numpy as np
import polars as pl

TAKER_BPS = 4.5


def prepare(df: pl.DataFrame, base_win=60):
    df = df.filter((pl.col('mid') > 0) & (pl.col('oracle') > 0)).sort(['coin', 't'])
    df = df.with_columns(((pl.col('mid') / pl.col('oracle') - 1) * 1e4).alias('dev'),
                         ((pl.col('iask') - pl.col('ibid')) / pl.col('mid') * 1e4).alias('ispread'))
    return df.with_columns(
        pl.col('dev').rolling_median(base_win, min_samples=base_win // 2).shift(1).over('coin').alias('base'),
        pl.col('ispread').rolling_median(60, min_samples=30).shift(1).over('coin').alias('ispread_med'))


def trades(df: pl.DataFrame, K=60.0, H=60, max_ispread=20.0, side='both', spread_cost=0.5, entry_gate=2.0,
           exit_wait=10):
    """Simulate per coin; returns a trade frame with net bps.
    entry_gate: skip the entry when the impact spread at the fill minute exceeds entry_gate x its trailing median
    (live: the book is visible before sending). exit_wait: when the exit minute's spread is abnormal, wait up to
    exit_wait more minutes for the first normal-spread minute (else exit at the last one, paying its spread)."""
    out = []
    for coin, g in df.group_by('coin'):
        t = g['t'].to_numpy(); mid = g['mid'].to_numpy(); x = (g['dev'] - g['base']).to_numpy()
        isp = g['ispread'].to_numpy(); ism = g['ispread_med'].to_numpy()
        n = len(t); i = 0
        while i < n - H - exit_wait - 1:
            xi = x[i]
            if np.isfinite(xi) and abs(xi) >= K and np.isfinite(ism[i]) and ism[i] < max_ispread \
                    and (side == 'both' or (side == 'long') == (xi < 0)):
                j, k = i + 1, i + 1 + H
                ok_entry = entry_gate is None or (np.isfinite(isp[j]) and isp[j] <= entry_gate * ism[i])
                if ok_entry and t[j] - t[i] <= 120_000 and t[k] - t[j] <= (H + 2) * 60_000:   # contiguous data only
                    if entry_gate is not None:
                        while k < i + 1 + H + exit_wait and not (np.isfinite(isp[k]) and isp[k] <= entry_gate * ism[i]):
                            k += 1
                    s = -np.sign(xi)
                    gross = s * (mid[k] / mid[j] - 1) * 1e4
                    c_in = TAKER_BPS + spread_cost * np.nan_to_num(isp[j], nan=ism[i])
                    c_out = TAKER_BPS + spread_cost * np.nan_to_num(isp[k], nan=ism[i])
                    out.append((coin[0] if isinstance(coin, tuple) else coin, t[i], s, xi, ism[i], gross,
                                gross - c_in - c_out))
                    i = k
                    continue
            i += 1
    return pl.DataFrame(out, schema=['coin', 't', 'side', 'x', 'ispread', 'gross_bps', 'net_bps'], orient='row').sort('t')


def summary(tr: pl.DataFrame, notional=1.0):
    if tr.is_empty():
        return {'n': 0}
    d = tr.with_columns((pl.col('t') // 86_400_000).alias('day')).group_by('day').agg(pl.col('net_bps').sum()).sort('day')
    days = int(d['day'].max() - d['day'].min()) + 1
    daily = np.zeros(days)
    daily[(d['day'] - d['day'].min()).to_numpy()] = d['net_bps'].to_numpy()
    pos = np.sort(daily)[::-1]
    tot = daily.sum()
    return {'n': len(tr), 'per_day': round(len(tr) / days, 2), 'mean_net_bps': round(float(tr['net_bps'].mean()), 2),
            'median_net_bps': round(float(tr['net_bps'].median()), 2), 'mean_gross_bps': round(float(tr['gross_bps'].mean()), 2),
            'win_rate': round(float((tr['net_bps'] > 0).mean()), 3),
            'sharpe_daily': round(float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(365)), 2),
            'top5_days_share': round(float(pos[:5].sum() / tot), 3) if tot > 0 else None,
            'pos_days': round(float((daily > 0).mean()), 3), 'coins': tr['coin'].n_unique()}
