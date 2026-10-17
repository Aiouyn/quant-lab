"""Hourly HL funding/premium event strategies on the minute HL ctx store (all perps, delisted included).

Per coin and hour h: f[h] = the `funding` snapshot of minute h-1:59 (~57 s before payment; known at h).
Signals (from the operator's prior HL research, re-tested on 3 years):
  flip      : sign(f[h]) != sign(f[h-1]) and |f[h]| >= thr  -> position to RECEIVE the new rate (f>0 short, f<0 long)
  prem_ext  : |premium[h]| >= rolling 30-day p90 of |premium| -> fade (premium>0 short)
Execution: enter at the mid of minute h+1 (snapshot ~h:01:03, ~60 s after the signal), exit at the mid of minute
h+hold+1; taker fee each side + half the impact spread at entry/exit (capped at cap_bps); funding received/paid
for hours h+1..h+hold at the per-hour rates f.
"""
from __future__ import annotations

import numpy as np
import polars as pl

TAKER_BPS = 4.5


def hourly(df: pl.DataFrame):
    """Minute ctx -> per coin-hour rows with funding f (rate paid at h), premium, entry/exit-relevant mids."""
    df = df.filter((pl.col('mid') > 0) & (pl.col('oracle') > 0)).with_columns(
        (pl.col('t') // 3_600_000 * 3_600_000).alias('h'), ((pl.col('t') // 60_000) % 60).alias('minute'),
        ((pl.col('iask') - pl.col('ibid')) / pl.col('mid') * 1e4).alias('isp'))
    last = (df.filter(pl.col('minute') == 59).select(pl.col('h') + 3_600_000, 'coin', pl.col('funding').alias('f'),
                                                      pl.col('premium').alias('prem')))
    first = df.filter(pl.col('minute') == 1).select('h', 'coin', pl.col('mid').alias('mid1'), pl.col('isp').alias('isp1'),
                                                    pl.col('vlm'))
    return last.join(first, on=['h', 'coin'], how='inner').sort(['coin', 'h'])


def events(hr: pl.DataFrame, kind='flip', thr=1e-5, hold=8, cap_bps=40.0, min_vlm=5e6, q=0.9):
    hr = hr.sort(['coin', 'h']).with_columns(pl.col('f').shift(1).over('coin').alias('f_prev'),
                                             pl.col('h').shift(1).over('coin').alias('h_prev'))
    if kind == 'flip':
        sig = (pl.col('h') - pl.col('h_prev') == 3_600_000) & (pl.col('f').sign() != pl.col('f_prev').sign()) \
            & (pl.col('f_prev') != 0) & (pl.col('f').abs() >= thr)
        side = -pl.col('f').sign()
    else:
        hr = hr.with_columns(pl.col('prem').abs().rolling_quantile(q, window_size=720, min_samples=360).shift(1)
                             .over('coin').alias('pq'))
        sig = pl.col('prem').abs() >= pl.col('pq')
        side = -pl.col('prem').sign()
    hr = hr.with_columns(sig.fill_null(False).alias('sig'), side.alias('side'))
    # exit price, funding over the hold, exit spread
    hr = hr.with_columns(pl.col('mid1').shift(-hold).over('coin').alias('mid_exit'),
                         pl.col('isp1').shift(-hold).over('coin').alias('isp_exit'),
                         pl.col('h').shift(-hold).over('coin').alias('h_exit'),
                         pl.col('f').rolling_sum(hold).shift(-hold).over('coin').alias('f_sum'))
    e = hr.filter(pl.col('sig') & (pl.col('vlm') >= min_vlm) & (pl.col('h_exit') - pl.col('h') == hold * 3_600_000))
    e = e.with_columns(
        (pl.col('side') * (pl.col('mid_exit') / pl.col('mid1') - 1) * 1e4).alias('price_bps'),
        (-pl.col('side') * pl.col('f_sum') * 1e4).alias('fund_bps'),
        (2 * TAKER_BPS + 0.5 * pl.col('isp1').clip(0, cap_bps) + 0.5 * pl.col('isp_exit').clip(0, cap_bps)).alias('cost_bps'))
    e = e.with_columns((pl.col('price_bps') + pl.col('fund_bps') - pl.col('cost_bps')).alias('net_bps'))
    # one position per coin: drop events while a previous one on the same coin is still open
    keep, last_exit = [], {}
    for c, h, he in e.select('coin', 'h', 'h_exit').iter_rows():
        ok = h >= last_exit.get(c, -1)
        keep.append(ok)
        if ok:
            last_exit[c] = he
    return e.filter(pl.Series(keep))


def summary(e: pl.DataFrame):
    if e.is_empty():
        return {'n': 0}
    d = e.group_by(pl.col('h') // 86_400_000).agg(pl.col('net_bps').sum()).sort('h')['net_bps'].to_numpy()
    n = len(e)
    a, b = e.head(n // 2), e.tail(n - n // 2)
    tot = d.sum()
    return {'n': n, 'coins': e['coin'].n_unique(), 'mean_net': round(float(e['net_bps'].mean()), 2),
            'median_net': round(float(e['net_bps'].median()), 2), 'price': round(float(e['price_bps'].mean()), 2),
            'fund': round(float(e['fund_bps'].mean()), 2), 'cost': round(float(e['cost_bps'].mean()), 2),
            'win': round(float((e['net_bps'] > 0).mean()), 3), 't_stat': round(float(e['net_bps'].mean() / (e['net_bps'].std() / np.sqrt(n))), 2),
            'half_A': round(float(a['net_bps'].mean()), 2), 'half_B': round(float(b['net_bps'].mean()), 2),
            'top5_days_share': round(float(np.sort(d)[::-1][:5].sum() / tot), 3) if tot > 0 else None}
