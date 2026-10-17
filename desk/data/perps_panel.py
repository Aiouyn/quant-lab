"""Hourly multi-asset panel for the PERPS desk: every Hyperliquid perp (delisted included -> no survivorship bias).

Row (ts = hour h, coin) holds ONLY what a live system knows at h:
  px          HL mid of the minute-h:00 snapshot (taken ~h:00:03)           -> features
  px_fill     HL mid of the minute-h:01 snapshot (~60 s later)               -> execution price (decision -> fill latency)
  oracle, mark, premium, oi (contracts), vlm (24 h notional), isp (impact spread bps at h:01)
  f           HL funding rate PAID at h (the h-1:59 snapshot)                (fraction per hour; longs pay when > 0)
  lo, hi      min / max HL mid over minutes h:01..h+1:00 = the hour AFTER the decision. FUTURE data: used only by the
              engine to simulate limit-order fills; never use them in signals (the causality test would not see it,
              so perps_panel() for research drops them)
  dvol_btc/eth, vix, vix3m, dxy, spx_ret_1d, tnx, gold_ret_1d: market-wide context, availability-joined
              (Deribit implied vol hourly; daily macro closes usable the next day 01:00 UTC)
  bn_*        Binance positioning metrics joined by AVAILABILITY time (desk.data.availability): OI value, global
              account long/short, top-trader position long/short, taker buy/sell ratio (only coins listed on Binance)
Sources: minute HL ctx store (archive to 2026-10-01, then the verified live recorder for the forward period),
Binance metrics parquet (results/v3/bn_metrics). Output: <store>/perps/panel.parquet
Run: python -m desk.data.perps_panel
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import polars as pl

HLCTX = Path('D:/quant_lab/hlctx')
HOUR = 3_600_000


def hl_hourly(files):
    lf = pl.scan_parquet([str(f) for f in files]).with_columns(
        (pl.col('t') // HOUR * HOUR).alias('h'), ((pl.col('t') // 60_000) % 60).alias('minute'))
    lf = lf.filter(pl.col('minute').is_in([0, 1, 59]) & (pl.col('mid') > 0))
    df = lf.collect()
    m0 = df.filter(pl.col('minute') == 0).select(pl.col('h').alias('ts'), 'coin', pl.col('mid').alias('px'), 'oracle', 'mark',
                                                  'premium', 'oi', 'vlm')
    m1 = df.filter(pl.col('minute') == 1).select(pl.col('h').alias('ts'), 'coin', pl.col('mid').alias('px_fill'),
                                                  ((pl.col('iask') - pl.col('ibid')) / pl.col('mid') * 1e4).cast(pl.Float32).alias('isp'))
    m59 = df.filter(pl.col('minute') == 59).select((pl.col('h') + HOUR).alias('ts'), 'coin', pl.col('funding').cast(pl.Float64).alias('f'))
    # range of the HL mid over the hour AFTER the decision (minutes h:01 .. h+1:00): maker fill simulation
    rng = (pl.scan_parquet([str(f) for f in files]).filter(pl.col('mid') > 0)
           .with_columns(((pl.col('t') - 60_000) // HOUR * HOUR).alias('ts'))
           .group_by(['ts', 'coin']).agg(pl.col('mid').min().alias('lo'), pl.col('mid').max().alias('hi')).collect())
    return (m0.join(m1, on=['ts', 'coin'], how='inner').join(m59, on=['ts', 'coin'], how='left')
            .join(rng, on=['ts', 'coin'], how='left').sort(['coin', 'ts']))


def bn_metrics(settings):
    uni = {u['binance']: u['coin'] for u in json.loads((settings.store / 'carry' / 'universe.json').read_text())}
    cols = ['sum_open_interest_value', 'count_long_short_ratio', 'sum_toptrader_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    frames = []
    for d in glob.glob(str(settings.home / 'bn_metrics' / '*')):
        sym = Path(d).name
        coin = uni.get(sym) or (sym[:-4] if sym.endswith('USDT') else None)
        if sym.startswith('1000') and sym.endswith('USDT'):
            coin = 'k' + sym[4:-4]
        fs = sorted(glob.glob(d + '/*.parquet'))
        if not fs or not coin:
            continue
        m = pl.concat([pl.read_parquet(f) for f in fs], how='diagonal_relaxed')
        m = m.select(pl.col('create_time').cast(pl.Utf8),
                     *[pl.col(c).cast(pl.Utf8).str.strip_chars().replace('', None).cast(pl.Float64, strict=False) for c in cols])
        frames.append(m.with_columns(pl.lit(coin).alias('coin')))
    return pl.concat(frames).unique(['coin', 'create_time']).sort(['coin', 'create_time']) if frames else None


DERIBIT = 'https://www.deribit.com/api/v2/public/get_volatility_index_data'


def dvol(settings):
    """Deribit DVOL (30-day implied vol index) hourly for BTC and ETH since 2021-03; cached and extended."""
    import httpx, time as _t
    cache = settings.store / 'perps' / 'dvol.parquet'
    have = pl.read_parquet(cache) if cache.exists() else None
    frames = [have] if have is not None else []
    with httpx.Client(timeout=30) as c:
        for cur in ('BTC', 'ETH'):
            start = int(have.filter(pl.col('cur') == cur)['ts'].max() or 0) + 1 if have is not None and len(have.filter(pl.col('cur') == cur)) else 1614556800000
            end = int(_t.time() * 1000)
            while start < end:
                stop = min(end, start + 900 * HOUR)
                r = c.get(DERIBIT, params={'currency': cur, 'start_timestamp': start, 'end_timestamp': stop, 'resolution': 3600}).json()
                rows = (r.get('result') or {}).get('data') or []
                if rows:
                    frames.append(pl.DataFrame({'ts': [int(x[0]) for x in rows], 'dvol': [float(x[4]) for x in rows]})
                                  .with_columns(pl.lit(cur).alias('cur')))
                start = stop + 1
                _t.sleep(0.2)
    d = pl.concat(frames, how='diagonal_relaxed').unique(['cur', 'ts']).sort(['cur', 'ts'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(cache)
    return d


def market_features(settings, panel):
    """Market-wide columns joined by AVAILABILITY time:
      dvol_btc, dvol_eth   Deribit implied vol; the hourly bar opening at T is usable from T + 1 h
      vix, vix3m, dxy, spx_ret_1d, tnx, gold_ret_1d   daily closes of date D usable from D+1 01:00 UTC"""
    try:
        d = dvol(settings)
        for cur in ('BTC', 'ETH'):
            x = d.filter(pl.col('cur') == cur).select((pl.col('ts') + HOUR).alias('avail'), pl.col('dvol').alias(f'dvol_{cur.lower()}')).sort('avail')
            panel = panel.sort('ts').join_asof(x, left_on='ts', right_on='avail', strategy='backward').drop('avail')
    except Exception as e:
        print('dvol unavailable:', repr(e)[:200])
    mp = settings.store / 'macro' / 'panel.parquet'
    if mp.exists():
        m = pl.read_parquet(mp, columns=['date', 'symbol', 'adjclose'])
        def series(sym, name, ret=False):
            x = m.filter(pl.col('symbol') == sym).sort('date')
            v = (pl.col('adjclose') / pl.col('adjclose').shift(1) - 1) if ret else pl.col('adjclose')
            return x.select(((pl.col('date').cast(pl.Datetime('ms')).dt.epoch('ms')) + 86_400_000 + HOUR).alias('avail'), v.alias(name))
        for sym, name, ret in (('^VIX', 'vix', False), ('^VIX3M', 'vix3m', False), ('DX-Y.NYB', 'dxy', False),
                               ('^GSPC', 'spx_ret_1d', True), ('^TNX', 'tnx', False), ('GLD', 'gold_ret_1d', True)):
            x = series(sym, name, ret).drop_nulls().sort('avail')
            if len(x):
                panel = panel.sort('ts').join_asof(x, left_on='ts', right_on='avail', strategy='backward').drop('avail')
    return panel


def build(settings, out=None):
    from .availability import asof
    files = sorted(HLCTX.glob('*.parquet'))
    panel = hl_hourly(files)
    m = bn_metrics(settings)
    if m is not None:
        rename = {'sum_open_interest_value': 'bn_oi_value', 'count_long_short_ratio': 'bn_ls_accounts',
                  'sum_toptrader_long_short_ratio': 'bn_ls_top_positions', 'sum_taker_long_short_vol_ratio': 'bn_taker_ratio'}
        panel = asof(panel.sort('ts'), m, 'bn_metrics', left_on='ts', by='coin').rename(rename)
    panel = market_features(settings, panel)
    panel = panel.unique(['ts', 'coin'], keep='first', maintain_order=True).sort(['ts', 'coin'])   # one row per key
    out = out or settings.store / 'perps' / 'panel.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out, compression='zstd')
    return {'rows': len(panel), 'coins': panel['coin'].n_unique(), 'first': int(panel['ts'].min()), 'last': int(panel['ts'].max()),
            'path': str(out)}


if __name__ == '__main__':
    from ..config import SETTINGS
    print(build(SETTINGS))
