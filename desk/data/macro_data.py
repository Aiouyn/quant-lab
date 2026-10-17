"""Daily multi-asset panel for the MACRO desk (equities, sectors, countries, factors, bonds, commodities, currencies,
FX, crypto) from Yahoo Finance's public chart API, back to the 1990s where available.

Row (date, symbol): open, high, low, close (raw), adjclose (total return incl. dividends), adj_open (open scaled
by adjclose/close), volume, plus metadata: cls (asset class), tradable (1 = can be held; 0 = feature-only series
such as indices, yields, VIX, continuous futures with roll gaps), cost_bps (one-way, incl. half spread).
Bars are exchange dates; strategies decide at the close of day t and execute at the close of day t+1 (so
cross-time-zone closes can never leak). Output: <store>/macro/panel.parquet. Run: python -m desk.data.macro_data
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import polars as pl

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
CHART = 'https://query1.finance.yahoo.com/v8/finance/chart/{}'

# symbol: (class, tradable, cost_bps one-way)
UNIVERSE = {
    # US equity broad
    'SPY': ('equity_us', 1, 1.5), 'QQQ': ('equity_us', 1, 1.5), 'IWM': ('equity_us', 1, 2), 'DIA': ('equity_us', 1, 2),
    'MDY': ('equity_us', 1, 3),
    # sectors
    'XLK': ('sector', 1, 2), 'XLF': ('sector', 1, 2), 'XLE': ('sector', 1, 2), 'XLV': ('sector', 1, 2), 'XLI': ('sector', 1, 2),
    'XLY': ('sector', 1, 2), 'XLP': ('sector', 1, 2), 'XLU': ('sector', 1, 2), 'XLB': ('sector', 1, 2), 'XLRE': ('sector', 1, 3),
    'XLC': ('sector', 1, 3), 'SMH': ('sector', 1, 3), 'KRE': ('sector', 1, 3), 'XBI': ('sector', 1, 3), 'ITB': ('sector', 1, 4),
    # countries / regions
    'EFA': ('equity_intl', 1, 2), 'EEM': ('equity_intl', 1, 2), 'VGK': ('equity_intl', 1, 3), 'EWJ': ('equity_intl', 1, 3),
    'EWG': ('equity_intl', 1, 4), 'EWU': ('equity_intl', 1, 4), 'EWC': ('equity_intl', 1, 4), 'EWA': ('equity_intl', 1, 4),
    'EWZ': ('equity_intl', 1, 4), 'FXI': ('equity_intl', 1, 4), 'EWY': ('equity_intl', 1, 4), 'EWT': ('equity_intl', 1, 4),
    'EWW': ('equity_intl', 1, 5), 'INDA': ('equity_intl', 1, 5), 'EWH': ('equity_intl', 1, 5), 'EWS': ('equity_intl', 1, 5),
    # style / factor ETFs
    'IWD': ('factor', 1, 3), 'IWF': ('factor', 1, 3), 'MTUM': ('factor', 1, 4), 'QUAL': ('factor', 1, 4), 'USMV': ('factor', 1, 4),
    'VLUE': ('factor', 1, 5), 'SPLV': ('factor', 1, 4), 'RSP': ('factor', 1, 3),
    # bonds
    'TLT': ('bond', 1, 2), 'IEF': ('bond', 1, 2), 'IEI': ('bond', 1, 2), 'SHY': ('bond', 1, 1), 'AGG': ('bond', 1, 2),
    'LQD': ('bond', 1, 3), 'HYG': ('bond', 1, 3), 'JNK': ('bond', 1, 4), 'TIP': ('bond', 1, 3), 'EMB': ('bond', 1, 4),
    'BWX': ('bond', 1, 5), 'MUB': ('bond', 1, 4), 'BIL': ('bond', 1, 1),
    # commodities / real assets
    'GLD': ('commodity', 1, 2), 'SLV': ('commodity', 1, 3), 'USO': ('commodity', 1, 3), 'UNG': ('commodity', 1, 5),
    'DBC': ('commodity', 1, 4), 'DBA': ('commodity', 1, 5), 'DBB': ('commodity', 1, 6), 'PDBC': ('commodity', 1, 5),
    'VNQ': ('real_estate', 1, 2), 'GDX': ('commodity', 1, 3),
    # currency ETFs
    'UUP': ('currency', 1, 3), 'FXE': ('currency', 1, 3), 'FXY': ('currency', 1, 3), 'FXB': ('currency', 1, 4),
    'FXA': ('currency', 1, 4), 'FXC': ('currency', 1, 4), 'FXF': ('currency', 1, 4),
    # FX spot (price of 1 unit of base in quote; returns exclude the interest-rate carry)
    'EURUSD=X': ('fx', 1, 1), 'GBPUSD=X': ('fx', 1, 1), 'JPY=X': ('fx', 1, 1), 'AUDUSD=X': ('fx', 1, 1.5),
    'NZDUSD=X': ('fx', 1, 2), 'CAD=X': ('fx', 1, 1.5), 'CHF=X': ('fx', 1, 1.5), 'MXN=X': ('fx', 1, 3), 'EURJPY=X': ('fx', 1, 1.5),
    'BRL=X': ('fx', 1, 5), 'ZAR=X': ('fx', 1, 5), 'TRY=X': ('fx', 1, 8), 'INR=X': ('fx', 1, 4), 'CNY=X': ('fx', 1, 3),
    # crypto (daily, as an asset class next to the others)
    'BTC-USD': ('crypto', 1, 10), 'ETH-USD': ('crypto', 1, 10),
    # feature-only series
    '^GSPC': ('index', 0, 0), '^NDX': ('index', 0, 0), '^RUT': ('index', 0, 0), '^N225': ('index', 0, 0),
    '^GDAXI': ('index', 0, 0), '^FTSE': ('index', 0, 0), '^HSI': ('index', 0, 0), '^STOXX50E': ('index', 0, 0),
    '^VIX': ('vol', 0, 0), '^VIX3M': ('vol', 0, 0), '^VVIX': ('vol', 0, 0), '^MOVE': ('vol', 0, 0), '^OVX': ('vol', 0, 0),
    '^GVZ': ('vol', 0, 0),
    '^IRX': ('rate', 0, 0), '^FVX': ('rate', 0, 0), '^TNX': ('rate', 0, 0), '^TYX': ('rate', 0, 0),
    'DX-Y.NYB': ('index', 0, 0),
    'GC=F': ('future', 0, 0), 'SI=F': ('future', 0, 0), 'CL=F': ('future', 0, 0), 'NG=F': ('future', 0, 0),
    'HG=F': ('future', 0, 0), 'ZN=F': ('future', 0, 0), 'ZC=F': ('future', 0, 0), 'ZW=F': ('future', 0, 0),
    'ZS=F': ('future', 0, 0), 'ES=F': ('future', 0, 0),
}


def fetch(client, sym, period1=0):
    for attempt in range(4):
        try:
            r = client.get(CHART.format(sym), params={'period1': period1, 'period2': int(time.time()) + 86400,
                                                       'interval': '1d', 'events': 'div,split'}, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            res = r.json()['chart']['result']
            if not res:
                return None
            d = res[0]
            ts = d.get('timestamp') or []
            q = d['indicators']['quote'][0]
            adj = (d['indicators'].get('adjclose') or [{}])[0].get('adjclose') or q['close']
            gmt = int(d['meta'].get('gmtoffset') or 0)
            df = pl.DataFrame({'t': ts, 'open': q['open'], 'high': q['high'], 'low': q['low'], 'close': q['close'],
                               'adjclose': adj, 'volume': q.get('volume') or [None] * len(ts)},
                              schema_overrides={'open': pl.Float64, 'high': pl.Float64, 'low': pl.Float64, 'close': pl.Float64,
                                                'adjclose': pl.Float64, 'volume': pl.Float64}, strict=False)
            df = df.with_columns(pl.from_epoch(pl.col('t') + gmt, time_unit='s').dt.date().alias('date')).drop('t')
            df = df.filter(pl.col('close').is_not_null() & (pl.col('close') > 0)).unique('date', keep='last').sort('date')
            df = df.with_columns((pl.col('adjclose') / pl.col('close')).alias('f'))
            df = df.with_columns((pl.col('open') * pl.col('f')).alias('adj_open')).drop('f')
            return df.with_columns(pl.lit(sym).alias('symbol'))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def build(settings):
    with httpx.Client(headers=UA) as c, ThreadPoolExecutor(6) as pool:
        frames = list(pool.map(lambda s: fetch(c, s), UNIVERSE))
    got = [f for f in frames if f is not None and len(f)]
    df = pl.concat(got, how='diagonal_relaxed')
    meta = pl.DataFrame({'symbol': list(UNIVERSE), 'cls': [v[0] for v in UNIVERSE.values()],
                         'tradable': [v[1] for v in UNIVERSE.values()], 'cost_bps': [float(v[2]) for v in UNIVERSE.values()]})
    df = df.join(meta, on='symbol', how='left').sort(['date', 'symbol'])
    out = settings.store / 'macro' / 'panel.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression='zstd')
    missing = sorted(set(UNIVERSE) - set(df['symbol'].unique().to_list()))
    cov = df.group_by('symbol').agg(pl.col('date').min().alias('first'), pl.len().alias('rows'))
    rep = {'rows': len(df), 'symbols': df['symbol'].n_unique(), 'missing': missing,
           'first_dates': {r['symbol']: str(r['first']) for r in cov.iter_rows(named=True)}}
    (settings.store / 'macro' / 'coverage.json').write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    from ..config import SETTINGS
    r = build(SETTINGS)
    print({k: r[k] for k in ('rows', 'symbols', 'missing')})
