"""Cross-venue 1-second grid: Hyperliquid L2 (archive, ~0.5 s snapshots, 20 levels) vs Binance USDT-M aggTrades.

Row s (UTC second, ms) holds EXCHANGE-time state as of s (no latency applied here; the simulator applies live
receive/fill latencies by index offsets):
  hl_age      s - exchange time of the latest HL book snapshot <= s (ms, staleness)
  bid, ask, bsz, asz            HL best bid/ask and sizes
  bdep, adep                    HL notional (USD) in the top 10 levels per side
  bn_px, bn_age                 last Binance trade price <= s and its age (ms)
  bn_buy, bn_sell               Binance taker buy/sell base quantity in (s-1 s, s]
Store: results/v3/xvenue_store/<COIN>/<YYYY-MM-DD>.parquet.  Run: python -m desk.data.xvenue dev
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .harvest import BN_SYMBOL, lz4_frames

ROOT = Path(__file__).resolve().parents[2] / 'results' / 'v3' / 'xvenue_store'   # on C: (D: is the operator's often-full game drive)
DRIVE = 'loopfactory:QUANT_DATA'
COINS = ['SOL', 'HYPE', 'XRP', 'DOGE', 'SUI', 'ENA', 'ZEC', 'NEAR', 'PUMP', 'FARTCOIN', 'WLD', 'XPL', 'AAVE', 'LINK',
         'AVAX', 'TAO', 'ONDO', 'kPEPE']


def _cat(path):
    return subprocess.run(['rclone', 'cat', f'{DRIVE}/{path}'], capture_output=True, timeout=1800).stdout


def hl_day(coin, d):
    raw = _cat(f'hyperliquid/l2Book/{coin}/{d}.lz4')
    if not raw:
        return None
    rows = []
    for line in lz4_frames(raw).decode().splitlines():
        x = json.loads(line)['raw']['data']
        b, a = x['levels']
        if not b or not a:
            continue
        rows.append((int(x['time']), float(b[0]['px']), float(a[0]['px']), float(b[0]['sz']), float(a[0]['sz']),
                     sum(float(l['px']) * float(l['sz']) for l in b[:10]), sum(float(l['px']) * float(l['sz']) for l in a[:10])))
    return pl.DataFrame(rows, schema=['hl_t', 'bid', 'ask', 'bsz', 'asz', 'bdep', 'adep'], orient='row').sort('hl_t')


def bn_month(coin, ym):
    sym = BN_SYMBOL.get(coin, f'{coin}USDT')
    raw = _cat(f'binance_um/aggTrades/{sym}/{ym}.parquet')
    if not raw:
        return None
    df = pl.read_parquet(io.BytesIO(raw), columns=['price', 'quantity', 'transact_time', 'is_buyer_maker'])
    df = df.with_columns(pl.col('is_buyer_maker').cast(pl.Utf8).str.to_lowercase().eq('true')
                         if df.schema['is_buyer_maker'] == pl.Utf8 else pl.col('is_buyer_maker'))
    return df.sort('transact_time')


def build_day(coin, d, bn):
    out = ROOT / coin / f'{d}.parquet'
    if out.exists():
        return 0
    hl = hl_day(coin, d)
    if hl is None or hl.is_empty():
        return -1
    t0 = int(np.datetime64(d, 'ms').astype(np.int64))
    grid = pl.DataFrame({'s': np.arange(t0, t0 + 86_400_000, 1000, dtype=np.int64)})
    g = grid.join_asof(hl, left_on='s', right_on='hl_t', strategy='backward')
    b = bn.filter((pl.col('transact_time') >= t0 - 3_600_000) & (pl.col('transact_time') < t0 + 86_400_000))
    last = b.select(pl.col('transact_time').alias('bn_t'), pl.col('price').alias('bn_px'))
    g = g.join_asof(last, left_on='s', right_on='bn_t', strategy='backward')
    flow = (b.filter(pl.col('transact_time') >= t0)
            .with_columns(((pl.col('transact_time') // 1000) * 1000 + 1000).alias('s'))   # (s-1, s] -> row s
            .group_by('s').agg(pl.col('quantity').filter(~pl.col('is_buyer_maker')).sum().alias('bn_buy'),
                               pl.col('quantity').filter(pl.col('is_buyer_maker')).sum().alias('bn_sell')))
    g = g.join(flow, on='s', how='left').with_columns(pl.col('bn_buy').fill_null(0.0), pl.col('bn_sell').fill_null(0.0))
    g = g.select('s', (pl.col('s') - pl.col('hl_t')).cast(pl.Int32).alias('hl_age'),
                 *[pl.col(c).cast(pl.Float32) for c in ('bid', 'ask', 'bsz', 'asz', 'bdep', 'adep', 'bn_px')],
                 (pl.col('s') - pl.col('bn_t')).cast(pl.Int32).alias('bn_age'),
                 pl.col('bn_buy').cast(pl.Float32), pl.col('bn_sell').cast(pl.Float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    g.write_parquet(tmp, compression='zstd', compression_level=6)
    tmp.replace(out)                                    # atomic: a full disk never leaves a truncated parquet
    return len(hl)


def build(start, end, coins=COINS, workers=4):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    months = sorted({(d0 + timedelta(days=i)).strftime('%Y-%m') for i in range((d1 - d0).days + 1)})
    jobs = [(c, ym) for c in coins for ym in months]

    def run(c, ym):
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        days = [x.isoformat() for x in days if x.strftime('%Y-%m') == ym]
        if all((ROOT / c / f'{x}.parquet').exists() for x in days):
            return c, ym, 0
        bn = bn_month(c, ym)
        if bn is None:
            return c, ym, -1
        n = sum(max(build_day(c, x, bn), 0) for x in days)
        return c, ym, n

    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(run, c, ym) for c, ym in jobs]):
            try:
                print('xvenue', *f.result(), flush=True)
            except Exception as e:
                print('xvenue error', repr(e)[:300], flush=True)
    print('xvenue done', flush=True)


def load(coin, start, end):
    files = sorted(f for f in (ROOT / coin).glob('*.parquet') if start <= f.stem <= end)
    return pl.concat([pl.read_parquet(f) for f in files]) if files else None


if __name__ == '__main__':
    sp = json.loads((ROOT / 'splits.json').read_text())
    a, b = sp[sys.argv[1] if len(sys.argv) > 1 else 'dev']
    build(a, b, coins=sys.argv[2].split(',') if len(sys.argv) > 2 else COINS)
