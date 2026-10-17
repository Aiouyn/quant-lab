"""Compact minute store of Hyperliquid asset contexts (every perp, delisted ones included -> no survivorship bias).

Source: Drive QUANT_DATA/hyperliquid/asset_ctxs/<YYYYMMDD>.csv.lz4 (HL public archive, one row per coin-minute; a row
labelled minute T is a snapshot at ~T+3 s, see DATA_PARITY.md). Kept: coins whose day notional volume >= MIN_VLM.
Store: D:/quant_lab/hlctx/<YYYY-MM-DD>.parquet  (t ms, coin, mid, oracle, mark, ibid, iask, funding, premium, oi, vlm)
Run: python -m desk.data.hl_ctx [start YYYYMMDD] [end YYYYMMDD]
"""
from __future__ import annotations

import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

SRC = 'loopfactory:QUANT_DATA/hyperliquid/asset_ctxs'
ROOT = Path('D:/quant_lab/hlctx')
MIN_VLM = 2e6


def build_day(name):
    out = ROOT / f'{name[:4]}-{name[4:6]}-{name[6:8]}.parquet'
    if out.exists():
        return 0
    import lz4.frame
    raw = subprocess.run(['rclone', 'cat', f'{SRC}/{name}.csv.lz4'], capture_output=True, timeout=600).stdout
    if not raw:
        return -1
    df = pl.read_csv(io.BytesIO(lz4.frame.decompress(raw)), infer_schema_length=None)
    liquid = df.group_by('coin').agg(pl.col('day_ntl_vlm').max()).filter(pl.col('day_ntl_vlm') >= MIN_VLM)['coin']
    df = df.filter(pl.col('coin').is_in(liquid)).select(
        pl.col('time').str.to_datetime(time_zone='UTC').dt.epoch('ms').alias('t'), 'coin',
        *[pl.col(a).cast(pl.Float64).alias(b) for a, b in [('mid_px', 'mid'), ('oracle_px', 'oracle'), ('mark_px', 'mark'),
                                                          ('impact_bid_px', 'ibid'), ('impact_ask_px', 'iask')]],
        *[pl.col(a).cast(pl.Float32).alias(b) for a, b in [('funding', 'funding'), ('premium', 'premium'),
                                                          ('open_interest', 'oi'), ('day_ntl_vlm', 'vlm')]])
    df.sort(['coin', 't']).write_parquet(out, compression='zstd')
    return len(df)


def build(start='20230520', end='29991231', workers=8):
    ROOT.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(['rclone', 'lsf', SRC], capture_output=True, text=True).stdout.split()
    names = [n.split('.')[0] for n in names if start <= n.split('.')[0] <= end]
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(build_day, n) for n in names]):
            done += 1
            if done % 50 == 0:
                print(f'hl_ctx {done}/{len(names)}', flush=True)
    print('hl_ctx done', flush=True)


def load(start, end, coins=None, columns=None):
    files = sorted(f for f in ROOT.glob('*.parquet') if start <= f.stem <= end)
    lf = pl.scan_parquet([str(f) for f in files])
    if coins:
        lf = lf.filter(pl.col('coin').is_in(list(coins)))
    if columns:
        lf = lf.select(columns)
    return lf.collect()


if __name__ == '__main__':
    build(*sys.argv[1:3])
