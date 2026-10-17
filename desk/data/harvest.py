"""Data harvester: stream public market data straight to Google Drive (rclone), never staging on the local disk.

Destination (rclone remote `loopfactory:` = the operator's 5 TB Google Drive):
  QUANT_DATA/hyperliquid/asset_ctxs/<date>.csv.lz4          every HL perp, ~minute mark/oracle/mid/premium/funding/OI/impact px
  QUANT_DATA/hyperliquid/l2Book/<COIN>/<date>.lz4             24 hourly L2 snapshot files concatenated per coin-day
  QUANT_DATA/hyperliquid/node_fills/<date>/<hour>.lz4         every fill incl. liquidations (requester-pays egress!)
  QUANT_DATA/binance_um/<kind>/<SYMBOL>/<YYYY-MM>.parquet     metrics / bookDepth / klines_1m / aggTrades per month
Run:  python -m desk.data.harvest all     (resumable: anything already on Drive is skipped)
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import polars as pl
import requests

DRIVE = 'loopfactory:QUANT_DATA'
HL = 'hl_archive:hyperliquid-archive'
HL_NODE = 'hl_archive:hl-mainnet-node-data'
BV = 'https://data.binance.vision/data/futures/um'
LOG = None

COINS20 = ['BTC', 'ETH', 'SOL', 'HYPE', 'XRP', 'DOGE', 'SUI', 'ENA', 'ZEC', 'NEAR', 'PUMP', 'FARTCOIN', 'WLD', 'XPL',
           'AAVE', 'LINK', 'AVAX', 'TAO', 'ONDO', 'kPEPE']
BN_SYMBOL = {'kPEPE': '1000PEPEUSDT', 'kSHIB': '1000SHIBUSDT', 'kBONK': '1000BONKUSDT'}


def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)


def rclone(*args, inp=None, timeout=1800):
    r = subprocess.run(['rclone', *args], input=inp, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr.decode('utf-8', 'replace')


def existing(path):
    code, out, _ = rclone('lsf', '-R', '--files-only', f'{DRIVE}/{path}', timeout=600)
    return set(out.decode().split()) if code == 0 else set()


def mkdirs(paths):
    """Create target folders one by one BEFORE parallel uploads: concurrent rcat into a missing folder makes Google
    Drive create several folders with the same name (files then scatter across them)."""
    for d in sorted(set(paths)):
        rclone('mkdir', f'{DRIVE}/{d}', timeout=120)


def put(data: bytes, path):
    for attempt in range(4):
        code, _, err = rclone('rcat', f'{DRIVE}/{path}', inp=data)
        if code == 0:
            return True
        time.sleep(5 * (attempt + 1))
    log(f'upload failed {path}: {err[-300:]}')
    return False


def days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# --------------------------------------------------------------------------- Hyperliquid
def hl_asset_ctxs():
    log('HL asset_ctxs: server-side copy of the full history')
    code, _, err = rclone('copy', f'{HL}/asset_ctxs', f'{DRIVE}/hyperliquid/asset_ctxs', '--transfers', '8',
                          '--checkers', '16', '--s3-requester-pays', '--stats', '0', timeout=6 * 3600)
    log(f'HL asset_ctxs done code={code} {err[-200:]}')


_S3 = None


def s3():
    """One shared boto3 client (requester-pays, us-east-1) - far faster than an rclone process per object."""
    global _S3
    if _S3 is None:
        import boto3
        from botocore.config import Config
        _S3 = boto3.client('s3', region_name='us-east-1', config=Config(max_pool_connections=64, retries={'max_attempts': 5}))
    return _S3


def _hl_get(key, bucket='hyperliquid-archive'):
    try:
        return s3().get_object(Bucket=bucket, Key=key, RequestPayer='requester')['Body'].read()
    except Exception:
        return None


def lz4_frames(raw: bytes) -> bytes:
    """Decode ALL concatenated LZ4 frames (lz4.frame.decompress stops after the first one = 1 hour of a coin-day)."""
    import lz4.frame
    out = []
    while raw:
        d = lz4.frame.LZ4FrameDecompressor()
        out.append(d.decompress(raw))
        raw = d.unused_data
    return b''.join(out)


def _hl_coin_day(coin, d):
    ds = d.strftime('%Y%m%d')
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(lambda h: _hl_get(f'market_data/{ds}/{h}/l2Book/{coin}.lz4'), range(24)))
    chunks = [c for c in parts if c]
    if not chunks:
        return 0
    data = b''.join(chunks)                                # concatenated LZ4 frames decode as one stream
    return len(data) if put(data, f'hyperliquid/l2Book/{coin}/{d.isoformat()}.lz4') else 0


def hl_l2book(coins=COINS20, start=date(2025, 10, 1), end=None, workers=6):
    end = end or date.today() - timedelta(days=2)
    have = existing('hyperliquid/l2Book')
    todo = [(c, d) for c in coins for d in days(start, end) if f'{c}/{d.isoformat()}.lz4' not in have]
    log(f'HL l2Book: {len(todo)} coin-days to fetch ({len(have)} already on Drive)')
    mkdirs(f'hyperliquid/l2Book/{c}' for c, _ in todo)
    total, n = 0, 0
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(_hl_coin_day, c, d) for c, d in todo]):
            total += f.result() or 0; n += 1
            if n % 200 == 0:
                log(f'HL l2Book: {n}/{len(todo)} coin-days, {total / 1e9:.2f} GB uploaded')
    log(f'HL l2Book done: {total / 1e9:.2f} GB')


def hl_node_fills(n_days=45):
    end = date.today() - timedelta(days=1)
    log(f'HL node_fills: last {n_days} days (about 1 GB/day of requester-pays egress)')
    for d in days(end - timedelta(days=n_days - 1), end):
        ds = d.strftime('%Y%m%d')
        code, _, err = rclone('copy', f'{HL_NODE}/node_fills_by_block/hourly/{ds}', f'{DRIVE}/hyperliquid/node_fills/{ds}',
                              '--s3-requester-pays', '--transfers', '4', timeout=3 * 3600)
        log(f'HL node_fills {ds} code={code}')


# --------------------------------------------------------------------------- Binance (free)
def _zip_frame(url, kind):
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0])
    has_header = not raw[:1].isdigit()
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema_length=None)   # full scan: integer-looking prefixes broke dtype guesses
    if not has_header and kind == 'klines_1m':
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count',
                      'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'][:len(df.columns)]
    if not has_header and kind == 'aggTrades':
        df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'transact_time',
                      'is_buyer_maker'][:len(df.columns)]
    return df


def _bn_month(kind, sym, ym, monthly):
    if monthly:
        sub = {'klines_1m': f'monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip',
               'aggTrades': f'monthly/aggTrades/{sym}/{sym}-aggTrades-{ym}.zip'}[kind]
        frames = [_zip_frame(f'{BV}/{sub}', kind)]
    else:
        y, m = map(int, ym.split('-'))
        d0 = date(y, m, 1)
        d1 = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        urls = [f'{BV}/daily/{kind}/{sym}/{sym}-{kind}-{d.isoformat()}.zip' for d in days(d0, min(d1, date.today()))]
        with ThreadPoolExecutor(8) as pool:
            frames = list(pool.map(lambda u: _zip_frame(u, kind), urls))
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return 0
    df = pl.concat(frames, how='diagonal_relaxed')
    buf = io.BytesIO()
    df.write_parquet(buf, compression='zstd', compression_level=9)
    data = buf.getvalue()
    return len(data) if put(data, f'binance_um/{kind}/{sym}/{ym}.parquet') else 0


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f'{y}-{m:02d}'
        m += 1
        if m == 13:
            y, m = y + 1, 1


def binance(kind, symbols, start, end=None, workers=6):
    end = end or date.today() - timedelta(days=1)
    monthly = kind in ('klines_1m', 'aggTrades')
    have = existing(f'binance_um/{kind}')
    last_full = (date.today().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    todo = [(s, ym) for s in symbols for ym in months(start, end)
            if f'{s}/{ym}.parquet' not in have and (not monthly or ym <= last_full)]
    log(f'Binance {kind}: {len(todo)} symbol-months to build ({len(have)} already on Drive)')
    mkdirs(f'binance_um/{kind}/{s}' for s, _ in todo)
    total, n = 0, 0
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(_bn_month, kind, s, ym, monthly) for s, ym in todo]
        for f in as_completed(futs):
            try:
                total += f.result() or 0
            except Exception as e:
                log(f'Binance {kind} error: {e!r}'[:300])
            n += 1
            if n % 100 == 0:
                log(f'Binance {kind}: {n}/{len(todo)}, {total / 1e9:.2f} GB')
    log(f'Binance {kind} done: {total / 1e9:.2f} GB')


def universe_symbols():
    from pathlib import Path
    u = json.loads((Path(__file__).resolve().parents[2] / 'results' / 'v3' / 'store' / 'carry' / 'universe.json').read_text())
    return [x['binance'] for x in u]


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    syms20 = [BN_SYMBOL.get(c, f'{c}USDT') for c in COINS20]
    if what in ('all', 'hl_ctx'):
        hl_asset_ctxs()
    if what in ('all', 'bn_small'):
        uni = universe_symbols()
        binance('metrics', uni, date(2023, 6, 1))
        binance('klines_1m', uni, date(2023, 6, 1))
    if what in ('all', 'hl_l2'):
        hl_l2book()
    if what in ('all', 'bn_depth'):
        binance('bookDepth', syms20, date(2025, 10, 1))
    if what in ('all', 'bn_trades'):
        binance('aggTrades', [s for s in syms20 if s not in ('BTCUSDT', 'ETHUSDT')], date(2025, 10, 1), workers=3)
    if what in ('all', 'hl_fills'):
        hl_node_fills(45)
    log(f'harvest {what} finished')


if __name__ == '__main__':
    main()
