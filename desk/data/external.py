"""Public external data: OKX trade prints (-> 1s aggregates) and funding (Hyperliquid hourly, OKX 8h).

OKX daily trade files are bucketed by UTC+8 day: file D covers [D-1 16:00, D 16:00) UTC. We aggregate
each file to 1s buckets and re-split them by UTC day, so L2 day D uses files D and D+1.
Alignment (causal): a trade at time t (ms) attaches to the row whose timestamp is floor(t/1000)*1000+1000,
i.e. trades during second [k-1, k) are visible at row k. Raw prints are discarded after aggregation.
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

OKX_TRADES = 'https://static.okx.com/cdn/okex/traderecords/trades/daily/{ymd}/BTC-USDT-SWAP-trades-{d}.zip'
HL_INFO = 'https://api.hyperliquid.xyz/info'


def _aggregate(csv_bytes: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(csv_bytes), columns=['side', 'price', 'size', 'created_time'],
                     schema_overrides={'price': pl.Float64, 'size': pl.Float64, 'created_time': pl.Int64})
    df = df.with_columns(((pl.col('created_time') // 1000) * 1000 + 1000).alias('ts'),
                         (pl.col('side') == 'buy').alias('is_buy'))
    return (df.group_by('ts').agg(
        pl.col('size').filter(pl.col('is_buy')).sum().alias('buy_sz'),
        pl.col('size').filter(~pl.col('is_buy')).sum().alias('sell_sz'),
        pl.col('is_buy').sum().cast(pl.Float64).alias('buy_n'),
        (~pl.col('is_buy')).sum().cast(pl.Float64).alias('sell_n'),
        pl.col('size').max().alias('max_trade'),
        ((pl.col('price') * pl.col('size')).sum() / pl.col('size').sum()).alias('vwap'),
        pl.col('price').sort_by('created_time').last().alias('last_px'),
    ).sort('ts'))


def fetch_okx_trade_file(d: str, raw_dir: Path, client: httpx.Client) -> Path | None:
    out = raw_dir / f'{d}.parquet'
    if out.exists():
        return out
    url = OKX_TRADES.format(ymd=d.replace('-', ''), d=d)
    for attempt in range(4):
        try:
            r = client.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            agg = _aggregate(z.read(z.namelist()[0]))
            tmp = out.with_suffix('.partial')
            agg.write_parquet(tmp, compression='zstd')
            tmp.replace(out)
            return out
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def build_trades(dates: list[str], store: Path, workers=8, progress=print):
    """dates = UTC L2 dates. Writes store/trades_1s/<date>.parquet."""
    raw = store / 'okx_trades_agg'; raw.mkdir(parents=True, exist_ok=True)
    out_dir = store / 'trades_1s'; out_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted(set(dates) | {(Date.fromisoformat(d) + timedelta(days=1)).isoformat() for d in dates})
    missing = []
    with httpx.Client(follow_redirects=True) as client, ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_okx_trade_file, d, raw, client): d for d in needed}
        for i, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                if f.result() is None:
                    missing.append(d)
            except Exception as e:
                missing.append(d)
                progress(f'trades {d} failed: {e!r}')
            if i % 25 == 0:
                progress(f'trades: {i}/{len(needed)} files')
    for d in dates:
        target = out_dir / f'{d}.parquet'
        if target.exists():
            continue
        start = int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)
        parts = [pl.read_parquet(raw / f'{x}.parquet') for x in (d, (Date.fromisoformat(d) + timedelta(days=1)).isoformat())
                 if (raw / f'{x}.parquet').exists()]
        if not parts:
            continue
        df = pl.concat(parts).filter((pl.col('ts') >= start) & (pl.col('ts') < start + 86_400_000))
        # a second can straddle two OKX files: re-aggregate
        df = df.group_by('ts').agg(pl.col('buy_sz').sum(), pl.col('sell_sz').sum(), pl.col('buy_n').sum(),
                                   pl.col('sell_n').sum(), pl.col('max_trade').max(),
                                   ((pl.col('vwap') * (pl.col('buy_sz') + pl.col('sell_sz'))).sum()
                                    / (pl.col('buy_sz') + pl.col('sell_sz')).sum()).alias('vwap'),
                                   pl.col('last_px').last()).sort('ts')
        df.write_parquet(target, compression='zstd')
    return {'files': len(needed), 'missing': missing}


def fetch_hl_funding(store: Path, start_ms: int, end_ms: int | None = None, coin='BTC') -> pl.DataFrame:
    """Hourly Hyperliquid funding rates (fraction per hour) -> store/funding_hl.parquet."""
    end_ms = end_ms or int(time.time() * 1000)
    rows, cursor = [], start_ms
    with httpx.Client() as c:
        while cursor < end_ms:
            r = c.post(HL_INFO, json={'type': 'fundingHistory', 'coin': coin, 'startTime': cursor, 'endTime': end_ms}, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += batch
            nxt = max(int(b['time']) for b in batch) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.25)
    df = pl.DataFrame({'ts': [int(r['time']) for r in rows], 'funding_rate': [float(r['fundingRate']) for r in rows],
                       'premium': [float(r['premium']) for r in rows]}).unique('ts').sort('ts')
    # snap to the hour boundary the payment belongs to
    df = df.with_columns(((pl.col('ts') + 30 * 60_000) // 3_600_000 * 3_600_000).alias('hour')).unique('hour', keep='last').sort('hour')
    df.write_parquet(store / 'funding_hl.parquet')
    return df


def fetch_okx_funding(store: Path) -> pl.DataFrame:
    """OKX 8h funding (API exposes ~3 months; used only as a reference feature)."""
    rows, after = [], None
    with httpx.Client() as c:
        for _ in range(60):
            params = {'instId': 'BTC-USDT-SWAP', 'limit': 100}
            if after:
                params['after'] = after
            r = c.get('https://www.okx.com/api/v5/public/funding-rate-history', params=params, timeout=30)
            data = r.json().get('data', [])
            if not data:
                break
            rows += data
            after = data[-1]['fundingTime']
            time.sleep(0.2)
    df = pl.DataFrame({'ts': [int(r['fundingTime']) for r in rows], 'okx_funding': [float(r['realizedRate'] or r['fundingRate']) for r in rows]}).unique('ts').sort('ts')
    df.write_parquet(store / 'funding_okx.parquet')
    return df


# ---------------------------------------------------------------------------------------------- trade tape
TICK = 0.1


def _tape_aggregate(csv_bytes: bytes) -> pl.DataFrame:
    """Per (row second, price tick, aggressor side) traded size. side: +1 buyer-initiated (lifts asks),
    -1 seller-initiated (hits bids). Row alignment matches trades_1s: trades in [k-1, k) seconds -> row k."""
    df = pl.read_csv(io.BytesIO(csv_bytes), columns=['side', 'price', 'size', 'created_time'],
                     schema_overrides={'price': pl.Float64, 'size': pl.Float64, 'created_time': pl.Int64})
    return (df.with_columns(((pl.col('created_time') // 1000) * 1000 + 1000).alias('ts'),
                            (pl.col('price') / TICK).round(0).cast(pl.Int32).alias('px'),
                            pl.when(pl.col('side') == 'buy').then(1).otherwise(-1).cast(pl.Int8).alias('side'))
              .group_by(['ts', 'px', 'side']).agg(pl.col('size').sum().cast(pl.Float32))
              .sort(['ts', 'side', 'px']))


def fetch_tape_file(d: str, raw_dir: Path, client: httpx.Client):
    out = raw_dir / f'{d}.parquet'
    if out.exists():
        return out
    url = OKX_TRADES.format(ymd=d.replace('-', ''), d=d)
    for attempt in range(4):
        try:
            r = client.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            tape = _tape_aggregate(z.read(z.namelist()[0]))
            tmp = out.with_suffix('.partial')
            tape.write_parquet(tmp, compression='zstd')
            tmp.replace(out)
            return out
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def build_tape(dates: list[str], store: Path, workers=4, progress=print):
    """store/tape/<UTC date>.parquet: price-level trade tape used by the queue-aware maker fill model."""
    raw = store / 'tape_raw'; raw.mkdir(parents=True, exist_ok=True)
    out_dir = store / 'tape'; out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if not (out_dir / f'{d}.parquet').exists()]
    needed = sorted(set(todo) | {(Date.fromisoformat(d) + timedelta(days=1)).isoformat() for d in todo})
    missing = []
    with httpx.Client(follow_redirects=True) as client, ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_tape_file, d, raw, client): d for d in needed}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                if f.result() is None:
                    missing.append(futs[f])
            except Exception as e:
                missing.append(futs[f]); progress(f'tape {futs[f]} failed: {e!r}')
            if i % 25 == 0:
                progress(f'tape: {i}/{len(needed)} files')
    for d in todo:
        start = int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)
        nxt = (Date.fromisoformat(d) + timedelta(days=1)).isoformat()
        parts = [pl.read_parquet(raw / f'{x}.parquet') for x in (d, nxt) if (raw / f'{x}.parquet').exists()]
        if not parts:
            continue
        t = (pl.concat(parts).filter((pl.col('ts') >= start) & (pl.col('ts') < start + 86_400_000))
             .group_by(['ts', 'px', 'side']).agg(pl.col('size').sum()).sort(['ts', 'side', 'px']))
        t.write_parquet(out_dir / f'{d}.parquet', compression='zstd')
    # raw per-file aggregates are only needed to re-split days; drop the ones no longer needed
    done = {p.stem for p in out_dir.glob('*.parquet')}
    for p in raw.glob('*.parquet'):
        prev = (Date.fromisoformat(p.stem) - timedelta(days=1)).isoformat()
        if p.stem in done and (prev in done or prev not in dates):
            p.unlink(missing_ok=True)
    return {'days': len(todo), 'missing_files': missing}
