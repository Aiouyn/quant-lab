"""Parallel, resumable ingestion: Drive L2 archives (+ OKX trades) -> per-day 1s feature store + bars."""
from __future__ import annotations

import json
import re
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
try:
    import orjson
except ImportError:
    import json as orjson
import polars as pl

from .atomic import atomic_write_json, atomic_write_parquet
from .book import GridBook, COLUMNS

DAY_MS = 86_400_000
STORE_VERSION = 'v3.1'


def day_start_ms(date: str) -> int:
    return int(datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)


def archive_inventory(data_dir: Path):
    out = []
    for p in sorted(Path(data_dir).glob('*.tar.gz')):
        m = re.search(r'(\d{4}-\d{2}-\d{2})\.tar\.gz$', p.name)
        if m:
            out.append({'date': m[1], 'path': str(p), 'size': p.stat().st_size})
    return out


def reconstruct_day(path: str, date: str):
    """Return (dict of column -> np.ndarray[86400], quality dict)."""
    start = day_start_ms(date)
    n = 86_400
    cols = {c: np.full(n, np.nan, np.float64 if c in ('bid_px', 'ask_px', 'mid') else np.float32) for c in COLUMNS}
    valid = np.zeros(n, np.int8)
    book = GridBook()
    last_ts = None
    max_gap = 0
    gaps = 0
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            for line in stream:
                if len(line) < 3:
                    continue
                msg = orjson.loads(line)
                rows = [dict(item, action=msg.get('action')) for item in msg['data']] if isinstance(msg.get('data'), list) else [msg]
                for row in rows:
                    ts = int(row['ts'])
                    if last_ts is not None:
                        gap = ts - last_ts
                        if gap < 0:
                            raise ValueError('out of order timestamps')
                        max_gap = max(max_gap, gap)
                        if gap > 5000:
                            gaps += 1
                            book.prev_top = None
                    last_ts = ts
                    if not book.apply(row):
                        continue
                    k = (ts - start) // 1000
                    if k < 0 or k >= n:
                        continue
                    f = book.features()
                    if f is None:
                        continue
                    for c, v in f.items():
                        cols[c][k] = v
                    valid[k] = 1
    q = dict(book.q)
    q.update({'max_gap_ms': int(max_gap), 'gaps_over_5s': gaps, 'valid_rows': int(valid.sum())})
    q['resync_top20_mismatch_mean'] = q.pop('resync_top20_mismatch_sum') / max(q['resyncs'], 1)
    return cols, valid, q


def to_frame(date: str, cols, valid):
    start = day_start_ms(date)
    ts = start + np.arange(86_400, dtype=np.int64) * 1000
    df = pl.DataFrame({'ts': ts, 'valid': valid, **cols})
    # forward-fill state on invalid seconds (marked valid=0; flows set to 0) so time index == row index
    flow = [c for c in df.columns if c.startswith('ofi')]
    df = df.with_columns([pl.col(c).fill_nan(None).forward_fill() for c in df.columns if c not in ('ts', 'valid', *flow)]
                         + [pl.col(c).fill_nan(0.0) for c in flow])
    return df


def trades_frame(date: str, trades_dir: Path):
    """1s aggregated trades aligned to the L2 grid: trades in [k-1, k) seconds attach to row k (causal)."""
    p = trades_dir / f'{date}.parquet'
    if not p.exists():
        return None
    return pl.read_parquet(p)


def process_day(rec, store: str, trades_dir: str | None = None, overwrite=False):
    store = Path(store)
    date = rec['date']
    out = store / '1s' / f'{date}.parquet'
    meta = store / 'meta' / f'{date}.json'
    if out.exists() and meta.exists() and not overwrite:
        return {'date': date, 'skipped': True}
    t0 = time.time()
    cols, valid, q = reconstruct_day(rec['path'], date)
    df = to_frame(date, cols, valid)
    if trades_dir:
        tr = trades_frame(date, Path(trades_dir))
        if tr is not None:
            df = df.join(tr, on='ts', how='left').with_columns(
                [pl.col(c).fill_null(0) for c in tr.columns if c not in ('ts', 'last_px', 'vwap')]
                + [pl.col('last_px').forward_fill(), pl.col('vwap').forward_fill()])
    for sub in ('1s', '10s', '1m', 'meta'):
        (store / sub).mkdir(parents=True, exist_ok=True)
    f32 = [pl.col(c).cast(pl.Float32) for c in df.columns if df[c].dtype == pl.Float64 and c not in ('bid_px', 'ask_px', 'mid', 'last_px', 'vwap')]
    df = df.with_columns(f32)
    atomic_write_parquet(df, out, compression_level=6)
    for res in ('10s', '1m'):
        atomic_write_parquet(bars(df, res), store / res / f'{date}.parquet')
    q.update({'date': date, 'seconds': round(time.time() - t0, 1), 'store_version': STORE_VERSION,
              'source_size': rec['size'], 'has_trades': 'buy_sz' in df.columns})
    atomic_write_json(meta, q, indent=1)
    return q


FLOW_PREFIXES = ('ofi', 'buy_', 'sell_')
# state features whose within-bar MEAN carries far more information than the last second's snapshot
AVG_COLS = ('qi1', 'qi5', 'qi10', 'micro_dev_bps', 'spread_bps', 'bdep_10', 'adep_10', 'bdep_50', 'adep_50',
            'bimp_10', 'aimp_10')


def bars(df: pl.DataFrame, every: str) -> pl.DataFrame:
    """Aggregate the 1s store into bars. Bar timestamp = bar END (last second included) -> causal."""
    period = {'10s': 10_000, '1m': 60_000}[every]
    flows = [c for c in df.columns if c.startswith(FLOW_PREFIXES)]
    state = [c for c in df.columns if c not in flows and c not in ('ts', 'valid', 'mid', 'max_trade', 'vwap', 'last_px')]
    g = df.with_columns(((pl.col('ts') // period) * period + period - 1000).alias('bar_ts'))
    agg = [pl.col('mid').first().alias('open'), pl.col('mid').max().alias('high'), pl.col('mid').min().alias('low'),
           pl.col('mid').last().alias('mid'), pl.col('valid').mean().alias('valid')]
    agg += [pl.col(c).last() for c in state]
    agg += [pl.col(c).mean().alias(f'{c}_avg') for c in AVG_COLS if c in df.columns]
    agg += [pl.col(c).sum() for c in flows]
    if 'max_trade' in df.columns:
        agg += [pl.col('max_trade').max(), pl.col('last_px').last()]
    out = g.group_by('bar_ts', maintain_order=True).agg(agg).rename({'bar_ts': 'ts'})
    return out.sort('ts')


def rebuild_bars(store: Path, workers=8, progress=print):
    """Recompute 10s/1m bars for every 1s day file (after a bar-schema change)."""
    days = sorted(p.stem for p in (Path(store) / '1s').glob('*.parquet'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_rebuild_one, [(str(store), d) for d in days]))
    progress(f'rebuilt bars for {len(days)} days')
    return len(days)


def _rebuild_one(arg):
    store, d = arg
    store = Path(store)
    df = pl.read_parquet(store / '1s' / f'{d}.parquet')
    for res in ('10s', '1m'):
        atomic_write_parquet(bars(df, res), store / res / f'{d}.parquet')


def ingest(settings, workers=10, limit=None, overwrite=False, progress=print):
    inv = archive_inventory(settings.data_dir)
    if limit:
        inv = inv[:limit]
    trades_dir = settings.store / 'trades_1s'
    todo = [r for r in inv if overwrite or not (settings.store / '1s' / f"{r['date']}.parquet").exists()]
    progress(f'ingest: {len(inv)} archives, {len(todo)} to process with {workers} workers')
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_day, r, str(settings.store), str(trades_dir) if trades_dir.exists() else None, overwrite): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                q = f.result()
                results.append(q)
                progress(f"[{i}/{len(todo)}] {r['date']} ok {q.get('seconds')}s valid={q.get('valid_rows')} resync_mismatch={q.get('resync_top20_mismatch_mean', 0):.3f}")
            except Exception as e:
                err = {'date': r['date'], 'error': repr(e), 'trace': traceback.format_exc()[-2000:]}
                results.append(err)
                progress(f"[{i}/{len(todo)}] {r['date']} FAILED {e!r}")
    return results
