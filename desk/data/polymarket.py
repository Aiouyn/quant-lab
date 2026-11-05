"""Prediction-market data (Polymarket public APIs) for the PRED desk.

1. markets: every CLOSED binary market from gamma-api (paged), with question, category, end date, volume, liquidity,
   the YES token id and the resolved outcome (final outcomePrices: YES won = 1 / 0).
2. prices: hourly YES price history per market from clob prices-history (fidelity 60 min), for markets with
   volume >= MIN_VOLUME.
Panel: <store>/pred/panel.parquet  rows (ts hour ms, market, p = YES price at ts, end_ms, outcome, category, volume)
   plus, once the volume history is backfilled (desk.data.pred_volume): vol_usd_1h, vol_usd_cum, and each market only
   from the hour its volume-to-date crossed MIN_VOLUME (point-in-time universe, no selection on future popularity).
Run: python -m desk.data.polymarket  (resumable; prices cached per market under <store>/pred/prices/)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import polars as pl

GAMMA = 'https://gamma-api.polymarket.com/markets'
CLOB = 'https://clob.polymarket.com/prices-history'
UA = {'User-Agent': 'Mozilla/5.0'}
MIN_VOLUME = 250_000.0      # tradable size only (1.3M+ resolved binaries exist, mostly tiny in-game markets)


def fetch_markets(c, root: Path):
    """Closed binary markets with lifetime volume >= MIN_VOLUME via keyset pagination, CHECKPOINTED: kept markets go to
    root/markets_parts/*.parquet and the cursor to root/markets_cursor.txt every 200 pages, so an interrupted run (or
    the daily refresh) resumes from the last cursor instead of re-reading every market. Caveat: a market created long
    ago that closes later may sit behind the saved cursor and be missed by incremental refreshes."""
    parts = root / 'markets_parts'
    parts.mkdir(parents=True, exist_ok=True)
    cur_f = root / 'markets_cursor.txt'
    cursor = cur_f.read_text().strip() if cur_f.exists() else None
    out, pages = [], 0

    def flush():
        if out:
            pl.DataFrame(out).write_parquet(parts / f'part_{int(time.time() * 1000)}.parquet')
            out.clear()
        if cursor:
            cur_f.write_text(cursor)

    while True:
        params = {'closed': 'true', 'limit': 100}
        if cursor:
            params['after_cursor'] = cursor
        d = None
        for attempt in range(6):
            try:
                d = c.get(GAMMA + '/keyset', params=params, timeout=60).json(); break
            except Exception:
                time.sleep(3 * (attempt + 1))
        if not d:
            break
        for m in d.get('markets') or []:
            try:
                outs = json.loads(m.get('outcomes') or '[]'); px = json.loads(m.get('outcomePrices') or '[]')
                toks = json.loads(m.get('clobTokenIds') or '[]')
            except Exception:
                continue
            if len(outs) != 2 or len(px) != 2 or len(toks) != 2:
                continue
            try:
                yes_i = [o.lower() for o in outs].index('yes')
            except ValueError:
                yes_i = 0                                         # first outcome treated as "YES"
            try:
                final = float(px[yes_i])
            except ValueError:
                continue
            if final not in (0.0, 1.0):
                continue                                          # unresolved / invalid / 50-50
            if float(m.get('volumeNum') or m.get('volume') or 0) < MIN_VOLUME:
                continue
            out.append({'market': str(m['id']), 'question': (m.get('question') or '')[:300], 'category': m.get('category') or '',
                        'end': m.get('endDate') or m.get('closedTime') or '', 'volume': float(m.get('volumeNum') or m.get('volume') or 0),
                        'liquidity': float(m.get('liquidityNum') or 0), 'yes_token': toks[yes_i], 'outcome': int(final)})
        pages += 1
        nxt = d.get('next_cursor')
        if not nxt or not d.get('markets'):
            break
        cursor = nxt
        if pages % 200 == 0:
            flush()
            print('market pages', pages, flush=True)
    flush()
    files = sorted(parts.glob('*.parquet'))
    return pl.concat([pl.read_parquet(f) for f in files]).unique('market', keep='last') if files else pl.DataFrame()


def fetch_prices(c, token, cache: Path, end_s=None, window_d=14, max_windows=26):
    """Hourly YES price history. The API returns nothing for interval=max on closed markets and rejects windows over
    ~14 days, so walk BACKWARDS from the market end in 14-day windows until two consecutive windows are empty (before
    the market existed) or a one-year cap. Empty cache files (from the old method) are refetched."""
    f = cache / f'{token[-24:]}.parquet'
    if f.exists():
        df = pl.read_parquet(f)
        if len(df):
            return df
    if not end_s:
        return None
    pts, empty, stop = [], 0, int(end_s) + 86400
    for _ in range(max_windows):
        start = stop - window_d * 86400
        h = None
        for attempt in range(4):
            try:
                h = c.get(CLOB, params={'market': token, 'startTs': start, 'endTs': stop, 'fidelity': 60}, timeout=60).json().get('history')
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if h is None:
            return None                                           # network failure: retry on a later run
        if h:
            pts += h; empty = 0
        else:
            empty += 1
            if empty >= 2 and pts:
                break
            if empty >= 4:
                break
        stop = start
    df = pl.DataFrame({'t': [int(x['t']) for x in pts], 'p': [float(x['p']) for x in pts]},
                      schema={'t': pl.Int64, 'p': pl.Float64}).unique('t').sort('t')
    df.write_parquet(f)
    return df


PIT_MIN_COVERAGE = 0.98     # share of markets whose volume history is cached before the point-in-time universe applies


def point_in_time_status(root: Path, sel: pl.DataFrame) -> dict:
    """Whether the panel can use the point-in-time universe: volume history cached for >= PIT_MIN_COVERAGE of the
    markets and not disabled with PRED_PIT_UNIVERSE=0. Until then the panel keeps the lifetime-volume universe (and
    coverage.json says so), so a half-finished backfill never silently shrinks the desk."""
    import os
    from .pred_volume import coverage
    cov = coverage(root, sel['yes_token'].to_list())
    enabled = os.environ.get('PRED_PIT_UNIVERSE', '1') != '0'
    return {'applied': bool(enabled and cov >= PIT_MIN_COVERAGE), 'volume_coverage': round(cov, 4), 'enabled': enabled}


def build(settings, workers=8, volume_minutes=None):
    root = settings.store / 'pred'
    cache = root / 'prices'
    cache.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=UA) as c:
        mk = fetch_markets(c, root)
        mk.write_parquet(root / 'markets.parquet')
        sel = mk.filter(pl.col('volume') >= MIN_VOLUME)
        print(f'resolved binary markets with lifetime volume >= {MIN_VOLUME:.0f}: {len(sel)}', flush=True)
        toks = sel['yes_token'].to_list()
        with ThreadPoolExecutor(workers) as pool:                  # results are cached per market on disk
            ends = [int(x.timestamp()) if x is not None else None for x in
                    sel['end'].str.to_datetime(strict=False, time_zone='UTC').to_list()]
            for i, _ in enumerate(pool.map(lambda a: fetch_prices(c, a[0], cache, a[1]) is not None, zip(toks, ends))):
                if i % 500 == 0:
                    print('prices', i, '/', len(toks), flush=True)
    if volume_minutes:                    # incremental volume-history fetch for newly resolved markets (time-boxed)
        from .pred_volume import backfill
        try:
            backfill(settings, minutes=volume_minutes, tokens=sel['yes_token'].to_list())
        except Exception as e:
            print('pred volume backfill failed:', repr(e)[:200], flush=True)
    # memory-light assembly: stream the per-market cache files instead of holding every history in RAM
    keymap = pl.DataFrame({'key': [tk[-24:] for tk in toks], 'yes_token': toks})
    files = [str(cache / f'{k}.parquet') for k in keymap['key'].to_list() if (cache / f'{k}.parquet').exists()]
    px = (pl.scan_parquet(files, include_file_paths='path')
          .with_columns(pl.col('path').str.extract(r'([^\\/]+)\.parquet$').alias('key'),   # file name (Windows or POSIX)
                        ((pl.col('t') * 1000) // 3_600_000 * 3_600_000).alias('ts'))
          .group_by(['key', 'ts']).agg(pl.col('p').last())
          .join(keymap.lazy(), on='key').drop('key'))
    meta = sel.select('market', 'yes_token', 'end', 'outcome', 'category', 'volume').lazy()
    lz = (px.join(meta, on='yes_token')
          .with_columns(pl.col('end').str.to_datetime(strict=False, time_zone='UTC').dt.epoch('ms').alias('end_ms'))
          .drop('end').filter(pl.col('end_ms').is_not_null() & (pl.col('ts') <= pl.col('end_ms'))))
    pit = point_in_time_status(root, sel)
    if pit['applied']:                    # universe membership dated by volume-to-date (see desk.data.pred_volume)
        from .pred_volume import apply_point_in_time, volume_frame
        act, entry = volume_frame(root, sel.select('yes_token', 'volume'), MIN_VOLUME)
        pit['markets_with_entry'] = len(entry)
        lz = apply_point_in_time(lz, act, entry)
    panel = lz.drop('yes_token').with_columns(pl.col('market').cast(pl.Int64)).sort(['ts', 'market']).collect(engine='streaming')
    panel.write_parquet(root / 'panel.parquet', compression='zstd')
    rep = {'markets': panel['market'].n_unique(), 'rows': len(panel), 'first': int(panel['ts'].min()), 'last': int(panel['ts'].max()),
           'point_in_time_universe': pit}
    (root / 'coverage.json').write_text(json.dumps(rep))
    return rep


if __name__ == '__main__':
    from ..config import SETTINGS
    print(build(SETTINGS))
