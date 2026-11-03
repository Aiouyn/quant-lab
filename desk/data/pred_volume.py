"""Point-in-time trading volume for the PRED desk (removes the "final volume >= $250k" look-ahead).

Why: the pred universe is every resolved Polymarket binary whose LIFETIME volume reached MIN_VOLUME. Selecting on the
final number means a strategy trades a market from its first hour already "knowing" it will become popular. A live
trader only sees the volume traded so far. This module rebuilds that history so the panel can:
  1. show a market only from the hour its volume-to-date actually crossed MIN_VOLUME (point-in-time universe), and
  2. give strategies observable activity columns: `vol_usd_1h` and `vol_usd_cum` (fill USDC, lagged one hour).

Source (free, no key): Polymarket's public Goldsky orderbook subgraph, `orderFilledEvents` of the CTF exchange. Every
fill swaps an outcome token against USDC (asset id "0"); the USDC leg (6 decimals) is the dollar volume. We read the
fills of the market's YES token (maker or taker side). Coverage: CLOB era only (late 2022 onward); older AMM markets
have no fills here and are reported as `no_fills`.

Crossing hour: the subgraph counts fills differently from gamma's displayed `volume` (each match emits one event per
order; NO-vs-NO trades are not on the YES token), so we do not compare raw dollars to MIN_VOLUME. Instead the market
qualifies at the first hour where cum_fills / total_fills >= MIN_VOLUME / gamma_volume: the hour the displayed volume
(what a live trader would see) crossed the threshold, assuming fills and displayed volume accrue at the same pace.

Cache: <store>/pred/volume/<token[-24:]>.parquet (ts hour ms, usd, n), one file per market, written once (markets
are closed, history is final). Run the backfill: python -m desk.data.pred_volume [--minutes N] [--workers W]
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import polars as pl

SUBGRAPH = ('https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/'
            'orderbook-subgraph/0.0.1/gn')
HOUR = 3_600_000
PAGE = 1000
USDC = '0'
SCHEMA = {'ts': pl.Int64, 'usd': pl.Float64, 'n': pl.Int64}

_Q = """query($tok: String!, $ts: BigInt!, $id: String!) {
  orderFilledEvents(first: %d, orderBy: %s, orderDirection: asc,
                    where: {%s: $tok, %s}) {
    id timestamp makerAssetId takerAssetId makerAmountFilled takerAmountFilled
  }
}"""


def _query(side: str, same_ts: bool) -> str:
    # normal pages walk forward by timestamp; when >= PAGE fills share one timestamp we page that second by id
    cond = 'timestamp: $ts, id_gt: $id' if same_ts else 'timestamp_gte: $ts'
    return _Q % (PAGE, 'id' if same_ts else 'timestamp', side, cond)


def _post(c: httpx.Client, query: str, variables: dict):
    last = None
    for attempt in range(6):
        try:
            r = c.post(SUBGRAPH, json={'query': query, 'variables': variables}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f'HTTP {r.status_code}')
            d = r.json()
            if d.get('errors'):
                raise RuntimeError(str(d['errors'])[:300])
            return d['data']['orderFilledEvents']
        except Exception as e:                                   # network / throttling: back off and retry
            last = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f'goldsky subgraph failed after retries: {last!r}')


def fill_usd(ev: dict) -> float:
    """USDC leg of one fill, in dollars."""
    if ev['makerAssetId'] == USDC:
        return int(ev['makerAmountFilled']) / 1e6
    if ev['takerAssetId'] == USDC:
        return int(ev['takerAmountFilled']) / 1e6
    return 0.0


def fetch_fills(c: httpx.Client, token: str, post=_post) -> list[tuple[int, float]]:
    """Every fill of `token` as (timestamp seconds, usd), maker and taker sides, deduplicated by event id."""
    seen, out = set(), []
    for side in ('makerAssetId', 'takerAssetId'):
        ts = 0
        while True:
            rows = post(c, _query(side, False), {'tok': token, 'ts': str(ts), 'id': ''})
            for ev in rows:
                if ev['id'] not in seen:
                    seen.add(ev['id']); out.append((int(ev['timestamp']), fill_usd(ev)))
            if len(rows) < PAGE:
                break
            last_ts = int(rows[-1]['timestamp'])
            if last_ts == ts:                                     # a full page inside one second: page it by id
                last_id = ''
                while True:
                    rr = post(c, _query(side, True), {'tok': token, 'ts': str(ts), 'id': last_id})
                    for ev in rr:
                        if ev['id'] not in seen:
                            seen.add(ev['id']); out.append((int(ev['timestamp']), fill_usd(ev)))
                    if len(rr) < PAGE:
                        break
                    last_id = rr[-1]['id']
                ts += 1
            else:
                ts = last_ts                                      # gte + dedupe: no fill at the boundary is lost
    return out


def hourly(fills: list[tuple[int, float]]) -> pl.DataFrame:
    if not fills:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame({'t': [f[0] for f in fills], 'usd': [f[1] for f in fills]}, schema={'t': pl.Int64, 'usd': pl.Float64})
    return (df.with_columns(((pl.col('t') * 1000) // HOUR * HOUR).alias('ts'))
            .group_by('ts').agg(pl.col('usd').sum(), pl.len().cast(pl.Int64).alias('n')).sort('ts'))


def cache_path(root: Path, token: str) -> Path:
    return root / 'volume' / f'{token[-24:]}.parquet'


def fetch_market(c, root: Path, token: str, post=_post) -> bool:
    f = cache_path(root, token)
    if f.exists():
        return True
    f.parent.mkdir(parents=True, exist_ok=True)
    df = hourly(fetch_fills(c, token, post))
    tmp = f.with_suffix('.tmp')
    df.write_parquet(tmp)                                         # an empty file = fetched, no CLOB fills
    tmp.replace(f)
    return True


def backfill(settings, minutes: float | None = None, workers: int = 6, tokens=None) -> dict:
    """Fetch the volume history of every market in <store>/pred/markets.parquet that is not cached yet, stopping
    cleanly after `minutes` (None = until done). Resumable: each market is one file."""
    root = settings.store / 'pred'
    if tokens is None:
        mk = pl.read_parquet(root / 'markets.parquet')
        tokens = mk.sort('volume', descending=True)['yes_token'].to_list()
    todo = [t for t in tokens if not cache_path(root, t).exists()]
    deadline = time.time() + minutes * 60 if minutes else None
    done = failed = 0

    def one(tok):
        if deadline and time.time() > deadline:
            return None
        try:
            return fetch_market(c, root, tok)
        except Exception:
            return False

    with httpx.Client(headers={'User-Agent': 'quant-lab/1.0'}) as c, ThreadPoolExecutor(workers) as pool:
        for i, ok in enumerate(pool.map(one, todo)):
            if ok:
                done += 1
            elif ok is False:
                failed += 1
            if i and i % 500 == 0:
                print(f'pred volume {i}/{len(todo)} fetched={done} failed={failed}', flush=True)
    rep = {'markets': len(tokens), 'cached_before': len(tokens) - len(todo), 'fetched': done, 'failed': failed,
           'remaining': sum(1 for t in tokens if not cache_path(root, t).exists())}
    (root / 'volume_status.json').write_text(json.dumps({'updated': time.time(), **rep}, indent=1))
    return rep


def coverage(root: Path, tokens) -> float:
    tokens = list(tokens)
    return sum(cache_path(root, t).exists() for t in tokens) / max(1, len(tokens))


def volume_frame(root: Path, meta: pl.DataFrame, min_volume: float) -> pl.DataFrame:
    """Per (yes_token, ts) point-in-time activity for the panel join, plus each market's qualifying hour.

    Columns: yes_token, ts, vol_usd_1h, vol_usd_cum (fills in the PREVIOUS hour / up to the start of hour ts: one hour
    of lag so nothing from inside the bar leaks), and a per-token frame `entry` (yes_token, entry_ts, total_usd).
    `meta` needs yes_token and volume (gamma lifetime volume, used only as the scale of the crossing fraction)."""
    keymap = meta.select('yes_token', 'volume').with_columns(pl.col('yes_token').str.slice(-24).alias('key'))
    files = [str(cache_path(root, t)) for t in meta['yes_token'].to_list() if cache_path(root, t).exists()]
    if not files:
        empty = pl.DataFrame(schema={'yes_token': pl.String, 'ts': pl.Int64, 'vol_usd_1h': pl.Float64,
                                     'vol_usd_cum': pl.Float64})
        return empty, pl.DataFrame(schema={'yes_token': pl.String, 'entry_ts': pl.Int64, 'total_usd': pl.Float64})
    v = (pl.scan_parquet(files, include_file_paths='path', schema=SCHEMA)
         .with_columns(pl.col('path').str.extract(r'([^\\/]+)\.parquet$').alias('key'))
         .join(keymap.lazy(), on='key').drop('key', 'path')
         .with_columns((pl.col('ts') + HOUR).alias('ts'))             # fills of hour h become visible at h + 1
         .sort(['yes_token', 'ts'])
         .with_columns(pl.col('usd').cum_sum().over('yes_token').alias('cum'),
                       pl.col('usd').sum().over('yes_token').alias('total_usd'))
         .collect())
    frac = (min_volume / pl.col('volume')).clip(None, 1.0)
    entry = (v.filter(pl.col('total_usd') > 0)
             .filter(pl.col('cum') >= frac * pl.col('total_usd') * (1 - 1e-9))
             .group_by('yes_token').agg(pl.col('ts').min().alias('entry_ts'), pl.col('total_usd').first()))
    act = v.select('yes_token', 'ts', pl.col('usd').alias('vol_usd_1h'), pl.col('cum').alias('vol_usd_cum'))
    return act, entry


def apply_point_in_time(panel: pl.LazyFrame, act: pl.DataFrame, entry: pl.DataFrame) -> pl.LazyFrame:
    """Keep each market only from its qualifying hour and attach activity columns (forward-filled cumulative volume;
    0 in hours without fills). Markets without a known entry are dropped: their universe membership can't be dated."""
    p = (panel.join(entry.lazy().select('yes_token', 'entry_ts'), on='yes_token', how='inner')
         .filter(pl.col('ts') >= pl.col('entry_ts')).drop('entry_ts'))
    p = p.join(act.lazy(), on=['yes_token', 'ts'], how='left').sort(['yes_token', 'ts'])
    # cumulative volume as of ts: the last known cumulative at or before ts (fills may occur in hours with no price)
    asof = (act.lazy().sort(['yes_token', 'ts']).select('yes_token', pl.col('ts').alias('ts_v'),
                                                          pl.col('vol_usd_cum').alias('cum_asof')))
    p = (p.sort('ts').join_asof(asof.sort('ts_v'), left_on='ts', right_on='ts_v', by='yes_token', strategy='backward',
                                check_sortedness=False)
         .with_columns(pl.coalesce('cum_asof', pl.lit(0.0)).alias('vol_usd_cum'),
                       pl.col('vol_usd_1h').fill_null(0.0)).drop('cum_asof', 'ts_v'))
    return p


if __name__ == '__main__':
    import argparse
    from ..config import SETTINGS
    ap = argparse.ArgumentParser(description='Backfill Polymarket per-market volume history (free Goldsky subgraph).')
    ap.add_argument('--minutes', type=float, default=None, help='stop after N minutes (resumable)')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    print(backfill(SETTINGS, minutes=a.minutes, workers=a.workers))
