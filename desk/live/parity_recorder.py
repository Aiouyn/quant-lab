"""Live parity recorder: capture, in REAL TIME, the exact schemas and cadences of the historical datasets harvested to
Drive, so that every feature researched on history can be computed identically live (and verified against the archive).

Streams (each row carries recv_ms = local receive time, to measure publication latency):
  hl_asset_ctxs   every 60 s, all HL perps       archive schema: time,coin,funding,open_interest,prev_day_px,day_ntl_vlm,
                                                 premium,oracle_px,mark_px,mid_px,impact_bid_px,impact_ask_px
  hl_l2book       every ~5 s, COINS20, 20 lvls   archive: market_data/<d>/<h>/l2Book/<coin> (time, levels)
  bn_metrics      polled every 30 s, SYMS20      archive: create_time,symbol + 6 metrics; recorded LONG (column,value)
                                                 with archive-aligned create_time and first-seen availability lag
  bn_bookdepth    every 30 s, SYMS20             archive: timestamp,percentage(+-1..5),depth,notional (+max_pct_covered)
  bn_klines_1m    every 60 s, SYMS20             archive: klines 1m columns
  bn_trades_1s    continuous, SYMS20             per-second aggregates of aggTrades (same aggregation as research)
Local: results/v3/parity/<stream>/<date>/*.parquet (hourly chunks); finished days are compacted, pushed to
Drive (QUANT_DATA/live/<stream>/<date>.parquet) and pruned locally after keep_days.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from ..data.harvest import COINS20, BN_SYMBOL

HL_INFO = 'https://api.hyperliquid.xyz/info'
HL_WS = 'wss://api.hyperliquid.xyz/ws'
FAPI = 'https://fapi.binance.com'
BN_WS = 'wss://fstream.binance.com/market/stream?streams='   # futures market streams moved under /market (old path hangs)
DRIVE = 'loopfactory:QUANT_DATA/live'
PCTS = (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def now_ms():
    return int(time.time() * 1000)


def day_of(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%d')


class Sink:
    """Buffered hourly parquet chunks per stream; daily compaction + Drive upload + local pruning."""

    def __init__(self, root: Path, stream: str, keep_days=2, upload=True):
        self.dir = root / stream
        self.stream, self.keep_days, self.upload = stream, keep_days, upload
        self.rows, self.hour = [], None

    def add(self, rows):
        if not rows:
            return
        h = rows[0]['recv_ms'] // 3_600_000
        if self.hour is not None and h != self.hour:
            self.flush()
        self.hour = h
        self.rows.extend(rows)

    def flush(self):
        if not self.rows:
            return
        d = day_of(self.rows[0]['recv_ms'])
        p = self.dir / d
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self.rows, infer_schema_length=None).write_parquet(p / f'{self.rows[0]["recv_ms"]}.parquet', compression='zstd')
        self.rows = []

    def close_day(self, d):
        p = self.dir / d
        files = sorted(p.glob('*.parquet')) if p.exists() else []
        if not files:
            return
        out = self.dir / f'{d}.parquet'
        pl.concat([pl.read_parquet(f) for f in files], how='diagonal_relaxed').write_parquet(out, compression='zstd', compression_level=9)
        shutil.rmtree(p, ignore_errors=True)
        if self.upload:
            r = subprocess.run(['rclone', 'copyto', str(out), f'{DRIVE}/{self.stream}/{d}.parquet'], capture_output=True)
            if r.returncode != 0:
                return
        days = sorted(self.dir.glob('*.parquet'))
        for old in days[:-self.keep_days] if len(days) > self.keep_days else []:
            old.unlink(missing_ok=True)


async def hl_asset_ctxs(sink, stop):
    async with httpx.AsyncClient(timeout=20) as c:
        while not stop.is_set():
            await asyncio.sleep(60 - time.time() % 60 + 3.0)           # archive minute rows are snapshots at ~T+2-3 s (verified)
            try:
                t0 = now_ms()
                meta, ctx = (await c.post(HL_INFO, json={'type': 'metaAndAssetCtxs'})).json()
                recv = now_ms()
                minute = datetime.fromtimestamp((t0 // 60_000) * 60, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                rows = []
                for i, a in enumerate(meta['universe']):
                    x = ctx[i]
                    imp = x.get('impactPxs') or [None, None]
                    f = lambda k: float(x[k]) if x.get(k) not in (None, '') else None
                    rows.append({'time': minute, 'coin': a['name'], 'funding': f('funding'), 'open_interest': f('openInterest'),
                                 'prev_day_px': f('prevDayPx'), 'day_ntl_vlm': f('dayNtlVlm'), 'premium': f('premium'),
                                 'oracle_px': f('oraclePx'), 'mark_px': f('markPx'), 'mid_px': f('midPx'),
                                 'impact_bid_px': float(imp[0]) if imp and imp[0] else None,
                                 'impact_ask_px': float(imp[1]) if imp and len(imp) > 1 and imp[1] else None,
                                 'recv_ms': recv})
                sink.add(rows)
            except Exception as e:
                print(f'hl_asset_ctxs error {e!r}', flush=True)


async def hl_l2book(sink, stop, coins=COINS20, every_s=5.0):
    import websockets
    latest = {}

    async def reader():
        while not stop.is_set():
            try:
                async with websockets.connect(HL_WS, ping_interval=20, max_size=2 ** 24) as ws:
                    for c in coins:
                        await ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'l2Book', 'coin': c}}))
                    while not stop.is_set():
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                        if msg.get('channel') == 'l2Book':
                            latest[msg['data']['coin']] = (msg['data'], now_ms())
            except Exception as e:
                print(f'hl_l2book ws {e!r}', flush=True)
                await asyncio.sleep(3)

    async def sampler():
        while not stop.is_set():
            await asyncio.sleep(every_s - time.time() % every_s)
            rows = []
            for c, (d, recv) in list(latest.items()):
                bids, asks = d['levels']
                rows.append({'coin': c, 'time': int(d['time']), 'recv_ms': recv, 'sample_ms': now_ms(),
                             'bid_px': [float(l['px']) for l in bids], 'bid_sz': [float(l['sz']) for l in bids],
                             'bid_n': [int(l['n']) for l in bids], 'ask_px': [float(l['px']) for l in asks],
                             'ask_sz': [float(l['sz']) for l in asks], 'ask_n': [int(l['n']) for l in asks]})
            sink.add(rows)
    await asyncio.gather(reader(), sampler())


BN_METRIC_EP = {'openInterestHist': ('sumOpenInterest', 'sumOpenInterestValue'),
                'topLongShortAccountRatio': ('longShortRatio',), 'topLongShortPositionRatio': ('longShortRatio',),
                'globalLongShortAccountRatio': ('longShortRatio',), 'takerlongshortRatio': ('buySellRatio',)}
# Verified 2026-10-02 against data.binance.vision metrics for 2026-10-01 (100% exact for OI, 1e-4 = API 4-dp rounding
# for ratios): archive create_time = API timestamp - 5 min for OI and long/short ratios, = API timestamp for the taker
# ratio. Each endpoint publishes on its own delay, so values are recorded per endpoint with their first-seen time.
BN_ARCHIVE_SHIFT_MS = {'openInterestHist': 300_000, 'topLongShortAccountRatio': 300_000,
                       'topLongShortPositionRatio': 300_000, 'globalLongShortAccountRatio': 300_000,
                       'takerlongshortRatio': 0}
BN_ARCHIVE_COL = {('openInterestHist', 'sumOpenInterest'): 'sum_open_interest',
                  ('openInterestHist', 'sumOpenInterestValue'): 'sum_open_interest_value',
                  ('topLongShortAccountRatio', 'longShortRatio'): 'count_toptrader_long_short_ratio',
                  ('topLongShortPositionRatio', 'longShortRatio'): 'sum_toptrader_long_short_ratio',
                  ('globalLongShortAccountRatio', 'longShortRatio'): 'count_long_short_ratio',
                  ('takerlongshortRatio', 'buySellRatio'): 'sum_taker_long_short_vol_ratio'}


async def bn_metrics(sink, stop, syms, every_s=30):
    """Long format: one row per (archive column, symbol, API timestamp), written the first time it is seen.
    first_seen_lag_s = recv - (archive create_time) is the true availability delay to apply in research
    (resolution = every_s). Rows from the very first poll are flagged backfill (their lag is not a measurement)."""
    seen, first = set(), True
    async with httpx.AsyncClient(timeout=20) as c:
        async def one(s, ep):
            try:
                return s, ep, (await c.get(f'{FAPI}/futures/data/{ep}', params={'symbol': s, 'period': '5m', 'limit': 2})).json()
            except Exception as e:
                print(f'bn_metrics {s} {ep} {e!r}', flush=True)
                return s, ep, []
        while not stop.is_set():
            res = await asyncio.gather(*[one(s, ep) for s in syms for ep in BN_METRIC_EP])
            recv, rows = now_ms(), []
            for s, ep, recs in res:
                for r in recs if isinstance(recs, list) else []:
                    ts = int(r['timestamp'])
                    if (s, ep, ts) in seen:
                        continue
                    seen.add((s, ep, ts))
                    ct = ts - BN_ARCHIVE_SHIFT_MS[ep]
                    for k in BN_METRIC_EP[ep]:
                        rows.append({'create_time': datetime.fromtimestamp(ct / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                     'symbol': s, 'column': BN_ARCHIVE_COL[(ep, k)], 'value': float(r[k]), 'api_ts': ts,
                                     'recv_ms': recv, 'first_seen_lag_s': (recv - ct) / 1000, 'backfill': first})
            sink.add(rows)
            first = False
            if len(seen) > 200_000:
                seen = {x for x in seen if x[2] > recv - 86_400_000}
            await asyncio.sleep(every_s - time.time() % every_s + 3)


async def bn_bookdepth(sink, stop, syms):
    async with httpx.AsyncClient(timeout=20) as c:
        while not stop.is_set():
            await asyncio.sleep(30 - time.time() % 30 + 1)
            rows = []
            for s in syms:
                try:
                    d = (await c.get(f'{FAPI}/fapi/v1/depth', params={'symbol': s, 'limit': 1000})).json()
                    recv = now_ms()
                    bids = [(float(p), float(q)) for p, q in d['bids']]; asks = [(float(p), float(q)) for p, q in d['asks']]
                    mid = (bids[0][0] + asks[0][0]) / 2
                    cover_b = (1 - bids[-1][0] / mid) * 100; cover_a = (asks[-1][0] / mid - 1) * 100
                    stamp = datetime.fromtimestamp(recv / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    for pct in PCTS:
                        side = bids if pct < 0 else asks
                        lim = mid * (1 + pct / 100)
                        sel = [(p, q) for p, q in side if (p >= lim if pct < 0 else p <= lim)]
                        rows.append({'timestamp': stamp, 'symbol': s, 'percentage': pct, 'depth': sum(q for _, q in sel),
                                     'notional': sum(p * q for p, q in sel), 'max_pct_covered': cover_b if pct < 0 else cover_a,
                                     'complete': (cover_b if pct < 0 else cover_a) >= abs(pct), 'recv_ms': recv})
                except Exception as e:
                    print(f'bn_bookdepth {s} {e!r}', flush=True)
            sink.add(rows)


async def bn_klines(sink, stop, syms):
    async with httpx.AsyncClient(timeout=20) as c:
        while not stop.is_set():
            await asyncio.sleep(60 - time.time() % 60 + 2)
            rows = []
            for s in syms:
                try:
                    k = (await c.get(f'{FAPI}/fapi/v1/klines', params={'symbol': s, 'interval': '1m', 'limit': 2})).json()[0]
                    rows.append({'symbol': s, 'open_time': int(k[0]), 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]),
                                 'close': float(k[4]), 'volume': float(k[5]), 'close_time': int(k[6]), 'quote_volume': float(k[7]),
                                 'count': int(k[8]), 'taker_buy_volume': float(k[9]), 'taker_buy_quote_volume': float(k[10]),
                                 'recv_ms': now_ms()})
                except Exception as e:
                    print(f'bn_klines {s} {e!r}', flush=True)
            sink.add(rows)


async def bn_trades(sink, stop, syms):
    """Per-second aggregates of aggTrades (buyer-maker => seller-initiated), the same aggregation research applies."""
    import websockets
    agg = {}
    url = BN_WS + '/'.join(f'{s.lower()}@aggTrade' for s in syms)

    async def reader():
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, max_size=2 ** 22) as ws:
                    while not stop.is_set():
                        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))['data']
                        sec = int(m['T']) // 1000 * 1000 + 1000          # trades in [k-1s, k) -> row k (causal)
                        a = agg.setdefault((m['s'], sec), [0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0])
                        q, p = float(m['q']), float(m['p'])
                        if m['m']:
                            a[1] += q; a[3] += 1
                        else:
                            a[0] += q; a[2] += 1
                        a[4] = max(a[4], q); a[5] += p * q; a[6] = p; a[7] = now_ms() - int(m['T'])   # true arrival latency
            except Exception as e:
                print(f'bn_trades ws {e!r}', flush=True)
                await asyncio.sleep(3)

    async def flusher():
        while not stop.is_set():
            await asyncio.sleep(10)
            cutoff = now_ms() - 3000
            done = [k for k in agg if k[1] < cutoff]
            rows = []
            for k in done:
                b, s_, nb, ns, mx, pq, last, lat = agg.pop(k)
                rows.append({'symbol': k[0], 'ts': k[1], 'buy_qty': b, 'sell_qty': s_, 'buy_n': nb, 'sell_n': ns,
                             'max_trade': mx, 'vwap': pq / (b + s_) if b + s_ else None, 'last_px': last,
                             'arrival_latency_ms': lat, 'recv_ms': now_ms()})
            sink.add(rows)
    await asyncio.gather(reader(), flusher())


def _parity_later(settings, day, wait_s=12 * 3600):
    """Archives publish ~1 day late: check `day` against them half a day after it closed (report -> parity/reports)."""
    time.sleep(wait_s)
    try:
        from ..data.parity import run as parity_run
        parity_run(settings, day)
    except Exception as e:
        print(f'parity {day} failed: {e!r}', flush=True)


async def run(settings, hours=720.0):
    root = settings.home / 'parity'
    stop = asyncio.Event()
    syms = [BN_SYMBOL.get(c, f'{c}USDT') for c in COINS20]
    sinks = {n: Sink(root, n) for n in ('hl_asset_ctxs', 'hl_l2book', 'bn_metrics', 'bn_bookdepth', 'bn_klines_1m', 'bn_trades_1s')}

    async def housekeeping():
        cur = day_of(now_ms())
        end = time.time() + hours * 3600
        while time.time() < end:
            await asyncio.sleep(30)
            for s in sinks.values():
                if s.rows and now_ms() // 3_600_000 != s.hour:
                    s.flush()
            d = day_of(now_ms())
            if d != cur:
                for s in sinks.values():
                    s.flush()
                    await asyncio.to_thread(s.close_day, cur)
                prev, cur = cur, d
                asyncio.create_task(asyncio.to_thread(_parity_later, settings, prev))
            (root / 'status.json').write_text(json.dumps({'updated': time.time(), 'pid': __import__('os').getpid(),
                                                          'buffered': {k: len(v.rows) for k, v in sinks.items()}}))
        stop.set()

    root.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(housekeeping(), hl_asset_ctxs(sinks['hl_asset_ctxs'], stop), hl_l2book(sinks['hl_l2book'], stop),
                         bn_metrics(sinks['bn_metrics'], stop, syms), bn_bookdepth(sinks['bn_bookdepth'], stop, syms),
                         bn_klines(sinks['bn_klines_1m'], stop, syms), bn_trades(sinks['bn_trades_1s'], stop, syms))


if __name__ == '__main__':
    import sys
    from ..config import SETTINGS
    asyncio.run(run(SETTINGS, float(sys.argv[1]) if len(sys.argv) > 1 else 720.0))
