"""Live OKX BTC-USDT-SWAP features computed with the EXACT research code path (GridBook on the 5000-level book).

Strategies are researched on OKX features; computing their live inputs from OKX removes feature-transfer risk.
Execution is still priced on Hyperliquid's live book (desk.live.paper). Book: REST books-full (5000 levels) once
per second; trades: public WebSocket (sizes in contracts, aggressor side) attached causally like the archive
(trades during [k-1, k) seconds are visible at row k).
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from ..data.book import GridBook

BOOKS_FULL = 'https://www.okx.com/api/v5/market/books-full'
WS = 'wss://ws.okx.com:8443/ws/v5/public'
INST = 'BTC-USDT-SWAP'


def okx_root(settings):
    return settings.home / 'okx_live'


async def okx_features(settings, hours=24.0, stop=None, on_row=None, write=True, funding=lambda: 0.0):
    import websockets
    stop = stop or asyncio.Event()
    end = time.time() + hours * 3600
    book = GridBook()
    trades = []                          # (ts_ms, side, size_contracts, px)
    rows, written = [], 0
    root = okx_root(settings) / '1s'
    stats = {'polls': 0, 'errors': 0, 'rows': 0}

    def flush():
        nonlocal rows, written
        if rows and write:
            day = datetime.fromtimestamp(rows[0]['ts'] / 1000, timezone.utc).strftime('%Y-%m-%d')
            d = root / day
            d.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).write_parquet(d / f'{rows[0]["ts"]}.parquet')
            written += len(rows)
            (okx_root(settings) / 'status.json').write_text(json.dumps({'updated': time.time(), 'rows': written, **stats}))
        rows = []

    async def trade_stream():
        while time.time() < end and not stop.is_set():
            try:
                async with websockets.connect(WS, ping_interval=20) as ws:
                    await ws.send(json.dumps({'op': 'subscribe', 'args': [{'channel': 'trades', 'instId': INST}]}))
                    while time.time() < end and not stop.is_set():
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        except asyncio.TimeoutError:
                            continue
                        for t in msg.get('data', []) or []:
                            trades.append((int(t['ts']), 1 if t['side'] == 'buy' else -1, float(t['sz']), float(t['px'])))
            except Exception:
                stats['errors'] += 1
                await asyncio.sleep(3)

    async def poller():
        last_k = None
        async with httpx.AsyncClient(timeout=5) as c:
            while time.time() < end and not stop.is_set():
                t0 = time.time()
                try:
                    r = await c.get(BOOKS_FULL, params={'instId': INST, 'sz': 5000})
                    d = r.json()['data'][0]
                    stats['polls'] += 1
                    ts = int(d['ts'])
                    k = ts // 1000
                    if k != last_k and book.apply({'action': 'snapshot', 'bids': d['bids'], 'asks': d['asks']}):
                        f = book.features()
                        if f is not None:
                            row_ts = k * 1000
                            mine = [t for t in trades if t[0] < row_ts]
                            del trades[:len(mine)]
                            buys = [t for t in mine if t[1] > 0]; sells = [t for t in mine if t[1] < 0]
                            vol = sum(t[2] for t in mine)
                            row = {'ts': row_ts, 'valid': 1, **f,
                                   'buy_sz': sum(t[2] for t in buys), 'sell_sz': sum(t[2] for t in sells),
                                   'buy_n': float(len(buys)), 'sell_n': float(len(sells)),
                                   'max_trade': max([t[2] for t in mine], default=0.0),
                                   'vwap': sum(t[2] * t[3] for t in mine) / vol if vol else f['mid'],
                                   'last_px': mine[-1][3] if mine else f['mid'], 'funding_rate': funding()}
                            rows.append(row)
                            stats['rows'] += 1
                            last_k = k
                            if on_row is not None:
                                on_row(row)
                            if len(rows) >= 60:
                                flush()
                except Exception:
                    stats['errors'] += 1
                await asyncio.sleep(max(0.05, 1.0 - (time.time() - t0)))
        flush()

    await asyncio.gather(trade_stream(), poller())
    return {'rows_written': written, **stats}


def load_okx_live(settings, since_ms=None):
    files = sorted((okx_root(settings) / '1s').glob('*/*.parquet'))
    if since_ms:
        files = [f for f in files if int(f.stem) >= since_ms - 120_000]
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how='diagonal_relaxed').sort('ts').unique('ts', keep='last')
    return df.filter(pl.col('ts') >= since_ms) if since_ms else df
