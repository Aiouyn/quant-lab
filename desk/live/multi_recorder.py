"""Hyperliquid multi-asset recorder: per-second research-schema features for many perps, built for weeks of 24/7
capture on a small disk. Layout: hl_store/coins/<COIN>/chunks/*.parquet (10-minute chunks) -> compacted daily into
hl_store/coins/<COIN>/1s/<YYYY-MM-DD>.parquet; compacted days are copied to an archive dir (e.g. Google Drive) and
local days older than keep_days are pruned. Funding for every coin is refreshed from metaAndAssetCtxs.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from .recorder import FeatureState, WS, INFO

DEFAULT_COINS = ('BTC', 'ETH', 'SOL', 'HYPE', 'XRP', 'DOGE', 'SUI', 'ENA')


def coin_root(settings, coin):
    return settings.home / 'hl_store' / 'coins' / coin


def _levels(book_msg, ct):
    bids, asks = book_msg['levels']
    b = np.array([[float(l['px']), float(l['sz']) * ct, float(l.get('n', 0))] for l in bids]) if bids else np.zeros((0, 3))
    a = np.array([[float(l['px']), float(l['sz']) * ct, float(l.get('n', 0))] for l in asks]) if asks else np.zeros((0, 3))
    return b, a


def compact_day(root: Path, day: str, archive: Path | None):
    chunks = sorted((root / 'chunks').glob(f'{day}_*.parquet'))
    if not chunks:
        return 0
    df = pl.concat([pl.read_parquet(c) for c in chunks], how='diagonal_relaxed').unique('ts', keep='last').sort('ts')
    (root / '1s').mkdir(parents=True, exist_ok=True)
    out = root / '1s' / f'{day}.parquet'
    if out.exists():
        df = pl.concat([pl.read_parquet(out), df], how='diagonal_relaxed').unique('ts', keep='last').sort('ts')
    tmp = out.with_suffix('.partial')
    df.write_parquet(tmp, compression='zstd', compression_level=9)
    tmp.replace(out)
    for c in chunks:
        c.unlink(missing_ok=True)
    if archive is not None:
        try:
            dst = archive / root.name / f'{day}.parquet'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, dst)
        except Exception:
            pass                                          # archiving is best-effort (Drive offline etc.)
    return len(df)


def prune(root: Path, keep_days: int):
    days = sorted((root / '1s').glob('*.parquet'))
    for p in days[:-keep_days] if len(days) > keep_days else []:
        p.unlink(missing_ok=True)


def prune_legacy(settings, keep_days: int):
    """Per-minute legacy feeds (BTC HL book features, live OKX features) only need recent days for paper warm-up."""
    for base in (settings.home / 'hl_store' / '1s', settings.home / 'okx_live' / '1s'):
        if not base.exists():
            continue
        days = sorted(d for d in base.iterdir() if d.is_dir())
        for d in days[:-keep_days] if len(days) > keep_days else []:
            shutil.rmtree(d, ignore_errors=True)


async def record_multi(settings, coins=DEFAULT_COINS, hours=720.0, archive: Path | None = None, keep_days=3,
                       stop=None, chunk_rows=600):
    import websockets
    stop = stop or asyncio.Event()
    end = time.time() + hours * 3600
    states = {c: FeatureState() for c in coins}
    # sizes in BTC-equivalent notional units of 0.01 BTC (OKX contract scale) so depth/impact features compare across
    # coins: ct = 100 * mid(coin) / mid(BTC), fixed at start and recorded in the status file
    async with httpx.AsyncClient() as cl:
        mids = (await cl.post(INFO, json={'type': 'allMids'}, timeout=20)).json()
    btc = float(mids['BTC'])
    cts = {c: 100.0 * float(mids.get(c, btc)) / btc for c in coins}
    rows = {c: [] for c in coins}
    stats = {'rows': 0, 'reconnects': 0, 'started': time.time(), 'coins': list(coins), 'size_units_per_coin': cts}
    status = settings.home / 'hl_store' / 'multi_status.json'
    status.parent.mkdir(parents=True, exist_ok=True)

    def flush(c, final=False):
        if not rows[c]:
            return
        root = coin_root(settings, c)
        (root / 'chunks').mkdir(parents=True, exist_ok=True)
        day = datetime.fromtimestamp(rows[c][0]['ts'] / 1000, timezone.utc).strftime('%Y-%m-%d')
        pl.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows[c]]) \
            .write_parquet(root / 'chunks' / f'{day}_{rows[c][0]["ts"]}.parquet', compression='zstd')
        rows[c] = []

    async def stream(wide):
        while time.time() < end and not stop.is_set():
            try:
                async with websockets.connect(WS, ping_interval=20, max_size=2 ** 24) as ws:
                    for c in coins:
                        sub = {'type': 'l2Book', 'coin': c, **({'nSigFigs': 3} if wide else {})}
                        await ws.send(json.dumps({'method': 'subscribe', 'subscription': sub}))
                        if not wide:
                            await ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'trades', 'coin': c}}))
                    while time.time() < end and not stop.is_set():
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        except asyncio.TimeoutError:
                            continue
                        ch, data = msg.get('channel'), msg.get('data')
                        if ch == 'l2Book' and data.get('coin') in states:
                            c = data['coin']
                            lv = _levels(data, cts[c])
                            if wide:
                                states[c].wide = lv
                            else:
                                states[c].full = lv
                        elif ch == 'trades':
                            for t in data:
                                c = t.get('coin')
                                if c in states:
                                    states[c].trades.append((1 if t['side'] == 'B' else -1, float(t['sz']) * cts[c], float(t['px'])))
            except Exception:
                stats['reconnects'] += 1
                if not wide:
                    for s in states.values():
                        s.full = None; s.prev_top = None
                await asyncio.sleep(min(30, 2 + stats['reconnects']))

    async def funding():
        async with httpx.AsyncClient() as cl:
            while time.time() < end and not stop.is_set():
                try:
                    meta, ctx = (await cl.post(INFO, json={'type': 'metaAndAssetCtxs'}, timeout=20)).json()
                    for i, a in enumerate(meta['universe']):
                        if a['name'] in states:
                            states[a['name']].funding_rate = float(ctx[i].get('funding') or 0)
                except Exception:
                    pass
                await asyncio.sleep(max(0.5, min(300, end - time.time())))

    async def ticker():
        last_sec = int(time.time())
        cur_day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        while time.time() < end and not stop.is_set():
            await asyncio.sleep(max(0.0, last_sec + 1 - time.time()) + 0.002)
            now = int(time.time())
            if now <= last_sec:
                continue
            last_sec = now
            for c, s in states.items():
                r = s.row(now * 1000)
                if r:
                    rows[c].append(r); stats['rows'] += 1
                if len(rows[c]) >= chunk_rows:
                    flush(c)
            day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if day != cur_day:                            # UTC day rolled: compact yesterday for every coin
                for c in coins:
                    flush(c)
                    await asyncio.to_thread(compact_day, coin_root(settings, c), cur_day, archive)
                    await asyncio.to_thread(prune, coin_root(settings, c), keep_days)
                await asyncio.to_thread(prune_legacy, settings, keep_days)
                cur_day = day
            if now % 60 == 0:
                status.write_text(json.dumps({**stats, 'updated': time.time(), 'pid': __import__('os').getpid(),
                                              'archive': str(archive) if archive else None}))

    await asyncio.gather(stream(False), stream(True), funding(), ticker())
    for c in coins:
        flush(c, final=True)
    return stats


def load_coin(settings, coin, days=None):
    """All recorded 1 s rows for a coin (compacted days + pending chunks)."""
    root = coin_root(settings, coin)
    files = sorted((root / '1s').glob('*.parquet')) + sorted((root / 'chunks').glob('*.parquet'))
    if days:
        files = [f for f in files if f.stem[:10] in set(days)]
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files], how='diagonal_relaxed').unique('ts', keep='last').sort('ts')
