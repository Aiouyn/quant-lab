"""Hyperliquid public recorder -> 1s feature store with the SAME schema as the OKX research store.

Subscriptions: l2Book BTC (full precision, 20 levels/side), l2Book BTC nSigFigs=3 (~$100 buckets -> wide
depth bands), trades BTC, activeAssetCtx BTC. Sizes are converted to OKX contract units (1 ct = 0.01 BTC) so
quantity-based features keep their research scale. Rows are written per minute to hl_store/1s/<date>/<HHMM>.parquet.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from ..data.book import BANDS_BPS, SWEEP_CONTRACTS, TOP

WS = 'wss://api.hyperliquid.xyz/ws'
INFO = 'https://api.hyperliquid.xyz/info'
CT = 100.0      # contracts per BTC (OKX BTC-USDT-SWAP)


def hl_root(settings):
    return settings.home / 'hl_store'


def levels(book_msg):
    bids, asks = book_msg['levels']
    b = np.array([[float(l['px']), float(l['sz']) * CT, float(l.get('n', 0))] for l in bids]) if bids else np.zeros((0, 3))
    a = np.array([[float(l['px']), float(l['sz']) * CT, float(l.get('n', 0))] for l in asks]) if asks else np.zeros((0, 3))
    return b, a


class FeatureState:
    def __init__(self):
        self.full = None
        self.wide = None
        self.prev_top = None
        self.trades = []
        self.funding_rate = 0.0
        self.ctx = {}

    def row(self, ts_ms):
        if self.full is None:
            return None
        b, a = self.full
        if len(b) == 0 or len(a) == 0:
            return None
        bp, ap = b[0, 0], a[0, 0]
        if bp >= ap:
            return None
        mid = (bp + ap) / 2
        out = {'ts': ts_ms, 'valid': 1, 'bid_px': bp, 'ask_px': ap, 'mid': mid, 'spread_bps': (ap - bp) / mid * 1e4}
        bsz0, asz0 = b[0, 1], a[0, 1]
        micro = (ap * bsz0 + bp * asz0) / max(bsz0 + asz0, 1e-12)
        out.update({'micro_dev_bps': (micro - mid) / mid * 1e4, 'bid_sz0': bsz0, 'ask_sz0': asz0,
                    'bid_n0': b[0, 2], 'ask_n0': a[0, 2]})

        def imb(x, y):
            return (x - y) / (x + y) if x + y > 0 else 0.0
        out['qi1'] = imb(bsz0, asz0)
        out['qi5'] = imb(b[:5, 1].sum(), a[:5, 1].sum())
        out['qi10'] = imb(b[:10, 1].sum(), a[:10, 1].sum())
        wb, wa = self.wide if self.wide is not None else (b, a)
        for d in BANDS_BPS:
            lo, hi = mid * (1 - d / 1e4), mid * (1 + d / 1e4)
            # use the full book while it covers the band, else the aggregated wide book
            sb = b if b[-1, 0] <= lo else wb
            sa = a if a[-1, 0] >= hi else wa
            out[f'bdep_{d}'] = float(sb[sb[:, 0] >= lo, 1].sum()); out[f'adep_{d}'] = float(sa[sa[:, 0] <= hi, 1].sum())
            if d in (10, 50):
                out[f'bcnt_{d}'] = float(sb[sb[:, 0] >= lo, 2].sum()); out[f'acnt_{d}'] = float(sa[sa[:, 0] <= hi, 2].sum())
        ofi = np.zeros(TOP)
        tb, ta = b[:TOP], a[:TOP]
        if self.prev_top is not None:
            pb, pa = self.prev_top
            m = min(len(tb), len(pb), len(ta), len(pa))
            for i in range(m):
                eb = tb[i, 1] if tb[i, 0] > pb[i, 0] else (tb[i, 1] - pb[i, 1] if tb[i, 0] == pb[i, 0] else -pb[i, 1])
                ea = ta[i, 1] if ta[i, 0] < pa[i, 0] else (ta[i, 1] - pa[i, 1] if ta[i, 0] == pa[i, 0] else -pa[i, 1])
                ofi[i] = eb - ea
        self.prev_top = (tb.copy(), ta.copy())
        out.update({'ofi1': float(ofi[0]), 'ofi5': float(ofi[:5].sum()), 'ofi10': float(ofi.sum())})
        lo50, hi50 = mid * (1 - 50 / 1e4), mid * (1 + 50 / 1e4)
        bb50 = wb[wb[:, 0] >= lo50]; aa50 = wa[wa[:, 0] <= hi50]
        jb = int(np.argmax(bb50[:, 1])) if len(bb50) else 0
        ja = int(np.argmax(aa50[:, 1])) if len(aa50) else 0
        out['bwall_sz'] = float(bb50[jb, 1]) if len(bb50) else 0.0
        out['bwall_bps'] = float((mid - bb50[jb, 0]) / mid * 1e4) if len(bb50) else 0.0
        out['awall_sz'] = float(aa50[ja, 1]) if len(aa50) else 0.0
        out['awall_bps'] = float((aa50[ja, 0] - mid) / mid * 1e4) if len(aa50) else 0.0
        for q in SWEEP_CONTRACTS:
            for side, lv, sgn in (('b', b, -1), ('a', a, 1)):
                book = lv if lv[:, 1].sum() >= q else (wb if side == 'b' else wa)
                cum = np.cumsum(book[:, 1])
                j = min(int(np.searchsorted(cum, q)), len(cum) - 1)
                out[f'{side}imp_{q // 100}'] = float(abs(book[j, 0] - mid) / mid * 1e4)
        # trades during the last second (aggressor side)
        buys = [t for t in self.trades if t[0] > 0]
        sells = [t for t in self.trades if t[0] < 0]
        vol = sum(t[1] for t in self.trades)
        out.update({'buy_sz': float(sum(t[1] for t in buys)), 'sell_sz': float(sum(t[1] for t in sells)),
                    'buy_n': float(len(buys)), 'sell_n': float(len(sells)),
                    'max_trade': float(max([t[1] for t in self.trades], default=0.0)),
                    'vwap': float(sum(t[1] * t[2] for t in self.trades) / vol) if vol else float('nan'),
                    'last_px': float(self.trades[-1][2]) if self.trades else float('nan'),
                    'funding_rate': self.funding_rate,
                    '_trades': list(self.trades)})        # raw (side, size_ct, px) of this second: maker fills (not stored)
        self.trades = []
        return out


async def _funding_loop(state, stop):
    async with httpx.AsyncClient() as c:
        while not stop.is_set():
            try:
                r = await c.post(INFO, json={'type': 'fundingHistory', 'coin': 'BTC',
                                             'startTime': int(time.time() * 1000) - 6 * 3600_000}, timeout=20)
                hist = r.json()
                if hist:
                    state.funding_rate = float(hist[-1]['fundingRate'])
            except Exception:
                pass
            await asyncio.sleep(300)


async def record_async(settings, hours=24.0, stop=None, on_row=None, write=True):
    import websockets
    stop = stop or asyncio.Event()
    root = hl_root(settings) / '1s'
    root.mkdir(parents=True, exist_ok=True)
    status_path = hl_root(settings) / 'status.json'
    state = FeatureState()
    rows, reconnects, written = [], 0, 0
    end = time.time() + hours * 3600
    asyncio.create_task(_funding_loop(state, stop))

    def flush():
        nonlocal rows, written
        if not rows or not write:
            rows = []
            return
        df = pl.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows])
        day = datetime.fromtimestamp(rows[0]['ts'] / 1000, timezone.utc).strftime('%Y-%m-%d')
        d = root / day
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f'{rows[0]["ts"]}.parquet')
        written += len(rows)
        rows = []
        status_path.write_text(json.dumps({'updated': time.time(), 'rows_written': written, 'reconnects': reconnects,
                                           'pid': __import__('os').getpid(), 'end': end}))

    async def stream(subs, wide):
        nonlocal reconnects
        while time.time() < end and not stop.is_set():
            try:
                async with websockets.connect(WS, ping_interval=20, max_size=2 ** 23) as ws:
                    for sub in subs:
                        await ws.send(json.dumps({'method': 'subscribe', 'subscription': sub}))
                    while time.time() < end and not stop.is_set():
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                        except asyncio.TimeoutError:
                            continue
                        ch, data = msg.get('channel'), msg.get('data')
                        if ch == 'l2Book':
                            if wide:
                                state.wide = levels(data)
                            else:
                                state.full = levels(data)
                        elif ch == 'trades':
                            for t in data:
                                state.trades.append((1 if t['side'] == 'B' else -1, float(t['sz']) * CT, float(t['px'])))
                        elif ch == 'activeAssetCtx':
                            state.ctx = data.get('ctx', {})
            except Exception:
                reconnects += 1
                if not wide:
                    state.full = None
                    state.prev_top = None
                await asyncio.sleep(min(30, 2 * reconnects))

    async def ticker():
        last_sec = int(time.time())
        while time.time() < end and not stop.is_set():
            await asyncio.sleep(max(0.0, last_sec + 1 - time.time()) + 0.002)
            now_sec = int(time.time())
            if now_sec <= last_sec:              # early timer wake-up (Windows granularity): never emit a second twice
                continue
            last_sec = now_sec
            r = state.row(last_sec * 1000)
            if r:
                rows.append(r)
                if on_row is not None:
                    try:
                        on_row(r)
                    except Exception as e:
                        print(f'on_row error: {e!r}', flush=True)
            if len(rows) >= 60:
                flush()

    await asyncio.gather(
        stream([{'type': 'l2Book', 'coin': 'BTC'}, {'type': 'trades', 'coin': 'BTC'},
                {'type': 'activeAssetCtx', 'coin': 'BTC'}], wide=False),
        stream([{'type': 'l2Book', 'coin': 'BTC', 'nSigFigs': 3}], wide=True),
        ticker())
    flush()
    return {'rows_written': written, 'reconnects': reconnects}


def recorder_alive(settings, max_age=30):
    p = hl_root(settings) / 'status.json'
    if not p.exists():
        return False
    try:
        st = json.loads(p.read_text())
        import psutil
        return time.time() - st['updated'] < max_age + 60 and psutil.pid_exists(st['pid'])
    except Exception:
        return False


def record(settings, hours=24.0):
    """Record Hyperliquid (execution venue) and live OKX research-identical features side by side."""
    from .okx_feed import okx_features

    async def both():
        return await asyncio.gather(record_async(settings, hours), okx_features(settings, hours))
    res = asyncio.run(both())
    print(json.dumps(res))
    return 0


def load_hl(settings, since_ms=None, resolution='1s'):
    root = hl_root(settings) / '1s'
    files = sorted(root.glob('*/*.parquet'))
    if since_ms:
        files = [f for f in files if int(f.stem) >= since_ms - 120_000]
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how='diagonal_relaxed').sort('ts').unique('ts', keep='last')
    if since_ms:
        df = df.filter(pl.col('ts') >= since_ms)
    if resolution != '1s':
        from ..data.ingest import bars
        df = bars(df.drop('funding_rate'), resolution).join(df.select('ts', 'funding_rate'), on='ts', how='left') \
            .with_columns(pl.col('funding_rate').forward_fill())
    return df


def venue_stats(settings, hours=24):
    """Hyperliquid vs OKX: spreads, depth and sweep impact (median), plus a suggested impact multiplier."""
    hl = load_hl(settings, since_ms=int((time.time() - hours * 3600) * 1000))
    if hl is None or len(hl) < 600:
        return {'error': 'not enough Hyperliquid captures yet; start `python -m desk record` (needs >= 10 min)',
                'rows': 0 if hl is None else len(hl)}
    from ..data.store import FeatureStore
    st = FeatureStore(settings, allowed=('dev',))
    d = st.days('dev')
    okx = st.load(d[-5:], '1s')
    cols = ['spread_bps', 'bdep_10', 'adep_10', 'bdep_25', 'bimp_1', 'aimp_1', 'bimp_10', 'aimp_10', 'bimp_50']

    def med(df):
        return {c: round(float(df[c].median()), 4) for c in cols if c in df.columns}
    h, o = med(hl), med(okx)
    hs_h, hs_o = h['spread_bps'] / 2, o['spread_bps'] / 2
    mult = ((h['aimp_10'] - hs_h) / max(o['aimp_10'] - hs_o, 1e-6) + (h['bimp_10'] - hs_h) / max(o['bimp_10'] - hs_o, 1e-6)) / 2
    return {'hyperliquid_rows': len(hl), 'hyperliquid_hours': round(len(hl) / 3600, 2), 'hyperliquid_median': h,
            'okx_median_last5dev': o, 'suggested_impact_mult_10btc': round(float(mult), 2),
            'note': 'HL depth beyond ~20 levels comes from the nSigFigs=3 aggregated book (coarser).'}
