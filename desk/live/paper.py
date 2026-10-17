"""Hyperliquid paper trading for strategy bundles and portfolios (no wallet, no orders: simulated fills).

Live rows come from an in-process HL feed with the research feature schema (desk.live.recorder), so the
strategy code runs exactly as in research. Fills: 1 s latency, mid +/- (half spread + average depth-sweep
impact x mult + slippage), taker fee; hourly funding on open positions. Every decision and fill is journaled.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from ..lab.strategy import run_signal
from ..pipeline.bundle import load_bundle
from .recorder import load_hl, record_async, recorder_alive

RES_MS = {'1s': 1000, '10s': 10_000, '1m': 60_000}


def _impact_extra(size_btc, i1, i10, i50, hs):
    if size_btc <= 1:
        worst = hs + (i1 - hs) * size_btc
    elif size_btc <= 10:
        worst = i1 + (i10 - i1) * (size_btc - 1) / 9
    elif size_btc <= 50:
        worst = i10 + (i50 - i10) * (size_btc - 10) / 40
    else:
        worst = i50 * size_btc / 50
    return max(0.0, (worst - hs) / 2)


class Sleeve:
    def __init__(self, sid, path, weight, notional):
        self.manifest, self.mod, self.model = load_bundle(path)
        self.id, self.weight = sid, weight
        self.unit = notional * weight
        self.res = self.mod.RESOLUTION
        self.cost = self.manifest.get('cost_model') or {}
        self.cash, self.pos, self.fees, self.funding = 0.0, 0.0, 0.0, 0.0
        self.pending = None
        self.target = 0.0
        self.status = 'warming'
        self.fills = 0

    def equity(self, mid):
        return self.cash + self.pos * mid


class PaperSession:
    def __init__(self, settings, target, notional=100_000.0, feature_source='okx'):
        self.s = settings
        self.feature_source = feature_source      # 'okx' = research-identical features; 'hl' = Hyperliquid book features
        self.last_exec = None
        self.sleeves = []
        if target.startswith('P-') or (Path(target) / 'manifest.json').exists() and 'weights' in json.loads((Path(target) / 'manifest.json').read_text()):
            mpath = Path(target) if Path(target).exists() else settings.home / 'portfolios' / target
            m = json.loads((mpath / 'manifest.json').read_text())
            for sid, w in m['weights'].items():
                self.sleeves.append(Sleeve(sid, m['bundles'][sid], w, notional))
            self.kind, self.target_id = 'portfolio', m['id']
        else:
            bpath = Path(target) if Path(target).exists() else settings.home / 'bundles' / target
            sid = json.loads((bpath / 'manifest.json').read_text())['id']
            self.sleeves.append(Sleeve(sid, bpath, 1.0, notional))
            self.kind, self.target_id = 'strategy', sid
        self.id = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
        self.dir = settings.home / 'paper' / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=self._max_rows())
        self.started = time.time()
        self.last_hour = None
        self.last_mid = None
        (self.dir / 'session.json').write_text(json.dumps({
            'id': self.id, 'kind': self.kind, 'target': self.target_id, 'started': self.started,
            'feature_source': self.feature_source, 'execution_venue': 'hyperliquid (simulated fills on live HL book)',
            'sleeves': [{'id': x.id, 'weight': x.weight, 'unit_notional': x.unit, 'resolution': x.res,
                         'code_sha256': x.manifest['code_sha256'], 'expected': x.manifest.get('expected', {}).get('val')}
                        for x in self.sleeves]}, indent=1))

    def _max_rows(self):
        need = 0
        for x in self.sleeves:
            need = max(need, (x.mod.LOOKBACK + 5) * RES_MS[x.res] // 1000)
        return int(max(need, 3600) * 1.2)

    def warm_start(self):
        need = self.buffer.maxlen
        since = int((time.time() - need) * 1000)
        if self.feature_source == 'okx':
            from .okx_feed import load_okx_live
            df = load_okx_live(self.s, since_ms=since)
        else:
            df = load_hl(self.s, since_ms=since)
        if df is not None:
            for r in df.to_dicts():
                self.buffer.append(r)
        return 0 if df is None else len(df)

    def log(self, name, obj):
        with open(self.dir / name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, default=float) + '\n')

    def on_hl_row(self, row):
        """Hyperliquid book row: execution prices, funding and fills (and decisions when feature_source='hl')."""
        if self.feature_source == 'hl':
            return self.on_row(row)
        if self.last_exec is not None and row['ts'] <= self.last_exec['ts']:
            return
        self.last_exec = row
        self._execute(row)

    def on_okx_row(self, row):
        """OKX feature row (identical to research): decisions only; fills happen on the next HL row."""
        if self.buffer and row['ts'] <= self.buffer[-1]['ts']:
            return
        self.buffer.append({k: v for k, v in row.items() if not k.startswith('_')})
        for x in self.sleeves:
            if self._new_bar(x, row['ts']):
                self._decide(x, row['ts'])

    @staticmethod
    def _new_bar(x, ts):
        """Decide once per completed bar, even when a polled second is skipped."""
        step = RES_MS[x.res]
        bar = (ts + 1000) // step
        prev = getattr(x, 'last_decision_bar', None)
        x.last_decision_bar = max(bar, prev or bar)
        return prev is None or bar > prev

    def _execute(self, row):
        ts, mid = row['ts'], row['mid']
        self.last_mid = mid
        if not hasattr(self, 'hl_mids'):
            self.hl_mids = deque(maxlen=61)
        self.hl_mids.append(mid)
        hour = ts // 3_600_000
        for x in self.sleeves:
            if self.last_hour is not None and hour != self.last_hour and x.pos != 0:
                f = x.pos * mid * (row.get('funding_rate') or 0.0)
                x.cash -= f; x.funding += f
            if x.mod.EXECUTION:
                self._maker_step(x, row)
            elif x.pending is not None and x.pending['decided'] < ts:
                self._fill(x, row)
        self.last_hour = hour
        if ts % 60_000 == 0:
            self._snapshot(ts)

    def on_row(self, row):
        if self.buffer and row['ts'] <= self.buffer[-1]['ts']:
            return                               # strictly increasing time: a decision can never fill in its own second
        self.buffer.append({k: v for k, v in row.items() if not k.startswith('_')})
        ts, mid = row['ts'], row['mid']
        self.last_mid = mid
        hour = ts // 3_600_000
        for x in self.sleeves:
            if self.last_hour is not None and hour != self.last_hour and x.pos != 0:
                f = x.pos * mid * (row.get('funding_rate') or 0.0)
                x.cash -= f; x.funding += f
            if x.pending is not None:
                self._fill(x, row)
        self.last_hour = hour
        for x in self.sleeves:
            step = RES_MS[x.res]
            if ts % step == step - 1000 or step == 1000:
                self._decide(x, ts)
        if ts % 60_000 == 0:
            self._snapshot(ts)

    def _frame(self, res):
        df = pl.DataFrame(list(self.buffer), infer_schema_length=None)
        if res == '1s':
            return df
        from ..data.ingest import bars
        fr = df.select(pl.col('ts'), pl.col('funding_rate')) if 'funding_rate' in df.columns else None
        b = bars(df.drop('funding_rate') if fr is not None else df, res)
        step = RES_MS[res]
        if len(b) and (int(df['ts'][-1]) + 1000) % step != 0:
            b = b.head(len(b) - 1)                  # the last bar is still forming: decide on completed bars only
        if fr is not None:
            b = b.join(fr, on='ts', how='left').with_columns(pl.col('funding_rate').forward_fill().fill_null(0.0))
        return b

    def _decide(self, x, ts):
        df = self._frame(x.res)
        if len(df) < max(x.mod.LOOKBACK, 2):
            x.status = f'warming {len(df)}/{x.mod.LOOKBACK}'
            return
        x.status = 'live'
        try:
            sig = run_signal(x.mod, df, model=x.model)
            t = float(sig[-1])
        except Exception as e:
            self.log('errors.jsonl', {'ts': ts, 'sleeve': x.id, 'error': repr(e)[:500]})
            t = 0.0
        x.target = t
        want = t * x.unit / df['mid'][-1]
        deadband = x.cost.get('deadband', 0.05)
        if abs(want - x.pos) * df['mid'][-1] > deadband * x.unit or (t == 0 and x.pos != 0):
            x.pending = {'target': t, 'decided': ts}
        self.log('decisions.jsonl', {'ts': ts, 'sleeve': x.id, 'target': t})

    def _fill(self, x, row):
        p = row['mid']
        want = x.pending['target'] * x.unit / p
        delta = want - x.pos
        if abs(delta) < 1e-12:
            x.pending = None
            return
        hs = row['spread_bps'] / 2
        side = 'a' if delta > 0 else 'b'
        extra = _impact_extra(abs(delta), row.get(f'{side}imp_1', 0.5), row.get(f'{side}imp_10', 2.0),
                              row.get(f'{side}imp_50', 6.0), hs) * x.cost.get('impact_mult', 1.0)
        hist = getattr(self, 'hl_mids', None)
        move = abs(math.log(p / hist[0])) * 1e4 if hist and len(hist) == hist.maxlen and hist[0] > 0 else 0.0
        slip = x.cost.get('slippage_bps', 0.5) + x.cost.get('vol_slip_coef', 0.05) * move   # calibrated on live fills
        px = p * (1 + math.copysign(1, delta) * (hs + extra + slip) / 1e4)
        fee = abs(delta) * px * x.cost.get('taker_bps', 4.5) / 1e4
        x.cash -= delta * px + fee
        x.fees += fee
        x.pos = 0.0 if x.pending['target'] == 0 else want
        x.fills += 1
        self.log('fills.jsonl', {'ts': row['ts'], 'sleeve': x.id, 'qty_btc': delta, 'price': px, 'mid': p, 'fee': fee,
                                 'decided': x.pending['decided']})
        x.pending = None

    def _maker_step(self, x, row):
        """Hyperliquid limit-order simulation with the research queue model: join the back of the touch queue,
        queue ahead consumed only by aggressor trades AT our price, trade-through fills fully, ttl/repeg/exit rules."""
        ex = x.mod.EXECUTION
        ts, p = row['ts'], row['mid']
        o = getattr(x, 'order', None)
        # 1) trades of this second can fill an order placed in an earlier second
        if o is not None and o['placed'] < ts:
            for side_t, size_ct, px_t in row.get('_trades', []):
                if o['rem'] <= 1e-12:
                    break
                if side_t != -o['side']:
                    continue
                if (o['side'] > 0 and px_t < o['px'] - 1e-9) or (o['side'] < 0 and px_t > o['px'] + 1e-9):
                    q = o['rem']
                elif abs(px_t - o['px']) <= 1e-9:
                    v = size_ct / 100.0
                    if v <= o['q']:
                        o['q'] -= v; q = 0.0
                    else:
                        q = min(o['rem'], v - o['q']); o['q'] = 0.0
                else:
                    q = 0.0
                if q > 0:
                    fee = q * o['px'] * x.cost.get('maker_bps', 1.5) / 1e4
                    x.cash -= o['side'] * q * o['px'] + fee; x.fees += fee; x.pos += o['side'] * q; o['rem'] -= q
                    x.fills += 1
                    self.log('fills.jsonl', {'ts': ts, 'sleeve': x.id, 'qty_btc': o['side'] * q, 'price': o['px'], 'mid': p,
                                             'fee': fee, 'maker': True, 'decided': o['decided'], 'placed': o['placed']})
            if o['rem'] <= 1e-12:
                x.order = o = None
        # 2) age / re-peg / expiry
        if o is not None:
            o['age'] += 1
            touch = row['bid_px'] if o['side'] > 0 else row['ask_px']
            o['behind'] = o['behind'] + 1 if abs(touch - o['px']) > 1e-9 else 0
            if o['age'] > ex['ttl_s'] or o['behind'] >= ex['repeg_s']:
                if o['age'] > ex['ttl_s'] and ex['on_expiry'] == 'cross':
                    x.pending = {'target': o['tgt'], 'decided': o['decided']}
                    self._fill(x, row)
                x.order = o = None
        # 3) a new decision (made in an earlier second) reaches the venue now
        if x.pending is not None and x.pending['decided'] < ts:
            t = x.pending['target']
            if o is not None and abs(o['tgt'] - t) > 1e-9:
                x.order = o = None
            want = t * x.unit / p
            delta = want - x.pos
            reducing = abs(want) < abs(x.pos) - 1e-12 or want * x.pos < 0
            if ex['exit'] == 'taker' and reducing:
                self._fill(x, row)
            elif o is None and abs(delta) * p > x.cost.get('deadband', 0.05) * x.unit * 0.5:
                side = 1 if delta > 0 else -1
                x.order = {'side': side, 'px': row['bid_px'] if side > 0 else row['ask_px'],
                           'q': (row['bid_sz0'] if side > 0 else row['ask_sz0']) / 100.0 * x.cost.get('queue_mult', 1.0),
                           'rem': abs(delta), 'age': 0, 'behind': 0, 'tgt': t, 'placed': ts, 'decided': x.pending['decided']}
                self.log('orders.jsonl', {'ts': ts, 'sleeve': x.id, **{k: x.order[k] for k in ('side', 'px', 'q', 'rem', 'tgt')}})
            x.pending = None

    def _snapshot(self, ts):
        mid = self.last_mid
        sl = [{'id': x.id, 'status': x.status, 'target': x.target, 'pos_btc': x.pos, 'equity': x.equity(mid),
               'fees': x.fees, 'funding': x.funding, 'fills': x.fills, 'unit': x.unit} for x in self.sleeves]
        total = sum(v['equity'] for v in sl)
        self.log('equity.jsonl', {'ts': ts, 'mid': mid, 'equity': total, 'sleeves': {v['id']: v['equity'] for v in sl}})
        (self.dir / 'status.json').write_text(json.dumps({'id': self.id, 'updated': time.time(), 'started': self.started,
                                                          'kind': self.kind, 'target': self.target_id, 'pnl': total,
                                                          'hours': (time.time() - self.started) / 3600, 'sleeves': sl,
                                                          'pid': __import__('os').getpid()}, default=float, indent=1))

    async def run(self, hours):
        n = self.warm_start()
        print(f'paper {self.id}: {self.kind} {self.target_id}, {len(self.sleeves)} sleeve(s), warm rows {n}', flush=True)
        write = not recorder_alive(self.s)
        if self.feature_source == 'okx':
            from .okx_feed import okx_features
            state = {}
            await asyncio.gather(
                record_async(self.s, hours=hours, on_row=lambda r: (state.__setitem__('f', r.get('funding_rate', 0.0)), self.on_hl_row(r)),
                             write=write),
                okx_features(self.s, hours=hours, on_row=self.on_okx_row, write=write, funding=lambda: state.get('f', 0.0)))
        else:
            await record_async(self.s, hours=hours, on_row=self.on_row, write=write)
        if self.last_mid:
            self._snapshot(int(time.time() * 1000))


def run_paper(settings, target, hours=168.0, feature_source='okx'):
    bpath = Path(target) if Path(target).exists() else settings.home / 'bundles' / target
    if (bpath / 'strategy.py').exists() and "DESK = 'carry'" in (bpath / 'strategy.py').read_text(encoding='utf-8').replace('"', "'"):
        from .carry_paper import CarryPaper
        return CarryPaper(settings, str(bpath)).run(hours)
    sess = PaperSession(settings, target, feature_source=feature_source)
    asyncio.run(sess.run(hours))
    return 0
