"""Forward paper test of funding_flip_v1 (results/v3/preregistered/funding_flip_v1.json) on live Hyperliquid data.

Exactly the pre-registered rule, live:
  at hh:59:03 poll metaAndAssetCtxs -> f = funding about to be paid at the hour (same instant as the archive row)
  at the hour h: sign(f[h]) != sign(f[h-1]) and |f[h]| >= 3e-5 and day volume >= $5M -> RECEIVE the new rate
  entry at ~h:01:03 at the real touch (l2Book ask for longs, bid for shorts) + 4.5 bps taker; exit 8 h later the same
  way; funding accrued hourly at the paid rates; one position per coin; $NOTIONAL per trade.
EXPLORATORY: dev and val means were positive (+76 / +52 bps) but validation FAILED its concentration gate and the
holdout window is contaminated for this family -> this forward test is the only clean evidence left.
Run: python -m desk.live.flip_paper [hours]
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx

HL = 'https://api.hyperliquid.xyz/info'
THR, HOLD_H, MIN_VLM, NOTIONAL, TAKER_BPS = 3e-5, 8, 5e6, 1_000.0, 4.5
HOUR = 3_600_000


def ctx(c):
    meta, ctxs = c.post(HL, json={'type': 'metaAndAssetCtxs'}, timeout=20).json()
    return {a['name']: {'f': float(x.get('funding') or 0), 'vlm': float(x.get('dayNtlVlm') or 0),
                        'mid': float(x.get('midPx') or 0) or None}
            for a, x in zip(meta['universe'], ctxs) if not a.get('isDelisted')}


def touch(c, coin):
    b = c.post(HL, json={'type': 'l2Book', 'coin': coin}, timeout=20).json()['levels']
    return float(b[0][0]['px']), float(b[1][0]['px'])


def wait_until(sec_of_hour):
    now = time.time()
    t = now - now % 3600 + sec_of_hour
    if t <= now:
        t += 3600
    time.sleep(t - now)


class FlipPaper:
    def __init__(self, home: Path):
        self.id = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
        self.dir = home / 'paper' / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prev, self.open, self.pnl, self.n, self.started = {}, {}, 0.0, 0, time.time()
        (self.dir / 'session.json').write_text(json.dumps({'id': self.id, 'kind': 'funding_flip', 'target': 'funding_flip_v1',
                                                           'started': self.started, 'rule': __doc__}, indent=1))

    def log(self, name, obj):
        with open(self.dir / name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj) + '\n')

    def status(self):
        st = {'id': self.id, 'kind': 'funding_flip', 'target': 'funding_flip_v1', 'updated': time.time(),
              'started': self.started, 'hours': (time.time() - self.started) / 3600, 'pnl': self.pnl, 'closed_trades': self.n,
              'open': {c: {k: v for k, v in p.items() if k != 'fund'} for c, p in self.open.items()},
              'pid': __import__('os').getpid()}
        (self.dir / 'status.json').write_text(json.dumps(st, indent=1))

    def run(self, hours=720.0):
        end = time.time() + hours * 3600
        with httpx.Client() as c:
            while time.time() < end:
                try:
                    wait_until(59 * 60 + 3)                             # hh:59:03 - same instant as the archive row
                    snap = ctx(c)
                    h = (int(time.time() * 1000) // HOUR + 1) * HOUR    # the hour this rate is paid at
                    for coin, p in self.open.items():                   # funding paid at h on open positions
                        if coin in snap:
                            p['fund'] += -p['side'] * snap[coin]['f'] * NOTIONAL
                    signals = []
                    for coin, x in snap.items():
                        fp = self.prev.get(coin)
                        if fp is not None and fp != 0 and x['f'] != 0 and (x['f'] > 0) != (fp > 0) \
                                and abs(x['f']) >= THR and x['vlm'] >= MIN_VLM and coin not in self.open:
                            signals.append((coin, -1.0 if x['f'] > 0 else 1.0, x['f']))
                    self.prev = {k: v['f'] for k, v in snap.items()}
                    self.log('signals.jsonl', {'h': h, 'signals': signals})
                    wait_until(60 + 3)                                  # h:01:03
                    now_ms = int(time.time() * 1000)
                    for coin, p in list(self.open.items()):             # exits due
                        if now_ms >= p['exit_due']:
                            bid, ask = touch(c, coin)
                            px = bid if p['side'] > 0 else ask
                            price = p['side'] * (px / p['entry_px'] - 1) * NOTIONAL
                            fee = TAKER_BPS / 1e4 * NOTIONAL * 2
                            net = price + p['fund'] - fee
                            self.pnl += net; self.n += 1
                            self.log('trades.jsonl', {**p, 'exit_px': px, 'exit_ms': now_ms, 'price_usd': price,
                                                      'fee_usd': fee, 'net_usd': net, 'net_bps': net / NOTIONAL * 1e4})
                            del self.open[coin]
                    for coin, side, f in signals:
                        bid, ask = touch(c, coin)
                        self.open[coin] = {'coin': coin, 'side': side, 'f': f, 'signal_h': h, 'entry_ms': now_ms,
                                           'entry_px': ask if side > 0 else bid, 'spread_bps': (ask / bid - 1) * 1e4,
                                           'exit_due': now_ms + HOLD_H * HOUR - 30_000, 'fund': 0.0}
                    self.status()
                except Exception as e:
                    self.log('errors.jsonl', {'ts': time.time(), 'error': repr(e)[:400]})
                    time.sleep(30)


if __name__ == '__main__':
    from ..config import SETTINGS
    FlipPaper(SETTINGS.home).run(float(sys.argv[1]) if len(sys.argv) > 1 else 720.0)
