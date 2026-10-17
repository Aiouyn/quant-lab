"""Forward paper test of spot_carry_v1 (results/v3/preregistered/spot_carry_v1.json): short HL perp + long Binance spot.

Pre-registered rule, live (hourly):
  stat = trailing 504 h mean of HL hourly funding (fundingHistory, then the rate polled at hh:59:03 each hour)
  enter when stat > 0.2 bps/h and Binance perp 24 h quote volume / 24 >= $1M; exit when stat < 0.1 bps/h or illiquid
  gross cap 1.0: per-coin notional = CAPITAL / max(1, number of open coins)
  maker execution (as validated): targets decided at hour h fill at hour h+1 prices; fees HL maker 1.5 bps + Binance
  spot 10 bps + 0.5 bps adverse per leg on traded notional.
PnL each hour: funding received by the short (N x rate) + short-perp leg (-N x HL mid change) + spot leg (+N x spot change).
STATUS: v1 passed validation (+7.1%/yr) and FAILED holdout (-2.8%/yr, low-funding regime). This forward test checks
whether it works now that the regime monitor shows HL funding back near 15%/yr. Run: python -m desk.live.spot_carry_paper
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import httpx

HL = 'https://api.hyperliquid.xyz/info'
FAPI = 'https://fapi.binance.com'
SPOT = 'https://api.binance.com'
W_H, ENTER, EXIT, MIN_QV_H, CAPITAL = 504, 0.2, 0.1, 1e6, 100_000.0
FEE_BPS = 1.5 + 10.0 + 2 * 0.5
HOUR = 3_600_000


def wait_until(sec_of_hour):
    now = time.time()
    t = now - now % 3600 + sec_of_hour
    if t <= now:
        t += 3600
    time.sleep(t - now)


class SpotCarryPaper:
    def __init__(self, settings):
        self.home = settings.home
        self.coins = json.loads((settings.home / 'spot_universe.json').read_text())
        self.id = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
        self.dir = settings.home / 'paper' / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hist = {c: deque(maxlen=W_H) for c in self.coins}
        self.N, self.pending, self.last = {}, None, {}
        self.cash = self.funding = self.fees = 0.0
        self.started = time.time()
        (self.dir / 'session.json').write_text(json.dumps({'id': self.id, 'kind': 'spot_carry', 'target': 'spot_carry_v1',
                                                           'started': self.started, 'coins': self.coins, 'rule': __doc__}, indent=1))

    def log(self, name, obj):
        with open(self.dir / name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj) + '\n')

    def warm(self, c):
        start = int(time.time() * 1000) - (W_H + 2) * HOUR
        for coin in self.coins:
            try:
                rows = c.post(HL, json={'type': 'fundingHistory', 'coin': coin, 'startTime': start}, timeout=30).json()
                while rows and len(rows) % 500 == 0:
                    more = c.post(HL, json={'type': 'fundingHistory', 'coin': coin, 'startTime': int(rows[-1]['time']) + 1},
                                  timeout=30).json()
                    if not more:
                        break
                    rows += more
                self.hist[coin].extend(float(r['fundingRate']) for r in rows)
            except Exception as e:
                self.log('errors.jsonl', {'ts': time.time(), 'warm': coin, 'error': repr(e)[:200]})
            time.sleep(0.3)

    def prices(self, c):
        meta, ctx = c.post(HL, json={'type': 'metaAndAssetCtxs'}, timeout=20).json()
        hl = {a['name']: {'mid': float(x.get('midPx') or 0), 'f': float(x.get('funding') or 0)} for a, x in zip(meta['universe'], ctx)}
        spot = {x['symbol']: float(x['price']) for x in c.get(f'{SPOT}/api/v3/ticker/price', timeout=20).json()}
        qv = {x['symbol']: float(x.get('quoteVolume') or 0) / 24 for x in c.get(f'{FAPI}/fapi/v1/ticker/24hr', timeout=20).json()}
        return hl, spot, qv

    def step(self, c):
        hl, spot, qv = self.prices(c)                         # polled at hh:59:03: rates about to be paid at the hour
        for coin, n in self.N.items():                        # 1) funding paid at the hour + mark to market
            if coin in hl and n:
                fnd = n * hl[coin]['f']; self.cash += fnd; self.funding += fnd
        for coin, n in self.N.items():
            sym = f'{coin}USDT'
            if n and coin in self.last and coin in hl and sym in spot and hl[coin]['mid'] > 0:
                l = self.last[coin]
                self.cash += -n * (hl[coin]['mid'] / l['hl'] - 1) + n * (spot[sym] / l['spot'] - 1)
        for coin in self.coins:
            sym = f'{coin}USDT'
            if coin in hl and sym in spot and hl[coin]['mid'] > 0:
                self.last[coin] = {'hl': hl[coin]['mid'], 'spot': spot[sym]}
            if coin in hl:
                self.hist[coin].append(hl[coin]['f'])
        if self.pending is not None:                          # 2) maker fills of the targets decided an hour ago
            for coin, want in self.pending.items():
                d = abs(want - self.N.get(coin, 0.0))
                if d > 1.0:
                    fee = d * FEE_BPS / 1e4; self.cash -= fee; self.fees += fee
                    self.log('fills.jsonl', {'ts': time.time(), 'coin': coin, 'from': self.N.get(coin, 0.0), 'to': want, 'fee': fee})
                    self.N[coin] = want
        state = {}                                            # 3) new targets (filled next hour)
        for coin in self.coins:
            h = self.hist[coin]
            stat = sum(h) / len(h) * 1e4 if len(h) >= W_H // 2 else None
            liquid = qv.get(f'{coin}USDT', 0) >= MIN_QV_H and f'{coin}USDT' in spot
            was = self.N.get(coin, 0.0) > 0 or (self.pending or {}).get(coin, 0.0) > 0
            on = stat is not None and liquid and (stat > ENTER or (was and stat >= EXIT))
            state[coin] = on
        n_on = sum(state.values())
        self.pending = {coin: (CAPITAL / max(1, n_on) if on else 0.0) for coin, on in state.items()}
        st = {'id': self.id, 'kind': 'spot_carry', 'target': 'spot_carry_v1', 'updated': time.time(), 'started': self.started,
              'hours': (time.time() - self.started) / 3600, 'pnl': self.cash, 'funding': self.funding, 'fees': self.fees,
              'positions': {k: round(v) for k, v in self.N.items() if v}, 'targets_next_hour': sorted(k for k, v in state.items() if v),
              'pid': __import__('os').getpid()}
        (self.dir / 'status.json').write_text(json.dumps(st, indent=1))
        self.log('equity.jsonl', {'ts': time.time(), 'equity': self.cash, 'funding': self.funding, 'fees': self.fees,
                                  'gross': sum(self.N.values())})

    def run(self, hours=2160.0):
        end = time.time() + hours * 3600
        with httpx.Client() as c:
            self.warm(c)
            while time.time() < end:
                wait_until(59 * 60 + 3)
                try:
                    self.step(c)
                except Exception as e:
                    self.log('errors.jsonl', {'ts': time.time(), 'error': repr(e)[:400]})


if __name__ == '__main__':
    from ..config import SETTINGS
    SpotCarryPaper(SETTINGS).run(float(sys.argv[1]) if len(sys.argv) > 1 else 2160.0)
