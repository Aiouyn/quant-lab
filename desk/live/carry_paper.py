"""Live dry run for carry strategies: Hyperliquid perp vs Binance perp, both legs simulated at live marks.

Every minute: poll HL metaAndAssetCtxs (funding for the current hour, premium, mark) and Binance premiumIndex
(mark, index, last funding rate, next funding time) + 24h quote volume. At each hour boundary:
  1. settle HL funding (rate of the hour that just ended) and Binance funding if a Binance funding time passed;
  2. mark both legs to market at live prices (real basis PnL, not the premium approximation);
  3. append an hourly panel row per coin (same schema as research) and re-run the strategy;
  4. trade to the new positions: taker fees + slippage on both legs, gross-capped like the backtest.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from ..lab.carry import CarryCost, MAJORS, carry_signal, PANEL_COLS
from ..pipeline.bundle import load_bundle

HL = 'https://api.hyperliquid.xyz/info'
FAPI = 'https://fapi.binance.com'
HOUR = 3_600_000


def snapshot(client, coins, bn_symbols):
    """Pairs are 'COIN' (hedge on Binance) or 'COIN@BY' (hedge on Bybit); the HL leg is keyed by the pair name."""
    base = {c.split('@')[0] for c in coins}
    meta, ctx = client.post(HL, json={'type': 'metaAndAssetCtxs'}, timeout=20).json()
    hl = {}
    for i, a in enumerate(meta['universe']):
        if a['name'] in base:
            c = ctx[i]
            hl[a['name']] = {'funding': float(c.get('funding') or 0), 'premium': float(c.get('premium') or 0),
                             'mark': float(c.get('markPx') or 0)}
    for c in coins:                                            # HL leg is shared by a coin's pairs
        if '@' in c and c.split('@')[0] in hl:
            hl[c] = hl[c.split('@')[0]]
    pi = {x['symbol']: x for x in client.get(f'{FAPI}/fapi/v1/premiumIndex', timeout=20).json()}
    tk = {x['symbol']: x for x in client.get(f'{FAPI}/fapi/v1/ticker/24hr', timeout=20).json()}
    bn = {}
    if any(c.endswith('@BY') for c in coins):
        by = {x['symbol']: x for x in client.get('https://api.bybit.com/v5/market/tickers', params={'category': 'linear'},
                                                 timeout=20).json()['result']['list']}
        for c in coins:
            if c.endswith('@BY'):
                x = by.get(c.split('@')[0] + 'USDT')
                if x and x.get('fundingRate'):
                    mark, index = float(x['markPrice']), float(x['indexPrice'] or x['markPrice'])
                    bn[c] = {'mark': mark, 'premium': (mark / index - 1) if index else 0.0, 'rate': float(x['fundingRate']),
                             'next': int(x['nextFundingTime']), 'qvol_h': float(x.get('turnover24h') or 0) / 24}
    for c in coins:
        if c.endswith('@BY'):
            continue
        s = bn_symbols.get(c)
        if s in pi:
            mark, index = float(pi[s]['markPrice']), float(pi[s]['indexPrice'])
            bn[c] = {'mark': mark, 'premium': (mark / index - 1) if index else 0.0, 'rate': float(pi[s]['lastFundingRate']),
                     'next': int(pi[s]['nextFundingTime']), 'qvol_h': float(tk.get(s, {}).get('quoteVolume', 0) or 0) / 24}
    return hl, bn


class CarryPaper:
    def __init__(self, settings, target, notional=100_000.0):
        self.s = settings
        bpath = Path(target) if Path(target).exists() else settings.home / 'bundles' / target
        self.manifest, self.mod, _ = load_bundle(bpath)
        if self.mod.DESK != 'carry':
            raise ValueError('not a carry strategy bundle')
        self.cost = CarryCost(**{k: v for k, v in (self.manifest.get('carry_cost') or {}).items()})
        self.cost.unit_notional = notional
        uni = json.loads((settings.store / 'carry' / 'universe.json').read_text())
        self.bn_symbols = {u['coin']: u['binance'] for u in uni}
        from ..data.carry_data import load_panel
        import types
        hist = load_panel(types.SimpleNamespace(store=settings.store))
        # pairs: 'binance' (COIN), 'bybit' (COIN@BY) or 'all' - must match what the strategy was evaluated on
        venue = (self.manifest.get('pairs') or 'binance')
        coins = sorted(hist['coin'].unique().to_list())
        if venue == 'binance':
            coins = [c for c in coins if '@' not in c]
        elif venue == 'bybit':
            coins = [c for c in coins if c.endswith('@BY')]
        hist = hist.filter(pl.col('coin').is_in(coins))
        self.coins = coins
        cutoff = hist['ts'].max() - 90 * 86_400_000
        self.panel = hist.filter(pl.col('ts') >= cutoff).select([c for c in PANEL_COLS if c in hist.columns])
        # live inputs must be the SAME quantities as history: last PAID hedge-venue rate and the observed interval between
        # payments (premiumIndex.lastFundingRate is the predicted current-period rate, and intervals are 4 h or 8 h)
        tail = self.panel.sort('ts').group_by('coin').agg(pl.col('bn_rate_last').last(), pl.col('bn_interval_h').last())
        self.bn_paid = {r['coin']: r['bn_rate_last'] for r in tail.iter_rows(named=True)}
        self.bn_iv = {r['coin']: r['bn_interval_h'] for r in tail.iter_rows(named=True)}
        self.bn_paid_at = {}
        self.N = {c: 0.0 for c in self.coins}                  # signed USD notional (+ = long HL / short BN)
        self.cash, self.fees, self.funding = 0.0, 0.0, 0.0
        self.last = {}                                          # last marks per coin
        self.last_bn_next = {}
        self.id = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
        self.dir = settings.home / 'paper' / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        (self.dir / 'session.json').write_text(json.dumps({'id': self.id, 'kind': 'carry', 'target': self.manifest['id'],
                                                           'started': self.started, 'coins': self.coins,
                                                           'execution_venues': 'hyperliquid + binance (simulated)'}, indent=1))

    def log(self, name, obj):
        with open(self.dir / name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, default=float) + '\n')

    def mtm(self, hl, bn):
        pnl = 0.0
        for c, n in self.N.items():
            if n == 0 or c not in hl or c not in bn or c not in self.last:
                continue
            l = self.last[c]
            pnl += n * (hl[c]['mark'] / l['hl'] - 1) - n * (bn[c]['mark'] / l['bn'] - 1)
        return pnl

    def hour(self, ts, hl, bn, hl_hour_rates):
        # 1) funding settlement for the hour that just ended
        for c, n in self.N.items():
            if n == 0:
                continue
            if c in hl_hour_rates:
                f = -n * hl_hour_rates[c]; self.cash += f; self.funding += f
            if c in bn and c in self.last_bn_next and self.last_bn_next[c] <= ts:
                f = n * bn[c]['rate_settled']; self.cash += f; self.funding += f
        # 2) mark to market
        self.cash += self.mtm(hl, bn)
        for c in self.coins:
            if c in hl and c in bn:
                self.last[c] = {'hl': hl[c]['mark'], 'bn': bn[c]['mark']}
        # 3) panel rows
        rows = []
        for c in self.coins:
            if c not in hl or c not in bn:
                continue
            paid = c in self.last_bn_next and self.last_bn_next[c] <= ts
            if paid:
                ev = self.last_bn_next[c]
                if c in self.bn_paid_at and ev > self.bn_paid_at[c]:
                    self.bn_iv[c] = round((ev - self.bn_paid_at[c]) / HOUR)
                self.bn_paid_at[c], self.bn_paid[c] = ev, bn[c]['rate_settled']
            rows.append({'ts': ts, 'coin': c, 'hl_funding': hl_hour_rates.get(c, hl[c]['funding']),
                         'hl_premium': hl[c]['premium'], 'bn_funding': bn[c]['rate_settled'] if paid else 0.0,
                         'bn_rate_last': self.bn_paid.get(c, bn[c]['rate']), 'bn_interval_h': self.bn_iv.get(c, 8.0),
                         'bn_premium': bn[c]['premium'],
                         'bn_close': bn[c]['mark'], 'bn_qvol': bn[c]['qvol_h']})
            self.last_bn_next[c] = bn[c]['next']
        if rows:
            self.panel = pl.concat([self.panel, pl.DataFrame(rows).select(self.panel.columns)], how='vertical_relaxed') \
                .unique(['ts', 'coin'], keep='last').sort(['ts', 'coin'])
        # 4) new positions from the strategy (on the latest hour only), gross-capped like the backtest.
        #    maker execution: targets decided now are posted passively and fill at the NEXT hour (as in the backtest)
        pos = carry_signal(self.mod, self.panel)
        latest = self.panel.with_columns(pl.Series('pos', pos)).filter(pl.col('ts') == ts)
        decided = {r['coin']: r['pos'] for r in latest.iter_rows(named=True)}
        maker = getattr(self.mod, 'CARRY_EXECUTION', 'taker') == 'maker'
        if maker:
            s, self.pending = getattr(self, 'pending', {}), decided
            s = {**{c: 0.0 for c, v in self.N.items() if v}, **s}
        else:
            s = decided
        gross = max(1.0, sum(abs(v) for v in s.values()) / self.cost.max_gross)
        for c, v in s.items():
            want = self.cost.unit_notional * v / gross
            d = abs(want - self.N.get(c, 0.0))
            if d > 1.0 and c in hl and c in bn:
                if maker:
                    fee = d * (self.cost.hl_maker_bps + self.cost.bn_maker_bps + 2 * self.cost.maker_adverse_bps) / 1e4
                else:
                    slip = (self.cost.major_slip_bps if c in MAJORS else self.cost.alt_slip_bps) + \
                        min(self.cost.impact_coef * d / max(bn.get(c, {}).get('qvol_h', 1e5), 1e5), 50.0)
                    fee = d * (self.cost.hl_taker_bps + self.cost.bn_taker_bps + 2 * slip) / 1e4
                self.cash -= fee; self.fees += fee
                self.log('fills.jsonl', {'ts': ts, 'coin': c, 'from': self.N.get(c, 0.0), 'to': want, 'cost': fee,
                                         'hl_mark': hl[c]['mark'], 'bn_mark': bn[c]['mark']})
                self.N[c] = want
        self.status(ts)

    def status(self, ts):
        st = {'id': self.id, 'kind': 'carry', 'target': self.manifest['id'], 'updated': time.time(), 'started': self.started,
              'hours': (time.time() - self.started) / 3600, 'pnl': self.cash, 'fees': self.fees, 'funding': self.funding,
              'gross_notional': sum(abs(v) for v in self.N.values()),
              'positions': {c: round(v, 2) for c, v in self.N.items() if v}, 'pid': __import__('os').getpid(),
              'sleeves': [{'id': self.manifest['id'], 'status': 'live', 'equity': self.cash, 'unit': self.cost.unit_notional,
                           'target': None, 'pos_btc': None, 'fills': None}]}
        (self.dir / 'status.json').write_text(json.dumps(st, indent=1, default=float))
        self.log('equity.jsonl', {'ts': ts, 'equity': self.cash, 'funding': self.funding, 'fees': self.fees})

    def run(self, hours=168.0):
        end = time.time() + hours * 3600
        hl_rates, prev_bn, last_hour = {}, {}, None
        with httpx.Client() as client:
            while time.time() < end:
                try:
                    hl, bn = snapshot(client, set(self.coins), self.bn_symbols)
                    now = int(time.time() * 1000)
                    hour = now // HOUR * HOUR
                    # a boundary poll already shows the NEXT period's rates: settle with the previous poll's values
                    for c in bn:
                        bn[c]['rate_settled'] = prev_bn.get(c, bn[c]['rate'])
                    if last_hour is None:
                        last_hour = hour
                        for c in self.coins:
                            if c in hl and c in bn:
                                self.last[c] = {'hl': hl[c]['mark'], 'bn': bn[c]['mark']}
                                self.last_bn_next[c] = bn[c]['next']
                        self.hour(hour, hl, bn, {})     # initial positions (no settlement yet)
                    elif hour > last_hour:
                        self.hour(hour, hl, bn, dict(hl_rates))
                        last_hour = hour
                    hl_rates = {c: hl[c]['funding'] for c in hl}
                    prev_bn = {c: bn[c]['rate'] for c in bn}
                except Exception as e:
                    self.log('errors.jsonl', {'ts': time.time(), 'error': repr(e)[:400]})
                time.sleep(60)
        return 0
