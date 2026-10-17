"""Forward paper of the HL-native cash-and-carry (portfolio margin): long spot + short perp on Hyperliquid, BTC and HYPE.

Why: cross-venue carry needs ~2x capital (the short is margined apart from its spot hedge). HL portfolio margin
(docs: trading/portfolio-margin; BTC LTV 0.5, HYPE LTV 0.65 collateral) offsets spot and perp pnl in one account,
so capital ~ 1.15 x notional. Backtest 2023-06..2026-10, ALWAYS ON (no parameters): every half-year positive for
both coins on 1.15x capital (BTC +2.4..+26%/yr, HYPE +7.2..+102%/yr; current half +7.0% / +8.4%).
Paper model (hourly): equal notional per coin; spot leg priced at the HL oracle (spot index), perp leg at HL mid;
funding received by the short at the rate polled at hh:59:03; entry fees 4.5 bps perp + 7 bps spot (assumed).
Run: python -m desk.live.hl_carry_paper [hours]
"""
from __future__ import annotations

import json
import sys
import time
import uuid

import httpx

HL = 'https://api.hyperliquid.xyz/info'
COINS = ('BTC', 'HYPE')
NOTIONAL_PER_COIN, CAPITAL_MULT, ENTRY_FEE_BPS = 50_000.0, 1.15, 4.5 + 7.0


def wait_until(sec_of_hour):
    now = time.time()
    t = now - now % 3600 + sec_of_hour
    if t <= now:
        t += 3600
    time.sleep(t - now)


def snap(c):
    meta, ctx = c.post(HL, json={'type': 'metaAndAssetCtxs'}, timeout=20).json()
    return {a['name']: {'mid': float(x['midPx']), 'oracle': float(x['oraclePx']), 'f': float(x['funding'])}
            for a, x in zip(meta['universe'], ctx) if a['name'] in COINS}


def run(home, hours=2160.0):
    sid = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
    d = home / 'paper' / sid
    d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (d / 'session.json').write_text(json.dumps({'id': sid, 'kind': 'hl_carry', 'target': 'hl_native_carry_btc_hype',
                                                'started': started, 'rule': __doc__}, indent=1))
    cash = funding = 0.0
    fees = ENTRY_FEE_BPS / 1e4 * NOTIONAL_PER_COIN * len(COINS)
    cash -= fees
    end = started + hours * 3600
    with httpx.Client() as c:
        last = snap(c)
        while time.time() < end:
            try:
                wait_until(59 * 60 + 3)
                s = snap(c)
                for coin in COINS:
                    n = NOTIONAL_PER_COIN
                    fnd = n * s[coin]['f']                                    # short receives positive funding
                    basis = -n * (s[coin]['mid'] / last[coin]['mid'] - 1) + n * (s[coin]['oracle'] / last[coin]['oracle'] - 1)
                    cash += fnd + basis; funding += fnd
                last = s
                days = (time.time() - started) / 86400
                st = {'id': sid, 'kind': 'hl_carry', 'target': 'hl_native_carry_btc_hype', 'updated': time.time(),
                      'started': started, 'hours': days * 24, 'pnl': cash, 'funding': funding, 'fees': fees,
                      'positions': {k: -NOTIONAL_PER_COIN for k in COINS},
                      'apr_on_capital': cash / (NOTIONAL_PER_COIN * len(COINS) * CAPITAL_MULT) / max(days, 1e-9) * 365,
                      'pid': __import__('os').getpid()}
                (d / 'status.json').write_text(json.dumps(st, indent=1))
                with open(d / 'equity.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'ts': time.time(), 'equity': cash, 'funding': funding, 'fees': fees,
                                        'rates': {k: s[k]['f'] for k in COINS}}) + '\n')
            except Exception as e:
                with open(d / 'errors.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'ts': time.time(), 'error': repr(e)[:300]}) + '\n')


if __name__ == '__main__':
    from ..config import SETTINGS
    run(SETTINGS.home, float(sys.argv[1]) if len(sys.argv) > 1 else 2160.0)
