"""Funding-regime monitor: how far are the carry sleeves from re-opening?

For every coin of the carry universe, from live APIs (no local data needed):
  spread_bps_h = trailing-504h mean of (HL hourly funding - Binance last paid rate / interval)   (carry v2 statistic)
  hl_bps_h     = trailing-504h mean of HL hourly funding                                         (spot-carry statistic)
Carry v2 enters at |spread| > 0.3 bps/h (exits below 0.15); spot-carry v1 at hl > 0.2 bps/h. Writes
results/v3/regime_monitor.json and prints the closest coins. Run: python -m desk.live.regime
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

HL = 'https://api.hyperliquid.xyz/info'
FAPI = 'https://fapi.binance.com'
W_H = 504
CARRY_ENTER, SPOT_ENTER = 0.3, 0.2


def coin_state(c, coin, sym, start_ms):
    hl = c.post(HL, json={'type': 'fundingHistory', 'coin': coin, 'startTime': start_ms}, timeout=30).json()
    rows = list(hl)
    while len(hl) >= 500:                                        # paginate (500 rows per call)
        hl = c.post(HL, json={'type': 'fundingHistory', 'coin': coin, 'startTime': int(hl[-1]['time']) + 1}, timeout=30).json()
        rows += hl
    bn = c.get(f'{FAPI}/fapi/v1/fundingRate', params={'symbol': sym, 'startTime': start_ms - 3 * 86_400_000, 'limit': 1000},
               timeout=30).json()
    if not rows or not isinstance(bn, list) or len(bn) < 3:
        return None
    ht = np.array([int(r['time']) for r in rows]); hf = np.array([float(r['fundingRate']) for r in rows])
    bt = np.array([int(r['fundingTime']) for r in bn]); br = np.array([float(r['fundingRate']) for r in bn])
    iv = np.r_[np.diff(bt)[:1], np.diff(bt)] / 3_600_000                          # hours between payments, as history
    k = np.searchsorted(bt, ht, side='right') - 1                                 # last PAID Binance rate at each HL hour
    ok = k >= 0
    spr = hf[ok] - br[k[ok]] / np.maximum(iv[k[ok]], 1)
    return {'coin': coin, 'hours': int(ok.sum()), 'spread_bps_h': float(spr[-W_H:].mean() * 1e4),
            'hl_bps_h': float(hf[-W_H:].mean() * 1e4), 'bn_interval_h': float(iv[-1])}


def snapshot(settings):
    uni = json.loads((settings.store / 'carry' / 'universe.json').read_text())
    start = int(time.time() * 1000) - (W_H + 24) * 3_600_000
    with httpx.Client() as c, ThreadPoolExecutor(6) as pool:
        res = [r for r in pool.map(lambda u: _safe(c, u['coin'], u['binance'], start), uni) if r]
    for r in res:
        r['carry_v2_gap'] = round(CARRY_ENTER - abs(r['spread_bps_h']), 3)        # <= 0 means entry condition met
        r['spot_carry_gap'] = round(SPOT_ENTER - r['hl_bps_h'], 3)
    out = {'updated': time.time(), 'coins': len(res),
           'carry_v2_open': [r['coin'] for r in res if r['carry_v2_gap'] <= 0],
           'spot_carry_open': [r['coin'] for r in res if r['spot_carry_gap'] <= 0],
           'median_abs_spread_bps_h': float(np.median([abs(r['spread_bps_h']) for r in res])) if res else None,
           'median_hl_funding_bps_h': float(np.median([r['hl_bps_h'] for r in res])) if res else None,
           'rows': sorted(res, key=lambda r: r['carry_v2_gap'])}
    (settings.home / 'regime_monitor.json').write_text(json.dumps(out, indent=1))
    return out


def _safe(c, coin, sym, start):
    for attempt in range(3):
        try:
            return coin_state(c, coin, sym, start)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def loop(settings, every_h=6.0):
    """Snapshot every `every_h` hours; history appended to results/v3/regime_history.jsonl."""
    while True:
        try:
            o = snapshot(settings)
            with open(settings.home / 'regime_history.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({k: o[k] for k in ('updated', 'coins', 'carry_v2_open', 'spot_carry_open',
                                                      'median_abs_spread_bps_h', 'median_hl_funding_bps_h')}) + '\n')
        except Exception as e:
            print('regime error', repr(e)[:300], flush=True)
        try:
            from .forward_review import review
            review(settings)
        except Exception as e:
            print('review error', repr(e)[:300], flush=True)
        time.sleep(every_h * 3600)


if __name__ == '__main__':
    import sys
    from ..config import SETTINGS
    if '--loop' in sys.argv:
        loop(SETTINGS)
    o = snapshot(SETTINGS)
    print(f"{o['coins']} coins | carry v2 open: {o['carry_v2_open'] or 'none'} | spot-carry open: {o['spot_carry_open'] or 'none'}")
    print(f"median |HL-BN spread| {o['median_abs_spread_bps_h']:.3f} bps/h (entry 0.3) | median HL funding {o['median_hl_funding_bps_h']:.3f} bps/h (entry 0.2)")
    for r in o['rows'][:10]:
        print(f"  {r['coin']:9s} spread {r['spread_bps_h']:+.3f} gap {r['carry_v2_gap']:+.3f} | HL funding {r['hl_bps_h']:+.3f} gap {r['spot_carry_gap']:+.3f} | BN interval {r['bn_interval_h']:.0f}h")
