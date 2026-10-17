"""Independent, deliberately simple event-loop re-implementation of the fill/cost/funding contract.

Used to certify the numba engine on submitted strategies (parity within tolerance) so that a bug in one
implementation cannot silently promote a strategy. Written from the contract, not from backtest.py.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from .backtest import CostModel, RES_SECONDS


def _avg_extra_bps(size_btc, imp1, imp10, imp50, half_spread):
    pts = [(0.0, half_spread), (1.0, imp1), (10.0, imp10), (50.0, imp50)]
    if size_btc > 50.0:
        worst = imp50 * size_btc / 50.0
    else:
        worst = half_spread
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if size_btc <= x1:
                worst = y0 + (y1 - y0) * (size_btc - x0) / (x1 - x0)
                break
    return max(0.0, (worst - half_spread) / 2.0)


def reference_backtest(df: pl.DataFrame, target, resolution: str, cost: CostModel):
    rows = df.to_dicts()
    res_s = RES_SECONDS[resolution]
    by_ts = {r['ts']: r for r in rows}
    lookback = max(1, int(round(60 / res_s)))
    cash, pos, fees, funding_paid = 0.0, 0.0, 0.0, 0.0
    equity = []
    prev_hour = None
    for i, r in enumerate(rows):
        hour = r['ts'] // 3_600_000
        if cost.funding and prev_hour is not None and hour != prev_hour and pos != 0.0:
            f = pos * r['mid'] * r.get('funding_rate', 0.0)
            cash -= f; funding_paid += f
        prev_hour = hour
        t = float(target[i])
        t = 0.0 if math.isnan(t) else max(-1.0, min(1.0, t))
        # fill price: mid observed `latency` later (1s data) or next bar's first mid (bars)
        if res_s == 1:
            nxt = by_ts.get(r['ts'] + 1000 * max(1, cost.latency_s))
            p = nxt['mid'] if nxt is not None else None
        else:
            extra = max(0, int(round((cost.latency_s - 1) / res_s)))
            j = i + 1 + extra
            p = rows[j].get('open', rows[j]['mid']) if j < len(rows) and rows[j]['ts'] - r['ts'] <= (1 + extra) * res_s * 1000 + 5000 else None
        if p is not None and r.get('valid', 1) >= 0.5:
            want = t * cost.unit_notional / p
            delta = want - pos
            if abs(delta) * p > cost.deadband * cost.unit_notional or (t == 0.0 and pos != 0.0):
                hs = r.get('spread_bps', 0.01) / 2
                side = 'a' if delta > 0 else 'b'
                extra_bps = _avg_extra_bps(abs(delta), r.get(f'{side}imp_1', 0.5), r.get(f'{side}imp_10', 2.0),
                                           r.get(f'{side}imp_50', 6.0), hs) * cost.impact_mult
                move = abs(math.log(r['mid'] / rows[i - lookback]['mid'])) * 1e4 if i >= lookback else 0.0
                slip = cost.slippage_bps + getattr(cost, 'vol_slip_coef', 0.0) * move
                px = p * (1 + math.copysign(1, delta) * (hs + extra_bps + slip) / 1e4)
                fee = abs(delta) * px * cost.taker_bps / 1e4
                cash -= delta * px + fee
                fees += fee
                pos = 0.0 if t == 0.0 else want
        equity.append(cash + pos * r['mid'])
    return {'pnl': equity[-1], 'fees': fees, 'funding_paid': funding_paid, 'equity': np.array(equity)}


def certify(df, target, resolution, cost, engine_result, tol=1e-6):
    ref = reference_backtest(df, target, resolution, cost)
    scale = max(1.0, abs(ref['pnl']), cost.unit_notional * 1e-4)
    diff = abs(ref['pnl'] - engine_result['pnl']) / scale
    return {'ok': diff < tol or abs(ref['pnl'] - engine_result['pnl']) < 1e-4 * cost.unit_notional / 100,
            'engine_pnl': engine_result['pnl'], 'reference_pnl': ref['pnl'], 'relative_diff': diff}


def reference_maker(rows, tape_by_ts, target, cost, ex):
    """Plain-Python re-implementation of the limit-order contract (see desk/lab/maker.py docstring).
    rows: list of dicts (1 s); tape_by_ts: {ts: [(px_tick, side, size_contracts), ...]}; target per row."""
    tick, ct = 0.1, 100.0
    cash = pos = fees = 0.0
    order = None                      # dict(side, px, q, rem, age, behind, tgt, placed)
    lag = max(1, int(cost.latency_s))
    eq = []
    prev_hour = None

    def taker(i, delta):
        r = rows[i]
        hs = r['spread_bps'] / 2
        s = 'a' if delta > 0 else 'b'
        extra = _avg_extra_bps(abs(delta), r[f'{s}imp_1'], r[f'{s}imp_10'], r[f'{s}imp_50'], hs) * cost.impact_mult
        px = r['mid'] * (1 + math.copysign(1, delta) * (hs + extra + cost.slippage_bps) / 1e4)
        return px, abs(delta) * px * cost.taker_bps / 1e4

    for i, r in enumerate(rows):
        m = r['mid']
        hour = r['ts'] // 3_600_000
        if cost.funding and prev_hour is not None and hour != prev_hour and pos != 0.0:
            cash -= pos * m * r.get('funding_rate', 0.0)
        prev_hour = hour
        if order is not None and order['placed'] < i:
            for px_t, side_t, size_t in tape_by_ts.get(r['ts'], []):
                if order['rem'] <= 1e-12:
                    break
                if side_t != -order['side']:
                    continue
                if (order['side'] > 0 and px_t < order['px']) or (order['side'] < 0 and px_t > order['px']):
                    q = order['rem']
                elif px_t == order['px']:
                    v = size_t / ct
                    if v <= order['q']:
                        order['q'] -= v; q = 0.0
                    else:
                        q = min(order['rem'], v - order['q']); order['q'] = 0.0
                else:
                    q = 0.0
                if q > 0:
                    px = order['px'] * tick
                    fee = q * px * cost.maker_bps / 1e4
                    cash -= order['side'] * q * px + fee; fees += fee
                    pos += order['side'] * q; order['rem'] -= q
            if order['rem'] <= 1e-12:
                order = None
        if order is not None:
            order['age'] += 1
            touch = round((r['bid_px'] if order['side'] > 0 else r['ask_px']) / tick)
            order['behind'] = order['behind'] + 1 if touch != order['px'] else 0
            if order['age'] > ex['ttl_s'] and ex['on_expiry'] == 'cross' and r.get('valid', 1) >= 0.5:
                d = order['side'] * order['rem']
                px, fee = taker(i, d)
                cash -= d * px + fee; fees += fee; pos += d
                order = None
            elif order['age'] > ex['ttl_s'] or order['behind'] >= ex['repeg_s']:
                order = None
        if i >= lag and r.get('valid', 1) >= 0.5 and rows[i - lag].get('valid', 1) >= 0.5:
            t = target[i - lag]
            t = 0.0 if t != t else t
            want = t * cost.unit_notional / m
            if order is not None and abs(order['tgt'] - t) > 1e-9:
                order = None
            pending = order['side'] * order['rem'] if order is not None else 0.0
            delta = want - pos
            reducing = abs(want) < abs(pos) - 1e-12 or want * pos < 0
            if abs(delta - pending) * m > cost.deadband * cost.unit_notional or (t == 0.0 and pos != 0.0 and order is None):
                if ex['exit'] == 'taker' and reducing:
                    order = None
                    px, fee = taker(i, delta)
                    cash -= delta * px + fee; fees += fee; pos += delta
                elif order is None and abs(delta) * m > cost.deadband * cost.unit_notional * 0.5:
                    side = 1 if delta > 0 else -1
                    order = {'side': side, 'px': round((r['bid_px'] if side > 0 else r['ask_px']) / tick),
                             'q': (r['bid_sz0'] if side > 0 else r['ask_sz0']) / ct * cost.queue_mult,
                             'rem': abs(delta), 'age': 0, 'behind': 0, 'tgt': t, 'placed': i}
        eq.append(cash + pos * m)
    return {'pnl': eq[-1], 'fees': fees, 'equity': eq}
