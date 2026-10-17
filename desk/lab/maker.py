"""Queue-aware limit-order (maker) execution on the 1 s timeline, driven by the price-level OKX trade tape.

Model (conservative, designed for 1 Hz book data):
- A decision at row i reaches the venue 1 s later (row j = i+1). The order joins the BACK of the queue at the
  touch (buy at best bid, sell at best ask) with queue_ahead = displayed size at that level x queue_mult.
- Trades are only visible per second (tape rows attach trades of (k-1, k] to row k). From row j+1 on:
    aggressor trades AT our price consume queue_ahead first, then fill us;
    any aggressor trade THROUGH our price (worse for the aggressor than our level) fills us completely.
  Cancellations ahead of us are ignored (pessimistic: real queues also shrink by cancels).
- If the touch moves away from our price for >= repeg_s seconds we cancel and re-join the new touch (back of queue).
- After ttl_s seconds any unfilled remainder is cancelled (on_expiry='cancel') or crossed as taker ('cross').
- exit='taker' makes position reductions immediate taker orders; entries stay passive.
- Maker fills pay maker_bps; taker crosses pay half spread + depth impact + slippage + taker_bps.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

from .backtest import CostModel, _impact, summarize

TICK = 0.1
CT_PER_BTC = 100.0
DEFAULT_EXECUTION = {'mode': 'maker', 'ttl_s': 60, 'on_expiry': 'cancel', 'repeg_s': 5, 'exit': 'maker'}
EXEC_COLS = ['ts', 'valid', 'mid', 'bid_px', 'ask_px', 'bid_sz0', 'ask_sz0', 'spread_bps',
             'bimp_1', 'aimp_1', 'bimp_10', 'aimp_10', 'bimp_50', 'aimp_50', 'funding_rate']

# state vector layout (carried across day blocks)
S_CASH, S_POS, S_ACTIVE, S_SIDE, S_PX, S_Q, S_REM, S_AGE, S_BEHIND, S_TGT, S_FEES, S_FUND, S_PLACE = range(13)
STATE_SIZE = 13


def validate_execution(ex):
    if ex is None:
        return None
    if not isinstance(ex, dict) or ex.get('mode', 'taker') not in ('taker', 'maker'):
        raise ValueError("EXECUTION must be a dict with mode 'taker' or 'maker'")
    if ex.get('mode', 'taker') == 'taker':
        return None
    out = dict(DEFAULT_EXECUTION, **ex)
    if not (1 <= int(out['ttl_s']) <= 3600) or not (1 <= int(out['repeg_s']) <= 600):
        raise ValueError('ttl_s must be 1..3600 and repeg_s 1..600 seconds')
    if out['on_expiry'] not in ('cancel', 'cross') or out['exit'] not in ('maker', 'taker'):
        raise ValueError("on_expiry must be 'cancel'|'cross' and exit 'maker'|'taker'")
    return out


@nb.njit(cache=True)
def _taker_px(p, delta, hs, b1, b10, b50, a1, a10, a50, impact_mult, slip):
    size = abs(delta)
    if delta > 0:
        return p * (1.0 + (hs + _impact(size, a1, a10, a50, hs) * impact_mult + slip) / 1e4)
    return p * (1.0 - (hs + _impact(size, b1, b10, b50, hs) * impact_mult + slip) / 1e4)


@nb.njit(cache=True)
def _run_maker(mid, bid, ask, bsz, asz, hs, b1, b10, b50, a1, a10, a50, valid, funding, target,
               t_start, t_end, t_px, t_side, t_size, st, unit, deadband, maker_bps, taker_bps, slip, impact_mult,
               queue_mult, ttl, repeg, cross_on_expiry, exit_taker, lag, fills_out):
    """One block of 1 s rows. Returns (equity, pos, n_fills). fills_out rows: row, qty_btc, px, maker(1/0), placed_row."""
    n = len(mid)
    equity = np.empty(n)
    pos_out = np.empty(n)
    nf = 0
    for i in range(n):
        m = mid[i]
        if funding[i] != 0.0 and st[S_POS] != 0.0:
            f = st[S_POS] * m * funding[i]
            st[S_CASH] -= f
            st[S_FUND] += f
        # ---- 1) resting order: trades during the last second (attached to this row) may fill it
        if st[S_ACTIVE] == 1.0 and st[S_PLACE] < i:     # live since an earlier row: this second's trades count
            side = st[S_SIDE]; P = st[S_PX]
            for k in range(t_start[i], t_end[i]):
                if st[S_REM] <= 1e-12:
                    break
                # a resting BUY is hit by seller-initiated trades (side -1); a resting SELL by buyers (+1)
                if (side > 0 and t_side[k] < 0) or (side < 0 and t_side[k] > 0):
                    through = (side > 0 and t_px[k] < P) or (side < 0 and t_px[k] > P)
                    at = t_px[k] == P
                    if through:
                        q = st[S_REM]
                    elif at:
                        v = t_size[k] / CT_PER_BTC
                        if v <= st[S_Q]:
                            st[S_Q] -= v
                            q = 0.0
                        else:
                            q = min(st[S_REM], v - st[S_Q])
                            st[S_Q] = 0.0
                    else:
                        q = 0.0
                    if q > 0.0:
                        px = P * TICK
                        fee = q * px * maker_bps / 1e4
                        st[S_CASH] -= side * q * px + fee
                        st[S_FEES] += fee
                        st[S_POS] += side * q
                        st[S_REM] -= q
                        if nf < fills_out.shape[0]:
                            fills_out[nf, 0] = i; fills_out[nf, 1] = side * q; fills_out[nf, 2] = px
                            fills_out[nf, 3] = 1.0; fills_out[nf, 4] = st[S_PLACE]
                        nf += 1
            if st[S_REM] <= 1e-12:
                st[S_ACTIVE] = 0.0
        # ---- 2) manage the resting order (age, re-peg, expiry)
        if st[S_ACTIVE] == 1.0:
            st[S_AGE] += 1.0
            touch = round(bid[i] / TICK) if st[S_SIDE] > 0 else round(ask[i] / TICK)
            if touch != st[S_PX]:
                st[S_BEHIND] += 1.0
            else:
                st[S_BEHIND] = 0.0
            expired = st[S_AGE] > ttl
            if expired and cross_on_expiry and valid[i] == 1:
                delta = st[S_SIDE] * st[S_REM]
                px = _taker_px(m, delta, hs[i], b1[i], b10[i], b50[i], a1[i], a10[i], a50[i], impact_mult, slip)
                fee = abs(delta) * px * taker_bps / 1e4
                st[S_CASH] -= delta * px + fee
                st[S_FEES] += fee
                st[S_POS] += delta
                if nf < fills_out.shape[0]:
                    fills_out[nf, 0] = i; fills_out[nf, 1] = delta; fills_out[nf, 2] = px
                    fills_out[nf, 3] = 0.0; fills_out[nf, 4] = st[S_PLACE]
                nf += 1
                st[S_ACTIVE] = 0.0
            elif expired or st[S_BEHIND] >= repeg:
                st[S_ACTIVE] = 0.0                       # cancel; a new order may be placed below
        # ---- 3) new decisions (target decided at row i-1 reaches the venue now, at row i)
        if i >= lag and valid[i] == 1 and valid[i - lag] == 1:
            t = target[i - lag]
            if t != t:
                t = 0.0
            want = t * unit / m
            pending = st[S_SIDE] * st[S_REM] if st[S_ACTIVE] == 1.0 else 0.0
            delta = want - st[S_POS]
            if st[S_ACTIVE] == 1.0 and abs(st[S_TGT] - t) > 1e-9:
                st[S_ACTIVE] = 0.0                       # target changed: cancel and re-decide
                pending = 0.0
            reducing = abs(want) < abs(st[S_POS]) - 1e-12 or want * st[S_POS] < 0
            if abs(delta - pending) * m > deadband * unit or (t == 0.0 and st[S_POS] != 0.0 and st[S_ACTIVE] == 0.0):
                if exit_taker and reducing:
                    st[S_ACTIVE] = 0.0
                    px = _taker_px(m, delta, hs[i], b1[i], b10[i], b50[i], a1[i], a10[i], a50[i], impact_mult, slip)
                    fee = abs(delta) * px * taker_bps / 1e4
                    st[S_CASH] -= delta * px + fee
                    st[S_FEES] += fee
                    st[S_POS] += delta
                    if nf < fills_out.shape[0]:
                        fills_out[nf, 0] = i; fills_out[nf, 1] = delta; fills_out[nf, 2] = px
                        fills_out[nf, 3] = 0.0; fills_out[nf, 4] = i
                    nf += 1
                elif st[S_ACTIVE] == 0.0 and abs(delta) * m > deadband * unit * 0.5:
                    side = 1.0 if delta > 0 else -1.0
                    st[S_ACTIVE] = 1.0; st[S_SIDE] = side
                    st[S_PX] = round(bid[i] / TICK) if side > 0 else round(ask[i] / TICK)
                    st[S_Q] = (bsz[i] if side > 0 else asz[i]) / CT_PER_BTC * queue_mult
                    st[S_REM] = abs(delta); st[S_AGE] = 0.0; st[S_BEHIND] = 0.0; st[S_TGT] = t; st[S_PLACE] = i
        equity[i] = st[S_CASH] + st[S_POS] * m
        pos_out[i] = st[S_POS]
    return equity, pos_out, nf


def load_tape(store_root, dates):
    from pathlib import Path
    parts = [pl.read_parquet(Path(store_root) / 'tape' / f'{d}.parquet') for d in dates
             if (Path(store_root) / 'tape' / f'{d}.parquet').exists()]
    if not parts:
        return None
    return pl.concat(parts).sort(['ts', 'side', 'px'])


def run_block(frame: pl.DataFrame, target, tape: pl.DataFrame | None, cost: CostModel, ex: dict, st=None):
    """frame: 1 s execution rows; target: per-row target decided at that row. Mutates/returns state."""
    n = len(frame)
    st = np.zeros(STATE_SIZE) if st is None else st
    col = lambda c, d=0.0: frame[c].to_numpy().astype(np.float64) if c in frame.columns else np.full(n, d)
    ts = frame['ts'].to_numpy()
    if tape is not None and len(tape):
        tts = tape['ts'].to_numpy()
        t_start = np.searchsorted(tts, ts, 'left').astype(np.int64)
        t_end = np.searchsorted(tts, ts, 'right').astype(np.int64)
        t_px = tape['px'].to_numpy().astype(np.float64)
        t_side = tape['side'].to_numpy().astype(np.float64)
        t_size = tape['size'].to_numpy().astype(np.float64)
    else:
        t_start = t_end = np.zeros(n, np.int64)
        t_px = t_side = t_size = np.zeros(0)
    hour = ts // 3_600_000
    funding = np.zeros(n)
    if cost.funding and 'funding_rate' in frame.columns:
        b = np.zeros(n, bool); b[1:] = hour[1:] != hour[:-1]
        funding[b] = frame['funding_rate'].to_numpy()[b]
    valid = (col('valid', 1.0) >= 0.5).astype(np.int8)
    if st[S_ACTIVE] == 1.0:
        st[S_PLACE] = -1.0                               # order carried in from the previous block
    fills = np.zeros((max(1000, n // 2), 5))
    eq, pos, nf = _run_maker(col('mid'), col('bid_px'), col('ask_px'), col('bid_sz0'), col('ask_sz0'),
                             col('spread_bps', 0.01) / 2, col('bimp_1', .5), col('bimp_10', 2.), col('bimp_50', 6.),
                             col('aimp_1', .5), col('aimp_10', 2.), col('aimp_50', 6.), valid, funding,
                             np.asarray(target, np.float64), t_start, t_end, t_px, t_side, t_size, st,
                             cost.unit_notional, cost.deadband, cost.maker_bps, cost.taker_bps, cost.slippage_bps,
                             cost.impact_mult, cost.queue_mult, float(ex['ttl_s']), float(ex['repeg_s']),
                             ex['on_expiry'] == 'cross', ex['exit'] == 'taker', max(1, int(cost.latency_s)), fills)
    return eq, pos, fills[:min(nf, len(fills))], st


def to_1s_target(sig_frame: pl.DataFrame, sig, exec_frame: pl.DataFrame):
    """Map a bar-resolution signal onto the 1 s timeline: the value decided at a bar's last second holds until
    the next decision (as-of join, never forward-looking)."""
    s = pl.DataFrame({'ts': sig_frame['ts'], 'target': np.asarray(sig, np.float64)}).sort('ts')
    return exec_frame.select('ts').join_asof(s, on='ts', strategy='backward')['target'].fill_null(0.0).to_numpy()


def maker_summary(ts, equity, pos, mid, st, fills, cost):
    frow = fills[:, 0].astype(np.int64) if len(fills) else np.zeros(0, np.int64)
    fqty = fills[:, 1] if len(fills) else np.zeros(0)
    fpx = fills[:, 2] if len(fills) else np.zeros(0)
    out = summarize(ts, equity, pos, mid, st[S_FEES], st[S_FUND], frow, fqty, fpx, np.zeros(len(frow)), cost)
    if len(fills):
        maker = fills[:, 3] == 1.0
        out['maker_fill_fraction'] = float(np.sum(np.abs(fqty[maker])) / max(np.sum(np.abs(fqty)), 1e-12))
        # adverse selection: mid move 60 s after a maker fill, signed by our side (negative = picked off)
        idx = np.clip(frow[maker] + 60, 0, len(mid) - 1)
        side = np.sign(fqty[maker])
        mk = (mid[idx] / fpx[maker] - 1) * 1e4 * side
        out['maker_markout_60s_bps'] = float(np.mean(mk)) if len(mk) else None
        out['median_seconds_to_fill'] = float(np.median(frow[maker] - fills[maker, 4])) if maker.any() else None
    out['execution'] = 'maker'
    return out
