"""Fast, honest backtester (numba) for target-position strategies on the 1s store or bars.

Contract: a strategy emits target[i] in [-1, 1] using data up to and including row i.
The order is filled `latency` later at the then-current mid, crossing the half spread, plus
depth-sweep impact (from the deep OKX book at decision time), extra slippage and taker fees.
Validity: both the decision row AND the fill row must be `valid` (and the fill row contiguous in time).
If the fill row is invalid/forward-filled or across a gap, the order is dropped (no fill, no carry-over);
the strategy's target on the next valid decision row is re-evaluated as usual. certify.py implements the same.
Hourly Hyperliquid funding is charged on open positions at each hour boundary.
Position unit: target 1.0 == `unit_notional` USD (fixed, non-compounding -> clean daily PnL series).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numba as nb
import numpy as np
import polars as pl

RES_SECONDS = {'1s': 1, '10s': 10, '1m': 60}


@dataclass
class CostModel:
    taker_bps: float = 4.5
    maker_bps: float = 1.5        # Hyperliquid base-tier maker fee (limit orders that rest and get filled)
    queue_mult: float = 1.0       # maker stress: queue ahead of our limit order x queue_mult
    slippage_bps: float = 0.5
    impact_mult: float = 1.0      # OKX->HL depth transfer multiplier (calibrated by the execution analyst)
    latency_s: int = 2            # real HL IOC round trip measured 1.2 s median / 2.5 s mean (operator's live ledger)
    vol_slip_coef: float = 0.05   # extra bps per side per bps of |mid move over the last 60 s| (live: ~11 bps RT on 1% moves)
    deadband: float = 0.05        # ignore rebalances smaller than 5% of unit
    unit_notional: float = 100_000.0
    funding: bool = True

    def stressed(self, fee_mult=1.5, extra_latency_s=0, impact_mult=None, queue_mult=None):
        d = asdict(self)
        d['taker_bps'] *= fee_mult
        d['maker_bps'] *= fee_mult
        if queue_mult is not None:
            d['queue_mult'] = queue_mult
        d['slippage_bps'] *= fee_mult
        d['latency_s'] += extra_latency_s
        if impact_mult is not None:
            d['impact_mult'] = impact_mult
        return CostModel(**d)


@nb.njit(cache=True)
def _impact(size_btc, i1, i10, i50, hs):
    """Average extra bps beyond the touch to sweep `size_btc`, from bps-to-sweep at 1/10/50 BTC."""
    if size_btc <= 1.0:
        worst = hs + (i1 - hs) * size_btc
    elif size_btc <= 10.0:
        worst = i1 + (i10 - i1) * (size_btc - 1.0) / 9.0
    elif size_btc <= 50.0:
        worst = i10 + (i50 - i10) * (size_btc - 10.0) / 40.0
    else:
        worst = i50 * size_btc / 50.0
    extra = (worst - hs) / 2.0          # average fill sits ~halfway between touch and worst level
    return extra if extra > 0.0 else 0.0


@nb.njit(cache=True)
def _run(exec_mid, mark_mid, hs_bps, bi1, bi10, bi50, ai1, ai10, ai50, target, valid, funding,
         fee_bps, slip, impact_mult, unit, deadband):
    n = len(target)
    equity = np.empty(n)
    pos_out = np.empty(n)
    cash = 0.0
    pos = 0.0                       # BTC
    fees = 0.0
    fund_paid = 0.0
    nfills = 0
    fill_row = np.empty(n, np.int64)
    fill_qty = np.empty(n)
    fill_px = np.empty(n)
    fill_cost = np.empty(n)
    for i in range(n):
        m = mark_mid[i]
        if funding[i] != 0.0 and pos != 0.0:
            f = pos * m * funding[i]
            cash -= f
            fund_paid += f
        t = target[i]
        p = exec_mid[i]
        if valid[i] == 1 and p == p and t == t:
            if t > 1.0:
                t = 1.0
            elif t < -1.0:
                t = -1.0
            want = t * unit / p
            delta = want - pos
            if abs(delta) * p > deadband * unit or (t == 0.0 and pos != 0.0):
                size = abs(delta)
                if delta > 0:
                    extra = _impact(size, ai1[i], ai10[i], ai50[i], hs_bps[i]) * impact_mult
                    px = p * (1.0 + (hs_bps[i] + extra + slip[i]) / 1e4)
                else:
                    extra = _impact(size, bi1[i], bi10[i], bi50[i], hs_bps[i]) * impact_mult
                    px = p * (1.0 - (hs_bps[i] + extra + slip[i]) / 1e4)
                fee = size * px * fee_bps / 1e4
                cash -= delta * px + fee
                fees += fee
                fill_row[nfills] = i
                fill_qty[nfills] = delta
                fill_px[nfills] = px
                fill_cost[nfills] = fee + size * abs(px - p)
                nfills += 1
                pos = want if t != 0.0 else 0.0
                if t == 0.0:
                    pos = 0.0
        equity[i] = cash + pos * m
        pos_out[i] = pos
    return equity, pos_out, fees, fund_paid, fill_row[:nfills], fill_qty[:nfills], fill_px[:nfills], fill_cost[:nfills]


def _arrays(df: pl.DataFrame, resolution: str, cost: CostModel):
    res_s = RES_SECONDS[resolution]
    n = len(df)
    mid = df['mid'].to_numpy().astype(np.float64)
    lat_rows = max(1, int(round(cost.latency_s / res_s))) if res_s == 1 else 1
    if res_s == 1:
        shift = lat_rows                     # fill `latency` rows later
        src = mid
    else:
        # bars: decision at bar end, fill at the next bar's first observed mid (+ extra latency rows if stressed)
        extra = max(0, int(round((cost.latency_s - 1) / res_s)))
        shift = 1 + extra
        src = df['open'].to_numpy().astype(np.float64) if 'open' in df.columns else mid
    exec_mid = np.full(n, np.nan)
    if shift < n:
        exec_mid[:n - shift] = src[shift:]
    # gaps across days: exec only if the fill row is contiguous in time (same `shift` rows as the fill itself)
    ts = df['ts'].to_numpy()
    step = res_s * 1000 * shift
    nxt = np.full(n, np.iinfo(np.int64).max)
    if shift < n:
        nxt[:n - shift] = ts[shift:]
    exec_mid[(nxt - ts) > step + 5000] = np.nan
    # FILL-row validity: an order whose execution row is invalid (forward-filled book state) does not fill at that
    # stale mid. Contract: the order is DROPPED (no fill on this decision); the next decision row re-targets.
    if 'valid' in df.columns:
        v = df['valid'].to_numpy() >= 0.5
        fill_ok = np.zeros(n, bool)
        if shift < n:
            fill_ok[:n - shift] = v[shift:]
        exec_mid[~fill_ok] = np.nan
    hs = (df['spread_bps'].to_numpy().astype(np.float64) / 2) if 'spread_bps' in df.columns else np.full(n, 0.005)

    def col(c, default):
        return df[c].to_numpy().astype(np.float64) if c in df.columns else np.full(n, default)
    bi1, bi10, bi50 = col('bimp_1', 0.5), col('bimp_10', 2.0), col('bimp_50', 6.0)
    ai1, ai10, ai50 = col('aimp_1', 0.5), col('aimp_10', 2.0), col('aimp_50', 6.0)
    valid = (df['valid'].to_numpy() >= 0.5).astype(np.int8) if 'valid' in df.columns else np.ones(n, np.int8)
    funding = np.zeros(n)
    if cost.funding and 'funding_rate' in df.columns:
        hour = ts // 3_600_000
        boundary = np.zeros(n, bool)
        boundary[1:] = hour[1:] != hour[:-1]
        funding[boundary] = df['funding_rate'].to_numpy()[boundary]
    # volatility-scaled slippage: fast markets cost more than the displayed book suggests (calibrated on live fills)
    k = max(1, int(round(60 / res_s)))
    move = np.zeros(n)
    move[k:] = np.abs(np.log(mid[k:] / mid[:-k])) * 1e4
    slip = cost.slippage_bps + cost.vol_slip_coef * move
    return exec_mid, mid, hs, bi1, bi10, bi50, ai1, ai10, ai50, valid, funding, slip


def backtest(df: pl.DataFrame, target, resolution='1s', cost: CostModel | None = None, keep_series=False):
    cost = cost or CostModel()
    target = np.asarray(target, dtype=np.float64)
    if len(target) != len(df):
        raise ValueError(f'target length {len(target)} != data length {len(df)}')
    if not np.all(np.isfinite(target[~np.isnan(target)])):
        raise ValueError('target must be finite')
    exec_mid, mid, hs, bi1, bi10, bi50, ai1, ai10, ai50, valid, funding, slip = _arrays(df, resolution, cost)
    equity, pos, fees, fund_paid, frow, fqty, fpx, fcost = _run(
        exec_mid, mid, hs, bi1, bi10, bi50, ai1, ai10, ai50, np.nan_to_num(target, nan=0.0), valid, funding,
        cost.taker_bps, slip, cost.impact_mult, cost.unit_notional, cost.deadband)
    ts = df['ts'].to_numpy()
    return summarize(ts, equity, pos, mid, fees, fund_paid, frow, fqty, fpx, fcost, cost, keep_series)


def summarize(ts, equity, pos, mid, fees, fund_paid, frow, fqty, fpx, fcost, cost, keep_series=False):
    unit = cost.unit_notional
    day = ts // 86_400_000
    ends = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
    eq_end = equity[ends]
    daily_pnl = np.diff(np.r_[0.0, eq_end])
    daily_ret = daily_pnl / unit
    # position episodes: entries from flat or sign flips
    sgn = np.sign(np.round(pos * mid / unit, 6))
    prev = np.r_[0.0, sgn[:-1]]
    entries = int(np.sum((sgn != 0) & (sgn != prev)))
    exposure = float(np.mean(sgn != 0))
    turnover = float(np.sum(np.abs(fqty * fpx)) / unit)
    from .stats import sharpe, max_drawdown
    out = {
        'days': int(len(daily_ret)), 'pnl': float(equity[-1]), 'return_on_unit': float(equity[-1] / unit),
        'sharpe': sharpe(daily_ret), 'mean_daily_bps': float(np.mean(daily_ret) * 1e4) if len(daily_ret) else 0.0,
        'max_drawdown': max_drawdown(equity / unit), 'fees': float(fees), 'funding_paid': float(fund_paid),
        'execution_cost': float(np.sum(fcost)), 'fills': int(len(frow)), 'trades': entries,
        'turnover_units': turnover, 'exposure': exposure,
        'positive_days': float(np.mean(daily_ret > 0)) if len(daily_ret) else 0.0,
        'pnl_per_turnover_bps': float(equity[-1] / max(turnover * unit, 1e-9) * 1e4),
        'cost': asdict(cost),
    }
    out['daily'] = {'day': [int(d) for d in day[ends]], 'ret': [float(r) for r in daily_ret]}
    if keep_series:
        out['_equity'] = equity; out['_pos'] = pos
        out['_fills'] = (frow, fqty, fpx)
    return out
