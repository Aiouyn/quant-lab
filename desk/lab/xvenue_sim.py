"""Cross-venue convergence on Hyperliquid, simulated on the 1 s grid (desk.data.xvenue) with live latencies.

basis[s] = hl_mid / bn_px - 1 (bps);  x[s] = basis[s] - rolling median(basis, base_s) (persistent premium removed)
Decision at s uses only data VISIBLE at s: HL book as of s-1 (WS arrival ~1 s), Binance as of s-1.
Entry when |x| >= K, HL book fresh (age <= 2 s) and HL spread <= max_spread: side = -sign(x) (HL converges to BN).
Fill: touch price of the HL book as of s+2 (decision->fill latency 1.2-2.5 s measured live), + taker fee.
Exit: when visible |x| <= exit_frac*K or after H seconds; filled at touch 2 s later, + taker fee. One position/coin.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

TAKER_BPS = 4.5


def arrays(g: pl.DataFrame, base_s=1800):
    g = g.with_columns(((pl.col('bid') + pl.col('ask')) / 2).alias('mid'))
    g = g.with_columns(((pl.col('mid') / pl.col('bn_px') - 1) * 1e4).alias('basis'))
    g = g.with_columns(pl.col('basis').rolling_median(base_s, min_samples=base_s // 2).alias('base'))
    ok = (pl.col('hl_age') <= 2000) & (pl.col('bn_age') <= 5000) & (pl.col('bid') > 0) & (pl.col('ask') > pl.col('bid'))
    g = g.with_columns(pl.when(ok).then(pl.col('basis') - pl.col('base')).otherwise(None).alias('x'),
                       ((pl.col('ask') - pl.col('bid')) / pl.col('mid') * 1e4).alias('spr'))
    f = lambda c: g[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return g['s'].to_numpy(), f('x'), f('bid'), f('ask'), f('spr')


@njit(cache=True)
def _sim(x, bid, ask, spr, K, H, exit_frac, max_spread, fee):
    n = len(x)
    out = np.full((n, 5), np.nan)                 # entry idx, side, gross bps, net bps, hold s
    m = 0
    i = 1
    while i < n - H - 6:
        xv = x[i - 1]                              # visible at i
        if np.isfinite(xv) and abs(xv) >= K and spr[i - 1] <= max_spread:
            side = -1.0 if xv > 0 else 1.0
            j = i + 2
            pin = ask[j] if side > 0 else bid[j]
            if not (np.isfinite(pin) and pin > 0):
                i += 1
                continue
            k = j + 1
            while k < i + H:
                xv2 = x[k - 1]
                if np.isfinite(xv2) and abs(xv2) <= exit_frac * K:
                    break
                k += 1
            e = k + 2
            pout = bid[e] if side > 0 else ask[e]
            q = 0
            while not (np.isfinite(pout) and pout > 0) and e < n - 1 and q < 30:
                e += 1; q += 1
                pout = bid[e] if side > 0 else ask[e]
            gross = side * (pout / pin - 1) * 1e4
            out[m, 0] = i; out[m, 1] = side; out[m, 2] = gross; out[m, 3] = gross - 2 * fee; out[m, 4] = e - j
            m += 1
            i = e + 1
            continue
        i += 1
    return out[:m]


def simulate(g, K=20.0, H=300, exit_frac=0.25, max_spread=10.0, base_s=1800, fee=TAKER_BPS):
    s, x, bid, ask, spr = arrays(g, base_s)
    r = _sim(x, bid, ask, spr, float(K), int(H), float(exit_frac), float(max_spread), float(fee))
    if len(r) == 0:
        return pl.DataFrame(schema={'s': pl.Int64, 'side': pl.Float64, 'gross_bps': pl.Float64, 'net_bps': pl.Float64,
                                    'hold_s': pl.Float64})
    return pl.DataFrame({'s': s[r[:, 0].astype(np.int64)], 'side': r[:, 1], 'gross_bps': r[:, 2], 'net_bps': r[:, 3],
                         'hold_s': r[:, 4]})


# --------------------------------------------------------------------------------------------- resting-order variant
@njit(cache=True)
def _sim_maker(fair, bid, ask, delta, H, maker_fee, taker_fee, lag, exit_frac):
    """Resting bid at fair[s-lag]*(1-delta) and ask at fair[s-lag]*(1+delta) (fair = Binance price x baseline HL
    premium, both as of s-lag). Filled during second s only if the HL book TRADES THROUGH the quote: best ask
    <= our bid (or best bid >= our ask) at the snapshot of second s. Exit: taker at the touch when the HL mid comes
    back within exit_frac*delta of fair (visible a second later), or after H seconds; exit fill 2 s after decision."""
    n = len(fair)
    out = np.full((n, 5), np.nan)
    m = 0
    i = lag + 1
    while i < n - H - 6:
        f = fair[i - lag]
        if not np.isfinite(f) or not (np.isfinite(bid[i]) and np.isfinite(ask[i])):
            i += 1
            continue
        mv0 = (bid[i - lag] + ask[i - lag]) / 2
        if not (np.isfinite(mv0) and abs(mv0 / f - 1) < 0.03):      # corrupted / stale reference: never quote
            i += 1
            continue
        qb = f * (1 - delta / 1e4); qa = f * (1 + delta / 1e4)
        side = 0.0
        if ask[i] <= qb:
            side = 1.0; pin = qb
        elif bid[i] >= qa:
            side = -1.0; pin = qa
        if side == 0.0:
            i += 1
            continue
        k = i + 1
        while k < i + H:
            fv = fair[k - 1]; mv = (bid[k - 1] + ask[k - 1]) / 2
            if np.isfinite(fv) and np.isfinite(mv) and abs(mv / fv - 1) * 1e4 <= exit_frac * delta:
                break
            k += 1
        e = k + 2
        pout = bid[e] if side > 0 else ask[e]
        q = 0
        while not (np.isfinite(pout) and pout > 0) and e < n - 1 and q < 30:
            e += 1; q += 1
            pout = bid[e] if side > 0 else ask[e]
        if not (np.isfinite(pout) and pout > 0):
            i = e + 1
            continue
        gross = side * (pout / pin - 1) * 1e4
        out[m, 0] = i; out[m, 1] = side; out[m, 2] = gross; out[m, 3] = gross - maker_fee - taker_fee; out[m, 4] = e - i
        m += 1
        i = e + 1
    return out[:m]


def simulate_maker(g, delta=20.0, H=120, lag=2, base_s=1800, exit_frac=0.25, maker_fee=1.5, taker_fee=TAKER_BPS):
    g = g.with_columns(((pl.col('bid') + pl.col('ask')) / 2).alias('mid'))
    g = g.with_columns(((pl.col('mid') / pl.col('bn_px') - 1) * 1e4).alias('basis'))
    g = g.with_columns(pl.col('basis').rolling_median(base_s, min_samples=base_s // 2).alias('base'))
    ok = (pl.col('hl_age') <= 2000) & (pl.col('bid') > 0) & (pl.col('ask') > pl.col('bid'))
    fair_ok = pl.col('bn_age') <= 5000
    g = g.with_columns(pl.when(fair_ok).then(pl.col('bn_px') * (1 + pl.col('base') / 1e4)).otherwise(None).alias('fair'),
                       pl.when(ok).then(pl.col('bid')).otherwise(None).alias('b'),
                       pl.when(ok).then(pl.col('ask')).otherwise(None).alias('a'))
    f = lambda c: g[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
    r = _sim_maker(f('fair'), f('b'), f('a'), float(delta), int(H), float(maker_fee), float(taker_fee), int(lag), float(exit_frac))
    s = g['s'].to_numpy()
    if len(r) == 0:
        return pl.DataFrame(schema={'s': pl.Int64, 'side': pl.Float64, 'gross_bps': pl.Float64, 'net_bps': pl.Float64,
                                    'hold_s': pl.Float64})
    return pl.DataFrame({'s': s[r[:, 0].astype(np.int64)], 'side': r[:, 1], 'gross_bps': r[:, 2], 'net_bps': r[:, 3],
                         'hold_s': r[:, 4]})
