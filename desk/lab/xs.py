"""Cross-sectional perp desk: hourly panel over ~79 coins listed on both Hyperliquid and Binance.

Row (t, coin) holds only what is known at hour t (carry panel, Binance bars re-keyed to bar close):
  ret      : Binance perp close-to-close return of the hour ending at t (HL price proxy; basis corr 0.8-0.9)
  hl_funding (paid at t), hl_premium, bn_funding/bn_rate_last/bn_interval_h, bn_qvol (bar ending at t)
Strategy weights w[t, coin] (decided at t) are held over (t, t+1]:
  pnl[t+1] = sum_c w * ret[t+1] - sum_c w * hl_funding[t+1] - sum_c |w[t] - w[t-1]| * cost_bps(c) / 1e4
Survivorship caveat: the universe is coins still listed on both venues in 2026-10 (delisted coins are missing).
"""
from __future__ import annotations

import types

import numpy as np
import polars as pl

HOUR = 3_600_000
MAJORS = {'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE'}


def load_xs(settings, start_ms=None, end_ms=None):
    from ..data.carry_data import load_panel
    p = load_panel(types.SimpleNamespace(store=settings.store)).filter(~pl.col('coin').str.contains('@'))
    p = p.sort(['coin', 'ts']).with_columns(
        (pl.col('bn_close') / pl.col('bn_close').shift(1).over('coin') - 1).alias('ret'))
    if start_ms is not None:
        p = p.filter(pl.col('ts') >= start_ms)
    if end_ms is not None:
        p = p.filter(pl.col('ts') < end_ms)
    return p


def wide(p, col, coins=None, ts=None):
    w = p.pivot(on='coin', index='ts', values=col, aggregate_function='first').sort('ts')
    coins = coins or sorted(c for c in w.columns if c != 'ts')
    return w['ts'].to_numpy(), coins, w.select(coins).to_numpy().astype(float)


class XS:
    """Dense hour x coin matrices for fast research."""

    def __init__(self, p):
        self.ts, self.coins, self.R = wide(p, 'ret')
        _, _, self.F = wide(p, 'hl_funding', self.coins)
        _, _, self.P = wide(p, 'hl_premium', self.coins)
        _, _, self.BR = wide(p, 'bn_rate_last', self.coins)
        _, _, self.BI = wide(p, 'bn_interval_h', self.coins)
        _, _, self.V = wide(p, 'bn_qvol', self.coins)
        _, _, self.C = wide(p, 'bn_close', self.coins)
        self.R[np.abs(self.R) > 0.5] = np.nan                       # bad ticks / relistings
        self.live = np.isfinite(self.R) & np.isfinite(self.C)
        self.cost_bps = np.array([5.5 if c in MAJORS else 7.5 for c in self.coins])   # HL taker 4.5 + spread/slip


def rolling_sum(X, n):
    Z = np.nan_to_num(X)
    c = np.cumsum(Z, 0)
    out = c.copy()
    out[n:] = c[n:] - c[:-n]
    cnt = np.cumsum(np.isfinite(X), 0)
    k = cnt.copy(); k[n:] = cnt[n:] - cnt[:-n]
    out[k < n * 0.8] = np.nan
    return out


def rolling_mean(X, n):
    Z = np.nan_to_num(X)
    c = np.cumsum(Z, 0); cnt = np.cumsum(np.isfinite(X), 0)
    s = c.copy(); s[n:] = c[n:] - c[:-n]
    k = cnt.copy(); k[n:] = cnt[n:] - cnt[:-n]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = s / k
    out[k < n * 0.8] = np.nan
    return out


def xrank(X):
    """Cross-sectional rank in [-0.5, 0.5] per row (NaN-aware)."""
    out = np.full_like(X, np.nan)
    for i in range(len(X)):
        m = np.isfinite(X[i])
        n = m.sum()
        if n >= 5:
            r = X[i, m].argsort().argsort()
            out[i, m] = r / (n - 1) - 0.5
    return out


def forward(R, h):
    """Sum of returns over (t, t+h] aligned at t."""
    c = np.cumsum(np.nan_to_num(R), 0)
    out = np.full_like(R, np.nan)
    out[:-h] = c[h:] - c[:-h]
    return out


def rank_ic(S, Y, step=1):
    ics = []
    for i in range(0, len(S), step):
        m = np.isfinite(S[i]) & np.isfinite(Y[i])
        if m.sum() >= 10:
            a, b = S[i, m].argsort().argsort(), Y[i, m].argsort().argsort()
            ics.append(np.corrcoef(a, b)[0, 1])
    ics = np.array(ics)
    return float(np.nanmean(ics)), float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12) * np.sqrt(len(ics))), len(ics)


def weights_from_score(S, live, rebalance=1, gross=1.0, top=None, smooth=None):
    """Dollar-neutral weights: demeaned cross-sectional ranks (or top/bottom quantile), sum |w| = gross,
    held between rebalances; optional EMA smoothing (in hours) to cut turnover."""
    W = np.zeros_like(S)
    cur = np.zeros(S.shape[1])
    for i in range(len(S)):
        if i % rebalance == 0:
            s = np.where(live[i] & np.isfinite(S[i]), S[i], np.nan)
            m = np.isfinite(s)
            w = np.zeros(S.shape[1])
            if m.sum() >= 10:
                if top:
                    k = max(1, int(m.sum() * top))
                    idx = np.where(m)[0][np.argsort(s[m])]
                    w[idx[:k]] = -1.0; w[idx[-k:]] = 1.0
                else:
                    r = s[m].argsort().argsort() / (m.sum() - 1) - 0.5
                    w[m] = r
                w -= w[m].mean() * m if not top else 0
                g = np.abs(w).sum()
                w = w / g * gross if g > 0 else w
            cur = w if smooth is None else cur + (w - cur) / smooth
        cur = np.where(live[i], cur, 0.0)
        W[i] = cur
    return W


def simulate(xs: XS, W, from_idx=0, to_idx=None, cost_mult=1.0, funding=True):
    """Hourly PnL (fraction of gross capital) of weights W decided at t, held over (t, t+1]."""
    to_idx = to_idx or len(xs.ts)
    R = np.nan_to_num(xs.R); F = np.nan_to_num(xs.F)
    Wp = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])        # weight held during the hour ending at t
    gross_ret = (Wp * R).sum(1)
    fund = -(Wp * F).sum(1) if funding else np.zeros(len(R))
    turn = np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0))
    cost = (turn * xs.cost_bps * cost_mult / 1e4).sum(1)
    pnl = gross_ret + fund - cost
    sl = slice(from_idx, to_idx)
    return {'pnl': pnl[sl], 'gross': gross_ret[sl], 'funding': fund[sl], 'cost': cost[sl], 'ts': xs.ts[sl],
            'turnover': turn[sl].sum(1)}


def stats(res):
    p = res['pnl']
    days = len(p) / 24
    d = p[: int(days) * 24].reshape(-1, 24).sum(1) if days >= 1 else p
    sh = d.mean() / (d.std() + 1e-12) * np.sqrt(365)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {'sharpe': round(float(sh), 2), 'ann_ret': round(float(p.sum() / days * 365), 3), 'max_dd': round(dd, 3),
            'gross_ann': round(float(res['gross'].sum() / days * 365), 3), 'fund_ann': round(float(res['funding'].sum() / days * 365), 3),
            'cost_ann': round(float(res['cost'].sum() / days * 365), 3), 'turnover_day': round(float(res['turnover'].sum() / days), 2),
            'days': int(days)}
