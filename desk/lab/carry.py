"""Cross-venue funding carry: engine, split-guarded panel store, statistics.

Position s[t, c] in [-1, 1] decided at hour t (with data up to and including row t):
  +1 = long Hyperliquid perp / short Binance perp, -1 = the reverse. Held over (t, t+1]:
  funding  : s * (bn_funding[t+1] - hl_funding[t+1])          (each venue: longs pay when the rate is > 0)
  basis    : s * ((hl_premium[t+1] - hl_premium[t]) - (bn_premium[t+1] - bn_premium[t]))
  costs    : |s[t] - s[t-1]| * (hl_taker + bn_taker + 2 * slippage(c))   on both legs, charged at t
Gross exposure is capped: per-coin notional = unit * s / max(1, sum_c |s|).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import polars as pl

HOUR = 3_600_000
PANEL_COLS = ('ts', 'coin', 'hl_funding', 'hl_premium', 'bn_funding', 'bn_rate_last', 'bn_interval_h',
              'bn_premium', 'bn_close', 'bn_qvol')


@dataclass
class CarryCost:
    hl_taker_bps: float = 4.5
    bn_taker_bps: float = 5.0
    major_slip_bps: float = 1.0          # BTC/ETH per leg
    alt_slip_bps: float = 3.0            # others per leg, plus impact vs hourly volume
    impact_coef: float = 50.0            # bps per (notional / hourly quote volume)
    basis_noise_bps: float = 0.0         # stress: extra cost per unit turnover for basis uncertainty
    unit_notional: float = 100_000.0
    max_gross: float = 1.0
    execution: str = 'taker'            # 'maker': post limit orders on both legs (positions fill one hour later)
    hl_maker_bps: float = 1.5
    bn_maker_bps: float = 2.0
    maker_adverse_bps: float = 0.5      # per leg allowance for adverse selection on passive fills

    def stressed(self, fee_mult=1.5, slip_mult=2.0, basis_noise_bps=None):
        d = asdict(self)
        d['hl_taker_bps'] *= fee_mult; d['bn_taker_bps'] *= fee_mult
        d['major_slip_bps'] *= slip_mult; d['alt_slip_bps'] *= slip_mult
        d['hl_maker_bps'] *= fee_mult; d['bn_maker_bps'] *= fee_mult; d['maker_adverse_bps'] *= slip_mult
        if basis_noise_bps is not None:
            d['basis_noise_bps'] = basis_noise_bps
        return CarryCost(**d)

    def as_taker(self):
        d = asdict(self); d['execution'] = 'taker'
        return CarryCost(**d)


MAJORS = {'BTC', 'ETH'}


class CarryStore:
    """Panel access with the carry desk's own frozen chronological split; history before a split may be used
    as warmup but outcomes are only ever evaluated inside allowed splits."""

    def __init__(self, settings, allowed=('dev',)):
        from ..data.carry_data import carry_root
        self.root = carry_root(settings)
        self.allowed = set(allowed)
        self._panel = None

    def splits(self):
        return json.loads((self.root / 'splits.json').read_text())

    def range_ms(self, split):
        if split not in self.allowed:
            from ..data.store import SplitViolation
            raise SplitViolation(f'carry {split} data is not available in this context')
        a, b = self.splits()[split]
        to = lambda d: int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)
        return to(a), to(b) + 86_400_000

    def panel(self):
        if self._panel is None:
            from ..data.carry_data import load_panel
            import types
            p = load_panel(types.SimpleNamespace(store=self.root.parent))
            self._panel = p.select([c for c in PANEL_COLS if c in p.columns])
        return self._panel

    def load(self, split, warmup_days=30, coins=None):
        a, b = self.range_ms(split)
        latest_allowed = max(self.range_ms(s)[1] for s in self.allowed)
        p = self.panel().filter((pl.col('ts') >= a - warmup_days * 86_400_000) & (pl.col('ts') < min(b, latest_allowed)))
        if coins:
            p = p.filter(pl.col('coin').is_in(list(coins)))
        return p, a, b


def pivot(panel: pl.DataFrame, col):
    w = panel.pivot(on='coin', index='ts', values=col, aggregate_function='last').sort('ts')
    return w


def run_carry(panel: pl.DataFrame, positions, cost: CarryCost, trade_from_ms=None):
    """panel rows sorted by (ts, coin); positions aligned to panel rows. Returns dict with hourly series."""
    p = panel.with_columns(pl.Series('pos', np.clip(np.nan_to_num(np.asarray(positions, float)), -1, 1)))
    if trade_from_ms is not None:
        p = p.with_columns(pl.when(pl.col('ts') >= trade_from_ms).then(pl.col('pos')).otherwise(0.0).alias('pos'))
    coins = sorted(p['coin'].unique().to_list())
    def mat(col, fill=0.0):
        w = pivot(p, col)
        return w['ts'].to_numpy(), np.nan_to_num(w.select(coins).to_numpy().astype(float), nan=fill)
    ts, S = mat('pos')
    _, HF = mat('hl_funding'); _, BF = mat('bn_funding')
    _, HP = mat('hl_premium'); _, BP = mat('bn_premium')
    _, QV = mat('bn_qvol')
    gross = np.maximum(1.0, np.abs(S).sum(1, keepdims=True) / cost.max_gross)
    W = S / gross                                                  # capped weights
    maker = cost.execution == 'maker'
    if maker:                                                      # passive orders fill over the next hour
        W = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    N = W * cost.unit_notional                                     # signed USD notional per coin
    T = len(ts)
    fund = np.zeros(T); basis = np.zeros(T); fees = np.zeros(T)
    slip = np.array([cost.major_slip_bps if c in MAJORS else cost.alt_slip_bps for c in coins])
    qv = pl.DataFrame(QV).with_columns([pl.all().rolling_mean(24, min_samples=1)]).to_numpy()   # trailing 24h avg
    prevN = np.zeros(len(coins))
    for t in range(T):
        if t > 0:
            fund[t] = np.sum(prevN * (BF[t] - HF[t]))
            basis[t] = np.sum(prevN * ((HP[t] - HP[t - 1]) - (BP[t] - BP[t - 1])))
        dN = np.abs(N[t] - prevN)
        if dN.any():
            if maker:
                fees[t] = np.sum(dN * (cost.hl_maker_bps + cost.bn_maker_bps + 2 * cost.maker_adverse_bps
                                       + cost.basis_noise_bps) / 1e4)
            else:
                impact = cost.impact_coef * dN / np.maximum(qv[t - 1] if t > 0 else qv[t], 1e5)
                per_leg = slip + np.minimum(impact, 50.0)
                fees[t] = np.sum(dN * (cost.hl_taker_bps + cost.bn_taker_bps + 2 * per_leg + cost.basis_noise_bps) / 1e4)
        prevN = N[t]
    pnl = fund + basis - fees
    return {'ts': ts, 'pnl': pnl, 'funding': fund, 'basis': basis, 'fees': fees, 'weights': W, 'coins': coins}


def summarize_carry(res, cost: CarryCost, from_ms=None, to_ms=None):
    ts, pnl = res['ts'], res['pnl']
    m = np.ones(len(ts), bool)
    if from_ms is not None:
        m &= ts >= from_ms
    if to_ms is not None:
        m &= ts < to_ms
    ts, pnl = ts[m], pnl[m]
    W = res['weights'][m]
    day = ts // 86_400_000
    ud, inv = np.unique(day, return_inverse=True)
    daily = np.bincount(inv, weights=pnl) / cost.unit_notional
    from .stats import sharpe, max_drawdown
    turnover = float(np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0)).sum())
    sgn = np.sign(np.round(W, 6)); prev = np.vstack([np.zeros((1, W.shape[1])), sgn[:-1]])
    trades = int(np.sum((sgn != 0) & (sgn != prev)))
    return {'days': int(len(ud)), 'pnl': float(pnl.sum()), 'sharpe': sharpe(daily), 'mean_daily_bps': float(daily.mean() * 1e4) if len(daily) else 0.0,
            'max_drawdown': max_drawdown(np.cumsum(daily)), 'funding_pnl': float(res['funding'][m].sum()),
            'basis_pnl': float(res['basis'][m].sum()), 'fees': float(res['fees'][m].sum()), 'trades': trades,
            'turnover_units': turnover, 'exposure': float(np.mean(np.abs(W).sum(1) > 0)),
            'positive_days': float(np.mean(daily > 0)) if len(daily) else 0.0,
            'avg_gross': float(np.abs(W).sum(1).mean()), 'coins_traded': int((np.abs(W).sum(0) > 0).sum()),
            'daily': {'day': [int(d) for d in ud], 'ret': [float(r) for r in daily]}, 'execution': 'carry'}


def carry_signal(mod, panel, params=None):
    p = dict(mod.PARAMS) if params is None else params
    out = mod.signal(panel, p)
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(out) != len(panel):
        raise ValueError(f'carry signal length {len(out)} != panel rows {len(panel)}')
    return np.clip(np.where(np.isfinite(out), out, 0.0), -1, 1)


def carry_causality(mod, panel, cuts=(0.4, 0.7)):
    full = carry_signal(mod, panel)
    ts = panel['ts'].to_numpy()
    uts = np.unique(ts)
    issues = []
    for c in cuts:
        cut = uts[int(len(uts) * c)]
        k = int(np.sum(ts <= cut))
        pre = carry_signal(mod, panel.head(k))
        bad = int(np.sum(np.abs(pre - full[:k]) > 1e-6))
        if bad:
            issues.append(f'prefix cut at {cut}: {bad} positions changed when later hours were removed (future data use)')
    return {'ok': not issues, 'issues': issues, 'rows': len(panel)}


def evaluate_carry(code, cstore, split='dev', cost=None, params=None, variants=None, with_causality=True, keep=False):
    from .strategy import load_strategy, code_hash
    mod = load_strategy(code)
    cost = cost or CarryCost()
    panel, a, b = cstore.load(split, warmup_days=int(getattr(mod, 'WARMUP_DAYS', 30)))
    pos = carry_signal(mod, panel, params)
    res = run_carry(panel, pos, cost, trade_from_ms=a)
    out = summarize_carry(res, cost, a, b)
    out['code_hash'] = code_hash(code)
    out['resolution'] = '1h'
    out['causality'] = carry_causality(mod, panel) if with_causality else None
    out['variants'] = {}
    for name, (c, tr) in (variants or {}).items():
        pp = pos if tr is None else (np.clip(pos, 0, 1) if tr == 'long_only' else np.clip(pos, -1, 0))
        out['variants'][name] = summarize_carry(run_carry(panel, pp, c, trade_from_ms=a), c, a, b)
    if keep:
        out['_panel'], out['_pos'], out['_res'] = panel, pos, res
    return out
