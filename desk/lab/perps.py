"""PERPS desk: multi-asset Hyperliquid perps, hourly decisions (panel: desk.data.perps_panel).

Strategy contract (DESK = 'perps'):
    PARAMS = {...};  WARMUP_DAYS = 30
    def signal(panel, params) -> array aligned to panel rows (sorted by ts, coin): target weight in [-1, 1]
Weights decided at hour t (using rows with ts <= t only) are scaled to a gross cap (sum |w| <= 1) and held over
(t, t+1].  Engine:
    pnl[t+1] = sum_c W[t] * (px_fill[t+1] / px_fill[t] - 1)          px_fill = HL mid ~60 s after the decision hour
             - sum_c W[t] * f[t+1]                                  funding paid at t+1 (longs pay when f > 0)
             - sum_c |W[t] - W[t-1]| * (taker 4.5 bps + 0.5 * impact spread (capped 50 bps)) * fee/impact multipliers
A coin with no row at t (illiquid day, delisted, data gap) cannot be traded: its position is HELD through the gap and
its price is carried forward, so the move is realized when it reappears.
Returns are fractions of gross capital (unit_notional USD for pnl). Splits: dev / val / holdout (sealed) / forward
(live data appended daily, never visible to research).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import polars as pl

from ..data.atomic import atomic_write_json, atomic_write_parquet

HOUR = 3_600_000
SPLITS = {'dev': ['2023-06-01', '2025-05-31'], 'val': ['2025-06-01', '2026-04-01'], 'holdout': ['2026-04-02', '2026-10-01'],
          'forward': ['2026-10-02', '2099-12-31'],
          'note': 'perps desk, frozen 2026-10-05. val/holdout were already used once by 2026-10-03 family tests '
                  '(RESEARCH_LOG_2026-10-03.md); forward = verified live recordings, appended daily.'}


@dataclass
class PerpsCost:
    taker_bps: float = 4.5
    spread_frac: float = 0.5           # fraction of the impact spread paid per unit turnover
    spread_cap_bps: float = 50.0
    fee_mult: float = 1.0
    impact_mult: float = 1.0
    delay_h: int = 0                   # extra decision->execution delay in hours (robustness)
    max_gross: float = 1.0
    unit_notional: float = 100_000.0
    execution: str = 'taker'           # 'maker': limit order at the h:01 price, filled only if the next hour trades
    maker_bps: float = 1.5             #   THROUGH it by through_bps (else the position is unchanged, retried next hour)
    through_bps: float = 2.0

    def as_taker(self):
        d = asdict(self); d['execution'] = 'taker'
        return PerpsCost(**d)

    def stressed(self, fee_mult=1.5, impact_mult=1.0, delay_h=0):
        d = asdict(self); d.update(fee_mult=self.fee_mult * fee_mult, impact_mult=self.impact_mult * impact_mult,
                                   delay_h=self.delay_h + delay_h)
        return PerpsCost(**d)


def _ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


class PerpsStore:
    def __init__(self, settings, allowed=('dev',), panel_path=None):
        self.root = settings.store / 'perps'
        self.allowed = set(allowed)
        self.panel_path = panel_path or self.root / 'panel.parquet'
        self._panel = None

    def splits(self):
        f = self.root / 'splits.json'
        if not f.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(f, SPLITS, indent=1)
        return json.loads(f.read_text())

    def range_ms(self, split):
        if split not in self.allowed:
            from ..data.store import SplitViolation
            raise SplitViolation(f'perps {split} data is not available in this context')
        a, b = self.splits()[split]
        return _ms(a), _ms(b) + 86_400_000

    def panel(self):
        if self._panel is None:
            self._panel = pl.read_parquet(self.panel_path).sort(['ts', 'coin'])
        return self._panel

    def load(self, split, warmup_days=30, coins=None):
        a, b = self.range_ms(split)
        latest = max(self.range_ms(s)[1] for s in self.allowed)
        p = self.panel().filter((pl.col('ts') >= a - warmup_days * 86_400_000) & (pl.col('ts') < min(b, latest)))
        if coins:
            p = p.filter(pl.col('coin').is_in(list(coins)))
        return p, a, b


def write_dev_view(settings):
    """Dev-only copy of the panel for research sandboxes (the data-layer split guard)."""
    st = PerpsStore(settings, allowed=('dev',))
    p, a, b = st.load('dev', warmup_days=0)
    if 'lo' in p.columns:                                             # fill ranges for research maker backtests
        (st.root / 'views').mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(p.select('ts', 'coin', 'lo', 'hi'), st.root / 'views' / 'dev_fills.parquet')
    p = visible(p)
    out = st.root / 'views' / 'dev.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(p, out)
    return out


def _wide(panel, col, coins, ts_index):
    w = panel.select('ts', 'coin', col).pivot(on='coin', index='ts', values=col, aggregate_function='first')
    w = pl.DataFrame({'ts': ts_index}).join(w, on='ts', how='left')
    for c in coins:
        if c not in w.columns:
            w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return w.select(coins).to_numpy().astype(float)


ENGINE_ONLY = ('lo', 'hi')            # post-decision price range: fill simulation only, never visible to strategies


def visible(panel):
    return panel.drop([c for c in ENGINE_ONLY if c in panel.columns])


def perps_signal(mod, panel, params=None):
    p = dict(mod.PARAMS) if params is None else params
    out = mod.signal(visible(panel), p)
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(out) != len(panel):
        raise ValueError(f'perps signal length {len(out)} != panel rows {len(panel)}')
    return np.clip(np.where(np.isfinite(out), out, 0.0), -1, 1)


def run_perps(panel: pl.DataFrame, weights, cost: PerpsCost, trade_from_ms=None, side=None):
    p = panel.with_columns(pl.Series('w', np.asarray(weights, float)))
    if trade_from_ms is not None:
        p = p.with_columns(pl.when(pl.col('ts') >= trade_from_ms).then(pl.col('w')).otherwise(0.0).alias('w'))
    if side == 'long_only':
        p = p.with_columns(pl.col('w').clip(0, None))
    elif side == 'short_only':
        p = p.with_columns(pl.col('w').clip(None, 0))
    coins = sorted(p['coin'].unique().to_list())
    t0, t1 = int(p['ts'].min()), int(p['ts'].max())
    ts = np.arange(t0, t1 + HOUR, HOUR, dtype=np.int64)                  # contiguous hourly grid
    S = _wide(p, 'w', coins, ts)
    present = np.isfinite(S)
    PX = pl.DataFrame(_wide(p, 'px_fill', coins, ts)).fill_null(strategy='forward').to_numpy()
    F = np.nan_to_num(_wide(p, 'f', coins, ts))
    ISP = pl.DataFrame(_wide(p, 'isp', coins, ts)).fill_null(strategy='forward').to_numpy()
    ISP = np.nan_to_num(ISP, nan=cost.spread_cap_bps)
    S = np.nan_to_num(S)
    T, C = S.shape
    maker = cost.execution == 'maker'
    if maker:
        if 'lo' not in p.columns:
            raise ValueError('maker execution needs the engine-only lo/hi columns of the perps panel')
        LO = _wide(p, 'lo', coins, ts); HI = _wide(p, 'hi', coins, ts)
        thr = cost.through_bps / 1e4
    W = np.zeros((T, C))
    cur = np.zeros(C)
    for t in range(T):
        d = t - cost.delay_h
        if d >= 0:
            held = np.where(present[t], 0.0, cur)                          # untradeable coins keep their position
            want = np.where(present[t], S[d], 0.0)
            budget = max(0.0, cost.max_gross - np.abs(held).sum())         # gross cap includes the held positions
            g = np.abs(want).sum()
            if g > budget:
                want = want * (budget / g) if g > 0 else want
            new = held + want
            if maker:                                                       # only trades-through fill
                buy_ok = LO[t] <= PX[t] * (1 - thr); sell_ok = HI[t] >= PX[t] * (1 + thr)
                fill = np.where(new > cur, buy_ok, np.where(new < cur, sell_ok, True))
                new = np.where(np.nan_to_num(fill, nan=0).astype(bool), new, cur)
            cur = new
        W[t] = cur
    R = np.zeros((T, C))
    with np.errstate(invalid='ignore', divide='ignore'):
        R[1:] = np.nan_to_num(PX[1:] / PX[:-1] - 1)
    R = np.clip(R, -0.9, 5.0)
    Wp = np.vstack([np.zeros((1, C)), W[:-1]])
    price = (Wp * R).sum(1)
    funding = -(Wp * F).sum(1)
    if maker:
        unit = np.full_like(ISP, cost.maker_bps * cost.fee_mult / 1e4)
    else:
        unit = (cost.taker_bps * cost.fee_mult + cost.spread_frac * np.minimum(ISP, cost.spread_cap_bps) * cost.impact_mult) / 1e4
    turn = np.abs(W - Wp * (1 + R))          # positions drift with their price before rebalancing back to target
    fees = (turn * unit).sum(1)
    pnl = price + funding - fees
    return {'ts': ts, 'pnl': pnl * cost.unit_notional, 'price': price * cost.unit_notional,
            'funding': funding * cost.unit_notional, 'fees': fees * cost.unit_notional, 'weights': W, 'coins': coins}


def summarize_perps(res, cost: PerpsCost, from_ms=None, to_ms=None):
    from .stats import sharpe, max_drawdown
    ts, pnl = res['ts'], res['pnl']
    m = np.ones(len(ts), bool)
    if from_ms is not None:
        m &= ts >= from_ms
    if to_ms is not None:
        m &= ts < to_ms
    ts, pnl, W = ts[m], pnl[m], res['weights'][m]
    day = ts // 86_400_000
    ud, inv = np.unique(day, return_inverse=True)
    daily = np.bincount(inv, weights=pnl) / cost.unit_notional
    sgn = np.sign(np.round(W, 6)); prev = np.vstack([np.zeros((1, W.shape[1])), sgn[:-1]])
    srt = np.sort(daily)
    return {'days': int(len(ud)), 'pnl': float(pnl.sum()), 'sharpe': sharpe(daily),
            'mean_daily_bps': float(daily.mean() * 1e4) if len(daily) else 0.0,
            'median_daily_bps': float(np.median(daily) * 1e4) if len(daily) else 0.0,
            'mean_ex_top5_days_bps': float(srt[:-5].mean() * 1e4) if len(srt) > 10 else 0.0,
            'max_drawdown': max_drawdown(np.cumsum(daily)), 'price_pnl': float(res['price'][m].sum()),
            'funding_pnl': float(res['funding'][m].sum()), 'fees': float(res['fees'][m].sum()),
            'trades': int(np.sum((sgn != 0) & (sgn != prev))),
            'turnover_units': float(np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0)).sum()),
            'exposure': float(np.mean(np.abs(W).sum(1) > 0)), 'avg_gross': float(np.abs(W).sum(1).mean()),
            'net_exposure': float(W.sum(1).mean()), 'coins_traded': int((np.abs(W).sum(0) > 0).sum()),
            'positive_days': float(np.mean(daily > 0)) if len(daily) else 0.0,
            'daily': {'day': [int(d) for d in ud], 'ret': [float(r) for r in daily]}, 'execution': 'perps'}


def perps_causality(mod, panel, cuts=(0.4, 0.7)):
    full = perps_signal(mod, panel)
    ts = panel['ts'].to_numpy()
    uts = np.unique(ts)
    issues = []
    for c in cuts:
        cut = uts[int(len(uts) * c)]
        k = int(np.sum(ts <= cut))
        pre = perps_signal(mod, panel.head(k))
        bad = int(np.sum(np.abs(pre - full[:k]) > 1e-6))
        if bad:
            issues.append(f'prefix cut at {cut}: {bad} weights changed when later hours were removed (future data use)')
    return {'ok': not issues, 'issues': issues, 'rows': len(panel)}


def evaluate_perps(code, store: PerpsStore, split='dev', cost=None, params=None, variants=None, with_causality=True):
    from .strategy import load_strategy, code_hash
    mod = load_strategy(code)
    cost = cost or PerpsCost()
    if getattr(mod, 'PERPS_EXECUTION', 'taker') == 'maker':
        from dataclasses import replace
        cost = replace(cost, execution='maker')
    panel, a, b = store.load(split, warmup_days=int(getattr(mod, 'WARMUP_DAYS', 30)))
    w = perps_signal(mod, panel, params)
    out = summarize_perps(run_perps(panel, w, cost, trade_from_ms=a), cost, a, b)
    out['code_hash'] = code_hash(code)
    out['causality'] = perps_causality(mod, panel) if with_causality else None
    out['variants'] = {}
    for name, (c, side) in (variants or {}).items():
        if cost.execution == 'maker' and name != 'taker_execution' and c.execution == 'taker':
            from dataclasses import replace
            c = replace(c, execution='maker')
        out['variants'][name] = summarize_perps(run_perps(panel, w, c, trade_from_ms=a, side=side), c, a, b)
    return out
