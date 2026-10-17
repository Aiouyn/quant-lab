"""MACRO desk: daily cross-asset strategies (equities, sectors, countries, factor ETFs, bonds, commodities, currencies,
FX spot, crypto) on decades of data (panel: desk.data.macro_data).

Strategy contract (DESK = 'macro'):
    PARAMS = {...};  WARMUP_DAYS = 300
    def signal(panel, params) -> array aligned to panel rows (sorted by date, symbol): target weight in [-1, 1]
Rows of non-tradable series (tradable == 0: indices, yields, VIX, continuous futures) are FEATURES: their weights are
ignored. Weights decided with data up to the close of day t execute at the close of day t+1 (no time-zone leaks:
an asset closing later than another can never inform a same-day trade) and are held to the next close.
    pnl[t+2] = sum W[t] * (adjclose[t+2] / adjclose[t+1] - 1)                (total return incl. dividends)
             - |W[t] - W[t-1]| * cost_bps(symbol) * fee_mult                 (charged at execution)
             - short notional * short_borrow / 252
Gross is capped at max_gross (default 2: diversified multi-asset books need some leverage). FX spot returns exclude
the interest-rate carry. Splits: dev 2000..2014, val 2015..2019, holdout 2020..2025-06, forward 2025-07.. (daily).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import polars as pl

SPLITS = {'dev': ['2000-01-01', '2014-12-31'], 'val': ['2015-01-01', '2019-12-31'], 'holdout': ['2020-01-01', '2025-06-30'],
          'forward': ['2025-07-01', '2099-12-31'],
          'note': 'macro desk, frozen 2026-10-06; forward grows daily (Yahoo refresh). Known history: famous episodes '
                  '(2008, 2020, 2022) are public knowledge - only the forward split is free of hindsight.'}


@dataclass
class MacroCost:
    fee_mult: float = 1.0
    short_borrow: float = 0.01          # annual, on short notional (not FX)
    max_gross: float = 2.0
    delay_d: int = 1                    # decision at close t -> execution at close t + delay_d
    unit_notional: float = 100_000.0

    def stressed(self, fee_mult=1.5, delay_d=0):
        d = asdict(self); d.update(fee_mult=self.fee_mult * fee_mult, delay_d=self.delay_d + delay_d)
        return MacroCost(**d)


class MacroStore:
    def __init__(self, settings, allowed=('dev',), panel_path=None):
        self.root = settings.store / 'macro'
        self.allowed = set(allowed)
        self.panel_path = panel_path or self.root / 'panel.parquet'
        self._panel = None

    def splits(self):
        f = self.root / 'splits.json'
        if not f.exists():
            f.write_text(json.dumps(SPLITS, indent=1))
        return json.loads(f.read_text())

    def range_dates(self, split):
        if split not in self.allowed:
            from ..data.store import SplitViolation
            raise SplitViolation(f'macro {split} data is not available in this context')
        a, b = self.splits()[split]
        return date.fromisoformat(a), date.fromisoformat(b)

    def panel(self):
        if self._panel is None:
            self._panel = pl.read_parquet(self.panel_path).sort(['date', 'symbol'])
        return self._panel

    def load(self, split, warmup_days=300, symbols=None):
        a, b = self.range_dates(split)
        latest = max(self.range_dates(s)[1] for s in self.allowed)
        start = date.fromordinal(a.toordinal() - int(warmup_days * 1.5))
        p = self.panel().filter((pl.col('date') >= start) & (pl.col('date') <= min(b, latest)))
        if symbols:
            p = p.filter(pl.col('symbol').is_in(list(symbols)))
        return p, a, b


def write_dev_view(settings):
    st = MacroStore(settings, allowed=('dev',))
    p, a, b = st.load('dev', warmup_days=0)
    out = st.root / 'views' / 'dev.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    p.write_parquet(out, compression='zstd')
    return out


def macro_signal(mod, panel, params=None):
    p = dict(mod.PARAMS) if params is None else params
    out = mod.signal(panel, p)
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(out) != len(panel):
        raise ValueError(f'macro signal length {len(out)} != panel rows {len(panel)}')
    return np.clip(np.where(np.isfinite(out), out, 0.0), -1, 1)


def run_macro(panel: pl.DataFrame, weights, cost: MacroCost, trade_from=None, side=None):
    p = panel.with_columns(pl.Series('w', np.asarray(weights, float)))
    p = p.with_columns(pl.when(pl.col('tradable') == 1).then(pl.col('w')).otherwise(0.0).alias('w'))
    if trade_from is not None:
        p = p.with_columns(pl.when(pl.col('date') >= trade_from).then(pl.col('w')).otherwise(0.0).alias('w'))
    if side == 'long_only':
        p = p.with_columns(pl.col('w').clip(0, None))
    elif side == 'short_only':
        p = p.with_columns(pl.col('w').clip(None, 0))
    t = p.filter(pl.col('tradable') == 1)
    syms = sorted(t['symbol'].unique().to_list())
    dates_s = t.select('date').unique().sort('date')['date']
    dates = np.array(dates_s.to_list(), dtype=object)

    def wide(col):
        w = t.select('date', 'symbol', col).pivot(on='symbol', index='date', values=col, aggregate_function='first')
        w = pl.DataFrame({'date': dates_s}).join(w, on='date', how='left')
        return w.select([s if s in w.columns else pl.lit(None).alias(s) for s in syms])
    S = wide('w').to_numpy().astype(float)
    present = np.isfinite(S)
    PX = wide('adjclose').fill_null(strategy='forward').to_numpy().astype(float)
    meta = t.group_by('symbol').agg(pl.col('cost_bps').first(), pl.col('cls').first()).sort('symbol')
    cost_v = meta['cost_bps'].to_numpy() / 1e4 * cost.fee_mult
    borrow_ok = (meta['cls'].to_numpy() != 'fx')
    S = np.nan_to_num(S)
    T, C = S.shape
    W = np.zeros((T, C)); cur = np.zeros(C)
    for i in range(T):
        d = i - cost.delay_d
        if d >= 0:
            held = np.where(present[i], 0.0, cur)
            want = np.where(present[i], S[d], 0.0)
            budget = max(0.0, cost.max_gross - np.abs(held).sum())
            g = np.abs(want).sum()
            if g > budget and g > 0:
                want = want * (budget / g)
            cur = held + want
        W[i] = cur
    R = np.zeros((T, C))
    with np.errstate(invalid='ignore', divide='ignore'):
        R[1:] = np.nan_to_num(PX[1:] / PX[:-1] - 1)
    R = np.clip(R, -0.9, 3.0)
    Wp = np.vstack([np.zeros((1, C)), W[:-1]])
    price = (Wp * R).sum(1)
    fees = (np.abs(W - Wp * (1 + R)) * cost_v).sum(1)     # drifted positions are rebalanced back to target
    borrow = (np.clip(-Wp, 0, None) * borrow_ok).sum(1) * cost.short_borrow / 252
    pnl = price - fees - borrow
    u = cost.unit_notional
    return {'dates': dates, 'pnl': pnl * u, 'price': price * u, 'fees': fees * u, 'borrow': borrow * u, 'weights': W,
            'symbols': syms}


def summarize_macro(res, cost: MacroCost, from_d=None, to_d=None):
    from .stats import sharpe, max_drawdown
    d, pnl, W = res['dates'], res['pnl'], res['weights']
    m = np.ones(len(d), bool)
    if from_d is not None:
        m &= d >= from_d
    if to_d is not None:
        m &= d <= to_d
    d, pnl, W = d[m], pnl[m], W[m]
    daily = pnl / cost.unit_notional
    sgn = np.sign(np.round(W, 6)); prev = np.vstack([np.zeros((1, W.shape[1])), sgn[:-1]])
    srt = np.sort(daily)
    years = max(len(daily) / 252, 1e-9)
    return {'days': int(len(daily)), 'pnl': float(pnl.sum()), 'sharpe': sharpe(daily, periods=252),
            'ann_return': float(daily.sum() / years), 'ann_vol': float(daily.std() * np.sqrt(252)) if len(daily) > 1 else 0.0,
            'mean_daily_bps': float(daily.mean() * 1e4) if len(daily) else 0.0,
            'median_daily_bps': float(np.median(daily) * 1e4) if len(daily) else 0.0,
            'mean_ex_top5_days_bps': float(srt[:-5].mean() * 1e4) if len(srt) > 10 else 0.0,
            'max_drawdown': max_drawdown(np.cumsum(daily)), 'fees': float(res['fees'][m].sum()),
            'borrow': float(res['borrow'][m].sum()), 'trades': int(np.sum((sgn != 0) & (sgn != prev))),
            'turnover_units': float(np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), W]), axis=0)).sum()),
            'exposure': float(np.mean(np.abs(W).sum(1) > 0)), 'avg_gross': float(np.abs(W).sum(1).mean()),
            'net_exposure': float(W.sum(1).mean()), 'assets_traded': int((np.abs(W).sum(0) > 0).sum()),
            'positive_days': float(np.mean(daily > 0)) if len(daily) else 0.0,
            'daily': {'day': [int(x.toordinal() - 719163) for x in d], 'ret': [float(r) for r in daily]}, 'execution': 'macro'}


def macro_causality(mod, panel, cuts=(0.4, 0.7)):
    full = macro_signal(mod, panel)
    ds = panel['date'].to_numpy()
    ud = np.unique(ds)
    issues = []
    for c in cuts:
        cut = ud[int(len(ud) * c)]
        k = int(np.sum(ds <= cut))
        pre = macro_signal(mod, panel.head(k))
        bad = int(np.sum(np.abs(pre - full[:k]) > 1e-6))
        if bad:
            issues.append(f'prefix cut at {cut}: {bad} weights changed when later days were removed (future data use)')
    return {'ok': not issues, 'issues': issues, 'rows': len(panel)}


def evaluate_macro(code, store: MacroStore, split='dev', cost=None, params=None, variants=None, with_causality=True):
    from .strategy import load_strategy, code_hash
    mod = load_strategy(code)
    cost = cost or MacroCost()
    panel, a, b = store.load(split, warmup_days=int(getattr(mod, 'WARMUP_DAYS', 300)))
    w = macro_signal(mod, panel, params)
    out = summarize_macro(run_macro(panel, w, cost, trade_from=a), cost, a, b)
    out['code_hash'] = code_hash(code)
    out['causality'] = macro_causality(mod, panel) if with_causality else None
    out['variants'] = {}
    for name, (c, side) in (variants or {}).items():
        out['variants'][name] = summarize_macro(run_macro(panel, w, c, trade_from=a, side=side), c, a, b)
    return out
