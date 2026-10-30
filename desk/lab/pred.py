"""PRED desk: binary prediction markets (Polymarket), hourly decisions (panel: desk.data.polymarket).

Strategy contract (DESK = 'pred'):
    PARAMS = {...};  WARMUP_DAYS = 0
    def signal(panel, params) -> array aligned to panel rows (sorted by ts, market): weight in [-1, 1]
      +w = hold YES worth w of capital, -w = hold NO worth |w|; gross capped at max_gross across markets.
Visible columns: ts, market, p (YES price at ts), end_ms (scheduled end, public), category, hours_to_end.
ENGINE-ONLY (never visible): outcome (the resolution = the future) and volume (FINAL lifetime volume).
Execution: decided at ts, filled at the next observed price of that market (~1 h later) paying half_spread
(default 1 cent, capped at half the distance to 0/1, min one 0.1-cent tick) per share: cost = |d yes| * h / p + |d no| * h / (1 - p). At resolution YES pays 1 or 0.
Caveat: the universe is resolved markets with lifetime volume >= $250k (a mild look-ahead selection on popularity).
Splits on ts: dev <= 2026-02, val 2026-03..05, holdout 2026-06..07, forward 2026-08+ (resolved markets refreshed daily).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import polars as pl

HOUR = 3_600_000
SPLITS = {'dev': ['2020-01-01', '2026-02-28'], 'val': ['2026-03-01', '2026-05-31'], 'holdout': ['2026-06-01', '2026-07-31'],
          'forward': ['2026-08-01', '2099-12-31'],
          'note': 'pred desk, frozen 2026-10-06 by market-count share (~57/22/19/rest) before any outcome analysis: the '
                  '>= $250k universe is concentrated in 2025-26 (43 markets end before 2024)'}
ENGINE_ONLY = ('outcome', 'volume')


@dataclass
class PredCost:
    half_spread: float = 0.01
    fee_mult: float = 1.0
    max_gross: float = 1.0
    delay_rows: int = 1
    unit_notional: float = 100_000.0

    def stressed(self, fee_mult=1.5, delay_rows=0):
        d = asdict(self); d.update(fee_mult=self.fee_mult * fee_mult, delay_rows=self.delay_rows + delay_rows)
        return PredCost(**d)


def _ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def visible(panel):
    p = panel.drop([c for c in ENGINE_ONLY if c in panel.columns])
    if 'end_ms' in p.columns and 'hours_to_end' not in p.columns:
        p = p.with_columns(((pl.col('end_ms') - pl.col('ts')) / HOUR).alias('hours_to_end'))
    return p


class PredStore:
    def __init__(self, settings, allowed=('dev',)):
        self.root = settings.store / 'pred'
        self.allowed = set(allowed)
        self._panel = None

    def splits(self):
        from ..data.provenance import read_splits, write_splits
        f = self.root / 'splits.json'
        if not f.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            write_splits(f, SPLITS)
        return read_splits(f)

    def range_ms(self, split, _lock=True):
        if split not in self.allowed:
            from ..data.store import SplitViolation
            raise SplitViolation(f'pred {split} data is not available in this context')
        a, b = self.splits()[split]
        if split == 'holdout' and _lock:
            from .holdout import assert_holdout_open
            assert_holdout_open('pred', [a, b])
        return _ms(a), _ms(b) + 86_400_000

    def panel(self):
        if self._panel is None:
            self._panel = pl.read_parquet(self.root / 'panel.parquet').sort(['ts', 'market'])
        return self._panel

    def load(self, split, warmup_days=0, coins=None):
        a, b = self.range_ms(split)
        latest = max(self.range_ms(s, _lock=False)[1] for s in self.allowed)
        # lazy scan: filter on disk (the full panel is ~25M rows); compact dtypes
        # only markets that RESOLVED inside the allowed window can be scored (their outcome is in the past)
        p = (pl.scan_parquet(self.root / 'panel.parquet')
             .filter((pl.col('ts') >= a - warmup_days * 86_400_000) & (pl.col('ts') < min(b, latest)) & (pl.col('end_ms') < latest))
             .with_columns(pl.col('market').cast(pl.Int64), pl.col('p').cast(pl.Float32), pl.col('outcome').cast(pl.Int8),
                           pl.col('category').cast(pl.Categorical))
             .collect())                                # the panel file is already sorted by (ts, market)
        return p, a, b


def write_dev_view(settings):
    st = PredStore(settings, allowed=('dev',))
    p, a, b = st.load('dev')
    out = st.root / 'views' / 'dev.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    visible(p).write_parquet(out, compression='zstd')
    p.select('ts', 'market', 'outcome').write_parquet(st.root / 'views' / 'dev_outcomes.parquet', compression='zstd')
    return out


def pred_signal(mod, panel, params=None):
    p = dict(mod.PARAMS) if params is None else params
    out = mod.signal(visible(panel), p)
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(out) != len(panel):
        raise ValueError(f'pred signal length {len(out)} != panel rows {len(panel)}')
    return np.clip(np.where(np.isfinite(out), out, 0.0), -1, 1)


def run_pred(panel: pl.DataFrame, weights, cost: PredCost, trade_from_ms=None, side=None):
    """Share-holding semantics (how binaries are really held): when a market's target weight changes, the position is
    re-sized to it (paying the half-spread on the traded part); while the target is unchanged the SHARES are held and
    the position value drifts with the price (no trading, no cost). Strategies should therefore emit stable targets:
    a target that changes every hour is re-traded every hour. Only markets with a non-zero target are simulated."""
    p = panel.sort(['ts', 'market']).with_columns(pl.Series('w', np.asarray(weights, float)))
    if trade_from_ms is not None:
        p = p.with_columns(pl.when(pl.col('ts') >= trade_from_ms).then(pl.col('w')).otherwise(0.0).alias('w'))
    if side == 'long_only':                     # YES only
        p = p.with_columns(pl.col('w').clip(0, None))
    elif side == 'short_only':                  # NO only
        p = p.with_columns(pl.col('w').clip(None, 0))
    traded = p.filter(pl.col('w') != 0).select('market').unique()
    p = p.join(traded, on='market', how='semi').sort(['market', 'ts'])
    cols = ['market', 'ts', 'p', 'end_ms', 'outcome', 'w']
    if p.is_empty():
        return pl.DataFrame(schema={'market': pl.Int64, 'ts': pl.Int64, 'W': pl.Float64, 'price_pnl': pl.Float64,
                                    'cost': pl.Float64, 'pnl': pl.Float64})
    # gross exposure per hour needs every market at once: compute it on a light frame, then simulate markets in
    # independent chunks (memory stays bounded however many markets a strategy trades)
    g = (p.select('market', 'ts', pl.col('w').cast(pl.Float64)).lazy()
         .with_columns(pl.col('w').shift(cost.delay_rows).over('market').fill_null(0.0).abs().alias('a'))
         .group_by('ts').agg(pl.col('a').sum().alias('g')).collect())
    mk = p['market'].unique().sort().to_list()
    out = []
    for i in range(0, len(mk), 1500):
        chunk = p.filter(pl.col('market').is_in(mk[i:i + 1500]))
        out.append(_simulate(chunk.lazy(), g, cost, cols))
    return pl.concat(out).sort(['market', 'ts'])


def _simulate(p, g, cost, cols):
    # terminal row per market: at end_ms the YES price becomes the outcome (payout, no trading cost)
    term = p.group_by('market').agg(pl.col('end_ms').first().alias('ts'), pl.col('outcome').first().cast(pl.Float64).alias('p'),
                                    pl.col('end_ms').first(), pl.col('outcome').first())
    term = term.with_columns(pl.lit(0.0).alias('w'), pl.lit(True).alias('_term'))
    norm = [pl.col('p').cast(pl.Float64), pl.col('outcome').cast(pl.Int64), pl.col('w').cast(pl.Float64), pl.col('end_ms').cast(pl.Int64)]
    p = pl.concat([p.select(cols).with_columns(*norm, pl.lit(False).alias('_term')),
                   term.select(cols + ['_term']).with_columns(*norm)]).sort(['market', 'ts', '_term'])
    # execution delay (rows within a market), gross cap on TARGETS per timestamp
    p = p.with_columns(pl.col('w').shift(cost.delay_rows).over('market').fill_null(0.0).alias('we'))
    p = p.with_columns(pl.when(pl.col('_term')).then(0.0).otherwise(pl.col('we')).alias('we'))
    p = p.join(g.lazy(), on='ts', how='left').with_columns(pl.col('g').fill_null(0.0)).with_columns((pl.col('we') / pl.max_horizontal(pl.lit(1.0), pl.col('g') / cost.max_gross)).alias('_traw'))
    p = p.sort(['market', 'ts', '_term'])
    # the cap scales a position when the STRATEGY's target changes; afterwards it is held (other markets entering or
    # leaving must not re-trade it every hour). Gross can therefore drift somewhat above max_gross between decisions.
    p = p.with_columns((pl.col('we') != pl.col('we').shift(1).over('market').fill_null(0.0)).cast(pl.Int32).cum_sum()
                       .over('market').alias('_wseg'))
    p = p.with_columns(pl.col('_traw').first().over(['market', '_wseg']).alias('T'))
    # ONE consistent valuation series for both ends of every return: live rows clipped to [0.5c, 99.5c], the terminal
    # row exactly 0 or 1. (Clipping only the denominator booked a phantom ~0.5%/h on decided markets - fixed 2026-10-06.)
    pc = pl.when(pl.col('_term')).then(pl.col('p')).otherwise(pl.col('p').clip(0.005, 0.995))
    p = p.with_columns(pc.alias('pc'))
    p = p.with_columns(pl.col('pc').shift(-1).over('market').alias('p_next'))
    # holding segments: a new segment starts whenever the target changes
    p = p.with_columns((pl.col('T') != pl.col('T').shift(1).over('market').fill_null(0.0)).alias('_chg'))
    p = p.with_columns(pl.col('_chg').cast(pl.Int32).cum_sum().over('market').alias('_seg'))
    p = p.with_columns(pl.col('pc').first().over(['market', '_seg']).alias('_pe'))
    yes = pl.col('T').clip(0, None) * pl.col('pc') / pl.col('_pe')
    no = (-pl.col('T')).clip(0, None) * (1 - pl.col('pc')) / (1 - pl.col('_pe'))
    p = p.with_columns(yes.alias('yes'), no.alias('no'))
    ret_yes = pl.col('p_next') / pl.col('pc') - 1
    ret_no = (1 - pl.col('p_next')) / (1 - pl.col('pc')) - 1
    p = p.with_columns(pl.when(pl.col('p_next').is_null()).then(0.0)
                       .otherwise(pl.col('yes') * ret_yes + pl.col('no') * ret_no).alias('price_pnl'),
                       (pl.col('yes') - pl.col('no')).alias('W'))
    # half-spread: half_spread (1 cent) but never wider than half the distance to 0 or 1, and at least one 0.1-cent
    # tick (a 0.05-cent contract cannot be quoted 1 cent wide)
    h = pl.min_horizontal(pl.lit(cost.half_spread), pl.max_horizontal(pl.lit(0.001), 0.5 * pl.min_horizontal(pl.col('pc'), 1 - pl.col('pc')))) * cost.fee_mult
    pp = pl.col('pc').shift(1).over('market')
    y0 = (pl.col('yes').shift(1).over('market') * pl.col('pc') / pp).fill_null(0.0)      # previous shares, marked now
    n0 = (pl.col('no').shift(1).over('market') * (1 - pl.col('pc')) / (1 - pp)).fill_null(0.0)
    trade_cost = ((pl.col('T').clip(0, None) - y0).abs() * h / pl.col('pc')
                  + ((-pl.col('T')).clip(0, None) - n0).abs() * h / (1 - pl.col('pc')))
    p = p.with_columns(pl.when(pl.col('_chg') & ~pl.col('_term')).then(trade_cost).otherwise(0.0).alias('cost'))
    p = p.with_columns((pl.col('price_pnl') - pl.col('cost')).alias('pnl'))
    # only what the summary needs: projection pushdown keeps the other ~10 columns from ever materializing
    return p.select('market', 'ts', 'W', 'price_pnl', 'cost', 'pnl').collect()


def summarize_pred(p, cost: PredCost, from_ms=None, to_ms=None):
    from .stats import sharpe, max_drawdown
    q = p.lazy()
    if from_ms is not None:
        q = q.filter(pl.col('ts') >= from_ms)
    if to_ms is not None:
        q = q.filter(pl.col('ts') < to_ms)
    d = q.group_by((pl.col('ts') // 86_400_000).alias('day')).agg(pl.col('pnl').sum(), pl.col('cost').sum(),
                                                                  pl.col('W').abs().sum().alias('gw')).sort('day').collect()
    lo_d = int(from_ms // 86_400_000) if from_ms is not None else (int(d['day'].min()) if len(d) else None)
    hi_d = int((to_ms - 1) // 86_400_000) if to_ms is not None else (int(d['day'].max()) if len(d) else None)
    if lo_d is not None and hi_d is not None:          # every day of the evaluation window counts (idle days = 0)
        hi_d = min(hi_d, int(d['day'].max())) if len(d) and to_ms is not None and hi_d > 40_000 else hi_d
        full = pl.DataFrame({'day': np.arange(lo_d, hi_d + 1, dtype=np.int64)}).join(d.with_columns(pl.col('day').cast(pl.Int64)), on='day', how='left').fill_null(0.0)
    else:
        full = d
    daily = full['pnl'].to_numpy() if len(full) else np.zeros(0)
    srt = np.sort(daily)
    agg = q.select((pl.col('W').abs() > 0).alias('on'), 'market', 'W', 'cost')         .with_columns(((pl.col('W').abs() > 0) & (pl.col('W').shift(1).over('market').fill_null(0.0).abs() == 0)).alias('entry'))         .select(pl.col('entry').sum().alias('entries'), pl.col('market').filter(pl.col('on')).n_unique().alias('mk'),
                pl.col('cost').sum().alias('fees')).collect().row(0, named=True)
    return {'days': int(len(daily)), 'pnl': float(daily.sum() * cost.unit_notional), 'sharpe': sharpe(daily),
            'mean_daily_bps': float(daily.mean() * 1e4) if len(daily) else 0.0,
            'median_daily_bps': float(np.median(daily) * 1e4) if len(daily) else 0.0,
            'mean_ex_top5_days_bps': float(srt[:-5].mean() * 1e4) if len(srt) > 10 else 0.0,
            'max_drawdown': max_drawdown(np.cumsum(daily)) if len(daily) else 0.0,
            'fees': float((agg['fees'] or 0.0) * cost.unit_notional), 'trades': int(agg['entries'] or 0),
            'markets_traded': int(agg['mk'] or 0), 'exposure': float(np.mean(full['gw'].to_numpy() > 0)) if len(full) else 0.0,
            'positive_days': float(np.mean(daily > 0)) if len(daily) else 0.0,
            'daily': {'day': [int(x) for x in full['day'].to_list()] if len(full) else [], 'ret': [float(r) for r in daily]},
            'execution': 'pred'}


def pred_causality(mod, panel, cuts=(0.4, 0.7)):
    full = pred_signal(mod, panel)
    ts = panel['ts'].to_numpy()
    ut = np.unique(ts)
    issues = []
    for c in cuts:
        cut = ut[int(len(ut) * c)]
        k = int(np.sum(ts <= cut))
        pre = pred_signal(mod, panel.head(k))
        bad = int(np.sum(np.abs(pre - full[:k]) > 1e-6))
        if bad:
            issues.append(f'prefix cut at {cut}: {bad} weights changed when later hours were removed (future data use)')
    return {'ok': not issues, 'issues': issues, 'rows': len(panel)}


def evaluate_pred(code, store: PredStore, split='dev', cost=None, params=None, variants=None, with_causality=True):
    from .strategy import load_strategy, code_hash
    mod = load_strategy(code)
    cost = cost or PredCost()
    panel, a, b = store.load(split, warmup_days=int(getattr(mod, 'WARMUP_DAYS', 0)))
    panel = panel.sort(['ts', 'market'])
    # score from the first day the split actually has markets: the >=250k universe only starts ~2023, and counting
    # the empty years as flat days diluted every Sharpe and failed month/half-sample gates for every strategy
    if len(panel):
        a = max(a, int(panel['ts'].min()) // 86_400_000 * 86_400_000)
    w = pred_signal(mod, panel, params)
    causal = pred_causality(mod, panel) if with_causality else None
    # memory: only markets the strategy ever trades need simulating (15M+ panel rows otherwise, per variant)
    keep = panel.select('market').with_columns(pl.Series('w', w)).filter(pl.col('w') != 0).select('market').unique()
    mask = panel.select(pl.col('market').is_in(keep['market'].implode())).to_series().to_numpy()
    panel, w = panel.filter(pl.Series(mask)).select('ts', 'market', 'p', 'end_ms', 'outcome'), w[mask]
    out = summarize_pred(run_pred(panel, w, cost, trade_from_ms=a), cost, a, b)
    out['code_hash'] = code_hash(code)
    out['causality'] = causal
    out['variants'] = {}
    for name, (c, side) in (variants or {}).items():
        out['variants'][name] = summarize_pred(run_pred(panel, w, c, trade_from_ms=a, side=side), c, a, b)
    return out
