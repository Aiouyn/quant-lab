"""Strategy contract v3, loader, causality test and (walk-forward) evaluation over stored days.

A strategy is a Python module (string) defining:

    RESOLUTION = '1m'          # '1s' | '10s' | '1m'
    LOOKBACK = 720             # rows of history the signal needs before it is reliable
    PARAMS = {...}             # tunable parameters (used by robustness neighborhoods)
    def signal(df, params, model=None):   # -> target position per row in [-1, 1] (np.ndarray / pl.Series)
        ...
    # optional, for fitted strategies:
    TRAIN_DAYS = 30; REFIT_DAYS = 7
    def fit(train_df, params):            # -> picklable model; called ONLY on strictly earlier days
        ...

`df` holds store columns (+ funding_rate). The signal must be causal: row i may use rows <= i only.
"""
from __future__ import annotations

import hashlib
import math
import types

import numpy as np
import polars as pl

from .backtest import backtest, CostModel, RES_SECONDS

FORBIDDEN = ('import os', 'import subprocess', 'import socket', 'shutil', 'requests', 'httpx', 'urllib',
             'open(', '__import__', 'eval(', 'exec(', 'fwd_returns', 'forward_returns')


class StrategyError(ValueError):
    pass


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def load_strategy(code: str):
    for bad in FORBIDDEN:
        if bad in code:
            raise StrategyError(f'strategy code may not use `{bad}` (strategies are pure functions of past data)')
    mod = types.ModuleType('strategy_' + code_hash(code))
    mod.__dict__['np'] = np
    mod.__dict__['pl'] = pl
    exec(compile(code, f'<strategy {code_hash(code)}>', 'exec'), mod.__dict__)
    if not callable(getattr(mod, 'signal', None)):
        raise StrategyError('strategy must define signal(df, params, model=None)')
    mod.DESK = getattr(mod, 'DESK', 'l2')
    if mod.DESK not in ('l2', 'carry', 'perps', 'macro', 'pred'):
        raise StrategyError("DESK must be 'l2' (OKX order-book strategies), 'carry' (cross-venue funding), "
                            "'perps' (multi-asset Hyperliquid perps, hourly), 'macro' (cross-asset daily) or "
                            "'pred' (prediction markets, hourly)")
    if mod.DESK in ('perps', 'macro', 'pred'):
        mod.RESOLUTION, mod.PARAMS, mod.LOOKBACK = ('1d' if mod.DESK == 'macro' else '1h'), dict(getattr(mod, 'PARAMS', {})), 0
        mod.FITTED, mod.EXECUTION, mod.TRAIN_DAYS, mod.REFIT_DAYS = False, None, 0, 0
        mod.WARMUP_DAYS = int(getattr(mod, 'WARMUP_DAYS', {'perps': 30, 'macro': 300, 'pred': 0}[mod.DESK]))
        mod.PERPS_EXECUTION = getattr(mod, 'PERPS_EXECUTION', 'taker')
        if mod.PERPS_EXECUTION not in ('maker', 'taker'):
            raise StrategyError("PERPS_EXECUTION must be 'maker' or 'taker'")
        return mod
    if mod.DESK == 'carry':
        mod.RESOLUTION, mod.PARAMS, mod.LOOKBACK = '1h', dict(getattr(mod, 'PARAMS', {})), 0
        mod.FITTED, mod.EXECUTION, mod.TRAIN_DAYS, mod.REFIT_DAYS = False, None, 0, 0
        mod.CARRY_EXECUTION = getattr(mod, 'CARRY_EXECUTION', 'taker')
        if mod.CARRY_EXECUTION not in ('maker', 'taker'):
            raise StrategyError("CARRY_EXECUTION must be 'maker' or 'taker'")
        return mod
    res = getattr(mod, 'RESOLUTION', '1m')
    if res not in RES_SECONDS:
        raise StrategyError('RESOLUTION must be 1s, 10s or 1m')
    mod.RESOLUTION = res
    mod.PARAMS = dict(getattr(mod, 'PARAMS', {}))
    mod.LOOKBACK = int(getattr(mod, 'LOOKBACK', 0))
    mod.FITTED = callable(getattr(mod, 'fit', None))
    mod.TRAIN_DAYS = int(getattr(mod, 'TRAIN_DAYS', 30))
    mod.REFIT_DAYS = int(getattr(mod, 'REFIT_DAYS', 7))
    from .maker import validate_execution
    try:
        mod.EXECUTION = validate_execution(getattr(mod, 'EXECUTION', None))
    except ValueError as e:
        raise StrategyError(str(e))
    return mod


def run_signal(mod, df, params=None, model=None):
    p = dict(mod.PARAMS) if params is None else params
    out = mod.signal(df, p, model) if mod.FITTED else _call(mod.signal, df, p)
    if isinstance(out, pl.Series):
        out = out.to_numpy()
    elif isinstance(out, pl.DataFrame):
        raise StrategyError('signal must return a Series/array, not a DataFrame')
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(out) != len(df):
        raise StrategyError(f'signal length {len(out)} != rows {len(df)}')
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, -1.0, 1.0)


def _call(fn, df, p):
    try:
        return fn(df, p)
    except TypeError:
        return fn(df, p, None)


def causality_test(mod, df, cuts=(0.35, 0.6, 0.85), model=None, seed=11):
    """Prefix invariance + future-perturbation invariance. Returns dict(ok, details)."""
    full = run_signal(mod, df, model=model)
    n = len(df)
    issues = []
    for c in cuts:
        k = int(n * c)
        pre = run_signal(mod, df.head(k), model=model)
        diff = np.abs(pre - full[:k])
        bad = int(np.sum(diff > 1e-6))
        if bad:
            first = int(np.argmax(diff > 1e-6))
            issues.append(f'prefix cut at row {k}: {bad} rows differ (first at {first}); signal uses future rows or full-sample statistics')
    # perturb the future: shuffle numeric columns after k, outputs before k must not move
    k = int(n * 0.5)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.arange(k, n))
    pert = df.with_row_index('_r').with_columns(
        [pl.when(pl.col('_r') >= k).then(pl.col(c).gather(np.r_[np.arange(k), perm]))
         .otherwise(pl.col(c)).alias(c) for c in df.columns if c != 'ts' and df[c].dtype.is_numeric()]).drop('_r')
    out = run_signal(mod, pert, model=model)
    bad = int(np.sum(np.abs(out[:k] - full[:k]) > 1e-6))
    if bad:
        issues.append(f'future perturbation changed {bad} earlier outputs; signal reads future data')
    return {'ok': not issues, 'issues': issues, 'rows': n, 'exposure': float(np.mean(full != 0))}


def _blocks(days, resolution):
    size = {'1s': 5, '10s': 60, '1m': 400}[resolution]
    for i in range(0, len(days), size):
        yield days[i:i + size]


def _prev_day(date):
    from datetime import date as D, timedelta
    return (D.fromisoformat(date) - timedelta(days=1)).isoformat()


def _load_with_warmup(store, block, res):
    """Prepend the calendar-previous day as history when the store can provide it."""
    try:
        return store.load([_prev_day(block[0])] + list(block), res)
    except Exception:
        return store.load(list(block), res)


def evaluate(code, store, days, cost: CostModel | None = None, params=None, fit_days=None, keep_signal=False,
             with_causality=True, variants=None):
    """Run a strategy over `days` (continuous, chunked with one day of warmup history) and backtest it.
    For fitted strategies, walk-forward: models are fit on the TRAIN_DAYS days strictly before each
    REFIT_DAYS block, drawn from `fit_days` (default: the evaluated days themselves + earlier dev days)."""
    mod = load_strategy(code)
    res = mod.RESOLUTION
    cost = cost or CostModel()
    days = sorted(days)
    parts, sigs = [], []
    causality = None
    if mod.FITTED:
        frames, models = _walk_forward(mod, store, days, fit_days, params)
        for df, model in frames:
            sigs.append(run_signal(mod, df, params, model)); parts.append(df)
            if with_causality and causality is None and len(df) > 200:
                causality = causality_test(mod, df, model=model)
    else:
        for block in _blocks(days, res):
            df = _load_with_warmup(store, block, res)
            s = run_signal(mod, df, params)
            keep = (df['ts'] >= _day_ms(block[0])).to_numpy()
            if with_causality and causality is None and len(df) > 200:
                causality = causality_test(mod, df.head(min(len(df), 200_000)))
            df, s = df.filter(pl.Series(keep)), s[keep]
            parts.append(df); sigs.append(s)
    df = pl.concat(parts, how='diagonal_relaxed') if len(parts) > 1 else parts[0]
    sig = np.concatenate(sigs)
    variants = variants or {}
    if mod.EXECUTION:
        runs = {'base': (cost, None), **variants}
        outs = maker_execute(store, df, sig, days, runs, mod.EXECUTION)
        result = outs.pop('base')
        result['variants'] = outs
    else:
        result = backtest(df, sig, res, cost)
        result['variants'] = {name: backtest(df, _transform(sig, tr), res, c) for name, (c, tr) in variants.items()}
    if causality is not None and not causality.get('ok'):
        from .asof import LookAheadError
        raise LookAheadError('l2: look-ahead detected. ' + ' | '.join(causality.get('issues', [])[:3]))
    result['resolution'] = res
    result['causality'] = causality
    result['code_hash'] = code_hash(code)
    if keep_signal:
        result['_df'] = df; result['_signal'] = sig
    return result


def miss_entries(sig, frac=0.25, seed=7):
    """Drop a deterministic random `frac` of position EPISODES (an entry and its whole holding period):
    live IOC orders failed to fill ~24% of the time, precisely in fast markets."""
    out = np.array(sig, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    skipping, prev = False, 0.0
    for i in range(len(out)):
        s = out[i]
        new_episode = s != 0 and (prev == 0 or np.sign(s) != np.sign(prev))
        if new_episode:
            skipping = rng.random() < frac
        if s == 0:
            skipping = False
        prev = s
        if skipping:
            out[i] = 0.0
    return out


def _transform(sig, tr):
    if tr == 'miss25':
        return miss_entries(sig, 0.25)
    if tr == 'long_only':
        return np.clip(sig, 0, 1)
    if tr == 'short_only':
        return np.clip(sig, -1, 0)
    return sig


def maker_execute(store, df, sig, days, runs, ex):
    """Run the queue-aware limit-order engine on the 1 s timeline for each named (cost, transform) run."""
    from .maker import EXEC_COLS, load_tape, run_block, to_1s_target, maker_summary, STATE_SIZE
    state = {k: np.zeros(STATE_SIZE) for k in runs}
    acc = {k: {'ts': [], 'eq': [], 'pos': [], 'mid': [], 'fills': []} for k in runs}
    offset = 0
    sig_frame = df.select('ts')
    for block in [days[i:i + 4] for i in range(0, len(days), 4)]:
        ef = store.load(block, '1s', columns=EXEC_COLS)
        tape = load_tape(store.root, block)
        tgt = to_1s_target(sig_frame, sig, ef)
        ts, mid = ef['ts'].to_numpy(), ef['mid'].to_numpy()
        for name, (c, tr) in runs.items():
            eq, pos, fills, state[name] = run_block(ef, _transform(tgt, tr), tape, c, ex, state[name])
            a = acc[name]
            a['ts'].append(ts); a['eq'].append(eq); a['pos'].append(pos); a['mid'].append(mid)
            if len(fills):
                f = fills.copy(); f[:, 0] += offset; f[:, 4] += offset
                a['fills'].append(f)
        offset += len(ef)
    out = {}
    for name, (c, _) in runs.items():
        a = acc[name]
        fills = np.vstack(a['fills']) if a['fills'] else np.zeros((0, 5))
        out[name] = maker_summary(np.concatenate(a['ts']), np.concatenate(a['eq']), np.concatenate(a['pos']),
                                  np.concatenate(a['mid']), state[name], fills, c)
    return out


def _day_ms(date):
    from ..data.ingest import day_start_ms
    return day_start_ms(date)


def _walk_forward(mod, store, days, fit_days, params):
    """Yields (frame, model) blocks; each model sees only days strictly before its block (1-day embargo)."""
    pool = sorted(set(fit_days or []) | set(days))
    res = mod.RESOLUTION
    out = []
    for b in range(0, len(days), mod.REFIT_DAYS):
        block = days[b:b + mod.REFIT_DAYS]
        first = block[0]
        prior = [d for d in pool if d < first][:-1]          # 1-day embargo
        train = prior[-mod.TRAIN_DAYS:]
        if len(train) < max(3, mod.TRAIN_DAYS // 3):
            continue                                        # not enough history yet: no trading
        model = mod.fit(store.load(train, res), dict(mod.PARAMS) if params is None else params)
        warm = [d for d in pool if d < first][-1:]
        df = store.load(warm + block, res)
        if warm:
            df = df.filter(pl.col('ts') >= _day_ms(first))
        out.append((df, model))
    if not out:
        raise StrategyError('walk-forward produced no tradable block (need more prior days than TRAIN_DAYS/3)')
    return out, None
