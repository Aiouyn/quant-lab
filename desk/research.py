"""Research API available inside agent sandboxes:  `from desk.research import *`

Only DEVELOPMENT days are reachable here (the sandbox store root is a dev-only view).

    days()                                   -> list of dev dates 'YYYY-MM-DD' (chronological)
    sample_days(n, seed=0)                   -> n dev dates spread across the period (NOT contiguous: gaps between days)
    contiguous_days(n, start=None)           -> n consecutive dev dates (use for rolling features / forward returns)
    load(days, resolution='1m', columns=None)-> polars DataFrame (ts ms, mid, book/trade features, funding_rate)
    fwd_returns(df, horizons_s, resolution)  -> dict horizon -> np.ndarray of forward bps returns (labels; research only)
    factor_report(df, values, resolution)    -> standardized IC/decay/regime/cost report
    backtest(df, target, resolution, **cost) -> net PnL report (HL fees, spread, depth impact, latency, funding)
    run_strategy(code, days)                 -> evaluate a strategy module string over days (walk-forward if fitted)
    causality(code, days)                    -> prefix/future-perturbation test of a strategy module
    library_factors() / factor_code(id)      -> the shared Factor Library (read-only)
    carry_panel(coins=None)                  -> CARRY DESK: hourly Hyperliquid-vs-Binance panel (dev period): ts, coin,
                                                hl_funding, hl_premium, bn_funding (paid this hour), bn_rate_last,
                                                bn_interval_h, bn_premium, bn_close, bn_qvol
    carry_backtest(panel, positions, **cost) -> carry PnL (funding + basis - fees on both legs)
    perps_panel(coins=None)                  -> PERPS DESK (main research ground): hourly panel of EVERY Hyperliquid perp
                                                (delisted included, ~230 coins, dev period 2023-06..2025-05), one row per
                                                (ts, coin), only data known at ts: px (HL mid at h:00), px_fill (mid at
                                                h:01 = execution price), oracle, mark, premium, oi, vlm (24h notional),
                                                isp (impact spread bps), f (funding PAID at ts), bn_oi_value,
                                                bn_ls_accounts, bn_ls_top_positions, bn_taker_ratio (Binance positioning,
                                                availability-lagged; null for coins not on Binance)
    perps_backtest(panel, weights, **cost)   -> PnL of hourly target weights in [-1,1] per row (gross capped at 1,
                                                held over the next hour, HL taker 4.5 bps + half impact spread, funding).
                                                Reports mean AND median daily bps and mean EXCLUDING the 5 best days:
                                                lumpy payoffs (crash rebounds, pump shorts) look good on the mean only.
                                                execution='maker' simulates limit orders at the h:01 price filled only
                                                if the next hour trades through them by 2 bps (1.5 bps fee, unfilled =
                                                retried next hour); strategies opt in with PERPS_EXECUTION = 'maker'.
    macro_panel(symbols=None)                -> MACRO DESK: daily cross-asset panel (dev 2000..2014; ~119 symbols:
                                                US/intl equity ETFs, sectors, countries, factor ETFs, bonds, commodities,
                                                REITs, currency ETFs, FX spot, BTC/ETH; plus FEATURE series with
                                                tradable=0: ^GSPC ^NDX ^RUT ^N225 ^GDAXI ^FTSE ^HSI ^STOXX50E, ^VIX ^VIX3M
                                                ^VVIX ^MOVE ^OVX ^GVZ, yields ^IRX ^FVX ^TNX ^TYX, DX-Y.NYB, continuous
                                                futures GC=F CL=F HG=F ZN=F ...). Columns: date, symbol, open, high, low,
                                                close, adjclose (total return), adj_open, volume, cls, tradable, cost_bps
    macro_backtest(panel, weights, **cost)   -> daily PnL: weights decided at close t execute at close t+1 (no
                                                time-zone leaks), per-asset costs, 1%/yr borrow on shorts, gross <= 2.
                                                Reports Sharpe (252), ann return/vol, median, mean ex top-5 days.
    tape(days)                               -> OKX trade tape: per second (ts), price tick (px, x0.1 = USD), aggressor
                                                side (+1 buy lifts asks / -1 sell hits bids), size (contracts). Use it to
                                                study fill probability, queue depletion and adverse selection.
    run_strategy(code) also handles EXECUTION={'mode':'maker',...} limit-order strategies (queue-aware fills).
    pred_panel() / pred_backtest(panel, weights) -> PRED DESK: hourly YES prices of resolved Polymarket binary markets
                                                (dev to 2026-02; columns ts, market, p, end_ms, category, hours_to_end);
                                                weight +w = hold YES, -w = hold NO; 1-cent half-spread; pays 1/0 at end
    event_study(panel, mask, horizons, side) -> event returns: mean, MEDIAN, mean excluding the 5 best, win rate, t
    xs_factor_report(panel, values)          -> cross-sectional IC + mean quintile spreads + cost hurdle (no registration)
    walk_forward(panel, features, (target, h), fit, predict) -> causal out-of-sample model predictions
    example_strategy(desk) / source(module)  -> a WORKING strategy template per desk; source code of desk modules
    FEATURES                                 -> column dictionary

Columns (1s store; bars keep last state + summed flows, plus open/high/low, plus within-bar MEANS
  <col>_avg for qi1, qi5, qi10, micro_dev_bps, spread_bps, bdep_10, adep_10, bdep_50, adep_50, bimp_10, aimp_10 -
  prefer the _avg versions for state signals on bars: the last-second snapshot is very noisy):
  bid_px, ask_px, mid, spread_bps, micro_dev_bps, bid_sz0/ask_sz0 (contracts, 1 ct = 0.01 BTC), bid_n0/ask_n0 (orders),
  qi1/qi5/qi10 (queue imbalance by level rank), bdep_D/adep_D (contracts within D bps of mid, D in 2,5,10,25,50,100),
  bcnt_10/acnt_10/bcnt_50/acnt_50 (order counts), ofi1/ofi5/ofi10 (multi-level order-flow imbalance, contracts/second),
  bwall_sz/bwall_bps/awall_sz/awall_bps (largest level within 50 bps and its distance), bimp_Q/aimp_Q (bps from mid
  to sweep Q BTC, Q in 1,10,50), buy_sz/sell_sz/buy_n/sell_n/max_trade/vwap/last_px (OKX trade prints per second,
  aggressor side), funding_rate (last paid Hyperliquid hourly funding), valid (1 = real observation).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from .lab.backtest import CostModel, backtest as _backtest
from .lab.factors import evaluate_factor, forward_returns
from .lab import strategy as _strategy

FEATURES = __doc__.split('Columns')[1]


class _Settings:
    def __init__(self, root):
        self.store = Path(root)
        self.data_dir = Path(root)
        self.dev_fraction, self.val_fraction = 1.0, 0.0


def _store():
    from .data.store import FeatureStore
    root = os.environ.get('DESK_STORE_ROOT')
    if root:
        return FeatureStore(_Settings(root), root=Path(root), allowed=('dev',))
    from .config import SETTINGS
    return FeatureStore(SETTINGS, allowed=('dev',))


STORE = _store()


def days():
    return STORE.days('dev')


def sample_days(n=20, seed=0):
    d = days()
    if n >= len(d):
        return d
    idx = np.unique(np.linspace(0, len(d) - 1, n).astype(int))
    if seed:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(d), n, replace=False))
    return [d[i] for i in idx]


def contiguous_days(n=20, start=None):
    """n consecutive dev days (rolling windows and forward returns never span calendar gaps).
    start: index into days() or a date string; default = a block from the middle of the dev period."""
    d = days()
    if start is None:
        i = max(0, len(d) // 2 - n // 2)
    elif isinstance(start, str):
        i = d.index(start)
    else:
        i = int(start)
    return d[i:i + n]


def load(days_, resolution='1m', columns=None):
    return STORE.load(days_, resolution, columns)


def fwd_returns(df, horizons_s=(60, 300, 900, 3600), resolution='1m'):
    return forward_returns(df, horizons_s, resolution)


def factor_report(df, values, resolution='1m', horizons_s=(10, 60, 300, 900, 3600, 14400)):
    return evaluate_factor(df, values, resolution, horizons_s)


def _log_trial(kind, sharpe, days_, ident):
    """Self-report only (charter R7c): inside the sandbox this sends an INFORMATIONAL record over the host pipe
    (DESK_TRIALS_FD); the host logs it but never counts it or uses its metrics. The counted trial is written host-side,
    one per run_python call, by the tool layer."""
    fd = os.environ.get('DESK_TRIALS_FD')
    if fd:
        try:
            os.write(int(fd), (json.dumps({'kind': kind, 'sharpe': float(sharpe), 'days': int(days_), 'id': ident})
                               + chr(10)).encode('utf-8'))
        except (OSError, ValueError, TypeError):
            pass
    return None


def backtest(df, target, resolution='1m', **cost):
    r = _backtest(df, target, resolution, CostModel(**cost))
    r.pop('daily', None)
    import hashlib
    _log_trial('dev_backtest', r['sharpe'], r['days'], hashlib.sha1(np.asarray(target, float).tobytes()).hexdigest()[:16])
    return r


def run_strategy(code, days_=None, **cost):
    r = _strategy.evaluate(code, STORE, days_ or sample_days(30), CostModel(**cost))
    r.pop('daily', None)
    _log_trial('dev_backtest', r['sharpe'], r['days'], r['code_hash'])
    return r


def causality(code, days_=None):
    mod = _strategy.load_strategy(code)
    df = STORE.load((days_ or sample_days(3)), mod.RESOLUTION)
    return _strategy.causality_test(mod, df)


def tape(days_):
    from .lab.maker import load_tape
    if isinstance(days_, str):
        days_ = [days_]
    allowed = set(days())
    bad = [d for d in days_ if d not in allowed]
    if bad:
        raise PermissionError(f'tape is only available for dev days: {bad[:3]}')
    t = load_tape(STORE.root, days_)             # the sandbox root is the dev-only view (hard links)
    if t is None:
        raise FileNotFoundError('tape not built for these days')
    return t


def carry_panel(coins=None, warmup_days=0):
    """Hourly cross-venue panel (Hyperliquid vs Binance) for the carry desk's DEV period."""
    from .lab.carry import CarryStore
    import types
    root = os.environ.get('DESK_CARRY_ROOT')
    s = types.SimpleNamespace(store=Path(root).parent, ensure=lambda: None) if root else _carry_settings()
    if not (s.store / 'carry' / 'splits.json').exists():
        from .data.store_builder import build_perps_and_carry_stores
        build_perps_and_carry_stores(s, try_remote=False)
    cs = CarryStore(s, allowed=('dev',))
    p, a, b = cs.load('dev', warmup_days=warmup_days, coins=coins)
    return p


def _carry_settings():
    from .config import SETTINGS
    return SETTINGS


def carry_backtest(panel, positions, **cost):
    """Backtest carry positions (aligned to panel rows; +1 = long HL / short Binance). Counts as a trial."""
    from .lab.carry import CarryCost, run_carry, summarize_carry
    c = CarryCost(**cost)
    r = summarize_carry(run_carry(panel, positions, c), c)
    r.pop('daily', None)
    import hashlib
    _log_trial('dev_backtest', r['sharpe'], r['days'], hashlib.sha1(np.asarray(positions, float).tobytes()).hexdigest()[:16])
    return r


def perps_panel(coins=None, warmup_days=0):
    """Hourly multi-asset Hyperliquid panel for the PERPS desk's DEV period (all perps, delisted included)."""
    path = os.environ.get('DESK_PERPS_DEV')
    if path and Path(path).exists():
        p = pl.read_parquet(path)
    else:
        s = _carry_settings()
        if not (s.store / 'perps' / 'panel.parquet').exists():
            from .data.store_builder import build_perps_and_carry_stores
            build_perps_and_carry_stores(s, try_remote=False)
        from .lab.perps import PerpsStore, visible
        p, a, b = PerpsStore(s, allowed=('dev',)).load('dev', warmup_days=0)
        p = visible(p)
    if coins:
        p = p.filter(pl.col('coin').is_in(list(coins)))
    return p.sort(['ts', 'coin'])


def perps_backtest(panel, weights, **cost):
    """Backtest hourly target weights aligned to panel rows (sorted by ts, coin). Counts as a trial."""
    from .lab.perps import PerpsCost, run_perps, summarize_perps
    c = PerpsCost(**cost)
    panel = panel.sort(['ts', 'coin'])
    if c.execution == 'maker' and 'lo' not in panel.columns:          # engine-only fill ranges (never in signals)
        fp = os.environ.get('DESK_PERPS_DEV_FILLS')
        if fp and Path(fp).exists():
            fills = pl.read_parquet(fp)
        else:
            from .lab.perps import PerpsStore
            fills = PerpsStore(_carry_settings(), allowed=('dev',)).load('dev', warmup_days=0)[0].select('ts', 'coin', 'lo', 'hi')
        panel = panel.join(fills, on=['ts', 'coin'], how='left').sort(['ts', 'coin'])
    r = summarize_perps(run_perps(panel, weights, c), c)
    r.pop('daily', None)
    import hashlib
    _log_trial('dev_backtest', r['sharpe'], r['days'], hashlib.sha1(np.asarray(weights, float).tobytes()).hexdigest()[:16])
    return r


def macro_panel(symbols=None, warmup_days=0):
    """Daily cross-asset panel for the MACRO desk's DEV period (2000..2014)."""
    path = os.environ.get('DESK_MACRO_DEV')
    if path and Path(path).exists():
        p = pl.read_parquet(path)
    else:
        s = _carry_settings()
        if not (s.store / 'macro' / 'panel.parquet').exists():
            from .data.store_builder import build_macro_store
            build_macro_store(s, try_remote=False)
        from .lab.macro import MacroStore
        p, a, b = MacroStore(s, allowed=('dev',)).load('dev', warmup_days=0)
    if symbols:
        p = p.filter(pl.col('symbol').is_in(list(symbols)))
    return p.sort(['date', 'symbol'])


def macro_backtest(panel, weights, **cost):
    """Backtest daily target weights aligned to panel rows (sorted by date, symbol). Counts as a trial."""
    from .lab.macro import MacroCost, run_macro, summarize_macro
    c = MacroCost(**cost)
    r = summarize_macro(run_macro(panel.sort(['date', 'symbol']), weights, c), c)
    r.pop('daily', None)
    import hashlib
    _log_trial('dev_backtest', r['sharpe'], r['days'], hashlib.sha1(np.asarray(weights, float).tobytes()).hexdigest()[:16])
    return r


from .research_tools import event_study, xs_factor_report, walk_forward, example_strategy, example_factor, source  # noqa: E402


def pred_panel(warmup_days=0):
    """Hourly prediction-market panel for the PRED desk DEV period (resolved markets only; no outcome column)."""
    path = os.environ.get('DESK_PRED_DEV')
    if path and Path(path).exists():
        # the view is written sorted by (ts, market); re-sorting would double peak memory (~2 GB) for nothing
        return pl.read_parquet(path).with_columns(pl.col('hours_to_end').cast(pl.Float32)).set_sorted('ts')
    s = _carry_settings()
    if not (s.store / 'pred' / 'panel.parquet').exists():
        from .data.store_builder import build_pred_store
        build_pred_store(s)
    from .lab.pred import PredStore, visible
    p, a, b = PredStore(s, allowed=('dev',)).load('dev')
    return visible(p).sort(['ts', 'market'])


def pred_backtest(panel, weights, **cost):
    """Backtest hourly YES(+)/NO(-) weights aligned to panel rows (sorted by ts, market). Counts as a trial."""
    from .lab.pred import PredCost, run_pred, summarize_pred
    c = PredCost(**cost)
    weights = np.asarray(weights, float).reshape(-1)
    if len(weights) != len(panel):
        raise ValueError(f'weights length {len(weights)} != panel rows {len(panel)}')
    key = panel.select('ts', 'market')
    if not key.select(((pl.col('ts').diff() > 0) | ((pl.col('ts').diff() == 0) & (pl.col('market').diff() > 0))
                       | pl.col('ts').diff().is_null()).all()).item():
        order = key.with_row_index('_i').sort(['ts', 'market'])['_i'].to_numpy()   # reorder rows AND weights together
        panel, weights = panel[order], weights[order]
    if 'outcome' not in panel.columns:                               # engine-only resolution outcomes (one per market)
        fp = os.environ.get('DESK_PRED_DEV_OUTCOMES')
        src = pl.scan_parquet(fp) if fp and Path(fp).exists() else None
        if src is None:
            from .lab.pred import PredStore
            src = PredStore(_carry_settings(), allowed=('dev',)).load('dev')[0].lazy()
        oc = src.group_by('market').agg(pl.col('outcome').first()).collect()
        panel = panel.join(oc, on='market', how='left', maintain_order='left')
    r = summarize_pred(run_pred(panel, weights, c), c)
    r.pop('daily', None)
    import hashlib
    _log_trial('dev_backtest', r['sharpe'], r['days'], hashlib.sha1(np.asarray(weights, float).tobytes()).hexdigest()[:16])
    return r


def research_agenda():
    """The lab director's agenda: where edges can persist at our speed, and how to test each idea here."""
    return (Path(__file__).resolve().parents[1] / 'RESEARCH_AGENDA.md').read_text(encoding='utf-8')


def library_factors():
    p = os.environ.get('DESK_LIBRARY')
    if not p or not Path(p).exists():
        return []
    return json.loads(Path(p).read_text())


def factor_code(fid):
    for f in library_factors():
        if f['id'] == fid:
            return f['code']
    raise KeyError(fid)


def factor_values(fid, df):
    """Compute a library factor on df. Factor modules define factor(df, params) -> Series and PARAMS."""
    ns = {'np': np, 'pl': pl}
    exec(factor_code(fid), ns)
    out = ns['factor'](df, dict(ns.get('PARAMS', {})))
    return np.asarray(out.to_numpy() if isinstance(out, pl.Series) else out, dtype=float)


__all__ = ['days', 'sample_days', 'contiguous_days', 'tape', 'carry_panel', 'carry_backtest', 'perps_panel', 'perps_backtest', 'macro_panel', 'macro_backtest', 'event_study', 'xs_factor_report', 'walk_forward', 'pred_panel', 'pred_backtest', 'research_agenda', 'example_strategy', 'example_factor', 'source', 'load', 'fwd_returns', 'factor_report', 'backtest', 'run_strategy', 'causality',
           'library_factors', 'factor_code', 'factor_values', 'FEATURES', 'np', 'pl', 'CostModel']
