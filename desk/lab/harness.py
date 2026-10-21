"""Trusted evaluation harness (runs as `python -m desk.lab.harness job.json out.json`).

Jobs
  strategy_eval  : evaluate strategy code on explicit days or a split, optional robustness + certification
  factor_eval    : standardized factor report on the benchmark day set + novelty vs. the library
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

from ..config import SETTINGS
from ..data.store import FeatureStore
from .backtest import CostModel, backtest
from .strategy import evaluate, load_strategy
from . import stats


def _days(store, job):
    if job.get('days'):
        return list(job['days'])
    split = job.get('split', 'dev')
    d = store.days(split)
    if job.get('max_days') and len(d) > job['max_days']:
        idx = np.unique(np.linspace(0, len(d) - 1, job['max_days']).astype(int))
        d = [d[i] for i in idx]
    return d


def _brief(r):
    keep = ('days', 'pnl', 'sharpe', 'mean_daily_bps', 'max_drawdown', 'fees', 'funding_paid', 'execution_cost', 'fills',
            'trades', 'turnover_units', 'exposure', 'positive_days', 'pnl_per_turnover_bps')
    return {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in keep if k in r}


def _settings(job):
    import dataclasses
    if job.get('_home'):
        return dataclasses.replace(SETTINGS, home=Path(job['_home']), store_dir=Path(job['_store']))
    return SETTINGS


def carry_eval(job):
    from .carry import CarryStore, CarryCost, evaluate_carry
    cs = CarryStore(_settings(job), allowed=tuple(job.get('allowed', ('dev',))))
    split = job.get('split', 'dev')
    cost = CarryCost(**(job.get('carry_cost') or {}))
    ex = getattr(load_strategy(job['code']), 'CARRY_EXECUTION', None)
    if ex in ('maker', 'taker'):
        cost.execution = ex
    variants = {}
    if job.get('stress'):
        variants['stress'] = (cost.stressed(1.5, 2.0), None)
    if job.get('robustness') and cost.execution == 'maker':
        variants['taker_execution'] = (cost.as_taker(), None)
    if job.get('robustness'):
        variants.update({'fees_x1.5': (cost.stressed(1.5, 1.0), None),
                         'latency_+2s': (cost.stressed(1.0, 2.0), None),
                         'impact_x2': (cost.stressed(1.0, 1.0, 2.0), None),
                         'long_only': (cost, 'long_only'), 'short_only': (cost, 'short_only')})
    t0 = time.time()
    base = evaluate_carry(job['code'], cs, split, cost, variants=variants)
    var = base.pop('variants', {})
    daily = np.array(base['daily']['ret'])
    summ = _brief(base)
    summ.update({k: round(base[k], 2) for k in ('funding_pnl', 'basis_pnl', 'avg_gross', 'coins_traded')})
    out = {'summary': summ, 'causality': base['causality'], 'code_hash': base['code_hash'], 'execution': 'carry',
           'desk': 'carry', 'resolution': '1h', 'evaluated_days': base['days'], 'split': split,
           'certification': {'ok': True, 'note': 'carry engine certified by tests/v3/test_carry.py reference parity'}}
    out['stats'] = {'psr': round(stats.psr(daily), 4),
                    'bootstrap_lower_mean_bps': round(stats.block_bootstrap_lower(daily) * 1e4, 4) if len(daily) >= 5 else None,
                    'positive_weeks': round(stats.weekly_positive_fraction(base['daily']['day'], daily), 3),
                    'top5_day_share': round(stats.top_day_share(daily), 3),
                    'mean_ex_top5_days_bps': round(stats.mean_ex_top(daily) * 1e4, 3)}
    if job.get('n_trials'):
        out['stats']['dsr'] = round(stats.dsr(daily, job['n_trials'], job.get('trial_sr_std')), 4)
        out['stats']['n_trials'] = job['n_trials']
    if job.get('return_daily'):
        out['daily'] = base['daily']
    if job.get('stress'):
        out['stress'] = _brief(var['stress'])
    if job.get('robustness'):
        out['robustness'] = robustness(job['code'], None, None, cost, base, var)
        mod = load_strategy(job['code'])
        numeric = [(k, v) for k, v in mod.PARAMS.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0][:4]
        nb = []
        for k, v in numeric:
            for f in (0.8, 1.2):
                pp = dict(mod.PARAMS)
                pp[k] = type(v)(v * f) if isinstance(v, float) else max(1, int(round(v * f)))
                if pp[k] == v:
                    continue
                r = evaluate_carry(job['code'], cs, split, cost, params=pp, with_causality=False)
                nb.append({'param': k, 'value': pp[k], 'pnl': round(r['pnl'], 1), 'sharpe': round(r['sharpe'], 3)})
        if nb:
            out['robustness']['param_neighborhood'] = {
                'variants': nb, 'fraction_profitable': round(float(np.mean([x['pnl'] > 0 for x in nb])), 3),
                'median_sharpe': round(float(np.median([x['sharpe'] for x in nb])), 3)}
    out['seconds'] = round(time.time() - t0, 1)
    return out


def _desk_kit(desk, job):
    """(store, cost, evaluate, stress(cost), robustness variants, sharpe periods) for panel desks."""
    if desk == 'perps':
        from .perps import PerpsStore, PerpsCost, evaluate_perps
        st = PerpsStore(_settings(job), allowed=tuple(job.get('allowed', ('dev',))))
        cost = PerpsCost(**(job.get('perps_cost') or {}))
        rob = {'fees_x1.5': (cost.stressed(1.5, 1.0), None), 'latency_+2s': (cost.stressed(1.0, 1.0, 1), None),
               'impact_x2': (cost.stressed(1.0, 2.0), None), 'long_only': (cost, 'long_only'), 'short_only': (cost, 'short_only')}
        if getattr(load_strategy(job['code']), 'PERPS_EXECUTION', 'taker') == 'maker':
            from dataclasses import replace
            mk = replace(cost, execution='maker')
            rob.update({'taker_execution': (mk.as_taker(), None),
                        'queue_x2': (replace(mk, through_bps=mk.through_bps * 2 + 2), None)})
        return st, cost, evaluate_perps, cost.stressed(1.5, 2.0), rob, 365
    if desk == 'pred':
        from .pred import PredStore, PredCost, evaluate_pred
        st = PredStore(_settings(job), allowed=tuple(job.get('allowed', ('dev',))))
        cost = PredCost(**(job.get('pred_cost') or {}))
        rob = {'fees_x1.5': (cost.stressed(1.5), None), 'latency_+2s': (cost.stressed(1.0, 1), None),
               'impact_x2': (cost.stressed(2.0), None), 'long_only': (cost, 'long_only'), 'short_only': (cost, 'short_only')}
        return st, cost, evaluate_pred, cost.stressed(2.0, 0), rob, 365
    from .macro import MacroStore, MacroCost, evaluate_macro
    st = MacroStore(_settings(job), allowed=tuple(job.get('allowed', ('dev',))))
    cost = MacroCost(**(job.get('macro_cost') or {}))
    rob = {'fees_x1.5': (cost.stressed(1.5), None), 'latency_+2s': (cost.stressed(1.0, 1), None),
           'impact_x2': (cost.stressed(2.0), None), 'long_only': (cost, 'long_only'), 'short_only': (cost, 'short_only')}
    return st, cost, evaluate_macro, cost.stressed(2.0, 0), rob, 252


def panel_desk_eval(job, desk):
    st, cost, evaluate_fn, stress_cost, rob_variants, periods = _desk_kit(desk, job)
    split = job.get('split', 'dev')
    variants = {}
    if job.get('stress'):
        variants['stress'] = (stress_cost, None)
    if job.get('robustness'):
        variants.update(rob_variants)
    t0 = time.time()
    base = evaluate_fn(job['code'], st, split, cost, variants=variants)
    var = base.pop('variants', {})
    daily = np.array(base['daily']['ret'])
    summ = _brief(base)
    extra = [k for k in ('price_pnl', 'funding_pnl', 'avg_gross', 'net_exposure', 'coins_traded', 'assets_traded', 'markets_traded', 'ann_return',
                         'ann_vol', 'median_daily_bps', 'mean_ex_top5_days_bps') if k in base]
    summ.update({k: round(base[k], 4) for k in extra})
    out = {'summary': summ, 'causality': base['causality'], 'code_hash': base['code_hash'], 'execution': desk,
           'desk': desk, 'resolution': '1d' if desk == 'macro' else '1h', 'evaluated_days': base['days'], 'split': split,
           'certification': {'ok': True, 'note': f'{desk} engine certified by tests/v3/test_{desk}.py reference parity'}}
    out['stats'] = {'psr': round(stats.psr(daily), 4),
                    'bootstrap_lower_mean_bps': round(stats.block_bootstrap_lower(daily) * 1e4, 4) if len(daily) >= 5 else None,
                    'positive_weeks': round(stats.weekly_positive_fraction(base['daily']['day'], daily), 3),
                    'top5_day_share': round(stats.top_day_share(daily), 3),
                    'mean_ex_top5_days_bps': round(base['mean_ex_top5_days_bps'], 3),
                    'median_daily_bps': round(base['median_daily_bps'], 3)}
    if job.get('n_trials'):
        out['stats']['dsr'] = round(stats.dsr(daily, job['n_trials'], job.get('trial_sr_std')), 4)
        out['stats']['n_trials'] = job['n_trials']
    if job.get('return_daily'):
        out['daily'] = base['daily']
    if job.get('stress'):
        out['stress'] = _brief(var['stress'])
    if job.get('robustness'):
        out['robustness'] = robustness(job['code'], None, None, cost, base, var)
        mod = load_strategy(job['code'])
        numeric = [(k, v) for k, v in mod.PARAMS.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0][:4]
        nb = []
        for k, v in numeric:
            for f in (0.8, 1.2):
                pp = dict(mod.PARAMS)
                pp[k] = type(v)(v * f) if isinstance(v, float) else max(1, int(round(v * f)))
                if pp[k] == v:
                    continue
                r = evaluate_fn(job['code'], st, split, cost, params=pp, with_causality=False)
                nb.append({'param': k, 'value': pp[k], 'pnl': round(r['pnl'], 1), 'sharpe': round(r['sharpe'], 3)})
        if nb:
            out['robustness']['param_neighborhood'] = {
                'variants': nb, 'fraction_profitable': round(float(np.mean([x['pnl'] > 0 for x in nb])), 3),
                'median_sharpe': round(float(np.median([x['sharpe'] for x in nb])), 3)}
    out['seconds'] = round(time.time() - t0, 1)
    return out


def perps_eval(job):
    return panel_desk_eval(job, 'perps')


def strategy_eval(job):
    if load_strategy(job['code']).DESK in ('perps', 'macro', 'pred'):
        return panel_desk_eval(job, load_strategy(job['code']).DESK)
    if load_strategy(job['code']).DESK == 'carry':
        return carry_eval(job)
    store = FeatureStore(_settings(job), allowed=tuple(job.get('allowed', ('dev',))))
    cost = CostModel(**job.get('cost', {}))
    days = _days(store, job)
    t0 = time.time()
    fit_pool = [d for sp in job.get('allowed', ('dev',)) for d in store.days(sp)]   # fitted models: any EARLIER day
    mod0 = load_strategy(job['code'])
    from .backtest import RES_SECONDS
    lat = 2 if mod0.RESOLUTION == '1s' or mod0.EXECUTION else RES_SECONDS[mod0.RESOLUTION]
    variants = {}
    if job.get('stress'):
        variants['stress'] = (cost.stressed(1.5), None)
    if job.get('robustness'):
        variants.update({'fees_x1.5': (cost.stressed(1.5), None), 'latency_+2s': (cost.stressed(1.0, lat), None),
                         'impact_x2': (cost.stressed(1.0, 0, cost.impact_mult * 2), None),
                         'long_only': (cost, 'long_only'), 'short_only': (cost, 'short_only'),
                         'missed_entries_25%': (cost, 'miss25')})
        if mod0.EXECUTION:
            variants['queue_x2'] = (cost.stressed(1.0, 0, None, cost.queue_mult * 2), None)
    base = evaluate(job['code'], store, days, cost, keep_signal=True, fit_days=fit_pool, variants=variants)
    var = base.pop('variants', {})
    df, sig = base.pop('_df'), base.pop('_signal')
    res = base['resolution']
    daily = np.array(base['daily']['ret'])
    out = {'summary': _brief(base), 'causality': base.get('causality'), 'code_hash': base['code_hash'],
           'execution': base.get('execution', 'taker'),
           'maker_stats': {k: base.get(k) for k in ('maker_fill_fraction', 'maker_markout_60s_bps', 'median_seconds_to_fill')}
           if base.get('execution') == 'maker' else None,
           'resolution': res, 'evaluated_days': len(days), 'first_day': days[0], 'last_day': days[-1]}
    out['stats'] = {
        'psr': round(stats.psr(daily), 4),
        'bootstrap_lower_mean_bps': round(stats.block_bootstrap_lower(daily) * 1e4, 4) if len(daily) >= 5 else None,
        'positive_weeks': round(stats.weekly_positive_fraction(base['daily']['day'], daily), 3),
        'top5_day_share': round(stats.top_day_share(daily), 3),
        'mean_ex_top5_days_bps': round(stats.mean_ex_top(daily) * 1e4, 3),
    }
    if job.get('n_trials'):
        out['stats']['dsr'] = round(stats.dsr(daily, job['n_trials'], job.get('trial_sr_std')), 4)
        out['stats']['n_trials'] = job['n_trials']
    if job.get('return_daily'):
        out['daily'] = base['daily']
    if job.get('stress'):
        out['stress'] = _brief(var['stress'])
    if job.get('robustness'):
        out['robustness'] = robustness(job['code'], store, days, cost, base, var)
    if job.get('certify') and out['execution'] == 'maker':
        out['certification'] = {'ok': True, 'note': 'maker engine is certified by tests/v3 reference parity (queue model)'}
    elif job.get('certify'):
        from .certify import certify
        k = min(len(df), {'1s': 2 * 86400, '10s': 4 * 8640, '1m': 10 * 1440}[res])
        sub, s2 = df.head(k), sig[:k]
        er = backtest(sub, s2, res, cost)
        out['certification'] = certify(sub, s2, res, cost, er)
    out['seconds'] = round(time.time() - t0, 1)
    return out


def robustness(code, store, days, cost, base, var):
    rob = {k: _brief(v) for k, v in var.items() if k != 'stress'}
    daily = np.array(base['daily']['ret']); dd = np.array(base['daily']['day'])
    h = len(daily) // 2
    rob['first_half_sharpe'] = round(stats.sharpe(daily[:h]), 3)
    rob['second_half_sharpe'] = round(stats.sharpe(daily[h:]), 3)
    months = {}
    for d, r in zip(dd, daily):
        m = time.strftime('%Y-%m', time.gmtime(int(d) * 86400))
        months.setdefault(m, []).append(r)
    rob['monthly_bps'] = {m: round(float(np.sum(v)) * 1e4, 1) for m, v in months.items()}
    active = [np.sum(v) for v in months.values() if np.any(np.asarray(v) != 0)]   # flat months carry no information
    rob['positive_months'] = round(float(np.mean([x > 0 for x in active])), 3) if active else 0.0
    # parameter neighborhood on a subsample of days
    mod = load_strategy(code)
    numeric = [(k, v) for k, v in mod.PARAMS.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0][:4]
    if numeric and store is not None:
        sub = days[::3] if len(days) > 30 else days
        ref = evaluate(code, store, sub, cost, with_causality=False)
        nb = []
        for k, v in numeric:
            for f in (0.8, 1.2):
                p = dict(mod.PARAMS)
                p[k] = type(v)(v * f) if isinstance(v, float) else max(1, int(round(v * f)))
                if p[k] == v:
                    continue
                try:
                    r = evaluate(code, store, sub, cost, params=p, with_causality=False)
                    nb.append({'param': k, 'value': p[k], 'pnl': round(r['pnl'], 1), 'sharpe': round(r['sharpe'], 3)})
                except Exception as e:
                    nb.append({'param': k, 'value': p[k], 'error': repr(e)[:200]})
        ok = [x for x in nb if 'pnl' in x]
        rob['param_neighborhood'] = {'reference_sharpe': round(ref['sharpe'], 3), 'variants': nb,
                                     'fraction_profitable': round(float(np.mean([x['pnl'] > 0 for x in ok])), 3) if ok else None,
                                     'median_sharpe': round(float(np.median([x['sharpe'] for x in ok])), 3) if ok else None}
    return rob


def panel_factor_eval(job):
    from .panel_factor import evaluate
    desk = job['desk']
    s = _settings(job)
    if desk == 'perps':
        from .perps import PerpsStore
        panel, a, b = PerpsStore(s, allowed=('dev',)).load('dev', warmup_days=0)
    elif desk == 'pred':
        from .pred import PredStore, visible
        panel, a, b = PredStore(s, allowed=('dev',)).load('dev')
        panel = visible(panel)
    else:
        from .macro import MacroStore
        panel, a, b = MacroStore(s, allowed=('dev',)).load('dev', warmup_days=0)
    vdir = Path(job['values_dir'])
    lib = {}
    for fid in job.get('library_ids', []):
        f = vdir / f'{fid}.parquet'
        if f.exists():
            lib[fid] = pl.read_parquet(f)
    out = evaluate(job['code'], panel, desk, lib)
    vals = out.pop('_values')
    if job.get('factor_id'):
        vdir.mkdir(parents=True, exist_ok=True)
        vals.write_parquet(vdir / f"{job['factor_id']}.parquet", compression='zstd')
    return out


def factor_eval(job):
    if job.get('desk') in ('perps', 'macro', 'pred'):
        return panel_factor_eval(job)
    from .factors import evaluate_factor, novelty
    store = FeatureStore(_settings(job), allowed=('dev',))
    res = job.get('resolution', '1m')
    # benchmark days are pinned to the FROZEN dev list (stable across ingestion progress -> comparable novelty)
    frozen = store.days('dev', ingested_only=False)
    k = {'1s': 10, '10s': 25, '1m': 60}[res]
    picks = [frozen[i] for i in np.unique(np.linspace(0, len(frozen) - 1, k).astype(int))]
    days = [d for d in picks if (store.root / res / f'{d}.parquet').exists()]
    if len(days) < len(picks):
        raise RuntimeError(f'benchmark days not fully ingested yet ({len(days)}/{len(picks)})')
    df = store.load(days, res)
    ns = {'np': np, 'pl': pl}
    exec(job['code'], ns)
    params = dict(ns.get('PARAMS', {}))
    vals = ns['factor'](df, params)
    vals = np.asarray(vals.to_numpy() if isinstance(vals, pl.Series) else vals, dtype=float)
    if len(vals) != len(df):
        raise ValueError(f'factor returned {len(vals)} values for {len(df)} rows')
    # causality on a prefix
    k = len(df) // 2
    pre = ns['factor'](df.head(k), params)
    pre = np.asarray(pre.to_numpy() if isinstance(pre, pl.Series) else pre, dtype=float)
    a, b = pre, vals[:k]
    bad = int(np.sum(~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-6 * (1 + np.abs(b))))))
    report = evaluate_factor(df, vals, res)
    report['causal'] = bad == 0
    if bad:
        report['causality_issue'] = f'{bad} values changed when future rows were removed'
    cache = Path(job['values_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{job['factor_id']}_{res}.npy"
    lib = {}
    for p in cache.glob(f'*_{res}.npy'):
        if p.name != key:
            v = np.load(p)
            if len(v) == len(vals):
                lib[p.name.rsplit('_', 1)[0]] = v
    report['novelty'] = novelty(vals, lib)
    np.save(cache / key, vals.astype(np.float32))
    report['benchmark_days'] = len(days)
    return report


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    if not job.get('_ticket'):                                # charter R7b: only run_job (which redeems it) launches us
        Path(sys.argv[2]).write_text(json.dumps({'error': 'ticket_refused: job has no ledger trial ticket'}))
        return
    try:
        out = {'strategy_eval': strategy_eval, 'factor_eval': factor_eval}[job['type']](job)
    except Exception as e:
        out = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-3000:]}
    Path(sys.argv[2]).write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    main()
