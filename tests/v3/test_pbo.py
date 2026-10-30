"""Cycle 1b item 4: PBO (CSCV + CPCV with purge/embargo) as strict validation gates. Synthetic data only."""
import asyncio
import math

import numpy as np

from desk.company.ledger import Ledger
from desk.config import Settings
from desk.lab import stats
from desk.lab.harness import strategy_eval
from desk.lab.strategy import code_hash
from desk.pipeline.validation import Pipeline, check, overfit_job, val_gates
from synth_desk import TREND, job, macro_settings

NOPARAMS = TREND.replace("PARAMS = {'look': 20}", "PARAMS = {}").replace("params['look']", "20")


def _good_report(**over):
    r = {'summary': {'sharpe': 3.0, 'trades': 500, 'max_drawdown': 0.05, 'pnl': 1.0},
         'stats': {'dsr': 0.99, 'bootstrap_lower_mean_bps': 5.0, 'positive_weeks': 0.8, 'top5_day_share': 0.1,
                   'mean_ex_top5_days_bps': 3.0},
         'stress': {'pnl': 1.0}, 'causality': {'ok': True},
         'overfit': {'pbo': 0.2, 'cpcv': {'oos_sharpe_median': 1.5}}}
    r.update(over)
    return r


def test_cpcv_folds_purge_and_embargo():
    T, purge, emb = 120, 4, 3
    folds = list(stats.cpcv_folds(T, 6, 2, purge, emb))
    assert len(folds) == math.comb(6, 2)
    groups = np.array_split(np.arange(T), 6)
    for test, train in folds:
        assert not train[test].any()
        for g in groups:
            if g[0] in test:
                lo, hi = g[0], g[-1]
                assert not train[max(0, lo - purge):lo].any()                 # purged before
                assert not train[hi + 1:min(T, hi + 1 + purge + emb)].any()   # purged + embargoed after


def test_pbo_flags_selection_that_reverses_out_of_sample():
    rng = np.random.default_rng(0)
    T, N = 600, 6
    # config j earns in its own regime block and loses everywhere else: the in-sample winner is an OOS loser
    blocks = np.array_split(np.arange(T), N)
    M = rng.normal(0, 0.01, (T, N)) - 0.002
    for j, b in enumerate(blocks):
        M[b, j] += 0.012
    rep = stats.overfit_report([(list(range(T)), M[:, j]) for j in range(N)])
    assert rep['pbo'] > 0.5 and rep['pbo_cpcv'] > 0.5
    # a configuration that dominates everywhere: PBO ~ 0, positive purged OOS Sharpe
    M2 = rng.normal(0, 0.01, (T, N))
    M2[:, 0] += 0.004
    rep2 = stats.overfit_report([(list(range(T)), M2[:, j]) for j in range(N)])
    assert rep2['pbo'] <= 0.2 and rep2['cpcv']['oos_sharpe_median'] > 0 and rep2['cpcv']['embargo'] == 6


def test_val_gates_reject_pbo_above_half_and_fail_closed():
    s = Settings()
    assert s.max_pbo == 0.50 and s.pbo_fail_closed
    g = val_gates(s)
    assert check(g, _good_report()) == []
    assert 'pbo<=0.50' in check(g, _good_report(overfit={'pbo': 0.51, 'cpcv': {'oos_sharpe_median': 1.0}}))
    assert 'cpcv_oos_sharpe>0' in check(g, _good_report(overfit={'pbo': 0.1, 'cpcv': {'oos_sharpe_median': -0.2}}))
    missing = _good_report()
    missing.pop('overfit')
    assert {'pbo<=0.50', 'cpcv_oos_sharpe>0'} <= set(check(g, missing))       # undefined PBO -> reject (strict)
    nan = _good_report(overfit={'pbo': float('nan'), 'cpcv': {'oos_sharpe_median': float('nan')}})
    assert 'pbo<=0.50' in check(g, nan)
    import dataclasses
    loose = dataclasses.replace(s, pbo_fail_closed=False, max_pbo=0.6)       # configurable, opt-in only
    assert check(val_gates(loose), missing) == []


def test_harness_reports_pbo_over_the_param_grid_and_logs_each_config(tmp_path):
    s = macro_settings(tmp_path)
    out = strategy_eval(job(s, TREND, split='dev', overfit=overfit_job(Settings())))
    ov = out['overfit']
    assert ov['n_configs'] == 3 and len(ov['configs']) == 2                # look=20 plus 16 and 24
    assert math.isfinite(ov['pbo']) and ov['pbo'] == max(ov['pbo_cscv'], ov['pbo_cpcv'])
    assert ov['cpcv']['purge'] == 5 and ov['cpcv']['embargo'] >= 1 and ov['cpcv']['n_paths'] == 15
    assert sum(1 for x in out['trial_log'] if x['variant'].startswith('grid:')) == 2


def test_pipeline_val_stage_applies_pbo_gate(tmp_path):
    s = macro_settings(tmp_path)
    P = Pipeline(s, ledger=Ledger(tmp_path / 'p.sqlite3'))
    seen = []

    async def in_process(j, timeout=0):
        seen.append(j)
        j = dict(j, _home=str(s.home), _store=str(s.store))
        try:
            return strategy_eval(j)
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}
    P._job = in_process
    sid = P.L.add_strategy('noparams', 'a', 'macro', NOPARAMS, code_hash(NOPARAMS), '1d', 't')
    P.L.update_strategy(sid, status='validating')
    asyncio.run(P.process_async(sid))
    st = P.L.strategy(sid)
    assert seen and seen[-1]['split'] == 'val' and seen[-1]['overfit']
    assert st['status'] == 'rejected_val' and 'pbo<=0.50' in st['gate_reasons']   # no grid -> PBO undefined -> reject
