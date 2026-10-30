"""Cycle 1b item 3: structural look-ahead guard + planted leaky canary. Synthetic data only."""
import asyncio

import numpy as np
import pytest

from desk.company.ledger import Ledger
from desk.lab import asof
from desk.lab.asof import LookAheadError, asof_guard
from desk.lab.canary import CANARY_MACRO, HarnessCompromised, canary_selfcheck, synthetic_macro_panel
from desk.lab.harness import strategy_eval
from desk.lab.macro import MacroCost, MacroStore, evaluate_macro, macro_causality, macro_signal
from desk.lab.strategy import code_hash, load_strategy
from synth_desk import TREND, job, macro_settings

# leaks only after 2002-01-01 (last ~15% of the 300-day panel): invisible to the old fixed 40%/70% prefix cuts
LATE_LEAK = '''DESK = 'macro'
PARAMS = {}


def signal(panel, params):
    import datetime
    late = pl.col('date') > datetime.date(2002, 1, 1)
    paid = pl.col('adjclose').shift(-2).over('symbol') / pl.col('adjclose').shift(-1).over('symbol') - 1
    honest = pl.col('adjclose') / pl.col('adjclose').shift(5).over('symbol') - 1
    return panel.select(pl.when(late).then(paid).otherwise(honest).sign().fill_null(0).alias('w'))['w'].to_numpy()
'''
# clean at its default params, leaks for any other value: the old harness never checked parameter neighbours
NEIGHBOUR_LEAK = '''DESK = 'macro'
PARAMS = {'look': 20}


def signal(panel, params):
    k = params['look']
    if k == 20:
        r = pl.col('adjclose') / pl.col('adjclose').shift(k).over('symbol') - 1
    else:
        r = pl.col('adjclose').shift(-2).over('symbol') / pl.col('adjclose').shift(-1).over('symbol') - 1
    return panel.select(r.sign().fill_null(0).alias('w'))['w'].to_numpy()
'''


def test_canary_would_pass_the_stats_but_is_rejected(tmp_path, monkeypatch):
    s = macro_settings(tmp_path)
    st = MacroStore(s, allowed=('dev',))
    with monkeypatch.context() as m:                          # what stats alone would say (guard switched off)
        m.setattr(asof, 'asof_guard', lambda *a, **k: {'ok': True})
        r = evaluate_macro(CANARY_MACRO, st, 'dev', MacroCost())
    assert r['sharpe'] > 5 and r['pnl'] > 0                   # a spectacular, entirely fake edge
    for _ in range(5):                                        # rejected on every run (fresh random cuts each time)
        with pytest.raises(LookAheadError):
            evaluate_macro(CANARY_MACRO, st, 'dev', MacroCost())
        with pytest.raises(LookAheadError):
            strategy_eval(job(s, CANARY_MACRO, robustness=True, stress=True))


def test_harness_refuses_to_evaluate_if_canary_is_not_rejected(tmp_path, monkeypatch):
    s = macro_settings(tmp_path)
    assert canary_selfcheck()
    monkeypatch.setattr(asof, 'asof_guard', lambda *a, **k: {'ok': True})
    with pytest.raises(HarnessCompromised):
        strategy_eval(job(s, TREND))


def test_late_leak_evades_old_fixed_cuts_but_not_the_guard():
    mod = load_strategy(LATE_LEAK)
    panel = synthetic_macro_panel(n_days=300)
    assert macro_causality(mod, panel)['ok']                  # the old soft check misses it
    with pytest.raises(LookAheadError):
        asof_guard(lambda v: macro_signal(mod, v), panel, 'date', n_cuts=4, seed=1)


def test_leak_only_in_parameter_neighbours_is_caught(tmp_path):
    s = macro_settings(tmp_path)
    out = strategy_eval(job(s, TREND, robustness=True))      # control: clean strategy is fine
    assert out['causality']['ok'] and out['causality']['guard'] == 'asof'
    with pytest.raises(LookAheadError):
        strategy_eval(job(s, NEIGHBOUR_LEAK, robustness=True))


def test_engine_asserts_execution_after_decision(tmp_path):
    s = macro_settings(tmp_path)
    with pytest.raises(LookAheadError, match='delay_d'):
        evaluate_macro(TREND, MacroStore(s, allowed=('dev',)), 'dev', MacroCost(delay_d=0))


def test_pipeline_rejects_the_canary_as_look_ahead(tmp_path):
    from desk.pipeline.validation import Pipeline
    s = macro_settings(tmp_path)
    P = Pipeline(s, ledger=Ledger(tmp_path / 'p.sqlite3'))

    async def in_process(j, timeout=0):                       # same as harness.main: exceptions become {'error'}
        j = dict(j, _home=str(s.home), _store=str(s.store))
        try:
            return strategy_eval(j)
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}
    P._job = in_process
    sid = P.L.add_strategy('canary', 'a', 'macro', CANARY_MACRO, code_hash(CANARY_MACRO), '1d', 't')
    asyncio.run(P.process_async(sid))
    st = P.L.strategy(sid)
    assert st['status'] == 'rejected_dev' and st['gate_reasons'] == ['look_ahead']
