"""Cycle 1b item 2: every backtest (dev/val, any variant) is a trial row; DSR deflates by the real cumulative count
per family and across families, using the variance of trial Sharpes. Synthetic data only."""
import asyncio
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from desk.company.ledger import Ledger
from desk.config import Settings
from desk.lab import stats
from desk.lab.harness import deflate, strategy_eval
from desk.lab.strategy import code_hash
from synth_desk import TREND, job, macro_settings

OTHER = TREND.replace("r.sign()", "(-r).sign()").replace("shift(params['look'])", "shift(params['look'] * 2)")


def test_every_backtest_variant_appends_a_ledger_row(tmp_path):
    s = macro_settings(tmp_path)
    out = strategy_eval(job(s, robustness=True, stress=True))
    nb = out['robustness']['param_neighborhood']['variants']
    expected = 1 + 1 + 5 + len(nb)                 # base + stress + 5 robustness variants + parameter neighbours
    assert len(out['trial_log']) == expected
    L = Ledger(tmp_path / 'l.sqlite3')
    L.record_eval_trials(TREND, out, 'test')
    rows = L.q('SELECT * FROM eval_trials')
    assert len(rows) == expected
    assert {r['family'] for r in rows} == {L.family_of(TREND)}
    assert {r['split'] for r in rows} == {'dev'} and all(r['n_obs'] > 100 and r['created'] for r in rows)
    assert len({r['params_hash'] for r in rows}) == expected            # every variant/param set is its own row
    with pytest.raises(sqlite3.DatabaseError, match='insert-only'):
        L.x('DELETE FROM eval_trials')
    with pytest.raises(sqlite3.DatabaseError, match='insert-only'):
        L.x('UPDATE eval_trials SET sharpe=0')


def test_dsr_counts_cumulative_trials_per_family_and_globally(tmp_path):
    s = macro_settings(tmp_path, drift=0.0006)
    L = Ledger(tmp_path / 'l.sqlite3')
    fresh = strategy_eval(job(s, split='val', stress=True, trials=L.trial_moments(TREND)))
    for i in range(300):                                         # other families searched a lot
        L.record_eval_trial(f'sk:other{i % 7}', 'macro', 'dev', 'base', 0.5 + (i % 5) * 0.1, 500, 252, 'test')
    for i in range(40):                                          # this family was tuned 40 times
        L.record_eval_trial(L.family_of(TREND), 'macro', 'dev', f'v{i}', 0.8, 500, 252, 'test')
    m = L.trial_moments(TREND)
    assert m['global']['n'] == 340 and m['family']['n'] == 40
    out = strategy_eval(job(s, split='val', stress=True, trials=m))
    n_cur = len(out['trial_log'])
    assert out['stats']['n_trials'] == 340 + n_cur and out['stats']['n_trials_family'] == 40 + n_cur
    assert out['stats']['dsr'] == min(out['stats']['dsr_family'], out['stats']['dsr_global'])
    assert out['stats']['dsr'] < fresh['stats']['dsr']           # more trials -> harder to pass


def test_dsr_uses_trial_sharpe_variance_never_below_null():
    rng = np.random.default_rng(3)
    daily = 0.001 + 0.01 * rng.standard_normal(500)
    log = [{'sharpe': 1.0, 'periods': 252}]
    null = stats.dsr(daily, 101, None)
    wide = {'n': 100, 'sum': 0.0, 'sumsq': 100 * 0.2 ** 2}       # per-period SR std 0.2 >> null 1/sqrt(500)
    d_wide = deflate(daily, {'trials': {'family': wide, 'global': wide}}, log)
    assert d_wide['global']['sr_var'] > 1 / 500 and d_wide['global']['dsr'] < null
    tight = {'n': 100, 'sum': 100 * 0.063, 'sumsq': 100 * 0.063 ** 2}   # identical trials: variance ~0
    d_tight = deflate(daily, {'trials': {'family': tight, 'global': tight}}, log)
    assert d_tight['global']['sr_var'] == pytest.approx(1 / 500)        # floored at the null, never looser
    assert d_tight['global']['dsr'] == pytest.approx(null)


def test_min_dsr_is_095_in_config_and_readme():
    from desk.pipeline.validation import val_gates
    assert Settings().min_dsr == 0.95
    assert 'deflated_sharpe>=0.95' in val_gates(Settings())
    readme = (Path(__file__).resolve().parents[2] / 'README.md').read_text(encoding='utf-8')
    assert re.search(r'deflated Sharpe ≥ 0\.95', readme)


def test_pipeline_passes_ledger_moments_and_records_val_trials(tmp_path):
    from desk.pipeline.validation import Pipeline
    s = Settings(home=tmp_path / 'home', store_dir=tmp_path / 'store')
    P = Pipeline(s, ledger=Ledger(tmp_path / 'p.sqlite3'))
    P.L.record_eval_trial('sk:x', 'macro', 'dev', 'base', 1.0, 300, 252, 'test')
    seen = []

    async def fake_job(j, timeout=0):
        seen.append(j)
        return {'error': 'synthetic', 'desk': 'macro', 'split': 'val', 'code_hash': 'h',
                'trial_log': [{'variant': 'base', 'params_hash': 'p', 'sharpe': 1.2, 'n_obs': 250, 'periods': 252}]}
    P._job = fake_job
    sid = P.L.add_strategy('n', 'a', 'macro', TREND, code_hash(TREND), 'd', 't')
    P.L.update_strategy(sid, status='validating')
    asyncio.run(P.process_async(sid))
    assert seen[0]['trials']['global']['n'] == 1
    sid2 = P.L.add_strategy('n', 'a', 'macro', TREND, code_hash(TREND), 'd', 't')
    P.L.update_strategy(sid2, status='validating')

    async def ok_job(j, timeout=0):
        seen.append(j)
        return {'desk': 'macro', 'split': 'val', 'code_hash': 'h', 'summary': {'sharpe': 0.0}, 'stats': {},
                'trial_log': [{'variant': 'base', 'params_hash': 'p', 'sharpe': 1.2, 'n_obs': 250, 'periods': 252}]}
    P._job = ok_job
    asyncio.run(P.process_async(sid2))
    assert P.L.q("SELECT COUNT(*) n FROM eval_trials WHERE source='pipeline_val'")[0]['n'] == 1
    assert seen[-1]['trials']['family']['n'] == 0 and seen[-1]['trials']['global']['n'] == 1
