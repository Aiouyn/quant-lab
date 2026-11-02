"""Phase 4: forward verdicts are a pre-registered test, not a Sharpe threshold that luck passes 40% of the time."""
from __future__ import annotations

import asyncio
import dataclasses
import json

import numpy as np

from desk.lab.forward import LOOKS, firm_book, firm_forward, forward_verdict


def _sims(sharpe_ann, n_days, n_sims, seed=0, ann=365):
    rng = np.random.default_rng(seed)
    mu = sharpe_ann / np.sqrt(ann) * 0.01
    return rng.normal(mu, 0.01, (n_sims, n_days))


def test_no_edge_is_rarely_confirmed():
    sims = _sims(0.0, LOOKS[-1], 3000)
    conf = np.mean([forward_verdict(x)['verdict'] == 'FORWARD_CONFIRMED' for x in sims])
    assert conf <= 0.05                                       # family-wise over the three looks
    old_rule = np.mean([(x[:60].mean() / x[:60].std() * np.sqrt(365) >= 0.5) and x[:60].sum() > 0 for x in sims])
    assert old_rule > 0.3                                     # what the previous 60-day Sharpe >= 0.5 rule allowed


def test_real_edge_is_confirmed_and_losers_fail():
    strong = _sims(4.0, LOOKS[-1], 300, seed=1)
    assert np.mean([forward_verdict(x)['verdict'] == 'FORWARD_CONFIRMED' for x in strong]) > 0.8
    loser = _sims(-3.0, 130, 1, seed=2)[0]
    v = forward_verdict(loser)
    assert v['verdict'] == 'FORWARD_FAILED' and 'not positive' in v['reason']


def test_days_between_looks_cannot_change_the_answer():
    x = _sims(0.0, 100, 1, seed=3)[0]
    a = forward_verdict(x[:61])
    y = x.copy()
    y[60:] = 0.05                                             # a lucky streak after the look is not looked at
    b = forward_verdict(y[:100])
    assert a['look'] == b['look'] == 60 and a['t'] == b['t'] and a['verdict'] == b['verdict']
    assert forward_verdict(x[:59])['verdict'] == 'RUNNING'


def test_more_strategies_tested_means_a_stricter_bar():
    one, many = forward_verdict([0.001] * 10, 1), forward_verdict([0.001] * 10, 20)
    assert many['p_critical'] < one['p_critical']
    assert many['detectable_sharpe_next_look'] > one['detectable_sharpe_next_look']


def test_firm_book_is_equal_weight_and_keeps_failures():
    a = {'desk': 'perps', 'daily': {'day': [1, 2, 3], 'ret': [0.01, 0.01, 0.01]}, 'verdict': 'FORWARD_CONFIRMED'}
    b = {'desk': 'perps', 'daily': {'day': [2, 3], 'ret': [-0.03, -0.03]}, 'verdict': 'FORWARD_FAILED'}
    days, ret = firm_book([a, b])
    assert days == [1, 2, 3] and np.allclose(ret, [0.01, -0.01, -0.01])
    f = firm_forward([a, b])
    assert f['strategies'] == 2 and f['verdict'] == 'RUNNING'


def test_pipeline_retires_forward_failures_and_records_the_firm_book(tmp_path):
    from desk.company.ledger import Ledger
    from desk.company.rsi import compute_generation_scoreboard
    from desk.config import SETTINGS
    from desk.lab.perps import PerpsStore
    from desk.pipeline.validation import Pipeline
    from tests.v3.test_panel_drills import perps_panel
    s = dataclasses.replace(SETTINGS, home=tmp_path, store_dir=None)
    (s.store / 'perps').mkdir(parents=True)
    perps_panel(n_coins=2, start='2026-09-20', end='2027-02-15').write_parquet(s.store / 'perps' / 'panel.parquet')
    PerpsStore(s).splits()
    L = Ledger(tmp_path / 'desk.sqlite3')
    code = "DESK='perps'\nPARAMS={}\ndef signal(panel, p):\n    return panel['px'] * 0\n"
    ids = []
    for name in ('loser', 'winner'):
        sid = L.add_strategy(name, 'r1', 'perps', code + f'# {name}\n', name, '1h', 't')
        L.update_strategy(sid, status='approved')
        ids.append(sid)
    pipe = Pipeline.__new__(Pipeline)
    pipe.L, pipe.s, pipe.queue = L, s, None
    pipe.cost = lambda: {}
    rng = np.random.default_rng(0)

    async def job(j):
        n = 130
        mu = -0.002 if '# loser' in j['code'] else 0.004
        r = rng.normal(mu, 0.01, n)
        return {'summary': {'sharpe': 0.0, 'pnl': float(r.sum())}, 'stats': {},
                'daily': {'day': list(range(20730, 20730 + n)), 'ret': [float(v) for v in r]}}
    pipe._job = job
    res = asyncio.run(pipe.forward_check())
    assert res['forward_days'].get('perps', 0) > 100
    st = {i: L.strategy(i) for i in ids}
    assert st[ids[0]]['status'] == 'forward_failed' and st[ids[0]]['forward_report']['verdict'] == 'FORWARD_FAILED'
    assert st[ids[1]]['forward_report']['verdict'] == 'FORWARD_CONFIRMED'
    assert st[ids[1]]['forward_report']['test']['tests_counted'] == 2
    firm = L.get('firm_forward')
    assert firm['strategies'] == 2 and firm['days'] == 130
    board = compute_generation_scoreboard(L)
    assert board['forward_decided'] == 2 and board['generation_fitness'] == 0.5
    assert board['firm_forward_verdict'] == firm['verdict']
