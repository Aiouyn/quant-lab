"""Phase 1b: out-of-sample discipline. Each test is a way to spend or peek at out-of-sample data that must be stopped."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import types
from datetime import date, timedelta

import numpy as np
import polars as pl

from desk.lab.causality import date_literals, panel_causality
from desk.lab.strategy import family_hash, load_strategy, structural_hash


def _panel(n_days=60, syms=('A', 'B', 'C'), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        for s in syms:
            rows.append({'date': date(2010, 1, 1) + timedelta(days=i), 'symbol': s, 'x': float(rng.normal())})
    return pl.DataFrame(rows).sort(['date', 'symbol'])


def _sig(mod, panel):
    return np.asarray(mod.signal(panel, {}), float)


# -------------------------------------------------------------------------------------------------- family budget
def test_family_hash_groups_retuned_variants_but_not_new_logic():
    a = "PARAMS={'n': 20, 'z': 1.5}\ndef signal(df, p):\n    return (df['mid'] - df['mid'].rolling_mean(p['n'])).sign()\n"
    tuned = a.replace('20', '21').replace('1.5', '1.6')
    other = a.replace('.sign()', '.clip(-1, 1)')
    assert structural_hash(a) != structural_hash(tuned)            # a different strategy for the holdout ...
    assert family_hash(a) == family_hash(tuned)                    # ... but the same family for the val budget
    assert family_hash(a) != family_hash(other)


def _pipeline(tmp_path):
    from desk.company.ledger import Ledger
    from desk.pipeline.validation import Pipeline
    from desk.config import SETTINGS
    L = Ledger(tmp_path / 'desk.sqlite3')
    pipe = Pipeline.__new__(Pipeline)
    pipe.L, pipe.s, pipe.queue = L, dataclasses.replace(SETTINGS, home=tmp_path), None
    pipe.jobs = []

    async def job(j, timeout=0):
        pipe.jobs.append(j)
        return {'error': 'stub'}
    pipe._job = job
    pipe._notify = lambda s, m: None
    pipe.cost = lambda: {}
    return pipe, L


def _submit(L, code, author='r1'):
    sid = L.add_strategy('s', author, 'macro', code, structural_hash(code), '1d', 'thesis')
    L.update_strategy(sid, status='validating')
    return sid


def test_val_budget_per_family_is_enforced(tmp_path):
    pipe, L = _pipeline(tmp_path)
    base = "DESK='macro'\nPARAMS={'n': %d}\ndef signal(panel, p):\n    return panel['x'] * 0\n"
    sids = [_submit(L, base % n) for n in (10, 11, 12)]
    for sid in sids:
        asyncio.run(pipe.process_async(sid))
    assert len(pipe.jobs) == 2                                       # two looks, then the family is out of budget
    third = L.strategy(sids[2])
    assert third['status'] == 'rejected_val' and 'val_budget_exhausted_for_family' in third['gate_reasons']
    fresh = _submit(L, "DESK='macro'\nPARAMS={}\ndef signal(panel, p):\n    return panel['x'].sign()\n")
    asyncio.run(pipe.process_async(fresh))
    assert len(pipe.jobs) == 3                                       # a genuinely different idea still gets a look
    assert pipe.jobs[-1]['n_trials'] >= 3                            # every look counts toward the deflation


def test_submit_refuses_a_family_without_budget(tmp_path):
    from desk.company.tools import Toolbox
    pipe, L = _pipeline(tmp_path)
    base = "DESK='macro'\nPARAMS={'n': %d}\ndef signal(panel, p):\n    return panel['x'] * 0\n"
    for n in (10, 11):
        L.x('INSERT INTO val_uses VALUES(?,?,?)', (family_hash(base % n), f'S-{n}', 0.0))
    t = Toolbox.__new__(Toolbox)
    t.L, t.s = L, pipe.s
    t.agent = types.SimpleNamespace(name='r1', pod='macro', role='researcher')
    t.org = types.SimpleNamespace(pipeline=types.SimpleNamespace(enqueue=lambda sid: None))
    out = asyncio.run(t.t_submit_strategy('again', base % 13, 'retuned'))
    assert 'error' in out and 'validation' in out['error']


# ---------------------------------------------------------------------------------------- trials are not voluntary
def test_unlogged_sandbox_backtests_still_count(tmp_path):
    from desk.company import tools as T
    assert T.BACKTEST_CALL.search("r = macro_backtest(p, w)")
    assert T.BACKTEST_CALL.search("x = run_strategy(code)")
    assert not T.BACKTEST_CALL.search("print(macro_panel().shape)")


# ---------------------------------------------------------------------------------------------- stronger causality
def test_date_literals_are_flagged():
    assert date_literals("CRASH = '2020-03-16'\ndef signal(p, q):\n    return 0")
    assert date_literals("T = 1584316800000\n")                      # epoch ms
    assert date_literals("import datetime\nd = datetime.date(2020, 3, 16)\n")
    assert not date_literals("PARAMS={'n': 20, 'z': 1.5, 'cap': 100000}\n")


def test_full_panel_detector_is_caught_by_future_perturbation():
    # Honest on any truncated panel, cheats only when it sees the full one (prefix cuts can't see this).
    code = """DESK='macro'
PARAMS={}
def signal(panel, p):
    import numpy as np
    x = panel['x'].to_numpy()
    if panel.height < 180:
        return np.sign(x)
    nxt = panel.select(pl.col('x').shift(-1).over('symbol'))['x'].fill_null(0).to_numpy()
    return np.sign(nxt)
"""
    mod = load_strategy(code)
    res = panel_causality(_sig, mod, _panel(), 'date', code=code, cuts=(0.95,))
    assert not res['ok'] and any('perturbation' in i for i in res['issues'])


def test_date_literal_fails_panel_causality():
    code = "DESK='macro'\nPARAMS={}\nCRASH='2010-02-01'\ndef signal(panel, p):\n    return (panel['date'].cast(pl.Utf8) == CRASH).cast(pl.Float64)\n"
    res = panel_causality(_sig, load_strategy(code), _panel(), 'date', code=code)
    assert not res['ok'] and 'date literals' in res['issues'][0]


def test_honest_panel_signal_passes():
    code = "DESK='macro'\nPARAMS={}\ndef signal(panel, p):\n    return panel.select(pl.col('x').rolling_mean(3).over('symbol').fill_null(0))['x']\n"
    res = panel_causality(_sig, load_strategy(code), _panel(), 'date', code=code)
    assert res['ok'], res['issues']
    assert res['checks'] == 5


# --------------------------------------------------------------------------------------------- point-in-time data
def test_macro_adjclose_does_not_reveal_future_dividends():
    from desk.lab.macro import visible
    d = [date(2000, 1, 3) + timedelta(days=i) for i in range(6)]
    raw = pl.DataFrame({'date': d, 'symbol': ['A'] * 6, 'close': [100.0] * 6,
                        'adjclose': [99.0, 99.0, 99.0, 99.0, 100.0, 100.0],       # $1 dividend on day 4, back-adjusted
                        'adj_open': [99.0] * 4 + [100.0] * 2})
    v = visible(raw)
    assert v['adjclose'][:4].to_list() == [100.0] * 4              # before the dividend: no trace of it
    assert np.allclose(v['adjclose'].to_numpy()[1:] / v['adjclose'].to_numpy()[:-1],
                       raw['adjclose'].to_numpy()[1:] / raw['adjclose'].to_numpy()[:-1])   # same returns
    assert visible(raw.head(3))['adjclose'].to_list() == v['adjclose'][:3].to_list()       # causal


def test_carry_dev_view_holds_no_sealed_hours(tmp_path):
    from desk.lab.carry import CarryStore, dev_view_root, write_dev_view
    s = types.SimpleNamespace(store=tmp_path / 'store')
    root = s.store / 'carry'
    (root / 'coins').mkdir(parents=True)
    day = 86_400_000
    t0 = 1_700_000_000_000 // day * day
    ts = np.arange(t0, t0 + 20 * day, 3_600_000)
    pl.DataFrame({'ts': ts, 'coin': 'BTC', 'hl_funding': np.zeros(len(ts))}).write_parquet(root / 'coins' / 'BTC.parquet')
    iso = lambda ms: date.fromtimestamp(ms / 1000).isoformat()
    (root / 'splits.json').write_text(json.dumps({'dev': [iso(t0), iso(t0 + 9 * day)],
                                                  'val': [iso(t0 + 10 * day), iso(t0 + 14 * day)],
                                                  'holdout': [iso(t0 + 15 * day), iso(t0 + 19 * day)]}))
    out = write_dev_view(s)
    assert out == dev_view_root(s)
    v = pl.read_parquet(out / 'coins' / 'BTC.parquet')
    _, end = CarryStore(s, allowed=('dev',)).range_ms('dev')
    assert len(v) and v['ts'].max() < end
    assert set(json.loads((out / 'splits.json').read_text())) == {'dev'}
