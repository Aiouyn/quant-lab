"""Cycle 1b item 1: one-shot holdout keyed by strategy FAMILY, insert-only ledger, burned/forward-only windows,
sha256-manifested splits. Synthetic stores only; no real holdout data is read."""
import asyncio
import json
import sqlite3

import pytest

from desk.company.ledger import Ledger
from desk.config import Settings
from desk.data.provenance import SplitsIntegrityError, read_splits, write_splits
from desk.lab.holdout import HoldoutAlreadyUsed, HoldoutLocked, assert_holdout_open, family_keys, structural_family
from desk.lab.strategy import code_hash

CODE = '''"""Trend on adjusted closes."""
DESK = 'macro'
PARAMS = {'fast': 20, 'slow': 100}


def signal(panel, params):
    fast = panel['adjclose'].rolling_mean(params['fast'])
    slow = panel['adjclose'].rolling_mean(params['slow'])
    return (fast > slow).cast(float)
'''
# the same family after "tuning": new params, renamed locals, new docstring, comments
TWEAKED = '''"""A totally different idea (honest)."""
DESK = 'macro'
PARAMS = {'fast': 25, 'slow': 120}


def signal(panel, params):
    # moving-average crossover
    a = panel['adjclose'].rolling_mean(params['fast'])
    b = panel['adjclose'].rolling_mean(params['slow'])
    return (a > b).cast(float)
'''
NEAR_COPY = TWEAKED.replace("    return (a > b).cast(float)", "    c = a * 1.0\n    return (c > b).cast(float)")
DIFFERENT = '''DESK = 'macro'
PARAMS = {'look': 63}


def signal(panel, params):
    r = panel['adjclose'].pct_change(params['look'])
    z = (r - r.mean()) / r.std()
    return z.clip(-1, 1).fill_null(0.0)
'''
FORWARD = ['2026-11-02', '2027-04-30']


def _settings(tmp_path):
    return Settings(home=tmp_path / 'home', store_dir=tmp_path / 'store')


def _seal(settings, desk, holdout):
    root = settings.store / desk
    root.mkdir(parents=True, exist_ok=True)
    write_splits(root / 'splits.json', {'dev': ['2010-01-01', '2016-12-31'], 'val': ['2017-01-01', '2019-12-31'],
                                        'holdout': holdout, 'forward': ['2027-05-01', '2099-12-31']})


# ------------------------------------------------------------------------------------------- insert-only ledger
def test_holdout_tables_are_insert_only(tmp_path):
    L = Ledger(tmp_path / 'l.sqlite3')
    L.claim_holdout_shot('S-1', CODE, code_hash(CODE), FORWARD)
    for sql in ("UPDATE holdout_shots SET strategy='x'", 'DELETE FROM holdout_shots',
                "UPDATE holdout_uses SET passed=1", 'DELETE FROM holdout_uses',
                "INSERT OR REPLACE INTO holdout_uses VALUES('%s','S-2',0,NULL)" % code_hash(CODE),
                "INSERT OR REPLACE INTO holdout_shots(family,strategy) SELECT family,'S-2' FROM holdout_shots"):
        with pytest.raises(sqlite3.DatabaseError, match='insert-only'):
            L.x(sql)
    L.record_holdout_result('f', 'S-1', True)
    with pytest.raises(sqlite3.DatabaseError, match='insert-only'):
        L.x('UPDATE holdout_results SET passed=0')
    assert L.q('SELECT strategy FROM holdout_shots')[0]['strategy'] == 'S-1'


# ------------------------------------------------------------------------------------------- family keys
def test_family_survives_param_name_docstring_edits():
    assert structural_family(CODE) == structural_family(TWEAKED)
    assert structural_family(CODE) != structural_family(DIFFERENT)
    assert code_hash(CODE) != code_hash(TWEAKED)             # the old code_hash key gave TWEAKED a fresh shot


def test_one_shot_per_family_not_per_code_hash(tmp_path):
    L = Ledger(tmp_path / 'l.sqlite3')
    L.claim_holdout_shot('S-1', CODE, code_hash(CODE), FORWARD)
    for code in (CODE, TWEAKED, NEAR_COPY):
        with pytest.raises(HoldoutAlreadyUsed):
            L.claim_holdout_shot('S-2', code, code_hash(code), FORWARD)
    L.claim_holdout_shot('S-3', DIFFERENT, code_hash(DIFFERENT), FORWARD)   # a genuinely different family is fine


def test_declared_family_and_lineage_inherit_the_shot(tmp_path):
    L = Ledger(tmp_path / 'l.sqlite3')
    p = L.add_strategy('parent', 'a', 'macro', CODE, code_hash(CODE), 'd', 't')
    L.claim_holdout_shot(p, CODE, code_hash(CODE), FORWARD)
    c = L.add_strategy('child', 'a', 'macro', DIFFERENT, code_hash(DIFFERENT), 'd', 't')
    L.add_lineage(c, 'strategy', [p], 'mutate', 'macro')
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_shot(c, DIFFERENT, code_hash(DIFFERENT), FORWARD)
    decl = "FAMILY = 'trend'\n" + DIFFERENT
    L.claim_holdout_shot('S-x', decl, code_hash(decl), FORWARD)
    other = "FAMILY = 'Trend'\n" + DIFFERENT.replace("pct_change(params['look'])", "diff()").replace('.clip(-1, 1)', '')
    assert any(k.startswith('decl:') for k in family_keys(other))
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_shot('S-y', other, code_hash(other), FORWARD)


def test_legacy_holdout_uses_are_burned_on_open(tmp_path):
    db = tmp_path / 'l.sqlite3'
    con = sqlite3.connect(db)                                # a pre-lock ledger: code_hash-keyed, mutable
    con.executescript('''CREATE TABLE strategies(id TEXT PRIMARY KEY, name TEXT, author TEXT, pod TEXT, code TEXT,
      code_hash TEXT, resolution TEXT, thesis TEXT, status TEXT, dev_report TEXT, review TEXT, val_report TEXT,
      holdout_report TEXT, forward_report TEXT, paper_report TEXT, gate_reasons TEXT, created REAL, updated REAL);
      CREATE TABLE holdout_uses(code_hash TEXT PRIMARY KEY, strategy TEXT, created REAL, passed INT);''')
    con.execute("INSERT INTO strategies(id,code,code_hash,status) VALUES('S-old',?,?,'rejected_holdout')",
                (CODE, code_hash(CODE)))
    con.execute("INSERT INTO holdout_uses VALUES(?, 'S-old', 0, 0)", (code_hash(CODE),))
    con.commit()
    con.close()
    L = Ledger(db)
    assert {r['reason'] for r in L.q('SELECT reason FROM holdout_shots')} == {'burned_legacy'}
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_shot('S-new', TWEAKED, code_hash(TWEAKED), FORWARD)
    Ledger(db)                                               # idempotent re-open
    assert L.q("SELECT COUNT(*) n FROM holdout_shots WHERE strategy='S-old'")[0]['n'] >= 1


# ------------------------------------------------------------------------------------------- burned windows
@pytest.mark.parametrize('desk,window', [('macro', ['2020-01-01', '2025-06-30']), ('perps', ['2026-04-02', '2026-10-01']),
                                         ('pred', ['2026-06-01', '2026-07-31']), ('carry', ['2026-08-01', '2026-10-29']),
                                         ('l2', ['2026-10-20', '2026-10-21', '2026-10-30'])])
def test_existing_and_pre_lock_holdouts_are_burned(desk, window):
    with pytest.raises(HoldoutLocked):
        assert_holdout_open(desk, window)


def test_forward_only_window_is_open():
    assert str(assert_holdout_open('macro', FORWARD)[0]) == '2026-11-02'
    assert str(assert_holdout_open('macro', ['2026-10-30', '2027-01-01'])[0]) == '2026-10-30'


def test_store_refuses_burned_holdout_but_forward_loads_still_work(tmp_path):
    from desk.lab.macro import MacroStore
    s = _settings(tmp_path)
    _seal(s, 'macro', ['2020-01-01', '2025-06-30'])
    st = MacroStore(s, allowed=('dev', 'val', 'holdout', 'forward'))
    with pytest.raises(HoldoutLocked):
        st.range_dates('holdout')
    assert str(st.range_dates('forward')[0]) == '2027-05-01'     # `latest` over allowed splits must not trip the lock


# ------------------------------------------------------------------------------------------- pipeline
def _pipeline(tmp_path, holdout):
    from desk.pipeline.validation import Pipeline
    s = _settings(tmp_path)
    _seal(s, 'macro', holdout)
    P = Pipeline(s, ledger=Ledger(tmp_path / 'p.sqlite3'))
    calls = []

    async def fake_job(job, timeout=0):
        calls.append(job['split'])
        return {'error': 'synthetic: no data in tests'}
    P._job = fake_job
    return P, calls


def _holdout_candidate(P, code):
    sid = P.L.add_strategy('n', 'a', 'macro', code, code_hash(code), 'd', 't')
    P.L.update_strategy(sid, status='holdout')
    return sid


def test_pipeline_refuses_second_shot_for_tweaked_family(tmp_path):
    P, calls = _pipeline(tmp_path, FORWARD)
    a = _holdout_candidate(P, CODE)
    asyncio.run(P.process_async(a))
    assert calls == ['holdout']
    b = _holdout_candidate(P, TWEAKED)
    asyncio.run(P.process_async(b))
    assert calls == ['holdout'], 'a re-parameterized copy must not get a second holdout run'
    assert P.L.strategy(b)['status'] == 'rejected_holdout'
    assert P.L.strategy(b)['gate_reasons'] == ['holdout_already_used_by_family']


def test_pipeline_never_runs_a_burned_holdout(tmp_path):
    P, calls = _pipeline(tmp_path, ['2020-01-01', '2025-06-30'])
    a = _holdout_candidate(P, CODE)
    asyncio.run(P.process_async(a))
    assert calls == [] and P.L.strategy(a)['status'] == 'holdout_locked'
    assert P.L.q('SELECT COUNT(*) n FROM holdout_shots')[0]['n'] == 0      # no shot consumed while locked


# ------------------------------------------------------------------------------------------- splits manifest
def test_splits_manifest_detects_tampering(tmp_path):
    from desk.lab.macro import MacroStore
    s = _settings(tmp_path)
    _seal(s, 'macro', FORWARD)
    f = s.store / 'macro' / 'splits.json'
    assert MacroStore(s).splits()['holdout'] == FORWARD
    sp = json.loads(f.read_text())
    sp['dev'][1] = '2018-12-31'                              # silently move val days into dev
    f.write_text(json.dumps(sp, indent=1))
    with pytest.raises(SplitsIntegrityError, match='does not match'):
        MacroStore(s).splits()
    with pytest.raises(SplitsIntegrityError, match='does not match'):
        MacroStore(s, allowed=('dev',)).load('dev')


def test_unsealed_splits_are_a_hard_error(tmp_path):
    f = tmp_path / 'splits.json'
    f.write_text(json.dumps({'dev': ['2020-01-01', '2020-02-01']}))
    with pytest.raises(SplitsIntegrityError, match='no sha256 manifest'):
        read_splits(f)
    from desk.data.provenance import seal_splits
    seal_splits(f)
    assert read_splits(f)['dev'][0] == '2020-01-01'


# ------------------------------------------------------------------------------------------- RSI choke point
def test_rsi_holdout_eval_requires_a_claimed_shot(tmp_path, monkeypatch):
    from desk.company import rsi
    from desk.data import store_builder
    s = _settings(tmp_path)
    L = Ledger(tmp_path / 'l.sqlite3')
    monkeypatch.setattr(store_builder, 'ensure_store_panels',
                        lambda *a, **k: pytest.fail('holdout data path reached without a claimed shot'))
    for kw in ({}, {'ledger': L}, {'ledger': L, 'holdout_family': 'sk:not-claimed'}):
        with pytest.raises(HoldoutLocked):
            rsi._evaluate_truth_machine_split(CODE, 'S-1', 'holdout', s, **kw)
    fam = L.claim_holdout_shot('S-1', CODE, code_hash(CODE), FORWARD)
    with pytest.raises(HoldoutLocked):                       # someone else's claim does not unlock it either
        rsi._evaluate_truth_machine_split(CODE, 'S-2', 'holdout', s, ledger=L, holdout_family=fam)
