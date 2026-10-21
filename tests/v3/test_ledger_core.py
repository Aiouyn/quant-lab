"""Charter R6 / R7a / R4 / R7c ledger core (plumbing only; no gate thresholds or decisions change)."""
import asyncio
import sqlite3
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

from desk.company.ledger import HoldoutAlreadyUsed, Ledger
from desk.config import SETTINGS
from desk.lab.identity import evaluation_identity, normalized_ast
from desk.lab.strategy import code_hash
from desk.pipeline.validation import Pipeline

BASE = "DESK = 'macro'\nPARAMS = {'k': 3}\n\ndef signal(panel, params):\n    return panel['adjclose'] * 0.0\n"
COSMETIC = ('"""A docstring."""\nDESK = \'macro\'\n# a comment\nPARAMS = {\'k\': 3}   \n\n\n'
            'def signal(panel, params):\n    """doc"""\n    # noop\n    return panel[\'adjclose\'] * 0.0\n')
CONST = BASE.replace("'k': 3", "'k': 4")


@pytest.fixture
def L(tmp_path):
    L = Ledger(tmp_path / 'company.db')
    L.trial('a', 'dev_backtest', 'h', 1.0, 10, desk='macro', split='dev', family='macro_tsmom')
    L.claim_holdout('id1', 'S-1')
    L.holdout_result('id1', 'S-1', True)
    return L


@pytest.mark.parametrize('sql', [
    "UPDATE trials SET sharpe=9", "DELETE FROM trials",
    "INSERT OR REPLACE INTO trials(id,author,kind,code_hash,sharpe,days,created) VALUES(1,'x','dev_backtest','h',9,10,0)",
    "UPDATE holdout_uses SET strategy='S-9'", "DELETE FROM holdout_uses",
    "INSERT OR REPLACE INTO holdout_uses VALUES('id1','S-2',0,NULL)",
    "INSERT INTO holdout_uses VALUES('id1','S-2',0,NULL)",
    "UPDATE holdout_results SET passed=0", "DELETE FROM holdout_results",
])
def test_insert_only_tables_abort(L, sql):
    with pytest.raises(sqlite3.DatabaseError, match='append-only|already used'):
        L.x(sql)
    assert L.q('SELECT COUNT(*) n FROM trials')[0]['n'] == 1
    assert L.q('SELECT strategy FROM holdout_uses')[0]['strategy'] == 'S-1'


def test_trials_have_desk_split_family_identity(L):
    cols = {r['name']: r for r in L.q('PRAGMA table_info(trials)')}
    for c in ('desk', 'split', 'family'):
        assert cols[c]['notnull'] == 1
    assert 'identity' in cols
    with pytest.raises(sqlite3.IntegrityError):
        L.x("INSERT INTO trials(author,kind,code_hash,desk,split,family) VALUES('a','k','h',NULL,'dev','f')")
    r = L.q('SELECT desk, split, family FROM trials')[0]
    assert (r['desk'], r['split'], r['family']) == ('macro', 'dev', 'macro_tsmom')


def test_old_ledger_is_migrated(tmp_path):
    db = tmp_path / 'old.db'
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE trials(id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, kind TEXT, code_hash TEXT,"
                      " sharpe REAL, days INT, created REAL);"
                      "CREATE TABLE holdout_uses(code_hash TEXT PRIMARY KEY, strategy TEXT, created REAL, passed INT);"
                      "INSERT INTO trials(author,kind,code_hash,sharpe,days,created) VALUES('a','submission','h',1,10,0);"
                      "INSERT INTO holdout_uses VALUES('rawhash','S-1',0,1);")
    con.commit(); con.close()
    L = Ledger(db)
    assert L.q('SELECT desk, split, family FROM trials')[0]['family'] == 'unassigned'
    with pytest.raises(HoldoutAlreadyUsed):              # legacy raw-code rows still block
        L.claim_holdout('new-identity', 'S-2', code_hash='rawhash')
    with pytest.raises(sqlite3.DatabaseError):
        L.x('DELETE FROM trials')


def _claim(args):
    db, i = args
    try:
        Ledger(db).claim_holdout('same-identity', f'S-{i}')
        return True
    except HoldoutAlreadyUsed:
        return False


def test_double_holdout_refused_under_concurrency(tmp_path):
    db = tmp_path / 'c.db'
    Ledger(db)
    with ProcessPoolExecutor(8, mp_context=mp.get_context('spawn')) as ex:                    # separate processes and connections: SQLite lock only
        wins = list(ex.map(_claim, [(db, i) for i in range(16)]))
    assert sum(wins) == 1
    L = Ledger(db)
    res, ts = [], [threading.Thread(target=lambda i=i: res.append(_claim((db, 100 + i)))) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not any(res)
    assert L.q('SELECT COUNT(*) n FROM holdout_uses')[0]['n'] == 1


def test_identity_stable_under_cosmetic_edits(tmp_path):
    assert normalized_ast(BASE) == normalized_ast(COSMETIC)
    i0, c0 = evaluation_identity(BASE, {'split': 'holdout'}, tmp_path)
    assert evaluation_identity(COSMETIC, {'split': 'holdout'}, tmp_path)[0] == i0
    assert evaluation_identity(CONST, {'split': 'holdout'}, tmp_path)[0] != i0          # constants matter
    assert evaluation_identity(BASE, {'split': 'holdout', 'cost': 2}, tmp_path)[0] != i0  # settings matter
    (tmp_path / 'macro').mkdir()
    (tmp_path / 'macro' / 'splits.json').write_text('{"dev": 1}')
    assert evaluation_identity(BASE, {'split': 'holdout'}, tmp_path)[0] != i0             # data version matters
    deps = dict(c0['deps'])
    assert {'desk.lab.macro', 'desk.lab.stats', 'desk.lab.harness'} <= set(deps)


def _report():
    return {'desk': 'macro', 'summary': {'sharpe': -1.0, 'pnl': -1.0, 'days': 100, 'trades': 0, 'max_drawdown': 1.0},
            'stats': {}, 'causality': {'ok': True}}


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    p = Pipeline(SETTINGS, ledger=Ledger(tmp_path / 'company.db'))
    p.jobs = []

    async def fake_job(job, timeout=5400):
        p.jobs.append(job)
        return dict(_report(), code_hash=code_hash(job['code']))
    monkeypatch.setattr(p, '_job', fake_job)
    return p


def _strategy(L, code, status):
    sid = L.add_strategy('s', 'alice', 'macro', code, code_hash(code), '1d', 't')
    L.update_strategy(sid, status=status)
    return sid


@pytest.mark.parametrize('second', ['renamed', 'cosmetic', 'same_id_reset'])
def test_pipeline_holdout_is_one_shot_per_identity(pipe, second):
    L = pipe.L
    sid = _strategy(L, BASE, 'holdout')
    asyncio.run(pipe.process_async(sid))
    assert len(pipe.jobs) == 1 and L.strategy(sid)['status'] == 'rejected_holdout'
    if second == 'same_id_reset':
        L.update_strategy(sid, status='holdout'); sid2 = sid
    else:
        sid2 = _strategy(L, BASE if second == 'renamed' else COSMETIC, 'holdout')
    asyncio.run(pipe.process_async(sid2))
    assert len(pipe.jobs) == 1
    assert L.strategy(sid2)['gate_reasons'] == ['holdout_already_used_by_identical_code']
    assert [r['passed'] for r in L.q('SELECT passed FROM holdout_results')] == [0]   # outcome as a new row


def test_dev_trial_written_host_side_with_desk_split(pipe):
    asyncio.run(pipe.process_async(_strategy(pipe.L, BASE, 'submitted')))
    r = pipe.L.q('SELECT kind, desk, split, family FROM trials')
    assert [dict(x) for x in r] == [{'kind': 'submission', 'desk': 'macro', 'split': 'dev', 'family': 'unassigned'}]


def test_sandbox_cannot_write_trials(tmp_path):
    """R7c: research helpers no longer emit anything the host turns into a trial."""
    import os
    from desk import research
    os.environ['DESK_TRIALS'] = str(tmp_path / 't.jsonl')
    try:
        research._log_trial('dev_backtest', 9.0, 100, 'forged')
    finally:
        os.environ.pop('DESK_TRIALS')
    assert not (tmp_path / 't.jsonl').exists()
    import inspect
    from desk.company import tools
    src = inspect.getsource(tools.Toolbox.t_run_python)
    assert 'extra_env' not in src and '_collect_trials' not in inspect.getsource(tools)


# ------------------------------------------------------------------ R7b: job tickets
def _settings(tmp_path):
    import dataclasses
    return dataclasses.replace(SETTINGS, home=tmp_path / 'home', store_dir=tmp_path / 'store')


@pytest.mark.parametrize('case', ['missing', 'unknown', 'reused', 'other_job'])
def test_run_job_refuses_bad_ticket(tmp_path, case, monkeypatch):
    from desk.lab import sandbox
    s = _settings(tmp_path)
    (s.home).mkdir(parents=True)
    L = Ledger(s.ledger)
    job = {'type': 'strategy_eval', 'code': BASE, 'split': 'dev', 'allowed': ['dev']}
    launched = []

    async def fake_exec(*a, **k):
        launched.append(a)
        raise RuntimeError('launched')
    monkeypatch.setattr(sandbox.asyncio, 'create_subprocess_exec', fake_exec)
    if case == 'missing':
        j = job
    elif case == 'unknown':
        j = dict(job, _ticket='T-' + '0' * 32)
    elif case == 'other_job':
        j = dict(job, code=CONST, _ticket=L.issue_ticket('a', job))
    else:
        t = L.issue_ticket('a', job)
        L.redeem_ticket(t, job)
        j = dict(job, _ticket=t)
    r = asyncio.run(sandbox.run_job(j, s))
    assert r['error'].startswith('ticket_refused') and not launched


def test_valid_ticket_redeems_once(tmp_path):
    from desk.company.ledger import TicketRefused
    L = Ledger(tmp_path / 'c.db')
    job = {'type': 'strategy_eval', 'code': BASE, 'split': 'dev'}
    t = L.issue_ticket('a', job)
    L.redeem_ticket(t, dict(job, _ticket=t, _store='/x'))           # host plumbing keys don't change the binding
    with pytest.raises(TicketRefused, match='already used'):
        L.redeem_ticket(t, job)
    with pytest.raises(sqlite3.DatabaseError):
        L.x('DELETE FROM job_ticket_uses')


def test_harness_entrypoint_refuses_unticketed_job(tmp_path, monkeypatch):
    import json, sys
    from desk.lab import harness
    jp, op = tmp_path / 'j.json', tmp_path / 'o.json'
    jp.write_text(json.dumps({'type': 'strategy_eval', 'code': BASE}))
    monkeypatch.setattr(sys, 'argv', ['h', str(jp), str(op)])
    harness.main()
    assert json.loads(op.read_text())['error'].startswith('ticket_refused')


# ------------------------------------------------------------------ ruling (1): dev DSR N from host-attested identities
def test_dev_n_counts_host_attested_identities_per_family(tmp_path):
    L = Ledger(tmp_path / 'n.db')
    L.trial('a', 'dev_backtest', 'self', 9.0, 100)                       # self-reported style: no host identity
    L.x("INSERT INTO trials(author,kind,code_hash,sharpe,days,created,family,identity) "
        "VALUES('a','dev_backtest','h',1,100,0,'f1','forged')")         # identity but not attested
    assert L.trial_stats()['distinct_strategies'] == 0
    for i in range(3):                                                   # every sandbox_run counts, even 1-day/no-sharpe
        L.trial('a', 'sandbox_run', f'c{i}', None, None, family='f1', identity=f'id{i}')
    L.trial('a', 'sandbox_run', 'c0', None, None, family='f1', identity='id0')   # same (identity, config): once
    L.trial('a', 'submission', 's', 1.0, 3, family='f2', identity='idX')
    assert L.trial_stats(family='f1')['distinct_strategies'] == 3
    assert L.trial_stats(family='f2')['distinct_strategies'] == 1
    assert L.trial_stats()['distinct_strategies'] == 4


def test_pipeline_dev_n_uses_attested_count(pipe):
    L = pipe.L
    for i in range(4):
        L.trial('a', 'sandbox_run', f'c{i}', None, None, family='unassigned', identity=f'id{i}')
    L.trial('a', 'dev_backtest', 'self', 2.0, 100)                       # does not count
    asyncio.run(pipe.process_async(_strategy(L, BASE, 'submitted')))
    assert pipe.jobs[-1]['n_trials'] == 4
    assert L.q("SELECT attested FROM trials WHERE kind='submission'")[0]['attested'] == 1


# ------------------------------------------------------------------ ruling (3): infra failure burns the token
@pytest.fixture
def infra_pipe(pipe, monkeypatch):
    async def failing_job(job, timeout=5400):
        pipe.jobs.append(job)
        return {'error': 'harness failed (memory)'}
    monkeypatch.setattr(pipe, '_job', failing_job)
    return pipe


def test_infra_failure_burns_token_without_attestation(infra_pipe):
    L = infra_pipe.L
    sid = _strategy(L, BASE, 'holdout')
    asyncio.run(infra_pipe.process_async(sid))
    assert len(infra_pipe.jobs) == 1                                     # no views table here: cannot attest pre-open
    assert L.q('SELECT pre_open FROM holdout_infra')[0]['pre_open'] == 0
    L.update_strategy(sid, status='holdout')                             # asking again does nothing
    asyncio.run(infra_pipe.process_async(sid))
    assert len(infra_pipe.jobs) == 1
    ident = L.q('SELECT identity FROM holdout_uses')[0]['identity']
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_retry(ident)


def test_retry_only_on_host_attested_pre_open_failure(tmp_path):
    L = Ledger(tmp_path / 'r.db')
    L.x('CREATE TABLE holdout_views(identity TEXT PRIMARY KEY, job_id TEXT, created REAL)')
    L.claim_holdout('pre', 'S-1'); L.claim_holdout('post', 'S-2')
    with pytest.raises(HoldoutAlreadyUsed):                              # never on request
        L.claim_holdout_retry('pre')
    L.x("INSERT INTO holdout_views VALUES('post','j',0)")                # data was opened for this one
    assert L.attest_holdout_infra_failure('pre', 'S-1', 'oom') is True
    assert L.attest_holdout_infra_failure('post', 'S-2', 'oom') is False
    assert L.attest_holdout_infra_failure('pre', 'S-1', 'again') is False   # one attestation per token
    L.claim_holdout_retry('pre')
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_retry('pre')                                     # exactly one retry
    with pytest.raises(HoldoutAlreadyUsed):
        L.claim_holdout_retry('post')
    with pytest.raises(sqlite3.DatabaseError):
        L.x('DELETE FROM holdout_infra')
