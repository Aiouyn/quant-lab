"""Phase 3: a change to how the firm works goes live only when a paired drill trial shows it helps."""
from __future__ import annotations

import dataclasses
import json

import pytest

from desk.lab import process_trials as PT


# ------------------------------------------------------------------------------------------------ decision rule
def test_decide_accepts_only_consistent_lift_with_clean_nulls():
    good = [{'base': 0.1, 'cand': 0.5}, {'base': 0.0, 'cand': 0.4}, {'base': 0.3, 'cand': 0.3}]
    assert PT.decide(good, [0, 0, 0])['verdict'] == 'ACCEPT'
    assert PT.decide(good, [0, 1, 0])['verdict'] == 'REJECT'                     # a false discovery vetoes
    one_lucky = [{'base': 0.0, 'cand': 0.9}, {'base': 0.3, 'cand': 0.0}, {'base': 0.2, 'cand': 0.2}]
    assert PT.decide(one_lucky, [0, 0, 0])['verdict'] == 'REJECT'                # big mean, one win: not enough
    tiny = [{'base': 0.30, 'cand': 0.33}] * 3
    assert PT.decide(tiny, [0, 0, 0])['verdict'] == 'REJECT'                     # wins, but below the minimum lift
    assert PT.decide(good[:2] + [{'base': None, 'cand': 0.5}], [0, 0, 0])['verdict'] == 'ERROR'
    assert PT.decide(good, [0, None, 0])['verdict'] == 'ERROR'


# ------------------------------------------------------------------------------------------------ variants
def _ledger(tmp_path, name='desk.sqlite3'):
    from desk.company.ledger import Ledger
    return Ledger(tmp_path / name)


def test_proposals_do_not_reach_charters_until_accepted(tmp_path):
    from desk.company.roles import charter
    L = _ledger(tmp_path)
    assert 'error' in PT.propose(L, 'wizard', 'x', 'y', 'a1')
    assert 'error' in PT.propose(L, 'researcher', 'x' * (PT.MAX_PATCH_CHARS + 1), 'y', 'a1')
    assert 'error' in PT.propose(L, 'researcher', 'rule', '', 'a1')
    vid = PT.propose(L, 'researcher', 'Always split dev in halves.', 'drill PT-x', 'a1', pod='xsec')['id']
    assert 'split dev in halves' not in charter('researcher', 'xsec', ledger=L)
    L.x("UPDATE process_variants SET status='active' WHERE id=?", (vid,))         # flipped by hand, no trial
    assert 'split dev in halves' not in charter('researcher', 'xsec', ledger=L)
    L.x("INSERT INTO process_trials(id,variant_id,verdict) VALUES('PT-1',?, 'ACCEPT')", (vid,))
    assert 'split dev in halves' in charter('researcher', 'xsec', ledger=L)
    assert 'split dev in halves' not in charter('researcher', 'macro', ledger=L)  # pod-scoped
    for _ in range(PT.MAX_OPEN_PER_AUTHOR):
        PT.propose(L, 'officer', 'r', 'e', 'spammer')
    assert 'error' in PT.propose(L, 'officer', 'r', 'e', 'spammer')


def test_retired_engine_charter_versions_are_ignored(tmp_path):
    from desk.company.roles import charter
    L = _ledger(tmp_path)
    L.add_charter_version('researcher', 'OLD ENGINE OVERLAY', 'synthetic gate', status='promoted')
    L.add_charter_version('researcher', 'OLD ENGINE OVERLAY', 'synthetic gate', status='active')
    assert 'OLD ENGINE' not in charter('researcher', 'xsec', ledger=L)


def test_agents_get_propose_and_history_tools():
    from desk.company.tools import ROLE_TOOLS
    assert 'propose_process_change' in ROLE_TOOLS['lead'] and 'process_history' in ROLE_TOOLS['cio']
    assert 'propose_process_change' not in ROLE_TOOLS['researcher']


# ------------------------------------------------------------------------------------------------ full trial
def _settings(tmp_path):
    from desk.config import SETTINGS
    from desk.lab.perps import PerpsStore
    from tests.v3.test_panel_drills import perps_panel
    s = dataclasses.replace(SETTINGS, home=tmp_path, store_dir=None)
    root = s.store / 'perps'
    root.mkdir(parents=True)
    perps_panel(n_coins=4).write_parquet(root / 'panel.parquet')
    PerpsStore(s).splits()
    return s


ORACLE = """def signal(panel, p):
    x = pl.col(F).cast(pl.Float64)
    if T == 'log_change':
        x = (x.abs() + 1e-9).log() - (x.abs() + 1e-9).log().shift(W).over('coin')
    f = x.rolling_mean(W, min_samples=max(2, W // 2)).over('coin') if T == 'level' else x
    q = panel.with_columns(f.alias('_f'))
    z = ((pl.col('_f') - pl.col('_f').mean().over('ts')) / (pl.col('_f').std().over('ts') + 1e-12))
    return q.select((z.fill_null(0).fill_nan(0).tanh() * SIGN).alias('s'))['s']
"""


def _fake_firm(approve_on_null=False, seen=None):
    """A stand-in for run_company: the firm finds the planted signal only when the candidate rule is in its charter."""
    from desk.company.ledger import Ledger
    from desk.company.roles import charter

    def runner(s, drill, hours, agents):
        home = s.home / 'drills' / drill
        L = Ledger(home / 'desk.sqlite3')
        secret = json.loads((home / 'drill_secret.json').read_text())
        if seen is not None:
            seen.append((s.home.name, drill, secret.get('feature'), secret.get('window')))
        helped = 'check every feature' in charter('researcher', 'xsec', ledger=L)
        if drill.endswith('null'):
            if approve_on_null and helped:
                code = "DESK='perps'\nPARAMS={}\ndef signal(panel, p):\n    return panel['premium'].fill_null(0) * 0 + 1\n"
                L.add_strategy('fluke', 'r1', 'perps', code, 'h0', '1h', 't')
                L.update_strategy(L.q('SELECT id FROM strategies')[0]['id'], status='approved')
            return 0
        if helped:                                           # an oracle that rebuilds the hidden score exactly
            code = (f"DESK='perps'\nPARAMS={{}}\nF, W, T, SIGN = {secret['feature']!r}, {secret['window']}, "
                    f"{secret['transform']!r}, {secret['sign']}\n" + ORACLE)
            L.add_strategy('oracle', 'r1', 'perps', code, 'h1', '1h', 't')
            L.update_strategy(L.q('SELECT id FROM strategies')[0]['id'], status='approved')
        return 0
    return runner


def test_trial_promotes_a_rule_that_helps_and_rejects_one_that_overfits(tmp_path):
    from desk.company.roles import charter
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    vid = PT.propose(L, 'researcher', 'Rank-test check every feature cross-sectionally before modelling.',
                     'planted drills were missed', 'perps-lead', pod='xsec')['id']
    seen = []
    res = PT.run_trial(s, vid, desk='perps', pairs=2, hours=0.0, agents=1, runner=_fake_firm(seen=seen))
    assert res['verdict'] == 'ACCEPT', res
    assert all(p['cand'] > 0.9 and p['base'] == 0 for p in res['pair_scores'])
    by_pair = {}
    for home, drill, feat, win in seen:                      # both arms of a pair face the same hidden signal
        if drill.endswith('planted'):
            by_pair.setdefault(home.split('-')[1], set()).add((feat, win))
    assert all(len(v) == 1 for v in by_pair.values())
    assert 'check every feature' in charter('researcher', 'xsec', ledger=L)
    assert L.q('SELECT status FROM process_variants WHERE id=?', (vid,))[0]['status'] == 'active'

    # a rule that also "finds" things on a market with nothing to find is rejected, even with a planted lift
    bad = PT.propose(L, 'researcher', 'Submit whatever looks good and check every feature.', 'more output',
                     'perps-r1', pod='xsec')['id']
    res2 = PT.run_trial(s, bad, desk='perps', pairs=2, hours=0.0, agents=1, runner=_fake_firm(approve_on_null=True))
    assert res2['verdict'] == 'REJECT' and res2['null_false_discoveries'] > 0
    assert 'Submit whatever' not in charter('researcher', 'xsec', ledger=L)
    assert 'check every feature cross-sectionally' in charter('researcher', 'xsec', ledger=L)   # still the live one

    hist = PT.history(L)
    assert {h['verdict'] for h in hist} == {'ACCEPT', 'REJECT'}


def test_failed_run_is_an_error_and_promotes_nothing(tmp_path):
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    vid = PT.propose(L, 'researcher', 'rule', 'why', 'a1')['id']

    def broken(*_):
        raise RuntimeError('no LLM key')
    res = PT.run_trial(s, vid, desk='perps', pairs=1, hours=0.0, agents=1, runner=broken)
    assert res['verdict'] == 'ERROR'
    assert L.q('SELECT status FROM process_variants WHERE id=?', (vid,))[0]['status'] == 'proposed'
    assert not PT.live_variants(L)


# ------------------------------------------------------------------------------------------------ research tools
def _tool_firm():
    """The firm finds the planted signal only when its brief announces the 'xsec_rank' primitive."""
    from desk.company.ledger import Ledger
    inner = _fake_firm()

    def runner(s, drill, hours, agents):
        L = Ledger(s.home / 'drills' / drill / 'desk.sqlite3')
        if 'xsec_rank' in PT.primitives_text(L) and not drill.endswith('null'):
            L.x("INSERT INTO process_variants(id,role,pod,patch_text,rationale,author,status,created) "
                "VALUES('PV-shim','researcher','xsec','check every feature','t','t','arm',0)")
        return inner(s, drill, hours, agents)
    return runner


def test_new_research_tools_are_announced_only_after_a_trial(tmp_path):
    from desk.company.roles import charter
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    assert 'error' in PT.propose_primitive(L, 'xsec_rank', '', '', 'toolsmith')
    vid = PT.propose_primitive(L, 'xsec_rank', 'cross-sectional rank IC per feature', 'leads rank by hand', 'toolsmith')['id']
    assert PT.primitives_text(L) == ''                                  # installed, not announced
    res = PT.run_trial(s, vid, desk='perps', pairs=2, hours=0.0, agents=1, runner=_tool_firm())
    assert res['verdict'] == 'ACCEPT', res
    assert 'desk.lib.contrib.xsec_rank' in PT.primitives_text(L)
    assert 'xsec_rank' not in charter('researcher', 'xsec', ledger=L)   # a tool is not a charter rule


def test_contribute_primitive_registers_a_variant_instead_of_announcing(tmp_path, monkeypatch):
    import asyncio
    import types
    from pathlib import Path
    from desk.company import tools as T
    L = _ledger(tmp_path)
    contrib = Path(T.__file__).resolve().parent.parent / 'lib' / 'contrib'
    idx = contrib / 'INDEX.md'
    before = idx.read_text(encoding='utf-8') if idx.exists() else None
    sent = []
    monkeypatch.setattr(L, 'send', lambda *a, **k: sent.append(a), raising=False)
    tb = T.Toolbox.__new__(T.Toolbox)
    tb.L, tb.agent = L, types.SimpleNamespace(name='toolsmith', role='toolsmith', workspace=tmp_path)
    tb.s, tb.org = None, types.SimpleNamespace(sandbox_slots=None)

    async def fake_run(*a, **k):
        return {'status': 'ok', 'stdout': 'OK'}
    monkeypatch.setattr(T, 'run_python', fake_run)
    target = contrib / 'zz_trial_probe.py'
    try:
        res = asyncio.run(tb.t_contribute_primitive('zz_trial_probe', 'def f():\n    return 1\n', 'print("OK")',
                                                    'returns one'))
        assert res.get('ok') and res['variant'].startswith('PV-'), res
        assert sent == [] and (idx.read_text(encoding='utf-8') if idx.exists() else None) == before
        assert L.q('SELECT role, pod, status FROM process_variants')[0] == \
            {'role': 'primitive', 'pod': 'zz_trial_probe', 'status': 'proposed'}
    finally:
        target.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ speed
def test_early_stop_only_when_accept_is_impossible():
    win, loss = {'base': 0.0, 'cand': 0.6}, {'base': 0.5, 'cand': 0.0}
    assert PT.futile([win], [0], 3) is None
    assert PT.futile([loss], [0], 3) is None                         # 2 wins still possible
    assert PT.futile([loss, loss], [0, 0], 3)                         # cannot win 2 of 3
    assert PT.futile([win], [1], 3)                                   # a false discovery always rejects
    assert PT.futile([{'base': None, 'cand': 0.5}], [0], 3) is None   # failures finish and end in ERROR
    # whenever futile() fires, every possible completion is a REJECT
    import itertools
    grid = [{'base': b, 'cand': c} for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)]
    for done in itertools.product(grid, repeat=1):
        for nul in ([0], [1]):
            if PT.futile(list(done), nul, 3):
                for rest in itertools.product(grid, repeat=2):
                    assert PT.decide(list(done) + list(rest), nul + [0, 0])['verdict'] == 'REJECT'


def test_trial_runs_pair_arms_together_and_stops_a_hopeless_trial(tmp_path):
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    bad = PT.propose(L, 'researcher', 'Submit whatever looks good and check every feature.', 'x', 'r1', pod='xsec')['id']
    res = PT.run_trial(s, bad, desk='perps', pairs=3, hours=0.0, agents=1, runner=_fake_firm(approve_on_null=True))
    assert res['verdict'] == 'REJECT' and 'stopped early' in res['reason'] and len(res['pair_scores']) == 1
    runs = json.loads(L.q('SELECT results FROM process_trials')[0]['results'])['runs']
    assert sorted((r['arm'], r['kind']) for r in runs) == [('base', 'planted'), ('cand', 'null'), ('cand', 'planted')]


def test_planted_score_gives_half_credit_for_found_but_not_approved():
    rep = lambda c, best: {'planted_signal_corr': c, 'best_corr_with_hidden': best}
    assert PT._score(rep(0.6, [(0.6, 's1', 'approved')]), 'planted') == 0.6
    assert PT._score(rep(0.0, [(0.5, 's1', 'rejected_holdout'), (0.2, 's2', 'dev')]), 'planted') == 0.25
    assert PT._score(rep(0.0, [(0.05, 's1', 'dev')]), 'planted') == 0.0          # noise is not a discovery
    assert PT._score(rep(0.0, []), 'planted') == 0.0
    assert PT._score(rep(None, []), 'planted') is None
    assert PT._score({'false_discoveries': 0}, 'null') == 0                   # the null veto still counts approvals


def test_registered_factors_earn_a_quarter_credit():
    rep = {'planted_signal_corr': 0.0, 'best_corr_with_hidden': [], 'best_factor_corr_with_hidden': [(0.8, 'F-1', 'active')]}
    assert PT._score(rep, 'planted') == 0.2
    rep['best_corr_with_hidden'] = [(0.6, 'S-1', 'revise')]
    assert PT._score(rep, 'planted') == 0.3                                       # a submitted strategy counts more
    rep['best_factor_corr_with_hidden'] = [(0.05, 'F-2', 'weak')]
    rep['best_corr_with_hidden'] = []
    assert PT._score(rep, 'planted') == 0.0


def test_proposer_briefs_show_the_process_pipeline_and_nudge_when_empty(tmp_path):
    from desk.company.runtime import process_pipeline_text
    L = _ledger(tmp_path)
    assert 'Nothing is queued' in process_pipeline_text(L)               # nothing ever proposed: nudge
    vid = PT.propose(L, 'researcher', 'Split dev in halves.', 'drills', 'a1')['id']
    txt = process_pipeline_text(L)
    assert vid in txt and 'Nothing is queued' not in txt
    L.x("UPDATE process_variants SET status='rejected' WHERE id=?", (vid,))
    assert 'Nothing is queued' in process_pipeline_text(L)


def test_briefs_show_where_drills_lost_the_planted_signal(tmp_path):
    from desk.company.runtime import process_pipeline_text
    L = _ledger(tmp_path)
    runs = [{'kind': 'planted', 'score': 0.17, 'approved': 0, 'best_corr': [[0.33, 'S-1', 'rejected_review']]},
            {'kind': 'planted', 'score': 0.4, 'approved': 1, 'best_corr': [[0.4, 'S-2', 'approved']]},
            {'kind': 'planted', 'score': 0.1, 'approved': 0, 'best_corr': [], 'best_factor_corr': [[0.46, 'F-1', 'active']]},
            {'kind': 'null', 'score': 0}]
    L.x("INSERT INTO process_trials(id,variant_id,verdict,results,created) VALUES('PT-1','PV-x','REJECT',?,1)",
        (json.dumps({'runs': runs}),))
    txt = process_pipeline_text(L)
    assert '3 planted runs' in txt and 'approved in 1' in txt
    assert 'rejected_review: 1' in txt and 'factor only, never submitted: 1' in txt


def test_cio_gets_a_mandatory_proposal_objective_when_the_queue_is_dry(tmp_path):
    import time as _t
    from desk.company.runtime import proposal_due
    L = _ledger(tmp_path)
    assert proposal_due(L)                                                # never proposed
    vid = PT.propose(L, 'lead', 'rule', 'why', 'cio')['id']
    assert not proposal_due(L)                                            # queued
    L.x("UPDATE process_variants SET status='rejected' WHERE id=?", (vid,))
    assert not proposal_due(L)                                            # proposed recently
    L.x("UPDATE process_variants SET created=? WHERE id=?", (_t.time() - 13 * 3600, vid))
    assert proposal_due(L)


def test_trial_loop_requeues_interrupted_trials_and_runs_the_queue(tmp_path):
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    vid = PT.propose(L, 'researcher', 'Rank-test check every feature cross-sectionally.', 'r', 'lead', pod='xsec')['id']
    L.x("UPDATE process_variants SET status='testing' WHERE id=?", (vid,))
    L.x("INSERT INTO process_trials(id,variant_id,verdict,created) VALUES('PT-dead',?,'RUNNING',1)", (vid,))
    calls = {'n': 0}

    def stop():
        calls['n'] += 1
        return calls['n'] > 1                                             # one pass through the loop
    logs = []
    PT.trial_loop(s, pairs=1, hours=0.0, agents=1, idle_s=0, stop=stop, log=logs.append, runner=_fake_firm())
    assert L.q("SELECT verdict FROM process_trials WHERE id='PT-dead'")[0]['verdict'] == 'ABORTED'
    assert any('ABORTED' in m for m in logs) and any('trial start' in m for m in logs)
    assert L.q('SELECT status FROM process_variants WHERE id=?', (vid,))[0]['status'] in ('active', 'rejected')


def test_prune_drops_market_copies_but_keeps_ledgers_and_secrets(tmp_path):
    import os
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    L.x("INSERT INTO process_trials(id,variant_id,verdict,created) VALUES('PT-done','PV-x','REJECT',1)")
    L.x("INSERT INTO process_trials(id,variant_id,verdict,created) VALUES('PT-live','PV-y','RUNNING',1)")
    real = tmp_path / 'real_store'
    (real / 'perps').mkdir(parents=True)
    (real / 'perps' / 'keep.parquet').write_bytes(b'x' * 10)
    for tid in ('PT-done', 'PT-live'):
        run = tmp_path / 'trials' / tid / 'base-0' / 'drills' / 'base_planted'
        for sub in ('store/perps', 'artifacts', 'workspaces/r1'):
            (run / sub).mkdir(parents=True)
        (run / 'store' / 'perps' / 'panel.parquet').write_bytes(b'x' * 1000)
        (run / 'desk.sqlite3').write_bytes(b'db')
        (run / 'drill_secret.json').write_text('{}')
    linked = tmp_path / 'trials' / 'PT-done' / 'cand-0' / 'drills' / 'cand_null'
    linked.mkdir(parents=True)
    try:
        os.symlink(real, linked / 'store', target_is_directory=True)
    except OSError:
        pass
    out = PT.prune_finished(s, L)
    done = tmp_path / 'trials' / 'PT-done' / 'base-0' / 'drills' / 'base_planted'
    assert out['runs'] == 2 and not (done / 'store').exists() and not (done / 'workspaces').exists()
    assert (done / 'desk.sqlite3').exists() and (done / 'drill_secret.json').exists()
    assert (tmp_path / 'trials' / 'PT-live' / 'base-0' / 'drills' / 'base_planted' / 'store').exists()
    assert (real / 'perps' / 'keep.parquet').exists()                   # a linked store is never followed


def test_disk_guard_is_per_drive_and_configurable(tmp_path, monkeypatch):
    from desk.ops import supervisor as S
    monkeypatch.setattr(S, 'HOME', tmp_path)
    assert S.disk_guard_cfg()['D:\\'][0] < S.disk_guard_cfg()['C:\\'][0]
    (tmp_path / 'supervisor.json').write_text(json.dumps({'disk_guard': {'D:\\': [0.5, 1]}}))
    assert S.disk_guard_cfg() == {'D:\\': (0.5, 1.0)}


def test_history_shows_each_variant_once_with_its_latest_verdict(tmp_path):
    L = _ledger(tmp_path)
    vid = PT.propose(L, 'researcher', 'Rank-test check every feature cross-sectionally.', 'r', 'lead', pod='xsec')['id']
    for i, v in enumerate(('ABORTED', 'REJECT', 'ACCEPT')):
        L.x("INSERT INTO process_trials(id,variant_id,verdict,created) VALUES(?,?,?,?)", (f'PT-{i}', vid, v, i + 1))
    h = [r for r in PT.history(L) if r['id'] == vid]
    assert len(h) == 1 and h[0]['verdict'] == 'ACCEPT' and h[0]['trials'] == 3


def test_operator_directive_leads_every_brief_and_the_firm_never_self_freezes(tmp_path):
    import time as _t
    from types import SimpleNamespace
    from desk.company.roles import charter
    from desk.company.runtime import Org
    L = _ledger(tmp_path)
    L.memo('cio', 'desk', 'Stop all new theses until the operator decides.')
    L.memo('operator', 'operator', 'Continue: focus on the macro data.')
    org = SimpleNamespace(ledger=L, started=_t.time(), deadline=_t.time() + 3600, settings=SimpleNamespace(home=tmp_path),
                          cost_params=lambda: dict(taker_bps=4.5, slippage_bps=0.5, impact_mult=1, latency_s=1))
    agent = SimpleNamespace(name='xsec-r1', role='researcher', pod='xsec', workspace=tmp_path)
    txt = Org.brief(org, agent)
    assert txt.index('OPERATOR DIRECTIVE') < txt.index('DESK MEMO') and 'macro data' in txt
    assert 'never pauses itself' in charter('researcher', 'xsec')


def test_cio_can_only_staff_agents_that_actually_run(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from desk.company.tools import Toolbox
    L = _ledger(tmp_path)
    for n, r, p in (('cio', 'cio', 'exec'), ('xsec-r1', 'researcher', 'xsec'), ('xsec-r2', 'researcher', 'xsec')):
        L.upsert_agent(n, r, p)
    live = [SimpleNamespace(name='cio'), SimpleNamespace(name='xsec-r1')]   # xsec-r2 is a leftover of a bigger run
    org = SimpleNamespace(ledger=L, settings=None, agents=live)
    tb = Toolbox(org, SimpleNamespace(name='cio', role='cio', pod='exec'))
    out = asyncio.run(tb.t_reassign_agent('xsec-r2', 'positioning', 'untested'))
    assert 'not running' in out['error'] and 'xsec-r1' in out['error']
    assert 'error' in asyncio.run(tb.t_board_post('thesis', 't', 'b', assignee='positioning-lead'))
    assert asyncio.run(tb.t_reassign_agent('xsec-r1', 'positioning', 'untested')) == {'ok': True}
    moved = L.q("SELECT payload FROM events WHERE kind='REASSIGN' AND payload LIKE ?", ('%"agent": "xsec-r1"%',))
    assert moved and json.loads(moved[0]['payload'])['pod'] == 'positioning'          # what Org restores on restart


def test_trial_loop_backs_off_after_errors_and_waits_for_a_readable_store(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    L = _ledger(tmp_path)
    vid = PT.propose(L, 'researcher', 'Rank-test check every feature cross-sectionally.', 'r', 'lead', pod='xsec')['id']
    sleeps, calls = [], {'n': 0}
    monkeypatch.setattr(PT.time, 'sleep', lambda x: sleeps.append(x))

    def broken(s_, drill, hours, agents):
        raise OSError(22, 'Invalid argument')

    def stop():
        calls['n'] += 1
        return calls['n'] > 3
    PT.trial_loop(s, pairs=1, hours=0.0, agents=1, idle_s=0, stop=stop, log=lambda m: None, runner=broken)
    assert L.q("SELECT COUNT(*) n FROM process_trials")[0]['n'] == 3              # one trial per pass, never a spin
    assert sleeps == [900, 1800, 3600]                                            # backoff doubles
    assert len(L.q('SELECT id FROM process_trials')[0]['id']) == len('PT-') + 10
    assert PT.purge_empty_errors(L) == 3 and not L.q('SELECT id FROM process_trials')
    L.x("UPDATE process_variants SET status='testing' WHERE id=?", (vid,))       # stuck without a running trial
    PT.recover_stale(L)
    assert PT.next_proposed(L) == vid
    monkeypatch.setattr(PT, 'store_problem', lambda *a: 'D:/store/splits.json: OSError(22)')
    calls['n'] = 0
    PT.trial_loop(s, pairs=1, hours=0.0, agents=1, idle_s=0, stop=stop, log=lambda m: None, runner=broken)
    assert not L.q('SELECT id FROM process_trials')                              # nothing started on a bad store


def test_supervisor_holds_store_users_when_the_store_is_unreadable(tmp_path, monkeypatch):
    from desk.ops import supervisor as S
    monkeypatch.setattr(S, 'HOME', tmp_path)
    (tmp_path / 'store').mkdir()
    (tmp_path / 'store' / 'splits.json').write_text('{}')
    assert S.store_problem() is None
    real_open = open

    def failing(path, *a, **k):
        if str(path).endswith('splits.json'):
            raise OSError(22, 'Invalid argument')
        return real_open(path, *a, **k)
    monkeypatch.setattr('builtins.open', failing)
    assert 'Invalid argument' in S.store_problem()


def test_a_store_link_to_a_dead_drive_is_unreadable_but_does_not_kill_other_services(tmp_path, monkeypatch):
    import os
    from desk.config import SETTINGS
    from desk.ops import supervisor as S
    os.symlink(tmp_path / 'dead_drive', tmp_path / 'store', target_is_directory=True)
    s = dataclasses.replace(SETTINGS, home=tmp_path, store_dir=None)
    s.ensure()                                                     # recorders and dashboard still start
    assert (tmp_path / 'workspaces').is_dir()
    monkeypatch.setattr(S, 'HOME', tmp_path)
    assert 'missing drive' in S.store_problem()
    assert 'missing drive' in PT.store_problem(s)
