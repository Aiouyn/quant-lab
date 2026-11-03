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
