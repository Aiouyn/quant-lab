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
