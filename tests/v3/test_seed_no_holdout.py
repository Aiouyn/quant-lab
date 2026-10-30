"""Cycle 1b item 5: ensure_seeded() must never consume a holdout shot (or touch holdout data). Synthetic only."""
import dataclasses

import pytest

from desk.company import rsi
from desk.company.ledger import Ledger
from desk.config import SETTINGS
from desk.lab import holdout
from desk.lab.holdout import HoldoutLocked, holdout_forbidden
from desk.lab.strategy import code_hash
from synth_desk import TREND

FORWARD = ('2026-11-02', '2027-04-30')


def _passing(split):
    return {'summary': {'sharpe': 3.0, 'trades': 500, 'max_drawdown': 0.05, 'pnl': 1.0, 'days': 100,
                        'mean_daily_bps': 5.0},
            'stats': {'dsr': 0.99, 'bootstrap_lower_mean_bps': 5.0, 'positive_weeks': 0.8, 'top5_day_share': 0.1,
                      'mean_ex_top5_days_bps': 3.0},
            'stress': {'pnl': 1.0}, 'causality': {'ok': True}, 'code_hash': 'x', 'split': split,
            'overfit': {'pbo': 0.1, 'cpcv': {'oos_sharpe_median': 1.0}}, 'daily': {'day': [1, 2], 'ret': [0.01, 0.01]}}


@pytest.fixture
def open_forward_world(monkeypatch):
    """Worst case for the seeder: a forward holdout window is OPEN and every candidate clears dev/val."""
    import desk.data.store_builder as sb
    import desk.pipeline.validation as pv
    holdout_evals = []

    def fake_eval(code, sid, split, *a, **k):
        if split == 'holdout':
            holdout.assert_holdout_allowed(sid)          # the real function's first check (structural guard)
            holdout_evals.append(sid)
        return _passing(split)
    monkeypatch.setattr(rsi, '_evaluate_truth_machine_split', fake_eval)
    monkeypatch.setattr(sb, 'ensure_store_panels', lambda *a, **k: {})
    monkeypatch.setattr(pv, 'check', lambda gates, r: [])
    monkeypatch.setattr(holdout, 'desk_holdout_window', lambda s, d: FORWARD)
    return holdout_evals


def _candidate(L):
    sid = L.add_strategy('trend', 'a', 'macro', TREND, code_hash(TREND), '1d', 't')
    L.update_strategy(sid, status='review')
    return sid


def test_control_rsi_gate_spends_a_shot_when_allowed(tmp_path, open_forward_world):
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = Ledger(tmp_path / 'l.sqlite3')
    _candidate(L)
    rsi.validate_and_promote_population(L, settings=s, desk='macro', max_candidates=2)
    assert L.q('SELECT COUNT(*) n FROM holdout_shots')[0]['n'] == 1 and open_forward_world


def test_ensure_seeded_never_consumes_a_holdout_shot(tmp_path, open_forward_world, monkeypatch):
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = Ledger(tmp_path / 'l.sqlite3')
    _candidate(L)
    before = L.q('SELECT COUNT(*) n FROM holdout_shots')[0]['n']
    calls = []
    real = rsi.validate_and_promote_population

    def spy(*a, **k):
        calls.append(holdout.holdout_forbidden_reason())
        return real(*a, **k)
    monkeypatch.setattr(rsi, 'validate_and_promote_population', spy)
    rsi.ensure_seeded(s, L)
    assert calls and set(calls) == {'ensure_seeded'}       # the seeder DID reach the promotion gate ...
    assert open_forward_world == []                        # ... but never evaluated holdout data
    assert L.q('SELECT COUNT(*) n FROM holdout_shots')[0]['n'] == before
    assert L.q("SELECT COUNT(*) n FROM events WHERE kind='HOLDOUT_SKIPPED'")[0]['n'] >= 1


def test_no_holdout_scope_blocks_every_entry_point(tmp_path):
    L = Ledger(tmp_path / 'l.sqlite3')
    with holdout_forbidden('ensure_seeded'):
        with pytest.raises(HoldoutLocked):
            L.claim_holdout_shot('S-x', TREND, code_hash(TREND), FORWARD)
        with pytest.raises(HoldoutLocked):
            holdout.assert_holdout_open('macro', FORWARD)
        with pytest.raises(HoldoutLocked):
            rsi._evaluate_truth_machine_split(TREND, 'S-x', 'holdout', SETTINGS)
    assert L.q('SELECT COUNT(*) n FROM holdout_shots')[0]['n'] == 0
    holdout.assert_holdout_open('macro', FORWARD)          # outside the scope the forward window is open again
