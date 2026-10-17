"""Only a never-requested role can continue; prior real artifact scope persists."""
import json
import numpy as np
import polars as pl
import pytest
import verify_real_native_review as protocol
from quantdesk import models, venue_study
from quantdesk.mission_review import prepare
from quantdesk.store import Store
from tests.test_agent_completion import runtime, response
from tests.test_scientific_decisions import call
from tests.test_mission_review import critique
from tests.test_venue_decisions import setup


def test_threshold_and_signal_scale_cannot_activate_below_cost_forecasts(monkeypatch):
    frame = pl.DataFrame({'spread_bps': [0.1] * 5})
    monkeypatch.setattr(models, 'feature_matrix', lambda *args: np.zeros((5, 1)))
    # Fix forecasts, independently vary the signal scaling and decision threshold.
    # This tests the real numerical gate, not the wording in the prompt contract.
    monkeypatch.setattr(models, 'predict', lambda *args: np.full(5, 8.15))
    for scale in (0.1, 1, 10, 100):
        artifact = {'recipe': {'min_edge_bps': 0, 'prediction_scale_bps': scale},
            'fitted': {}, 'execution': {'taker_fee_bps': 4.5, 'slippage_bps': 0.5}}
        signal = models.fitted_signal(frame, np.zeros(5), artifact)
        assert np.all(signal == 0)
        assert not any(np.any(np.abs(signal) > threshold) for threshold in (0.01, 0.1, 0.5))


def test_only_unrequested_historian_finishes_without_repeating_prior_roles(tmp_path, monkeypatch):
    monkeypatch.setattr(venue_study, 'process_alive', lambda pid: False)
    store, settings, _, _, _, _, _, replay, stop = setup(tmp_path)
    cid = store.campaign({'hash': 'continuation-fixture', 'development': []}, 750000)
    monkeypatch.setattr(protocol, 'CID', cid)
    monkeypatch.setattr(protocol, 'REPLAY', replay['replay_id'])
    calls = []
    def create(**kwargs):
        task = json.loads(kwargs['messages'][1]['content'])
        role = 'historian' if task.get('required_mission_decision') else 'falsifier' if task.get('mission_stop') else 'scientist'
        calls.append(role)
        if role == 'scientist': return response(call('stop_mission', stop))
        if role == 'falsifier': return response(content=json.dumps(critique(replay['replay_id'])))
        assert task['critique'] == critique(replay['replay_id'])
        assert task['verified_venue_evidence'][0]['current_tool_constraints']['native_replay_supported_modes'] == ['taker']
        assert {t['function']['name'] for t in kwargs['tools']} == {'record_mission_lesson'}
        return response(call('record_mission_lesson', {'decision_id': task['required_mission_decision'],
            'lesson': 'Full actual critic and supported mode constraints preserved', 'next_experiment': 'A predeclared supported taker comparison'}))
    def factory(s, cfg): return runtime(s, cfg, create)
    original_reserve = Store.reserve
    def blocked(self, campaign, role, tokens, task_id=None):
        if role == 'historian': raise RuntimeError('Synthetic pre-request reservation failure')
        return original_reserve(self, campaign, role, tokens, task_id=task_id)
    path = tmp_path / 'real-run.json'
    with monkeypatch.context() as patch:
        patch.setattr(Store, 'reserve', blocked)
        initial = protocol.execute(settings, store, path, factory)
    assert initial['status'] == 'FAILED' and calls == ['scientist', 'falsifier']
    result = protocol.finish_unrequested_historian(settings, store, path, factory)
    assert result['status'] == 'COMPLETE_WITH_INTERVENTION' and calls == ['scientist', 'falsifier', 'historian']
    assert result['new_provider_requests'] == 3 and result['protected_unchanged']
    assert result['continuation']['new_provider_requests'] == 1
    assert json.loads((tmp_path / 'real-run-initial-reservation-failure.json').read_text())['status'] == 'FAILED'
    assert protocol.finish_unrequested_historian(settings, store, path, factory)['new_request_started'] is False
    with store.connect() as db:
        assert db.execute('SELECT token_limit FROM task_budgets WHERE task_id=?', (result['cycle_id'] + ':historian:mission',)).fetchone()[0] == 10000


def test_already_requested_historian_cannot_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(venue_study, 'process_alive', lambda pid: False)
    store, settings, _, _, _, _, _, replay, stop = setup(tmp_path)
    cid = store.campaign({'hash': 'requested-fixture', 'development': []}, 750000)
    monkeypatch.setattr(protocol, 'CID', cid)
    monkeypatch.setattr(protocol, 'REPLAY', replay['replay_id'])
    def create(**kwargs):
        task = json.loads(kwargs['messages'][1]['content'])
        if task.get('required_mission_decision'): raise TimeoutError('Synthetic already charged request')
        if task.get('mission_stop'): return response(content=json.dumps(critique(replay['replay_id'])))
        return response(call('stop_mission', stop))
    path = tmp_path / 'real-run.json'
    protocol.execute(settings, store, path, lambda s, cfg: runtime(s, cfg, create))
    with pytest.raises(ValueError, match='zero historian requests'):
        protocol.finish_unrequested_historian(settings, store, path)
