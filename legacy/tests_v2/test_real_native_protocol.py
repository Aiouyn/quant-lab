"""Declared actual-review runner with explicitly synthetic transport/data."""
import json
from dataclasses import replace
import pytest
import verify_real_native_review as protocol
from quantdesk import venue_study
from quantdesk.store import Store
from tests.test_agent_completion import runtime, response
from tests.test_scientific_decisions import call
from tests.test_mission_review import critique
from tests.test_venue_decisions import setup


@pytest.fixture(autouse=True)
def dead_capture(monkeypatch): monkeypatch.setattr(venue_study, 'process_alive', lambda pid: False)


@pytest.mark.parametrize('fail_at', [None, 'scientist', 'falsifier', 'historian'])
def test_protocol_is_bounded_idempotent_and_partial_never_completes(tmp_path, monkeypatch, fail_at):
    store, settings, _, _, _, _, _, replay, stop = setup(tmp_path)
    cid = store.campaign({'hash': 'protocol-fixture', 'development': []}, 750000)
    store.set_status(cid, 'STOPPED')
    monkeypatch.setattr(protocol, 'CID', cid)
    monkeypatch.setattr(protocol, 'REPLAY', replay['replay_id'])
    requested = []
    def create(**kwargs):
        task = json.loads(kwargs['messages'][1]['content'])
        role = 'historian' if task.get('required_mission_decision') else 'falsifier' if task.get('mission_stop') else 'scientist'
        requested.append(role)
        if role == fail_at: raise TimeoutError('Explicit synthetic provider timeout')
        if role == 'scientist': return response(call('stop_mission', stop))
        if role == 'falsifier': return response(content=json.dumps(critique(replay['replay_id'])))
        return response(call('record_mission_lesson', {'decision_id': task['required_mission_decision'],
            'lesson': 'Synthetic critic disagreement preserved; configuration evidence only',
            'next_experiment': 'Predeclare a matched comparison on other permitted development dates'}))
    def factory(s, cfg): return runtime(s, cfg, create)
    path = tmp_path / 'protocol.json'
    result = protocol.execute(settings, store, path, factory)
    assert result['status'] == ('FAILED' if fail_at else 'COMPLETE')
    assert result['protected_unchanged'] and result['orders_submitted_to_exchange'] == 0
    assert result['new_provider_requests'] == len(requested) <= 3
    if fail_at: assert result['remaining_original_tokens'] < 750000 - 100 * (len(requested) - 1)
    before = len(requested)
    assert protocol.execute(settings, store, path, factory)['new_request_started'] is False
    assert len(requested) == before
    with store.connect() as db:
        status = db.execute('SELECT status FROM cycles WHERE id=?', (result['cycle_id'],)).fetchone()[0]
        assert status == ('FAILED' if fail_at else 'DECISION_COMPLETE')
        assert db.execute('SELECT count(*) FROM leases').fetchone()[0] == 0
        assert store.get_campaign(cid)['token_limit'] == 750000 and store.get_campaign(cid)['status'] == 'STOPPED'


def test_exhausted_campaign_cannot_be_replenished_by_protocol(tmp_path, monkeypatch):
    store = Store(tmp_path)
    cid = store.campaign({}, 750000)
    monkeypatch.setattr(protocol, 'CID', cid)
    store.reserve(cid, 'operator', 750000)
    with pytest.raises(ValueError, match='No remaining original allowance'):
        protocol.execute(replace(protocol.Settings(), home=tmp_path), store, tmp_path / 'run.json')
    assert not (tmp_path / 'run.json').exists()
