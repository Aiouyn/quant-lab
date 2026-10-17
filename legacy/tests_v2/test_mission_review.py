"""Real native services/runtime, synthetic data and explicit provider responses."""
import json
from dataclasses import replace
import pytest
from quantdesk import mission_review, venue_study
from quantdesk.agents import ResearchTools, task_system_prompt
from quantdesk.research_map import research_map
from tests.test_agent_completion import runtime, response
from tests.test_scientific_decisions import call
from tests.test_venue_decisions import setup


@pytest.fixture(autouse=True)
def dead_capture(monkeypatch):
    monkeypatch.setattr(venue_study, 'process_alive', lambda pid: False)


def critique(ident):
    return {'verdict': 'disagree', 'evidence': [{'replay_id': ident,
            'observation': 'Measured fixture costs apply to this configuration only'}],
            'counterarguments': 'A configuration loss does not disprove a family',
            'limitations': 'Synthetic retrospective evidence; no simulator certification',
            'next_experiment': 'Predeclare a cash comparison on other permitted development data'}


def review_setup(tmp_path):
    store, settings, _, candidate, _, _, _, replay, stop = setup(tmp_path)
    cid = store.campaign({'hash': 'synthetic-provider-review', 'development': []}, 100000)
    cycle = store.start_cycle(cid)
    tools = ResearchTools(cid, store, None, None, settings, cycle)
    return store, settings, cid, candidate, cycle, cycle + ':scientist', tools, replay, stop


def test_complete_native_review_with_disagreement_checkpoint_and_retrieval(tmp_path):
    store, settings, cid, _, cycle, scientist, tools, replay, stop = review_setup(tmp_path)
    settings = replace(settings, max_agent_steps=1, max_completion_tokens=1000)
    requests = []
    review = critique(replay['replay_id'])
    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        task = json.loads(kwargs['messages'][1]['content'])
        assert task['venue_review'] and task['verified_venue_evidence'][0]['replay_id'] == replay['replay_id']
        assert 'NUMERICAL TOOL CONTRACT' not in kwargs['messages'][0]['content']
        assert kwargs['extra_body']['thinking']['type'] == 'enabled'
        if task.get('required_mission_decision'):
            assert task['critique'] == review
            return response(call('record_mission_lesson', {'decision_id': task['required_mission_decision'],
                'lesson': 'Critic disagrees with family-level rejection; observed costs only',
                'next_experiment': review['next_experiment']}))
        if task.get('mission_stop'):
            assert kwargs['tools'] == [] and kwargs['tool_choice'] == 'none'
            return response(content=json.dumps(review))
        assert [t['function']['name'] for t in kwargs['tools']] == ['stop_mission']
        params = kwargs['tools'][0]['function']['parameters']
        assert params['properties']['basis']['enum'] == ['MEASURED_VENUE']
        assert 'proposed_spec' not in params['properties']
        return response(call('stop_mission', stop))
    agent = runtime(store, settings, create)
    task = {'venue_review': True, 'proposal_workflow': True, 'replay_ids': [replay['replay_id']],
            'verified_venue_evidence': [{'pnl': 999999}], 'task_token_limit': 12000}
    decision = agent.run(cid, 'scientist', task, tools, scientist)
    scoped = {'venue_review': True, 'mission_decision_id': decision['decision_id']}
    agent.run(cid, 'falsifier', scoped, tools, cycle + ':falsifier:mission')
    agent.run(cid, 'historian', dict(scoped, critique={'verdict': 'support'}), tools, cycle + ':historian:mission')
    store.complete_cycle(cycle, 'MISSION_STOP')
    count = len(requests)
    assert agent.run(cid, 'scientist', task, tools, scientist)['decision_id'] == decision['decision_id']
    assert agent.run(cid, 'historian', scoped, tools, cycle + ':historian:mission')['artifact_receipt']
    assert len(requests) == count == 3
    with pytest.raises(ValueError, match='scope is immutable'):
        agent.run(cid, 'scientist', dict(task, replay_ids=['other']), tools, scientist)
    assert len(requests) == count
    with store.connect() as db:
        lesson = json.loads(db.execute('SELECT payload FROM mission_lessons').fetchone()[0])
        assert lesson['grounding']['independent_critique'] == review
        assert db.execute('SELECT status FROM cycles WHERE id=?', (cycle,)).fetchone()[0] == 'DECISION_COMPLETE'
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM holdouts').fetchone()[0] == 0
        assert 'opaque context must stay ephemeral' not in ''.join(r[0] for r in db.execute('SELECT payload FROM events'))
    assert research_map(store, cid)['reviewed_mission_lessons'] == 1
    assert any(r['kind'] == 'MISSION_LESSON' for r in store.recall(4))


def test_bad_grounding_rejected_before_any_provider_call(tmp_path):
    store, settings, cid, _, _, scientist, tools, replay, _ = setup(tmp_path)
    path = store.home / 'venue_replays' / replay['replay_id'] / 'quote.events.jsonl'
    path.write_text('edited', encoding='utf-8')
    def forbidden(**kwargs): pytest.fail('No paid call on corrupt evidence')
    with pytest.raises(ValueError):
        runtime(store, settings, forbidden).run(cid, 'scientist', {'venue_review': True,
            'proposal_workflow': True, 'replay_ids': [replay['replay_id']]}, tools, scientist)
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM research_decisions').fetchone()[0] == 0


def test_empty_critique_cannot_complete_and_historian_context_requires_review(tmp_path):
    store, settings, cid, _, cycle, scientist, tools, replay, stop = review_setup(tmp_path)
    store.start_task(scientist, cid, 'scientist', {})
    tools.active_task_id = scientist
    decision = tools.call('scientist', 'stop_mission', stop)
    task = {'venue_review': True, 'mission_decision_id': decision['decision_id']}
    with pytest.raises(ValueError, match='Independent mission review'):
        mission_review.prepare(tools, 'historian', task)
    settings = replace(settings, max_agent_steps=1)
    with pytest.raises(RuntimeError):
        runtime(store, settings, lambda **kwargs: response(content='{}')).run(
            cid, 'falsifier', task, tools, cycle + ':falsifier:mission')
    with store.connect() as db:
        assert db.execute('SELECT status FROM tasks WHERE id=?', (cycle + ':falsifier:mission',)).fetchone()[0] == 'FAILED'
        assert db.execute('SELECT count(*) FROM mission_lessons').fetchone()[0] == 0


def test_scope_does_not_change_general_research_and_cannot_escape_ids(tmp_path):
    store, _, _, _, _, _, tools, replay, _ = setup(tmp_path)
    original = tools.available('scientist')
    selected = mission_review.tool_specs('scientist', {'replay_ids': [replay['replay_id']]}, original)
    assert len(json.dumps(selected)) < len(json.dumps(original)) / 4
    assert 'proposed_spec' in next(t for t in original if t['function']['name'] == 'stop_mission')['function']['parameters']['properties']
    assert 'NUMERICAL TOOL CONTRACT' in task_system_prompt('scientist')
    with pytest.raises(ValueError):
        mission_review.validate_call({'replay_ids': [replay['replay_id']]}, 'stop_mission',
            {'basis': 'MEASURED_DEVELOPMENT', 'replay_ids': [replay['replay_id']]})
    with pytest.raises(ValueError):
        mission_review.validate_call({'replay_ids': [replay['replay_id']]}, 'read_venue_replay', {'replay_id': 'other'})
    with pytest.raises(ValueError):
        mission_review.validate_review({'replay_ids': [replay['replay_id']]}, critique('other'))


def test_verifier_reservations_exclude_other_campaigns_and_other_tasks(tmp_path):
    from quantdesk.store import Store
    from verify_native_review import request_reservations
    store = Store(tmp_path)
    cid = store.campaign({}, 10000)
    other = store.campaign({'hash': 'other'}, 10000)
    for campaign, task, amount in [(cid, 'current-scientist', 100),
                                  (cid, 'earlier-scientist', 1000),
                                  (other, 'other-scientist', 2000)]:
        store.start_task(task, campaign, 'scientist', {})
        store.reserve(campaign, 'scientist', amount, task_id=task)
    assert request_reservations(store, ['current-scientist']) == [
        {'agent': 'scientist', 'reserved': 100, 'actual': None}]
