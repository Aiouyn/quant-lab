"""Exact native receipts ground allocation decisions, never alpha approval."""
import json
import pytest
from quantdesk import venue_study,venue_replay
from quantdesk.agents import ResearchTools
from quantdesk.proposals import load
from quantdesk.research_map import research_map
from quantdesk.store import digest
from tests.test_venue_replay import environment


@pytest.fixture(autouse=True)
def dead_capture(monkeypatch): monkeypatch.setattr(venue_study,'process_alive',lambda pid:False)


def setup(tmp_path):
    store,settings,cid,candidate,study,key,_=environment(tmp_path)
    replay=venue_replay.replay(store,settings,cid,candidate,study)
    cycle=store.start_cycle(cid);task=cycle+':scientist'
    store.start_task(task,cid,'scientist',{})
    tools=ResearchTools(cid,store,None,None,settings,cycle);tools.active_task_id=task
    args={'basis':'MEASURED_VENUE','replay_ids':[replay['replay_id']],
        'reason':'Observed flat-price fixture loses to fees; stop this allocation only',
        'next_experiment':'Predeclare a supported mechanism with incremental net evidence versus cash',
        'limitations':'Synthetic retrospective fixture, no independent confirmation or class verdict'}
    return store,settings,cid,candidate,cycle,task,tools,replay,args


def test_venue_stop_binds_exact_registered_receipt_and_is_immutable(tmp_path):
    store,_,cid,candidate,_,_,tools,replay,args=setup(tmp_path)
    result=tools.call('scientist','stop_mission',args)
    decision=load(store,result['decision_id'],cid);source=decision['grounding']['observations'][0]
    assert decision['basis']=='MEASURED_VENUE' and decision['grounding']['phase']=='venue_development'
    assert source['source_digest']==replay['replay_id']==digest(source['observed_venue_replay'])
    assert source['candidate_id']==candidate and source['source_kind']=='VENUE_REPLAY'
    assert source['observed_venue_replay']==replay['report']
    assert decision['scope']=='CURRENT_RESEARCH_ALLOCATION_ONLY'
    assert tools.call('scientist','stop_mission',args)['decision_id']==result['decision_id']
    with pytest.raises(ValueError,match='immutable'):
        tools.call('scientist','stop_mission',dict(args,reason='Replace the original direction'))
    assert store.get_candidate(candidate)['status']=='PROPOSED'
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM holdouts').fetchone()[0]==0


@pytest.mark.parametrize('invalid',[[],['bogus'],['a'*64],['a'*64,'a'*64]])
def test_unknown_or_duplicate_venue_ids_cannot_create_decisions(tmp_path,invalid):
    store,_,cid,_,_,_,tools,_,args=setup(tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)): tools.call('scientist','stop_mission',dict(args,replay_ids=invalid))
    with store.connect() as db: assert db.execute('SELECT count(*) FROM research_decisions').fetchone()[0]==0


@pytest.mark.parametrize('damage',['journal','registration','candidate'])
def test_tampered_or_unregistered_venue_grounding_cannot_commit(tmp_path,damage):
    store,_,_,candidate,_,_,tools,replay,args=setup(tmp_path)
    if damage=='journal':
        (store.home/'venue_replays'/replay['replay_id']/'depth.events.jsonl').write_text('edited')
    elif damage=='registration':
        with store.connect() as db: db.execute("DELETE FROM memories WHERE kind='VENUE_REPLAY'")
    else:
        with store.connect() as db: db.execute('UPDATE candidates SET code=? WHERE id=?',('edited-code',candidate))
    with pytest.raises(ValueError): tools.call('scientist','stop_mission',args)
    with store.connect() as db: assert db.execute('SELECT count(*) FROM research_decisions').fetchone()[0]==0


def test_versions_of_same_economic_configuration_are_not_separate_grounding(tmp_path,monkeypatch):
    store,settings,cid,candidate,_,_,tools,replay,args=setup(tmp_path)
    real=venue_replay.implementation()
    monkeypatch.setattr(venue_replay,'implementation',lambda:dict(real,version_fixture=2))
    second=venue_replay.replay(store,settings,cid,candidate,replay['report']['study_id'])
    assert second['replay_id']!=replay['replay_id']
    with pytest.raises(ValueError,match='Repeated replay versions'):
        tools.call('scientist','stop_mission',dict(args,replay_ids=[replay['replay_id'],second['replay_id']]))


def test_cross_campaign_native_history_is_scoped_and_source_preserved(tmp_path):
    store,settings,source_cid,_,_,_,_,replay,args=setup(tmp_path)
    cid=store.campaign({'hash':'later','development':[]},1000);cycle=store.start_cycle(cid)
    task=cycle+':scientist';store.start_task(task,cid,'scientist',{})
    tools=ResearchTools(cid,store,None,None,settings,cycle);tools.active_task_id=task
    result=tools.call('scientist','stop_mission',args)
    assert result['decision']['grounding']['observations'][0]['source_campaign']==source_cid
    assert result['decision']['campaign']==cid and result['decision']['scope']=='CURRENT_RESEARCH_ALLOCATION_ONLY'


def test_independent_review_required_before_native_mission_lesson_and_map(tmp_path):
    store,_,cid,_,cycle,_,tools,replay,args=setup(tmp_path)
    result=tools.call('scientist','stop_mission',args)
    task=cycle+':historian';store.start_task(task,cid,'historian',{})
    tools.active_task_id=task
    lesson={'decision_id':result['decision_id'],'lesson':'Observed configuration loses modeled fees; do not generalize the family',
        'next_experiment':'Precommit a matched cash comparison with supported history and unchanged costs'}
    with pytest.raises(ValueError,match='Independent mission review'):
        tools.call('historian','record_mission_lesson',lesson)
    review=cycle+':falsifier:mission';store.start_task(review,cid,'falsifier',{})
    store.finish_task(review,{'critique':'Synthetic result has limited scope; exact receipt contains authoritative costs and source',
        'replay_id':replay['replay_id']})
    assert tools.call('historian','record_mission_lesson',lesson)['recorded']
    assert tools.call('historian','record_mission_lesson',lesson)['already_recorded']
    mapped=research_map(store,cid)
    assert mapped['reviewed_mission_lessons']==1
    suggestion=mapped['pending_mission_experiment_suggestions'][0]
    assert suggestion['basis']=='MEASURED_VENUE' and suggestion['decision_id']==result['decision_id']
    assert any(r['kind']=='MISSION_LESSON' for r in store.recall(4))
    with store.connect() as db:
        value=json.loads(db.execute('SELECT payload FROM mission_lessons').fetchone()[0])
        assert value['grounding']['decision']['grounding']['observations'][0]['observed_venue_replay']==replay['report']
        assert db.execute('SELECT count(*) FROM mission_lessons').fetchone()[0]==1
