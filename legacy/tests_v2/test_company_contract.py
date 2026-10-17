"""Artifact-based research and measured planning; synthetic fixtures are explicit."""
import json
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
from quantdesk.agents import ResearchTools,Desk,AgentTaskFailure
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.planning import planning_contract
from tests.test_fitted_models import desk,spec,CODE
from tests.test_data_feasibility import segmented
from tests.test_agent_completion import runtime,response


def test_artifact_build_check_revise_and_replay_preserve_original_evidence(tmp_path,monkeypatch):
    store,cid,_,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    tools=ResearchTools(cid,store,catalog,evaluator,settings,'cycle')
    built=tools.call('engineer','build_candidate',{'spec':spec(),'code':CODE})
    original=store.get_candidate(built['candidate_id'])
    checked=tools.call('engineer','check_candidate',{'candidate_id':original['id']})
    assert checked['status']=='BOUNDED_PROGRAM_CHECK_PASSED' and not built['submitted']
    assert checked['artifact_identity']['spec_hash']==digest(original['spec'])
    revised=tools.call('engineer','revise_candidate',{'candidate_id':original['id'],'spec_changes':{'model':{'inner_days':3},'threshold':0.3}})
    assert revised['candidate_id']!=original['id']
    assert revised['spec']['model']['family']==original['spec']['model']['family']
    assert revised['spec']['model']['inner_days']==3 and original['id'] in revised['spec']['parent_ids']
    report=tools.call('engineer','walk_forward_candidate',{'candidate_id':revised['candidate_id'],'folds':1,'start_index':8,'end_index':8})
    assert report['folds']==1 and report['outer_dates']==['2026-01-09']
    again=tools.call('engineer','walk_forward_candidate',{'candidate_id':revised['candidate_id'],'folds':1,'start_index':8,'end_index':8})
    assert again['already_measured'] and store.get_candidate(original['id'])==original
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM events WHERE kind='SUBMITTED'").fetchone()
    tools.call('engineer','register_existing_candidate',{'candidate_id':revised['candidate_id']})
    with store.connect() as db:
        assert json.loads(db.execute("SELECT payload FROM events WHERE kind='SUBMITTED'").fetchone()[0])['cycle']=='cycle'


def test_invalid_revision_never_changes_prior_artifact_and_foreign_id_is_rejected(tmp_path,monkeypatch):
    store,cid,original,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    tools=ResearchTools(cid,store,catalog,evaluator,settings)
    with pytest.raises(ValueError,match='inner_days'):
        tools.call('engineer','revise_candidate',{'candidate_id':original['id'],'spec_changes':{'model':{'inner_days':15}}})
    assert store.get_candidate(original['id'])==original
    foreign=store.campaign(store.get_campaign(cid)['plan'],1000)
    with pytest.raises(ValueError,match='in this campaign'):
        ResearchTools(foreign,store,catalog,evaluator,settings).call('engineer','check_candidate',{'candidate_id':original['id']})
    with pytest.raises(ValueError,match='does not own'):
        tools.call('scientist','revise_candidate',{'candidate_id':original['id']})


def test_cosmetic_build_returns_the_actual_persisted_identity(tmp_path,monkeypatch):
    store,cid,original,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    changed=dict(spec(),title='Cosmetic rename')
    built=ResearchTools(cid,store,catalog,evaluator,settings).call('engineer','build_candidate',{'spec':changed,'code':CODE+'# cosmetic\n'})
    assert not built['created'] and built['candidate_id']==original['id']
    assert built['spec']==original['spec'] and built['code_hash']==digest(original['code'])


def test_planning_profiles_only_availability_selected_development_frames(tmp_path):
    store=Store(tmp_path); records=[{'date':f'2026-01-{i:02d}'} for i in range(1,8)]
    cid=store.campaign({'development':records[:6],'holdout':records[6:],'hash':'fixture'},1000)
    reads=[]
    class Catalog:
        def cached(self,record,rows):return record['date'] in {'2026-01-02','2026-01-04','2026-01-06'}
        def load(self,record,rows):
            reads.append((record['date'],rows)); return segmented(record['date']),{'has_sequence_ids':False}
    contract=planning_contract(store,Catalog(),cid,Settings())
    assert reads==[('2026-01-02',None),('2026-01-04',None),('2026-01-06',None)]
    assert all(p['segment_coverage']['maximum_rows']==900 for p in contract['profiles'])
    assert all(p['maximum_observed_feasible_warmup_rows_by_holding_seconds']['300']==597 for p in contract['profiles'])
    assert 'PnL' in contract['interpretation'] and 'outcome' in contract['interpretation']


def test_failed_cycle_can_resume_or_be_explicitly_abandoned_without_token_reset(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},1000); cycle=store.start_cycle(cid)
    request=store.reserve(cid,'engineer',100); store.finish_request(request,90)
    store.suspend_cycle(cid,'FAILED','fixture contract failure')
    assert store.start_cycle(cid)==cycle
    store.suspend_cycle(cid,'PAUSED','fixture interruption')
    evidence=store.abandon_cycle(cid,'Measured mission incompatibility; retain its evidence')
    assert evidence['cycle_id']==cycle and store.start_cycle(cid)!=cycle
    with store.connect() as db:
        assert db.execute('SELECT actual FROM requests').fetchone()[0]==90
        assert db.execute('SELECT status FROM cycles WHERE id=?',(cycle,)).fetchone()[0]=='ABANDONED'
        assert db.execute("SELECT COUNT(*) FROM memories WHERE kind='OPERATIONAL_DIRECTION_CHANGE'").fetchone()[0]==1


def test_live_cycle_abandonment_is_rejected(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},1000); cycle=store.start_cycle(cid)
    store.start_task(cycle+':engineer',cid,'engineer',{})
    with pytest.raises(ValueError,match='live task'):
        store.abandon_cycle(cid,'Cannot interrupt its owning process')
    with store.connect() as db:
        assert db.execute('SELECT status FROM cycles').fetchone()[0]=='ACTIVE'


def test_scientist_repairs_impossible_history_before_its_handoff_is_accepted(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'fixture'},100000)
    class Catalog:
        def cached(self,*a):return True
        def load(self,*a):return segmented(),{}
    bad=spec(); bad['holding_seconds']=300; bad['warmup_rows']=1000
    good=spec(); good['holding_seconds']=300; good['warmup_rows']=300
    replies=[response(content=json.dumps({'strategy_spec':bad})),response(content=json.dumps({'strategy_spec':good}))]
    calls=[]
    def create(**kwargs):calls.append(kwargs); return replies.pop(0)
    result=runtime(store,replace(Settings(),max_agent_steps=2),create).run(cid,'scientist',{'require_strategy_spec':True},ResearchTools(cid,store,Catalog(),None,Settings()),'proposal')
    assert result['strategy_spec']['warmup_rows']==300 and len(calls)==2
    assert any('no usable labels' in m.get('content','') for m in calls[1]['messages'])
    assert not store.candidates(cid)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='PROPOSAL_CONTRACT_REJECTED'").fetchone()[0]==1


def test_built_unmeasured_draft_does_not_become_resume_finalization(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    tools=ResearchTools(cid,store,None,None,Settings())
    built=tools.call('engineer','build_candidate',{'spec':spec(),'code':CODE})
    store.start_task('unfinished-artifact',cid,'engineer',{})
    store.event(cid,'engineer','TOOL_RESULT',{'task_id':'unfinished-artifact','tool':'build_candidate','result':built})
    store.event(cid,'engineer','PROGRAM_ATTEMPT',{'task_id':'unfinished-artifact','tool':'check_candidate','args':{'candidate_id':built['candidate_id']}})
    store.fail_task('unfinished-artifact',RuntimeError('fixture'))
    def create(**kwargs):
        task=json.loads(kwargs['messages'][1]['content'])
        assert task['resume_implementation'] and not task.get('resume_finalization')
        assert task['candidate_checkpoints'][0]['id']==built['candidate_id']
        call=SimpleNamespace(id='submit',function=SimpleNamespace(name='register_existing_candidate',arguments=json.dumps({'candidate_id':built['candidate_id']})))
        return response(call)
    assert runtime(store,replace(Settings(),max_agent_steps=1),create).run(cid,'engineer',{},tools,'unfinished-artifact')['artifact_receipt']


def test_artifact_workflow_rejects_unadvertised_legacy_calls_then_accepts_actual_ids(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000); tools=ResearchTools(cid,store,None,None,Settings())
    calls=[]
    legacy=SimpleNamespace(id='legacy',function=SimpleNamespace(name='register_candidate',arguments=json.dumps({'spec':spec(),'code':CODE})))
    build=SimpleNamespace(id='build',function=SimpleNamespace(name='build_candidate',arguments=json.dumps({'spec':spec(),'code':CODE})))
    def create(**kwargs):
        names={t['function']['name'] for t in kwargs['tools']}
        assert 'register_candidate' not in names and 'build_candidate' in names
        calls.append(kwargs)
        if len(calls)==1:return response(legacy)
        if len(calls)==2:return response(build)
        ident=store.candidates(cid)[0]['id']
        return response(SimpleNamespace(id='submit',function=SimpleNamespace(name='register_existing_candidate',arguments=json.dumps({'candidate_id':ident}))))
    assert runtime(store,replace(Settings(),max_agent_steps=4),create).run(cid,'engineer',{'artifact_workflow':True},tools,'id-workflow')['artifact_receipt']
    assert len(calls)==3
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='SUBMITTED'").fetchone()[0]==1
        assert 'does not advertise' in json.loads(db.execute("SELECT payload FROM events WHERE kind='TOOL_RESULT' ORDER BY id LIMIT 1").fetchone()[0])['result']['error']
