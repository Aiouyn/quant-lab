"""Scientific allocation contracts with real services and explicit synthetic inputs."""
import json
import os
from dataclasses import replace
from types import SimpleNamespace
import pytest
from quantdesk.agents import ResearchTools, Desk, AgentTaskFailure
from quantdesk.autonomy import Supervisor
from quantdesk.evaluation import Evaluator
from quantdesk.proposals import load
from quantdesk.research_map import research_map
from quantdesk.settings import Settings
from quantdesk.store import Store, digest
from tests.test_agent_completion import runtime, response
from tests.test_data_feasibility import segmented
from tests.test_quantdesk import spec, CODE, Message


class Catalog:
    def __init__(self): self.reads=[]
    def cached(self, *args): return True
    def load(self, record, rows):
        self.reads.append((record['date'],rows))
        return segmented(record['date']), {'has_sequence_ids':False,'fixture':True}


def setup(tmp_path, role='scientist', start=True):
    store=Store(tmp_path)
    dates=[{'date':f'2026-01-{i:02d}'} for i in range(1,16)]
    plan={'development':dates[:12], 'validation':dates[12:14], 'holdout':dates[14:],
          'hash':'scientific-fixture', 'sampled':False, 'max_rows_per_day':None}
    cid=store.campaign(plan,200000)
    cycle=store.start_cycle(cid); task=cycle+':'+role
    settings=replace(Settings(),home=tmp_path,max_agent_steps=2)
    catalog=Catalog(); tools=ResearchTools(cid,store,catalog,None,settings,cycle)
    if start: store.start_task(task,cid,role,{})
    tools.active_task_id=task
    return store,cid,cycle,task,tools,settings


def proposal(**changes):
    value={'strategy_spec':spec(holding_seconds=30,warmup_rows=29),
           'effective_history_rows':30,'development_indices':[0,5,11],
           'hypothesis':'Persistent pressure may forecast delayed price motion',
           'implementation_design':'Causal trailing thirty-observation pressure',
           'baseline':'Cash and fixed trading on identical dates',
           'ablation':'Remove pressure while keeping execution identical',
           'stop_condition':'Reject when incremental net capture does not exceed modeled costs',
           'limitations':'Synthetic metadata demonstrates compatibility only'}
    return dict(value,**changes)


def stopping(**changes):
    return dict({'basis':'DATA_INCOMPATIBLE','proposed_spec':spec(holding_seconds=300,warmup_rows=1000),
                 'day_index':0,'reason':'History exceeds the observed segment length',
                 'next_experiment':'Test an explicitly shorter causal mechanism',
                 'limitations':'This rejects this allocation, not a feature family'},**changes)


def call(name,args,ident='call'):
    return SimpleNamespace(id=ident,function=SimpleNamespace(name=name,arguments=json.dumps(args)))


def test_proposal_attaches_actual_compatibility_and_is_immutable(tmp_path):
    store,cid,cycle,task,tools,_=setup(tmp_path)
    receipt=tools.call('scientist','submit_proposal',proposal())
    decision=load(store,receipt['decision_id'],cid)
    assert decision['task_id']==task and decision['cycle']==cycle and decision['kind']=='PROPOSAL'
    assert digest(decision)==receipt['decision_id']
    checks=decision['grounding']['feasibility']
    assert all(c['status']=='LABELS_AVAILABLE' and c['segment_coverage']['maximum_rows']==900 for c in checks)
    assert [c['development_index'] for c in checks]==[0,5,11]
    assert tools.catalog.reads==[('2026-01-01',None),('2026-01-06',None),('2026-01-12',None)]
    assert not store.candidates(cid) and not receipt['submitted_candidate']
    assert tools.call('scientist','submit_proposal',proposal())['decision_id']==receipt['decision_id']
    with pytest.raises(ValueError,match='immutable'):
        tools.call('scientist','submit_proposal',proposal(hypothesis='A different hypothesis'))
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM research_decisions').fetchone()[0]==1
        assert not db.execute("SELECT 1 FROM data_exposures WHERE phase!='development'").fetchone()
        assert not db.execute('SELECT 1 FROM requests').fetchone()


@pytest.mark.parametrize('changes,match',[
    ({'effective_history_rows':100},'mature effective history'),
    ({'effective_history_rows':True},'integer'),
    ({'development_indices':[12]},'validation/holdout'),
    ({'development_indices':[0,0]},'unique'),
    ({'strategy_spec':spec(holding_seconds=300,warmup_rows=1000)},'cannot mature'),
    ({'hypothesis':''},'nonempty')])
def test_invalid_proposal_never_creates_decision(tmp_path,changes,match):
    store,cid,_,_,tools,_=setup(tmp_path)
    with pytest.raises(ValueError,match=match): tools.call('scientist','submit_proposal',proposal(**changes))
    with store.connect() as db: assert db.execute('SELECT count(*) FROM research_decisions').fetchone()[0]==0
    assert not store.candidates(cid)


def test_fitted_proposal_requires_earlier_training_dates(tmp_path):
    store,cid,_,_,tools,_=setup(tmp_path)
    modeled=spec(holding_seconds=30,warmup_rows=29,model={'min_train_days':10})
    with pytest.raises(ValueError,match='minimum earlier training'):
        tools.call('scientist','submit_proposal',proposal(strategy_spec=modeled,development_indices=[9]))
    result=tools.call('scientist','submit_proposal',proposal(strategy_spec=modeled,development_indices=[10]))
    assert result['decision']['strategy_spec']['model']['min_train_days']==10
    assert len(tools.catalog.reads)==1


def test_stop_verifies_actual_incompatibility_without_candidate_or_replay(tmp_path):
    store,cid,_,_,tools,_=setup(tmp_path)
    result=tools.call('scientist','stop_mission',stopping())
    decision=load(store,result['decision_id'],cid)
    measured=decision['grounding']['feasibility']
    assert measured['status']=='NO_USABLE_LABELS' and measured['eligible_label_rows']==0
    assert measured['maximum_observed_feasible_warmup_rows']==597
    assert decision['scope']=='CURRENT_RESEARCH_ALLOCATION_ONLY'
    assert not store.candidates(cid)
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM events WHERE kind IN ('SUBMITTED','EXPERIMENT_STARTED')").fetchone()


def test_stop_cannot_invent_incompatibility_or_prior_evidence(tmp_path):
    store,cid,_,_,tools,_=setup(tmp_path)
    with pytest.raises(ValueError,match='supports labels'):
        tools.call('scientist','stop_mission',stopping(proposed_spec=spec(warmup_rows=0)))
    ident,_=store.candidate(cid,spec(),CODE)
    with pytest.raises(ValueError,match='No measured development'):
        tools.call('scientist','stop_mission',stopping(basis='MEASURED_DEVELOPMENT',candidate_ids=[ident]))
    store.memory(cid,ident,'VALIDATION',{'pnl':-100})
    with pytest.raises(ValueError,match='No measured development'):
        tools.call('scientist','stop_mission',stopping(basis='MEASURED_DEVELOPMENT',candidate_ids=[ident]))
    store.memory(cid,ident,'DEVELOPMENT_EXPERIMENT',{'pnl':-2,'daily_results':[{'date':'2026-01-01','trades':2}]})
    result=tools.call('scientist','stop_mission',stopping(basis='MEASURED_DEVELOPMENT',candidate_ids=[ident]))
    source=result['decision']['grounding']['observations'][0]
    assert source['source_kind']=='DEVELOPMENT_EXPERIMENT' and source['observed_development']['pnl']==-2
    assert source['source_digest']==digest(source['observed_development'])


def test_decision_ownership_roles_seal_and_integrity(tmp_path):
    store,cid,_,task,tools,_=setup(tmp_path,start=False)
    with pytest.raises(ValueError,match='owning role'): tools.call('scientist','submit_proposal',proposal())
    store.start_task(task,cid,'scientist',{})
    with pytest.raises(ValueError,match='does not own'): tools.call('engineer','submit_proposal',proposal())
    with store.connect() as db: db.execute('UPDATE tasks SET pid=? WHERE id=?',(os.getpid()+100000,task))
    with pytest.raises(ValueError,match='owning role'): tools.call('scientist','submit_proposal',proposal())
    with store.connect() as db: db.execute('UPDATE tasks SET pid=? WHERE id=?',(os.getpid(),task))
    store.set_status(cid,'SEALED')
    with pytest.raises(ValueError,match='Sealed'): tools.call('scientist','submit_proposal',proposal())
    store.set_status(cid,'READY'); result=tools.call('scientist','submit_proposal',proposal())
    other=store.campaign({},1000)
    with pytest.raises(ValueError,match='Unknown decision'): load(store,result['decision_id'],other)
    with store.connect() as db: db.execute('UPDATE research_decisions SET payload=?',('{}',))
    with pytest.raises(ValueError,match='integrity'): load(store,result['decision_id'],cid)


def test_runtime_tool_commit_finishes_without_followup_calls_and_retains_rejected_args(tmp_path):
    store,cid,_,task,tools,settings=setup(tmp_path,start=False)
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==1: return response(call('submit_proposal',proposal(effective_history_rows=100)))
        assert kwargs['tool_choice']=='auto'
        assert {t['function']['name'] for t in kwargs['tools']}=={'submit_proposal','stop_mission'}
        result=response(call('submit_proposal',proposal()))
        result.choices[0].message.tool_calls.append(call('experiment',{},'must-not-run'))
        return result
    result=runtime(store,settings,create).run(cid,'scientist',{'proposal_workflow':True},tools,task)
    assert result['artifact_receipt'] and len(calls)==2
    with store.connect() as db:
        rejected=json.loads(db.execute("SELECT payload FROM events WHERE kind='SCIENTIFIC_DECISION_ATTEMPT' ORDER BY id LIMIT 1").fetchone()[0])
        assert rejected['args']['effective_history_rows']==100
        assert db.execute('SELECT status FROM tasks WHERE id=?',(task,)).fetchone()[0]=='COMPLETE'
        assert not db.execute("SELECT 1 FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()


def test_runtime_prose_cannot_finish_scientific_task(tmp_path):
    store,cid,_,task,tools,settings=setup(tmp_path,start=False)
    with pytest.raises(RuntimeError,match='step task budget'):
        runtime(store,settings,lambda **kw:response(content=json.dumps({'strategy_spec':proposal()['strategy_spec']}))).run(
            cid,'scientist',{'proposal_workflow':True},tools,task)
    with store.connect() as db:
        assert not db.execute('SELECT 1 FROM research_decisions').fetchone()
        assert db.execute('SELECT status FROM tasks WHERE id=?',(task,)).fetchone()[0]=='FAILED'


def test_recorded_decision_recovers_after_task_crash_without_provider(tmp_path):
    store,cid,_,task,tools,settings=setup(tmp_path)
    result=tools.call('scientist','submit_proposal',proposal()); store.fail_task(task,RuntimeError('crash after commit'))
    recovered=runtime(store,settings,lambda **kw:pytest.fail('No paid call after a durable decision')).run(
        cid,'scientist',{'proposal_workflow':True},tools,task)
    assert recovered['decision_id']==result['decision_id'] and recovered['artifact_receipt']


def test_artifact_only_prompt_omits_code_syntax_but_enforces_exact_tool_commit(tmp_path):
    from quantdesk.agents import task_system_prompt
    store,cid,_,task,tools,settings=setup(tmp_path,start=False)
    settings=replace(settings,max_agent_steps=1)
    def create(**kwargs):
        prompt=kwargs['messages'][0]['content']
        assert 'NUMERICAL TOOL CONTRACT' not in prompt and 'fully mature history' in prompt
        assert 'min_edge_bps is an EXTRA margin' in prompt and 'thinking' in kwargs['extra_body']
        assert {t['function']['name'] for t in kwargs['tools']}=={'submit_proposal','stop_mission'}
        return response(call('submit_proposal',proposal()))
    result=runtime(store,settings,create).run(cid,'scientist',{'proposal_workflow':True},tools,task)
    assert result['artifact_receipt']
    assert 'NUMERICAL TOOL CONTRACT' in task_system_prompt('engineer')
    assert 'NUMERICAL TOOL CONTRACT' in task_system_prompt('scientist')
    assert 'NUMERICAL TOOL CONTRACT' not in task_system_prompt('director')
    assert 'NUMERICAL TOOL CONTRACT' not in task_system_prompt('historian')
    assert research_map(store,cid)['recent_scientific_decisions'][0]['decision_id']==result['decision_id']


def test_mission_lesson_requires_review_and_preserves_immutable_grounding(tmp_path):
    store,cid,cycle,task,tools,settings=setup(tmp_path)
    result=tools.call('scientist','stop_mission',stopping()); store.finish_task(task,result)
    historian=cycle+':historian:mission'; store.start_task(historian,cid,'historian',{}); tools.active_task_id=historian
    args={'decision_id':result['decision_id'],'lesson':'History does not mature inside these segments','next_experiment':'Test shorter history'}
    with pytest.raises(ValueError,match='Independent mission review'): tools.call('historian','record_mission_lesson',args)
    review=cycle+':falsifier:mission'; store.start_task(review,cid,'falsifier',{})
    store.finish_task(review,{'critique':'Evidence concerns this configuration only'})
    assert tools.call('historian','record_mission_lesson',args)['recorded']
    assert tools.call('historian','record_mission_lesson',args)['already_recorded']
    with pytest.raises(ValueError,match='immutable reviewed lesson'):
        tools.call('historian','record_mission_lesson',dict(args,lesson='A different interpretation'))
    store.fail_task(historian,RuntimeError('crash after lesson'))
    recovered=runtime(store,settings,lambda **kw:pytest.fail('Lesson checkpoint must not spend again')).run(
        cid,'historian',{'required_mission_decision':result['decision_id']},tools,historian)
    assert recovered['artifact_receipt']
    mapping=research_map(store,cid)
    assert mapping['reviewed_mission_lessons']==1 and mapping['completed_strategy_cohorts']==0
    assert mapping['pending_mission_experiment_suggestions'][0]['decision_id']==result['decision_id']
    with store.connect() as db:
        payload=json.loads(db.execute('SELECT payload FROM mission_lessons').fetchone()[0])
        assert payload['grounding']['decision']==result['decision']
        assert payload['grounding']['independent_critique']['critique'].startswith('Evidence concerns')
        assert db.execute("SELECT count(*) FROM memories WHERE kind='MISSION_LESSON'").fetchone()[0]==1


def test_full_stop_company_and_supervisor_separate_decisions_from_strategies(tmp_path):
    store,cid,_,_,tools,settings=setup(tmp_path,start=False)
    missions=[]; roles=[]
    def create(**kwargs):
        task=json.loads(kwargs['messages'][1]['content'])
        if 'proposal_workflow' in task:
            roles.append('scientist'); return response(call('stop_mission',stopping()))
        if 'required_mission_decision' in task:
            roles.append('historian')
            return response(call('record_mission_lesson',{'decision_id':task['required_mission_decision'],
                'lesson':'Observed segment cannot mature the current history','next_experiment':'Explicitly shorten mechanism and retain counterevidence'}))
        if 'mission_stop' in task:
            roles.append('falsifier'); return response(content=json.dumps({'critique':'Incompatibility is real, but does not establish lack of alpha in a revised design'}))
        assert 'objective' in task, 'Engineer/other roles must not run for an allocation stop'
        roles.append('director'); missions.append(task)
        return response(content=json.dumps({'mission':'Evaluate a one-thousand-row history and three-hundred-second holding'}))
    actual_runtime=runtime(store,settings,create)
    class NoValidation:
        def screen(self,*args): pytest.fail('Stops cannot trigger strategy validation')
        def validate(self,*args): pytest.fail('Stops cannot promote strategies')
    desk=Desk(store,tools.catalog,NoValidation(),settings,actual_runtime)
    report=Supervisor(desk,store).run(cid,2,1,0,0)
    assert report['status']=='COMPLETE' and report['completed_cycles']==2
    assert report['completed_mission_decisions']==2 and report['completed_strategy_cohorts']==0
    assert report['completed_unspecified_cycles']==0
    assert roles==['director','scientist','falsifier','historian']*2
    assert missions[1]['research_map']['completed_mission_decisions']==1
    assert missions[1]['research_map']['reviewed_mission_lessons']==1
    assert any(m['kind']=='MISSION_LESSON' for m in missions[1]['memory'])
    assert not store.candidates(cid)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM cycles WHERE status='DECISION_COMPLETE'").fetchone()[0]==2
        assert not db.execute("SELECT 1 FROM cycles WHERE status='COMPLETE'").fetchone()
        assert not db.execute('SELECT 1 FROM holdouts').fetchone()
        assert not db.execute("SELECT 1 FROM events WHERE kind IN ('SUBMITTED','EXPERIMENT_STARTED','CYCLE_COMPLETED')").fetchone()


def test_full_proposal_company_uses_exact_artifact_through_actual_runtime_and_replay(tmp_path):
    from tests.test_quantdesk import frame
    store,cid,_,_,_,settings=setup(tmp_path,start=False)
    settings=replace(settings,max_agent_steps=4)
    # Partial synthetic frames deliberately cannot reach confirmation.
    with store.connect() as db:
        plan=json.loads(db.execute('SELECT plan FROM campaigns WHERE id=?',(cid,)).fetchone()[0])
        plan.update(sampled=True,max_rows_per_day=80)
        db.execute('UPDATE campaigns SET plan=? WHERE id=?',(json.dumps(plan),cid))
    class PartialCatalog:
        def cached(self,*args): return True
        def load(self,*args): return frame(),{'fixture':True,'has_sequence_ids':False}
    engineer_turns=[]; seen_proposal=[]; scientific_spec=spec()
    def create(**kwargs):
        task=json.loads(kwargs['messages'][1]['content'])
        if 'proposal_workflow' in task:
            return response(call('submit_proposal',proposal(strategy_spec=scientific_spec,effective_history_rows=1,development_indices=[0])))
        if 'artifact_workflow' in task:
            seen_proposal.append(task['proposal']); engineer_turns.append(kwargs)
            if len(engineer_turns)==1: return response(call('build_candidate',{'spec':scientific_spec,'code':CODE}))
            ident=store.candidates(cid)[0]['id']
            if len(engineer_turns)==2:
                result=response(call('check_candidate',{'candidate_id':ident,'day_index':0}))
                result.choices[0].message.tool_calls.append(call('experiment_candidate',{'candidate_id':ident,'days':1,'day_indices':[0]},'economic'))
                return result
            return response(call('register_existing_candidate',{'candidate_id':ident}))
        if 'cohort' in task and 'critique' in task:
            return response(call('record_lesson',{'candidate_id':task['cohort'][0]['candidate_id'],
                'lesson':'Partial synthetic evidence cannot establish deployment readiness',
                'evidence':'Full-day coverage is absent','next_experiment':'Independently test complete development dates'}))
        if 'cohort' in task:
            return response(content=json.dumps({'critique':'This development-only partial fixture cannot support confirmation'}))
        assert 'objective' in task
        return response(content=json.dumps({'mission':'Test a bounded causal pressure signal with conservative execution'}))
    catalog=PartialCatalog()
    result=Desk(store,catalog,Evaluator(store,catalog,settings),settings,runtime(store,settings,create)).run(cid)
    assert result['status']=='READY' and len(engineer_turns)==3
    outcome=result['cycle_outcomes'][0]
    assert outcome['kind']=='STRATEGY_COHORT' and outcome['decision_id']==seen_proposal[0]['decision_id']
    assert load(store,outcome['decision_id'],cid)['strategy_spec']==seen_proposal[0]['strategy_spec']
    candidate=store.get_candidate(outcome['candidates'][0])
    assert candidate['code']==CODE and candidate['status']=='PROPOSED'
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM cycles WHERE status='COMPLETE'").fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM tasks WHERE status='COMPLETE'").fetchone()[0]==5
        assert db.execute("SELECT count(*) FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()[0]>=1
        assert db.execute("SELECT count(*) FROM memories WHERE kind='LESSON'").fetchone()[0]==1
        lesson=json.loads(db.execute("SELECT payload FROM memories WHERE kind='LESSON'").fetchone()[0])
        assert lesson['grounding']['development_experiment']
        assert not db.execute('SELECT 1 FROM holdouts').fetchone()
