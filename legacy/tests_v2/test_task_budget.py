"""Durable role allocation limits; provider fixtures are explicitly synthetic."""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from quantdesk.store import Store
from quantdesk.settings import Settings
from tests.test_agent_completion import runtime,response


def test_task_budget_is_atomic_and_charges_ambiguous_requests_after_resume(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10000)
    store.start_task('bounded',cid,'scientist',{'task_token_limit':100})
    def reserve():
        try:return store.reserve(cid,'scientist',60,task_id='bounded')
        except RuntimeError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:reserve(),range(2)))
    request=next(r for r in results if r)
    assert len([r for r in results if r])==1
    store.finish_request(request,20)
    uncertain=store.reserve(cid,'scientist',80,task_id='bounded')
    store.finish_request(uncertain,None,'FAILED')
    store.fail_task('bounded',RuntimeError('synthetic failure'))
    reopened=Store(tmp_path)
    reopened.start_task('bounded',cid,'scientist',{})
    with pytest.raises(RuntimeError,match='Task token budget exhausted'):
        reopened.reserve(cid,'scientist',1,task_id='bounded')
    with reopened.connect() as db:
        assert db.execute('SELECT SUM(COALESCE(actual,reserved)) FROM requests').fetchone()[0]==100
        assert db.execute('SELECT token_limit FROM campaigns').fetchone()[0]==10000


def test_task_limits_cannot_be_raised_or_attached_to_old_checkpoints(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10000)
    store.start_task('bounded',cid,'scientist',{'task_token_limit':100})
    store.fail_task('bounded',RuntimeError('synthetic failure'))
    with pytest.raises(ValueError,match='immutable'):
        store.start_task('bounded',cid,'scientist',{'task_token_limit':101})
    store.start_task('old',cid,'scientist',{})
    store.fail_task('old',RuntimeError('synthetic failure'))
    with pytest.raises(ValueError,match='unattributed'):
        store.start_task('old',cid,'scientist',{'task_token_limit':100})


def test_completed_task_and_requests_cannot_cross_role_or_campaign(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10000); other=store.campaign({},10000)
    store.start_task('complete',cid,'scientist',{'task_token_limit':100})
    store.finish_task('complete',{'fixture':True})
    for target,role in ((other,'scientist'),(cid,'engineer')):
        with pytest.raises(ValueError,match='campaign/role'):
            store.start_task('complete',target,role,{})
        with pytest.raises(ValueError,match='campaign/role'):
            store.reserve(target,role,10,task_id='complete')


def test_report_only_drops_unused_schemas_and_stops_before_overbudget_call(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000)
    class Tools:
        def available(self,role):
            return [{'type':'function','function':{'name':'unused','description':'x'*40000,'parameters':{'type':'object','properties':{}}}}]
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs['tool_choice']=='none' and kwargs['tools']==[]
        # Invalid final JSON forces a second attempted request. Actual billing
        # exceeds the first reservation deliberately; usage must still be charged.
        result=response(content='not JSON'); result.usage.total_tokens=29000
        return result
    settings=replace(Settings(),max_agent_steps=2)
    task={'resume_report_only':True,'task_token_limit':30000}
    with pytest.raises(RuntimeError,match='Task token budget exhausted'):
        runtime(store,settings,create).run(cid,'scientist',task,Tools(),'bounded-provider')
    assert len(calls)==1
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==1
        assert db.execute('SELECT actual FROM requests').fetchone()[0]==29000
        assert db.execute('SELECT task_id FROM request_tasks').fetchone()[0]=='bounded-provider'
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='FAILED'


def test_completed_budgeted_task_is_reused_without_a_new_request(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000)
    class Tools:
        def available(self,role):return []
    calls=[]
    def create(**kwargs):calls.append(kwargs); return response(content=json.dumps({'handoff':'synthetic'}))
    agent=runtime(store,Settings(),create)
    task={'resume_report_only':True,'task_token_limit':30000}
    first=agent.run(cid,'scientist',task,Tools(),'bounded-complete')
    assert agent.run(cid,'scientist',{},Tools(),'bounded-complete')==first
    assert len(calls)==1


def test_scientist_resume_uses_focused_public_artifacts_with_same_budget(tmp_path):
    from quantdesk.agents import ResearchTools
    from tests.test_data_feasibility import segmented
    from tests.test_fitted_models import spec
    from quantdesk.store import digest
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'synthetic'},200000)
    tid='bounded-scientist'; store.start_task(tid,cid,'scientist',{'task_token_limit':60000})
    request=store.reserve(cid,'scientist',20000,task_id=tid); store.finish_request(request,20000)
    original=[{'id':1,'campaign':cid,'candidate':'measured','kind':'DEVELOPMENT_EXPERIMENT',
        'payload':{'status':'DEVELOPMENT_ONLY','pnl':0.0,'trades':0,'interpretation':'Not alpha',
            'spec':{'rationale':'x'*100000},'model_artifacts':[{'fit_table':'y'*100000}]}}]
    store.event(cid,'scientist','TOOL_RESULT',{'task_id':tid,'tool':'recall','result':original})
    store.fail_task(tid,RuntimeError('Synthetic context reservation too large'))
    class Catalog:
        def cached(self,*a):return True
        def load(self,*a):return segmented(),{}
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs['tools']==[] and kwargs['tool_choice']=='none'
        assert kwargs['messages'][0]['role']=='system'
        assert all(m['role']=='user' for m in kwargs['messages'][1:])
        task=json.loads(kwargs['messages'][1]['content'])
        observation=task['measured_task_evidence'][0]
        assert observation['result_digest']==digest(original)
        assert observation['result']['memories'][0]['payload']['trades']==0
        assert len(kwargs['messages'][1]['content'])<10000
        return response(content=json.dumps({'strategy_spec':spec(),'supported_history':'Measured segmented fixture only'}))
    agent=runtime(store,replace(Settings(),max_agent_steps=1),create)
    result=agent.run(cid,'scientist',{'require_strategy_spec':True,'task_token_limit':60000},
        ResearchTools(cid,store,Catalog(),None,Settings()),tid)
    assert result['strategy_spec'] and len(calls)==1
    with store.connect() as db:
        assert db.execute('SELECT token_limit FROM task_budgets').fetchone()[0]==60000
        assert db.execute('SELECT COUNT(*) FROM request_tasks').fetchone()[0]==2
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='COMPLETE'


def test_rejected_public_scientist_handoff_is_preserved_without_reasoning(tmp_path):
    from quantdesk.agents import ResearchTools
    store=Store(tmp_path); cid=store.campaign({},100000)
    public={'decision':'SYNTHETIC DECLINE','reason':'Preserve this public contract failure, not a strategy approval'}
    def create(**kwargs):
        result=response(content=json.dumps(public))
        result.choices[0].message.reasoning_content='private synthetic context that must not persist'
        return result
    agent=runtime(store,replace(Settings(),max_agent_steps=1),create)
    with pytest.raises(RuntimeError,match='exceeded'):
        agent.run(cid,'scientist',{'require_strategy_spec':True,'task_token_limit':30000},
            ResearchTools(cid,store,None,None,Settings()),'rejected-public')
    with store.connect() as db:
        attempt=json.loads(db.execute("SELECT payload FROM events WHERE kind='PUBLIC_HANDOFF_ATTEMPT'").fetchone()[0])
        assert json.loads(attempt['content'])==public and not attempt['content_truncated']
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='FAILED'
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='PROPOSAL_CONTRACT_REJECTED'").fetchone()[0]==1
        assert 'private synthetic context' not in json.dumps([dict(r) for r in db.execute('SELECT * FROM events')])
