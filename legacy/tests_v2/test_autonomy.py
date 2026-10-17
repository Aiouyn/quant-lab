"""Bounded recovery, real-process leases and evidence-efficient retrieval."""
import os
import time
from types import SimpleNamespace
import pytest
from quantdesk.autonomy import Supervisor
from quantdesk.agents import AgentTaskFailure
from quantdesk.store import Store
from quantdesk.control import ResearchTimeLimit


def setup(tmp_path,outcomes,budget=100000):
    store=Store(tmp_path); cid=store.campaign({},budget)
    class Desk:
        calls=0
        def run(self,campaign,cycles,own_lease):
            assert cycles==1 and not own_lease
            self.calls+=1
            result=outcomes.pop(0)
            if isinstance(result,Exception):
                store.set_status(cid,'STOPPED')
                raise result
            return result
    return store,cid,Desk()


def test_transient_failure_recovers_once_and_records_budget(tmp_path):
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('scientist',TimeoutError()),{'status':'READY'},{'status':'READY'}])
    report=Supervisor(desk,store).run(cid,2,1,1,0)
    assert report['status']=='COMPLETE' and report['completed_cycles']==2
    assert report['retries']==1 and desk.calls==3
    assert report['remaining_tokens']==100000
    assert store.get_campaign(cid)['status']=='READY'


def test_authentication_failure_never_retries(tmp_path):
    error=RuntimeError('credentials invalid')
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('director',error)])
    result=Supervisor(desk,store).run(cid,2,1,2,0)
    assert result['status']=='FAILED' and result['retries']==0 and desk.calls==1


def test_ambiguous_failed_requests_remain_charged_before_retry(tmp_path):
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('scientist',TimeoutError())],budget=30000)
    original=desk.run
    def fail(*args,**kwargs):
        reservation=store.reserve(cid,'scientist',20000)
        store.finish_request(reservation,None,'FAILED')
        return original(*args,**kwargs)
    desk.run=fail
    result=Supervisor(desk,store).run(cid,2,1,2,0)
    assert result['status']=='BUDGET_EXHAUSTED' and result['remaining_tokens']==10000
    assert desk.calls==1


def test_retry_allowance_is_finite_across_the_run(tmp_path):
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('scientist',TimeoutError()),{'status':'READY'},AgentTaskFailure('engineer',TimeoutError())])
    result=Supervisor(desk,store).run(cid,2,1,1,0)
    assert result['status']=='FAILED' and result['completed_cycles']==1 and result['retries']==1


def test_stop_during_recovery_does_not_restart_provider(tmp_path):
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('director',TimeoutError())])
    now=[0.0]
    def sleep(seconds):
        now[0]+=seconds
        store.set_status(cid,'STOP_REQUESTED')
    result=Supervisor(desk,store,clock=lambda:now[0],sleep=sleep).run(cid,2,1,2,5)
    assert result['status']=='STOPPED' and desk.calls==1


def test_time_limit_preserves_finished_cycle(tmp_path):
    store,cid,desk=setup(tmp_path,[{'status':'READY'}])
    now=[0.0]; original=desk.run
    def finish(*args,**kwargs):
        result=original(*args,**kwargs); now[0]=10
        return result
    desk.run=finish
    result=Supervisor(desk,store,clock=lambda:now[0]).run(cid,2,0.001,0,0)
    assert result['status']=='TIME_LIMIT' and result['completed_cycles']==1


def test_deadline_interrupts_between_units_in_one_cycle_and_restores_boundary(tmp_path):
    store,cid,desk=setup(tmp_path,[])
    now=[0.0]; units=[]
    def run(*args,**kwargs):
        store.check_research_control(cid)
        units.append('first durable unit')
        store.event(cid,'fixture','UNIT_COMPLETED',{})
        now[0]=4
        store.check_research_control(cid)
        pytest.fail('A second unit must not start after the limit')
    desk.run=run
    report=Supervisor(desk,store,clock=lambda:now[0]).run(cid,2,0.001,2,0)
    assert report['status']=='TIME_LIMIT' and report['completed_cycles']==0
    assert report['retries']==0 and report['failures']==[]
    assert units==['first durable unit'] and store.get_campaign(cid)['status']=='STOPPED'
    store.check_research_control(cid)


def test_provider_finishing_after_limit_is_charged_but_tool_is_not_executed(tmp_path):
    import json
    from dataclasses import replace
    from quantdesk.agents import ResearchTools
    from quantdesk.settings import Settings
    from tests.test_agent_completion import runtime,response,submit
    from tests.test_quantdesk import CODE
    store=Store(tmp_path); cid=store.campaign({},100000)
    now=[0.0]; calls=[]
    def create(**kwargs):
        calls.append(kwargs); now[0]=4
        return response(submit(CODE))
    settings=replace(Settings(),max_agent_steps=1)
    tools=ResearchTools(cid,store,None,None,settings)
    with store.research_deadline(cid,3.6,lambda:now[0]):
        with pytest.raises(ResearchTimeLimit):
            runtime(store,settings,create).run(cid,'engineer',{},tools,'late-provider')
    assert len(calls)==1 and not store.candidates(cid)
    with store.connect() as db:
        request=db.execute('SELECT status,actual FROM requests').fetchone()
        assert request['status']=='COMPLETE' and request['actual']==100
        assert db.execute("SELECT status FROM tasks WHERE id='late-provider'").fetchone()[0]=='STOPPED'


def test_model_time_limit_does_not_start_second_archive(tmp_path,monkeypatch):
    from tests.test_fitted_models import desk as fitted_desk
    from quantdesk.models import ModelService
    store,cid,candidate,catalog,_,settings=fitted_desk(tmp_path,monkeypatch)
    now=[0.0]; original=catalog.load; reads=[]
    def load(record,rows=None):
        reads.append(record['date']); value=original(record,rows); now[0]=4
        return value
    catalog.load=load
    records=store.get_campaign(cid)['plan']['development'][:8]
    cutoff=int(catalog.frames['2026-01-09']['ts_ms'][0])
    with store.research_deadline(cid,3.6,lambda:now[0]):
        with pytest.raises(ResearchTimeLimit):
            ModelService(store,catalog,settings).fit(cid,candidate,records,cutoff)
    assert reads==['2026-01-01']


def test_deadline_inside_tool_propagates_as_control_not_hypothesis_error(tmp_path):
    from dataclasses import replace
    from quantdesk.agents import ResearchTools
    from quantdesk.settings import Settings
    from tests.test_agent_completion import runtime,response
    store=Store(tmp_path); cid=store.campaign({},100000); now=[0.0]
    call=SimpleNamespace(id='profile',function=SimpleNamespace(name='inspect_data',arguments='{}'))
    tools=ResearchTools(cid,store,None,None,Settings())
    def measured(*args):
        store.event(cid,'fixture','UNIT_COMPLETED',{})
        now[0]=4
        store.check_research_control(cid)
    tools.call=measured
    with store.research_deadline(cid,3.6,lambda:now[0]):
        with pytest.raises(ResearchTimeLimit):
            runtime(store,replace(Settings(),max_agent_steps=2),lambda **kw:response(call)).run(cid,'scientist',{},tools,'deadline-tool')
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='UNIT_COMPLETED'").fetchone()[0]==1
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='TOOL_RESULT'").fetchone()[0]==0
        assert db.execute("SELECT status FROM tasks WHERE id='deadline-tool'").fetchone()[0]=='STOPPED'


def test_expired_live_lease_cannot_be_stolen(tmp_path):
    store=Store(tmp_path)
    with store.connect() as db:
        db.execute('INSERT INTO leases(name,owner,expires,pid) VALUES(?,?,?,?)',('research','active',time.time()-10,os.getpid()))
    with pytest.raises(RuntimeError,match='Another desk process'):
        with store.lease('research'):
            pass


def test_retrieval_preserves_statistics_without_replay_payloads(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},1000)
    store.memory(cid,'candidate','WALK_FORWARD_EXPERIMENT',{'statistics':{'days':20,'pnl':2},
        'daily_results':[{'date':'2026-01-01','large':'table'}]})
    result=store.recall()[0]['payload']
    assert result['statistics']['days']==20
    assert result['evaluated_dates']==['2026-01-01'] and 'daily_results' not in result


def test_comments_and_formatting_do_not_create_new_candidates(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},1000)
    strategy={'title':'a','threshold':0.2}
    first,created=store.candidate(cid,strategy,'def compute_signal(df):\n    return df\n')
    second,created=store.candidate(cid,dict(strategy,title='renamed'),'# decoration\ndef compute_signal( df ):\n  return df # another decoration\n')
    assert first==second and not created
