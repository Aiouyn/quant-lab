"""Actual runtime contracts: finite repair, public drafts and honest task state."""
import json
import os
from dataclasses import replace
from types import SimpleNamespace
import pytest
from quantdesk.agents import AgentRuntime,ResearchTools
from quantdesk.agents import AgentTaskFailure
from quantdesk.settings import Settings
from quantdesk.store import Store
from tests.test_quantdesk import Message,CODE,spec


def response(call=None,content=None):
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=100),choices=[
        SimpleNamespace(message=Message(tool_calls=[call] if call else None,content=content,
            reasoning_content='opaque context must stay ephemeral'),finish_reason='stop')])


def submit(code):
    return SimpleNamespace(id='submission',function=SimpleNamespace(name='register_candidate',
        arguments=json.dumps({'spec':spec(),'code':code})))


def runtime(store,settings,create):
    return AgentRuntime(store,settings,client=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))),model='fixture',provider='mimo')


def test_last_engineer_turn_can_repair_rejected_code(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},150000); calls=[]
    bad=CODE.replace("df['qi_5'].clip(-1.0,1.0)","pl.arange(0,len(df),eager=True)")
    responses=[response(submit(bad)),response(submit(CODE))]
    def create(**kwargs):
        calls.append(json.loads(json.dumps(kwargs)))
        return responses.pop(0)
    tools=ResearchTools(cid,store,None,None,Settings())
    result=runtime(store,replace(Settings(),max_agent_steps=2),create).run(cid,'engineer',{},tools,'repair')
    assert result['artifact_receipt'] and len(store.candidates(cid))==1
    assert calls[-1].get('tool_choice')!='none'
    assert any('Unsupported Polars' in m.get('content','') for m in calls[-1]['messages'])
    with store.connect() as db:
        drafts=[json.loads(r[0]) for r in db.execute("SELECT payload FROM events WHERE kind='PROGRAM_ATTEMPT'")]
        assert drafts[0]['args']['code']==bad and drafts[1]['args']['code']==CODE
        assert db.execute("SELECT status FROM tasks WHERE id='repair'").fetchone()[0]=='COMPLETE'
        assert 'opaque context must stay ephemeral' not in '\n'.join(r[0] for r in db.execute('SELECT payload FROM events'))


def test_director_recovers_truncation_using_public_measurements_once(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},150000); calls=[]
    recall=SimpleNamespace(id='recall',function=SimpleNamespace(name='recall',arguments='{}'))
    partial=response(submit(CODE),'{"mission":"unfinished"'); partial.choices[0].finish_reason='length'
    responses=[response(recall),partial,response(content='{"mission":"Test a distinct cost-feasible mechanism"}')]
    def create(**kwargs):
        calls.append(json.loads(json.dumps(kwargs)))
        return responses.pop(0)
    tools=ResearchTools(cid,store,None,None,Settings())
    result=runtime(store,Settings(),create).run(cid,'director',{'objective':'Find a measured mission'},tools,'director-recovery')
    assert result['mission'] and len(calls)==3 and not store.candidates(cid)
    assert calls[0]['max_completion_tokens']==18000
    assert calls[1]['messages'][2]['reasoning_content']=='opaque context must stay ephemeral'
    assert len(calls[2]['messages'])==2 and calls[2]['tool_choice']=='none'
    recovery=json.loads(calls[2]['messages'][1]['content'])
    assert recovery['measured_task_evidence'][0]['tool']=='recall'
    assert not recovery['incomplete_public_output']['executable']
    assert 'opaque context must stay ephemeral' not in json.dumps(calls[2]['messages'])
    assert store.get_campaign(cid)['token_limit']==150000


def test_failed_director_resume_finishes_from_its_own_tool_evidence(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000); captured=[]
    store.start_task('same-director',cid,'director',{})
    store.event(cid,'director','TOOL_RESULT',{'task_id':'same-director','tool':'inspect_data','result':{'date':'2026-01-01','median_spread_bps':0.5}})
    store.fail_task('same-director',RuntimeError('truncated'))
    def create(**kwargs):
        captured.append(kwargs)
        return response(content='{"mission":"Use the measured costs"}')
    result=runtime(store,Settings(),create).run(cid,'director',{'objective':'Continue','memory':[{'huge':'unrelated'}]},ResearchTools(cid,store,None,None,Settings()),'same-director')
    assert result['mission'] and len(captured)==1
    task=json.loads(captured[0]['messages'][1]['content'])
    assert task['measured_task_evidence'][0]['result']['median_spread_bps']==0.5
    assert 'unrelated' not in json.dumps(task)
    assert captured[0]['tool_choice']=='none' and captured[0]['max_completion_tokens']==18000


def test_failed_draft_resumes_exact_code_without_unrelated_candidate_adoption(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000)
    store.candidate(cid,spec(title='Unrelated prior work'),CODE)
    bad=CODE.replace("df['qi_5'].clip(-1.0,1.0)","pl.arange(0,len(df),eager=True)")
    first=runtime(store,replace(Settings(),max_agent_steps=1),lambda **kw:response(submit(bad)))
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(RuntimeError,match='step task budget'):
        first.run(cid,'engineer',{},tools,'unfinished')
    with store.connect() as db:
        assert db.execute("SELECT status FROM tasks WHERE id='unfinished'").fetchone()[0]=='FAILED'
    def resume(**kwargs):
        task=json.loads(kwargs['messages'][1]['content'])
        assert task['unfinished_program']['args']['code']==bad
        assert task['resume_implementation'] and not task.get('resume_finalization')
        assert not any(t['function']['name']=='inspect_data' for t in kwargs['tools'])
        return response(submit(CODE))
    result=runtime(store,Settings(),resume).run(cid,'engineer',{},tools,'unfinished')
    assert result['artifact_receipt']


def test_budget_error_records_terminal_task_without_provider_call(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10)
    def unexpected(**kwargs): pytest.fail('Budget must be checked before provider')
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(RuntimeError,match='token budget exhausted'):
        runtime(store,Settings(),unexpected).run(cid,'engineer',{},tools,'budget')
    with store.connect() as db:
        row=db.execute("SELECT status,pid FROM tasks WHERE id='budget'").fetchone()
        assert row['status']=='FAILED' and row['pid'] is None
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==0


def test_repeated_identical_tool_failure_stops_paid_loop_and_retains_draft(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000); calls=[]
    bad=CODE.replace("df['qi_5'].clip(-1.0,1.0)","pl.arange(0,len(df),eager=True)")
    def create(**kwargs):
        calls.append(kwargs); return response(submit(bad))
    with pytest.raises(AgentTaskFailure) as caught:
        runtime(store,replace(Settings(),max_agent_steps=8),create).run(cid,'engineer',{},ResearchTools(cid,store,None,None,Settings()),'repeat-error')
    assert caught.value.error_type=='RepeatedToolFailure' and not caught.value.retryable
    assert len(calls)==3 and not store.candidates(cid)
    with store.connect() as db:
        assert db.execute("SELECT status FROM tasks WHERE id='repeat-error'").fetchone()[0]=='FAILED'
        assert db.execute("SELECT COUNT(*) FROM requests WHERE actual=100 AND status='COMPLETE'").fetchone()[0]==3
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='PROGRAM_ATTEMPT'").fetchone()[0]==3
        error=json.loads(db.execute("SELECT payload FROM events WHERE kind='REPEATED_TOOL_FAILURE'").fetchone()[0])
        assert error['count']==3


def test_summary_without_implementation_is_not_a_completed_engineer_task(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(RuntimeError,match='step task budget'):
        runtime(store,replace(Settings(),max_agent_steps=1),lambda **kw:response(content='{"claim":"done"}')).run(
            cid,'engineer',{},tools,'no-code')
    with store.connect() as db:
        assert db.execute("SELECT status FROM tasks WHERE id='no-code'").fetchone()[0]=='FAILED'
        assert not db.execute("SELECT 1 FROM events WHERE kind='AGENT_COMPLETED'").fetchone()


def test_live_task_owner_cannot_be_overwritten_or_marked_failed(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    store.start_task('owned',cid,'engineer',{})
    tools=ResearchTools(cid,store,None,None,Settings())
    # A second invocation in the same live process also cannot steal ownership.
    with pytest.raises(RuntimeError,match='live process'):
        runtime(store,Settings(),lambda **kw:pytest.fail()).run(cid,'engineer',{},tools,'owned')
    with store.connect() as db:
        row=db.execute('SELECT status,pid FROM tasks').fetchone()
        assert row['status']=='RUNNING' and row['pid']==os.getpid()


def test_truncated_public_submission_is_saved_but_never_executed(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    truncated=response(submit(CODE)); truncated.choices[0].finish_reason='length'
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(RuntimeError,match='no synthetic replacement'):
        runtime(store,Settings(),lambda **kw:truncated).run(cid,'engineer',{},tools,'truncated')
    assert not store.candidates(cid)
    with store.connect() as db:
        partial=json.loads(db.execute("SELECT payload FROM events WHERE kind='INCOMPLETE_PUBLIC_OUTPUT'").fetchone()[0])
        assert not partial['executable']
        assert json.loads(partial['tool_calls'][0]['arguments'])['code']==CODE
        assert 'opaque context must stay ephemeral' not in '\n'.join(r[0] for r in db.execute('SELECT payload FROM events'))
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==2
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='COMPLETION_RECOVERY'").fetchone()[0]==1


def test_historian_summary_cannot_replace_required_grounded_lessons(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(RuntimeError,match='step task budget'):
        runtime(store,replace(Settings(),max_agent_steps=1),lambda **kw:response(content='{"claim":"remembered"}')).run(
            cid,'historian',{'cohort':[{'candidate_id':'unrecorded'}]},tools,'missing-lesson')
    with store.connect() as db:
        assert db.execute("SELECT status FROM tasks WHERE id='missing-lesson'").fetchone()[0]=='FAILED'


def test_dead_task_is_not_displayed_as_live(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    store.start_task('dead',cid,'engineer',{})
    with store.connect() as db:
        db.execute("UPDATE tasks SET pid=1073741823 WHERE id='dead'")
    task=store.status()['tasks'][0]
    assert task['status']=='INTERRUPTED' and task['declared_status']=='RUNNING' and not task['running']


def test_budget_revision_is_explicit_and_preserves_ambiguous_charges(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},1000)
    request=store.reserve(cid,'engineer',800); store.finish_request(request,None,'FAILED')
    with pytest.raises(ValueError,match='below charged'):
        store.revise_budget(cid,799,'Cannot erase an ambiguous bill')
    result=store.revise_budget(cid,2000,'Operator changes the test allowance')
    assert result['charged_or_reserved']==800 and not result['automatic']
    with store.connect() as db:
        assert db.execute('SELECT token_limit FROM campaigns').fetchone()[0]==2000
        assert json.loads(db.execute("SELECT payload FROM events WHERE kind='BUDGET_REVISED'").fetchone()[0])['old_limit']==1000


def test_preflight_checks_actual_worker_and_rejects_future_dependence(tmp_path):
    from tests.test_quantdesk import frame
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'fixture'},100000)
    class Catalog:
        def load(self,record,rows):
            assert rows==3000
            return frame(),{}
    tools=ResearchTools(cid,store,Catalog(),None,Settings())
    passed=tools.call('engineer','check_program',{'spec':spec(),'code':CODE})
    assert passed['rows']==80 and passed['status']=='BOUNDED_PROGRAM_CHECK_PASSED'
    bad=CODE.replace("df['qi_5'].clip(-1.0,1.0)","(df['qi_5']-df['qi_5'].mean()).clip(-1.0,1.0)")
    with pytest.raises(ValueError,match='Causality failure'):
        tools.call('engineer','check_program',{'spec':spec(),'code':bad})
    assert not store.candidates(cid)


def test_paper_stops_after_a_strategy_failure_without_repeating_inference(tmp_path,monkeypatch):
    from tests.test_quantdesk import bundle
    import quantdesk.paper as paper
    calls=[]
    def broken(*args):
        calls.append(1)
        raise ValueError('Actual strategy failed')
    monkeypatch.setattr(paper,'run_signal',broken)
    session=paper.PaperSession(bundle(tmp_path),tmp_path,Settings(),True)
    def fetch(*args):
        return {'coin':'BTC','time':int(paper.time.time()*1000),
            'levels':[[{'px':'100','sz':'10','n':1}],[{'px':'100.01','sz':'10','n':1}]]}
    result=session.run(samples=3,interval=1,fetch=fetch)
    assert len(calls)==1 and result['strategy_errors']==1 and result['halted']
    state=json.loads((session.directory/'status.json').read_text())
    assert state['status']=='HALTED' and state['orders_submitted_to_exchange']==0


def test_development_coverage_checks_global_exposure_without_listing_holdout(tmp_path):
    store=Store(tmp_path)
    plan={'development':[{'date':'2026-01-01'},{'date':'2026-01-02'}],
        'holdout':[{'date':'2026-01-03'}]}
    cid=store.campaign(plan,100000)
    store.expose_dates('earlier-campaign',plan['development'][:1],'development')
    result=ResearchTools(cid,store,None,None,Settings()).call('director','development_coverage',{})
    assert len(result['records'])==2
    assert result['records'][0]['globally_exposed'] and not result['records'][1]['globally_exposed']
    assert result['records'][0]['first_recorded_exposure']['campaign']=='earlier-campaign'
    assert '2026-01-03' not in json.dumps(result)


def test_research_map_does_not_count_ungrounded_narratives_as_learning(tmp_path):
    from quantdesk.research_map import research_map
    store=Store(tmp_path); cid=store.campaign({},1000)
    store.memory(cid,'a','LESSON',{'lesson':'A bare assertion'})
    store.memory(cid,'b','LESSON',{'lesson':'Measured failure','grounding':{
        'candidate_status':'PROPOSED','development_experiment':{'pnl':0,'trades':0}}})
    result=research_map(store,cid)
    assert result['grounded_lessons']==1 and result['ungrounded_lesson_records']==1


def test_broker_cannot_fill_before_measured_decision_availability():
    from quantdesk.execution import Broker
    from tests.test_quantdesk import quote
    broker=Broker(spec(),Settings().execution_dict())
    broker.step(quote(0),1.0,decision_ts_ms=2400)
    assert broker.pending['arrival']==3400
    broker.step(quote(1000),0.0)
    broker.step(quote(3000),0.0)
    assert broker.account.quantity==0
    broker.step(quote(4000),0.0)
    assert broker.account.quantity>0 and broker.account.entry_time==4000


def test_decision_delay_cannot_cross_into_a_new_funding_hour():
    from quantdesk.execution import Broker
    from tests.test_quantdesk import quote
    broker=Broker(spec(),Settings().execution_dict())
    broker.step(quote(3590000),1.0,decision_ts_ms=3660000)
    assert broker.pending is None and broker.orders==0


@pytest.mark.parametrize('raises',[False,True])
def test_slow_inference_halts_without_fabricating_a_close_on_stale_prices(tmp_path,monkeypatch,raises):
    import numpy as np
    import quantdesk.paper as paper
    from tests.test_quantdesk import bundle,quote
    clock=[0.0]
    monkeypatch.setattr(paper.time,'monotonic',lambda:clock[0])
    def too_slow(code,frame,*args):
        clock[0]=10.0
        if raises:
            raise ValueError('Late worker failure')
        return np.ones(len(frame))
    monkeypatch.setattr(paper,'run_signal',too_slow)
    session=paper.PaperSession(bundle(tmp_path),tmp_path,Settings(),True)
    broker=session.members[0]['broker']; broker.fill(quote(0),1,1)
    book={'coin':'BTC','time':1000,'levels':[[{'px':'100','sz':'10','n':1}],[{'px':'100.01','sz':'10','n':1}]]}
    session.consume(book,now_ms=1000)
    assert session.halted and session.stale_decisions==1
    assert broker.account.quantity==1 and not broker.trades and broker.pending is None
