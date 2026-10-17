"""One real streamed implementation delivery, then independent local checks.

This new engineering scope is not a retry of either exhausted scientist test.
No additional requests, budget reset, cycle creation, holdout or promotion.
"""
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.agents import AgentRuntime,ResearchTools
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.planning import planning_contract
from quantdesk.settings import Settings
from quantdesk.store import Store,digest,process_alive

CID='2b22519842f6'
TASK='streamed-engineering:'+CID+':implementation-v1'
LIMIT=35000
INDEX=22


class DeliveryTools(ResearchTools):
    def available(self,role):
        if role=='engineer': return [self.tool_specs['register_candidate']]
        return super().available(role)


def snapshot(store):
    with store.connect() as db:
        return {'usage':dict(db.execute('SELECT COUNT(*) requests,COALESCE(SUM(actual),0) actual,COALESCE(SUM(COALESCE(actual,reserved)),0) charged FROM requests WHERE campaign=?',(CID,)).fetchone()),
                'cycles':[dict(r) for r in db.execute('SELECT id,status FROM cycles WHERE campaign=? ORDER BY created',(CID,))],
                'holdout_dates':db.execute('SELECT COUNT(*) FROM holdout_dates').fetchone()[0]}


def main():
    settings=replace(Settings(),max_agent_steps=1,max_completion_tokens=16000,request_timeout=900)
    directory=settings.home/'verification'/'streamed-engineer'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'run.json'
    if path.exists():
        old=json.loads(path.read_text()); print(json.dumps({'status':old['status'],'pid':old['pid'],
            'process_alive':process_alive(old['pid']),'receipt':str(path),'new_call_started':False})); return
    store=Store(settings.home); before=snapshot(store); campaign=store.get_campaign(CID)
    assert campaign['token_limit']==750000 and 750000-before['usage']['charged']>=LIMIT
    with store.connect() as db: assert not db.execute('SELECT 1 FROM tasks WHERE id=?',(TASK,)).fetchone()
    state={'status':'PREPARING','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'campaign':CID,'task_id':TASK,'task_token_limit':LIMIT,'before':before,
        'scope':'One real engineering implementation delivery through streamed SDK, followed by actual local numerical and development checks; not a company cohort',
        'criteria':{'maximum_provider_requests':1,'completion_limit':16000,'request_deadline_seconds':900,
            'cooperative_total_seconds':1800,'fixed_development_index':INDEX,'original_limit':750000,
            'budget_increase':False,'pilot_restart':False,'scientist_retry':False,'no_holdout_or_promotion':True},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')},
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with path.open('x',encoding='utf-8') as stream: json.dump(state,stream,indent=2)
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        stage=path.with_suffix('.partial'); stage.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8'); stage.replace(path)
    catalog=Catalog(settings.data_dir,settings.home); evaluator=Evaluator(store,catalog,settings)
    tools=DeliveryTools(CID,store,catalog,evaluator,settings,'standalone-streamed-engineering')
    try:
        with store.lease('research',seconds=2000),store.research_deadline(CID,time.monotonic()+1800):
            contract=planning_contract(store,catalog,CID,settings)
            assert catalog.cached(tools.plan['development'][INDEX],None)
            prior=store.get_candidate('fb9f302b011322bd')
            diagnostic=tools._call('engineering_preflight','data_feasibility',{'spec':prior['spec'],'day_index':INDEX})
            task={'task_token_limit':LIMIT,'mission':{'objective':'Build a genuinely implementable revision or distinct causal microstructure design supported by observed history. Think critically about pressure persistence, liquidity replenishment and continuation versus reversal; choose a falsifiable mechanism after modeled costs. No alpha is assumed.',
                    'prior_incompatible_spec':prior['spec'],'prior_candidate_id':prior['id'],
                    'earlier_real_counterevidence':'Candidate 37e7854f12d976e8 produced zero trades on observed complete development dates. This does not condemn a factor family.'},
                'data_contract':contract,'measured_data_feasibility':diagnostic,
                'costs':settings.execution_dict(),'fixed_development_index':INDEX,
                'prepared_training_information':'Development indices 0..24 are complete prepared November dates. Replay is predeclared on index 22, November 23. A fitted recipe must use earlier dates only and min_train_days <= 22; inner_days + 3 <= min_train_days <= train_days.',
                'required_deliverable':'You have one complete streamed response and the actual data contract. Deliver exact code and specification with register_candidate. Use your intelligence to redesign unsupported long-history mechanisms explicitly; do not lower warmup while still claiming fully mature 600-900-row inputs. Prefer an interpretable causal state/regime design and taker mode. You may use an earlier-only fitted recipe, with custom feature columns if helpful. Native checks and a fixed full-day development replay will run after delivery; registration itself proves no numerical correctness or economics. All code must follow the full supplied numerical contract. Rationale and falsification must identify baseline/ablation, effective feature history and the observation that would reject this exact mechanism. Do not call unavailable tools, return a prose-only promise, invent measured IC/PnL, or claim validation/approval.'}
            state.update(status='RUNNING',phase='PROVIDER',public_task=task); persist()
            store.event(CID,'operator','STREAMED_ENGINEERING_PLANNED',{'task_id':TASK,'scope':state['scope'],
                'task_token_limit':LIMIT,'maximum_provider_requests':1,'fixed_development_index':INDEX,
                'original_budget_increased':False})
            agent=AgentRuntime(store,settings); assert agent.streaming and agent.provider=='mimo'
            state.update(provider=agent.provider,model=agent.model,streaming=True); persist()
            result=agent.run(CID,'engineer',task,tools,TASK)
            ids=result.get('committed_candidate_ids',[])
            assert len(ids)==1
            candidate=store.get_candidate(ids[0]); assert candidate['campaign']==CID
            state.update(phase='LOCAL_NUMERICAL_CHECK',candidate_id=candidate['id'],
                code_hash=digest(candidate['code']),spec_hash=digest(candidate['spec']),delivery_receipt=result); persist()
            checks=ResearchTools(CID,store,catalog,evaluator,settings,tools.cycle_id)
            numerical=checks.call('engineer','check_candidate',{'candidate_id':candidate['id'],'day_index':INDEX})
            state['numerical_check']=numerical; persist()
            assert numerical['status']=='BOUNDED_PROGRAM_CHECK_PASSED'
            state.update(phase='LOCAL_DEVELOPMENT_REPLAY'); persist()
            if 'model' in candidate['spec']:
                feasibility=checks.call('engineer','candidate_feasibility',{'candidate_id':candidate['id'],'day_index':INDEX})
                state['recipe_feasibility']=feasibility; persist()
                assert feasibility['status']=='LABELS_AVAILABLE'
                replay=checks.call('engineer','walk_forward_candidate',{'candidate_id':candidate['id'],'folds':1,'start_index':INDEX,'end_index':INDEX})
            else:
                replay=checks.call('engineer','experiment_candidate',{'candidate_id':candidate['id'],'days':1,'day_indices':[INDEX]})
            state.update(status='COMPLETE',phase='COMPLETE',replay=replay,
                interpretation='Real streamed engineer authored an immutable implementation; independent numerical checks and one fixed actual development replay completed. Economics remain the observed replay only; no scientific handoff acceptance, full company cohort, validation, alpha, learning benefit, portfolio or promotion is established.')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            error_type=type(error).__name__,error=str(error)[:2000])
    finally:
        after=snapshot(store); state['after']=after
        state['new_provider_requests']=after['usage']['requests']-before['usage']['requests']
        state['new_known_actual_tokens']=after['usage']['actual']-before['usage']['actual']
        state['new_charged_tokens']=after['usage']['charged']-before['usage']['charged']
        state['remaining_original_tokens']=750000-after['usage']['charged']
        state['cycles_unchanged']=after['cycles']==before['cycles']
        state['new_holdout_exposures']=after['holdout_dates']-before['holdout_dates']
        with store.connect() as db:
            state['attributed_requests']=[dict(r) for r in db.execute('SELECT requests.id,status,reserved,actual FROM requests JOIN request_tasks ON request_tasks.request_id=requests.id WHERE task_id=?',(TASK,))]
            state['stream_progress']=[json.loads(r[0]) for r in db.execute("SELECT payload FROM events WHERE campaign=? AND kind='PROVIDER_PROGRESS' AND json_extract(payload,'$.task_id')=? ORDER BY id",(CID,TASK))]
        persist()
        assert state['cycles_unchanged'] and state['new_holdout_exposures']==0 and len(state['attributed_requests'])<=1
    print(json.dumps({k:v for k,v in state.items() if k not in {'public_task','source_sha256','stream_progress','replay','numerical_check'}},indent=2),flush=True)


if __name__=='__main__':main()
