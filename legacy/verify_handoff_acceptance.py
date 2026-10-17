"""One bounded real scientist contract test; never restarts the failed pilot."""
import hashlib
import argparse
import json
import os
import time
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.agents import AgentRuntime,ResearchTools
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.planning import planning_contract
from quantdesk.settings import Settings
from quantdesk.store import Store


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='Reviewed compact finalization, only within the original two-request/60k/15-minute contract')
    args=parser.parse_args()
    settings=replace(Settings(),max_agent_steps=2,max_completion_tokens=18000,request_timeout=300)
    directory=settings.home/'verification'/'handoff-acceptance'; directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'run.json'
    if receipt.exists() and not args.resume:
        print(receipt.read_text(encoding='utf-8'))
        return
    cid='2b22519842f6'; task_id='handoff-acceptance:'+cid+':scientist:spec-v1'; limit=60000
    resuming=receipt.exists()
    state={'status':'PREPARING','pid':os.getpid(),'campaign':cid,'task_id':task_id,'task_token_limit':limit,
        'started_utc':datetime.now(timezone.utc).isoformat(),'scope':'Real scientist design repair and validated handoff only; not a company cycle or fourth pilot recovery',
        'criteria':{'valid_strategy_spec':True,'actual_feasibility_gate':True,'new_provider_calls_max':2,
            'completion_limit_per_call':18000,'provider_timeout_seconds':300,'cooperative_minutes':15,
            'original_campaign_token_limit':750000,'budget_increase':False,'new_campaign':False,
            'no_holdout_or_promotion':True},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')}}
    if resuming:
        from quantdesk.store import process_alive
        old=json.loads(receipt.read_text(encoding='utf-8'))
        if old['status']=='RUNNING' and process_alive(old['pid']):
            raise ValueError('Acceptance process is live; do not duplicate it')
        if old['status']!='FAILED' or old.get('new_provider_requests')!=1 or old.get('error_type')!='RuntimeError' or 'Task token budget exhausted' not in old.get('error','') or old.get('resumptions',0):
            raise ValueError('No permitted reviewed compact finalization remains')
        if time.time()-datetime.fromisoformat(old['started_utc']).timestamp()>=900:
            raise ValueError('Original acceptance time boundary reached')
        (directory/'first-attempt.json').write_text(json.dumps(old,indent=2,allow_nan=False),encoding='utf-8')
        state.update(started_utc=old['started_utc'],before=old['before'],resumptions=1,
            original_source_sha256=old['source_sha256'],remaining_provider_requests=1,
            resume_reason='Reviewed oversized retrieval context; fresh public-artifact finalization with the same frozen task/campaign budget and original time boundary')
        settings=replace(settings,max_agent_steps=1)
    else:
        with receipt.open('x',encoding='utf-8') as stream:json.dump(state,stream,indent=2)
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        staged=receipt.with_suffix('.partial'); staged.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8'); staged.replace(receipt)
    store=Store(settings.home)
    def snapshot():
        with store.connect() as db:
            return {'usage':dict(db.execute('SELECT COUNT(*) requests,COALESCE(SUM(actual),0) actual,COALESCE(SUM(COALESCE(actual,reserved)),0) charged FROM requests WHERE campaign=?',(cid,)).fetchone()),
                'cycles':[dict(r) for r in db.execute('SELECT id,status FROM cycles WHERE campaign=? ORDER BY created',(cid,))],
                'holdout_dates':db.execute('SELECT COUNT(*) FROM holdout_dates').fetchone()[0]}
    before=state.get('before') or snapshot(); state['before']=before; persist()
    try:
        campaign=store.get_campaign(cid)
        assert campaign['token_limit']==750000 and campaign['token_limit']-before['usage']['charged']>=limit
        assert any(r['id']=='6792345b30f3' and r['status']=='ABANDONED' for r in before['cycles'])
        catalog=Catalog(settings.data_dir,settings.home); evaluator=Evaluator(store,catalog,settings)
        tools=ResearchTools(cid,store,catalog,evaluator,settings,cycle_id='standalone-handoff-test')
        elapsed=time.time()-datetime.fromisoformat(state['started_utc']).timestamp()
        with store.lease('research',seconds=1800),store.research_deadline(cid,time.monotonic()+max(0,900-elapsed)):
            contract=planning_contract(store,catalog,cid,settings)
            prior=store.get_candidate('fb9f302b011322bd')
            diagnostic=tools._call('acceptance_preflight','data_feasibility',{'spec':prior['spec'],'day_index':24})
            task={'task_token_limit':limit,'require_strategy_spec':True,'data_contract':contract,
                'mission':{'objective':'Repair or redesign the earlier slow-pressure proposal into a falsifiable implementation design that the actual segmented data and supported tools can evaluate. Use your scientific judgment to change the effective history or holding horizon explicitly; do not silently shrink warmup while claiming mature unsupported windows.',
                    'prior_candidate_id':prior['id'],'prior_spec':prior['spec'],
                    'earlier_director_horizon_seconds':1800,'earlier_scientist_proposed_holding_seconds':300,
                    'prior_window_idea':'Fully mature trailing 600-900 second deviation/flow persistence',
                    'known_economic_anchor':'Earlier real 60-second candidate 37e7854f12d976e8 had zero trades on measured full development dates under its costs/configuration. This does not condemn a feature family.',
                    'interpretation':'Prior narrative IC, uncertainty margins and diffusion extrapolations are assertions unless backed by actual tool evidence. No profitable mechanism is assumed.'},
                'measured_data_feasibility':diagnostic,'costs':settings.execution_dict(),
                'required_deliverable':'Use tools only for a decisive unresolved uncertainty. Return exactly one JSON handoff with hypothesis, supported effective history, baseline/ablation, test dates and stop conditions, limitations, and strategy_spec matching the schema. No authored code or submission is required in this scientist contract test. Distinguish structural feasibility from predictive and net economic evidence. Do not claim a completed company cycle, approval, confirmed alpha or learning benefit.'}
            state['status']='RUNNING'; state['provider']='configured actual provider'; persist()
            store.event(cid,'operator','HANDOFF_ACCEPTANCE_PLANNED',{'task_id':task_id,'task_token_limit':limit,
                'scope':state['scope'],'original_pilot_criterion_unchanged':True,'original_budget_increased':False})
            agent=AgentRuntime(store,settings)
            state.update(provider=agent.provider,model=agent.model); persist()
            result=agent.run(cid,'scientist',task,tools,task_id)
            state.update(status='COMPLETE',result=result)
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            error_type=type(error).__name__,error=str(error)[:2000])
    finally:
        after=snapshot(); state['after']=after
        state['new_provider_requests']=after['usage']['requests']-before['usage']['requests']
        state['new_actual_tokens']=after['usage']['actual']-before['usage']['actual']
        state['new_charged_tokens']=after['usage']['charged']-before['usage']['charged']
        state['remaining_original_tokens']=750000-after['usage']['charged']
        state['cycles_unchanged']=after['cycles']==before['cycles']
        state['new_holdout_exposures']=after['holdout_dates']-before['holdout_dates']
        with store.connect() as db:
            state['attributed_requests']=[dict(r) for r in db.execute('SELECT requests.id,status,reserved,actual FROM requests JOIN request_tasks ON request_tasks.request_id=requests.id WHERE task_id=?',(task_id,))]
        assert state['cycles_unchanged'] and state['new_holdout_exposures']==0
        persist()
    print(json.dumps({k:v for k,v in state.items() if k not in {'result','source_sha256'}},indent=2),flush=True)


if __name__=='__main__':main()
