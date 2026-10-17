"""Actual prepared-data decision checks, then an opt-in single real protocol call.

The paid probe has a new finite scope. It never resumes the exhausted historical
handoff allowance, restarts the failed pilot, or claims company acceptance.
"""
import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from quantdesk.agents import AgentRuntime, ResearchTools
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.planning import planning_contract
from quantdesk.proposals import load
from quantdesk.settings import Settings
from quantdesk.store import Store, digest, process_alive

CID='2b22519842f6'
TASK='scientific-protocol:'+CID+':scientist:decisions-v1'
LIMIT=35000


def sources():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')}


def snapshot(store):
    with store.connect() as db:
        return {'usage':dict(db.execute('SELECT COUNT(*) requests,COALESCE(SUM(actual),0) actual,COALESCE(SUM(COALESCE(actual,reserved)),0) charged FROM requests WHERE campaign=?',(CID,)).fetchone()),
                'cycles':[dict(r) for r in db.execute('SELECT id,status FROM cycles WHERE campaign=? ORDER BY created',(CID,))],
                'holdout_dates':db.execute('SELECT COUNT(*) FROM holdout_dates').fetchone()[0],
                'candidate_statuses':[dict(r) for r in db.execute('SELECT status,COUNT(*) n FROM candidates GROUP BY status')]}


def offline(settings,production,catalog,directory):
    before=snapshot(production)
    plan=production.get_campaign(CID)['plan']
    originals={i:production.get_candidate(i) for i in ('37e7854f12d976e8','fb9f302b011322bd')}
    # Only the isolated verification ledger receives synthetic role ownership.
    # Inputs and decisions are actual prepared-data compatibility, not LLM work.
    isolated=Store(directory/'offline-ledger')
    cid=isolated.campaign(plan,1000); cycle=isolated.start_cycle(cid)
    local_settings=replace(settings,home=isolated.home)
    tools=ResearchTools(cid,isolated,catalog,None,local_settings,cycle)
    contract=planning_contract(isolated,catalog,cid,local_settings)
    good=originals['37e7854f12d976e8']; bad=originals['fb9f302b011322bd']
    index=next(i for i,r in enumerate(plan['development']) if r['date']=='2025-11-23')
    assert catalog.cached(plan['development'][index],None)
    task=cycle+':scientist:proposal'; isolated.start_task(task,cid,'scientist',{}); tools.active_task_id=task
    accepted=tools.call('scientist','submit_proposal',{
        'strategy_spec':good['spec'],'effective_history_rows':good['spec']['warmup_rows']+1,'development_indices':[index],
        'hypothesis':'Offline compatibility check of an actual earlier MiMo design; no new scientific inference',
        'implementation_design':'Retain exact stored recipe; this probe does not execute its code',
        'baseline':'Existing cash and fixed-trading benchmark contract',
        'ablation':'Existing custom-feature removal benchmark contract',
        'stop_condition':'No economic conclusion from compatibility alone',
        'limitations':'Operator-authored verification arguments, not new discovered strategy or model output'})
    isolated.finish_task(task,accepted)
    task=cycle+':scientist:stop'; isolated.start_task(task,cid,'scientist',{}); tools.active_task_id=task
    stopped=tools.call('scientist','stop_mission',{'basis':'DATA_INCOMPATIBLE','proposed_spec':bad['spec'],
        'day_index':index,'reason':'Existing proposed warmup and holding cannot mature inside the observed segments',
        'next_experiment':'Design a supported mechanism and explicitly account for its mature history',
        'limitations':'Actual metadata check only; no independent scientific review or economics supplied'})
    isolated.finish_task(task,stopped)
    assert snapshot(production)==before
    assert all(production.get_candidate(i)==v for i,v in originals.items())
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),
        'input_campaign':CID,'verification_ledger':str(isolated.path),'development_date':plan['development'][index]['date'],
        'planning_contract':contract,'proposal':accepted,'stop':stopped,
        'original_candidates_preserved':True,'production_usage_cycles_holdout_unchanged':True,
        'new_provider_requests':0,'new_promotions':0,
        'interpretation':'Actual prepared data compatibility through the decision service; operator-authored probe in isolated ledger. No code replay, independent review, company completion, alpha or learning benefit.'}
    path=directory/'actual-data.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def real(settings,store,catalog,directory):
    path=directory/'real-run.json'
    if path.exists():
        state=json.loads(path.read_text(encoding='utf-8'))
        print(json.dumps({'status':state['status'],'pid':state['pid'],
            'process_alive':process_alive(state['pid']),'receipt':str(path),'new_call_started':False}))
        return
    before=snapshot(store); campaign=store.get_campaign(CID)
    assert campaign['token_limit']==750000 and campaign['token_limit']-before['usage']['charged']>=LIMIT
    assert any(c['id']=='6792345b30f3' and c['status']=='ABANDONED' for c in before['cycles'])
    with store.connect() as db:
        assert not db.execute('SELECT 1 FROM tasks WHERE id=?',(TASK,)).fetchone()
    state={'status':'PREPARING','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'campaign':CID,'task_id':TASK,'task_token_limit':LIMIT,'before':before,'source_sha256':sources(),
        'scope':'One separately bounded real scientist artifact protocol test; no company cycle or pilot recovery',
        'criteria':{'maximum_new_requests':1,'completion_limit':16000,'request_timeout_seconds':300,
                    'cooperative_seconds':600,'original_campaign_limit':750000,'budget_increase':False},
        'provider_tool_selection':'MiMo automatic selection; trusted local service enforces completion, not required mode'}
    with path.open('x',encoding='utf-8') as stream: json.dump(state,stream,indent=2)
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        staged=path.with_suffix('.partial'); staged.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8'); staged.replace(path)
    configured=replace(settings,max_agent_steps=1,max_completion_tokens=16000,request_timeout=300)
    tools=ResearchTools(CID,store,catalog,Evaluator(store,catalog,configured),configured,'standalone-scientific-protocol-test')
    try:
        with store.lease('research',seconds=900),store.research_deadline(CID,time.monotonic()+600):
            contract=planning_contract(store,catalog,CID,configured)
            prior=store.get_candidate('fb9f302b011322bd')
            feasibility=tools._call('protocol_preflight','data_feasibility',{'spec':prior['spec'],'day_index':24})
            task={'task_token_limit':LIMIT,'proposal_workflow':True,'data_contract':contract,
                'mission':{'objective':'Use your scientific judgment to redesign the earlier long-history pressure mechanism into a feasible falsifiable design, or stop this allocation with actual grounding. A compatible proposal is not assumed profitable.',
                    'prior_candidate_id':prior['id'],'prior_spec':prior['spec'],
                    'prior_history_claim':'Fully mature 600-900-second deviation/flow persistence',
                    'observed_counterevidence':'Earlier real candidate 37e7854f12d976e8 produced zero trades on measured complete development dates; this does not reject a feature family.'},
                'measured_data_feasibility':feasibility,'costs':configured.execution_dict(),
                'prepared_development_indices':list(range(25)),
                'required_deliverable':'You have one response and actual public metadata. Call exactly one submit_proposal or stop_mission tool with its complete arguments. For a proposal select 1..3 prepared outcome-blind development indices, supported fully mature history, precise implementation design, baseline, ablation and stop condition. Use a fitted outer date only with sufficient earlier training dates. Explicitly revise an unsupported history mechanism, not just its declared warmup. A grounded stop is equally valid. Do not return a prose-only summary or claim code, PnL, approval, a company completion, or learning benefit. The trusted service validates and persists the actual decision.'}
            state.update(status='RUNNING',public_task=task); persist()
            store.event(CID,'operator','SCIENTIFIC_PROTOCOL_PLANNED',{'task_id':TASK,'scope':state['scope'],
                'task_token_limit':LIMIT,'maximum_requests':1,'original_pilot_criterion_unchanged':True})
            agent=AgentRuntime(store,configured); state.update(model=agent.model,provider=agent.provider); persist()
            result=agent.run(CID,'scientist',task,tools,TASK)
            decision=load(store,result['decision_id'],CID)
            assert decision['task_id']==TASK and decision['cycle']==tools.cycle_id
            state.update(status='COMPLETE',result=result,
                interpretation='Real scientist committed a trusted grounded allocation artifact; no engineer, code replay, independent review, company cycle, alpha or learning-benefit claim')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            error_type=type(error).__name__,error=str(error)[:2000])
    finally:
        after=snapshot(store); state['after']=after
        state['new_provider_requests']=after['usage']['requests']-before['usage']['requests']
        state['new_actual_tokens']=after['usage']['actual']-before['usage']['actual']
        state['new_charged_tokens']=after['usage']['charged']-before['usage']['charged']
        state['remaining_original_tokens']=750000-after['usage']['charged']
        state['cycles_unchanged']=after['cycles']==before['cycles']
        state['new_holdout_exposures']=after['holdout_dates']-before['holdout_dates']
        state['candidate_statuses_unchanged']=after['candidate_statuses']==before['candidate_statuses']
        with store.connect() as db:
            state['attributed_requests']=[dict(r) for r in db.execute('SELECT requests.id,status,reserved,actual FROM requests JOIN request_tasks ON request_tasks.request_id=requests.id WHERE task_id=?',(TASK,))]
        persist()
        assert state['cycles_unchanged'] and state['new_holdout_exposures']==0 and state['candidate_statuses_unchanged']
        assert len(state['attributed_requests'])<=1
    print(json.dumps({k:v for k,v in state.items() if k not in {'public_task','result','source_sha256'}},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--real-provider',action='store_true'); args=parser.parse_args()
    settings=Settings(); store=Store(settings.home); catalog=Catalog(settings.data_dir,settings.home)
    directory=settings.home/'verification'/'scientific-protocol'; directory.mkdir(parents=True,exist_ok=True)
    if args.real_provider:
        real(settings,store,catalog,directory)
    else:
        result=offline(settings,store,catalog,directory)
        print(json.dumps({'receipt':str(directory/'actual-data.json'),'proposal_kind':result['proposal']['decision_kind'],
            'stop_status':result['stop']['decision']['grounding']['feasibility']['status'],'new_provider_requests':0},indent=2))


if __name__=='__main__': main()
