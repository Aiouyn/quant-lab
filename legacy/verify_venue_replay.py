"""Predeclared existing code/models and capture; no provider or promotion."""
import hashlib
import argparse
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.agents import ResearchTools,report_observation
from quantdesk.settings import Settings
from quantdesk.store import Store,process_alive
from quantdesk.planning import planning_contract
from quantdesk.data import Catalog
from verify_venue_study import snapshot

CAMPAIGN='2fb07460f9a0'
CAPTURE='hyperliquid-20260930-113005-436f57'
ARMS=[('existing-custom','37e7854f12d976e8','eb2ce19499dbc1779143e5b021632448d7f7eb94045e814692c010420b61577c'),
      ('existing-ablation','4987d273959987d1','0046af9e4f9eb063a91fea995f1e86314897cad3ba00f57a4722d159f917e8ce'),
      ('existing-fixed-reference','f1435f796b966260',None)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-current-implementation',action='store_true',help='Explicitly recheck the same predeclared artifacts after implementation changes; not new independent evidence')
    args=parser.parse_args()
    settings=Settings();store=Store(settings.home)
    directory=store.home/'verification'/'venue-replay';directory.mkdir(parents=True,exist_ok=True)
    output=directory/'run.json'
    if output.exists():
        old=json.loads(output.read_text())
        if old.get('status')=='RUNNING' and process_alive(old.get('pid')): raise RuntimeError('Existing verification process is live')
        if not args.verify_current_implementation:
            print('Existing bounded run:',old['status'],str(output));return
        backup=directory/'run-before-final-verification.json'
        if not backup.exists(): backup.write_bytes(output.read_bytes())
    state={'status':'RUNNING','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Retrospective target-venue execution sensitivity of three preselected existing artifacts. No newly discovered strategy or forward qualification.',
        'implementation_recheck':args.verify_current_implementation,
        'independence':'Repeated versions, fill arms and overlapping captures are not independent market evidence. No alpha significance claim.',
        'selection':'Known existing custom/ablated pair and fixed reference; frozen matching November 23 model cutoffs, selected before this capture replay. No model refitting or search.',
        'capture_id':CAPTURE,'predeclared_arms':ARMS,'before':snapshot(store),'results':[],
        'models_sha256_before':{key:hashlib.sha256((store.home/'models'/(key+'.json')).read_bytes()).hexdigest() for _,_,key in ARMS if key}}
    def persist():
        staged=output.with_suffix('.partial');staged.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8');staged.replace(output)
    started=time.monotonic();tools=ResearchTools(CAMPAIGN,store,None,None,settings)
    try:
        with store.lease('research',seconds=900),store.research_deadline(CAMPAIGN,started+600):
            persist()
            study=tools.call('falsifier','study_venue_capture',{'capture_id':CAPTURE})
            state['study_id']=study['study_id'];persist()
            for label,candidate,key in ARMS:
                print('REPLAY',label,candidate,flush=True)
                result=tools.call('falsifier','replay_venue_candidate',{'candidate_id':candidate,'study_id':study['study_id'],**({'model_key':key} if key else {})})
                state['results'].append({'label':label,**result});persist()
                print('RECORDED',label,result['report']['quote_arm']['pnl'],result['report']['depth_arm']['pnl'],flush=True)
            state['planning_context']=planning_contract(store,Catalog(settings.data_dir,settings.home),CAMPAIGN,settings)['target_venue_observations']
            state['condensed_memory']=report_observation({'tool':'recall','result':store.recall(10)})
            state['after']=snapshot(store)
            state['protected_ledger_unchanged']=state['before']==state['after']
            state['models_unchanged']=all(hashlib.sha256((store.home/'models'/(key+'.json')).read_bytes()).hexdigest()==checksum for key,checksum in state['models_sha256_before'].items())
            state['latest_context_has_exact_study']=state['study_id'] in {v['study_id'] for v in state['planning_context']}
            state['new_provider_requests']=0;state['orders_submitted_to_exchange']=0;state['new_holdout_exposures']=0
            state['new_economic_experiments']=sum(not r['already_measured'] for r in state['results'])
            state['fill_assumption_arms']=2*state['new_economic_experiments']
            if not all(state[k] for k in ['protected_ledger_unchanged','models_unchanged','latest_context_has_exact_study']):
                raise RuntimeError('Venue replay acceptance invariant failed')
            state['status']='COMPLETE'
    except Exception as error:
        state['status']='FAILED';state['error_type']=type(error).__name__;state['error']=str(error);persist();raise
    finally:
        state['elapsed_seconds']=round(time.monotonic()-started,3);state['completed_utc']=datetime.now(timezone.utc).isoformat();persist()
    print('COMPLETE',str(output),state['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__': main()
