"""Existing public captures only; zero provider, holdout or exchange requests."""
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.agents import ResearchTools
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.research_map import research_map
from quantdesk.venue_study import inventory

CAMPAIGN='2b22519842f6'
CAPTURES=['hyperliquid-20260930-112558-56191c',
          'hyperliquid-20260930-112115-d26336',
          'hyperliquid-20260930-113005-436f57']


def snapshot(store):
    with store.connect() as db:
        usage=dict(db.execute('SELECT count(*) requests,sum(actual) known_actual_tokens,sum(COALESCE(actual,reserved)) charged_tokens FROM requests WHERE campaign=?',(CAMPAIGN,)).fetchone())
        return {'usage':usage,'campaign':dict(db.execute('SELECT id,status,token_limit FROM campaigns WHERE id=?',(CAMPAIGN,)).fetchone()),
                'cycles':[dict(r) for r in db.execute('SELECT id,status FROM cycles ORDER BY id')],
                'candidates':[dict(r) for r in db.execute('SELECT id,status FROM candidates ORDER BY id')],
                'holdouts':[dict(r) for r in db.execute('SELECT * FROM holdouts ORDER BY plan_hash')],
                'holdout_exposures':[dict(r) for r in db.execute('SELECT * FROM holdout_dates ORDER BY source_hash')],
                'forward_plans':[dict(r) for r in db.execute('SELECT id FROM forward_plans ORDER BY id')]}


def main():
    settings=Settings();store=Store(settings.home)
    home=settings.home/'verification'/'venue-study';home.mkdir(parents=True,exist_ok=True)
    output=home/'run.json'
    state={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
           'scope':'Three preselected existing public captures. Metadata, unique prints and hypothetical independent sweeps only; no own fills, strategy PnL or promotion.',
           'capture_selection':'Known capture identities spanning a short fast probe, long slow-depth capture and long fast capture. No outcome-based selection.',
           'capture_ids':CAPTURES,'notionals_usd':[250,1000,2500,10000],
           'before':snapshot(store),'inventory':inventory(store.home), 'studies':[],
           'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('quantdesk/venue_study.py'),Path('quantdesk/agents.py'),Path('quantdesk/research_map.py')]}}
    def persist():
        partial=output.with_suffix('.partial');partial.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8');partial.replace(output)
    started=time.monotonic();deadline=started+600
    tools=ResearchTools(CAMPAIGN,store,None,None,settings)
    try:
        with store.lease('research',seconds=900),store.research_deadline(CAMPAIGN,deadline):
            persist()
            for capture in CAPTURES:
                print('MEASURING',capture,flush=True)
                result=tools.call('falsifier','study_venue_capture',{'capture_id':capture,'notionals_usd':state['notionals_usd']})
                report=result['report']
                size=len(json.dumps(result).encode())
                state['studies'].append({'capture_id':capture,'study_id':result['study_id'],'already_measured':result['already_measured'],
                    'artifact':str(store.home/'venue_studies'/(result['study_id']+'.json')),'tool_response_bytes':size,
                    'summary':{k:report[k] for k in ['effective_source_state','quality_counts','depth_observations','book_window','segments','observations','trades','hypothetical_sweeps']}})
                persist();print('RECORDED',capture,result['study_id'],size,'bytes',flush=True)
            mapped=research_map(store,CAMPAIGN)['available_venue_studies']
            state['exact_studies_available_to_agents']=all(s['study_id'] in {r['study_id'] for r in mapped} for s in state['studies'])
            with store.connect() as db:
                state['durable_study_memories']=db.execute("SELECT count(*) FROM memories WHERE campaign=? AND kind='VENUE_EXECUTION_STUDY'",(CAMPAIGN,)).fetchone()[0]
            state['after']=snapshot(store)
            state['research_ledger_invariants_unchanged']=state['before']==state['after']
            if not state['research_ledger_invariants_unchanged'] or not state['exact_studies_available_to_agents']:
                raise RuntimeError('Verification invariant failed')
            state['new_provider_requests']=0;state['new_holdout_exposures']=0;state['orders_submitted_to_exchange']=0
            state['status']='COMPLETE'
    except Exception as error:
        state['status']='FAILED';state['error_type']=type(error).__name__;state['error']=str(error);persist();raise
    finally:
        state['elapsed_seconds']=round(time.monotonic()-started,3);state['completed_utc']=datetime.now(timezone.utc).isoformat();persist()
    print('COMPLETE',str(output),state['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__': main()
