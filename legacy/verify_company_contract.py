"""Real prepared-data planning and ID replay; no paid model invocation or holdout."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.agents import ResearchTools
from quantdesk.planning import planning_contract


def main():
    settings=Settings(); store=Store(settings.home); catalog=Catalog(settings.data_dir,settings.home)
    candidate=store.get_candidate('37e7854f12d976e8'); cid=candidate['campaign']; plan=store.get_campaign(cid)['plan']
    date=next(i for i,r in enumerate(plan['development']) if r['date']=='2025-11-23')
    earlier=plan['development'][max(0,date-candidate['spec']['model']['train_days']):date]
    assert all(catalog.cached(r,None) for r in earlier+[plan['development'][date]])
    with store.connect() as db:
        before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
        seals=db.execute('SELECT COUNT(*) FROM holdout_dates').fetchone()[0]
    tools=ResearchTools(cid,store,catalog,Evaluator(store,catalog,settings),settings,'verification-only-not-company-cycle')
    contract=planning_contract(store,catalog,cid,settings)
    built=tools.call('engineer','build_candidate',{'spec':candidate['spec'],'code':candidate['code']})
    assert built['candidate_id']==candidate['id'] and not built['submitted']
    check=tools.call('engineer','check_candidate',{'candidate_id':candidate['id']})
    compatibility=tools.call('engineer','candidate_feasibility',{'candidate_id':candidate['id'],'day_index':date})
    replay=tools.call('engineer','walk_forward_candidate',{'candidate_id':candidate['id'],'folds':1,'start_index':date,'end_index':date})
    assert replay['outer_dates']==['2025-11-23'] and replay['trades']==0
    assert store.get_candidate(candidate['id'])==candidate
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before
        assert db.execute('SELECT COUNT(*) FROM holdout_dates').fetchone()[0]==seals
    legacy_args={'spec':candidate['spec'],'code':candidate['code'],'folds':1,'start_index':date,'end_index':date}
    id_args={'candidate_id':candidate['id'],'folds':1,'start_index':date,'end_index':date}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'candidate_id':candidate['id'],
        'planning_contract':contract,'build':built,'numerical_check':check,'compatibility':compatibility,
        'replay':replay,'original_candidate_and_report_preserved':True,
        'argument_bytes':{'legacy':len(json.dumps(legacy_args).encode()),'id_workflow':len(json.dumps(id_args).encode()),
            'interpretation':'Serialized call arguments only; not a measured provider-token or research-efficiency improvement'},
        'new_provider_requests':0,'new_holdout_exposures':0,'company_cycles_completed_by_this_probe':0,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')},
        'interpretation':'Actual cached development data and actual earlier agent program/model. Infrastructure replay and history constraints; zero trades, no new alpha, no company completion or controlled learning claim.'}
    directory=settings.home/'verification'/'company-contract'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'actual-replay.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'receipt':str(path),'outer_dates':replay['outer_dates'],'trades':replay['trades'],
        'argument_bytes':result['argument_bytes'],'new_provider_requests':0,'candidate_preserved':True},indent=2))


if __name__=='__main__':main()
