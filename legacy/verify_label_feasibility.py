"""Real development compatibility and exact refit parity, without provider calls."""
import json
import hashlib
import uuid
from dataclasses import replace
from pathlib import Path
from datetime import datetime,timezone
from quantdesk.data import Catalog
from quantdesk.models import ModelService,label_feasibility
from quantdesk.settings import Settings
from quantdesk.store import Store,digest


def main():
    settings=Settings(); source=Store(settings.home)
    target=settings.home/'verification'/'label-feasibility'
    target.mkdir(parents=True,exist_ok=True)
    isolated=Store(target/'refits'/uuid.uuid4().hex[:12])
    with source.connect() as src,isolated.connect() as dst:
        src.backup(dst)
    catalog=Catalog(settings.data_dir,settings.home)
    cid='2b22519842f6'; plan=source.get_campaign(cid)['plan']
    draft=source.get_candidate('fef90e1b12a2de46')
    record=next(r for r in plan['development'] if r['date'].endswith('11-25'))
    assert catalog.cached(record,None),'Verification must reuse a prepared full development date'
    frame,quality=catalog.load(record,None)
    incompatible=label_feasibility(frame,draft['spec'],settings.execution_dict(),draft['spec']['model'])
    candidate=source.get_candidate('37e7854f12d976e8')
    old_path=settings.home/'models'/'eb2ce19499dbc1779143e5b021632448d7f7eb94045e814692c010420b61577c.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    candidate_plan=source.get_campaign(candidate['campaign'])['plan']
    records=[r for r in candidate_plan['development'] if r['date'] in old['provenance']['training_dates']]
    assert all(catalog.cached(r,candidate_plan.get('max_rows_per_day')) for r in records)
    service=ModelService(isolated,catalog,replace(settings,home=isolated.home))
    # Preserve the exact earlier measured features; this check isolates refitting.
    original=ModelService(source,catalog,settings)
    def measured_alpha(candidate,record,frame,rows):
        custom=candidate['spec']['model'].get('custom_features',False)
        path=source.home/'signals'/(digest({'model_alpha':'1.1.0','record':record,'rows':rows,
            'code':candidate['code'],'custom_features':custom,'input_features':candidate['spec'].get('input_features')})+('.parquet' if custom else '.npy'))
        assert path.exists(),'Refit verification must reuse an actual existing feature artifact'
        return original.alpha(candidate,record,frame,rows)
    service.alpha=measured_alpha
    fitted,path=service.fit(candidate['campaign'],candidate,records,old['provenance']['prediction_cutoff_ms'])
    assert fitted==old,'Refactor changed a previously frozen actual fit'
    with source.connect() as db:
        usage=dict(db.execute('SELECT SUM(COALESCE(actual,reserved)) charged,SUM(actual) actual,COUNT(*) requests FROM requests WHERE campaign=?',(cid,)).fetchone())
        states=[dict(r) for r in db.execute('SELECT agent,status FROM tasks WHERE campaign=?',(cid,))]
        cycles=dict(db.execute("SELECT COUNT(*) total,SUM(status='COMPLETE') completed FROM cycles WHERE campaign=?",(cid,)).fetchone())
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'pilot':{'campaign':cid,'usage':usage,'tasks':states,'cycles':cycles},
        'actual_development':{'date':record['date'],'rows':len(frame),'quality':quality,'recipe_compatibility':incompatible},
        'numeric_parity':{'candidate_id':candidate['id'],'exact_artifact_equal':True,'hash':fitted['artifact_hash'],'original':str(old_path),'refit':str(path)},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')},
        'interpretation':'Compatibility and numerical-preservation evidence only. No new MiMo request, economic backtest, holdout exposure, promotion or exchange order.'}
    receipt=target/'verification.json'
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
