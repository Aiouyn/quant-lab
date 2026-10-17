"""Real full-day evidence for the upgrade; no provider calls or holdout access."""
import json
import os
from dataclasses import replace
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.data import Catalog
from quantdesk.strategy import validate_spec
from quantdesk.evaluation import Evaluator, daily_statistics


def main():
    settings=Settings(); store=Store(settings.home)
    catalog=Catalog(settings.data_dir,settings.home)
    root=settings.home/'verification'/'fitted-upgrade'
    root.mkdir(parents=True,exist_ok=True)
    state_path=root/'state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
    else:
        source=store.get_candidate('6bc072b37a0b9b55')
        plan=catalog.plan() # Complete days; future test dates stay sealed.
        cid=store.campaign(plan,250000)
        strategy=validate_spec(dict(source['spec'],
            title='VERIFICATION: fitted predictor using existing MiMo alpha',
            rationale='Infrastructure and economic benchmark of fitted lifecycle on full dates; not a new AI discovery',
            model={'family':'quadratic_ridge','train_days':12,'min_train_days':10,
                   'inner_days':3,'max_train_rows':30000,'stride_seconds':15},
            parent_ids=[source['id']]))
        ident,_=store.candidate(cid,strategy,source['code'])
        state={'campaign':cid,'candidate':ident,'source_candidate':source['id'],
               'status':'RUNNING','completed_outer_dates':[],
               'no_provider_calls':True,'holdout_accessed':False}
        state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    candidate=store.get_candidate(state['candidate'])
    state.update(pid=os.getpid(),status='RUNNING')
    (root/'runner.json').write_text(json.dumps({'pid':os.getpid(),'module':'scratch.verify_fitted_research'}),encoding='utf-8')
    state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    evaluator=Evaluator(store,catalog,settings)
    indices=[10,12,14,16,19]
    try:
        with store.lease('research',seconds=14400):
            for index in indices:
                output=root/f'outer-{index}.json'
                if output.exists():
                    continue
                print(f'Full-day outer development fold {index}: fit earlier dates, replay candidate, baseline and alpha ablation',flush=True)
                report=evaluator.walk_forward(state['campaign'],candidate,1,index,index)
                output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                state['completed_outer_dates']=sorted(set(state['completed_outer_dates']+report['outer_dates']))
                state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
                print(json.dumps({'date':report['outer_dates'],'net_pnl':report['pnl'],'trades':report['trades'],
                    'attribution':report['closed_trade_attribution']}),flush=True)
            reports=[json.loads((root/f'outer-{i}.json').read_text()) for i in indices]
            days=[d for r in reports for d in r['daily_results']]
            summary={'status':'COMPLETE','campaign':state['campaign'],'candidate':state['candidate'],
                'evidence_type':'Real full-day OKX data; existing MiMo code plus trusted fitted infrastructure',
                'outer_dates':[d['date'] for d in days], 'full_day_coverage_seconds':[d['duration_seconds'] for d in days],
                'statistics':daily_statistics([d['return_fraction'] for d in days],evaluator.trial_count('')),
                'pnl':sum(d['pnl'] for d in days),'trades':sum(d['trades'] for d in days),
                'baseline_pnl':sum(r['comparisons']['depth_flow_baseline']['pnl'] for r in reports),
                'without_custom_alpha_pnl':sum(r['comparisons']['without_custom_alpha']['pnl'] for r in reports),
                'training_cutoffs_verified':all(d['model']['max_label_end_ms']<d['model']['prediction_cutoff_ms']-300000 for d in days),
                'holdout_accessed':False,'provider_calls':0,
                'claims':{'new_ai_discovery':False,'promotion':False,'venue_transfer_proven':False}}
            (root/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
            state['status']='COMPLETE'
            state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
            print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        state['status']='FAILED'; state['error']=f'{type(error).__name__}: {error}'
        state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
