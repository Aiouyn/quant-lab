"""Independent full-day replay of the actual MiMo feature submission.

Development evidence and an exploratory frozen bundle only. Never consumes
confirmation dates or fabricates success after an incomplete agent task.
"""
import json
import os
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator,daily_statistics
from quantdesk.input_contract import program_input
from quantdesk.strategy import run_feature_program


def save(path,value):
    partial=path.with_suffix('.partial')
    partial.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    partial.replace(path)


def main():
    settings=Settings(); store=Store(settings.home)
    source=json.loads((settings.home/'verification'/'feature-agent'/'state.json').read_text())
    if source['status']!='COMPLETE' or not source.get('candidate_ids'):
        raise RuntimeError('A completed real agent submission is required before replay')
    candidate=store.get_candidate(source['candidate_ids'][-1]); cid=candidate['campaign']
    plan=store.get_campaign(cid)['plan']
    if plan.get('sampled') or plan.get('max_rows_per_day'):
        raise RuntimeError('Verification requires full daily archives')
    if not candidate['spec'].get('model',{}).get('custom_features'):
        raise RuntimeError('Expected the actual agent-authored custom-feature candidate')
    catalog=Catalog(settings.data_dir,settings.home); evaluator=Evaluator(store,catalog,settings)
    directory=settings.home/'verification'/'feature-replay'; directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'state.json'
    state={'status':'RUNNING','pid':os.getpid(),'campaign':cid,'candidate':candidate['id'],
        'code_hash':digest(candidate['code']),'spec_hash':digest(candidate['spec']),
        'completed_outer_dates':[],'holdout_accessed':False,'promotion':False}
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous['candidate']!=candidate['id']:
            raise RuntimeError('Existing verification belongs to a different candidate')
        if previous['status']=='COMPLETE':
            print(json.dumps(previous,indent=2)); return
    save(state_path,state)
    try:
        with store.lease('research',seconds=7200):
            frame,quality=catalog.load(plan['development'][0])
            if len(frame)<80000 or any(quality.get(k,0) for k in ('crossed_books','timestamp_gaps','sequence_gaps')):
                raise RuntimeError('Preflight full-day coverage/quality failed')
            features=run_feature_program(candidate['code'],program_input(frame,candidate['spec']),settings.worker_timeout)
            state['preflight']={'date':plan['development'][0]['date'],'rows':len(frame),
                'feature_columns':features.columns,'causal_prefix_checks':True}
            save(state_path,state)
            reports=[]
            for index in (10,14,19):
                path=directory/f'outer-{index}.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if report['candidate_id']!=candidate['id']:
                        raise RuntimeError('Replay checkpoint candidate mismatch')
                else:
                    print(f'Full-day causal feature/model replay at development index {index}',flush=True)
                    report=evaluator.walk_forward(cid,candidate,1,index,index)
                    save(path,report)
                reports.append(report)
                state['completed_outer_dates']=sorted({d for r in reports for d in r['outer_dates']})
                save(state_path,state)
                print(json.dumps({'dates':report['outer_dates'],'pnl':report['pnl'],'trades':report['trades']}),flush=True)
            bundle=evaluator.export(candidate,{},True)
            days=[d for r in reports for d in r['daily_results']]
            summary={'status':'COMPLETE','campaign':cid,'candidate':candidate['id'],
                'source':'Actual MiMo-authored custom features, independently fit and replayed on real full-day OKX data',
                'outer_dates':[d['date'] for d in days],'rows':[d['rows'] for d in days],
                'statistics':daily_statistics([d['return_fraction'] for d in days],evaluator.trial_count(plan['hash'])),
                'pnl':sum(d['pnl'] for d in days),'trades':sum(d['trades'] for d in days),
                'baseline_pnl':sum(r['comparisons']['depth_flow_baseline']['pnl'] for r in reports),
                'without_custom_features_pnl':sum(r['comparisons']['without_custom_alpha']['pnl'] for r in reports),
                'training_cutoffs_verified':all(d['model']['max_label_end_ms']<d['model']['prediction_cutoff_ms']-
                    candidate['spec']['model']['embargo_seconds']*1000 for d in days),
                'preflight':state['preflight'],'bundle':str(bundle),
                'holdout_accessed':False,'profitable_strategy_claim':False,'promotion':False,
                'limitation':'Three outer development dates do not establish alpha, autonomous-company learning or venue transfer'}
            save(directory/'report.json',summary)
            state.update(status='COMPLETE',bundle=str(bundle)); save(state_path,state)
            print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        state.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        save(state_path,state)
        raise


if __name__=='__main__':
    main()
