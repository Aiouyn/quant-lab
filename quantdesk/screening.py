"""Precommitted development triage; never validation, holdout or promotion."""
import json
import numpy as np
from .store import digest
from .strategy import VALIDATOR_VERSION
from .data import FEATURE_VERSION
from .models import MODEL_VERSION,validate_recipe
from .control import ResearchStopped

SCREEN_VERSION='1.0.0'


def available_indices(evaluator,cid,candidate=None):
    plan=evaluator.store.get_campaign(cid)['plan']
    records=plan['development']; recipe=validate_recipe(candidate['spec']['model']) if candidate and 'model' in candidate['spec'] else None
    minimum=recipe['min_train_days'] if recipe else 0
    cached=getattr(evaluator.catalog,'cached',None)
    prepared={i for i,r in enumerate(records) if cached and cached(r,plan.get('max_rows_per_day'))}
    eligible=[i for i in range(minimum,len(records)) if i in prepared and
        (not recipe or all(j in prepared for j in range(max(0,i-recipe['train_days']),i)))]
    # Without a prepared window, use the earliest bounded window rather than
    # hydrating a year's history. Selection never uses returns or predictions.
    pool=eligible or list(range(minimum,min(len(records),minimum+3)))
    positions=np.unique(np.linspace(0,len(pool)-1,min(3,len(pool)),dtype=int)) if pool else []
    return {'indices':[pool[int(i)] for i in positions],'prepared_eligible_indices':eligible,
        'selection':'First, middle and last eligible cached development date; otherwise earliest bounded window',
        'uses_outcomes_for_selection':False,'freshness_claim':False}


def classify(evidence,mode):
    days=evidence['daily_results']; reasons=[]
    if len(days)<3 or evidence.get('sampled') or any(d['duration_seconds']<18*3600 for d in days):
        reasons.append('Need three complete development dates for automatic confirmation triage; partial results remain descriptive')
    if any(any(d.get('quality',{}).get(k,0) for k in ('sequence_gaps','timestamp_gaps','crossed_books')) for d in days):
        reasons.append('Data quality faults require repair or independent review')
    if any(d.get('halted') or d.get('funding_boundary_crossings',0) for d in days):
        reasons.append('Risk halt or unmodeled funding exposure requires review')
    if reasons:
        return 'DEVELOPMENT_INCONCLUSIVE',reasons
    if mode=='maker':
        return 'DEVELOPMENT_RESEARCH_ONLY',['Maker proxy cannot authorize confirmation for promotion without trade and queue evidence']
    if evidence['trades']==0:
        return 'DEVELOPMENT_INACTIVE',['No executed trades on the selected development dates; this configuration provides no economic activity evidence']
    if evidence['pnl']<=0:
        return 'DEVELOPMENT_NEGATIVE',['Aggregate development net PnL does not exceed cash after modeled costs']
    if evidence['trades']<5 or sum(d['pnl']>0 for d in days)<2:
        return 'DEVELOPMENT_INCONCLUSIVE',['Positive aggregate PnL has insufficient activity or date breadth for automatic confirmation triage']
    ablation=evidence.get('comparisons',{}).get('without_custom_alpha')
    if ablation and evidence['pnl']<=ablation['pnl']:
        return 'DEVELOPMENT_NO_INCREMENT',['Agent-authored features did not improve matched net PnL over their removal on these development dates']
    return 'DEVELOPMENT_SCREEN_PASSED',[]


def screen(evaluator,cid,candidate,development_indices=None):
    store=evaluator.store; campaign=store.get_campaign(cid); plan=campaign['plan']
    if candidate['campaign']!=cid or campaign['status']=='SEALED':
        raise ValueError('Screen requires this candidate and an unsealed campaign')
    if candidate['status'] not in {'PROPOSED','EVALUATION_ERROR'}:
        raise ValueError('Screen cannot overwrite a completed confirmation decision')
    minimum=validate_recipe(candidate['spec']['model'])['min_train_days'] if 'model' in candidate['spec'] else 0
    if development_indices is not None and (not 1<=len(development_indices)<=3 or len(set(development_indices))!=len(development_indices) or any(type(i) is not int or not minimum<=i<len(plan['development']) for i in development_indices)):
        raise ValueError('Screen needs 1..3 development indices with sufficient earlier training')
    contract={'version':SCREEN_VERSION,'features':FEATURE_VERSION,'model':MODEL_VERSION,'validator':VALIDATOR_VERSION,
        'candidate':candidate['id'],'plan_hash':plan['hash'],'execution':evaluator.settings.execution_dict(),
        'requested_indices':development_indices}
    key=digest(contract)
    cached=store.cached_tool(key)
    if cached is not None:
        store.event(cid,'validator','DEVELOPMENT_SCREEN_REUSED',{'candidate':candidate['id'],'key':key})
        store.record_report(candidate['id'],'PROPOSED',cached)
        return dict(cached,already_measured=True)
    with store.connect() as db:
        rows=db.execute("SELECT payload FROM events WHERE campaign=? AND kind='DEVELOPMENT_SCREEN_PLANNED' ORDER BY id",(cid,)).fetchall()
    previous=next((json.loads(r[0]) for r in rows if json.loads(r[0]).get('key')==key),None)
    selection=previous or dict(available_indices(evaluator,cid,candidate),key=key,contract=contract)
    if previous is None:
        if development_indices is not None:
            selection.update(indices=list(development_indices),selection='Explicit development research dates, precommitted before replay')
        selection['dates']=[plan['development'][i]['date'] for i in selection['indices']]
        store.event(cid,'validator','DEVELOPMENT_SCREEN_PLANNED',selection)
    indices=selection['indices']
    evidence={}; diagnostic=None
    try:
        if not indices:
            status,reasons='DEVELOPMENT_INCONCLUSIVE',['Insufficient earlier development dates for this fitted recipe']
        else:
            if 'model' in candidate['spec']:
                evidence=evaluator.walk_forward(cid,candidate,development_indices=indices,checkpoint_key=key)
            else:
                def measured(member):
                    replay_key=digest({'screen':key,'candidate':member['id']})
                    result=store.cached_tool(replay_key)
                    if result is None:
                        result=evaluator.evaluate(cid,member,'development',development_indices=indices)
                        store.cache_tool(replay_key,result)
                    return result
                result=measured(candidate)
                from .evaluation import REFERENCE_CODE
                baseline_spec=dict({k:v for k,v in candidate['spec'].items() if k!='input_features'},title='REFERENCE: causal depth-flow screen baseline',
                    rationale='Fixed matched research comparator; not an agent discovery',falsification='Measure net value on the same development dates')
                baseline_id,_=store.candidate(cid,baseline_spec,REFERENCE_CODE)
                baseline=measured(store.get_candidate(baseline_id))
                evidence={'daily_results':result['days'],'statistics':result['statistics'],'pnl':result['pnl'],
                    'trades':result['trades'],'sampled':result['sampled'],'execution':result['execution'],
                    'comparisons':{'cash':{'pnl':0.0,'trades':0},'depth_flow_baseline':{
                        'candidate_id':baseline_id,'pnl':baseline['pnl'],'trades':baseline['trades']}}}
            status,reasons=classify(evidence,candidate['spec']['mode'])
            if evidence['trades']==0 and 'model' in candidate['spec']:
                diagnostic_key=digest({'screen':key,'prediction_diagnostic':indices[-1]})
                diagnostic=store.cached_tool(diagnostic_key)
                if diagnostic is None:
                    diagnostic=evaluator.diagnose_prediction(cid,candidate,indices[-1])
                    store.cache_tool(diagnostic_key,diagnostic)
    except ResearchStopped:
        raise
    except Exception as error:
        # Retain partial economic evidence, but never call a resource/worker
        # failure proof that the hypothesis lost money.
        status,reasons='EVALUATION_ERROR',[f'{type(error).__name__}: {error}']
    confirmation_reasons=[]
    if len(plan.get('validation',[]))<evaluator.settings.min_validation_days or len(plan.get('holdout',[]))<evaluator.settings.min_holdout_days:
        confirmation_reasons.append('Frozen plan lacks the minimum distinct validation or holdout dates')
    report={'status':status,'reasons':reasons,'phase':'development','candidate_id':candidate['id'],
        'screen_key':key,'screen_version':SCREEN_VERSION,'selection':selection,'development':evidence,
        'prediction_diagnostic':diagnostic,'confirmation_eligible':status=='DEVELOPMENT_SCREEN_PASSED' and not confirmation_reasons,
        'confirmation_reasons':confirmation_reasons,'promotion':False,'holdout_accessed':False,
        'interpretation':'Bounded development allocation decision only. Negative findings concern this configuration and these dates. No significance, independent confirmation, permanent class rejection or promotion is inferred.'}
    if status=='EVALUATION_ERROR':
        report['failure_kind']='operational; hypothesis not falsified'
    # A screened candidate remains a research proposal. Manual confirmation
    # stays available; screening never sets VALIDATED or PAPER_READY.
    store.record_report(candidate['id'],'PROPOSED',report)
    store.memory(cid,candidate['id'],'DEVELOPMENT_SCREEN',report)
    if status!='EVALUATION_ERROR':
        store.cache_tool(key,report)
    return report
