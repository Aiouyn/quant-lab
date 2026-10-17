"""Synthetic adversarial chronology, fitted replay/export and paper parity."""
import json
from datetime import datetime, timezone, timedelta
from dataclasses import replace
import numpy as np
import polars as pl
import pytest
from quantdesk.data import Book, enrich
from quantdesk.models import (validate_recipe, labelled_rows, fit_ridge, predict,
                              fitted_signal, ModelService, verify_artifact)
from quantdesk.store import Store, digest
from quantdesk.strategy import validate_spec, evaluate_in_process
from quantdesk.evaluation import Evaluator
from quantdesk.settings import Settings
from quantdesk.paper import load_bundle, PaperSession

CODE = "import polars as pl\ndef compute_signal(df):\n    return df['qi_5'].clip(-1.0,1.0)\n"


def spec():
    return validate_spec({'title':'Synthetic fitted fixture', 'rationale':'Known artificial relation',
        'falsification':'No research profitability claim', 'mode':'taker', 'holding_seconds':10,
        'threshold':0.2, 'warmup_rows':3, 'model':{'features':['qi_5','micro_dev_bps'],
        'train_days':8, 'min_train_days':6, 'inner_days':2, 'stride_seconds':5,
        'max_train_rows':4000, 'embargo_seconds':60}})


def frame(date, count=1200):
    start=int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()*1000)
    b=Book(); rows=[]; mid=100
    for i in range(count):
        pressure=0.8*np.sin(i/45)
        mid*=np.exp(pressure*0.0002)
        rows.append(b.apply({'action':'snapshot','_continuous_snapshot':True,'ts':start+i*1000,
            'bids':[[mid*0.99995-level*0.001,20*(1+pressure),1] for level in range(5)],
            'asks':[[mid*1.00005+level*0.001,20*(1-pressure),1] for level in range(5)]}))
    return enrich(pl.DataFrame(rows))


def public_fixture_book(mid,pressure,ts):
    return {'coin':'BTC','time':ts,'levels':[
        [{'px':str(mid*0.99995-level*0.001),'sz':str(20*(1+pressure)),'n':1} for level in range(5)],
        [{'px':str(mid*1.00005+level*0.001),'sz':str(20*(1-pressure)),'n':1} for level in range(5)]]}


class FixtureCatalog:
    def __init__(self, records):
        self.frames={r['date']:frame(r['date']) for r in records}
        self.reads=[]

    def load(self, record, max_rows=None):
        self.reads.append(record['date'])
        result=self.frames[record['date']]
        return result.head(max_rows) if max_rows else result, {'sequence_gaps':0,'timestamp_gaps':0,'crossed_books':0}


def desk(tmp_path, monkeypatch):
    import quantdesk.models as models
    import quantdesk.evaluation as evaluation
    # Existing worker regression tests cover process execution; here isolate finance boundaries.
    monkeypatch.setattr(models,'run_signal',lambda code,data,*a: evaluate_in_process(code,data))
    monkeypatch.setattr(evaluation,'run_signal',lambda code,data,*a: evaluate_in_process(code,data))
    records=[{'date':(datetime(2026,1,1)+timedelta(days=i)).date().isoformat()} for i in range(15)]
    plan={'development':records[:12], 'validation':records[12:14], 'holdout':records[14:],
          'sampled':False,'max_rows_per_day':None,'version':'fixture','hash':'frozen-fixture',
          'limitations':['Artificial data; no alpha claim']}
    store=Store(tmp_path); cid=store.campaign(plan,1000)
    candidate_id,_=store.candidate(cid,spec(),CODE)
    catalog=FixtureCatalog(records)
    settings=replace(Settings(),home=tmp_path)
    return store,cid,store.get_candidate(candidate_id),catalog,Evaluator(store,catalog,settings),settings


def test_train_only_scaling_and_nonlinear_fit():
    rng=np.random.default_rng(412); x=rng.normal(size=(800,2)); y=2*x[:,0]*x[:,1]+0.4*x[:,0]
    fitted=fit_ridge(x[:600],y[:600],0.0001,'quadratic_ridge')
    assert np.mean((predict(x[600:],fitted)-y[600:])**2)<0.01
    assert np.allclose(fitted['mean'],x[:600].mean(axis=0))
    # Extreme future inputs cannot change fitted parameters or earlier predictions.
    prefix=predict(x[:10],fitted)
    assert np.allclose(predict(np.vstack([x[:10],np.full((5,2),1e9)]),fitted)[:10],prefix)


def test_label_windows_reject_segments_gaps_and_funding():
    data=frame('2026-01-01',100)
    strategy=spec(); recipe=strategy['model']; execution=Settings().execution_dict()
    indices,target,ends=labelled_rows(data,strategy,execution,recipe)
    ts=data['ts_ms'].to_numpy()
    mid=data['mid_price'].to_numpy()
    assert np.allclose(target,(mid[indices+12]/mid[indices+1]-1)*10000)
    assert np.all(ends==ts[indices+12])
    broken=data.with_columns(pl.Series('segment',np.r_[np.zeros(50,dtype=int),np.ones(50,dtype=int)]))
    indices,_,_=labelled_rows(broken,strategy,execution,recipe)
    assert not any(38<=i<50 for i in indices)
    shifted=data.with_columns(pl.Series('ts_ms',ts-ts[0]+3590000))
    indices,_,_=labelled_rows(shifted,strategy,execution,recipe)
    assert not any(i<10 for i in indices)
    gapped=data.with_columns(pl.Series('ts_ms',ts+np.r_[np.zeros(50,dtype=int),np.full(50,10000)]))
    indices,_,_=labelled_rows(gapped,strategy,execution,recipe)
    assert not any(38<=i<50 for i in indices)


def test_embargo_purges_unresolved_labels():
    data=frame('2026-01-01',200); strategy=spec()
    cutoff=int(data['ts_ms'][180])
    _,_,ends=labelled_rows(data,strategy,Settings().execution_dict(),strategy['model'],cutoff)
    assert len(ends)>0
    assert ends.max()<cutoff-60000


def test_fit_refuses_validation_and_future_records(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    plan=store.get_campaign(cid)['plan']; service=ModelService(store,catalog,settings)
    cutoff=int(catalog.frames[plan['validation'][0]['date']]['ts_ms'][0])
    with pytest.raises(ValueError,match='development dates only'):
        service.fit(cid,candidate,plan['development'][:6]+plan['validation'],cutoff)
    with pytest.raises(ValueError,match='precede'):
        service.fit(cid,candidate,plan['development'][:6],int(catalog.frames[plan['development'][5]['date']]['ts_ms'][0]))


def test_outer_date_changes_cannot_change_fit_or_inner_selection(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    plan=store.get_campaign(cid)['plan']; record=plan['development'][8]
    service=ModelService(store,catalog,settings)
    cutoff=int(catalog.frames[record['date']]['ts_ms'][0])
    first,path=service.fit(cid,candidate,plan['development'][:8],cutoff)
    p=first['provenance']
    assert p['max_label_end_ms']<cutoff-60000
    assert p['inner_max_label_end_ms']<p['inner_prediction_cutoff_ms']-60000
    assert not set(p['inner_training_dates']) & set(p['inner_selection_dates'])
    assert record['date'] not in catalog.reads
    catalog.frames[record['date']]=catalog.frames[record['date']].with_columns(pl.lit(1e12).alias('mid_price'))
    second,_=service.fit(cid,candidate,plan['development'][:8],cutoff)
    assert first==second
    changed=json.loads(json.dumps(first)); changed['fitted']['intercept']+=1
    with pytest.raises(ValueError,match='integrity'):
        verify_artifact(changed)


def test_fitted_walk_forward_frozen_validation_and_export(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    result=evaluator.walk_forward(cid,candidate,2,8)
    assert result['folds']==2
    assert not set(result['outer_dates']) & {'2026-01-13','2026-01-14','2026-01-15'}
    for day in result['daily_results']:
        assert all(t<day['date'] for t in day['model']['training_dates'])
        assert day['model']['max_label_end_ms']<day['model']['prediction_cutoff_ms']-60000
    report=evaluator.validate(cid,candidate['id'])
    assert report['status']=='REJECTED' # 20 minutes of artificial data cannot qualify.
    base=report['base']; stress=report['stress']
    assert base['model_artifact']['hash']==stress['model_artifact']['hash']
    assert len({d['model']['hash'] for d in base['days']})==1
    candidate=store.get_candidate(candidate['id'])
    holdout=evaluator.evaluate(cid,candidate,'holdout')
    assert holdout['model_artifact']['hash']==base['model_artifact']['hash']
    directory=evaluator.export(candidate,report,True)
    manifest,code=load_bundle(directory,True)
    artifact=manifest['_model_artifact']
    data=catalog.frames['2026-01-15']
    alpha=evaluate_in_process(code,data)
    full=fitted_signal(data,alpha,artifact)
    prefix=fitted_signal(data.head(100),alpha[:100],artifact)
    assert np.allclose(full[:100],prefix)
    # Paper calls the same fitted inference and records the same last signal.
    import quantdesk.paper as paper
    monkeypatch.setattr(paper,'run_signal',lambda code,data,*a: evaluate_in_process(code,data,False))
    session=PaperSession(directory,tmp_path,settings,True)
    observed=[]
    original=session.members[0]['broker'].step
    def step(book,signal,terminal=False,**kwargs):
        observed.append(signal)
        return original(book,signal,terminal,**kwargs)
    session.members[0]['broker'].step=step
    b=Book(); rows=[]
    for i in range(15):
        ts=int(data['ts_ms'][i]); mid=float(data['mid_price'][i]); pressure=float(data['qi_5'][i])
        book=public_fixture_book(mid,pressure,ts)
        session.consume(book,ts)
        rows.append(b.apply(paper.normalize_book(book)))
    actual=enrich(pl.DataFrame(rows))
    expected=fitted_signal(actual,evaluate_in_process(code,actual),artifact)[-1]
    assert observed[-1]==pytest.approx(expected)
    path=directory/'model.json'; altered=json.loads(path.read_text()); altered['fitted']['intercept']+=1
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='integrity'):
        load_bundle(directory,True)


def test_operational_errors_are_retryable_not_economic_rejections(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    original=catalog.load
    catalog.load=lambda *args: (_ for _ in ()).throw(OSError('fixture unavailable'))
    report=evaluator.validate(cid,candidate['id'])
    assert report['status']=='EVALUATION_ERROR'
    catalog.load=original
    report=evaluator.validate(cid,candidate['id'])
    assert report['status']=='REJECTED' and 'base' in report


@pytest.mark.parametrize('recipe',[{'features':['ts_ms']},{'alphas':[float('nan')]},
    {'train_days':4},{'min_train_days':6,'inner_days':6},{'embargo_seconds':-1},
    {'family':'arbitrary_python'},{'max_train_rows':float('inf')}])
def test_invalid_fitting_contracts_fail(recipe):
    with pytest.raises(ValueError):
        validate_recipe(recipe)


FEATURE_CODE='''import polars as pl
import numpy as np
def compute_signal(df):
    pressure=df['qi_5']
    state=df['momentum_10s_bps'].clip(-30.0,30.0)/30.0
    return pl.DataFrame({'pressure':pressure,'state':state,'interaction':pressure*state})
'''


def test_causal_feature_program_runs_in_the_real_worker():
    from quantdesk.strategy import run_feature_program
    data=frame('2026-01-01',100)
    result=run_feature_program(FEATURE_CODE,data,15)
    assert result.columns==['pressure','state','interaction']
    expected=evaluate_in_process(FEATURE_CODE,data,True,True)
    assert np.allclose(result.to_numpy(),expected.to_numpy())
    assert np.allclose(result['pressure'],data['qi_5'])


@pytest.mark.parametrize('body',[
    "pl.DataFrame({'future':(df['qi_5']-df['qi_5'].mean()).clip(-1.0,1.0)})",
    "pl.DataFrame({('late' if len(df)>70 else 'early'):df['qi_5']})",
    "pl.DataFrame({'huge':df['mid_price']})",
    "pl.DataFrame({'text':pl.Series(['a'])})"
])
def test_feature_contract_rejects_future_schema_and_unbounded_values(body):
    code='import polars as pl\ndef compute_signal(df):\n    return '+body+'\n'
    with pytest.raises(ValueError):
        evaluate_in_process(code,frame('2026-01-01',100),True,True)


def test_frozen_feature_schema_rejects_column_reordering(tmp_path,monkeypatch):
    import quantdesk.models as models
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    monkeypatch.setattr(models,'run_feature_program',lambda code,data,*a:evaluate_in_process(code,data,True,True))
    strategy=validate_spec(dict(candidate['spec'],model=dict(candidate['spec']['model'],custom_features=True)))
    ident,_=store.candidate(cid,strategy,FEATURE_CODE)
    candidate=store.get_candidate(ident)
    records=store.get_campaign(cid)['plan']['development'][:8]
    data=catalog.frames['2026-01-09']
    artifact,_=ModelService(store,catalog,settings).fit(cid,candidate,records,int(data['ts_ms'][0]))
    custom=evaluate_in_process(FEATURE_CODE,data,True,True)
    fitted_signal(data,custom,artifact)
    with pytest.raises(ValueError,match='names/order'):
        fitted_signal(data,custom.select(list(reversed(custom.columns))),artifact)


def test_tree_learns_threshold_interactions_and_survives_native_json():
    from quantdesk.learners import fit_tree,predict_tree,tree_options
    rng=np.random.default_rng(935)
    x=rng.uniform(-1,1,(4000,3))
    y=np.where((x[:,0]>0.3)&(x[:,1]<-0.2),20.0,-5.0)+rng.normal(0,0.1,len(x))
    params=tree_options([{'max_depth':3,'min_child_weight':5,'eta':0.1,'rounds':100}])[0]
    tree=fit_tree(x[:3000],y[:3000],params)
    estimates=predict_tree(x[3000:],tree)
    linear=fit_ridge(x[:3000],y[:3000],0.1,'ridge')
    assert np.mean((estimates-y[3000:])**2)<np.mean((predict(x[3000:],linear)-y[3000:])**2)*0.25
    restored=json.loads(json.dumps(tree))
    assert np.allclose(predict_tree(x[3000:],restored),estimates)
    assert restored['library']=='xgboost' and 'model_json' in restored


def test_custom_tree_replay_freeze_export_and_paper_parity(tmp_path,monkeypatch):
    import quantdesk.models as models
    import quantdesk.evaluation as evaluation
    import quantdesk.paper as paper
    from quantdesk.strategy import run_feature_program
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    strategy=validate_spec(dict(candidate['spec'],model=dict(candidate['spec']['model'],
        family='xgboost',custom_features=True,tree_search=[{'rounds':30,'max_depth':2,'min_child_weight':10}])))
    ident,_=store.candidate(cid,strategy,FEATURE_CODE)
    candidate=store.get_candidate(ident)
    # Fitting passes through the real credential-free feature worker.
    monkeypatch.setattr(models,'run_feature_program',run_feature_program)
    monkeypatch.setattr(evaluation,'run_feature_program',run_feature_program)
    report=evaluator.validate(cid,ident)
    assert report['status']=='REJECTED' and 'base' in report
    assert report['base']['model_artifact']['hash']==report['stress']['model_artifact']['hash']
    assert all(d['model']['family']=='xgboost' for d in report['base']['days'])
    candidate=store.get_candidate(ident)
    holdout=evaluator.evaluate(cid,candidate,'holdout')
    assert holdout['model_artifact']['hash']==report['base']['model_artifact']['hash']
    directory=evaluator.export(candidate,report,True)
    manifest,code=load_bundle(directory,True)
    artifact=manifest['_model_artifact']
    assert artifact['fitted']['feature_names'][-3:]==['custom:pressure','custom:state','custom:interaction']
    assert artifact['provenance']['selected_inner_diagnostics']['observations']>=50
    # Paper builds the same named feature matrix and calls frozen native inference.
    monkeypatch.setattr(paper,'run_feature_program',lambda code,data,*a:evaluate_in_process(code,data,False,True))
    session=PaperSession(directory,tmp_path,settings,True)
    observed=[]; original=session.members[0]['broker'].step
    def step(book,signal,terminal=False,**kwargs):
        observed.append(signal); return original(book,signal,terminal,**kwargs)
    session.members[0]['broker'].step=step
    data=catalog.frames['2026-01-15']
    b=Book(); rows=[]
    for i in range(15):
        ts=int(data['ts_ms'][i]); mid=float(data['mid_price'][i]); pressure=float(data['qi_5'][i])
        book=public_fixture_book(mid,pressure,ts)
        session.consume(book,ts); rows.append(b.apply(paper.normalize_book(book)))
    actual=enrich(pl.DataFrame(rows))
    expected=fitted_signal(actual,evaluate_in_process(code,actual,False,True),artifact)[-1]
    assert observed[-1]==pytest.approx(expected)


@pytest.mark.parametrize('recipe',[
    {'family':'xgboost','tree_search':[{'max_depth':20}]},
    {'family':'xgboost','tree_search':[{'objective':'custom'}]},
    {'family':'xgboost','tree_search':[{},{}]},
    {'family':'ridge','tree_search':[{}]},
    {'custom_features':1}
])
def test_bounded_tree_and_feature_recipes(recipe):
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_alpha_cache_cannot_bypass_declared_inputs(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    record=store.get_campaign(cid)['plan']['development'][0]
    data=catalog.frames[record['date']]
    service=ModelService(store,catalog,settings)
    service.alpha(candidate,record,data,None)
    restricted=dict(candidate,spec=dict(candidate['spec'],input_features=['micro_dev_bps']))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        service.alpha(restricted,record,data,None)


def test_formula_cache_cannot_bypass_declared_inputs(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    formula={k:v for k,v in candidate['spec'].items() if k!='model'}
    candidate=dict(candidate,spec=formula)
    evaluator.evaluate(cid,candidate,'development',development_indices=[0])
    restricted=dict(candidate,spec=dict(formula,input_features=['micro_dev_bps']))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        evaluator.evaluate(cid,restricted,'development',development_indices=[0])


def test_v3_bundle_depth_is_enforced_and_cannot_be_weakened(tmp_path,monkeypatch):
    import quantdesk.paper as paper
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    strategy={k:v for k,v in candidate['spec'].items() if k!='model'}
    code=CODE.replace('qi_5','qi_20')
    ident,_=store.candidate(cid,strategy,code)
    candidate=store.get_candidate(ident)
    directory=evaluator.export(candidate,{'status':'PROPOSED'},True)
    manifest,_=load_bundle(directory,True)
    assert manifest['input_contract']['minimum_book_levels']==20
    monkeypatch.setattr(paper,'run_signal',lambda *args:pytest.fail('Insufficient depth reached inference'))
    session=PaperSession(directory,tmp_path,settings,True)
    ts=1790770000000
    session.consume(public_fixture_book(100,0.2,ts),ts)
    assert session.halted and session.members[0]['broker'].orders==0
    path=directory/'manifest.json'
    manifest=json.loads(path.read_text()); manifest['input_contract']['minimum_book_levels']=5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Input contract integrity'):
        load_bundle(directory,True)


def test_walkforward_reference_uses_its_own_feature_inputs(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    strategy=dict(candidate['spec'],input_features=['qi_5'])
    ident,_=store.candidate(cid,strategy,CODE)
    result=evaluator.walk_forward(cid,store.get_candidate(ident),1,8,8)
    reference=store.get_candidate(result['comparisons']['depth_flow_baseline']['candidate_id'])
    assert 'input_features' not in reference['spec']


def test_development_model_exports_as_exploratory_without_fake_validation(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    result=evaluator.walk_forward(cid,candidate,1,8,8)
    assert candidate['report'] is None
    directory=evaluator.export(candidate,{},True)
    manifest,_=load_bundle(directory,True)
    assert manifest['status']=='EXPLORATORY'
    assert manifest['model_evidence_phase']=='development'
    assert manifest['validation_summary']=={} and manifest['holdout_summary']=={}
    assert manifest['model_hash']==result['model_artifacts'][-1]['hash']
    assert manifest['development_summary']['days']==1
    with pytest.raises(ValueError,match='exploratory'):
        load_bundle(directory)


def test_prediction_frontier_separates_extra_margin_from_real_costs():
    from quantdesk.models import prediction_frontier
    data=frame('2026-01-01',100)
    strategy=validate_spec(dict(spec(),warmup_rows=30,model={
        'features':['qi_5'],'include_alpha':False,'min_edge_bps':20}))
    recipe=strategy['model']; execution=Settings().execution_dict()
    fitted=fit_ridge(np.zeros((100,1)),np.full(100,15.0),1.0,'ridge')
    artifact={'recipe':recipe,'execution':execution,'fitted':fitted}
    result=prediction_frontier(data,np.zeros(100),artifact,strategy)
    assert result['rows_above_modeled_costs']==100
    assert result['rows_above_costs_plus_margin']==0
    artifact['recipe']=dict(recipe,min_edge_bps=0)
    result=prediction_frontier(data,np.zeros(100),artifact,strategy)
    assert result['rows_above_margin_and_signal_threshold']==100
    assert result['rows_also_passing_warmup_funding_spread']==70


def test_prediction_diagnostics_are_development_only(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    result=evaluator.diagnose_prediction(cid,candidate,8)
    assert result['date']=='2026-01-09' and result['phase']=='development'
    assert result['outer_prediction_diagnostics']['interpretation'].startswith('Outer development')
    assert not set(catalog.reads)&{'2026-01-13','2026-01-14','2026-01-15'}
    with pytest.raises(ValueError,match='development index'):
        evaluator.diagnose_prediction(cid,candidate,12)


def test_walkforward_is_valid_grounding_for_an_organizational_lesson(tmp_path,monkeypatch):
    from quantdesk.agents import ResearchTools
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    evaluator.walk_forward(cid,candidate,1,8,8)
    tools=ResearchTools(cid,store,catalog,evaluator,settings)
    tools.call('historian','record_lesson',{'candidate_id':candidate['id'],
        'lesson':'Fixture research only','evidence':'Actual walk-forward record',
        'next_experiment':'No profitability claim'})
    with store.connect() as db:
        grounding=json.loads(db.execute("SELECT payload FROM memories WHERE kind='LESSON'").fetchone()[0])['grounding']
    assert grounding['development_evidence_kind']=='WALK_FORWARD_EXPERIMENT'
