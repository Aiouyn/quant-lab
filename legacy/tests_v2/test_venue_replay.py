"""Synthetic capture/candidate fixtures; no financial acceptance claims."""
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from quantdesk import venue_replay as module
from quantdesk import venue_study
from quantdesk.agents import ResearchTools,report_observation
from quantdesk.data import Book
from quantdesk.models import fit_ridge
from quantdesk.planning import planning_contract
from quantdesk.research_map import research_map
from quantdesk.store import digest
from quantdesk.strategy import validate_spec
from tests.test_venue_study import setup,book,T,IDENT,count
from tests.test_quantdesk import spec
from tests.test_data_feasibility import segmented

CODE='import numpy as np\nimport polars as pl\ndef compute_signal(df):\n    return pl.Series("signal",np.ones(df.height))\n'


@pytest.fixture(autouse=True)
def dead_capture(monkeypatch): monkeypatch.setattr(venue_study,'process_alive',lambda pid:False)


def environment(tmp_path,fitted=False):
    events=[book(T+i*500) for i in range(24)]
    store,settings,directory,path=setup(tmp_path,events)
    cid=store.campaign({'hash':'venue-replay-fixture','development':[{'date':'2026-01-01'}],
        'validation':[],'holdout':[],'sampled':False,'max_rows_per_day':None},1000)
    s=validate_spec(spec(holding_seconds=1,warmup_rows=0,model={'features':['qi_5'],
        'include_alpha':False,'train_days':8,'min_train_days':6,'inner_days':2}) if fitted else spec(holding_seconds=1,warmup_rows=0))
    candidate_id,_=store.candidate(cid,s,CODE)
    result=venue_study.study(store,settings,IDENT)
    model_key=None
    if fitted:
        x=np.linspace(-1,1,100)[:,None]
        artifact={'format':'quantdesk-fitted-v1','recipe':s['model'],'fitted':fit_ridge(x,20+10*x[:,0],.1,'ridge'),
            'code_hash':digest(CODE),'spec_hash':digest(s),'execution':settings.execution_dict(),
            'provenance':{'prediction_cutoff_ms':T-1000,'max_label_end_ms':T-1000000,
                'inner_prediction_cutoff_ms':T-200000,'inner_max_label_end_ms':T-2000000,
                'inner_training_dates':['2025-12-10'],'inner_selection_dates':['2025-12-11'],
                'training_dates':['2025-12-10','2025-12-11']}}
        artifact['artifact_hash']=digest(artifact);model_key='a'*64
        (store.home/'models').mkdir();(store.home/'models'/(model_key+'.json')).write_text(json.dumps(artifact))
    return store,settings,cid,candidate_id,result['study_id'],model_key,path


@pytest.mark.parametrize('fitted',[False,True])
def test_real_worker_formula_and_frozen_model_replay_reconcile_and_cache(tmp_path,fitted):
    store,settings,cid,candidate,study,key,_=environment(tmp_path,fitted)
    model_before=(store.home/'models'/(key+'.json')).read_bytes() if key else None
    result=module.replay(store,settings,cid,candidate,study,key);r=result['report']
    assert r['rows']==24 and r['quote_arm']['trades']>0 and r['depth_arm']['partial_fills']>0
    assert r['quote_arm']['fees']>r['depth_arm']['fees']>0
    assert r['quote_arm']['cash_position_fee_reconciled'] and r['depth_arm']['cash_position_fee_reconciled']
    assert not r['strategy_promoted'] and not r['forward_eligible'] and not r['orders_submitted_to_exchange']
    assert r['prediction_diagnostic'] is not None if fitted else r['prediction_diagnostic'] is None
    assert digest(r)==result['replay_id']
    assert module.replay(store,settings,cid,candidate,study,key)['already_measured']
    assert len(module.list_models(store,candidate,settings))==int(fitted)
    if key: assert (store.home/'models'/(key+'.json')).read_bytes()==model_before
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM memories WHERE kind='VENUE_REPLAY'").fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM data_exposures').fetchone()[0]==1
    assert store.get_candidate(candidate)['status']=='PROPOSED'
    for table in ('requests','holdouts','holdout_dates','forward_plans'): assert count(store,table)==0


def test_terminal_depth_residual_is_preserved_and_not_fabricated_as_closed_trade(tmp_path):
    store,settings,cid,candidate,_,_,path=environment(tmp_path)
    lines=[json.loads(v) for v in path.read_text().splitlines()]
    for event in lines:
        if event.get('channel')=='l2Book':
            for level in event['message']['data']['levels'][0]: level['sz']='.00001'
    path.write_text(''.join(json.dumps(e)+'\n' for e in lines))
    study=venue_study.study(store,settings,IDENT)['study_id']
    result=module.replay(store,settings,cid,candidate,study)['report']
    assert result['depth_arm']['open_quantity']>0 and result['depth_arm']['trades']==0
    assert result['depth_arm']['partial_fills']>1
    assert not result['forward_eligible']


@pytest.mark.parametrize('problem',['wrong_campaign','maker','depth','sealed','gap','model_missing','future_model','wrong_model','formula_model'])
def test_preflight_failures_cannot_start_economic_experiment(tmp_path,problem):
    fitted=problem in {'model_missing','future_model','wrong_model'}
    store,settings,cid,candidate,study,key,_=environment(tmp_path,fitted)
    if problem=='wrong_campaign': cid=store.campaign({'hash':'other'},1000)
    if problem in {'maker','depth'}:
        s=store.get_candidate(candidate)['spec']
        s=dict(s,mode='maker') if problem=='maker' else dict(s,input_features=['qi_20'])
        candidate,_=store.candidate(cid,s,CODE)
    if problem=='sealed': store.set_status(cid,'SEALED')
    if problem=='gap': settings=replace(settings,max_gap_ms=1000)
    if problem=='model_missing': key=None
    if problem in {'future_model','wrong_model'}:
        path=store.home/'models'/(key+'.json');a=json.loads(path.read_text());a.pop('artifact_hash')
        if problem=='future_model': a['provenance']['prediction_cutoff_ms']=T+1000
        else: a['code_hash']='wrong-code'
        a['artifact_hash']=digest(a);path.write_text(json.dumps(a))
    if problem=='formula_model': key='a'*64
    with pytest.raises(ValueError): module.replay(store,settings,cid,candidate,study,key)
    with store.connect() as db: assert not db.execute("SELECT 1 FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()
    assert not (store.home/'venue_replays').exists()


def test_changed_raw_source_and_tampered_replay_journal_are_refused(tmp_path):
    store,settings,cid,candidate,study,key,path=environment(tmp_path)
    result=module.replay(store,settings,cid,candidate,study)
    journal=store.home/'venue_replays'/result['replay_id']/'depth.events.jsonl'
    journal.write_text('changed')
    with pytest.raises(ValueError,match='journal integrity'): module.replay(store,settings,cid,candidate,study)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='exact study source'): module.replay(store,settings,cid,candidate,study)


def test_source_mutation_and_cancellation_during_compute_leave_no_replay(tmp_path,monkeypatch):
    store,settings,cid,candidate,study,_,path=environment(tmp_path)
    real=module.run_signal
    def changing(*args,**kwargs):
        result=real(*args,**kwargs);path.write_text(path.read_text()+'\n');return result
    monkeypatch.setattr(module,'run_signal',changing)
    with pytest.raises(ValueError,match='changed before replay commit'): module.replay(store,settings,cid,candidate,study)
    assert not (store.home/'venue_replays').exists()


def test_book_limit_and_cooperative_stop_do_not_write_replay(tmp_path,monkeypatch):
    store,settings,cid,candidate,study,_,_=environment(tmp_path)
    monkeypatch.setattr(module,'MAX_BOOKS',1)
    with pytest.raises(OverflowError): module.replay(store,settings,cid,candidate,study)
    monkeypatch.setattr(module,'MAX_BOOKS',200000)
    def stop(*args): raise RuntimeError('cooperative-stop-fixture')
    monkeypatch.setattr(store,'check_research_control',stop)
    with pytest.raises(RuntimeError,match='cooperative-stop'): module.replay(store,settings,cid,candidate,study)
    assert not (store.home/'venue_replays').exists()


def test_implementation_change_and_cached_report_tamper_are_refused(tmp_path,monkeypatch):
    store,settings,cid,candidate,study,_,_=environment(tmp_path)
    original=module.implementation; calls=[0]
    def changing():
        calls[0]+=1
        return original() if calls[0]==1 else {'sources':{'changed':'during-compute'}}
    monkeypatch.setattr(module,'implementation',changing)
    with pytest.raises(ValueError,match='implementation changed'): module.replay(store,settings,cid,candidate,study)
    assert not (store.home/'venue_replays').exists()
    monkeypatch.setattr(module,'implementation',original)
    result=module.replay(store,settings,cid,candidate,study)
    with store.connect() as db:
        row=db.execute('SELECT key,result FROM tool_cache').fetchone();cached=json.loads(row['result'])
        cached['report']['forward_eligible']=True
        db.execute('UPDATE tool_cache SET result=? WHERE key=?',(json.dumps(cached),row['key']))
    with pytest.raises(ValueError,match='replay integrity'): module.replay(store,settings,cid,candidate,study)


def test_connection_resets_and_configured_gap_reset_feature_history(tmp_path):
    store,settings,_,_,_,_,_=environment(tmp_path)
    events=[book(),book(T+500),{'kind':'DISCONNECTED'},book(T+1000),book(T+3000)]
    directory=store.home/'market_data'/IDENT;path=directory/'2026-01-01.jsonl'
    lines=path.read_text().splitlines();path.write_text(lines[0]+'\n'+''.join(json.dumps(e)+'\n' for e in events)+lines[-1]+'\n')
    settings=replace(settings,max_gap_ms=1000)
    report=venue_study.study(store,settings,IDENT)['report']
    frame,_=module.capture_frame(store,report,settings)
    assert frame['segment'].to_list()==[0,0,1,2]
    assert frame['ofi'].to_list()==[0,0,0,0]
    b=Book(max_gap_ms=1000)
    first={'action':'snapshot','_continuous_snapshot':True,'ts':0,'bids':[[100,10]],'asks':[[101,10]]}
    b.apply(first);b.apply(dict(first,ts=2000))
    assert b.quality.timestamp_gaps==1
    with pytest.raises(ValueError): Book(max_gap_ms=True)


def test_study_versions_are_not_independent_planning_entries_and_ids_survive_short_handoffs(tmp_path,monkeypatch):
    store,settings,cid,_,study,_,_=environment(tmp_path)
    first=venue_study.load(store,study);manifest=venue_study.algorithm_manifest()
    store.memory(cid,'','LESSON',{'lesson':'Grounded older lesson retained'})
    for i in range(3):
        monkeypatch.setattr(venue_study,'algorithm_manifest',lambda:dict(manifest,numpy_version='version-'+str(i)))
        receipt=venue_study.study(store,settings,IDENT)
        store.memory(cid,'','VENUE_EXECUTION_STUDY',{'study_id':receipt['study_id'],'capture_id':IDENT,'coverage_verified':False})
    summary=venue_study.summaries(store)
    assert len(summary)==1 and summary[0]['study_id']==receipt['study_id']
    assert venue_study.load(store,study)==first
    assert len(research_map(store,cid)['available_venue_studies'])==1
    recall=store.recall(4)
    assert recall[0]['kind']=='LESSON'
    assert len([r for r in recall if r['kind']=='VENUE_EXECUTION_STUDY'])==1
    condensed=report_observation({'tool':'recall','result':recall})
    assert receipt['study_id'] in json.dumps(condensed)
    catalog=SimpleNamespace(cached=lambda *a:False,load=lambda *a:(segmented('2026-01-01'),{}))
    contract=planning_contract(store,catalog,cid,settings)
    assert contract['target_venue_observations'][0]['study_id']==receipt['study_id']


def test_agent_roles_expose_native_replay_without_granting_portfolio_access(tmp_path):
    store,settings,cid,candidate,study,_,_=environment(tmp_path)
    tools=ResearchTools(cid,store,None,None,settings)
    result=tools.call('falsifier','replay_venue_candidate',{'candidate_id':candidate,'study_id':study})
    assert result['report']['forward_eligible'] is False
    assert tools.call('director','read_venue_replay',{'replay_id':result['replay_id']})['report']==result['report']
    mapped=research_map(store,cid)['available_venue_replays']
    assert mapped[0]['replay_id']==result['replay_id'] and not mapped[0]['forward_eligible']
    assert result['replay_id'] in json.dumps(report_observation({'tool':'recall','result':store.recall(4)}))
    assert tools.call('engineer','list_frozen_models',{'candidate_id':candidate})==[]
    with pytest.raises(ValueError,match='own that tool'):
        tools.call('director','replay_venue_candidate',{'candidate_id':candidate,'study_id':study})


def test_reserved_holdout_dates_block_economic_venue_access_before_worker(tmp_path,monkeypatch):
    store,settings,cid,candidate,study,_,_=environment(tmp_path)
    store.reserve_holdout('another-plan',[candidate],[{'date':'2026-01-01'}])
    def forbidden(*args): raise AssertionError('No numerical worker on sealed dates')
    monkeypatch.setattr(module,'run_signal',forbidden)
    with pytest.raises(ValueError,match='reserved holdout dates'): module.replay(store,settings,cid,candidate,study)
    with store.connect() as db: assert not db.execute("SELECT 1 FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()


def test_exact_replay_reads_verify_content_and_never_use_free_form_paths(tmp_path):
    store,settings,cid,candidate,study,_,_=environment(tmp_path)
    result=module.replay(store,settings,cid,candidate,study)
    assert module.load(store,result['replay_id'])['report']==result['report']
    with pytest.raises(ValueError,match='exact'): module.load(store,'../report.json')
    path=store.home/'venue_replays'/result['replay_id']/'report.json'
    bound=json.loads(path.read_text());bound['report']['forward_eligible']=True;path.write_text(json.dumps(bound))
    with pytest.raises(ValueError,match='integrity'): module.load(store,result['replay_id'])
