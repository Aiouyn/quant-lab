"""Public capture measurements: synthetic fixtures, never strategy evidence."""
import json
import os
from dataclasses import replace
from pathlib import Path
import pytest
from quantdesk import venue_study as module
from quantdesk.agents import ResearchTools
from quantdesk.research_map import research_map
from quantdesk.settings import Settings
from quantdesk.store import Store, digest

IDENT='hyperliquid-20260101-000000-abc123'
T=1767225600000


@pytest.fixture(autouse=True)
def dead_capture_process(monkeypatch):
    monkeypatch.setattr(module,'process_alive',lambda pid:False)


def book(t=T,received=None,levels=5):
    data={'coin':'BTC','time':t,'levels':[
        [{'px':str(99.99-i*.01),'sz':'1','n':2} for i in range(levels)],
        [{'px':str(100.01+i*.01),'sz':'1','n':2} for i in range(levels)]]}
    return packet('l2Book',data,t+10 if received is None else received)


def packet(channel,data,received):
    return {'kind':'PUBLIC_MESSAGE','channel':channel,'message':{'channel':channel,'data':data},'received_at_ms':received}


def trade(t,received,tid=7,px=100.02):
    return packet('trades',[{'coin':'BTC','time':t,'tid':tid,'px':str(px),'sz':'.1','side':'B',
                            'users':['PRIVATE_SENTINEL_BUYER','PRIVATE_SENTINEL_SELLER']}],received)


def setup(tmp_path,events=None,end=True,pid=999999,endpoint='wss://api.hyperliquid.xyz/ws'):
    store=Store(tmp_path); settings=replace(Settings(),home=tmp_path)
    directory=tmp_path/'market_data'/IDENT; directory.mkdir(parents=True)
    initial={'kind':'CAPTURE_START','coin':'BTC','endpoint':endpoint,'book_mode':'fast_5_levels'}
    records=[initial,{'kind':'CONNECTED'},*(events or [book(),book(T+500)])]
    if end: records.append({'kind':'CAPTURE_END','summary':{'coverage_verified':False}})
    path=directory/'2026-01-01.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in records),encoding='utf-8')
    status={'status':'COMPLETE' if end else 'RUNNING','book_mode':'fast_5_levels'}
    if pid is not None: status['pid']=pid
    (directory/'status.json').write_text(json.dumps(status),encoding='utf-8')
    return store,settings,directory,path


def count(store,table):
    with store.connect() as db: return db.execute('SELECT count(*) FROM '+table).fetchone()[0]


def test_sweep_hand_calculation_independent_scenarios_and_frozen_source(tmp_path):
    store,settings,directory,path=setup(tmp_path)
    original={p.name:p.read_bytes() for p in directory.iterdir()}
    result=module.study(store,settings,IDENT,[50,1000]); report=result['report']
    scenarios={(v['requested_notional_usd'],v['side']):v for v in report['hypothetical_sweeps']}
    assert scenarios[50,'BUY']['filled_fraction']['mean']==pytest.approx(1)
    assert scenarios[1000,'BUY']['filled_fraction']['mean']==pytest.approx(.05)
    assert scenarios[1000,'SELL']['filled_fraction']['mean']==pytest.approx(.05)
    assert scenarios[50,'BUY']['entry_drag_bps']['mean']==pytest.approx(3.50015)
    assert scenarios[50,'SELL']['entry_drag_bps']['mean']==pytest.approx(3.49985)
    assert scenarios[1000,'BUY']['fraction_fully_filled']==0
    assert scenarios[50,'BUY']['fraction_fully_filled']==1
    assert report['coverage_verified'] is report['strategy_pnl_computed'] is False
    assert report['orders_submitted_to_exchange']==0
    assert digest(report)==result['study_id']
    assert report['depth_observations']==[{'bid_levels':5,'ask_levels':5,'books':2}]
    assert {p.name:p.read_bytes() for p in directory.iterdir()}==original
    saved=json.loads((store.home/'venue_studies'/(result['study_id']+'.json')).read_text())
    assert saved['report']==report
    repeat=module.study(store,settings,IDENT,[1000,50])
    assert repeat['already_measured'] and repeat['study_id']==result['study_id']
    assert count(store,'venue_studies')==1
    for table in ('requests','candidates','holdouts','data_exposures','forward_plans'): assert count(store,table)==0


def test_print_join_is_causal_deduplicated_and_excludes_user_addresses(tmp_path):
    events=[trade(T-100,T-50),book(),trade(T-10,T+20),trade(T+5,T+5),
            trade(T+50,T+60),trade(T+50,T+70),trade(T+50,T+80,px=101),
            trade(T+6000,T+6010),{'kind':'DISCONNECTED'},trade(T+6001,T+6020)]
    store,settings,_,_=setup(tmp_path,events)
    report=module.study(store,settings,IDENT)['report']; q=report['quality_counts']
    assert q['valid_unique_trades']==6
    assert q['duplicate_trades']==2 and q['conflicting_duplicate_trades']==1
    assert q['prints_without_previous_quote']==2
    assert q['prints_excluded_future_or_unavailable_quote']==2
    assert q['prints_excluded_stale_quote']==1
    assert q['prints_with_causal_fresh_quote']==1 and q['prints_above_previous_ask']==1
    assert report['observations']['quote_age_at_print_ms']['mean']==50
    assert report['trades']['observed_volume_btc']==pytest.approx(.6)
    assert 'PRIVATE_SENTINEL' not in json.dumps(report)


def test_gap_bounded_freshness_and_connection_segments(tmp_path):
    events=[book(),book(T+500),book(T+8000),book(T+8500),{'kind':'DISCONNECTED'},book(T+9000)]
    store,settings,_,_=setup(tmp_path,events)
    r=module.study(store,settings,IDENT)['report']
    assert r['quality_counts']['book_intervals_over_gap_limit']==1
    assert r['segments']['count']==3
    assert r['segments']['rows']['mean']==pytest.approx(5/3)
    assert r['book_window']['span_seconds']==9
    assert r['book_window']['bounded_previous_quote_freshness_seconds']==6
    assert r['book_window']['bounded_freshness_fraction']==pytest.approx(2/3)
    assert not r['coverage_verified']


@pytest.mark.parametrize('breaker',[{'kind':'DISCONNECTED'},book(T-1),packet('l2Book',None,T+10)])
def test_repeated_timestamp_does_not_restore_broken_quote_continuity(tmp_path,breaker):
    store,settings,_,_=setup(tmp_path,[book(),breaker,book(),trade(T+100,T+110),book(T+500),trade(T+510,T+520)])
    r=module.study(store,settings,IDENT)['report'];q=r['quality_counts']
    assert q['same_timestamp_books']==1
    assert q['prints_without_previous_quote']==1 and q['prints_with_causal_fresh_quote']==1
    assert r['book_window']['bounded_previous_quote_freshness_seconds']==0


def test_duplicate_book_updates_available_quote_without_duplicate_sweep(tmp_path):
    changed=book();changed['message']['data']['levels'][1][0]['px']='100.005'
    store,settings,_,_=setup(tmp_path,[book(),changed,trade(T+10,T+20,px=100.007)])
    r=module.study(store,settings,IDENT)['report']
    assert r['quality_counts']['valid_unique_books']==1
    assert r['quality_counts']['prints_above_previous_ask']==1
    assert all(s['filled_fraction']['observations']==1 for s in r['hypothetical_sweeps'])


def test_invalid_records_and_wide_spreads_are_visible_and_not_calibrations(tmp_path):
    crossed=book(T+20);crossed['message']['data']['levels'][0][0]['px']='101'
    wide=book(T+30,levels=1);wide['message']['data']['levels'][1][0]['px']='101'
    context=packet('activeAssetCtx',{'coin':'BTC','ctx':{'funding':'.001','markPx':'100','oraclePx':'0'}},T+40)
    bad=packet('trades',[None,{'coin':'BTC','side':'B','time':T,'tid':'bad','px':'100','sz':'1'}],T+50)
    store,settings,_,_=setup(tmp_path,[book(),crossed,wide,context,bad])
    r=module.study(store,settings,IDENT)['report'];q=r['quality_counts']
    assert q['invalid_records']==2 and q['invalid_trades']==2
    assert q['books_excluded_by_spread_gate']==1
    assert r['observations']['indicative_funding_rate']['observations']==0
    assert r['observations']['spread_bps']['observations']==2
    assert all(s['filled_fraction']['observations']==1 for s in r['hypothetical_sweeps'])


def test_indicative_context_and_testnet_are_explicit(tmp_path):
    context=packet('activeAssetCtx',{'coin':'BTC','ctx':{'funding':'-.0001','markPx':'100.1','oraclePx':'100'}},T)
    store,settings,_,_=setup(tmp_path,[context,book()],endpoint='wss://api.hyperliquid-testnet.xyz/ws')
    r=module.study(store,settings,IDENT)['report']
    assert r['mainnet'] is False and r['coverage_verified'] is False
    assert r['observations']['indicative_funding_rate']['mean']==-.0001
    assert r['observations']['mark_oracle_basis_bps']['mean']==pytest.approx(10)


def test_interrupted_source_requires_observed_process_absence(tmp_path):
    store,settings,directory,_=setup(tmp_path,end=False)
    r=module.study(store,settings,IDENT)['report']
    assert r['effective_source_state']=='INTERRUPTED_NO_END_MARKER'
    assert r['capture_end_present'] is False
    assert module.inventory(store.home)[0]['effective_status']=='INTERRUPTED'
    (directory/'status.json').write_text('{"status":"RUNNING"}')
    with pytest.raises(ValueError,match='termination/liveness'): module.study(store,settings,IDENT)
    assert module.inventory(store.home)[0]['effective_status']=='UNVERIFIED_LIVENESS'
    assert count(store,'venue_studies')==1


def test_live_source_is_refused_even_when_status_claims_complete(tmp_path,monkeypatch):
    store,settings,_,_=setup(tmp_path)
    monkeypatch.setattr(module,'process_alive',lambda pid:True)
    with pytest.raises(ValueError,match='process is live'): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


@pytest.mark.parametrize('corrupt',['[]','not-json','{"pid":true,"status":"RUNNING"}'])
def test_inventory_handles_invalid_status_without_inventing_liveness(tmp_path,corrupt):
    store,settings,directory,_=setup(tmp_path)
    (directory/'status.json').write_text(corrupt)
    row=module.inventory(store.home)[0]
    assert row['process_alive'] is None
    with pytest.raises(ValueError): module.study(store,settings,IDENT)


@pytest.mark.parametrize('notionals',[[],[True],[49],[100001],[float('nan')],[50,50],(50,),[50]*7])
def test_invalid_scenario_parameters_leave_no_evidence(tmp_path,notionals):
    store,settings,_,_=setup(tmp_path)
    with pytest.raises(ValueError,match='notional'): module.study(store,settings,IDENT,notionals)
    assert count(store,'venue_studies')==0


@pytest.mark.parametrize('assumptions',[{'max_gap_ms':True},{'slippage_bps':-1},{'max_spread_bps':float('inf')}])
def test_invalid_execution_parameters_leave_no_evidence(tmp_path,assumptions):
    store,settings,_,_=setup(tmp_path)
    with pytest.raises(ValueError): module.study(store,replace(settings,**assumptions),IDENT)
    assert count(store,'venue_studies')==0


def test_limits_malformed_lines_and_cancellation_are_enforced(tmp_path,monkeypatch):
    store,settings,_,path=setup(tmp_path)
    monkeypatch.setattr(module,'MAX_RECORDS',2)
    with pytest.raises(OverflowError): module.study(store,settings,IDENT)
    path.write_text(path.read_text()+('not-json\n'*5))
    with pytest.raises(OverflowError): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0
    def stop(): raise RuntimeError('cooperative cancellation')
    with pytest.raises(RuntimeError,match='cancellation'): module.study(store,settings,IDENT,control=stop)
    assert count(store,'venue_studies')==0


@pytest.mark.parametrize('limit',[('MAX_BYTES',1),('MAX_TRADES',0),('MAX_BOOKS',0)])
def test_each_resource_limit_blocks_artifact(tmp_path,monkeypatch,limit):
    store,settings,_,_=setup(tmp_path,[book(),trade(T+10,T+20)])
    monkeypatch.setattr(module,*limit)
    with pytest.raises((ValueError,OverflowError)): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


@pytest.mark.parametrize('rewrite',['missing_start','multiple_start','after_end'])
def test_provenance_errors_cannot_become_studies(tmp_path,rewrite):
    store,settings,_,path=setup(tmp_path)
    lines=path.read_text().splitlines()
    if rewrite=='missing_start': lines=lines[1:]
    elif rewrite=='multiple_start': lines.insert(2,lines[0])
    else: lines.append(json.dumps(book(T+1000)))
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(RuntimeError): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


def test_source_change_during_scan_never_commits(tmp_path,monkeypatch):
    store,settings,_,path=setup(tmp_path)
    consume=module.Measurements.consume
    def changing(self,event):
        consume(self,event)
        if event['kind']=='PUBLIC_MESSAGE': path.write_text(path.read_text()+'\n')
    monkeypatch.setattr(module.Measurements,'consume',changing)
    with pytest.raises((ValueError,RuntimeError),match='changed|terminal'): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


def test_earlier_source_file_change_while_later_file_is_read_never_commits(tmp_path,monkeypatch):
    store,settings,directory,path=setup(tmp_path)
    lines=path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n')
    (directory/'2026-01-02.jsonl').write_text(json.dumps(book(T+86400000))+'\n'+lines[-1]+'\n')
    consume=module.Measurements.consume
    def changing(self,event):
        consume(self,event)
        if event.get('channel')=='l2Book' and event['message']['data']['time']>T+1000:
            path.write_text(path.read_text()+'\n')
    monkeypatch.setattr(module.Measurements,'consume',changing)
    with pytest.raises(ValueError,match='before study commit'): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


def test_resolved_capture_and_file_escape_guards_on_platform_without_symlinks(tmp_path,monkeypatch):
    store,settings,directory,path=setup(tmp_path)
    resolve=Path.resolve
    escaped=tmp_path/'outside'
    def directory_escape(self,*args,**kwargs):
        return escaped if self==directory else resolve(self,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',directory_escape)
    with pytest.raises(ValueError,match='Unknown public capture'): module.study(store,settings,IDENT)
    def file_escape(self,*args,**kwargs):
        return escaped/'2026-01-01.jsonl' if self==path else resolve(self,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',file_escape)
    with pytest.raises(ValueError,match='escapes'): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


def test_changed_input_creates_new_immutable_study_and_payload_tamper_is_refused(tmp_path):
    store,settings,_,path=setup(tmp_path)
    first=module.study(store,settings,IDENT)
    lines=path.read_text().splitlines();lines.insert(-1,json.dumps(book(T+1000)));path.write_text('\n'.join(lines)+'\n')
    second=module.study(store,settings,IDENT)
    assert first['study_id']!=second['study_id'] and not second['already_measured']
    assert module.load(store,first['study_id'])==first['report']
    assert count(store,'venue_studies')==2
    with store.connect() as db: db.execute('UPDATE venue_studies SET payload=? WHERE id=?',('{}',first['study_id']))
    with pytest.raises(ValueError,match='integrity'): module.load(store,first['study_id'])


def test_liquidity_engine_and_numeric_version_are_bound_to_cache_identity(tmp_path,monkeypatch):
    store,settings,_,_=setup(tmp_path)
    first=module.study(store,settings,IDENT)
    real=module.algorithm_manifest()
    assert 'liquidity.py' in real['sources_sha256'] and real['numpy_version']
    changed=dict(real,sources_sha256=dict(real['sources_sha256'],**{'liquidity.py':'different-engine'}))
    monkeypatch.setattr(module,'algorithm_manifest',lambda:changed)
    second=module.study(store,settings,IDENT)
    assert not second['already_measured'] and second['study_id']!=first['study_id']
    assert second['report']['hypothetical_sweeps']==first['report']['hypothetical_sweeps']
    assert module.load(store,first['study_id'])==first['report']


def test_algorithm_change_during_measurement_refuses_commit(tmp_path,monkeypatch):
    store,settings,_,_=setup(tmp_path)
    manifest=module.algorithm_manifest(); calls=[0]
    def changing():
        calls[0]+=1
        return manifest if calls[0]==1 else dict(manifest,numpy_version='different-runtime')
    monkeypatch.setattr(module,'algorithm_manifest',changing)
    with pytest.raises(ValueError,match='implementation changed'): module.study(store,settings,IDENT)
    assert count(store,'venue_studies')==0


def test_exact_capture_ids_and_source_symlinks_cannot_escape_home(tmp_path):
    store,settings,_,_=setup(tmp_path)
    for ident in ['../'+IDENT,'garbage','hyperliquid-20260101-000000-ffffff']:
        with pytest.raises(ValueError): module.study(store,settings,ident)
    outside=tmp_path/'outside';outside.mkdir();(outside/'2026-01-01.jsonl').write_text('{}')
    alias=tmp_path/'market_data'/'hyperliquid-20260101-000001-ffffff'
    try: os.symlink(outside,alias,target_is_directory=True)
    except OSError: pytest.skip('OS does not permit directory symlink fixture')
    # Outside the market_data subtree, even though still inside the test home.
    with pytest.raises(ValueError): module.study(store,settings,alias.name)
    assert alias.name not in [r['capture_id'] for r in module.inventory(store.home)]


def test_agent_tools_save_once_expose_exact_map_and_never_grant_promotion(tmp_path):
    store,settings,_,_=setup(tmp_path)
    plan={'hash':'metadata-only','development':[],'validation':[],'holdout':[],'sampled':False}
    cid=store.campaign(plan,1000)
    tools=ResearchTools(cid,store,None,None,settings)
    assert tools.call('director','list_venue_captures',{})[0]['capture_id']==IDENT
    first=tools.call('scientist','study_venue_capture',{'capture_id':IDENT,'notionals_usd':[50,1000]})
    repeated=tools.call('engineer','study_venue_capture',{'capture_id':IDENT,'notionals_usd':[1000,50]})
    assert repeated['already_measured'] and repeated['study_id']==first['study_id']
    assert tools.call('falsifier','read_venue_study',{'study_id':first['study_id']})['report']==first['report']
    mapped=research_map(store,cid)['available_venue_studies']
    assert mapped[0]['study_id']==first['study_id']
    assert mapped[0]['coverage_verified'] is mapped[0]['strategy_pnl_computed'] is False
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM memories WHERE kind='VENUE_EXECUTION_STUDY'").fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM events WHERE kind='VENUE_STUDY_REUSED'").fetchone()[0]==1
        assert not db.execute("SELECT 1 FROM events WHERE kind='EXPERIMENT_STARTED'").fetchone()
        db.execute("UPDATE campaigns SET status='SEALED' WHERE id=?",(cid,))
    with pytest.raises(ValueError,match='Sealed'): tools.call('historian','study_venue_capture',{'capture_id':IDENT})
    assert tools.call('historian','read_venue_study',{'study_id':first['study_id']})['report']==first['report']
    for table in ('requests','candidates','holdouts','data_exposures','forward_plans','research_decisions'): assert count(store,table)==0
