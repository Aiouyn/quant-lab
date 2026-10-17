"""Synthetic adversarial forward receipts; these fixtures are not venue evidence."""
import json
import time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import pytest
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.forward import DAY,FORMAT,bundle_identity,plan_forward,assess_forward,file_hash
from quantdesk.portfolio import build_portfolio
from tests.test_quantdesk import CODE,spec,bundle

START=int(time.time()*1000)//DAY*DAY-10*DAY


def setup(tmp_path,days=1):
    settings=replace(Settings(),home=tmp_path,max_gap_ms=60000,min_forward_days=days,min_forward_trades=2*days)
    store=Store(tmp_path); cid=store.campaign({},1000)
    strategy=spec(); ident,_=store.candidate(cid,strategy,CODE)
    store.record_report(ident,'PAPER_READY',{})
    directory=tmp_path/'bundles'/ident; directory.mkdir(parents=True)
    manifest={'format':'quantdesk-v2','id':ident,'title':'SYNTHETIC RECEIPT FIXTURE','status':'PAPER_READY',
        'spec':strategy,'spec_hash':digest(strategy),'code_hash':digest(CODE),'execution':settings.execution_dict(),
        'execution_hash':digest(settings.execution_dict()),'shadow_execution':settings.shadow_dict(),
        'shadow_execution_hash':digest(settings.shadow_dict())}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'strategy.py').write_text(CODE)
    plan=plan_forward(store,manifest,settings,now_ms=START-3000)
    return store,cid,manifest,settings,plan


def receipt(store,manifest,settings,plan,name='fixture',active=True):
    directory=store.home/'paper'/name; directory.mkdir(parents=True)
    metadata={'format':FORMAT,'created_ms':START-2000,'run_started_ms':START-1000,
        'data_source':'hyperliquid_public_mainnet','coin':'BTC','exploratory':False,'initial_equity':10000.0,
        'execution':settings.execution_dict(),'shadow_execution':settings.shadow_dict(),'forward_plan_id':plan['id'],
        'members':[{'id':manifest['id'],'status':'PAPER_READY','bundle_identity':bundle_identity(manifest),'execution':settings.execution_dict(),'shadow_execution':settings.shadow_dict(),'weight':1.0}]}
    events=[{'kind':'SESSION_START','initial_equity':10000,'signing_enabled':False,'exploratory':False,'received_at_ms':START-2000},
        {'kind':'SESSION_RUN_START','session_hash':digest(metadata),'received_at_ms':START-1000}]
    cash=10000.0; quantity=0.0; fees=0.0; trades=0; marks=0
    for day in range(settings.min_forward_days):
        for minute in range(1440):
            ts=START+day*DAY+minute*60000
            mid=101.0 if minute in (2,4) and active else 100.0
            bid=mid-0.01; ask=mid+0.01
            events.append({'kind':'BOOK','exchange_ts':ts,'received_at_ms':ts,'book':{'coin':'BTC','time':ts,'levels':[[{'px':str(bid),'sz':'1000','n':1}],[{'px':str(ask),'sz':'1000','n':1}]]}})
            if active and minute in (1,2,3,4):
                side=1 if minute in (1,3) else -1
                qty=2500/mid if side>0 else quantity
                price=(ask if side>0 else bid)*(1+side*settings.slippage_bps/10000)
                fee=qty*price*settings.taker_fee_bps/10000
                cash-=side*qty*price+fee; quantity+=side*qty; fees+=fee
                trades+=int(side<0)
                events.append({'kind':'FILL','ts_ms':ts,'received_at_ms':ts,'candidate':manifest['id'],
                    'side':side,'quantity':qty,'price':price,'fee':fee,'maker_proxy':False,
                    'requested_quantity':qty,'unfilled_quantity':0.0,'model':'displayed-depth-ioc-v1',
                    'levels':[{'price':price,'quantity':qty}]})
            events.append({'kind':'MARK','exchange_ts':ts,'received_at_ms':ts,'equity':cash+quantity*mid,'halted':False})
            marks+=1
    summary={'samples':marks,'equity':cash+quantity*mid,'fees':fees,'closed_trades':trades,
        'orders_submitted_to_exchange':0,'funding_boundary_crossings':0,'strategy_errors':0,'stale_decisions':0,
        'open_positions':{manifest['id']:quantity},'halted':False,'termination':'SAMPLE_LIMIT','session_hash':digest(metadata)}
    events.append({'kind':'SESSION_END','summary':summary.copy(),'received_at_ms':ts})
    save(directory,metadata,summary,events)
    return directory,metadata,summary,events


def save(directory,metadata,summary,events):
    # Recomputing checksums deliberately tests reconciliation beyond edit detection.
    summary['session_hash']=digest(metadata)
    for event in events:
        if event['kind']=='SESSION_RUN_START':event['session_hash']=digest(metadata)
        if event['kind']=='SESSION_END':event['summary']={k:v for k,v in summary.items() if k!='journal_sha256'}
    (directory/'session.json').write_text(json.dumps(metadata))
    (directory/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    summary['journal_sha256']=file_hash(directory/'journal.jsonl')
    (directory/'summary.json').write_text(json.dumps(summary))


def test_plan_is_future_frozen_and_retries_preserve_previous_windows(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    assert plan['start_ms']==START and plan['created_ms']<START
    assert plan_forward(store,manifest,settings,now_ms=START+DAY)==plan
    new=plan_forward(store,manifest,settings,True,START+DAY)
    assert new['start_ms']==START+2*DAY and new['previous_plan_id']==plan['id']
    assert assess_forward(store,manifest,settings,START+3*DAY)['planned_windows']==2


def test_reconciled_complete_window_can_qualify_only_for_paper_allocation(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    receipt(store,manifest,settings,plan)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert report['eligible'] and report['closed_trades']==2 and report['compound_net_return']>0
    assert report['days'][0]['coverage_fraction']>0.99
    assert 'no live' in report['interpretation']
    assert not assess_forward(store,manifest,settings,plan['end_ms']-1)['eligible']


@pytest.mark.parametrize('change',[{'data_source':'custom_fetch_unverified'},{'data_source':'hyperliquid_public_testnet'},{'exploratory':True}])
def test_testnet_custom_and_exploratory_sessions_cannot_qualify(tmp_path,change):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    metadata.update(change); save(directory,metadata,summary,events)
    assert not assess_forward(store,manifest,settings,plan['end_ms']+1)['eligible']


def test_accounting_edit_fails_even_after_all_checksums_are_recomputed(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    next(e for e in events if e['kind']=='MARK')['equity']+=50
    save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and 'reconcile' in report['invalid_sessions'][0]['reason']


def test_fill_fee_edit_is_not_accepted_as_a_profitable_receipt(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    next(e for e in events if e['kind']=='FILL')['fee']=0
    save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and 'fees differ' in report['invalid_sessions'][0]['reason']


def test_overlapping_sessions_are_rejected_instead_of_selecting_a_winner(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    receipt(store,manifest,settings,plan,'one'); receipt(store,manifest,settings,plan,'two')
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and any('Overlapping' in r for r in report['reasons'])


def test_missing_two_hours_cannot_be_disguised_as_a_full_day(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    events=[e for e in events if not(e['kind'] in {'BOOK','MARK'} and START+2*3600000<=e['exchange_ts']<START+4*3600000)]
    summary['samples']-=120
    save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and not report['days']


def test_cash_only_is_not_forward_acceptance(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    receipt(store,manifest,settings,plan,active=False)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and report['closed_trades']==0


def test_old_acceptance_and_modified_bundle_require_new_future_evidence(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    receipt(store,manifest,settings,plan)
    report=assess_forward(store,manifest,settings,plan['end_ms']+15*DAY)
    assert not report['eligible'] and any('stale' in r for r in report['reasons'])
    changed=dict(manifest,title='A different exact bundle')
    assert not assess_forward(store,changed,settings,plan['end_ms']+1)['eligible']


def test_nonfinite_accounting_and_post_receipt_events_are_rejected(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    summary['fees']=float('nan'); save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and 'Nonfinite' in report['invalid_sessions'][0]['reason']
    summary['fees']=0; events.append({'kind':'MARK','received_at_ms':plan['end_ms']-1})
    save(directory,metadata,summary,events)
    assert not assess_forward(store,manifest,settings,plan['end_ms']+1)['eligible']


def test_portfolio_requires_forward_evidence_and_uses_its_aligned_returns(tmp_path):
    store,cid,manifest,settings,plan=setup(tmp_path,5)
    with pytest.raises(ValueError,match='forward allocation gate'):
        build_portfolio(store,cid,settings)
    receipt(store,manifest,settings,plan)
    directory,result=build_portfolio(store,cid,settings)
    assert result['dates']==plan['dates'] and 'forward windows' in result['fitted_on']
    assert result['allocation']['cash_weight']==pytest.approx(0.65)
    assert result['status']=='PAPER_ONLY' and not result['portfolio_holdout_tested']
    assert (directory/'manifest.json').is_file()
    from quantdesk.paper import PaperSession
    session=PaperSession(directory,tmp_path,settings)
    assert session.is_portfolio and session.metadata['portfolio']
    assert session.members[0]['broker'].initial==pytest.approx(3500)


def test_unrelated_corrupt_session_does_not_replace_or_veto_bound_evidence(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    receipt(store,manifest,settings,plan)
    unrelated=store.home/'paper'/'unrelated'; unrelated.mkdir()
    (unrelated/'session.json').write_text('{unfinished')
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert report['eligible'] and len(report['unclassified_sessions'])==1


def test_actual_paper_run_labels_custom_fetch_and_writes_bound_receipt(tmp_path,monkeypatch):
    import time
    import quantdesk.paper as paper
    from tests.test_fitted_models import public_fixture_book
    session=paper.PaperSession(bundle(tmp_path,warmup=5000),tmp_path,Settings(),True)
    monkeypatch.setattr(paper.time,'sleep',lambda *a:None)
    summary=session.run(1,fetch=lambda *a:public_fixture_book(100,0,int(time.time()*1000)))
    metadata=json.loads((session.directory/'session.json').read_text())
    assert metadata['data_source']=='custom_fetch_unverified' and not metadata.get('forward_plan_id')
    assert summary['journal_sha256']==file_hash(session.journal) and summary['session_hash']==digest(metadata)
    assert summary['termination']=='SAMPLE_LIMIT' and summary['orders_submitted_to_exchange']==0
    with pytest.raises(ValueError,match='only once'):session.run(1)
