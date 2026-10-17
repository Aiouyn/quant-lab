"""Displayed-depth monetary and capacity invariants, not empirical fill calibration."""
import json
from dataclasses import replace
import pytest
from quantdesk.execution import Broker
from quantdesk.liquidity import DepthBudget
from quantdesk.settings import Settings
from quantdesk.store import digest
from quantdesk.paper import PaperSession,load_bundle
from tests.test_quantdesk import spec,CODE
from tests.test_forward_gate import setup,receipt,save
from quantdesk.forward import assess_forward,plan_forward


def raw(ts,bids=((99.99,10),),asks=((100.01,10),)):
    return {'coin':'BTC','time':ts,'levels':[[{'px':str(p),'sz':str(q),'n':1} for p,q in side] for side in (bids,asks)]}


def quote(ts,policy,**kwargs):
    book=raw(ts,**kwargs); bid=float(book['levels'][0][0]['px']); ask=float(book['levels'][1][0]['px'])
    return {'ts_ms':ts,'bid_p0':bid,'ask_p0':ask,'mid_price':(bid+ask)/2,
        'spread_bps':(ask-bid)/((bid+ask)/2)*10000,'_liquidity':DepthBudget(book,policy)}


def test_walks_prices_caps_participation_and_never_extrapolates_tail():
    policy=Settings().shadow_dict()
    budget=DepthBudget(raw(0,asks=((100.01,1),(100.02,2),(101,50))),policy)
    fill=budget.take(1,2,0.5)
    assert fill['quantity']==pytest.approx(0.3)
    assert fill['unfilled_quantity']==pytest.approx(1.7)
    assert fill['price']==pytest.approx((0.1*100.01+0.2*100.02)/0.3*1.00005)
    assert len(fill['levels'])==2
    assert budget.take(1,2,0.5)['quantity']==0


@pytest.mark.parametrize('levels',[((100.01,1),(100.00,1)),((100.01,1),(100.01,2)),((100.01,0),),((float('nan'),1),)])
def test_bad_depth_cannot_fall_back_to_unlimited_quotes(levels):
    with pytest.raises(ValueError):DepthBudget(raw(0,asks=levels),Settings().shadow_dict())


def test_partial_exits_preserve_residual_and_only_close_trade_when_flat():
    settings=Settings(); policy=settings.shadow_dict(); broker=Broker(spec(holding_seconds=1),settings.execution_dict(),policy)
    first=quote(1000,policy,asks=((100.01,10),))
    filled=broker.fill(first,1,2.0)
    assert filled==1 and broker.account.quantity==1
    broker.fill(quote(2000,policy,bids=((100.99,3),),asks=((101.01,10),)),-1,1)
    assert broker.account.quantity==pytest.approx(0.7) and not broker.trades
    assert broker.closing
    broker.fill(quote(3000,policy,bids=((101.99,10),),asks=((102.01,10),)),-1,0.7)
    assert broker.account.quantity==0 and len(broker.trades)==1 and not broker.closing
    trade=broker.trades[0]
    assert trade['quantity']==1
    assert trade['exit_price']==pytest.approx((0.3*100.99+0.7*101.99)*0.99995)
    assert broker.account.realized==pytest.approx(trade['pnl'])
    assert broker.account.cash-settings.initial_equity==pytest.approx(trade['pnl'])
    assert trade['gross_mid_pnl']-trade['execution_drag']-trade['fees']==pytest.approx(trade['pnl'])


def test_ioc_unfilled_entry_is_cancelled_and_exit_retries_with_latency():
    settings=Settings(); policy=settings.shadow_dict(); broker=Broker(spec(holding_seconds=1),settings.execution_dict(),policy)
    broker.step(quote(0,policy),1)
    broker.step(quote(1000,policy),0)
    assert broker.account.quantity==1 and broker.pending is None
    broker.step(quote(2000,policy),0)
    broker.step(quote(3000,policy,bids=((99.99,2),)),0)
    assert broker.account.quantity==pytest.approx(0.8)
    assert broker.pending['kind']=='exit' and broker.pending['arrival']==4000
    broker.step(quote(3500,policy),0)
    assert broker.account.quantity==pytest.approx(0.8)
    broker.step(quote(4000,policy),0)
    assert broker.account.quantity==0 and len(broker.trades)==1


def test_terminal_thin_book_retains_open_exposure_instead_of_fabricated_close():
    settings=Settings(); policy=settings.shadow_dict(); broker=Broker(spec(),settings.execution_dict(),policy)
    broker.fill(quote(0,policy),1,1)
    broker.step(quote(1000,policy,bids=((99.99,1),)),0,terminal=True)
    assert broker.account.quantity==pytest.approx(0.9) and not broker.trades
    assert broker.mark(quote(2000,policy))==pytest.approx(broker.account.cash+90)


def test_shared_book_does_not_allocate_same_depth_to_two_sleeves():
    settings=Settings(); policy=settings.shadow_dict(); book=quote(1000,policy,asks=((100.01,1),(100.02,2)))
    a=Broker(spec(),settings.execution_dict(),policy); b=Broker(spec(),settings.execution_dict(),policy)
    assert a.fill(book,1,0.2)==pytest.approx(0.2)
    assert b.fill(book,1,0.2)==pytest.approx(0.1)
    assert a.account.quantity+b.account.quantity==pytest.approx(0.3)


def test_short_partial_exits_have_correct_cash_fees_and_weighted_prices():
    settings=Settings(); policy=settings.shadow_dict(); broker=Broker(spec(),settings.execution_dict(),policy)
    broker.fill(quote(0,policy),-1,1)
    broker.fill(quote(1000,policy,bids=((98.99,10),),asks=((99.01,4),)),1,1)
    assert broker.account.quantity==pytest.approx(-0.6) and not broker.trades
    broker.fill(quote(2000,policy,bids=((97.99,10),),asks=((98.01,10),)),1,0.6)
    trade=broker.trades[0]
    assert trade['side']==-1 and trade['quantity']==1 and broker.account.quantity==0
    assert trade['exit_price']==pytest.approx((0.4*99.01+0.6*98.01)*1.00005)
    assert broker.account.realized==pytest.approx(trade['pnl'])
    assert broker.account.cash-settings.initial_equity==pytest.approx(trade['pnl'])
    assert trade['gross_mid_pnl']-trade['execution_drag']-trade['fees']==pytest.approx(trade['pnl'])


def test_paper_portfolio_uses_one_capacity_budget_even_during_resets(tmp_path,monkeypatch):
    import quantdesk.paper as paper
    settings=replace(Settings(),home=tmp_path)
    members=[]
    for ident in ('a','b'):
        directory=tmp_path/ident; directory.mkdir()
        strategy=spec(); policy=settings.shadow_dict()
        manifest={'format':'quantdesk-v2','id':ident,'title':ident,'status':'PAPER_READY','spec':strategy,
            'spec_hash':digest(strategy),'code_hash':digest(CODE),'execution':settings.execution_dict(),
            'execution_hash':digest(settings.execution_dict()),'shadow_execution':policy,'shadow_execution_hash':digest(policy)}
        (directory/'manifest.json').write_text(json.dumps(manifest)); (directory/'strategy.py').write_text(CODE)
        members.append({'candidate_id':ident,'weight':0.35,'bundle':str(directory)})
    directory=tmp_path/'portfolio'; directory.mkdir()
    manifest={'format':'quantdesk-portfolio-v2','members':members}; manifest['manifest_hash']=digest(manifest)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(paper,'run_signal',lambda code,frame,*args,**kw:[1.0]*len(frame))
    session=PaperSession(directory,tmp_path,settings)
    session.consume(raw(1000,asks=((100.01,1),)),now_ms=1000)
    session.consume(raw(2100,asks=((100.01,1),)),now_ms=2100)
    assert sum(m['broker'].account.quantity for m in session.members)==pytest.approx(0.1)
    # UTC day reset liquidations and subsequent broker steps share the book.
    session.consume(raw(86400000,bids=((99.99,0.1),),asks=((100.01,1),)),now_ms=86400000)
    assert sum(m['broker'].account.quantity for m in session.members)==pytest.approx(0.09)


def test_legacy_bundle_cannot_grant_new_forward_acceptance(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    del manifest['shadow_execution']; del manifest['shadow_execution_hash']
    with pytest.raises(ValueError,match='displayed-depth'):plan_forward(store,manifest,settings)
    assert not assess_forward(store,manifest,settings,plan['end_ms']+1)['eligible']


def test_receipt_overfill_is_rejected_even_after_rehash(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,events=receipt(store,manifest,settings,plan)
    first=next(e for e in events if e['kind']=='FILL')
    book=next(e for e in events if e['kind']=='BOOK' and e['exchange_ts']==first['ts_ms'])
    book['book']['levels'][1][0]['sz']='1'
    save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert not report['eligible'] and 'displayed depth' in report['invalid_sessions'][0]['reason']


def test_policy_edit_fails_bundle_hash_and_session_mismatch(tmp_path):
    store,_,manifest,settings,_=setup(tmp_path)
    directory=store.home/'bundles'/manifest['id']
    manifest['shadow_execution']['participation_fraction']=0.5
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='liquidity integrity'):load_bundle(directory)
    manifest['shadow_execution_hash']=digest(manifest['shadow_execution'])
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='shadow settings'):PaperSession(directory,tmp_path,settings)


def test_forward_reconciles_partial_exits_as_two_complete_trades(tmp_path):
    store,_,manifest,settings,plan=setup(tmp_path)
    directory,metadata,summary,old_events=receipt(store,manifest,settings,plan,active=False)
    broker=Broker(spec(),settings.execution_dict(),settings.shadow_dict())
    events=old_events[:2]
    for minute,old in enumerate(e for e in old_events if e['kind']=='BOOK'):
        ts=old['exchange_ts']; mid=101 if minute in (2,3,5,6) else 100
        bid_size=30 if minute in (2,5) else 70 if minute in (3,6) else 1000
        ask_size=100 if minute in (1,4) else 1000
        book=raw(ts,bids=((mid-0.01,bid_size),),asks=((mid+0.01,ask_size),))
        events.append({'kind':'BOOK','exchange_ts':ts,'received_at_ms':ts,'book':book})
        row={'ts_ms':ts,'mid_price':mid,'bid_p0':mid-0.01,'ask_p0':mid+0.01,
            '_liquidity':DepthBudget(book,settings.shadow_dict())}
        if minute in (1,2,3,4,5,6):
            side=1 if minute in (1,4) else -1
            requested=2500/mid if side>0 else abs(broker.account.quantity)
            broker.fill(row,side,requested)
            events.append(dict(broker.events[-1],candidate=manifest['id'],received_at_ms=ts))
        events.append({'kind':'MARK','exchange_ts':ts,'received_at_ms':ts,'equity':broker.mark(row),'halted':False})
    summary.update(equity=broker.mark(row),fees=broker.account.fees,closed_trades=len(broker.trades),
        open_positions={manifest['id']:broker.account.quantity})
    events.append({'kind':'SESSION_END','summary':summary.copy(),'received_at_ms':ts})
    save(directory,metadata,summary,events)
    report=assess_forward(store,manifest,settings,plan['end_ms']+1)
    assert len(broker.trades)==2 and report['eligible'] and report['closed_trades']==2
    assert report['compound_net_return']>0
