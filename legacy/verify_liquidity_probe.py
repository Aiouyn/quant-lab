"""Authored directional probe on actual public BTC books; no strategy/alpha claim."""
import hashlib
import json
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.paper import PaperSession


def audit_probe(directory):
    """Reconcile the exploratory journal without changing its eligibility."""
    from quantdesk.liquidity import DepthBudget
    metadata=json.loads((directory/'session.json').read_text(encoding='utf-8'))
    cash=float(metadata['initial_equity']); position=0.0; fees=0.0; trades=0; fills=0; marks=0; depth=None; mid=None
    for line in (directory/'journal.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event['kind']=='BOOK':
            depth=DepthBudget(event['book'],metadata['shadow_execution'])
            mid=(float(event['book']['levels'][0][0]['px'])+float(event['book']['levels'][1][0]['px']))/2
        elif event['kind']=='FILL':
            assert depth is not None and depth.ts_ms==event['ts_ms']
            requested=abs(position) if abs(position)>1e-12 else metadata['initial_equity']*metadata['execution']['allocation_fraction']/mid
            assert abs(event['requested_quantity']-requested)<1e-10
            if abs(position)>1e-12:assert event['side']*position<0
            expected=depth.take(event['side'],requested,metadata['execution']['slippage_bps'])
            assert expected['levels']==event['levels'] and abs(expected['quantity']-event['quantity'])<1e-10
            assert abs(expected['price']-event['price'])<1e-8
            fee=event['quantity']*event['price']*metadata['execution']['taker_fee_bps']/10000
            assert abs(fee-event['fee'])<1e-8
            old=position; cash-=event['side']*event['quantity']*event['price']+fee
            position+=event['side']*event['quantity']; fees+=fee; fills+=1
            trades+=int(abs(old)>1e-12 and abs(position)<1e-12)
        elif event['kind']=='MARK':
            assert depth.ts_ms==event['exchange_ts']
            assert abs(event['equity']-(cash+position*mid))<1e-6
            marks+=1
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert abs(summary['equity']-(cash+position*mid))<1e-6 and abs(summary['fees']-fees)<1e-6
    assert summary['closed_trades']==trades and summary['samples']==marks
    assert abs(summary['open_positions'][metadata['members'][0]['id']]-position)<1e-10
    return {'fills_reconciled':fills,'marks_reconciled':marks,'closed_trades':trades,
        'cash':cash,'quantity':position,'fees':fees,'equity':cash+position*mid,
        'interpretation':'Exploratory journal reconciled independently of Broker account state; no forward eligibility or calibration claim'}


def main():
    root=Path(__file__).resolve().parent
    directory=root/'results/desk/verification/liquidity'
    directory.mkdir(parents=True,exist_ok=True)
    # This isolated verification home cannot promote or mutate production candidates.
    settings=replace(Settings(),home=directory/'public-probe-home')
    store=Store(settings.home)
    code="import polars as pl\ndef compute_signal(df):\n    return pl.Series('signal', [1.0] * len(df))\n"
    spec={'title':'AUTHORED EXECUTION PROBE - no alpha claim','rationale':'Exercise local displayed-depth accounting on fresh public books',
        'falsification':'Not research: probe failures reject execution infrastructure only',
        'holding_seconds':2.0,'threshold':0.2,'warmup_rows':0,'mode':'taker','input_features':['qi_1']}
    manifest={'format':'quantdesk-v3','id':'authored-liquidity-probe','title':spec['title'],'spec':spec,
        'spec_hash':digest(spec),'code_hash':digest(code),'status':'EXPLORATORY',
        'execution':settings.execution_dict(),'execution_hash':digest(settings.execution_dict()),
        'shadow_execution':settings.shadow_dict(),'shadow_execution_hash':digest(settings.shadow_dict())}
    from quantdesk.input_contract import input_contract
    manifest['input_contract']=input_contract(code,spec)
    bundle=settings.home/'bundles'/manifest['id']; bundle.mkdir(parents=True,exist_ok=True)
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (bundle/'strategy.py').write_text(code,encoding='utf-8')
    with store.connect() as db:
        requests_before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    session=PaperSession(bundle,settings.home,settings,exploratory=True)
    summary=session.run(samples=12)
    events=[json.loads(line) for line in session.journal.read_text(encoding='utf-8').splitlines()]
    fills=[e for e in events if e['kind']=='FILL']
    books=[e for e in events if e['kind']=='BOOK']
    assert summary['orders_submitted_to_exchange']==0
    assert all(e['model']=='displayed-depth-ioc-v1' for e in fills)
    assert all(e['quantity']<=e['requested_quantity']+1e-12 for e in fills)
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==requests_before
    from quantdesk.forward import file_hash,bundle_identity,assess_forward
    metadata=json.loads((session.directory/'session.json').read_text(encoding='utf-8'))
    assert summary['journal_sha256']==file_hash(session.journal) and summary['session_hash']==digest(metadata)
    accounting=audit_probe(session.directory)
    assessment=assess_forward(store,manifest,settings)
    assert not assessment['eligible']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'probe':'Authored constant-long execution probe, not a discovered or qualified strategy',
        'summary':summary,'book_count':len(books),'fill_count':len(fills),
        'partial_fill_count':sum(e['unfilled_quantity']>1e-12 for e in fills),
        'maximum_levels_consumed':max((len(e['levels']) for e in fills),default=0),
        'shadow_policy':settings.shadow_dict(),'bundle_identity':bundle_identity(manifest),
        'metadata_matches_receipt':True,'journal_matches_receipt':True,
        'accounting_reconciliation':accounting,
        'forward_assessment':assessment,'new_provider_requests':0,'new_holdout_exposures':0,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'quantdesk').glob('*.py')},
        'interpretation':'Real public data and actual local model/worker integration. Positive probe PnL, if any, is directional chance; this is not alpha, fill calibration, portfolio acceptance, or capital readiness.'}
    path=directory/'public-probe.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'receipt':str(path),'summary':summary,'fills':len(fills),'partial_fills':result['partial_fill_count'],
        'maximum_levels_consumed':result['maximum_levels_consumed'],'forward_eligible':False,'new_provider_requests':0},indent=2))


if __name__=='__main__':main()
