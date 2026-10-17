"""Retrospective public-capture execution sensitivity, never forward approval."""
import hashlib
import json
import re
import uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import polars as pl
from . import venue_study
from .data import Book,enrich,segment_coverage
from .execution import Broker
from .input_contract import input_contract,program_input
from .liquidity import DepthBudget
from .models import verify_artifact,fitted_signal,prediction_frontier
from .paper import normalize_book
from .store import digest
from .strategy import validate_spec,run_signal,run_feature_program

MAX_BOOKS=200000


def load(store,replay_id):
    if not isinstance(replay_id,str) or not re.fullmatch('[a-f0-9]{64}',replay_id): raise ValueError('Use an exact venue replay ID')
    root=(store.home/'venue_replays').resolve();directory=(root/replay_id).resolve()
    if not directory.is_relative_to(root): raise ValueError('Replay path escapes evidence home')
    bound=json.loads((directory/'report.json').read_text())
    if digest(bound['report'])!=replay_id: raise ValueError('Venue replay integrity failed')
    checks=bound['report']['artifact_sha256']
    if set(checks)!={'quote.events.jsonl','quote.trades.jsonl','depth.events.jsonl','depth.trades.jsonl'}:
        raise ValueError('Unexpected replay journal set')
    for name,checksum in checks.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:
            raise ValueError('Cached replay journal integrity failed')
    return {'replay_id':replay_id,'directory':str(directory),'report':bound['report']}


def summaries(store,cid):
    with store.connect() as db:
        records=db.execute("SELECT payload FROM memories WHERE campaign=? AND kind='VENUE_REPLAY' ORDER BY id DESC LIMIT 40",(cid,)).fetchall()
    result=[];seen=set()
    for record in records:
        try:
            payload=json.loads(record[0]);report=load(store,payload['replay_id'])['report']
            key=(report['candidate_id'],report['capture_id'],report['model_hash'])
            if key in seen: continue
            seen.add(key)
            result.append({'replay_id':payload['replay_id'],**{k:report[k] for k in ['candidate_id','capture_id','study_id','model_key','rows','quote_arm','depth_arm','active_signals','forward_eligible','strategy_promoted']},
                'interpretation':'Exact retrospective source and artifacts; alternative fills and versions are not independent evidence. Read replay ID for diagnostics and assumptions.'})
        except (ValueError,KeyError,TypeError,OSError): continue
        if len(result)>=6: break
    return result


def implementation():
    names=('venue_replay.py','venue_study.py','data.py','paper.py','execution.py','liquidity.py','models.py','strategy.py','input_contract.py','worker.py','learners.py','store.py')
    return {'sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
            'numpy_version':np.__version__,'polars_version':pl.__version__}


def frozen_model(store,candidate,settings,key):
    if 'model' not in candidate['spec']:
        if key is not None: raise ValueError('Formula candidate cannot select a fitted model')
        return None
    if not isinstance(key,str) or not re.fullmatch('[a-f0-9]{64}',key):
        raise ValueError('Fitted replay needs the exact stored model filename key')
    root=(store.home/'models').resolve();path=(root/(key+'.json')).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size>4000000:
        raise ValueError('Unknown or oversized frozen model')
    artifact=verify_artifact(json.loads(path.read_text()))
    if artifact['code_hash']!=digest(candidate['code']) or artifact['spec_hash']!=digest(candidate['spec']) or artifact['recipe']!=candidate['spec']['model']:
        raise ValueError('Frozen model belongs to different code/specification')
    if artifact['execution']!=settings.execution_dict(): raise ValueError('Frozen model execution differs from replay')
    return artifact


def list_models(store,candidate_id,settings):
    candidate=store.get_candidate(candidate_id)
    if 'model' not in candidate['spec']: return []
    result=[]
    for path in sorted((store.home/'models').glob('*.json'))[:1000]:
        try: artifact=frozen_model(store,candidate,settings,path.stem)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError): continue
        result.append({'model_key':path.stem,'model_hash':artifact['artifact_hash'],
            'prediction_cutoff_ms':artifact['provenance']['prediction_cutoff_ms'],
            'training_dates':artifact['provenance'].get('training_dates',[]),
            'interpretation':'Exact existing frozen artifact; no refitting or new outcome evidence'})
    return sorted(result,key=lambda v:v['prediction_cutoff_ms'],reverse=True)[:10]


def capture_frame(store,report,settings,control=None):
    directory=venue_study.capture_directory(store.home,report['source']['capture_id'])
    if venue_study.source(directory,control)!=report['source']:
        raise ValueError('Capture no longer matches the exact study source')
    measured=venue_study.Measurements(dict(report['parameters'],notionals_usd=[]))
    rows=[];books=[];builder=None;segment=-1;scanned=0
    for item in report['source']['files']:
        hashed=hashlib.sha256();file=directory/item['file']
        with file.open('rb') as stream:
            while True:
                line=stream.readline(512001)
                if not line: break
                scanned+=1
                if len(line)>512000 or scanned>venue_study.MAX_RECORDS: raise OverflowError('Capture replay record limit')
                hashed.update(line)
                if control and scanned%1000==1: control()
                previous=measured.counts['valid_unique_books']
                try:
                    event=json.loads(line)
                    if not isinstance(event,dict): raise ValueError('Invalid capture object')
                    measured.consume(event)
                except (ValueError,TypeError,KeyError):
                    measured.reset();continue
                if measured.counts['valid_unique_books']==previous: continue
                raw=event['message']['data']
                if measured.segment_start==int(raw['time']):
                    segment+=1;builder=Book(settings.max_gap_ms)
                row=builder.apply(normalize_book(raw))
                if row is None: raise ValueError('Validated raw book failed reconstruction')
                row['segment']=segment;rows.append(row);books.append(raw)
                if len(rows)>MAX_BOOKS: raise OverflowError('Capture replay book limit')
        if hashed.hexdigest()!=item['sha256']: raise ValueError('Capture changed during replay reconstruction')
    if venue_study.source(directory,control)!=report['source']: raise ValueError('Capture changed after reconstruction')
    if not rows: raise ValueError('Capture has no usable books')
    return enrich(pl.DataFrame(rows)),books


def broker_arm(frame,signals,books,spec,settings,depth,control=None):
    broker=Broker(spec,settings.execution_dict(),settings.shadow_dict() if depth else None)
    last_segment=None;warm=0;events=[]
    for i,row in enumerate(frame.iter_rows(named=True)):
        if control and i%1000==0: control()
        if depth: row=dict(row,_liquidity=DepthBudget(books[i],settings.shadow_dict()))
        if last_segment!=row['segment']:
            if last_segment is not None: broker.step(row,0.,terminal=True)
            last_segment=row['segment'];warm=0
        warm+=1
        signal=float(signals[i]) if warm>spec['warmup_rows'] else 0.
        broker.step(row,signal,terminal=i==len(frame)-1)
        events.extend(broker.events);broker.events=[]
    fills=[e for e in events if e['kind']=='FILL']
    # Separate cash reconstruction from the broker's own account state.
    cash=settings.initial_equity;quantity=0.;fees=0.
    for fill in fills:
        cash-=fill['side']*fill['quantity']*fill['price']+fill['fee']
        quantity+=fill['side']*fill['quantity'];fees+=fill['fee']
    if not np.allclose([cash,quantity,fees],[broker.account.cash,broker.account.quantity,broker.account.fees],rtol=1e-10,atol=1e-8):
        raise ValueError('Replay cash/position/fee reconciliation failed')
    summary=broker.summary()
    summary.update(open_quantity=broker.account.quantity,cash=broker.account.cash,
        gross_mid_pnl=sum(t['gross_mid_pnl'] for t in broker.trades),
        execution_drag=sum(t['execution_drag'] for t in broker.trades),
        partial_fills=sum(e.get('unfilled_quantity',0)>1e-10 for e in fills),
        ioc_no_fills=sum(e['kind']=='IOC_NO_FILL' for e in events),
        fill_events=len(fills),cash_position_fee_reconciled=True)
    return summary,events,broker.trades


def replay(store,settings,cid,candidate_id,study_id,model_key=None):
    store.check_research_control(cid)
    if store.get_campaign(cid)['status']=='SEALED': raise ValueError('Sealed campaign cannot run venue research')
    candidate=store.get_candidate(candidate_id)
    if candidate['campaign']!=cid: raise ValueError('Candidate belongs to another campaign')
    spec=validate_spec(candidate['spec'])
    if spec['mode']!='taker': raise ValueError('Public depth cannot certify maker execution')
    report=venue_study.load(store,study_id)
    if report['parameters']['maximum_gap_ms']!=settings.max_gap_ms: raise ValueError('Study and replay gap policies differ')
    levels=input_contract(candidate['code'],spec)['minimum_book_levels']
    if not report['depth_observations'] or any(min(v['bid_levels'],v['ask_levels'])<levels for v in report['depth_observations']):
        raise ValueError('Capture depth cannot supply the candidate input contract')
    first=report['book_window']['first_exchange_ms'];last=report['book_window']['last_exchange_ms']
    if last-first>86400000+settings.max_gap_ms: raise ValueError('Replay requires a bounded one-day capture')
    start=datetime.fromtimestamp(first/1000,timezone.utc).date();end=datetime.fromtimestamp(last/1000,timezone.utc).date()
    possible_dates=[(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
    with store.connect() as db:
        if any(db.execute('SELECT 1 FROM holdout_dates WHERE source_hash=?',(digest({'date':d,'instrument':'BTC-USDT-SWAP'}),)).fetchone() for d in possible_dates):
            raise ValueError('Capture intersects globally reserved holdout dates; no venue research allowed')
    model=frozen_model(store,candidate,settings,model_key)
    if model and model['provenance']['prediction_cutoff_ms']>report['book_window']['first_exchange_ms']:
        raise ValueError('Capture observations precede frozen model prediction cutoff')
    algo=implementation()
    if venue_study.source(venue_study.capture_directory(store.home,report['source']['capture_id']),lambda:store.check_research_control(cid))!=report['source']:
        raise ValueError('Capture no longer matches the exact study source')
    key=digest({'candidate':candidate_id,'code':digest(candidate['code']),'spec':digest(spec),'study':study_id,
                'model':model['artifact_hash'] if model else None,'execution':settings.execution_dict(),
                'shadow':settings.shadow_dict(),'implementation':algo})
    cached=store.cached_tool(key)
    if cached:
        bound=load(store,cached['replay_id'])
        if bound['report']!=cached['report']: raise ValueError('Cached venue replay integrity failed')
        return dict(cached,already_measured=True)
    control=lambda:store.check_research_control(cid)
    frame,books=capture_frame(store,report,settings,control)
    if model and model['provenance']['prediction_cutoff_ms']>int(frame['ts_ms'][0]): raise ValueError('Prediction cutoff violated')
    dates=sorted({datetime.fromtimestamp(int(t)/1000,timezone.utc).date().isoformat() for t in frame['ts_ms']})
    store.expose_dates(cid,[{'date':d} for d in dates],'development')
    store.event(cid,'venue_replay','EXPERIMENT_STARTED',{'candidate':candidate_id,'phase':'venue_development',
        'study_id':study_id,'trials':2,'interpretation':'Same signals, alternative fill assumptions; no two independent samples'})
    runner=run_feature_program if spec.get('model',{}).get('custom_features',False) else run_signal
    alpha=runner(candidate['code'],program_input(frame,spec),settings.worker_timeout)
    signals=fitted_signal(frame,alpha,model) if model else alpha
    quote,qevents,qtrades=broker_arm(frame,signals,books,spec,settings,False,control)
    depth,devents,dtrades=broker_arm(frame,signals,books,spec,settings,True,control)
    if venue_study.source(venue_study.capture_directory(store.home,report['source']['capture_id']),control)!=report['source']:
        raise ValueError('Capture changed before replay commit')
    if implementation()!=algo: raise ValueError('Replay implementation changed before commit')
    if model and frozen_model(store,candidate,settings,model_key)['artifact_hash']!=model['artifact_hash']:
        raise ValueError('Frozen model changed before replay commit')
    artifacts={}
    for arm,events,trades in [('quote',qevents,qtrades),('depth',devents,dtrades)]:
        for kind,values in [('events',events),('trades',trades)]:
            artifacts[arm+'.'+kind+'.jsonl']=''.join(json.dumps(v,allow_nan=False)+'\n' for v in values)
    payload={'format':'quantdesk-venue-replay-v1','candidate_id':candidate_id,'study_id':study_id,
        'capture_id':report['source']['capture_id'],'model_key':model_key,'model_hash':model['artifact_hash'] if model else None,
        'code_hash':digest(candidate['code']),'spec_hash':digest(spec),'implementation':algo,
        'execution':settings.execution_dict(),'shadow_execution':settings.shadow_dict(),
        'rows':len(frame),'dates':dates,'segments':segment_coverage(frame),
        'mainnet':report['mainnet'],'source_state':report['effective_source_state'],
        'quote_arm':quote,'depth_arm':depth,'depth_minus_quote_pnl':depth['pnl']-quote['pnl'],
        'active_signals':int((np.abs(signals)>spec['threshold']).sum()),
        'signal_quantiles':{str(p):float(np.percentile(signals,p)) for p in [0,1,10,50,90,99,100]},
        'prediction_diagnostic':prediction_frontier(frame,alpha,model,spec) if model else None,
        'artifact_sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in artifacts.items()},
        'forward_eligible':False,'strategy_promoted':False,'orders_submitted_to_exchange':0,
        'limits':['Retrospective, already observed public capture; no untouched confirmation or forward acceptance.',
            'Frozen earlier model reused without refitting; venue feature distribution and effective history may differ.',
            'Both arms use the same causal signals; they are fill sensitivity, not two independent experiments.',
            'Displayed participation is assumed; no own-order impact, replenishment or execution calibration.',
            'Configured latency only; historical batch inference does not measure actual routing/inference delay.',
            'Interrupted sources may omit intervals; funding settlement is absent and residual positions remain risk.']}
    ident=digest(payload);directory=store.home/'venue_replays'/ident;directory.mkdir(parents=True,exist_ok=True)
    for name,value in artifacts.items():
        path=directory/name;temp=path.with_suffix('.'+uuid.uuid4().hex+'.partial');temp.write_text(value,encoding='utf-8',newline='');temp.replace(path)
    path=directory/'report.json';temp=path.with_suffix('.partial');temp.write_text(json.dumps({'replay_id':ident,'report':payload},indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
    result={'replay_id':ident,'directory':str(directory),'already_measured':False,'report':payload}
    store.cache_tool(key,result);store.memory(cid,candidate_id,'VENUE_REPLAY',{'replay_id':ident,**payload})
    return result
