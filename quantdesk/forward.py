"""Precommitted target-venue shadow evidence before paper portfolio formation.

This is an operational/economic allocation gate, not capital approval or a claim
of statistical alpha. Local checksums detect edits, not adversarial forgery.
"""
import hashlib
import json
import math
import time
from datetime import datetime,timezone
from pathlib import Path
from .store import digest

DAY=86400000
FORMAT='quantdesk-forward-session-v1'


def bundle_identity(manifest):
    return digest({k:v for k,v in manifest.items() if not k.startswith('_')})


def policy(settings):
    if not 1<=settings.min_forward_days<=90 or not 1<=settings.min_forward_trades<=10000 or not 0.8<=settings.min_forward_coverage<=1 or not 1<=settings.max_forward_age_days<=90:
        raise ValueError('Forward days 1..90, trades 1..10000, coverage 0.8..1')
    return {'version':1,'days':settings.min_forward_days,'minimum_closed_trades':settings.min_forward_trades,
        'minimum_daily_coverage':settings.min_forward_coverage,'maximum_gap_ms':settings.max_gap_ms,
        'maximum_age_days':settings.max_forward_age_days,
        'interpretation':'Minimum shadow coverage/activity and positive aggregate net return for paper allocation; not independent statistical confirmation or live approval'}


def identity(manifest,settings):
    return digest({'bundle':bundle_identity(manifest),'policy':policy(settings)})


def plan_forward(store,manifest,settings,new_window=False,now_ms=None):
    if manifest.get('status')!='PAPER_READY' or manifest.get('spec',{}).get('mode')!='taker':
        raise ValueError('Forward allocation plans require a PAPER_READY taker bundle')
    if manifest['execution']!=settings.execution_dict():
        raise ValueError('Forward settings must exactly match the frozen bundle execution')
    if manifest.get('shadow_execution')!=settings.shadow_dict() or manifest.get('shadow_execution_hash')!=digest(settings.shadow_dict()):
        raise ValueError('Forward plans require a matching frozen displayed-depth shadow policy')
    key=identity(manifest,settings); now=int(time.time()*1000) if now_ms is None else int(now_ms)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('SELECT payload FROM forward_plans JOIN forward_current ON forward_plans.id=forward_current.plan_id WHERE forward_current.identity=?',(key,)).fetchone()
        if old and not new_window:
            return json.loads(old[0])
        start=(now//DAY+1)*DAY
        result={'format':'quantdesk-forward-plan-v1','candidate_id':manifest['id'],'bundle_identity':bundle_identity(manifest),
            'identity':key,'created_ms':now,'start_ms':start,'end_ms':start+policy(settings)['days']*DAY,
            'minimum_book_levels':manifest.get('input_contract',{}).get('minimum_book_levels',1),
            'shadow_execution':manifest['shadow_execution'],
            'dates':[datetime.fromtimestamp((start+i*DAY)/1000,timezone.utc).date().isoformat() for i in range(policy(settings)['days'])],
            'policy':policy(settings),'previous_plan_id':json.loads(old[0])['id'] if old else None,
            'source':'Hyperliquid mainnet BTC public books; standalone qualified strategy; local shadow fills'}
        result['id']=digest(result)
        db.execute('INSERT INTO forward_plans VALUES(?,?,?,?,?)',(result['id'],manifest['id'],key,now/1000,json.dumps(result)))
        db.execute('INSERT OR REPLACE INTO forward_current VALUES(?,?)',(key,result['id']))
    store.event('paper', 'forward_gate','FORWARD_WINDOW_PLANNED',result)
    return result


def file_hash(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            sha.update(chunk)
    return sha.hexdigest()


def journal_days(directory,metadata,summary,plan,now_ms=None):
    """Streaming independent cash/position reconciliation from books and fills."""
    if summary.get('journal_sha256')!=file_hash(directory/'journal.jsonl') or summary.get('session_hash')!=digest(metadata):
        raise ValueError('Forward receipt integrity failed')
    if summary.get('termination')!='SAMPLE_LIMIT' or summary.get('halted'):
        raise ValueError('Forward session did not finish normally')
    if any(summary.get(key)!=0 for key in ('orders_submitted_to_exchange','funding_boundary_crossings','strategy_errors','stale_decisions')):
        raise ValueError('Forward record contains execution, funding or strategy failures')
    if metadata.get('format')!=FORMAT or metadata.get('forward_plan_id')!=plan['id']:
        raise ValueError('Session is not bound to the current precommitted forward window')
    if metadata.get('data_source')!='hyperliquid_public_mainnet' or metadata.get('coin')!='BTC' or metadata.get('exploratory') or metadata.get('portfolio') or len(metadata.get('members',[]))!=1:
        raise ValueError('Require standalone qualified mainnet BTC public observations')
    member=metadata['members'][0]
    if member['id']!=plan['candidate_id'] or member['bundle_identity']!=plan['bundle_identity'] or member['status']!='PAPER_READY' or member.get('weight')!=1.0:
        raise ValueError('Forward strategy identity differs from the frozen bundle')
    if metadata['execution']!=member['execution']:
        raise ValueError('Actual forward execution differs from the researched bundle')
    if metadata.get('shadow_execution')!=plan.get('shadow_execution') or member.get('shadow_execution')!=plan.get('shadow_execution'):
        raise ValueError('Forward shadow policy differs from the frozen plan')
    from .liquidity import DepthBudget,validate_policy
    validate_policy(plan.get('shadow_execution'))
    initial=float(metadata['initial_equity']); cash=initial; quantity=0.0; fees=0.0; trades=0; marks=0; books=0
    if not math.isfinite(initial) or initial<=0 or initial!=metadata['execution']['initial_equity']:
        raise ValueError('Invalid forward starting equity')
    if set(summary.get('open_positions',{}))!={member['id']} or not all(math.isfinite(float(v)) for v in (summary['equity'],summary['fees'],summary['open_positions'][member['id']])):
        raise ValueError('Nonfinite or inconsistent forward final accounting')
    previous=None; quote=None; depth=None; pending=[]; days={}; errors=set(); ended=False
    gap=plan['policy']['maximum_gap_ms']
    now=int(time.time()*1000) if now_ms is None else int(now_ms)
    started=False; run_started=False
    def day(ts):
        return datetime.fromtimestamp(ts/1000,timezone.utc).date().isoformat()
    def ledger(fill):
        nonlocal cash,quantity,fees,trades
        if fill.get('candidate')!=member['id'] or fill.get('maker_proxy'):
            raise ValueError('Unexpected forward fill member or maker proxy')
        side=fill['side']; qty=float(fill['quantity']); price=float(fill['price']); fee=float(fill['fee'])
        if side not in (-1,1) or not all(math.isfinite(v) for v in (qty,price,fee)) or qty<=0 or price<=0 or fee<0:
            raise ValueError('Invalid forward fill')
        if abs(quantity)>1e-12 and (side*quantity>=0 or qty>abs(quantity)+1e-10):
            raise ValueError('Forward exit does not reduce the recorded position')
        requested=abs(quantity) if abs(quantity)>1e-12 else initial*metadata['execution']['allocation_fraction']/quote[1]
        if not math.isfinite(float(fill.get('requested_quantity',float('nan')))) or abs(fill['requested_quantity']-requested)>1e-10:
            raise ValueError('Forward entry/exit sizing differs from the frozen execution model')
        expected=depth.take(side,requested,metadata['execution']['slippage_bps'])
        if abs(qty-expected['quantity'])>1e-10 or fill.get('model')!=expected['model'] or fill.get('levels')!=expected['levels'] or abs(float(fill.get('unfilled_quantity',float('nan')))-expected['unfilled_quantity'])>1e-10 or not math.isfinite(float(fill.get('unfilled_quantity',float('nan')))):
            raise ValueError('Forward fill exceeds or differs from observed displayed depth')
        expected_price=expected['price']
        expected_fee=qty*price*metadata['execution']['taker_fee_bps']/10000
        if abs(price-expected_price)>max(1e-8,price*1e-10) or abs(fee-expected_fee)>1e-8:
            raise ValueError('Forward fill price or fees differ from the frozen execution model')
        old=quantity; cash-=side*qty*price+fee; quantity+=side*qty; fees+=fee
        if abs(old)>1e-12 and abs(quantity)<1e-12:
            trades+=1
            d=days.setdefault(day(fill['ts_ms']),{'closed_trades':0,'coverage_ms':0})
            d['closed_trades']+=1
    with (directory/'journal.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line); kind=event.get('kind')
            received=int(event['received_at_ms'])
            if received>now or received<metadata['created_ms']:
                raise ValueError('Forward event lies outside the recorded observation time')
            if ended:
                raise ValueError('Journal contains events after SESSION_END')
            if kind=='SESSION_START':
                if started or float(event['initial_equity'])!=initial or event.get('signing_enabled') or event.get('exploratory'):
                    raise ValueError('Invalid forward session start')
                started=True
            elif kind=='SESSION_RUN_START':
                if not started or run_started or event['session_hash']!=digest(metadata) or metadata['run_started_ms']<plan['created_ms']:
                    raise ValueError('Forward run was not bound before observations')
                run_started=True
            elif kind=='BOOK':
                book=event['book']; ts=int(book['time'])
                if not run_started or received<metadata['run_started_ms'] or book.get('coin')!='BTC' or ts!=event['exchange_ts'] or abs(received-ts)>gap:
                    raise ValueError('Stale or incompatible forward book')
                if len(book.get('levels',[]))!=2 or min(map(len,book['levels']))<plan['minimum_book_levels']:
                    raise ValueError('Forward book depth does not satisfy the researched input contract')
                bid=float(book['levels'][0][0]['px']); ask=float(book['levels'][1][0]['px'])
                if not 0<bid<ask or not math.isfinite(bid+ask) or (quote and ts<=quote[0]):
                    raise ValueError('Invalid or nonmonotonic forward book')
                quote=(ts,(bid+ask)/2,bid,ask); books+=1
                depth=DepthBudget(book,plan['shadow_execution'])
                for fill in pending:
                    if fill['ts_ms']!=ts:
                        raise ValueError('Fill has no matching observed book')
                    ledger(fill)
                pending=[]
            elif kind=='FILL':
                if quote and quote[0]==event['ts_ms']:
                    ledger(event)
                else:
                    pending.append(event)
            elif kind=='MARK':
                ts=int(event['exchange_ts']); value=float(event['equity'])
                if not quote or quote[0]!=ts or previous and ts<=previous[0] or pending:
                    raise ValueError('Mark has no unique chronological observed book')
                if not math.isfinite(value) or value<=0 or abs(value-(cash+quantity*quote[1]))>max(1e-6,initial*1e-8):
                    raise ValueError('Forward equity does not reconcile to recorded fills/books')
                if event.get('halted'):
                    errors.add(day(ts))
                d=days.setdefault(day(ts),{'closed_trades':0,'coverage_ms':0})
                if 'first_ms' not in d:
                    d.update(first_ms=ts,start_equity=previous[1] if previous and ts-previous[0]<=gap else value)
                d.update(last_ms=ts,end_equity=value)
                if previous and ts-previous[0]<=gap:
                    cursor=previous[0]
                    while cursor<ts:
                        end=min(ts,(cursor//DAY+1)*DAY)
                        part=days.setdefault(day(cursor),{'closed_trades':0,'coverage_ms':0})
                        part['coverage_ms']+=end-cursor; cursor=end
                previous=(ts,value); marks+=1
            elif kind in {'STRATEGY_ERROR','STALE_DECISION','INPUT_CONTRACT_ERROR','INVALID_BOOK','STALE_BOOK','HISTORY_LIMIT','UNMODELED_FUNDING_BOUNDARY','DATA_GAP','FEED_ERROR','INTERRUPTED'}:
                errors.add(day(int(event.get('exchange_ts',event.get('ts_ms',event['received_at_ms'])))))
            elif kind=='SESSION_END':
                if event.get('summary')!={k:v for k,v in summary.items() if k!='journal_sha256'}:
                    raise ValueError('Session end and final receipt disagree')
                ended=True
    if not ended or pending or not previous or marks!=summary['samples'] or marks!=books:
        raise ValueError('Incomplete forward journal or inconsistent sample counts')
    if abs(previous[1]-summary['equity'])>max(1e-6,initial*1e-8) or abs(fees-summary['fees'])>1e-6 or trades!=summary['closed_trades']:
        raise ValueError('Forward final accounting summary does not reconcile')
    if abs(summary.get('open_positions',{}).get(member['id'],float('inf'))-quantity)>1e-10:
        raise ValueError('Forward open exposure does not reconcile')
    result=[]
    for date in plan['dates']:
        d=days.get(date,{})
        start=int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()*1000)
        if date in errors or not d.get('first_ms') or d['first_ms']-start>gap or start+DAY-d['last_ms']>gap or d['coverage_ms']/DAY<plan['policy']['minimum_daily_coverage']:
            continue
        result.append({'date':date,'return_fraction':d['end_equity']/d['start_equity']-1,
            'closed_trades':d['closed_trades'],'coverage_fraction':d['coverage_ms']/DAY,
            'session':str(directory),'journal_sha256':summary['journal_sha256']})
    return result


def assess_forward(store,manifest,settings,now_ms=None):
    key=identity(manifest,settings)
    with store.connect() as db:
        row=db.execute('SELECT payload FROM forward_plans JOIN forward_current ON forward_plans.id=forward_current.plan_id WHERE forward_current.identity=?',(key,)).fetchone()
        count=db.execute('SELECT COUNT(*) FROM forward_plans WHERE candidate=?',(manifest['id'],)).fetchone()[0]
    result={'candidate_id':manifest['id'],'status':'FORWARD_INCOMPLETE','eligible':False,'reasons':[],
        'policy':policy(settings),'planned_windows':count,'days':[],
        'interpretation':'Precommitted shadow evidence for paper portfolio formation only; no live capital approval or simulator certification'}
    if manifest.get('status')!='PAPER_READY' or manifest.get('spec',{}).get('mode')!='taker' or manifest.get('execution')!=settings.execution_dict():
        result['reasons']=['Require PAPER_READY taker status and exactly matching execution settings']; return result
    if manifest.get('shadow_execution')!=settings.shadow_dict() or manifest.get('shadow_execution_hash')!=digest(settings.shadow_dict()):
        result['reasons']=['Require a matching frozen displayed-depth shadow policy; legacy quote-only records remain observations']; return result
    if not row:
        result['reasons']=['No precommitted forward window for this exact bundle and policy']; return result
    plan=json.loads(row[0]); result['plan']=plan
    if digest({k:v for k,v in plan.items() if k!='id'})!=plan['id']:
        raise ValueError('Forward plan integrity failed')
    now=int(time.time()*1000) if now_ms is None else now_ms
    if now<plan['end_ms']:
        result['reasons'].append('Frozen future observation window is not complete')
    elif now-plan['end_ms']>plan['policy']['maximum_age_days']*DAY:
        result['reasons'].append('Forward acceptance is stale; explicitly precommit a new future window')
    invalid=[]; observed={}; unclassified=[]
    for path in sorted((store.home/'paper').glob('*/session.json')):
        try:
            metadata=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(metadata,dict):
                raise ValueError('Invalid paper session metadata')
        except (ValueError,OSError) as error:
            diagnostic={'session':str(path.parent),'reason':str(error)}
            try:
                binding=json.loads((path.parent/'summary.json').read_text(encoding='utf-8')).get('forward_plan_id')
            except (ValueError,OSError,AttributeError):
                binding=None
            (invalid if binding==plan['id'] else unclassified).append(diagnostic)
            continue
        if metadata.get('forward_plan_id')!=plan['id']:
            continue
        if metadata.get('run_started_ms',0)>=plan['end_ms']:
            continue
        directory=path.parent
        if not (directory/'summary.json').is_file():
            invalid.append({'session':str(directory),'reason':'No completed session receipt'}); continue
        try:
            summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            for d in journal_days(directory,metadata,summary,plan,now):
                observed.setdefault(d['date'],[]).append(d)
        except (ValueError,KeyError,TypeError,OSError,OverflowError) as error:
            invalid.append({'session':str(directory),'reason':str(error)})
    result['invalid_sessions']=invalid
    result['unclassified_sessions']=unclassified
    duplicates=[date for date,days in observed.items() if len(days)>1]
    if duplicates:
        result['reasons'].append('Overlapping full-date sessions cannot be cherry-picked: '+', '.join(duplicates))
    result['days']=[observed[date][0] for date in plan['dates'] if len(observed.get(date,[]))==1]
    if len(result['days'])!=len(plan['dates']):
        result['reasons'].append('Missing clean complete coverage for every frozen date')
    result['closed_trades']=sum(d['closed_trades'] for d in result['days'])
    result['compound_net_return']=math.prod(1+d['return_fraction'] for d in result['days'])-1
    if result['closed_trades']<plan['policy']['minimum_closed_trades']:
        result['reasons'].append('Insufficient observed closed-trade activity')
    if result['compound_net_return']<=0:
        result['reasons'].append('Frozen forward window did not produce a positive net return')
    result['eligible']=not result['reasons'] and not invalid
    result['status']='FORWARD_ALLOCATION_ELIGIBLE' if result['eligible'] else 'FORWARD_INCOMPLETE'
    return result
