"""Immutable studies of public captures; no PnL, own fills or promotion.

Quotes and prints are observations. Displayed-depth sweeps are explicitly
hypothetical under frozen assumptions, not calibrated routing or capacity.
"""
import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path
import numpy as np
from .liquidity import DepthBudget
from .store import digest,process_alive

VERSION=1
MAX_BYTES=512*1024*1024
MAX_RECORDS=2000000
MAX_TRADES=500000
MAX_BOOKS=500000


def algorithm_manifest():
    return {'sources_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ('venue_study.py','liquidity.py')},'numpy_version':np.__version__}


def statistics(values):
    if not values: return {'observations':0}
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all(): raise ValueError('Study measurements must be finite')
    return {'observations':len(a),'minimum':float(a.min()),'mean':float(a.mean()),
        **{'p'+str(q):float(np.percentile(a,q)) for q in (10,50,90,95,99)},'maximum':float(a.max())}


def timestamp(value):
    if isinstance(value,bool) or not isinstance(value,(int,str)): raise ValueError('Invalid timestamp')
    result=int(value)
    if result<0: raise ValueError('Invalid timestamp')
    return result


def number(value,positive=False):
    if isinstance(value,bool): raise ValueError('Invalid numeric field')
    result=float(value)
    if not math.isfinite(result) or positive and result<=0: raise ValueError('Invalid numeric field')
    return result


def capture_directory(home,capture_id):
    if not isinstance(capture_id,str) or not re.fullmatch(r'hyperliquid-\d{8}-\d{6}-[a-f0-9]{6}',capture_id):
        raise ValueError('Use an exact public capture ID')
    root=(Path(home)/'market_data').resolve(); directory=(root/capture_id).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir(): raise ValueError('Unknown public capture')
    return directory


def inventory(home,limit=8):
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=10: raise ValueError('Capture limit must be 1..10')
    rows=[]; root=(Path(home)/'market_data').resolve()
    for directory in sorted(root.glob('hyperliquid-*'),reverse=True):
        if not directory.is_dir() or not directory.resolve().is_relative_to(root): continue
        status_path=directory/'status.json'
        try: state=json.loads(status_path.read_text()) if status_path.exists() else {}
        except (OSError,ValueError): state={}
        if not isinstance(state,dict): state={}
        pid=state.get('pid')
        known_pid=isinstance(pid,int) and not isinstance(pid,bool) and 0<pid<2**32
        live=process_alive(pid) if known_pid else None
        files=[f for f in directory.glob('????-??-??.jsonl') if f.resolve().is_relative_to(directory.resolve())]
        effective=state.get('status','UNKNOWN')
        if effective=='RUNNING': effective='RUNNING' if live else 'INTERRUPTED' if live is False else 'UNVERIFIED_LIVENESS'
        rows.append({'capture_id':directory.name,'declared_status':state.get('status','UNKNOWN'),
            'process_alive':live,'effective_status':effective,
            'declared_book_mode':state.get('book_mode'),'raw_files':len(files),
            'raw_bytes':sum(f.stat().st_size for f in files),
            'interpretation':'Inventory only; reported counts/coverage require an actual immutable study'})
        if len(rows)>=limit: break
    return rows


def source(directory,control):
    state_path=directory/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if not isinstance(state,dict): raise ValueError('Capture status must be an object')
    pid=state.get('pid')
    if pid is not None and (isinstance(pid,bool) or not isinstance(pid,int) or not 0<pid<2**32): raise ValueError('Invalid reported capture process ID')
    if pid and process_alive(pid): raise ValueError('Capture process is live; study a frozen stopped capture')
    files=sorted(directory.glob('????-??-??.jsonl'))
    if not files or sum(f.stat().st_size for f in files)>MAX_BYTES: raise ValueError('Capture has no records or exceeds the study byte limit')
    manifests=[]
    for file in files:
        if not file.resolve().is_relative_to(directory): raise ValueError('Capture file escapes its directory')
        before=file.stat(); hashed=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):
                if control: control()
                hashed.update(block)
        after=file.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('Capture changed during fingerprinting')
        manifests.append({'file':file.name,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':hashed.hexdigest()})
    return {'capture_id':directory.name,'files':manifests,'declared_status':state.get('status','UNKNOWN'),
        'reported_pid':pid,'reported_pid_absent':bool(pid),'declared_book_mode':state.get('book_mode')}


class Measurements:
    def __init__(self,parameters):
        self.p=parameters; self.counts=Counter(); self.depths=Counter(); self.trade_sides=Counter()
        self.series={name:[] for name in ('spread_bps','book_step_ms','book_receipt_minus_exchange_ms','trade_receipt_minus_exchange_ms',
            'trade_size_btc','trade_notional_usd','print_mid_distance_abs_bps','quote_age_at_print_ms','indicative_funding_rate','mark_oracle_basis_bps')}
        self.sweeps={(n,side):{'filled_fraction':[],'entry_drag_bps':[],'levels_consumed':[]} for n in parameters['notionals_usd'] for side in (-1,1)}
        self.first=None; self.last=None; self.quote=None; self.seen_trades={}; self.first_trade=None; self.last_trade=None
        self.segment_start=None; self.segment_last=None; self.segment_rows=0; self.segments=[]; self.fresh_ms=0
        self.trade_notional=0.; self.trade_size=0.; self.endpoint=None; self.book_mode=None; self.started=False; self.ended=False

    def reset(self):
        if self.segment_rows: self.segments.append({'rows':self.segment_rows,'span_seconds':(self.segment_last-self.segment_start)/1000})
        self.segment_start=None; self.segment_last=None; self.segment_rows=0; self.quote=None

    def book(self,data,received):
        if not isinstance(data,dict) or not isinstance(data.get('levels'),list): raise ValueError('Invalid book packet')
        for levels in data['levels']:
            if not isinstance(levels,list): raise ValueError('Invalid book levels')
            for level in levels:
                if not isinstance(level,dict): raise ValueError('Invalid book level')
                number(level['px'],True); number(level['sz'],True)
        t=timestamp(data['time']); DepthBudget(data,self.p['shadow_policy'])
        if self.last is not None and t<self.last:
            self.counts['out_of_order_books']+=1; self.reset(); return
        if self.last is not None and t==self.last:
            self.counts['same_timestamp_books']+=1
            # A repeated timestamp cannot restore continuity after a bad packet,
            # out-of-order book or disconnection. Wait for a new valid book.
            if self.quote is not None: self.quote=(data,received)
            return
        if self.quote is not None:
            dt=t-self.last; self.series['book_step_ms'].append(dt)
            self.fresh_ms+=min(dt,self.p['maximum_gap_ms'])
            if dt>self.p['maximum_gap_ms']:
                self.counts['book_intervals_over_gap_limit']+=1; self.reset()
        if self.segment_start is None: self.segment_start=t
        self.segment_rows+=1; self.segment_last=t
        self.first=t if self.first is None else self.first; self.last=t; self.quote=(data,received)
        bid=float(data['levels'][0][0]['px']); ask=float(data['levels'][1][0]['px']); mid=(bid+ask)/2
        spread=(ask-bid)/mid*10000
        self.series['spread_bps'].append(spread); self.series['book_receipt_minus_exchange_ms'].append(received-t)
        self.depths[(len(data['levels'][0]),len(data['levels'][1]))]+=1; self.counts['valid_unique_books']+=1
        if self.counts['valid_unique_books']>MAX_BOOKS: raise OverflowError('Study book limit exceeded')
        if spread>self.p['maximum_spread_bps']:
            self.counts['books_excluded_by_spread_gate']+=1; return
        for (notional,side),values in self.sweeps.items():
            # Independent alternatives, NOT simultaneous sleeves on one budget.
            fill=DepthBudget(data,self.p['shadow_policy']).take(side,notional/mid,self.p['extra_slippage_bps'])
            values['filled_fraction'].append(fill['quantity']/fill['requested_quantity'])
            values['levels_consumed'].append(len(fill['levels']))
            if fill['price'] is not None: values['entry_drag_bps'].append(side*(fill['price']/mid-1)*10000)

    def trades(self,data,received):
        if not isinstance(data,list): raise ValueError('Trade packet must contain an array')
        for trade in data:
            try:
                if not isinstance(trade,dict): raise ValueError('Invalid trade object')
                if trade['coin']!='BTC' or trade['side'] not in {'A','B'}: raise ValueError('Invalid trade identity')
                t=timestamp(trade['time']); price=number(trade['px'],True); size=number(trade['sz'],True)
                tid=trade['tid']
                if isinstance(tid,bool) or not isinstance(tid,(int,str)) or not re.fullmatch(r'\d+',str(tid)): raise ValueError('Invalid trade ID')
                key=(t,'BTC',str(tid)); observed=(price,size,trade['side'])
                if key in self.seen_trades:
                    self.counts['duplicate_trades']+=1
                    self.counts['conflicting_duplicate_trades']+=int(self.seen_trades[key]!=observed)
                    continue
                if len(self.seen_trades)>=MAX_TRADES: raise OverflowError('Study unique-trade limit exceeded')
                self.seen_trades[key]=observed; self.counts['valid_unique_trades']+=1
                self.trade_sides[trade['side']]+=1; self.trade_size+=size; self.trade_notional+=price*size
                self.first_trade=t if self.first_trade is None else min(t,self.first_trade)
                self.last_trade=t if self.last_trade is None else max(t,self.last_trade)
                self.series['trade_size_btc'].append(size); self.series['trade_notional_usd'].append(price*size)
                self.series['trade_receipt_minus_exchange_ms'].append(received-t)
                if self.quote is None:
                    self.counts['prints_without_previous_quote']+=1; continue
                quote,quote_received=self.quote; age=t-timestamp(quote['time'])
                if quote_received>received or age<0:
                    self.counts['prints_excluded_future_or_unavailable_quote']+=1; continue
                if age>self.p['maximum_gap_ms']:
                    self.counts['prints_excluded_stale_quote']+=1; continue
                bid=float(quote['levels'][0][0]['px']); ask=float(quote['levels'][1][0]['px']); mid=(bid+ask)/2
                self.counts['prints_with_causal_fresh_quote']+=1
                self.counts['prints_below_previous_bid' if price<bid else 'prints_above_previous_ask' if price>ask else 'prints_inside_previous_quotes']+=1
                self.series['print_mid_distance_abs_bps'].append(abs(price/mid-1)*10000)
                self.series['quote_age_at_print_ms'].append(age)
            except (KeyError,TypeError,ValueError): self.counts['invalid_trades']+=1

    def consume(self,event):
        self.counts['records']+=1
        if self.counts['records']>MAX_RECORDS: raise OverflowError('Study record limit exceeded')
        kind=event.get('kind')
        if self.ended: raise RuntimeError('Capture records follow its terminal marker')
        if not self.started and kind!='CAPTURE_START': raise RuntimeError('Capture records lack preceding start provenance')
        if kind=='CAPTURE_START':
            if self.started: raise RuntimeError('Capture has multiple start markers')
            if event.get('coin')!='BTC' or event.get('endpoint') not in {'wss://api.hyperliquid.xyz/ws','wss://api.hyperliquid-testnet.xyz/ws'}:
                raise RuntimeError('Capture start must identify a BTC public venue endpoint')
            self.started=True; self.endpoint=event['endpoint']; self.book_mode=event.get('book_mode')
        elif kind=='CAPTURE_END': self.ended=True; self.reset()
        elif kind in {'CONNECTED','DISCONNECTED','INTERRUPTED'}:
            self.counts[kind.lower()]+=1; self.reset()
        elif kind=='PUBLIC_MESSAGE':
            received=timestamp(event['received_at_ms']); channel=event['channel']; message=event['message']
            if not isinstance(message,dict) or message.get('channel')!=channel: raise ValueError('Inconsistent channel envelope')
            if channel not in {'l2Book','trades','activeAssetCtx'}:
                self.counts['unsupported_public_channel']+=1; return
            data=message['data']; self.counts[channel+'_packets']+=1
            if channel=='l2Book': self.book(data,received)
            elif channel=='trades': self.trades(data,received)
            elif channel=='activeAssetCtx':
                if not isinstance(data,dict) or data.get('coin')!='BTC': raise ValueError('Unexpected asset context coin')
                ctx=data['ctx']
                if not isinstance(ctx,dict): raise ValueError('Invalid asset context')
                funding=number(ctx['funding']) if 'funding' in ctx else None
                basis=None
                if 'markPx' in ctx and 'oraclePx' in ctx:
                    basis=(number(ctx['markPx'],True)/number(ctx['oraclePx'],True)-1)*10000
                if funding is not None: self.series['indicative_funding_rate'].append(funding)
                if basis is not None: self.series['mark_oracle_basis_bps'].append(basis)

    def result(self,src):
        self.reset()
        if not self.started: raise ValueError('Missing capture start provenance')
        if not src['reported_pid_absent'] and not self.ended: raise ValueError('Capture termination/liveness cannot be established')
        span=(self.last-self.first) if self.first is not None else 0
        return {'endpoint':self.endpoint,'mainnet':self.endpoint=='wss://api.hyperliquid.xyz/ws','observed_book_mode':self.book_mode,
            'capture_end_present':self.ended,'effective_source_state':'TERMINAL_MARKER_PRESENT' if self.ended else 'INTERRUPTED_NO_END_MARKER',
            'quality_counts':dict(self.counts),'depth_observations':[{'bid_levels':b,'ask_levels':a,'books':n} for (b,a),n in sorted(self.depths.items())],
            'book_window':{'first_exchange_ms':self.first,'last_exchange_ms':self.last,'span_seconds':span/1000,
                'bounded_previous_quote_freshness_seconds':self.fresh_ms/1000,'bounded_freshness_fraction':self.fresh_ms/span if span else None,
                'interpretation':'Time covered by previous observed quotes up to the gap limit within recorded connections. Not complete event coverage or forward eligibility.'},
            'segments':{'count':len(self.segments),'rows':statistics([s['rows'] for s in self.segments]),
                'span_seconds':statistics([s['span_seconds'] for s in self.segments])},
            'observations':{name:statistics(values) for name,values in self.series.items()},
            'trades':{'unique':len(self.seen_trades),'reported_side_counts':dict(self.trade_sides),
                'observed_volume_btc':self.trade_size,'observed_notional_usd':self.trade_notional,
                'first_exchange_ms':self.first_trade,'last_exchange_ms':self.last_trade,
                'interpretation':'Observed deduplicated public prints, possibly with missing intervals; no own-order fill or complete market-volume assertion'},
            'hypothetical_sweeps':[{'requested_notional_usd':n,'side':'BUY' if side>0 else 'SELL',
                **{name:statistics(values) for name,values in series.items()},
                'fraction_fully_filled':sum(v>=1-1e-10 for v in series['filled_fraction'])/len(series['filled_fraction']) if series['filled_fraction'] else None}
                for (n,side),series in self.sweeps.items()]}


def load(store,study_id):
    with store.connect() as db: row=db.execute('SELECT payload FROM venue_studies WHERE id=?',(study_id,)).fetchone()
    if row is None: raise ValueError('Unknown venue study')
    payload=json.loads(row[0])
    if digest(payload)!=study_id: raise ValueError('Venue study integrity failed')
    return payload


def summaries(store,limit=6):
    """Latest implementation per capture/assumption set, not independent samples."""
    with store.connect() as db:
        records=db.execute('SELECT id,payload FROM venue_studies ORDER BY created DESC LIMIT 200').fetchall()
    result=[]; seen=set()
    for record in records:
        try:
            report=json.loads(record['payload'])
            if digest(report)!=record['id']: continue
            key=(report['source']['capture_id'],digest(report['parameters']))
            if key in seen: continue
            seen.add(key)
            result.append({'study_id':record['id'],'capture_id':key[0],'mainnet':report['mainnet'],
                'source_state':report['effective_source_state'],'depth_observations':report['depth_observations'],
                'book_step_ms':report['observations']['book_step_ms'],'segments':report['segments'],
                'book_window':report['book_window'],'maximum_gap_ms':report['parameters']['maximum_gap_ms'],
                'shadow_policy':report['parameters']['shadow_policy'],
                'coverage_verified':False,'strategy_pnl_computed':False,
                'interpretation':'Latest version per capture/assumptions. Read exact ID. Captures may overlap; versions are not independent evidence. WebSocket cadence does not establish REST or inference/routing timing. No calibrated own fills or forward approval.'})
        except (KeyError,TypeError,ValueError): continue
        if len(result)>=limit: break
    return result


def study(store,settings,capture_id,notionals=None,control=None):
    directory=capture_directory(store.home,capture_id); src=source(directory,control)
    notionals=[250.,1000.,2500.,10000.] if notionals is None else notionals
    if not isinstance(notionals,list) or not 1<=len(notionals)<=6 or any(isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not 50<=n<=100000 for n in notionals) or len(set(notionals))!=len(notionals):
        raise ValueError('Choose 1..6 unique finite notional scenarios in 50..100000 USD')
    parameters={'notionals_usd':sorted(map(float,notionals)),'shadow_policy':settings.shadow_dict(),
        'extra_slippage_bps':settings.slippage_bps,'maximum_gap_ms':settings.max_gap_ms,'maximum_spread_bps':settings.max_spread_bps,
        'quantity_convention':'Requested base quantity = scenario notional / contemporaneous midpoint. Independent alternatives per book; no simultaneous sleeve consumption.'}
    if isinstance(settings.max_gap_ms,bool) or not isinstance(settings.max_gap_ms,int) or settings.max_gap_ms<=0 or not math.isfinite(settings.slippage_bps) or settings.slippage_bps<0 or not math.isfinite(settings.max_spread_bps) or settings.max_spread_bps<=0:
        raise ValueError('Invalid study execution assumptions')
    implementation=algorithm_manifest(); algorithm=digest(implementation)
    study_key=digest({'source':src,'parameters':parameters,'version':VERSION,'algorithm_sha256':algorithm})
    with store.connect() as db: cached=db.execute('SELECT id FROM venue_studies WHERE study_key=?',(study_key,)).fetchone()
    if cached: return {'study_id':cached[0],'already_measured':True,'report':load(store,cached[0])}
    measured=Measurements(parameters)
    scanned=0
    for item in src['files']:
        file=directory/item['file']; hashed=hashlib.sha256()
        with file.open('rb') as stream:
            while True:
                line=stream.readline(512001)
                if not line: break
                if len(line)>512000: raise ValueError('Capture record exceeds bounded record size')
                scanned+=1
                if scanned>MAX_RECORDS: raise OverflowError('Study scanned-record limit exceeded')
                hashed.update(line)
                if scanned%1000==1 and control: control()
                try:
                    event=json.loads(line)
                    if not isinstance(event,dict): raise ValueError('Capture record must be an object')
                    measured.consume(event)
                except (ValueError,KeyError,TypeError):
                    measured.counts['invalid_records']+=1; measured.reset()
        after=file.stat()
        if hashed.hexdigest()!=item['sha256'] or (after.st_size,after.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):
            raise ValueError('Capture changed during study; no evidence recorded')
    # Earlier files and process/status provenance must still match after the
    # entire scan, including changes made while a later day's file was read.
    if source(directory,control)!=src: raise ValueError('Capture changed before study commit; no evidence recorded')
    if algorithm_manifest()!=implementation: raise ValueError('Study implementation changed before commit')
    payload={'version':VERSION,'algorithm_sha256':algorithm,'source':src,'parameters':parameters,
        'algorithm_manifest':implementation,
        **measured.result(src),'scanned_lines':scanned,'coverage_verified':False,'strategy_pnl_computed':False,'orders_submitted_to_exchange':0,
        'limits':['Displayed sweeps are assumptions, not calibrated market impact, replenishment or achievable capacity.',
            'Quote statistics are message-weighted. Receipt minus exchange time includes clock offset, batching and local/network delay; it is not measured one-way latency.',
            'Print matching uses only previously received quotes with exchange time no later than the print and age within the gap limit. Prints are not our fills or slippage calibration.',
            'Five-level captures cannot establish twenty-level features or unobserved liquidity.',
            'Asset funding context is indicative, not a historical settlement cashflow model.',
            'Malformed records, disconnects and out-of-order books conservatively break quote continuity; repeated book timestamps cannot restore it.',
            'Interrupted/reconnected sources may lose events. No maker, strategy, company, forward, portfolio or capital approval.']}
    ident=digest(payload)
    with store.connect() as db:
        db.execute('INSERT OR IGNORE INTO venue_studies VALUES(?,?,?,?)',(ident,study_key,json.dumps(payload,allow_nan=False),time.time()))
    output=store.home/'venue_studies'; output.mkdir(parents=True,exist_ok=True)
    path=output/(ident+'.json'); temp=output/(ident+'.'+uuid.uuid4().hex+'.partial')
    temp.write_text(json.dumps({'study_id':ident,'report':payload},indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)
    return {'study_id':ident,'already_measured':False,'report':payload}
