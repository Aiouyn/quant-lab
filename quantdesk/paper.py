"""Public Hyperliquid books -> local shadow account. No signing/order endpoint."""
import json
import os
import math
import time
import urllib.request
import uuid
from pathlib import Path
import numpy as np
import polars as pl
from .data import Book,enrich
from .execution import Broker
from .strategy import run_signal,run_feature_program,validate_code,validate_spec
from .store import digest
from .store import Store

MAINNET='https://api.hyperliquid.xyz/info'
TESTNET='https://api.hyperliquid-testnet.xyz/info'

def public_book(coin='BTC',testnet=False):
    request=urllib.request.Request(TESTNET if testnet else MAINNET,data=json.dumps({'type':'l2Book','coin':coin}).encode(),headers={'Content-Type':'application/json','User-Agent':'QuantLabResearch/2.0'},method='POST')
    with urllib.request.urlopen(request,timeout=10) as response:
        book=json.load(response)
    if book.get('coin')!=coin or len(book.get('levels',[]))!=2:
        raise ValueError('Invalid Hyperliquid book response')
    return book

def normalize_book(book):
    bids,asks=book['levels']
    return {'action':'snapshot','_continuous_snapshot':True,'ts':int(book['time']),'bids':[[l['px'],l['sz'],l.get('n',0)] for l in bids],'asks':[[l['px'],l['sz'],l.get('n',0)] for l in asks]}

def load_bundle(directory,exploratory=False):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') not in {'quantdesk-v2','quantdesk-v3'}:
        raise ValueError('Unsupported strategy bundle')
    if manifest['status']!='PAPER_READY' and not exploratory:
        raise ValueError('Bundle is exploratory; use --exploratory to explicitly shadow-test an unqualified system')
    code=(directory/'strategy.py').read_text(encoding='utf-8')
    if digest(code)!=manifest['code_hash'] or digest(manifest['spec'])!=manifest['spec_hash']:
        raise ValueError('Bundle integrity check failed; do not execute edited evidence artifacts')
    if digest(manifest['execution'])!=manifest.get('execution_hash'):
        raise ValueError('Execution integrity check failed; export a fresh bundle with recorded costs and risk limits')
    if 'shadow_execution' in manifest:
        from .liquidity import validate_policy
        validate_policy(manifest['shadow_execution'])
        if digest(manifest['shadow_execution'])!=manifest.get('shadow_execution_hash'):
            raise ValueError('Shadow liquidity integrity check failed')
    elif not exploratory:
        raise ValueError('Qualified shadow execution requires a fresh bundle with frozen displayed-depth policy')
    validate_code(code); validate_spec(manifest['spec'])
    if manifest['format']=='quantdesk-v3':
        from .input_contract import input_contract
        if manifest.get('input_contract')!=input_contract(code,manifest['spec']):
            raise ValueError('Input contract integrity check failed')
    if 'model' in manifest['spec']:
        from .models import verify_artifact
        artifact=json.loads((directory/'model.json').read_text(encoding='utf-8'))
        verify_artifact(artifact)
        if artifact['artifact_hash'] != manifest.get('model_hash') or artifact['code_hash'] != manifest['code_hash'] or artifact['spec_hash'] != manifest['spec_hash']:
            raise ValueError('Fitted model does not match bundle integrity hashes')
        if artifact['execution'] != manifest['execution']:
            raise ValueError('Fitted model execution assumptions differ from the bundle')
        manifest['_model_artifact']=artifact
    elif manifest.get('model_hash'):
        raise ValueError('Unexpected fitted artifact on formula-only bundle')
    return manifest,code

class PaperSession:
    def __init__(self,bundle,home,settings,exploratory=False):
        path=Path(bundle)
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        self.settings=settings
        self.members=[]
        self.is_portfolio=manifest.get('format')=='quantdesk-portfolio-v2'
        if manifest.get('format')=='quantdesk-portfolio-v2':
            if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest.get('manifest_hash'):
                raise ValueError('Portfolio manifest integrity check failed')
            total=sum(float(m['weight']) for m in manifest['members'])
            if total>1.000001 or any(not 0<float(m['weight'])<=settings.max_strategy_weight for m in manifest['members']):
                raise ValueError('Portfolio allocation violates gross/strategy limits')
            for member in manifest['members']:
                strategy,code=load_bundle(member['bundle'])
                if member['candidate_id']!=strategy['id'] or any(m['id']==strategy['id'] for m in self.members):
                    raise ValueError('Portfolio member identities are inconsistent or duplicated')
                execution=dict(strategy['execution'],initial_equity=settings.initial_equity*member['weight'])
                self.members.append({'id':strategy['id'],'manifest':strategy,'code':code,'broker':Broker(strategy['spec'],execution,strategy.get('shadow_execution')),'weight':member['weight']})
            self.cash_equity=settings.initial_equity*(1-total)
        else:
            strategy,code=load_bundle(path,exploratory)
            execution=dict(strategy['execution'],initial_equity=settings.initial_equity)
            self.members.append({'id':strategy['id'],'manifest':strategy,'code':code,'broker':Broker(strategy['spec'],execution,strategy.get('shadow_execution')),'weight':1.0})
            self.cash_equity=0.0
        self.directory=Path(home)/'paper'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        self.home=Path(home)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.journal=self.directory/'journal.jsonl'
        self.book=Book(settings.max_gap_ms); self.history=[]; self.last_book=None; self.halted=False
        self.initial=settings.initial_equity; self.peak=self.initial
        self.samples=0; self.total_errors=0; self.strategy_errors=0; self.inferences=0; self.stale_decisions=0
        if any(m['broker'].liquidity_policy is not None and m['broker'].liquidity_policy!=settings.shadow_dict() for m in self.members):
            raise ValueError('Session shadow settings differ from the frozen bundle')
        from .forward import FORMAT,bundle_identity
        self.metadata={'format':FORMAT,'created_ms':int(time.time()*1000),'exploratory':exploratory,
            'portfolio':self.is_portfolio,
            'initial_equity':self.initial,'execution':settings.execution_dict(),'shadow_execution':settings.shadow_dict(),'data_source':'manual_or_unverified',
            'members':[{'id':m['id'],'status':m['manifest']['status'],'bundle_identity':bundle_identity(m['manifest']),
                'execution':m['broker'].config,'shadow_execution':m['broker'].liquidity_policy,'weight':m['weight']} for m in self.members]}
        self._run_started=False
        self.write_metadata()
        self.log({'kind':'SESSION_START','bundle':str(path),'exploratory':exploratory,'venue':'Hyperliquid public data','initial_equity':self.initial,'signing_enabled':False})

    def write_metadata(self):
        path=self.directory/'session.json'; staged=path.with_suffix('.partial')
        staged.write_text(json.dumps(self.metadata,indent=2,allow_nan=False),encoding='utf-8')
        staged.replace(path)

    def publish_status(self,status):
        path=self.directory/'status.json'
        partial=path.with_suffix('.partial')
        partial.write_text(json.dumps({'status':status,'pid':os.getpid(),
            'directory':str(self.directory),'samples':self.samples,'inferences':self.inferences,
            'feed_errors':self.total_errors,'strategy_errors':self.strategy_errors,'stale_decisions':self.stale_decisions,
            'halted':self.halted,'updated_ms':int(time.time()*1000),
            'orders_submitted_to_exchange':0},allow_nan=False),encoding='utf-8')
        partial.replace(path)

    def log(self,event):
        with self.journal.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event,received_at_ms=int(time.time()*1000)),allow_nan=False)+'\n')

    def consume(self,book,now_ms=None):
        consumed_at=time.monotonic()
        now_ms=int(time.time()*1000) if now_ms is None else now_ms
        if abs(now_ms-int(book['time']))>self.settings.max_gap_ms:
            for m in self.members:
                m['broker'].pending=None
            self.log({'kind':'STALE_BOOK','exchange_ts':book['time']})
            return False
        if self.last_book and int(book['time'])<=self.last_book['ts_ms']:
            self.log({'kind':'DUPLICATE_OR_OLD_BOOK','exchange_ts':book['time']})
            return False
        row=self.book.apply(normalize_book(book))
        if row is None:
            self.log({'kind':'INVALID_BOOK'})
            return False
        from .liquidity import DepthBudget
        depth=DepthBudget(book,self.settings.shadow_dict()) if any(m['broker'].liquidity_policy is not None for m in self.members) else None
        execution_row=dict(row,_liquidity=depth)
        observed_depth=min(len(side) for side in book['levels'])
        incompatible=[m['id'] for m in self.members if observed_depth<m['manifest'].get('input_contract',{}).get('minimum_book_levels',1)]
        if incompatible:
            self.halted=True
            self.log({'kind':'INPUT_CONTRACT_ERROR','observed_book_levels':observed_depth,
                'candidates':incompatible,'reason':'Insufficient depth for researched features; no inference will run'})
        reset=bool(self.last_book and (row['segment']!=self.last_book['segment'] or
            row['ts_ms']//86400000!=self.last_book['ts_ms']//86400000))
        if reset:
            self.history=[]
            for member in self.members:
                broker=member['broker']; count=len(broker.events)
                broker.step(execution_row,0.0,terminal=True)
                for event in broker.events[count:]:
                    self.log(dict(event,candidate=member['id']))
                broker.events=[]
        self.history.append(row)
        self.log({'kind':'BOOK','exchange_ts':book['time'],'book':book})
        # Never silently drop the history used by cumulative/online signals.
        # A session segment is bounded by UTC day and an explicit hard limit.
        if len(self.history)>100000:
            self.halted=True
            self.log({'kind':'HISTORY_LIMIT','rows':len(self.history)})
        self.last_book=row
        frame=enrich(pl.DataFrame(self.history))
        total=self.cash_equity
        for member in self.members:
            broker=member['broker']; count=len(broker.events)
            try:
                warmup=member['manifest']['spec']['warmup_rows']
                signal=0.0
                if len(frame)>warmup and not self.halted and not broker.account.halted:
                    inference_started=time.monotonic()
                    artifact=member['manifest'].get('_model_artifact')
                    runner=run_feature_program if artifact and artifact['recipe'].get('custom_features',False) else run_signal
                    from .input_contract import program_input
                    alpha=runner(member['code'],program_input(frame,member['manifest']['spec']),self.settings.worker_timeout,False)
                    if artifact:
                        from .models import fitted_signal
                        if row['ts_ms'] < artifact['provenance']['prediction_cutoff_ms']:
                            raise ValueError('Paper observations precede model training cutoff')
                        alpha=fitted_signal(frame,alpha,artifact)
                    signal=float(alpha[-1])
                    self.inferences+=1
                    self.log({'kind':'INFERENCE','candidate':member['id'],'exchange_ts':row['ts_ms'],
                        'signal':signal,'elapsed_ms':(time.monotonic()-inference_started)*1000,
                        'decision_received_at_ms':int(time.time()*1000)})
                if self.halted:
                    signal=0.0
                decision_ts=now_ms+int((time.monotonic()-consumed_at)*1000)
                if decision_ts-row['ts_ms']>self.settings.max_gap_ms:
                    self.stale_decisions+=1; self.halted=True
                    broker.pending=None; broker.account.halted=True
                    self.log({'kind':'STALE_DECISION','candidate':member['id'],'exchange_ts':row['ts_ms'],
                        'decision_available_ts':decision_ts,'open_exposure_preserved':True})
                    total+=broker.mark(row)
                else:
                    total+=broker.step(execution_row,signal,terminal=self.halted,decision_ts_ms=decision_ts)
            except Exception as error:
                self.strategy_errors+=1
                broker.account.halted=True
                broker.pending=None
                self.log({'kind':'STRATEGY_ERROR','candidate':member['id'],'error':str(error)})
                if now_ms+int((time.monotonic()-consumed_at)*1000)-row['ts_ms']>self.settings.max_gap_ms:
                    self.stale_decisions+=1; self.halted=True
                    total+=broker.mark(row)
                    self.log({'kind':'STALE_DECISION','candidate':member['id'],'open_exposure_preserved':True})
                else:
                    total+=broker.step(execution_row,0.0,terminal=True)
            for event in broker.events[count:]:
                self.log(dict(event,candidate=member['id']))
            # Broker retains only the live state; durable fills live in the journal.
            broker.events=[]
        self.peak=max(self.peak,total)
        if (self.peak-total)/self.peak>=self.settings.max_drawdown_fraction:
            self.halted=True
            for member in self.members:
                broker=member['broker']; count=len(broker.events)
                if now_ms+int((time.monotonic()-consumed_at)*1000)-row['ts_ms']<=self.settings.max_gap_ms:
                    broker.step(execution_row,0.0,terminal=True)
                else:
                    broker.pending=None
                for event in broker.events[count:]:
                    self.log(dict(event,candidate=member['id']))
                broker.events=[]
            total=self.cash_equity+sum(m['broker'].mark(row) for m in self.members)
        self.samples+=1
        if all(m['broker'].account.halted for m in self.members):
            self.halted=True
        self.log({'kind':'MARK','exchange_ts':row['ts_ms'],'equity':total,'gross_notional':sum(abs(m['broker'].account.quantity)*row['mid_price'] for m in self.members),'net_notional':sum(m['broker'].account.quantity*row['mid_price'] for m in self.members),'halted':self.halted})
        return True

    def run(self,samples=300,interval=1.0,testnet=False,coin='BTC',fetch=public_book):
        if interval<1 or samples<1:
            raise ValueError('Use at least one sample and a polling interval >= 1 second')
        if self._run_started:
            raise ValueError('A paper session may run only once; start a new durable session')
        self._run_started=True
        self.metadata.update(coin=coin,testnet=testnet,requested_samples=samples,interval_seconds=interval,
            data_source='hyperliquid_public_testnet' if fetch is public_book and testnet else 'hyperliquid_public_mainnet' if fetch is public_book else 'custom_fetch_unverified')
        if fetch is public_book and not testnet and coin=='BTC' and not self.is_portfolio and len(self.members)==1 and not self.metadata['exploratory'] and self.members[0]['manifest']['status']=='PAPER_READY':
            from .forward import plan_forward
            plan=plan_forward(Store(self.home),self.members[0]['manifest'],self.settings)
            self.metadata['forward_plan_id']=plan['id']
        self.metadata['run_started_ms']=int(time.time()*1000)
        self.write_metadata()
        self.log({'kind':'SESSION_RUN_START','session_hash':digest(self.metadata),'data_source':self.metadata['data_source'],'forward_plan_id':self.metadata.get('forward_plan_id')})
        failures=0
        termination='SAMPLE_LIMIT'
        self.publish_status('RUNNING')
        try:
            for _ in range(samples):
                started=time.monotonic()
                try:
                    book=fetch(coin,testnet)
                    accepted=self.consume(book)
                    if not accepted:
                        raise ValueError('Feed did not provide a fresh valid book')
                    failures=0
                    if self.halted:
                        break
                except Exception as error:
                    failures+=1; self.total_errors+=1
                    for member in self.members:
                        member['broker'].pending=None
                    self.log({'kind':'FEED_ERROR','error_type':type(error).__name__,'consecutive_failures':failures})
                    if failures>=3:
                        self.halted=True
                        break
                time.sleep(max(0,interval-(time.monotonic()-started)))
                self.publish_status('RUNNING')
        except KeyboardInterrupt:
            termination='INTERRUPTED'
            self.log({'kind':'INTERRUPTED'})
        except Exception:
            termination='FAILED'
            raise
        finally:
            # No fabricated close on stale prices. Final open exposure is marked and explicit.
            summary=self.summary()
            summary.update(termination='HALTED' if self.halted else termination,forward_plan_id=self.metadata.get('forward_plan_id'),
                data_source=self.metadata['data_source'],session_hash=digest(self.metadata))
            self.log({'kind':'SESSION_END','summary':summary})
            from .forward import file_hash
            summary['journal_sha256']=file_hash(self.journal)
            staged=self.directory/'summary.partial'
            staged.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
            staged.replace(self.directory/'summary.json')
            self.publish_status('HALTED' if self.halted else termination if termination in {'INTERRUPTED','FAILED'} else 'COMPLETE')
            store=Store(self.home)
            for member in self.members:
                manifest=member['manifest']
                store.memory('paper',member['id'],'PAPER_OBSERVATION',{'title':manifest['title'],'bundle_status':manifest['status'],'summary':summary,'sleeve':member['broker'].summary(),'spec':manifest['spec'],'source':str(self.directory/'summary.json'),'interpretation':'Forward venue observation; short sessions and exploratory probes do not establish alpha or capital readiness'})
        return summary

    def summary(self):
        row=self.last_book
        equity=self.cash_equity+sum(m['broker'].mark(row) if row else m['broker'].initial for m in self.members)
        return {'directory':str(self.directory),'samples':self.samples,'inferences':self.inferences,'strategy_errors':self.strategy_errors,'stale_decisions':self.stale_decisions,'feed_errors':self.total_errors,'equity':equity,'pnl':equity-self.initial,'halted':self.halted,'open_positions':{m['id']:m['broker'].account.quantity for m in self.members},'fees':sum(m['broker'].account.fees for m in self.members),'closed_trades':sum(len(m['broker'].trades) for m in self.members),'funding_included':False,'funding_boundary_crossings':sum(m['broker'].funding_boundary_crossings for m in self.members),'orders_submitted_to_exchange':0,'shadow_fill_models':{m['id']:m['broker'].summary()['fill_model'] for m in self.members},'venue_transfer_status':'UNVERIFIED: displayed depth is sampled, participation is assumed, actual impact and fills remain uncalibrated','last_exchange_ts':row['ts_ms'] if row else None}
