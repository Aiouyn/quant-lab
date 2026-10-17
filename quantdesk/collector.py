"""Bounded public venue capture for future venue-specific execution research.

Snapshots, actual trades, and public asset context are preserved separately.
Reconnects mark missing intervals; capture never proves queue priority.
"""
import json
import time
import uuid
import os
from collections import Counter,deque
from datetime import datetime,timezone
from pathlib import Path


class Collector:
    def __init__(self,home,coin='BTC',testnet=False,book_fast=False):
        if coin!='BTC':
            raise ValueError('Current research dataset and collector support BTC')
        self.coin=coin
        self.book_fast=bool(book_fast)
        self.endpoint='wss://api.hyperliquid-testnet.xyz/ws' if testnet else 'wss://api.hyperliquid.xyz/ws'
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        self.directory=Path(home)/'market_data'/('hyperliquid-'+stamp+'-'+uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True,exist_ok=True)
        self.counts=Counter(); self.duplicates=0; self.reconnects=0; self.errors=0
        self.seen=set(); self.recent=deque(); self.last_book_ts=None
        self.started_ms=None; self.first_book_ts=None; self.last_progress=0.0

    def progress(self,status='RUNNING'):
        now_ms=int(time.time()*1000)
        payload={'status':status,'pid':os.getpid(),'directory':str(self.directory),
            'book_mode':'fast_5_levels' if self.book_fast else 'slow_20_levels',
            'started_ms':self.started_ms,'updated_ms':now_ms,'counts':dict(self.counts),
            'first_book_ts':self.first_book_ts,'last_book_ts':self.last_book_ts,
            'reconnects':self.reconnects,'errors':self.errors,'orders_submitted_to_exchange':0,
            'coverage_verified':False}
        target=self.directory/'status.json'; temporary=target.with_suffix('.partial')
        temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(target)
        self.last_progress=time.monotonic()

    def write(self,event):
        now_ms=int(time.time()*1000)
        day=datetime.fromtimestamp(now_ms/1000,timezone.utc).strftime('%Y-%m-%d')
        with (self.directory/(day+'.jsonl')).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event,received_at_ms=now_ms),allow_nan=False)+'\n')

    def consume(self,message):
        channel=message.get('channel'); data=message.get('data')
        if channel=='error':
            raise ValueError('Public subscription returned an error')
        if channel not in {'l2Book','trades','activeAssetCtx'}:
            return
        if channel=='trades':
            if not isinstance(data,list):
                raise ValueError('Trade feed must contain a list')
            fresh=[]
            for trade in data:
                if trade.get('coin')!=self.coin:
                    raise ValueError('Unexpected coin in trade feed')
                # tid alone is not globally unique according to the venue.
                key=(int(trade['time']),trade['coin'],trade['tid'])
                if key in self.seen:
                    self.duplicates+=1
                    continue
                self.seen.add(key); self.recent.append(key); fresh.append(trade)
                if len(self.recent)>100000:
                    self.seen.remove(self.recent.popleft())
            self.counts['trades']+=len(fresh)
            # Keep raw packets, including duplicates, for independent auditing.
        else:
            if not isinstance(data,dict) or data.get('coin')!=self.coin:
                raise ValueError('Unexpected public feed payload')
            if channel=='l2Book':
                ts=int(data['time'])
                if self.first_book_ts is None:
                    self.first_book_ts=ts
                if self.last_book_ts is not None and ts-self.last_book_ts>5000:
                    self.write({'kind':'BOOK_GAP','previous_ts':self.last_book_ts,'next_ts':ts})
                    self.counts['book_gaps']+=1
                if self.last_book_ts is not None and ts<self.last_book_ts:
                    self.write({'kind':'OUT_OF_ORDER_BOOK','previous_ts':self.last_book_ts,'next_ts':ts})
                    self.counts['out_of_order_books']+=1
                self.last_book_ts=ts
            self.counts[channel]+=1
        self.write({'kind':'PUBLIC_MESSAGE','channel':channel,'message':message})
        if time.monotonic()-self.last_progress>=10:
            self.progress()

    def run(self,seconds=3600,max_reconnects=5,connect=None):
        if not 1<=seconds<=86400 or not 0<=max_reconnects<=20:
            raise ValueError('Capture duration 1..86400 seconds; reconnect limit 0..20')
        if connect is None:
            from websockets.sync.client import connect
        deadline=time.monotonic()+seconds
        self.started_ms=int(time.time()*1000)
        self.progress()
        self.write({'kind':'CAPTURE_START','endpoint':self.endpoint,'coin':self.coin,'signing_enabled':False,
            'book_mode':'fast_5_levels' if self.book_fast else 'slow_20_levels'})
        stopped=False
        try:
            while time.monotonic()<deadline:
                try:
                    with connect(self.endpoint,open_timeout=min(10,max(1,deadline-time.monotonic()))) as socket:
                        for channel in ('l2Book','trades','activeAssetCtx'):
                            subscription={'type':channel,'coin':self.coin}
                            if channel=='l2Book':
                                subscription['fast']=self.book_fast
                            socket.send(json.dumps({'method':'subscribe','subscription':subscription}))
                        self.write({'kind':'CONNECTED','connection':self.reconnects})
                        self.progress()
                        while time.monotonic()<deadline:
                            try:
                                raw=socket.recv(timeout=min(5,max(0.01,deadline-time.monotonic())))
                            except TimeoutError:
                                if time.monotonic()-self.last_progress>=10:
                                    self.progress()
                                continue
                            self.consume(json.loads(raw))
                except Exception as error:
                    self.errors+=1
                    self.write({'kind':'DISCONNECTED','error_type':type(error).__name__,'missing_interval_possible':True})
                    if self.reconnects>=max_reconnects or time.monotonic()>=deadline:
                        stopped=True
                        break
                    self.reconnects+=1
                    time.sleep(min(2**(self.reconnects-1),10,max(0,deadline-time.monotonic())))
        except KeyboardInterrupt:
            stopped=True
            self.write({'kind':'INTERRUPTED'})
        summary={'directory':str(self.directory),'coin':self.coin,'endpoint':self.endpoint,'book_mode':'fast_5_levels' if self.book_fast else 'slow_20_levels','counts':dict(self.counts),'duplicate_trades':self.duplicates,'reconnects':self.reconnects,'errors':self.errors,'stopped_early':stopped,'orders_submitted_to_exchange':0,'coverage_verified':False,'limitation':'Public L2 is aggregated snapshots; trades support execution calibration but do not reveal queue priority. Disconnects may lose events. No automatic maker promotion.'}
        self.write({'kind':'CAPTURE_END','summary':summary})
        self.progress('STOPPED' if stopped else 'COMPLETE')
        (self.directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        from .store import Store
        Store(self.directory.parent.parent).memory('capture','','VENUE_OBSERVATION',summary)
        return summary
