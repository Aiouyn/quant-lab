"""Streaming reconstruction, bounded feature cache, chronological campaign plans."""
import hashlib
import heapq
import json
import math
import re
import tarfile
import uuid
import os
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np
import polars as pl
from .store import digest

FEATURE_VERSION = '2.0.1'
FEATURES = ('ts_ms','bid_p0','ask_p0','bid_v0','ask_v0','bid_n0','ask_n0','spread','spread_bps','mid_price','micro_price','micro_dev','micro_dev_bps','bid_v5','ask_v5','qi_1','qi_5','qi_20','count_skew_5','ofi','ofi_norm','vol_30s_bps','return_1s_bps','momentum_10s_bps','depth_total','segment')

@dataclass
class Quality:
    messages: int = 0
    snapshots: int = 0
    sequence_gaps: int = 0
    timestamp_gaps: int = 0
    crossed_books: int = 0
    duplicate_timestamps: int = 0
    has_sequence_ids: bool = False
    max_gap_ms: int = 0

class Book:
    def __init__(self,max_gap_ms=5000):
        if isinstance(max_gap_ms,bool) or not isinstance(max_gap_ms,int) or max_gap_ms<=0:
            raise ValueError('Book gap limit must be a positive integer')
        self.max_gap_ms=max_gap_ms
        self.bids,self.asks = {},{}
        self.bid_heap,self.ask_heap = [],[]
        self.previous = None
        self.sequence = None
        self.ready = False
        self.last_ts = None
        self.segment = 0
        self.quality = Quality()

    def apply(self,row):
        self.quality.messages+=1
        action = row.get('action')
        if action not in {'snapshot','update'}:
            return None
        ts = int(row['ts'])
        if self.last_ts is not None and ts < self.last_ts:
            raise ValueError('Out of order exchange timestamp')
        if self.last_ts is not None and ts == self.last_ts:
            self.quality.duplicate_timestamps+=1
        gap = ts-self.last_ts if self.last_ts is not None else 0
        self.quality.max_gap_ms = max(self.quality.max_gap_ms,gap)
        if gap>self.max_gap_ms:
            self.quality.timestamp_gaps+=1
            self.segment+=1
            self.previous=None
        self.last_ts = ts
        seq = row.get('seqId')
        if seq is not None:
            self.quality.has_sequence_ids=True
            if action=='update' and self.sequence is not None and int(row.get('prevSeqId',self.sequence))!=self.sequence:
                self.ready=False
                self.quality.sequence_gaps+=1
                self.segment+=1
            self.sequence=int(seq)
        if action=='snapshot':
            self.bids.clear(); self.asks.clear()
            self.bid_heap.clear(); self.ask_heap.clear()
            self.ready=True
            if not row.get('_continuous_snapshot'):
                self.previous=None
                self.segment+=1
            self.quality.snapshots+=1
        if not self.ready:
            return None
        for side,name,heap,multiplier in ((self.bids,'bids',self.bid_heap,-1),(self.asks,'asks',self.ask_heap,1)):
            for level in row.get(name,[]):
                p,v = float(level[0]),float(level[1])
                count = int(level[-1]) if len(level)>=3 else 0
                if not math.isfinite(p+v) or p<=0 or v<0 or count<0:
                    raise ValueError('Invalid orderbook level')
                if v==0:
                    side.pop(p,None)
                else:
                    if p not in side:
                        heapq.heappush(heap,multiplier*p)
                    side[p]=(v,count)
            if len(heap)>3*len(side)+100:
                heap[:]=[multiplier*p for p in side]
                heapq.heapify(heap)
        if not self.bids or not self.asks:
            return None
        bids = self.top_prices(self.bids,self.bid_heap,-1)
        asks = self.top_prices(self.asks,self.ask_heap,1)
        bp,ap = bids[0],asks[0]
        if bp>=ap:
            self.quality.crossed_books+=1
            self.previous=None
            return None
        bv,bn = self.bids[bp]; av,an = self.asks[ap]
        mid = (bp+ap)/2
        micro = (ap*bv+bp*av)/(bv+av)
        b5,a5 = sum(self.bids[p][0] for p in bids[:5]),sum(self.asks[p][0] for p in asks[:5])
        b20,a20 = sum(self.bids[p][0] for p in bids),sum(self.asks[p][0] for p in asks)
        n5b,n5a = sum(self.bids[p][1] for p in bids[:5]),sum(self.asks[p][1] for p in asks[:5])
        ofi=0.0
        if self.previous:
            pb,pbv,pa,pav=self.previous
            # Top of book OFI includes disappearance of the prior queue on worsening quotes.
            db = bv if bp>pb else bv-pbv if bp==pb else -pbv
            da = av if ap<pa else av-pav if ap==pa else -pav
            ofi=db-da
        self.previous=(bp,bv,ap,av)
        return {'ts_ms':ts,'bid_p0':bp,'ask_p0':ap,'bid_v0':bv,'ask_v0':av,'bid_n0':bn,'ask_n0':an,'spread':ap-bp,'spread_bps':(ap-bp)/mid*1e4,'mid_price':mid,'micro_price':micro,'micro_dev':micro-mid,'micro_dev_bps':(micro-mid)/mid*1e4,'bid_v5':b5,'ask_v5':a5,'qi_1':(bv-av)/(bv+av),'qi_5':(b5-a5)/(b5+a5),'qi_20':(b20-a20)/(b20+a20),'count_skew_5':(n5b-n5a)/max(n5b+n5a,1),'ofi':ofi,'ofi_norm':ofi/(bv+av),'depth_total':b20+a20,'segment':self.segment}

    @staticmethod
    def top_prices(side,heap,multiplier):
        prices=[]; taken=[]; seen=set()
        while heap and len(prices)<20:
            value=heapq.heappop(heap); price=multiplier*value
            if price not in side or price in seen:
                continue
            seen.add(price); prices.append(price); taken.append(value)
        for value in taken:
            heapq.heappush(heap,value)
        return prices

def enrich(frame):
    if frame.is_empty():
        raise ValueError('No valid synchronized orderbook observations')
    pieces=[]
    for segment in frame.partition_by('segment',maintain_order=True):
        ts=segment['ts_ms'].to_numpy()
        mid=segment['mid_price'].to_numpy()
        # Last observed price at/before t-h, never a future interpolation.
        outputs={}
        for seconds,name in ((1,'return_1s_bps'),(10,'momentum_10s_bps')):
            indices=np.searchsorted(ts,ts-seconds*1000,side='right')-1
            ret=np.zeros(len(ts)); good=indices>=0
            ret[good]=(mid[good]/mid[indices[good]]-1)*1e4
            outputs[name]=ret
        ticks=np.zeros(len(ts)); ticks[1:]=np.diff(np.log(mid))*1e4
        left=np.searchsorted(ts,ts-30000,side='left')
        cumulative=np.r_[0.0,np.cumsum(ticks)]
        squared=np.r_[0.0,np.cumsum(ticks*ticks)]
        n=np.arange(len(ts))+1-left
        variance=np.maximum(0,(squared[1:]-squared[left])/n-((cumulative[1:]-cumulative[left])/n)**2)
        outputs['vol_30s_bps']=np.sqrt(variance)
        pieces.append(segment.with_columns([pl.Series(k,v) for k,v in outputs.items()]))
    return pl.concat(pieces)


def segment_coverage(frame):
    groups=frame.group_by('segment').agg(pl.len().alias('rows'),
        (pl.col('ts_ms').max()-pl.col('ts_ms').min()).alias('span_ms'))
    return {'segments':len(groups),'minimum_rows':int(groups['rows'].min()),
        'median_rows':float(groups['rows'].median()),'maximum_rows':int(groups['rows'].max()),
        'minimum_span_seconds':float(groups['span_ms'].min()/1000),
        'median_span_seconds':float(groups['span_ms'].median()/1000),
        'maximum_span_seconds':float(groups['span_ms'].max()/1000),
        'interpretation':'Numerical state and fitting labels reset at reconstruction segments. A long day does not imply a long continuous model window.'}

def load_archive(path,max_rows=None):
    book=Book(); records=[]
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            # Separate archive members require independent snapshots.
            book.ready=False; book.previous=None; book.sequence=None
            stream=archive.extractfile(member)
            for line in stream:
                if not line.strip():
                    continue
                row=json.loads(line)
                if isinstance(row.get('data'),list):
                    rows=[dict(item,action=row.get('action')) for item in row['data']]
                else:
                    rows=[row]
                for item in rows:
                    state=book.apply(item)
                    if state:
                        records.append(state)
                    if max_rows and len(records)>=max_rows:
                        return enrich(pl.DataFrame(records)),asdict(book.quality)
    return enrich(pl.DataFrame(records)),asdict(book.quality)

class Catalog:
    def __init__(self,data_dir,home):
        self.data_dir=Path(data_dir)
        self.cache=Path(home)/'cache'
        self.cache.mkdir(parents=True,exist_ok=True)

    def inventory(self):
        records=[]
        for p in sorted(self.data_dir.glob('*.tar.gz')):
            date=re.search(r'(\d{4}-\d{2}-\d{2})\.tar\.gz$',p.name)
            if date:
                st=p.stat()
                records.append({'date':date[1],'path':str(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns})
        if not records:
            raise FileNotFoundError(f'No dated L2 archives at {self.data_dir}')
        if len({r['date'] for r in records})!=len(records):
            raise ValueError('Multiple archives for a date; explicit disambiguation required')
        return records

    def plan(self,days=None,max_rows=None):
        inventory=self.inventory()
        if days:
            if days<10:
                raise ValueError('Use at least 10 distinct dates for a research plan')
            # Even sampling spans the supplied history, rather than only recent mornings.
            indices=np.unique(np.linspace(0,len(inventory)-1,min(days,len(inventory)),dtype=int))
            inventory=[inventory[i] for i in indices]
        if len(inventory)<10:
            raise ValueError('Need at least 10 daily archives')
        first=int(len(inventory)*0.6); second=int(len(inventory)*0.85)
        plan={'version':FEATURE_VERSION,'development':inventory[:first],'validation':inventory[first:second],'holdout':inventory[second:],'max_rows_per_day':max_rows,'sampled':bool(max_rows),'quantity_units':'OKX contracts; signals use dimensionless depth ratios','limitations':['L2 archive does not identify trades versus cancels','Sequence continuity unverifiable when seqId is absent','Historical funding not included; holding limit 300 seconds','OKX behavior must be revalidated on Hyperliquid']}
        plan['hash']=digest(plan)
        return plan

    def cached(self,record,max_rows=None):
        """Availability only; do not hydrate an archive or inspect its outcomes."""
        fingerprint=digest({'record':record,'version':FEATURE_VERSION,'max_rows':max_rows})
        return (self.cache/(fingerprint+'.parquet')).is_file() and (self.cache/(fingerprint+'.json')).is_file()

    def load(self,record,max_rows=None):
        source=Path(record['path']); st=source.stat()
        if st.st_size!=record['size'] or st.st_mtime_ns!=record['mtime_ns']:
            raise ValueError('Archive changed since campaign plan was frozen')
        fingerprint=digest({'record':record,'version':FEATURE_VERSION,'max_rows':max_rows})
        parquet=self.cache/(fingerprint+'.parquet'); meta=self.cache/(fingerprint+'.json')
        if parquet.exists() and meta.exists():
            return pl.read_parquet(parquet),json.loads(meta.read_text())
        frame,quality=load_archive(source,max_rows)
        # Concurrent prefetch and replay can meet on one archive. Each writer
        # owns a temporary path; only complete cache artifacts become visible.
        temporary=self.cache/(fingerprint+'.'+uuid.uuid4().hex+'.partial')
        temporary_meta=temporary.with_suffix('.json.partial')
        try:
            frame.write_parquet(temporary)
            temporary_meta.write_text(json.dumps(quality),encoding='utf-8')
            # Immutable keys: publish the first completed writer without
            # replacing a cache file another Windows reader may have mapped.
            for staged,target in ((temporary,parquet),(temporary_meta,meta)):
                try:
                    os.link(staged,target)
                except FileExistsError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)
            temporary_meta.unlink(missing_ok=True)
        return frame,quality

    def profile(self,record,max_rows=3000):
        frame,quality=self.load(record,max_rows)
        return {'date':record['date'],'rows':len(frame),'duration_seconds':(frame['ts_ms'][-1]-frame['ts_ms'][0])/1000,'median_interval_ms':float(np.median(np.diff(frame['ts_ms'].to_numpy()))),'median_spread_bps':float(frame['spread_bps'].median()),'quality':quality,'features':list(FEATURES),
            'segment_coverage':segment_coverage(frame),'bounded_profile':max_rows is not None}
