import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
import polars as pl
from quantdesk.data import Catalog
from quantdesk.ingestion import prepare_data
from quantdesk.store import Store


def test_concurrent_cache_writers_publish_complete_artifacts(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    path=raw/'BTC-L2-2026-01-01.tar.gz'
    rows=[{'action':'snapshot','ts':i*1000,'bids':[[100,10,1]],'asks':[[101,10,1]]} for i in range(20)]
    data=''.join(json.dumps(r)+'\n' for r in rows).encode()
    with tarfile.open(path,'w:gz') as stream:
        member=tarfile.TarInfo('book.jsonl'); member.size=len(data)
        stream.addfile(member,io.BytesIO(data))
    catalog=Catalog(raw,tmp_path); record=catalog.inventory()[0]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda _:catalog.load(record),range(3)))
    assert all(len(frame)==20 for frame,_ in results)
    frame,quality=catalog.load(record)
    assert len(frame)==20 and quality['messages']==20
    assert not list(catalog.cache.glob('*.partial'))


def test_preparation_reads_only_frozen_development_dates(tmp_path):
    store=Store(tmp_path)
    records=[{'date':f'2026-01-{i:02d}'} for i in range(1,6)]
    cid=store.campaign({'development':records[:3],'validation':records[3:4],'holdout':records[4:]},1000)
    class Fixture:
        reads=[]
        def load(self,record,*args):
            self.reads.append(record['date'])
            return pl.DataFrame({'ts_ms':[0,1000]}),{'has_sequence_ids':False}
    catalog=Fixture(); result=prepare_data(store,catalog,cid,2,2)
    assert result['status']=='COMPLETE' and len(result['completed'])==2
    assert set(catalog.reads)=={r['date'] for r in records[:2]}
    assert not any(r['continuity_verified'] for r in result['completed'])
    assert not result['holdout_accessed']


def test_bad_archives_remain_failures_not_fake_prepared_data(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}]},1000)
    class Fixture:
        def load(self,*args): raise OSError('unavailable archive')
    result=prepare_data(store,Fixture(),cid,1,1)
    assert result['status']=='INCOMPLETE' and result['completed']==[]
    assert result['failures'][0]['error_type']=='OSError'
