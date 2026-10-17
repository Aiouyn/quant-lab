"""Bounded parallel preparation of development archives with quality receipts."""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor,as_completed


def prepare_data(store,catalog,cid,limit=None,workers=2):
    if not 1<=workers<=4 or (limit is not None and limit<1):
        raise ValueError('Preparation workers 1..4; positive date limit')
    campaign=store.get_campaign(cid)
    if campaign['status']=='SEALED':
        raise ValueError('Sealed campaign cannot expose further development data')
    records=campaign['plan']['development'][:limit]
    ident=uuid.uuid4().hex[:12]
    directory=store.home/'ingestion'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/(ident+'.json')
    state={'id':ident,'campaign':cid,'pid':os.getpid(),'status':'RUNNING',
           'started_at':time.time(),'requested_dates':len(records),'workers':workers,
           'completed':[],'failures':[],'holdout_accessed':False,
           'purpose':'Development feature cache and data-quality audit; no trading result'}

    def persist():
        state['updated_at']=time.time()
        temporary=path.with_suffix('.partial')
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(path)

    def load(record):
        frame,quality=catalog.load(record,campaign['plan'].get('max_rows_per_day'))
        return {'date':record['date'],'rows':len(frame),
            'duration_seconds':(int(frame['ts_ms'][-1])-int(frame['ts_ms'][0]))/1000,
            'quality':quality,'continuity_verified':bool(quality.get('has_sequence_ids')) and not quality.get('sequence_gaps',0)}

    with store.lease('ingestion',seconds=86400):
        store.expose_dates(cid,records,'development')
        persist()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(load,r):r for r in records}
                for future in as_completed(futures):
                    record=futures[future]
                    try:
                        receipt=future.result()
                        state['completed'].append(receipt)
                        store.event(cid,'ingestion','ARCHIVE_PREPARED',receipt)
                        print(f"Prepared {receipt['date']}: {receipt['rows']} rows ({len(state['completed'])}/{len(records)})",flush=True)
                    except Exception as error:
                        state['failures'].append({'date':record['date'],'error_type':type(error).__name__,'error':str(error)})
                    persist()
            state['status']='COMPLETE' if not state['failures'] else 'INCOMPLETE'
        except BaseException:
            state['status']='INTERRUPTED'
            raise
        finally:
            state['completed'].sort(key=lambda r:r['date'])
            persist()
    return dict(state,report_path=str(path))
