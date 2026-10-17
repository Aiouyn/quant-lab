"""Read-only job snapshots; a persisted RUNNING string alone is not liveness."""
import json
from .store import process_alive


def job_snapshot(store):
    jobs=[]
    patterns=[('ingestion','ingestion/*.json'),('autonomous','operations/*.json'),
              ('research_job','jobs/*/status.json'),
              ('venue_capture','market_data/*/status.json'),
              ('feature_verification','verification/feature-agent/state.json'),
              ('feature_replay','verification/feature-replay/state.json'),
              ('paper_forward','paper/*/status.json')]
    for kind,pattern in patterns:
        paths=sorted(store.home.glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True)[:10]
        for path in paths:
            try:
                value=json.loads(path.read_text(encoding='utf-8'))
                pid=value.get('pid')
                declared=value.get('status','UNKNOWN')
                running=declared=='RUNNING' and bool(pid and process_alive(pid))
                job={'kind':kind,'status':declared if declared!='RUNNING' or running else 'INTERRUPTED',
                    'running':running,'pid':pid,'campaign':value.get('campaign'),
                    'path':str(path),'updated_at':value.get('updated_at',value.get('updated_ms'))}
                if kind=='ingestion':
                    job.update(completed=len(value.get('completed',[])),requested=value.get('requested_dates'),failures=len(value.get('failures',[])))
                if kind in {'autonomous','research_job'}:
                    job.update(completed=value.get('completed_cycles'),requested=value.get('requested_cycles'),retries=value.get('retries'),remaining_tokens=value.get('remaining_tokens'))
                if kind=='venue_capture':
                    job.update(book_mode=value.get('book_mode','slow_20_levels'),counts=value.get('counts',{}),errors=value.get('errors'),coverage_verified=False)
                jobs.append(job)
            except (OSError,ValueError,TypeError):
                continue
    path=store.home/'verification'/'fitted-upgrade'/'state.json'
    if path.exists():
        try:
            value=json.loads(path.read_text(encoding='utf-8'))
            runner_path=path.with_name('runner.json')
            runner=json.loads(runner_path.read_text(encoding='utf-8')) if runner_path.exists() else {}
            pid=value.get('pid') or runner.get('pid')
            running=value['status']=='RUNNING' and bool(pid and process_alive(pid))
            jobs.append({'kind':'fitted_verification','campaign':value.get('campaign'),'pid':pid,
                'running':running,'status':value['status'] if value['status']!='RUNNING' or running else 'INTERRUPTED',
                'completed':len(value.get('completed_outer_dates',[])),'requested':5,'path':str(path)})
        except (OSError,ValueError,TypeError):
            pass
    return jobs
