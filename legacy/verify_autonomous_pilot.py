"""Launch one real, finite acceptance campaign; invoking again never duplicates it."""
import json
import argparse
import time
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.launch import start_research
from quantdesk.research_map import research_map
from quantdesk.store import process_alive


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='Resume remaining cycles after a reviewed terminal failure; keeps the original budget and time boundary')
    args=parser.parse_args()
    settings=Settings(); store=Store(settings.home)
    directory=settings.home/'verification'/'autonomous-pilot'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'launch.json'
    if path.exists():
        launch=json.loads(path.read_text(encoding='utf-8'))
        if args.resume:
            history=launch.get('attempts',[{'pid':launch['pid'],'job_directory':launch['job_directory']}])
            if len(history)>=3:
                raise ValueError('Pilot operator recovery allowance exhausted; review evidence instead of repeatedly restarting')
            status=json.loads((Path(launch['job_directory'])/'status.json').read_text(encoding='utf-8'))
            if status['status']=='RUNNING' and process_alive(status['pid']):
                raise ValueError('Pilot process is still live; do not duplicate it')
            cid=launch['campaign']
            with store.connect() as db:
                completed=db.execute("SELECT COUNT(*) FROM cycles WHERE campaign=? AND status='COMPLETE'",(cid,)).fetchone()[0]
            remaining=launch['criteria']['real_company_cycles']-completed
            first=json.loads((Path(history[0]['job_directory'])/'status.json').read_text(encoding='utf-8'))
            hours=launch['criteria']['bounded_hours']-(time.time()-first['started_at'])/3600
            from quantdesk.operations import job_snapshot
            used_retries=sum(j.get('retries',0) or 0 for j in job_snapshot(store) if j['kind']=='autonomous' and j['campaign']==cid)
            if remaining<=0 or hours<=0:
                raise ValueError('Original pilot cycle or time boundary reached')
            store.event(cid,'operator','PILOT_RECOVERY',{'reason':'Reviewed terminal task failure; same immutable campaign and remaining budget',
                'remaining_cycles':remaining,'remaining_hours':hours,'campaign_budget_raised':False})
            process,job=start_research(store,settings,cid,remaining,hours,max(0,2-used_retries))
            history.append({'pid':process.pid,'job_directory':str(job)})
            launch.update(pid=process.pid,job_directory=str(job),attempts=history,status='RESUMED')
            path.write_text(json.dumps(launch,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({k:v for k,v in launch.items() if k not in {'before','global_memory_before'}},indent=2))
        return
    # Claim the launch before creating a campaign/process. A crash leaves a
    # reviewable incomplete receipt instead of allowing a duplicate launch.
    with path.open('x',encoding='utf-8') as receipt:
        json.dump({'status':'LAUNCHING','automatic_duplicate_launch':False},receipt)
    plan=store.get_campaign('2fb07460f9a0')['plan']
    if plan['sampled']:
        raise ValueError('Acceptance run needs the frozen full-day plan')
    cid=store.campaign(plan,750000)
    criteria={'real_company_cycles':2,'bounded_hours':4,'global_recovery_allowance':2,
        'research_only':True,'provider':'configured actual provider; no stubs',
        'required_evidence':['Executable submissions','Measured development assessments',
            'Independent full-day replayability','Grounded critique and lessons','Role and cycle terminal states',
            'Actual and ambiguous token accounting','Previously measured work reused where identical'],
        'not_proven_by_this_run':['Profitable alpha','Controlled learning benefit','Calibrated execution','Portfolio performance']}
    store.event(cid,'operator','ACCEPTANCE_RUN_PLANNED',criteria)
    process,job=start_research(store,settings,cid,2,4,2)
    launch={'status':'LAUNCHED','campaign':cid,'pid':process.pid,'job_directory':str(job),
        'token_limit':750000,'criteria':criteria,'before':research_map(store,cid),
        'global_memory_before':store.recall(10)}
    path.write_text(json.dumps(launch,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in launch.items() if k not in {'before','global_memory_before'}},indent=2))


if __name__=='__main__':
    main()
