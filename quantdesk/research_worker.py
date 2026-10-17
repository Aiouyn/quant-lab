"""Fresh-process entry point for dashboard research; no provider stubs."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from .settings import Settings
from .store import Store
from .data import Catalog
from .evaluation import Evaluator
from .agents import AgentRuntime,Desk
from .autonomy import Supervisor


def run_request(path):
    path=Path(path); request=json.loads(path.read_text(encoding='utf-8'))
    state={'campaign':request['campaign'],'pid':os.getpid(),'status':'RUNNING','started_at':time.time(),
        'requested_cycles':request['cycles'],'completed_cycles':0,'request_path':str(path),
        'runtime':'fresh child interpreter; current modules loaded before research',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    status_path=path.with_name('status.json')
    def persist():
        state['updated_at']=time.time()
        staged=status_path.with_suffix('.partial')
        staged.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        staged.replace(status_path)
    persist()
    try:
        configuration=dict(request['settings'])
        for field in ('home','data_dir'):
            configuration[field]=Path(configuration[field])
        settings=Settings(**configuration)
        store=Store(settings.home)
        # Check campaign before constructing a credentialed provider client.
        store.get_campaign(request['campaign'])
        catalog=Catalog(settings.data_dir,settings.home)
        evaluator=Evaluator(store,catalog,settings)
        runtime=AgentRuntime(store,settings)
        state.update(provider=runtime.provider,model=runtime.model)
        persist()
        desk=Desk(store,catalog,evaluator,settings,runtime)
        report=Supervisor(desk,store).run(request['campaign'],request['cycles'],request['hours'],request['max_retries'])
        state.update(status=report['status'],completed_cycles=report['completed_cycles'],
            completed_strategy_cohorts=report.get('completed_strategy_cohorts'),
            completed_mission_decisions=report.get('completed_mission_decisions'),
            completed_unspecified_cycles=report.get('completed_unspecified_cycles'),
            remaining_tokens=report['remaining_tokens'],report_path=report['report_path'],failures=report['failures'])
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            error_type=type(error).__name__,error=str(error)[:1000])
        raise
    finally:
        persist()
    return state


if __name__=='__main__':
    run_request(sys.argv[1])
