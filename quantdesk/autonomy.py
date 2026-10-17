"""Bounded autonomous research with durable checkpoints and selective recovery."""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from .agents import AgentTaskFailure
from .control import ResearchTimeLimit


class Supervisor:
    def __init__(self,desk,store,clock=time.monotonic,sleep=time.sleep):
        self.desk,self.store,self.clock,self.sleep=desk,store,clock,sleep

    def remaining_tokens(self,cid):
        with self.store.connect() as db:
            used=db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests WHERE campaign=?',(cid,)).fetchone()[0]
        limit=self.store.get_campaign(cid)['token_limit']
        return None if limit is None else limit-used

    def run(self,cid,cycles=10,hours=8,max_retries=2,cooldown_seconds=5):
        if not 1<=cycles<=100 or not 0<hours<=72 or not 0<=max_retries<=5 or not 0<=cooldown_seconds<=300:
            raise ValueError('Cycles 1..100, hours (0,72], retries 0..5, cooldown 0..300 seconds')
        if self.store.get_campaign(cid)['status']=='SEALED':
            raise ValueError('Sealed campaign cannot resume research')
        ident=uuid.uuid4().hex[:12]
        directory=self.store.home/'operations'; directory.mkdir(parents=True,exist_ok=True)
        path=directory/(ident+'.json')
        deadline=self.clock()+hours*3600
        state={'id':ident,'campaign':cid,'pid':os.getpid(),'status':'RUNNING',
            'started_at':datetime.now(timezone.utc).isoformat(),'requested_cycles':cycles,
            'completed_cycles':0,'completed_strategy_cohorts':0,'completed_mission_decisions':0,'completed_unspecified_cycles':0,'retries':0,'failures':[],
            'time_limit_hours':hours,'deadline_policy':'Cooperative checks before provider calls, after their accounting, before tools and between model/replay archive units; current I/O or numerical unit may finish after the limit',
            'holdout_policy':'Research only; sealing and portfolio approval remain explicit'}

        def persist():
            state['remaining_tokens']=self.remaining_tokens(cid)
            state['updated_at']=datetime.now(timezone.utc).isoformat()
            partial=path.with_suffix('.partial')
            partial.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
            partial.replace(path)

        with self.store.lease('research',seconds=hours*3600+3600),self.store.research_deadline(cid,deadline,self.clock):
            persist()
            failures=0
            try:
                while state['completed_cycles']<cycles:
                    if self.store.get_campaign(cid)['status']=='STOP_REQUESTED':
                        state['status']='STOPPED'; break
                    if self.clock()>=deadline:
                        state['status']='TIME_LIMIT'; break
                    remaining=self.remaining_tokens(cid)
                    if remaining is not None and remaining<16000:
                        state['status']='BUDGET_EXHAUSTED'; break
                    self.store.set_status(cid,'RUNNING')
                    try:
                        result=self.desk.run(cid,1,own_lease=False)
                    except ResearchTimeLimit:
                        state['status']='TIME_LIMIT'; persist(); break
                    except AgentTaskFailure as error:
                        failures+=1
                        state['failures'].append({'role':error.role,'error_type':error.error_type,'retryable':error.retryable})
                        if not error.retryable or failures>max_retries:
                            state['status']='FAILED'; persist(); break
                        state['retries']+=1
                        delay=min(60,cooldown_seconds*2**(failures-1))
                        self.store.set_status(cid,'RUNNING')
                        self.store.event(cid,'supervisor','RECOVERY_WAIT',{'run':ident,'seconds':delay,'role':error.role,'reason':error.error_type})
                        persist()
                        wait_end=min(deadline,self.clock()+delay)
                        while self.clock()<wait_end and self.store.get_campaign(cid)['status']!='STOP_REQUESTED':
                            self.sleep(min(1,wait_end-self.clock()))
                        continue
                    except Exception as error:
                        state['failures'].append({'error_type':type(error).__name__,'retryable':False})
                        state['status']='BUDGET_EXHAUSTED' if 'token budget exhausted' in str(error).lower() else 'FAILED'
                        persist(); break
                    if result.get('status')=='STOPPED':
                        state['status']='STOPPED'; break
                    state['completed_cycles']+=1
                    outcomes=result.get('cycle_outcomes',[])
                    if len(outcomes)==1 and outcomes[0].get('kind')=='STRATEGY_COHORT':
                        state['completed_strategy_cohorts']+=1
                    elif len(outcomes)==1 and outcomes[0].get('kind')=='MISSION_STOP':
                        state['completed_mission_decisions']+=1
                    else:
                        state['completed_unspecified_cycles']+=1
                    # Recovery allowance is finite across the whole run, not reset
                    # after success; intermittent outages cannot consume unlimited tokens.
                    self.store.set_status(cid,'RUNNING')
                    persist()
                else:
                    state['status']='COMPLETE'
            except BaseException:
                state['status']='INTERRUPTED'
                raise
            finally:
                self.store.set_status(cid,'READY' if state['status']=='COMPLETE' else 'STOPPED')
                persist()
                self.store.event(cid,'supervisor','AUTONOMOUS_RUN_FINISHED',state)
        return dict(state,report_path=str(path))
