"""Durable task, evidence, lineage, and API budget accounting."""
import hashlib
import ast
import json
import sqlite3
import time
import uuid
import os
from contextlib import contextmanager
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Store:
    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir(parents=True,exist_ok=True)
        self.path = self.home / 'desk.sqlite3'
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, created REAL, plan TEXT, status TEXT, token_limit INTEGER);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created REAL, campaign TEXT, agent TEXT, kind TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, created REAL, campaign TEXT, fingerprint TEXT UNIQUE, title TEXT, spec TEXT, code TEXT, status TEXT, report TEXT);
                CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, created REAL, campaign TEXT, candidate TEXT, kind TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, campaign TEXT, agent TEXT, status TEXT, reserved INTEGER, actual INTEGER, created REAL);
                CREATE TABLE IF NOT EXISTS request_tasks(request_id TEXT PRIMARY KEY, task_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_budgets(task_id TEXT PRIMARY KEY, campaign TEXT NOT NULL, token_limit INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS research_decisions(id TEXT PRIMARY KEY, campaign TEXT NOT NULL, cycle TEXT, task TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL);
                CREATE TABLE IF NOT EXISTS mission_lessons(decision_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL);
                CREATE TABLE IF NOT EXISTS venue_studies(id TEXT PRIMARY KEY, study_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, created REAL);
                CREATE TABLE IF NOT EXISTS holdouts(plan_hash TEXT PRIMARY KEY, cohort TEXT, status TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS holdout_dates(source_hash TEXT PRIMARY KEY, plan_hash TEXT);
                CREATE TABLE IF NOT EXISTS data_exposures(source_hash TEXT PRIMARY KEY, date TEXT, phase TEXT, campaign TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS leases(name TEXT PRIMARY KEY, owner TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, campaign TEXT, agent TEXT, status TEXT, payload TEXT, result TEXT, updated REAL);
                CREATE TABLE IF NOT EXISTS cycles(id TEXT PRIMARY KEY, campaign TEXT, status TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS tool_cache(key TEXT PRIMARY KEY, result TEXT);
                CREATE TABLE IF NOT EXISTS forward_plans(id TEXT PRIMARY KEY, candidate TEXT, identity TEXT, created REAL, payload TEXT);
                CREATE TABLE IF NOT EXISTS forward_current(identity TEXT PRIMARY KEY, plan_id TEXT);
            ''')
            if 'pid' not in {r[1] for r in db.execute('PRAGMA table_info(leases)')}:
                db.execute('ALTER TABLE leases ADD COLUMN pid INTEGER')
            task_columns={r[1] for r in db.execute('PRAGMA table_info(tasks)')}
            if 'pid' not in task_columns:
                db.execute('ALTER TABLE tasks ADD COLUMN pid INTEGER')
            if 'started' not in task_columns:
                db.execute('ALTER TABLE tasks ADD COLUMN started REAL')
                db.execute('UPDATE tasks SET started=updated')
            if not db.execute("SELECT 1 FROM migrations WHERE name='date_exposure_v1'").fetchone():
                # Earlier versions did not log exact reads. Conservatively
                # quarantine their planned research dates from future holdouts.
                for row in db.execute('SELECT id,plan FROM campaigns').fetchall():
                    plan=json.loads(row['plan'])
                    for phase in ('development','validation'):
                        for record in plan.get(phase,[]):
                            db.execute('INSERT OR IGNORE INTO data_exposures VALUES(?,?,?,?,?)',
                                (digest({'date':record['date'],'instrument':'BTC-USDT-SWAP'}),record['date'],phase,row['id'],time.time()))
                db.execute("INSERT INTO migrations VALUES('date_exposure_v1')")

    def expose_dates(self,cid,records,phase):
        if phase not in {'development','validation'}:
            raise ValueError('Only research dates can be exposed')
        with self.connect() as db:
            db.executemany('INSERT OR IGNORE INTO data_exposures VALUES(?,?,?,?,?)',
                [(digest({'date':r['date'],'instrument':'BTC-USDT-SWAP'}),r['date'],phase,cid,time.time()) for r in records])

    def cached_tool(self,key):
        with self.connect() as db:
            row=db.execute('SELECT result FROM tool_cache WHERE key=?',(key,)).fetchone()
        return json.loads(row['result']) if row else None

    def cache_tool(self,key,result):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO tool_cache VALUES(?,?)',(key,json.dumps(result,allow_nan=False)))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path,timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def campaign(self, plan, token_limit):
        if token_limit is not None and (isinstance(token_limit,bool) or not isinstance(token_limit,int) or token_limit<=0):
            raise ValueError('Campaign token limit must be a positive integer or None for no local cap')
        cid = uuid.uuid4().hex[:12]
        with self.connect() as db:
            db.execute('INSERT INTO campaigns VALUES(?,?,?,?,?)',(cid,time.time(),json.dumps(plan),'READY',token_limit))
        return cid

    def get_campaign(self, cid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM campaigns WHERE id=?',(cid,)).fetchone()
        if row is None:
            raise ValueError('Unknown campaign')
        return dict(row) | {'plan':json.loads(row['plan'])}

    @contextmanager
    def research_deadline(self,cid,deadline,clock=time.monotonic):
        """Process-local cooperative boundary; does not change durable stop intent."""
        boundaries=getattr(self,'_research_boundaries',{})
        self._research_boundaries=boundaries
        previous=boundaries.get(cid)
        boundaries[cid]=(deadline,clock)
        try:
            yield
        finally:
            if previous is None:
                boundaries.pop(cid,None)
            else:
                boundaries[cid]=previous

    def check_research_control(self,cid):
        from .control import ResearchStopped,ResearchTimeLimit
        if self.get_campaign(cid)['status']=='STOP_REQUESTED':
            raise ResearchStopped('User stop requested; current unit completed')
        boundary=getattr(self,'_research_boundaries',{}).get(cid)
        if boundary and boundary[1]()>=boundary[0]:
            raise ResearchTimeLimit('Supervisor time limit reached; current unit preserved, no further research unit will start')

    def event(self,cid,agent,kind,payload):
        with self.connect() as db:
            db.execute('INSERT INTO events(created,campaign,agent,kind,payload) VALUES(?,?,?,?,?)',(time.time(),cid,agent,kind,json.dumps(payload,allow_nan=False)))

    def set_status(self,cid,status):
        with self.connect() as db:
            db.execute('UPDATE campaigns SET status=? WHERE id=?',(status,cid))

    def revise_budget(self,cid,tokens,reason):
        """Explicit budget revision; supervisors never call this automatically."""
        if isinstance(tokens,bool) or not isinstance(tokens,int) or tokens<=0 or not isinstance(reason,str) or not reason.strip():
            raise ValueError('A positive integer budget and an explicit reason are required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT token_limit,status FROM campaigns WHERE id=?',(cid,)).fetchone()
            if not row or row['status']=='SEALED':
                raise ValueError('Budget revision requires an existing unsealed campaign')
            used=db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests WHERE campaign=?',(cid,)).fetchone()[0]
            if tokens<used:
                raise ValueError('Budget cannot be reduced below charged and reserved usage')
            payload={'old_limit':row['token_limit'],'new_limit':tokens,'charged_or_reserved':used,'reason':reason,'automatic':False}
            db.execute('UPDATE campaigns SET token_limit=? WHERE id=?',(tokens,cid))
            db.execute('INSERT INTO events(created,campaign,agent,kind,payload) VALUES(?,?,?,?,?)',
                (time.time(),cid,'operator','BUDGET_REVISED',json.dumps(payload)))
        return payload

    def reserve(self,cid,agent,tokens,task_id=None):
        if isinstance(tokens,bool) or not isinstance(tokens,int) or tokens<=0:
            raise ValueError('Request reservation must be a positive integer')
        request = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            limit = db.execute('SELECT token_limit FROM campaigns WHERE id=?',(cid,)).fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0) FROM requests WHERE campaign=?",(cid,)).fetchone()[0]
            if limit is not None and used+tokens > limit:
                raise RuntimeError(f'Campaign token budget exhausted: {used}/{limit}; next reservation {tokens}')
            if task_id:
                task=db.execute('SELECT campaign,agent FROM tasks WHERE id=?',(task_id,)).fetchone()
                if not task or task['campaign']!=cid or task['agent']!=agent:
                    raise ValueError('Request task belongs to a different campaign/role or is missing')
                budget=db.execute('SELECT token_limit FROM task_budgets WHERE task_id=?',(task_id,)).fetchone()
                if budget:
                    task_used=db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests JOIN request_tasks ON requests.id=request_tasks.request_id WHERE request_tasks.task_id=?',(task_id,)).fetchone()[0]
                    if task_used+tokens>budget['token_limit']:
                        raise RuntimeError(f'Task token budget exhausted: {task_used}/{budget["token_limit"]}; next reservation {tokens}. Checkpoints retained; no automatic limit increase.')
            db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?)',(request,cid,agent,'RESERVED',tokens,None,time.time()))
            if task_id:
                db.execute('INSERT INTO request_tasks VALUES(?,?)',(request,task_id))
        return request

    def start_cycle(self,cid):
        with self.connect() as db:
            row=db.execute("SELECT id FROM cycles WHERE campaign=? AND status IN ('ACTIVE','PAUSED','FAILED') ORDER BY created DESC LIMIT 1",(cid,)).fetchone()
            if row:
                db.execute("UPDATE cycles SET status='ACTIVE' WHERE id=?",(row['id'],))
                return row['id']
            ident=uuid.uuid4().hex[:12]
            db.execute('INSERT INTO cycles VALUES(?,?,?,?)',(ident,cid,'ACTIVE',time.time()))
        return ident

    def complete_cycle(self,ident,outcome='STRATEGY_COHORT'):
        if outcome not in {'STRATEGY_COHORT','MISSION_STOP'}:
            raise ValueError('Unknown cycle outcome')
        with self.connect() as db:
            db.execute('UPDATE cycles SET status=? WHERE id=?',('COMPLETE' if outcome=='STRATEGY_COHORT' else 'DECISION_COMPLETE',ident))

    def suspend_cycle(self,cid,status,reason):
        if status not in {'PAUSED','FAILED'}:
            raise ValueError('Interrupted cycle status must be PAUSED or FAILED')
        with self.connect() as db:
            rows=db.execute("SELECT id FROM cycles WHERE campaign=? AND status='ACTIVE'",(cid,)).fetchall()
            db.execute("UPDATE cycles SET status=? WHERE campaign=? AND status='ACTIVE'",(status,cid))
        for row in rows:
            self.event(cid,'desk','CYCLE_'+status,{'cycle_id':row['id'],'reason':reason,'economic_rejection':False})

    def abandon_cycle(self,cid,reason):
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('An explicit reason is required; evidence is preserved')
        with self.lease('research'):
            if self.get_campaign(cid)['status']=='SEALED':
                raise ValueError('Sealed research cannot change direction')
            with self.connect() as db:
                row=db.execute("SELECT id FROM cycles WHERE campaign=? AND status IN ('ACTIVE','PAUSED','FAILED') ORDER BY created DESC LIMIT 1",(cid,)).fetchone()
                if row is None:
                    raise ValueError('No unfinished cycle to abandon')
                tasks=[dict(r) for r in db.execute('SELECT id,agent,status,pid,result FROM tasks WHERE campaign=? AND id LIKE ?',(cid,row['id']+':%'))]
                if any(t['status']=='RUNNING' and t['pid'] and process_alive(t['pid']) for t in tasks):
                    raise ValueError('A live task still owns this cycle')
                db.execute("UPDATE cycles SET status='ABANDONED' WHERE id=?",(row['id'],))
            evidence={'cycle_id':row['id'],'reason':reason,'tasks':[{k:t[k] for k in ('id','agent','status')} for t in tasks],
                'interpretation':'Operational research direction change; prior code, tasks, requests and measurements retained. No economic rejection or token reset.'}
            self.event(cid,'operator','CYCLE_ABANDONED',evidence)
            self.memory(cid,'','OPERATIONAL_DIRECTION_CHANGE',evidence)
        return evidence

    def start_task(self,ident,cid,agent,payload):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM tasks WHERE id=?',(ident,)).fetchone()
            if row and (row['campaign']!=cid or row['agent']!=agent):
                raise ValueError('Task checkpoint belongs to a different campaign/role')
            budget=payload.get('task_token_limit')
            if budget is not None:
                if isinstance(budget,bool) or not isinstance(budget,int) or budget<=0:
                    raise ValueError('Task token limit must be a positive integer')
                frozen=db.execute('SELECT token_limit FROM task_budgets WHERE task_id=?',(ident,)).fetchone()
                if frozen and frozen['token_limit']!=budget:
                    raise ValueError('Task token limit is immutable; resumption cannot reset or raise it')
                if not frozen:
                    if row:
                        raise ValueError('Cannot attach a new task budget to an older unattributed checkpoint; use a distinct reviewed task')
                    db.execute('INSERT INTO task_budgets VALUES(?,?,?)',(ident,cid,budget))
            if row and row['status']=='COMPLETE':
                return json.loads(row['result'])
            if row and row['status']=='RUNNING' and row['pid'] and process_alive(row['pid']):
                raise RuntimeError('Task checkpoint is already owned by a live process')
            now=time.time()
            db.execute('INSERT OR REPLACE INTO tasks(id,campaign,agent,status,payload,result,updated,pid,started) VALUES(?,?,?,?,?,?,?,?,?)',
                (ident,cid,agent,'RUNNING',json.dumps(payload),None,now,os.getpid(),(row['started'] or row['updated']) if row else now))
        return None

    def fail_task(self,ident,error,status='FAILED',after=0):
        with self.connect() as db:
            db.execute('UPDATE tasks SET status=?,result=?,updated=?,pid=NULL WHERE id=? AND status!=? AND pid=? AND updated>=?',
                (status,json.dumps({'error_type':type(error).__name__}),time.time(),ident,'COMPLETE',os.getpid(),after))

    def finish_task(self,ident,result):
        with self.connect() as db:
            db.execute("UPDATE tasks SET status='COMPLETE',result=?,updated=?,pid=NULL WHERE id=?",(json.dumps(result),time.time(),ident))

    def finish_request(self,request,actual,status='COMPLETE'):
        # An ambiguous failed request remains charged at its reservation.
        with self.connect() as db:
            db.execute('UPDATE requests SET actual=?,status=? WHERE id=?',(actual,status,request))

    def candidate(self,cid,spec,code):
        try:
            semantic_code=ast.dump(ast.parse(code),include_attributes=False)
        except SyntaxError:
            semantic_code=code.strip()
        fingerprint = digest({'campaign':cid,'code':semantic_code,'spec':{k:v for k,v in spec.items() if k not in {'title','rationale','parent_ids','falsification'}}})
        ident = fingerprint[:16]
        with self.connect() as db:
            existing = db.execute('SELECT * FROM candidates WHERE fingerprint=?',(fingerprint,)).fetchone()
            if existing:
                return existing['id'],False
            db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)',(ident,time.time(),cid,fingerprint,spec['title'],json.dumps(spec),code,'PROPOSED',None))
        self.event(cid,'engineer','CANDIDATE_REGISTERED',{'id':ident,'title':spec['title']})
        return ident,True

    def candidates(self,cid=None,status=None):
        sql,args = 'SELECT * FROM candidates WHERE 1=1',[]
        if cid:
            sql+=' AND campaign=?'; args.append(cid)
        if status:
            sql+=' AND status=?'; args.append(status)
        with self.connect() as db:
            rows = db.execute(sql+' ORDER BY created',args).fetchall()
        return [dict(r)|{'spec':json.loads(r['spec']),'report':json.loads(r['report']) if r['report'] else None} for r in rows]

    def get_candidate(self,ident):
        rows = [c for c in self.candidates() if c['id']==ident]
        if not rows:
            raise ValueError('Unknown candidate')
        return rows[0]

    def record_report(self,ident,status,report):
        with self.connect() as db:
            db.execute('UPDATE candidates SET status=?,report=? WHERE id=?',(status,json.dumps(report,allow_nan=False),ident))

    def memory(self,cid,candidate,kind,payload):
        with self.connect() as db:
            db.execute('INSERT INTO memories(created,campaign,candidate,kind,payload) VALUES(?,?,?,?,?)',(time.time(),cid,candidate,kind,json.dumps(payload,allow_nan=False)))

    def recall(self,limit=20):
        with self.connect() as db:
            # Frequent feature probes must not evict the lessons they support.
            important=db.execute("SELECT * FROM memories WHERE kind IN ('LESSON','MISSION_LESSON','EVIDENCE','WALK_FORWARD_EXPERIMENT','DEVELOPMENT_SCREEN','PREDICTION_DIAGNOSTIC','PAPER_OBSERVATION','VENUE_OBSERVATION','VENUE_EXECUTION_STUDY','VENUE_REPLAY','OPERATIONAL_DIRECTION_CHANGE','OPERATIONAL_ACCEPTANCE_FAILURE') ORDER BY CASE WHEN kind IN ('LESSON','MISSION_LESSON') THEN 0 ELSE 1 END,id DESC LIMIT ?",(max(1,limit//2),)).fetchall()
            recent=db.execute('SELECT * FROM memories ORDER BY id DESC LIMIT ?',(max(limit*4,20),)).fetchall()
            rows=[]; seen_ids=set(); seen_captures=set()
            for row in important+recent:
                if row['id'] in seen_ids: continue
                if row['kind']=='VENUE_EXECUTION_STUDY':
                    capture=json.loads(row['payload']).get('capture_id')
                    if capture and capture in seen_captures: continue
                    seen_captures.add(capture)
                rows.append(row);seen_ids.add(row['id'])
                if len(rows)>=limit: break
        return [dict(r)|{'payload':compact_memory(json.loads(r['payload']))} for r in rows]

    def reserve_holdout(self,plan_hash,cohort,dates=()):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                for record in dates:
                    source_hash=digest({'date':record['date'],'instrument':'BTC-USDT-SWAP'})
                    if db.execute('SELECT 1 FROM data_exposures WHERE source_hash=?',(source_hash,)).fetchone():
                        raise RuntimeError(f'Holdout date {record["date"]} was already exposed to research; choose genuinely untouched dates')
                db.execute('INSERT INTO holdouts VALUES(?,?,?,?)',(plan_hash,json.dumps(sorted(cohort)),'CONSUMED',time.time()))
                for record in dates:
                    # Renaming/copying/updating an archive cannot make its calendar
                    # observations a fresh financial test period.
                    db.execute('INSERT INTO holdout_dates VALUES(?,?)',(digest({'date':record['date'],'instrument':'BTC-USDT-SWAP'}),plan_hash))
            except sqlite3.IntegrityError:
                db.rollback()
                raise RuntimeError('This holdout has already been consumed. Create a new campaign using genuinely new dates.') from None

    @contextmanager
    def lease(self,name,seconds=3600):
        owner = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # A long replay can outlive its estimate. Never steal a live process's
            # lease merely because a wall-clock estimate has expired.
            for row in db.execute('SELECT name,pid,expires FROM leases').fetchall():
                dead=bool(row['pid'] and not process_alive(row['pid']))
                expired_unowned=not row['pid'] and row['expires']<time.time()
                if dead or expired_unowned:
                    db.execute('DELETE FROM leases WHERE name=?',(row['name'],))
            try:
                db.execute('INSERT INTO leases(name,owner,expires,pid) VALUES(?,?,?,?)',(name,owner,time.time()+seconds,os.getpid()))
            except sqlite3.IntegrityError:
                raise RuntimeError('Another desk process owns this research operation') from None
        try:
            yield
        finally:
            with self.connect() as db:
                db.execute('DELETE FROM leases WHERE name=? AND owner=?',(name,owner))

    def status(self):
        with self.connect() as db:
            campaigns = [dict(r) for r in db.execute('SELECT id,status,created,token_limit FROM campaigns ORDER BY created DESC')]
            counts = {r[0]:r[1] for r in db.execute('SELECT status,COUNT(*) FROM candidates GROUP BY status')}
            usage = [dict(r) for r in db.execute('SELECT campaign,agent,COUNT(*) calls,SUM(COALESCE(actual,reserved)) tokens FROM requests GROUP BY campaign,agent')]
            tasks=[dict(r) for r in db.execute('SELECT id,campaign,agent,status,updated,pid FROM tasks ORDER BY updated DESC LIMIT 30')]
            events = [dict(r)|{'payload':json.loads(r['payload'])} for r in db.execute('SELECT * FROM events ORDER BY id DESC LIMIT 50')]
        for task in tasks:
            task['running']=task['status']=='RUNNING' and bool(task['pid'] and process_alive(task['pid']))
            if task['status']=='RUNNING' and task['pid'] and not task['running']:
                task['declared_status']='RUNNING'
                task['status']='INTERRUPTED'
        return {'campaigns':campaigns,'candidates':counts,'usage':usage,'tasks':tasks,'events':events}

def compact_memory(value):
    """Keep complete evidence on disk, retrieve outcomes instead of whole replay tables."""
    if isinstance(value,dict):
        result={k:compact_memory(v) for k,v in value.items() if k not in {'daily_results','trade_rows','code'} and not (k=='days' and isinstance(v,list))}
        if isinstance(value.get('days'),list):
            result['evaluated_dates']=[d.get('date') for d in value['days'] if isinstance(d,dict)]
        if isinstance(value.get('daily_results'),list):
            result['evaluated_dates']=[d.get('date') for d in value['daily_results'] if isinstance(d,dict)]
        return result
    if isinstance(value,list):
        return [compact_memory(v) for v in value[:30]]
    return value

def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return ctypes.get_last_error()==5 # Access denied is not proof of death.
        try:
            status=wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(status))) and status.value==259
        finally:
            kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
