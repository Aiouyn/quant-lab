"""Loopback, read-focused dashboard with explicit bounded research starts."""
import json
import threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from .store import process_alive

def serve(store,catalog,evaluator,settings,port=8888):
    state={'running':False,'error':None,'campaign':None}
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def respond(self,payload,status=200):
            body=json.dumps(payload,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path=='/api/status':
                status=store.status()
                with store.connect() as db:
                    import time
                    lease=db.execute("SELECT pid,expires FROM leases WHERE name='research'").fetchone()
                    active=bool(lease and (process_alive(lease['pid']) if lease['pid'] else lease['expires']>time.time()))
                worker=dict(state,running=state['running'] or active)
                from .operations import job_snapshot
                self.respond(status|{'worker':worker,'jobs':job_snapshot(store),
                    'capabilities':{'research_runtime':'fresh child process','development_screen_before_confirmation':True}})
            elif self.path=='/api/candidates':
                self.respond(store.candidates())
            elif self.path in {'/','/index.html'}:
                body=(Path(__file__).parent/'dashboard.html').read_bytes()
                self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            else:
                self.respond({'error':'Not found'},404)

        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):
                self.respond({'error':'Cross-origin action rejected'},403); return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.respond({'error':'application/json required'},415); return
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<10000:
                    raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length))
                if self.path=='/api/plan':
                    days=data.get('days'); rows=data.get('rows_per_day')
                    if days is not None:
                        days=int(days)
                    if rows is not None:
                        rows=int(rows)
                        if rows<100:
                            raise ValueError('At least 100 rows per day')
                    budget=data.get('token_budget',settings.max_campaign_tokens)
                    if budget is not None and (isinstance(budget,bool) or not isinstance(budget,int) or budget<1000):
                        raise ValueError('Token budget out of range')
                    plan=catalog.plan(days,rows); cid=store.campaign(plan,budget)
                    self.respond({'campaign':cid,'dates':{p:len(plan[p]) for p in ('development','validation','holdout')},'sampled':plan['sampled'],'token_limit':budget})
                elif self.path=='/api/stop':
                    cid=str(data['campaign'])
                    from .operations import job_snapshot
                    active_job=any(j['running'] and j.get('campaign')==cid for j in job_snapshot(store))
                    if store.get_campaign(cid)['status']!='RUNNING' and not active_job:
                        raise ValueError('Campaign is not currently running')
                    store.set_status(cid,'STOP_REQUESTED')
                    self.respond({'status':'STOP_REQUESTED','effective':'After the current provider call or archive replay'})
                elif self.path=='/api/research':
                    cid=str(data['campaign']); cycles=int(data.get('cycles',1))
                    hours=float(data.get('hours',8)); max_retries=int(data.get('max_retries',2))
                    if not 1<=cycles<=10:
                        raise ValueError('Start 1..10 bounded cycles')
                    with lock:
                        if state['running']:
                            self.respond({'error':'Research already running'},409); return
                        campaign=store.get_campaign(cid)
                        if campaign['status']=='SEALED':
                            raise ValueError('Campaign is sealed')
                        state.update(running=True,error=None,campaign=cid)
                    try:
                        from .launch import start_research
                        process,directory=start_research(store,settings,cid,cycles,hours,max_retries)
                    except Exception:
                        state['running']=False
                        raise
                    state.update(pid=process.pid,job_directory=str(directory))
                    def worker():
                        exit_code=process.wait()
                        path=directory/'status.json'
                        try:
                            report=json.loads(path.read_text(encoding='utf-8'))
                            if report['status']!='COMPLETE':
                                state['error']=report.get('error') or f"Research ended: {report['status']}; inspect {directory}"
                        except (OSError,ValueError,KeyError):
                            state['error']=f'Research process exited ({exit_code}); inspect {directory}'
                        finally:
                            state['running']=False
                    threading.Thread(target=worker,daemon=True).start()
                    self.respond({'started':True,'campaign':cid,'cycles':cycles,'pid':process.pid,'job_directory':str(directory),'runtime':'fresh child process'})
                else:
                    self.respond({'error':'Not found'},404)
            except Exception as error:
                self.respond({'error':str(error)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Quant Lab evidence desk: http://127.0.0.1:{port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
