"""No local aggregate cap still preserves usage, task limits and failure controls."""
import json
import threading
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
import pytest
from quantdesk.store import Store
from quantdesk.settings import Settings
from quantdesk.cli import parser
from quantdesk.autonomy import Supervisor
from tests.test_autonomy import setup


def test_uncapped_usage_survives_reopen_without_erasing_finite_campaign(tmp_path):
    store=Store(tmp_path); old=store.campaign({},750000); cid=store.campaign({},None)
    first=store.reserve(cid,'scientist',800000)
    store.finish_request(first,None,'FAILED')
    reopened=Store(tmp_path)
    second=reopened.reserve(cid,'engineer',1000000)
    reopened.finish_request(second,900000)
    assert reopened.get_campaign(cid)['token_limit'] is None
    assert reopened.get_campaign(old)['token_limit']==750000
    assert sum(u['tokens'] for u in reopened.status()['usage'] if u['campaign']==cid)==1700000
    assert Supervisor(None,reopened).remaining_tokens(cid) is None
    with pytest.raises(RuntimeError,match='Campaign token budget exhausted'):
        reopened.reserve(old,'scientist',750001)


def test_uncapped_campaign_keeps_atomic_task_reservation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store=Store(tmp_path); cid=store.campaign({},None)
    store.start_task('bounded',cid,'scientist',{'task_token_limit':100})
    def reserve(_):
        try:return store.reserve(cid,'scientist',60,'bounded')
        except RuntimeError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(reserve,range(2)))
    assert sum(bool(r) for r in results)==1
    store.finish_request(next(r for r in results if r),None,'FAILED')
    with pytest.raises(RuntimeError,match='Task token budget exhausted'):
        Store(tmp_path).reserve(cid,'scientist',41,'bounded')


def test_supervisor_uncapped_completes_and_reports_null_remaining(tmp_path):
    store,cid,desk=setup(tmp_path,[{'status':'READY'},{'status':'READY'}],budget=None)
    request=store.reserve(cid,'scientist',2000000); store.finish_request(request,None,'FAILED')
    result=Supervisor(desk,store).run(cid,2,1,0,0)
    assert result['status']=='COMPLETE' and result['completed_cycles']==2
    assert result['remaining_tokens'] is None and desk.calls==2
    assert json.loads(open(result['report_path'],encoding='utf-8').read())['remaining_tokens'] is None


def test_supervisor_uncapped_still_stops_on_failure(tmp_path):
    from quantdesk.agents import AgentTaskFailure
    store,cid,desk=setup(tmp_path,[AgentTaskFailure('scientist',TimeoutError())],budget=None)
    result=Supervisor(desk,store).run(cid,2,1,0,0)
    assert result['status']=='FAILED' and result['retries']==0 and desk.calls==1
    assert result['remaining_tokens'] is None


def test_cli_default_has_no_local_cap_but_explicit_cap_is_preserved():
    assert Settings().max_campaign_tokens is None
    assert parser().parse_args(['plan']).token_budget is None
    assert parser().parse_args(['plan','--token-budget','250000']).token_budget==250000


def test_dashboard_plan_http_null_omitted_and_explicit_cap(tmp_path,monkeypatch):
    import quantdesk.web as web
    servers=[]
    def server(*args,**kwargs):
        value=ThreadingHTTPServer(*args,**kwargs); servers.append(value); return value
    monkeypatch.setattr(web,'ThreadingHTTPServer',server)
    store=Store(tmp_path)
    class Catalog:
        def plan(self,*args):return {'development':[{}],'validation':[{}],'holdout':[{}],'sampled':False}
    worker=threading.Thread(target=web.serve,args=(store,Catalog(),None,replace(Settings(),home=tmp_path),0),daemon=True)
    worker.start()
    # Listen creation is asynchronous; a short event wait avoids a fixed port.
    import time
    deadline=time.monotonic()+5
    while not servers and time.monotonic()<deadline:time.sleep(0.01)
    assert servers
    endpoint=f'http://127.0.0.1:{servers[0].server_port}/api/plan'
    try:
        for payload,expected in (({},None),({'token_budget':None},None),({'token_budget':250000},250000)):
            request=urllib.request.Request(endpoint,json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=5) as response:
                result=json.load(response)
            assert result['token_limit']==expected
            assert store.get_campaign(result['campaign'])['token_limit']==expected
    finally:
        servers[0].shutdown();worker.join(timeout=5)
    assert not worker.is_alive()
