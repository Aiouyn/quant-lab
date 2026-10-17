"""Streaming integration: real SDK/local SSE and explicit fault fixtures."""
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace
import pytest
from openai import OpenAI
from quantdesk.agents import AgentRuntime,AgentTaskFailure,ResearchTools
from quantdesk.control import ResearchStopped
from quantdesk.provider_stream import StreamAssembly,streaming_call
from quantdesk.settings import Settings
from quantdesk.store import Store


def chunk(delta=None,finish=None,usage=None):
    return {'id':'fixture-response','model':'fixture-model','created':1,
        'choices':[{'index':0,'delta':delta or {},'finish_reason':finish}] if delta is not None or finish is not None else [],
        **({'usage':usage} if usage is not None else {})}


def usage(total=50): return {'total_tokens':total,'prompt_tokens':total-10,'completion_tokens':10}


def tool(arguments='{}'):
    return {'tool_calls':[{'index':0,'id':'fixture-call','type':'function',
        'function':{'name':'recall','arguments':arguments}}]}


def fake_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_assembly_interleaved_public_tools_and_private_context():
    a=StreamAssembly()
    a.add(chunk({'reasoning_content':'opaque '},None))
    a.add(chunk({'reasoning_content':'thoughts','content':'public'}))
    a.add(chunk({'tool_calls':[{'index':1,'id':'b','type':'function','function':{'name':'second','arguments':'{"b":'}}]}))
    a.add(chunk({'tool_calls':[{'index':0,'id':'a','type':'function','function':{'name':'first','arguments':'{"a":1}'}}]}))
    a.add(chunk({'tool_calls':[{'index':1,'function':{'arguments':'2}'}}]},'tool_calls'))
    a.add(chunk(usage=usage()))
    response=a.response(); m=response.choices[0].message
    assert m.reasoning_content=='opaque thoughts' and m.content=='public'
    assert [c.id for c in m.tool_calls]==['a','b']
    assert json.loads(m.tool_calls[1].function.arguments)=={'b':2}
    assert response.usage.total_tokens==50
    assert 'opaque thoughts' not in json.dumps(a.public_partial())


@pytest.mark.parametrize('chunks,match',[
    ([chunk({'content':'no finish'})],'terminal completion'),
    ([chunk(tool(),'stop'),chunk({'content':'late'})],'after its terminal'),
    ([chunk({},'content_filter')],'content-filtered'),
    ([chunk({'tool_calls':[{'index':True}]})],'tool-call index'),
    ([chunk({'tool_calls':[{'index':0,'id':'id','function':{'arguments':'{}'}}]},'tool_calls')],'tool identity'),
    ([chunk({},'stop'),chunk(usage=usage(-1))],'nonnegative'),
    ([chunk({'content':'hello'}),dict(chunk({},'stop'),id='different')],'identity changed')])
def test_malformed_or_unterminated_stream_cannot_be_a_completion(chunks,match):
    a=StreamAssembly()
    with pytest.raises((ValueError,ConnectionError),match=match):
        for c in chunks: a.add(c)
        a.response()


def test_response_size_limit_includes_private_text():
    a=StreamAssembly(max_bytes=10)
    with pytest.raises(ValueError,match='bounded response size'):
        a.add(chunk({'reasoning_content':'opaque private data'}))
    assert a.reasoning_chars==0


def test_per_chunk_created_timestamps_and_interim_usage_are_not_identity_failures():
    a=StreamAssembly()
    a.add(dict(chunk({'reasoning_content':'opaque'},usage=usage(20)),created=1))
    assert a.usage is None
    a.add(dict(chunk({'content':'public'},usage=usage(30)),created=2))
    assert a.usage is None
    a.add(dict(chunk({},'stop',usage()),created=5))
    a.add(dict(chunk(usage=usage(60)),created=6))
    r=a.response()
    assert r.created==1 and r.usage.total_tokens==60
    assert r.choices[0].message.reasoning_content=='opaque'


def test_controlled_protocol_failure_diagnosis_never_contains_private_text():
    public=[]
    chunks=[chunk({'reasoning_content':'PRIVATE-SENTINEL'}),dict(chunk({},'stop'),id='another-response')]
    with pytest.raises(ValueError,match='identity changed'):
        streaming_call(fake_client(lambda **kw:iter(chunks)),{},1,partial=public.append)
    assert public[0]['protocol_error']=='Provider response identity changed during the stream'
    assert public[0]['failure_type']=='StreamProtocolError'
    assert 'PRIVATE-SENTINEL' not in json.dumps(public)


def test_deadline_preserves_public_draft_never_returns_late_tools():
    def generate():
        yield chunk({'reasoning_content':'secret private context'})
        yield chunk(tool('{"limit":4}'))
        time.sleep(0.15)
        yield chunk({},'tool_calls')
        yield chunk(usage=usage())
    public=[]; progress=[]; started=time.monotonic()
    with pytest.raises(TimeoutError):
        streaming_call(fake_client(lambda **kw:generate()),{},0.02,progress=progress.append,partial=public.append)
    assert time.monotonic()-started<0.12
    assert public[0]['tool_calls'][0]['arguments']=='{"limit":4}' and not public[0]['executable']
    assert 'secret private context' not in json.dumps(public+progress)
    assert progress[-1]['opaque_reasoning_characters']==len('secret private context')


@pytest.mark.parametrize('known_usage',[False,True])
def test_runtime_stream_failure_charges_ambiguity_or_actual_and_cannot_execute(tmp_path,known_usage,monkeypatch):
    store=Store(tmp_path); cid=store.campaign({},100000)
    now=[0.0]; release=threading.Event()
    def generate():
        yield chunk(tool())
        if known_usage: yield chunk({},'tool_calls',usage())
        # Even complete-looking calls and a terminal marker cannot bypass EOF.
        release.wait(5)
    def deterministic_deadline(*args,**kwargs):
        original_progress=kwargs.get('progress')
        def progress(value):
            if original_progress:original_progress(value)
            # Expire only after the intended fault evidence has been consumed.
            # Host scheduling/SQLite latency cannot turn known usage into unknown.
            if value['actual_tokens']==50 or not known_usage and value['chunks']:
                now[0]=200.0
            elif known_usage and value['chunks']==1:
                now[0]=15.1
        kwargs.update(clock=lambda:now[0],progress=progress)
        try:return streaming_call(*args,**kwargs)
        finally:release.set()
    monkeypatch.setattr('quantdesk.provider_stream.streaming_call',deterministic_deadline)
    client=fake_client(lambda **kw:generate())
    class Tools:
        def available(self,role): return [{'type':'function','function':{'name':'recall','parameters':{'type':'object'}}}]
        def call(self,*a): pytest.fail('Partial/late response must not run a tool')
    settings=replace(Settings(),request_timeout=100,max_agent_steps=1)
    agent=AgentRuntime(store,settings,client=client,model='fixture',provider='mimo',streaming=True)
    with pytest.raises(AgentTaskFailure): agent.run(cid,'director',{},Tools(),'stream-failure')
    with store.connect() as db:
        r=db.execute('SELECT status,actual,reserved FROM requests').fetchone()
        assert r['status']=='FAILED'
        assert r['actual']==(50 if known_usage else None)
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='FAILED'
        assert not db.execute("SELECT 1 FROM events WHERE kind='AGENT_COMPLETED'").fetchone()
        assert db.execute("SELECT 1 FROM events WHERE kind='INCOMPLETE_PUBLIC_OUTPUT'").fetchone()


def test_cooperative_stop_during_stream_finishes_request_and_prevents_tools(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    def generate():
        yield chunk({'content':'public partial'})
        store.set_status(cid,'STOP_REQUESTED')
        yield chunk(tool(),'tool_calls',usage())
    tools=ResearchTools(cid,store,None,None,Settings())
    agent=AgentRuntime(store,replace(Settings(),request_timeout=2),client=fake_client(lambda **kw:generate()),model='fixture',provider='mimo',streaming=True)
    with pytest.raises(ResearchStopped): agent.run(cid,'director',{},tools,'stream-stopped')
    with store.connect() as db:
        assert db.execute('SELECT status FROM requests').fetchone()[0]=='FAILED'
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='STOPPED'
        assert not db.execute("SELECT 1 FROM events WHERE kind='TOOL_RESULT'").fetchone()


def test_nonterminal_eof_is_retryable_but_has_no_success(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    agent=AgentRuntime(store,Settings(),client=fake_client(lambda **kw:iter([chunk(tool())])),model='fixture',provider='mimo',streaming=True)
    with pytest.raises(AgentTaskFailure) as caught:
        agent.run(cid,'director',{},ResearchTools(cid,store,None,None,Settings()),'stream-eof')
    assert caught.value.retryable and caught.value.error_type=='ConnectionError'
    with store.connect() as db: assert not db.execute("SELECT 1 FROM events WHERE kind='TOOL_RESULT'").fetchone()


def test_real_sdk_local_sse_preserves_opaque_context_and_accounts_usage(tmp_path):
    received=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(payload)
            assert payload['stream'] and payload['stream_options']=={'include_usage':True}
            if len(received)==1:
                chunks=[chunk({'reasoning_content':'OPAQUE-CONTEXT-SENTINEL'}),chunk(tool('{"limit":1}')),
                        chunk({},'tool_calls'),chunk(usage=usage(50))]
            else:
                chunks=[chunk({'content':'{"allocation":"observed recall used"}'}),chunk({},'stop'),chunk(usage=usage(80))]
            self.send_response(200); self.send_header('Content-Type','text/event-stream')
            self.send_header('Connection','close'); self.end_headers()
            for c in chunks:
                self.wfile.write(('data: '+json.dumps(c)+'\n\n').encode()); self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush(); self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    client=OpenAI(api_key='synthetic-local-fixture',base_url=f'http://127.0.0.1:{server.server_port}/v1',max_retries=0)
    try:
        store=Store(tmp_path); cid=store.campaign({},100000)
        settings=replace(Settings(),request_timeout=5,max_agent_steps=2)
        agent=AgentRuntime(store,settings,client=client,model='fixture',provider='mimo',streaming=True)
        result=agent.run(cid,'director',{},ResearchTools(cid,store,None,None,settings),'sdk-stream')
        assert result['allocation']=='observed recall used' and len(received)==2
        assistant=next(m for m in received[1]['messages'] if m['role']=='assistant')
        assert assistant['reasoning_content']=='OPAQUE-CONTEXT-SENTINEL'
        with store.connect() as db:
            assert db.execute('SELECT SUM(actual) FROM requests').fetchone()[0]==130
            assert db.execute('SELECT status FROM tasks').fetchone()[0]=='COMPLETE'
            public=' '.join(r[0] for r in db.execute('SELECT payload FROM events'))
            assert 'OPAQUE-CONTEXT-SENTINEL' not in public
            assert db.execute("SELECT count(*) FROM events WHERE kind='PROVIDER_PROGRESS'").fetchone()[0]>=2
            assert db.execute("SELECT count(*) FROM events WHERE kind='TOOL_RESULT'").fetchone()[0]==1
    finally:
        client.close(); server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_public_stream_draft_resumes_as_untrusted_input_with_original_budget(tmp_path):
    from tests.test_scientific_decisions import setup,proposal,call
    from tests.test_agent_completion import runtime,response
    store,cid,_,task,tools,settings=setup(tmp_path,start=False)
    def generate():
        yield chunk({'reasoning_content':'PRIVATE-DRAFT-SENTINEL'})
        yield chunk({'tool_calls':[{'index':0,'id':'partial-call','type':'function',
            'function':{'name':'submit_proposal','arguments':'{"hypothesis":"unfinished'}}]})
    agent=AgentRuntime(store,settings,client=fake_client(lambda **kw:generate()),model='fixture',provider='mimo',streaming=True)
    with pytest.raises(AgentTaskFailure):
        agent.run(cid,'scientist',{'proposal_workflow':True,'task_token_limit':100000},tools,task)
    with store.connect() as db:
        before=db.execute('SELECT SUM(COALESCE(actual,reserved)) FROM requests').fetchone()[0]
        assert not db.execute('SELECT 1 FROM research_decisions').fetchone()
    def create(**kwargs):
        resumed=json.loads(kwargs['messages'][1]['content'])
        assert resumed['resume_public_draft'] and resumed['resume_scientific_commit']
        draft=resumed['incomplete_public_output']
        assert not draft['executable'] and draft['tool_calls'][0]['arguments']=='{"hypothesis":"unfinished'
        assert 'PRIVATE-DRAFT-SENTINEL' not in json.dumps(kwargs)
        assert kwargs['tool_choice']=='auto'
        assert {t['function']['name'] for t in kwargs['tools']}=={'submit_proposal','stop_mission'}
        return response(call('submit_proposal',proposal()))
    result=runtime(store,settings,create).run(cid,'scientist',{'proposal_workflow':True,'task_token_limit':100000},tools,task)
    assert result['artifact_receipt']
    with store.connect() as db:
        assert db.execute('SELECT SUM(COALESCE(actual,reserved)) FROM requests').fetchone()[0]==before+100
        assert db.execute('SELECT token_limit FROM task_budgets WHERE task_id=?',(task,)).fetchone()[0]==100000
        assert 'PRIVATE-DRAFT-SENTINEL' not in ' '.join(r[0] for r in db.execute('SELECT payload FROM events'))
