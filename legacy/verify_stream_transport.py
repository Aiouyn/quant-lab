"""Tiny real endpoint transport diagnostic; no agent work or numerical delivery."""
import hashlib
import json
import math
import os
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from openai import OpenAI
import config
from quantdesk.provider_stream import streaming_call
from quantdesk.settings import Settings
from quantdesk.store import Store,process_alive


def main():
    settings=Settings(); store=Store(settings.home); cid='2b22519842f6'
    directory=settings.home/'verification'/'stream-transport'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'run.json'
    if path.exists():
        old=json.loads(path.read_text()); print(json.dumps({'status':old['status'],'pid':old['pid'],
            'process_alive':process_alive(old['pid']),'receipt':str(path),'new_call_started':False})); return
    task_id='stream-transport:'+cid+':metadata-v1'; allowance=4000
    state={'status':'PREPARING','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
        'campaign':cid,'task_id':task_id,'task_token_limit':allowance,
        'scope':'One tiny actual endpoint diagnostic for streamed metadata/usage only; no scientist retry, engineering retry, strategy or company acceptance',
        'criteria':{'maximum_requests':1,'completion_tokens_max':512,'deadline_seconds':120,'budget_increase':False},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')}}
    with path.open('x',encoding='utf-8') as f: json.dump(state,f,indent=2)
    def persist():
        stage=path.with_suffix('.partial'); stage.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8'); stage.replace(path)
    kwargs={'model':config.DEFAULT_MODEL,'messages':[{'role':'user','content':'Return exactly {"transport_probe":true}. No explanation or tools.'}],
        'max_completion_tokens':512,'tools':[], 'extra_body':{'thinking':{'type':'enabled'}}}
    reservation=None; recorded=False; metadata=[]; progress=[]; sdk=None
    try:
        assert store.get_campaign(cid)['token_limit']==750000
        with store.lease('research',seconds=180):
            store.start_task(task_id,cid,'transport_diagnostic',{'task_token_limit':allowance})
            estimate=math.ceil(len(json.dumps(kwargs).encode())/2)+512+2000
            reservation=store.reserve(cid,'transport_diagnostic',estimate,task_id=task_id)
            sdk=OpenAI(api_key=config.ACTIVE_KEY,base_url=config.ACTIVE_BASE_URL,max_retries=0,timeout=120)
            def create(**payload):
                stream=sdk.chat.completions.create(**payload)
                try:
                    for part in stream:
                        value=part.model_dump(exclude_none=True)
                        metadata.append({'created':value.get('created'),
                            'choices':[{'index':c.get('index'),'finish_reason':c.get('finish_reason'),
                                'delta_fields':list(c.get('delta',{})),
                                'opaque_characters':len(c.get('delta',{}).get('reasoning_content') or ''),
                                'public_characters':len(c.get('delta',{}).get('content') or '')} for c in value.get('choices',[])],
                            'usage':value.get('usage')})
                        yield part
                finally: stream.close()
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            state.update(status='RUNNING',request_id=reservation,reserved_tokens=estimate); persist()
            def on_progress(value):
                progress.append(value)
                store.event(cid,'transport_diagnostic','PROVIDER_PROGRESS',dict(task_id=task_id,request_id=reservation,**value))
            def on_partial(value): store.event(cid,'transport_diagnostic','INCOMPLETE_PUBLIC_OUTPUT',dict(task_id=task_id,request_id=reservation,**value))
            response=streaming_call(client,kwargs,120,progress=on_progress,partial=on_partial,
                control=lambda:store.check_research_control(cid))
            actual=response.usage.total_tokens if response.usage else None
            store.finish_request(reservation,actual); recorded=True
            state.update(status='COMPLETE',finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump(exclude_none=True) if response.usage else None,
                returned_model=response.model,response_id=response.id,
                public_content=response.choices[0].message.content,
                interpretation='Actual endpoint streamed metadata/usage assembled; no research artifact, code, scientific decision, learning benefit or company completion claimed')
            store.finish_task(task_id,{'transport_receipt':str(path),'finish_reason':response.choices[0].finish_reason})
    except BaseException as error:
        if reservation and not recorded: store.finish_request(reservation,getattr(error,'provider_actual_tokens',None),'FAILED')
        store.fail_task(task_id,error)
        state.update(status='FAILED',error_type=type(error).__name__,
            protocol_error=str(error) if type(error).__name__=='StreamProtocolError' else None)
    finally:
        if sdk: sdk.close()
        state.update(chunk_metadata=metadata,progress=progress,completed_utc=datetime.now(timezone.utc).isoformat())
        with store.connect() as db:
            state['request_usage']=dict(db.execute('SELECT COUNT(*) requests,COALESCE(SUM(actual),0) known_actual,COALESCE(SUM(COALESCE(actual,reserved)),0) charged FROM requests JOIN request_tasks ON requests.id=request_tasks.request_id WHERE task_id=?',(task_id,)).fetchone())
            state['campaign_usage']=dict(db.execute('SELECT COUNT(*) requests,COALESCE(SUM(actual),0) known_actual,COALESCE(SUM(COALESCE(actual,reserved)),0) charged FROM requests WHERE campaign=?',(cid,)).fetchone())
        state['remaining_original_tokens']=750000-state['campaign_usage']['charged']; persist()
    print(json.dumps({k:v for k,v in state.items() if k not in {'source_sha256','progress','chunk_metadata'}},indent=2),flush=True)


if __name__=='__main__':main()
