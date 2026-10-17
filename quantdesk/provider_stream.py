"""Bounded provider streaming; only a complete response may execute tools.

Opaque reasoning is retained in RAM for the next real tool turn. Persisted
callbacks receive counts or public fragments, never the reasoning text.
"""
import queue
import threading
import time
from openai.types.chat import ChatCompletion


class StreamProtocolError(ValueError):
    """Controlled protocol diagnosis; messages never embed response text."""
    pass


class StreamAssembly:
    def __init__(self,max_bytes=2000000):
        self.max_bytes=max_bytes; self.bytes=0; self.chunks=0
        self.content=[]; self.reasoning=[]; self.calls={}
        self.finish=None; self.usage=None; self.identity={}
        self.public_chars=0; self.reasoning_chars=0; self.seen_choice=False

    def fragment(self,value,target,private=False):
        if value is None: return
        if not isinstance(value,str): raise StreamProtocolError('Provider stream text must be a string')
        self.bytes+=len(value.encode())
        if self.bytes>self.max_bytes: raise StreamProtocolError('Provider stream exceeded bounded response size')
        target.append(value)
        if private: self.reasoning_chars+=len(value)
        else: self.public_chars+=len(value)

    def add(self,chunk):
        value=chunk.model_dump(exclude_none=True) if hasattr(chunk,'model_dump') else chunk
        if not isinstance(value,dict): raise StreamProtocolError('Provider stream chunk must be an object')
        self.chunks+=1
        for key in ('id','model','created','system_fingerprint'):
            if value.get(key) is not None:
                if key!='created' and key in self.identity and self.identity[key]!=value[key]:
                    raise StreamProtocolError('Provider response identity changed during the stream')
                if key!='created' or key not in self.identity:
                    self.identity[key]=value[key]
        usage=value.get('usage')
        if usage is not None:
            if not isinstance(usage,dict) or any(isinstance(usage.get(k),bool) or not isinstance(usage.get(k),int) or usage[k]<0 for k in ('total_tokens','prompt_tokens','completion_tokens')):
                raise StreamProtocolError('Provider usage must contain nonnegative integer token counts')
        choices=value.get('choices',[])
        if not isinstance(choices,list): raise StreamProtocolError('Provider choices must be an array')
        for choice in choices:
            if not isinstance(choice,dict) or choice.get('index')!=0:
                raise StreamProtocolError('Only one provider completion choice is supported')
            self.seen_choice=True; delta=choice.get('delta') or {}
            if not isinstance(delta,dict): raise StreamProtocolError('Provider delta must be an object')
            if self.finish is not None and any(delta.get(k) for k in ('content','reasoning_content','tool_calls','refusal')):
                raise StreamProtocolError('Provider produced text/tools after its terminal marker')
            if delta.get('refusal'): raise StreamProtocolError('Provider refused the requested response')
            self.fragment(delta.get('content'),self.content)
            self.fragment(delta.get('reasoning_content'),self.reasoning,private=True)
            for call in delta.get('tool_calls') or []:
                index=call.get('index')
                if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<32:
                    raise StreamProtocolError('Invalid streamed tool-call index')
                item=self.calls.setdefault(index,{'id':[],'name':[],'arguments':[]})
                if call.get('type') not in {None,'function'}: raise StreamProtocolError('Unsupported streamed tool type')
                self.fragment(call.get('id'),item['id'])
                function=call.get('function') or {}
                self.fragment(function.get('name'),item['name'])
                self.fragment(function.get('arguments'),item['arguments'])
            finish=choice.get('finish_reason')
            if finish is not None:
                if finish not in {'stop','tool_calls','length','content_filter'}:
                    raise StreamProtocolError('Unknown provider terminal reason')
                if self.finish is not None and self.finish!=finish: raise StreamProtocolError('Conflicting provider terminal markers')
                self.finish=finish
        if usage is not None and self.finish is not None:
            # Some compatible endpoints send placeholder/cumulative usage. Only
            # terminal usage can become a known request charge; later terminal
            # usage replaces a placeholder, rather than corrupting the stream.
            self.usage=usage

    def public_partial(self):
        # This is a draft receipt, never a reconstructed successful completion.
        return {'content':''.join(self.content)[:200000],
            'tool_calls':[{'index':i,'name':''.join(v['name'])[:200],
                'arguments':''.join(v['arguments'])[:200000]} for i,v in sorted(self.calls.items())],
            'executable':False,'terminal_marker_seen':self.finish is not None,
            'public_truncated':self.public_chars>200000,
            'reason':'Incomplete or interrupted provider stream; no tool executed'}

    def response(self):
        if not self.seen_choice or self.finish is None:
            raise ConnectionError('Provider stream ended without a terminal completion marker')
        if self.finish=='content_filter': raise StreamProtocolError('Provider response was content-filtered')
        calls=[]
        for _,item in sorted(self.calls.items()):
            ident=''.join(item['id']); name=''.join(item['name'])
            if not ident or not name: raise StreamProtocolError('Incomplete streamed tool identity')
            calls.append({'id':ident,'type':'function','function':{'name':name,'arguments':''.join(item['arguments'])}})
        if len({c['id'] for c in calls})!=len(calls): raise StreamProtocolError('Duplicate streamed tool-call IDs')
        message={'role':'assistant','content':''.join(self.content) or None,
                 'reasoning_content':''.join(self.reasoning),'tool_calls':calls or None}
        return ChatCompletion.model_validate({'id':self.identity.get('id','unreported'),
            'created':self.identity.get('created',int(time.time())),'model':self.identity.get('model','unreported'),
            'system_fingerprint':self.identity.get('system_fingerprint'),'object':'chat.completion',
            'choices':[{'index':0,'finish_reason':self.finish,'message':message}], 'usage':self.usage})


def streaming_call(client,kwargs,timeout,progress=None,partial=None,control=None,clock=time.monotonic):
    started=clock(); deadline=started+timeout; cancel=threading.Event()
    mailbox=queue.Queue(maxsize=32); assembly=StreamAssembly()
    def emit(kind,value):
        while not cancel.is_set():
            try: mailbox.put((kind,value),timeout=0.05); return
            except queue.Full: continue
    def invoke():
        stream=None
        try:
            stream=client.chat.completions.create(**dict(kwargs,stream=True,stream_options={'include_usage':True}))
            for chunk in stream:
                if cancel.is_set(): break
                emit('chunk',chunk)
            if not cancel.is_set(): emit('eof',None)
        except BaseException as error:
            emit('error',error)
        finally:
            # Close on the producer thread: a slow transport close must not
            # hold the research thread beyond its independent wall deadline.
            if stream is not None and hasattr(stream,'close'):
                try: stream.close()
                except Exception: pass
    threading.Thread(target=invoke,daemon=True).start()
    last_progress=None; last_control=started-1; first_chunk=None; last_chunk=None
    def report(final=False):
        nonlocal last_progress
        now=clock()
        if progress and (final or last_progress is None or now-last_progress>=15):
            progress({'elapsed_seconds':round(now-started,3),'chunks':assembly.chunks,
                'public_characters':assembly.public_chars,'opaque_reasoning_characters':assembly.reasoning_chars,
                'first_chunk_seconds':round(first_chunk-started,3) if first_chunk is not None else None,
                'last_chunk_age_seconds':round(now-last_chunk,3) if last_chunk is not None else None,
                'terminal_marker_seen':assembly.finish is not None,'final':final,
                'actual_tokens':assembly.usage['total_tokens'] if assembly.usage else None})
            last_progress=now
    try:
        while True:
            now=clock()
            if now>=deadline: raise TimeoutError('Provider stream exceeded task wall-clock deadline; partial output cannot execute')
            if control and now-last_control>=1:
                control(); last_control=now
            try: kind,value=mailbox.get(timeout=min(0.2,max(0.001,deadline-now)))
            except queue.Empty:
                report(); continue
            if clock()>=deadline: raise TimeoutError('Provider stream completed after task deadline; late output cannot execute')
            if kind=='error': raise value
            if kind=='eof':
                response=assembly.response()
                if control: control()
                report(final=True)
                return response
            last_chunk=clock()
            if first_chunk is None: first_chunk=last_chunk
            assembly.add(value); report()
    except BaseException as error:
        if partial:
            draft=assembly.public_partial()
            draft.update(failure_type=type(error).__name__,protocol_error=str(error) if isinstance(error,StreamProtocolError) else None)
            partial(draft)
        if assembly.usage is not None and assembly.usage['total_tokens']>0:
            error.provider_actual_tokens=assembly.usage['total_tokens']
        report(final=True)
        raise
    finally:
        cancel.set()
