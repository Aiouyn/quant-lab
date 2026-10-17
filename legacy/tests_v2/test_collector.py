import json
from quantdesk.collector import Collector


def test_actual_trades_are_deduplicated_by_venue_identity(tmp_path):
    c=Collector(tmp_path)
    trade={'coin':'BTC','time':1000,'tid':5,'px':'100','sz':'1','side':'B'}
    c.consume({'channel':'trades','data':[trade,trade,dict(trade,time=2000)]})
    assert c.counts['trades']==2 and c.duplicates==1
    messages=[json.loads(line) for p in c.directory.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert len(messages[0]['message']['data'])==3 # raw evidence retained


def test_missing_book_intervals_are_explicit_and_do_not_invent_trades(tmp_path):
    c=Collector(tmp_path)
    for ts in (1000,10000):
        c.consume({'channel':'l2Book','data':{'coin':'BTC','time':ts,'levels':[[],[]]}})
    assert c.counts['book_gaps']==1 and c.counts['trades']==0


def test_connection_failures_leave_a_durable_incomplete_capture(tmp_path):
    c=Collector(tmp_path)
    def unavailable(*args,**kwargs):
        raise OSError('test disconnect')
    summary=c.run(1,max_reconnects=0,connect=unavailable)
    assert summary['errors']==1 and summary['stopped_early']
    assert not summary['coverage_verified']
    assert summary['orders_submitted_to_exchange']==0
    assert (c.directory/'summary.json').exists()


def test_fast_subscription_is_explicit_and_status_is_durable(tmp_path,monkeypatch):
    import quantdesk.collector as module
    clock=[0.0]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    class Socket:
        sent=[]
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def send(self,value): self.sent.append(json.loads(value))
        def recv(self,timeout):
            clock[0]=2
            return json.dumps({'channel':'l2Book','data':{'coin':'BTC','time':1000,'levels':[[],[]]}})
    socket=Socket(); c=Collector(tmp_path,book_fast=True)
    summary=c.run(1,connect=lambda *args,**kwargs:socket)
    subscription=next(r['subscription'] for r in socket.sent if r['subscription']['type']=='l2Book')
    assert subscription['fast'] is True and summary['book_mode']=='fast_5_levels'
    status=json.loads((c.directory/'status.json').read_text())
    assert status['status']=='COMPLETE' and status['counts']['l2Book']==1
