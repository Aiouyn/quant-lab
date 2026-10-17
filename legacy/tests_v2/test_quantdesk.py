"""Regression tests for economics, causal isolation, gates, persistence, and tools."""
import json
import tarfile
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.data import Book,Catalog,enrich,load_archive
from quantdesk.strategy import evaluate_in_process,validate_code,run_signal
from quantdesk.strategy import validate_spec
from quantdesk.execution import Broker,replay
from quantdesk.evaluation import Evaluator,daily_statistics
from quantdesk.portfolio import allocate
from quantdesk.paper import PaperSession,load_bundle
from quantdesk.agents import AgentRuntime,ResearchTools

CODE="import polars as pl\ndef compute_signal(df: pl.DataFrame) -> pl.Series:\n    return df['qi_5'].clip(-1.0,1.0)\n"

def spec(**kw):
    return {'title':'Test pressure','rationale':'Test a causal book imbalance','falsification':'Reject after realistic costs or causal inconsistency','holding_seconds':2,'threshold':0.2,'warmup_rows':0,'mode':'taker',**kw}

def frame(n=80,spacing=1000):
    book=Book(); rows=[]
    for i in range(n):
        mid=100+np.sin(i/5)*0.2
        rows.append(book.apply({'action':'snapshot','_continuous_snapshot':True,'ts':i*spacing,'bids':[[mid-0.01,10+np.sin(i),3]],'asks':[[mid+0.01,10-np.sin(i),3]]}))
    return enrich(pl.DataFrame(rows))

def quote(ts,bid=99.99,ask=100.01,volume=100):
    return {'ts_ms':ts,'mid_price':(bid+ask)/2,'bid_p0':bid,'ask_p0':ask,'bid_v0':volume,'ask_v0':volume,'spread_bps':(ask-bid)/((bid+ask)/2)*1e4}

def test_correct_ofi_when_quotes_worsen():
    book=Book()
    book.apply({'action':'snapshot','ts':0,'bids':[[100,7,1]],'asks':[[102,9,1]]})
    row=book.apply({'action':'update','ts':1000,'bids':[[100,0,0],[99,5,1]],'asks':[[102,0,0],[103,8,1]]})
    assert row['ofi']==2 # -old bid volume - (-old ask volume)

def test_sequence_gap_waits_for_snapshot():
    book=Book()
    book.apply({'action':'snapshot','ts':0,'seqId':10,'bids':[[100,7,1]],'asks':[[102,9,1]]})
    assert book.apply({'action':'update','ts':1000,'seqId':15,'prevSeqId':11,'bids':[],'asks':[]}) is None
    assert book.apply({'action':'update','ts':2000,'seqId':16,'prevSeqId':15,'bids':[],'asks':[]}) is None
    assert book.apply({'action':'snapshot','ts':3000,'seqId':20,'bids':[[100,7,1]],'asks':[[102,9,1]]}) is not None
    assert book.quality.sequence_gaps==1

def test_update_before_snapshot_cannot_build_book():
    assert Book().apply({'action':'update','ts':0,'bids':[[100,7,1]],'asks':[[102,9,1]]}) is None

def test_time_features_use_exchange_time_and_causal_prefix():
    b=Book(); rows=[]
    for i,ts in enumerate([0,100,800,1800,2100]):
        rows.append(b.apply({'action':'snapshot','_continuous_snapshot':True,'ts':ts,'bids':[[99+i,10,1]],'asks':[[101+i,10,1]]}))
    full=enrich(pl.DataFrame(rows))
    prefix=enrich(pl.DataFrame(rows[:3]))
    assert np.allclose(full['return_1s_bps'][:3],prefix['return_1s_bps'])
    assert full['return_1s_bps'][3]==pytest.approx((103/102-1)*1e4)
    assert full['return_1s_bps'][1]==0

def test_archive_directory_and_multiple_members(tmp_path):
    path=tmp_path/'test.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        directory=tarfile.TarInfo('folder'); directory.type=tarfile.DIRTYPE; tar.addfile(directory)
        for i in range(2):
            row={'action':'snapshot','ts':i*1000,'bids':[[100,7,1]],'asks':[[102,9,1]]}
            payload=(json.dumps(row)+'\n').encode(); member=tarfile.TarInfo(f'folder/{i}.jsonl'); member.size=len(payload); tar.addfile(member,io.BytesIO(payload))
    loaded,quality=load_archive(path)
    assert len(loaded)==2 and quality['snapshots']==2

@pytest.mark.parametrize('body',["df['mid_price'].shift(-1).fill_null(0.0)","(df['qi_5']-df['qi_5'].mean()).clip(-1.0,1.0)","df['qi_5'].rolling_mean(30).fill_null(strategy='backward').fill_null(0.0)"])
def test_causal_prefix_rejects_future_dependence(body):
    code=f'import polars as pl\ndef compute_signal(df: pl.DataFrame) -> pl.Series:\n    return ({body}).clip(-1.0,1.0)\n'
    with pytest.raises(ValueError,match='Causality'):
        evaluate_in_process(code,frame())

def test_generated_code_cannot_import_or_read_files():
    for code in ["import os\ndef compute_signal(df):\n    return df['qi_5']\n", "import numpy as np\ndef compute_signal(df):\n    return np.load('secret.npy')\n", "def compute_signal(df):\n    return open('secret')\n"]:
        with pytest.raises(ValueError):
            validate_code(code)

def test_nested_provider_spec_is_decoded_without_fabrication():
    assert validate_spec(json.dumps(spec()))==validate_spec(spec())
    with pytest.raises(ValueError,match='JSON object'):
        validate_spec(99)

def test_standard_numpy_shape_and_types_are_usable():
    code="import numpy as np\nimport polars as pl\ndef compute_signal(df):\n    x = df['qi_5'].to_numpy().astype(np.float64)\n    if x.shape[0] == 0:\n        return pl.Series('signal', np.zeros(0))\n    return pl.Series('signal', np.tanh(x))\n"
    assert len(evaluate_in_process(code,frame()))==80

def test_subprocess_worker_uses_causal_columns_and_matches_local():
    f=frame().with_columns(pl.lit(999).alias('future_return'))
    assert np.allclose(run_signal(CODE,f,10),evaluate_in_process(CODE,f))

def test_worker_nans_are_rejected():
    code="import polars as pl\ndef compute_signal(df):\n    return df['qi_5'] / 0.0\n"
    with pytest.raises(ValueError,match='finite'):
        run_signal(code,frame(),10)

def test_flat_market_roundtrip_loses_spread_slippage_and_fees():
    settings=Settings(); broker=Broker(spec(holding_seconds=1),settings.execution_dict())
    broker.step(quote(0),1)
    broker.step(quote(1000),1)
    broker.step(quote(2000),0)
    broker.step(quote(3000),0)
    assert len(broker.trades)==1
    assert broker.trades[0]['pnl']<0
    assert broker.account.fees>0
    assert broker.account.quantity==0
    assert broker.summary()['pnl']==pytest.approx(broker.trades[0]['pnl'])

def test_latency_is_timestamp_based_and_prevents_early_fill():
    broker=Broker(spec(),Settings().execution_dict())
    broker.step(quote(0),1)
    broker.step(quote(100),0)
    broker.step(quote(999),0)
    assert broker.account.quantity==0
    broker.step(quote(1001),0)
    assert broker.account.quantity>0

def test_depth_depletion_and_repeated_drops_never_prove_maker_fills():
    broker=Broker(spec(mode='maker'),Settings().execution_dict())
    broker.step(quote(0),1)
    for ts in range(1000,10000,1000):
        broker.step(quote(ts,volume=1),1)
    assert broker.account.quantity==0 and len(broker.trades)==0

def test_terminal_liquidates_and_initial_loss_enters_drawdown():
    broker=Broker(spec(),Settings().execution_dict())
    broker.step(quote(0),1); broker.step(quote(1000),0); broker.step(quote(2000),0,terminal=True)
    assert broker.account.quantity==0
    assert broker.summary()['max_drawdown_fraction']>0

def test_data_gap_cancels_pending_order():
    broker=Broker(spec(),Settings().execution_dict())
    broker.step(quote(0),1); broker.step(quote(10000),1)
    assert broker.pending is None and broker.account.quantity==0

def test_budget_reservations_are_atomic(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100)
    def reserve():
        try:
            return store.reserve(cid,'test',60)
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations=list(pool.map(lambda _:reserve(),range(2)))
    assert len([r for r in reservations if r])==1

def test_candidate_hash_lineage_and_memory_survive_restart(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100)
    ident,created=store.candidate(cid,spec(),CODE)
    second,created_again=store.candidate(cid,spec(title='Different label'),CODE)
    assert ident==second and created and not created_again
    store.memory(cid,ident,'LESSON',{'evidence':'negative net execution PnL'})
    reopened=Store(tmp_path)
    assert reopened.recall()[0]['payload']['evidence']=='negative net execution PnL'

def test_holdout_reuse_rejected_even_with_different_plan_hash(tmp_path):
    store=Store(tmp_path); date={'date':'2026-01-01','path':'one','size':100,'mtime_ns':1}
    store.reserve_holdout('plan1',['a'],[date])
    with pytest.raises(RuntimeError,match='consumed'):
        store.reserve_holdout('plan2',['b'],[date])
    with pytest.raises(RuntimeError,match='consumed'):
        store.reserve_holdout('plan3',['c'],[dict(date,path='renamed',mtime_ns=999)])

def test_sampled_or_proxy_results_cannot_be_promoted(tmp_path):
    evaluator=Evaluator(Store(tmp_path),None,Settings())
    report={'statistics':{'days':100,'mean_daily_return':0.01,'lower_mean_95':0.005,'adjusted_p_value':0.001,'positive_days_fraction':0.8},'sampled':True,'days':[],'fill_model':'maker','trades':100}
    reasons=evaluator.gate(report)
    assert any('Partial' in r for r in reasons)
    assert any('Maker' in r for r in reasons)

def test_multiple_trials_increase_statistical_penalty():
    returns=np.random.default_rng(51).normal(0.001,0.003,60)
    one=daily_statistics(returns,1); many=daily_statistics(returns,100)
    assert many['adjusted_p_value']>=one['adjusted_p_value']
    assert daily_statistics([0,0,0],1)['adjusted_p_value']==1

def test_portfolio_caps_duplicates_and_cash():
    rng=np.random.default_rng(99); a=rng.normal(0,1,100); b=rng.normal(0,2,100)
    result=allocate(np.column_stack([a,a,b]),0.35)
    assert result['weights'][1]==0
    assert max(result['weights'])<=0.35000001
    assert result['cash_weight']==pytest.approx(0.3)
    assert sum(result['weights'])<=1

def bundle(tmp_path,status='EXPLORATORY',warmup=0):
    directory=tmp_path/'bundle'; directory.mkdir()
    strategy=spec(warmup_rows=warmup)
    manifest={'format':'quantdesk-v2','id':'test','title':'Test','status':status,'code_hash':digest(CODE),'spec_hash':digest(strategy),'spec':strategy,'execution':Settings().execution_dict()}
    manifest['execution_hash']=digest(manifest['execution'])
    (directory/'manifest.json').write_text(json.dumps(manifest)); (directory/'strategy.py').write_text(CODE)
    return directory

def test_bundle_hash_and_exploratory_gate(tmp_path):
    directory=bundle(tmp_path)
    with pytest.raises(ValueError,match='exploratory'):
        load_bundle(directory)
    load_bundle(directory,True)
    (directory/'strategy.py').write_text(CODE+'\n# modified')
    with pytest.raises(ValueError,match='integrity'):
        load_bundle(directory,True)

def test_paper_stale_books_cannot_trigger_orders(tmp_path):
    session=PaperSession(bundle(tmp_path),tmp_path,Settings(),True)
    book={'coin':'BTC','time':1000,'levels':[[{'px':'100','sz':'10','n':1}],[{'px':'100.02','sz':'2','n':1}]]}
    assert not session.consume(book,now_ms=20000)
    assert session.samples==0 and session.members[0]['broker'].pending is None

class Message:
    def __init__(self,content=None,tool_calls=None,reasoning_content=None):
        self.content=content; self.tool_calls=tool_calls; self.reasoning_content=reasoning_content
    def model_dump(self,exclude_none=True):
        data={'role':'assistant','content':self.content,'reasoning_content':self.reasoning_content}
        if self.tool_calls:
            data['tool_calls']=[{'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in self.tool_calls]
        return {k:v for k,v in data.items() if v is not None}

def test_mimo_thinking_and_tool_context_are_preserved(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},200000); calls=[]
    call=SimpleNamespace(id='call1',function=SimpleNamespace(name='recall',arguments='{}'))
    responses=[Message(tool_calls=[call],reasoning_content='opaque provider context'),Message(content='{"decision":"done"}')]
    def create(**kwargs):
        calls.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=100),choices=[SimpleNamespace(message=responses.pop(0),finish_reason='stop')])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class Tools:
        def available(self,role): return []
        def call(self,*args): return {'evidence':'prior measured failure'}
    runtime=AgentRuntime(store,Settings(),client=client,model='test',provider='mimo')
    assert runtime.run(cid,'director',{},Tools(),task_id='cycle1:director')['decision']=='done'
    assert runtime.run(cid,'director',{'new_context':'do not repeat completed work'},Tools(),task_id='cycle1:director')['decision']=='done'
    assert len(calls)==2
    assert calls[0]['extra_body']['thinking']['type']=='enabled'
    assert calls[1]['messages'][2]['reasoning_content']=='opaque provider context'
    assert sum(r['tokens'] for r in store.status()['usage'])==200
    assert 'opaque provider context' not in json.dumps(store.status())

def test_api_failure_never_produces_a_mock_hypothesis(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    def create(**kwargs): raise OSError('network unavailable')
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class Tools:
        def available(self,role): return []
    runtime=AgentRuntime(store,Settings(),client=client,model='test',provider='mimo')
    with pytest.raises(RuntimeError,match='no synthetic replacement'):
        runtime.run(cid,'scientist',{},Tools())
    assert not store.candidates()
    assert store.status()['usage'][0]['tokens']>0

def test_exploration_tools_cannot_read_holdout(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[]},10000)
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(ValueError,match='does not own'):
        tools.call('director','experiment',{})
    assert all(t['function']['name'] not in {'seal','holdout','submit_order'} for t in tools.available('engineer'))

def test_cycle_checkpoint_survives_restart(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10000)
    first=store.start_cycle(cid)
    assert Store(tmp_path).start_cycle(cid)==first
    store.complete_cycle(first)
    assert store.start_cycle(cid)!=first

def test_stale_process_lease_is_recovered_without_signalling(tmp_path):
    import os,time
    from quantdesk.store import process_alive
    assert process_alive(os.getpid())
    assert not process_alive(1073741823)
    store=Store(tmp_path)
    with store.connect() as db:
        db.execute('INSERT INTO leases VALUES(?,?,?,?)',('research','dead',time.time()+1000,1073741823))
    with store.lease('research'):
        with store.connect() as db:
            assert db.execute('SELECT pid FROM leases').fetchone()[0]==os.getpid()

def test_stop_request_prevents_the_next_provider_call(tmp_path):
    from quantdesk.agents import ResearchStopped
    store=Store(tmp_path); cid=store.campaign({},10000); store.set_status(cid,'STOP_REQUESTED')
    client=SimpleNamespace(chat=None)
    runtime=AgentRuntime(store,Settings(),client=client,model='test',provider='mimo')
    with pytest.raises(ResearchStopped):
        runtime.run(cid,'director',{},None)
    assert store.status()['usage']==[]

def test_feature_diagnostic_is_development_only_and_time_based(tmp_path):
    store=Store(tmp_path); plan={'development':[{'date':'2026-01-01'}],'max_rows_per_day':None,'hash':'test'}
    cid=store.campaign(plan,10000)
    class TestCatalog:
        def load(self,*args): return frame(100),{}
    tools=ResearchTools(cid,store,TestCatalog(),None,Settings())
    result=tools.call('scientist','measure_feature',{'feature':'qi_5','horizon_seconds':1})
    assert result['observations']>=20
    assert result['status']=='DESCRIPTIVE_DEVELOPMENT_ONLY'
    assert Evaluator(store,None,Settings()).trial_count('different-plan')==1


def test_previously_researched_dates_are_not_fresh_holdouts(tmp_path):
    store=Store(tmp_path); date={'date':'2026-01-01'}
    store.expose_dates('old',[date],'development')
    with pytest.raises(RuntimeError,match='already exposed'):
        store.reserve_holdout('new',['candidate'],[date])
    # Failed reservation is atomic; untouched test dates can still be sealed.
    store.reserve_holdout('new',['candidate'],[{'date':'2026-01-02'}])


def test_strategy_state_resets_between_reconstruction_segments():
    first=frame(40); second=frame(40).with_columns(pl.lit(99,dtype=pl.Int64).alias('segment'))
    combined=pl.concat([first,second])
    code="import polars as pl\ndef compute_signal(df):\n    return df['qi_5'].rolling_mean(8).fill_null(0.0)\n"
    signals=evaluate_in_process(code,combined)
    assert np.allclose(signals[40:],evaluate_in_process(code,second))
    assert np.all(signals[40:47]==0)


def test_memory_retrieval_keeps_grounded_lessons_after_many_probes(tmp_path):
    store=Store(tmp_path)
    store.memory('old','x','LESSON',{'lesson':'Conditional edge failed after target venue fees'})
    for i in range(30):
        store.memory('new','','FEATURE_DIAGNOSTIC',{'probe':i})
    recalled=store.recall(10)
    assert len(recalled)==10 and any(m['kind']=='LESSON' for m in recalled)


def test_paper_gap_resets_signal_warmup(tmp_path):
    session=PaperSession(bundle(tmp_path,warmup=2),tmp_path,Settings(),True)
    for ts in (1000,2000,3000,10000):
        book={'coin':'BTC','time':ts,'levels':[[{'px':'100','sz':'10','n':1}],[{'px':'100.02','sz':'2','n':1}]]}
        assert session.consume(book,now_ms=ts)
    assert len(session.history)==1
    assert session.members[0]['broker'].pending is None
    assert session.members[0]['broker'].account.quantity==0


def test_agent_can_build_online_estimator_with_numerical_helpers():
    code='''import numpy as np
import polars as pl

def online(x):
    state = 0.0
    out = np.zeros(len(x))
    for i in range(len(x)):
        state = 0.8 * state + 0.2 * x[i]
        out[i] = state
    return out

def compute_signal(df):
    return pl.Series('signal', np.tanh(online(df['qi_5'].to_numpy())))
'''
    f=frame()
    assert np.allclose(run_signal(code,f,10),evaluate_in_process(code,f))


def test_unbounded_loops_and_helper_defaults_remain_disallowed():
    for code in ["def compute_signal(df):\n    while True:\n        pass\n", "def helper(x=1):\n    return x\ndef compute_signal(df):\n    return df['qi_5']\n"]:
        with pytest.raises(ValueError):
            validate_code(code)


def test_repeated_tool_requests_reuse_evidence_without_extra_trials(tmp_path):
    store=Store(tmp_path); plan={'development':[{'date':'2026-01-01'}],'hash':'frozen'}
    cid=store.campaign(plan,10000)
    class TestCatalog:
        calls=0
        def profile(self,*args):
            self.calls+=1
            return {'rows':1000,'date':'2026-01-01'}
    catalog=TestCatalog(); tools=ResearchTools(cid,store,catalog,None,Settings())
    tools.call('scientist','inspect_data',{'day_index':0})
    resumed=ResearchTools(cid,Store(tmp_path),catalog,None,Settings())
    result=resumed.call('engineer','inspect_data',{'day_index':0})
    assert catalog.calls==1 and result['already_measured']


def test_exported_costs_cannot_be_silently_changed(tmp_path):
    directory=bundle(tmp_path)
    path=directory/'manifest.json'; manifest=json.loads(path.read_text())
    manifest['execution']['taker_fee_bps']=0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Execution integrity'):
        load_bundle(directory,True)


def test_broker_avoids_new_exposure_near_unmodeled_hourly_funding():
    broker=Broker(spec(holding_seconds=30),Settings().execution_dict())
    broker.step(quote(3590000),1)
    assert broker.pending is None
    broker.step(quote(3600000),1)
    broker.step(quote(3601000),1) # fresh quote after the deliberately skipped gap
    assert broker.pending is not None


def test_gap_across_funding_is_reported_for_promotion_review():
    broker=Broker(spec(holding_seconds=300),Settings().execution_dict())
    broker.step(quote(3200000),1); broker.step(quote(3201000),1)
    assert broker.account.quantity>0
    broker.step(quote(3601000),0)
    assert broker.summary()['funding_boundary_crossings']==1
    assert broker.account.quantity==0


def test_bootstrap_resolution_scales_with_large_research_families():
    stats=daily_statistics(np.random.default_rng(19).normal(0.01,0.001,30),300)
    assert stats['bootstrap_draws']==24000
    assert stats['adjusted_p_value']<0.05


def test_stop_during_provider_call_prevents_returned_tools(tmp_path):
    from quantdesk.agents import ResearchStopped
    store=Store(tmp_path); cid=store.campaign({},100000)
    call=SimpleNamespace(id='tool',function=SimpleNamespace(name='recall',arguments='{}'))
    def create(**kwargs):
        store.set_status(cid,'STOP_REQUESTED')
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=123),choices=[SimpleNamespace(message=Message(tool_calls=[call]),finish_reason='stop')])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class Tools:
        def available(self,role): return []
        def call(self,*args): raise AssertionError('Must stop before tools')
    runtime=AgentRuntime(store,Settings(),client=client,model='test',provider='mimo')
    with pytest.raises(ResearchStopped):
        runtime.run(cid,'director',{},Tools())
    assert store.status()['usage'][0]['tokens']==123


def test_historian_cannot_create_an_empirical_lesson_without_observations(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    ident,_=store.candidate(cid,spec(),CODE)
    tools=ResearchTools(cid,store,None,None,Settings())
    args={'candidate_id':ident,'lesson':'A claim','evidence':'An assertion','next_experiment':'Measure it'}
    with pytest.raises(ValueError,match='actual experiment'):
        tools.call('historian','record_lesson',args)
    store.memory(cid,ident,'DEVELOPMENT_EXPERIMENT',{'pnl':-5,'sampled':True})
    tools.call('historian','record_lesson',args)
    lesson=next(m for m in store.recall() if m['kind']=='LESSON')
    assert lesson['payload']['grounding']['development_experiment']['pnl']==-5


def test_slow_provider_cannot_hold_task_or_execute_late_tools(tmp_path):
    import time
    store=Store(tmp_path); cid=store.campaign({},100000)
    def create(**kwargs):
        time.sleep(0.1)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=100),choices=[SimpleNamespace(message=Message(content='{"late":true}'),finish_reason='stop')])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class Tools:
        def available(self,role): return []
    runtime=AgentRuntime(store,replace(Settings(),request_timeout=0.01),client=client,model='test',provider='mimo')
    with pytest.raises(RuntimeError,match='no synthetic replacement'):
        runtime.run(cid,'director',{},Tools())
    with store.connect() as db:
        request=db.execute('SELECT status,actual FROM requests').fetchone()
        assert request['status']=='FAILED' and request['actual'] is None
        assert not db.execute("SELECT 1 FROM events WHERE kind='AGENT_COMPLETED'").fetchone()


def test_two_complete_cycles_retrieve_grounded_partial_screen_lessons(tmp_path):
    from quantdesk.agents import Desk
    store=Store(tmp_path)
    records=[{'date':f'2026-01-{i:02d}'} for i in range(1,11)]
    plan={'development':records[:6],'validation':records[6:9],'holdout':records[9:],'hash':'synthetic-integration','version':'test','sampled':True,'max_rows_per_day':80,'limitations':['Synthetic regression fixture']}
    cid=store.campaign(plan,100000)
    class CatalogFixture:
        def load(self,*args): return frame(),{'sequence_gaps':0}
        def profile(self,*args): return {'rows':80,'date':args[0]['date']}
    class RuntimeFixture:
        missions=[]
        def run(self,cid,role,task,tools,task_id=None):
            if role=='director':
                self.missions.append(task)
                return {'mission':'Test causal pressure and reject unsupported profits'}
            if role=='scientist':
                store.start_task(task_id,cid,role,task); tools.active_task_id=task_id
                result=tools.call(role,'submit_proposal',{'strategy_spec':spec(),'effective_history_rows':1,'development_indices':[0],
                    'hypothesis':'Fixture causal pressure','implementation_design':'Bounded qi_5','baseline':'Cash and fixed trading',
                    'ablation':'Remove causal pressure','stop_condition':'Reject inadequate full-day evidence','limitations':'Synthetic partial samples'})
                store.finish_task(task_id,result)
                return result
            if role=='engineer':
                tools.call(role,'experiment',{'spec':spec(),'code':CODE,'days':1})
                return tools.call(role,'register_candidate',{'spec':spec(),'code':CODE})
            if role=='falsifier':
                c=tools.call(role,'read_candidate',{'candidate_id':task['cohort'][0]['candidate_id']})
                return {'status':c['status'],'reasons':c['report']['reasons']}
            for member in task['cohort']:
                tools.call(role,'record_lesson',{'candidate_id':member['candidate_id'],'lesson':'Partial samples cannot establish deployable evidence','evidence':json.dumps(member['assessment']['reasons']),'next_experiment':'Collect sufficient independent full dates'})
            return {'lessons_recorded':len(task['cohort'])}
    settings=replace(Settings(),home=tmp_path)
    catalog=CatalogFixture(); runtime=RuntimeFixture()
    result=Desk(store,catalog,Evaluator(store,catalog,settings),settings,runtime).run(cid,2)
    assert result['status']=='READY'
    submitted=next(c for c in store.candidates(cid) if c['title']==spec()['title'])
    assert submitted['status']=='PROPOSED'
    assert submitted['report']['status']=='DEVELOPMENT_INCONCLUSIVE'
    assert len(runtime.missions)==2
    assert all(outcome['kind']=='STRATEGY_COHORT' and outcome['decision_id'] for outcome in result['cycle_outcomes'])
    assert runtime.missions[1]['research_map']['completed_strategy_cohorts']==1
    assert any(m['kind']=='LESSON' for m in runtime.missions[1]['memory'])
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM cycles WHERE status='COMPLETE'").fetchone()[0]==2
        assert db.execute('SELECT COUNT(*) FROM holdouts').fetchone()[0]==0


def test_engineer_submits_existing_code_without_cross_campaign_adoption(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},10000); other=store.campaign({},10000)
    ident,_=store.candidate(cid,spec(),CODE)
    tools=ResearchTools(cid,store,None,None,Settings(),cycle_id='cycle')
    assert tools.call('engineer','register_existing_candidate',{'candidate_id':ident})['candidate_id']==ident
    assert store.get_candidate(ident)['code']==CODE
    with pytest.raises(ValueError,match='belong'):
        ResearchTools(other,store,None,None,Settings()).call('engineer','register_existing_candidate',{'candidate_id':ident})


def test_real_artifact_commit_finishes_engineer_without_a_summary_call(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    ident,_=store.candidate(cid,spec(),CODE); calls=[]
    tool_call=SimpleNamespace(id='submit',function=SimpleNamespace(name='register_existing_candidate',arguments=json.dumps({'candidate_id':ident})))
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=100),choices=[SimpleNamespace(message=Message(tool_calls=[tool_call]),finish_reason='stop')])
    runtime=AgentRuntime(store,Settings(),client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),model='test',provider='mimo')
    result=runtime.run(cid,'engineer',{},ResearchTools(cid,store,None,None,Settings(),cycle_id='cycle'),task_id='cycle:engineer')
    assert result['artifact_receipt'] and result['committed_candidate_ids']==[ident]
    assert len(calls)==1
    with store.connect() as db:
        assert db.execute('SELECT status FROM tasks').fetchone()[0]=='COMPLETE'


def test_trade_attribution_reconciles_gross_capture_costs_and_net_pnl():
    broker=Broker(spec(holding_seconds=1),Settings().execution_dict())
    broker.step(quote(0),1); broker.step(quote(1000),0)
    broker.step(quote(2000),0); broker.step(quote(3000),0)
    trade=broker.trades[0]
    assert trade['gross_mid_pnl']==0
    assert trade['execution_drag']>0 and trade['fees']>0
    assert trade['pnl']==pytest.approx(trade['gross_mid_pnl']-trade['execution_drag']-trade['fees'])


def test_execution_diagnostics_cannot_expose_validation_to_exploration(tmp_path):
    store=Store(tmp_path); cid=store.campaign({},100000)
    ident,_=store.candidate(cid,spec(),CODE)
    tools=ResearchTools(cid,store,None,None,Settings())
    with pytest.raises(ValueError,match='cannot inspect'):
        tools.call('scientist','diagnose_execution',{'candidate_id':ident,'phase':'validation','day_index':0})


def test_stop_between_archive_replays_preserves_unfinished_candidate(tmp_path):
    from quantdesk.control import ResearchStopped
    store=Store(tmp_path); records=[{'date':'2026-01-01'},{'date':'2026-01-02'}]
    plan={'development':[],'validation':records,'hash':'stop-fixture','version':'test','sampled':True,'limitations':[]}
    cid=store.campaign(plan,100000); ident,_=store.candidate(cid,spec(),CODE)
    class CatalogFixture:
        calls=0
        def load(self,*args):
            self.calls+=1
            store.set_status(cid,'STOP_REQUESTED')
            return frame(),{'sequence_gaps':0}
    catalog=CatalogFixture(); evaluator=Evaluator(store,catalog,replace(Settings(),home=tmp_path))
    with pytest.raises(ResearchStopped):
        evaluator.validate(cid,ident)
    assert catalog.calls==1
    assert store.get_candidate(ident)['status']=='PROPOSED'
