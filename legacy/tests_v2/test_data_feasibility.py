"""Impossible temporal windows must fail before expensive authored programs."""
import json
from dataclasses import replace
import numpy as np
import polars as pl
import pytest
from quantdesk.agents import ResearchTools
from quantdesk.models import ModelService,label_feasibility
from quantdesk.settings import Settings
from quantdesk.store import Store
from tests.test_fitted_models import desk,frame,spec
from tests.test_agent_completion import runtime,response
from types import SimpleNamespace


def segmented(date='2026-01-01'):
    return frame(date,1800).with_columns(pl.Series('segment',np.repeat([0,1],900)))


def test_warmup_and_delayed_holding_need_one_continuous_segment():
    data=segmented(); strategy=spec(); strategy['holding_seconds']=300
    strategy['warmup_rows']=1000
    result=label_feasibility(data,strategy,Settings().execution_dict(),strategy['model'])
    assert result['status']=='NO_USABLE_LABELS' and result['eligible_label_rows']==0
    assert result['segment_coverage']['maximum_rows']==900
    assert result['maximum_observed_feasible_warmup_rows']==597
    assert 'Warmup consumes' in result['reasons'][0]
    strategy['warmup_rows']=600
    result=label_feasibility(data,strategy,Settings().execution_dict(),strategy['model'])
    assert result['eligible_label_rows']==0 and 'delayed holding' in result['reasons'][0]
    strategy['warmup_rows']=300
    assert label_feasibility(data,strategy,Settings().execution_dict(),strategy['model'])['eligible_label_rows']>0


def test_model_preflight_never_evaluates_program_on_impossible_labels(tmp_path,monkeypatch):
    store,cid,candidate,catalog,_,settings=desk(tmp_path,monkeypatch)
    candidate['spec']['holding_seconds']=300; candidate['spec']['warmup_rows']=1000
    for date in catalog.frames:
        catalog.frames[date]=segmented(date)
    service=ModelService(store,catalog,settings)
    monkeypatch.setattr(service,'alpha',lambda *a:pytest.fail('Impossible recipe must fail before program execution'))
    records=store.get_campaign(cid)['plan']['development'][:8]
    cutoff=int(catalog.frames['2026-01-09']['ts_ms'][0])
    with pytest.raises(ValueError,match='inner training=0.*maximum observed segment=900'):
        service.fit(cid,candidate,records,cutoff)
    with store.connect() as db:
        recorded=json.loads(db.execute("SELECT payload FROM memories WHERE kind='MODEL_DATA_FEASIBILITY'").fetchone()[0])
    assert recorded['inner_selection_rows']==0 and len(recorded['dates'])==8
    assert not list((tmp_path/'models').glob('*.json'))


def test_feasibility_tool_prefers_complete_prepared_frame_and_rejects_holdout(tmp_path):
    store=Store(tmp_path); record={'date':'2026-01-01'}
    cid=store.campaign({'development':[record],'holdout':[{'date':'2026-01-02'}],'max_rows_per_day':None,'hash':'fixture'},1000)
    reads=[]
    class Catalog:
        def cached(self,record,rows):return True
        def load(self,record,rows):
            reads.append((record,rows)); return segmented(),{}
    tools=ResearchTools(cid,store,Catalog(),None,replace(Settings(),home=tmp_path))
    strategy=spec(); strategy['warmup_rows']=1000
    report=tools.call('director','data_feasibility',{'spec':strategy})
    assert report['full_prepared_frame_used'] and report['status']=='NO_USABLE_LABELS'
    assert reads==[(record,None)] and 'target' not in report
    with pytest.raises(ValueError,match='development date'):
        tools.call('engineer','data_feasibility',{'spec':strategy,'day_index':1})


def test_unprepared_feasibility_is_explicitly_bounded(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'fixture'},1000)
    class Catalog:
        def load(self,record,rows):
            assert rows==10000
            return segmented(),{}
    report=ResearchTools(cid,store,Catalog(),None,Settings()).call('scientist','data_feasibility',{'spec':spec()})
    assert not report['full_prepared_frame_used'] and report['bounded_rows_requested']==10000


def test_invalid_recipe_still_receives_observed_history_constraints(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'fixture'},1000)
    class Catalog:
        def cached(self,*a):return True
        def load(self,*a):return segmented(),{}
    strategy=spec(); strategy['warmup_rows']=1000
    strategy['model'].update(train_days=25,min_train_days=10,inner_days=15)
    report=ResearchTools(cid,store,Catalog(),None,Settings()).call('engineer','data_feasibility',{'spec':strategy})
    assert report['status']=='INVALID_RECIPE' and report['label_window_status']=='NO_USABLE_LABELS'
    assert report['segment_coverage']['maximum_rows']==900 and report['eligible_label_rows']==0
    assert 'inner_days <= 7' in report['recipe_error'] and 'min_train_days >= 18' in report['recipe_error']
    assert strategy['model']['inner_days']==15 and not store.candidates(cid)


def test_failed_engineer_receives_measured_recipe_incompatibility_on_resume(tmp_path):
    store=Store(tmp_path); cid=store.campaign({'development':[{'date':'2026-01-01'}],'hash':'fixture'},100000)
    strategy=spec(); strategy['warmup_rows']=1000
    code="import polars as pl\ndef compute_signal(df):\n    return df['qi_5'].clip(-1.0,1.0)\n"
    store.start_task('resume-fit',cid,'engineer',{})
    store.event(cid,'engineer','PROGRAM_ATTEMPT',{'task_id':'resume-fit','args':{'spec':strategy,'code':code}})
    store.event(cid,'engineer','TOOL_RESULT',{'task_id':'resume-fit','tool':'walk_forward','result':{'error':'Insufficient inner temporal validation labels'}})
    store.fail_task('resume-fit',RuntimeError('step budget'))
    class Catalog:
        def cached(self,*a):return True
        def load(self,*a):return segmented(),{}
    tools=ResearchTools(cid,store,Catalog(),None,Settings())
    def create(**kwargs):
        task=json.loads(kwargs['messages'][1]['content'])
        assert task['unfinished_program']['args']['code']==code
        assert task['measured_data_feasibility']['status']=='NO_USABLE_LABELS'
        assert task['measured_data_feasibility']['segment_coverage']['maximum_rows']==900
        assert any(t['function']['name']=='data_feasibility' for t in kwargs['tools'])
        submission=SimpleNamespace(id='candidate',function=SimpleNamespace(name='register_candidate',arguments=json.dumps({'spec':strategy,'code':code})))
        return response(submission)
    result=runtime(store,replace(Settings(),max_agent_steps=1),create).run(cid,'engineer',{},tools,'resume-fit')
    assert result['artifact_receipt']
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE agent='runtime' AND kind='DATA_FEASIBILITY'").fetchone()[0]==1
