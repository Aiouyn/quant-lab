"""Screening must preserve research/confirmation boundaries and exact checkpoints."""
import json
from dataclasses import replace
import pytest
from quantdesk.store import Store,digest
from quantdesk.settings import Settings
from quantdesk.evaluation import Evaluator,daily_statistics
from quantdesk.screening import classify,available_indices
from quantdesk.agents import ResearchTools,compact_cohort
from tests.test_quantdesk import CODE,spec,frame


def setup(tmp_path,sampled=False):
    records=[{'date':f'2026-01-{i:02d}'} for i in range(1,31)]
    plan={'development':records[:8],'validation':records[8:28],'holdout':records[28:],
        'hash':'screen-fixture','version':'fixture','sampled':sampled,'max_rows_per_day':None,
        'limitations':['Synthetic screening boundary test']}
    store=Store(tmp_path); cid=store.campaign(plan,100000)
    ident,_=store.candidate(cid,spec(),CODE)
    class Catalog:
        prepared={0,1,2,3,4,5}
        reads=[]
        def cached(self,record,*args):
            return plan['development'].index(record) in self.prepared
        def load(self,record,*args):
            assert record in plan['development']
            self.reads.append(record['date'])
            return frame(),{'sequence_gaps':0}
    catalog=Catalog(); settings=replace(Settings(),home=tmp_path)
    return store,cid,store.get_candidate(ident),catalog,Evaluator(store,catalog,settings)


def replay_fixture(evaluator,monkeypatch,pnls=(0,0,0),trades=0,fail_baseline=False):
    calls=[]
    def evaluate(cid,candidate,phase='validation',**kwargs):
        assert phase=='development'
        calls.append((candidate['id'],list(kwargs['development_indices'])))
        if fail_baseline and candidate['code']!=CODE:
            raise OSError('simulated storage fault')
        records=evaluator.store.get_campaign(cid)['plan']['development']
        days=[{'date':records[index]['date'],'duration_seconds':86399,'rows':86400,
            'quality':{'has_sequence_ids':False},'pnl':pnl,'return_fraction':pnl/10000,
            'trades':trades,'halted':False,'funding_boundary_crossings':0}
            for index,pnl in zip(kwargs['development_indices'],pnls)]
        return {'days':days,'statistics':daily_statistics([d['return_fraction'] for d in days]),
            'pnl':sum(d['pnl'] for d in days),'trades':sum(d['trades'] for d in days),
            'sampled':evaluator.store.get_campaign(cid)['plan']['sampled'],
            'execution':evaluator.settings.execution_dict()}
    monkeypatch.setattr(evaluator,'evaluate',evaluate)
    return calls


def test_actual_partial_replay_cannot_confirm_or_consume_holdout(tmp_path):
    store,cid,candidate,catalog,evaluator=setup(tmp_path,True)
    report=evaluator.screen(cid,candidate)
    assert report['status']=='DEVELOPMENT_INCONCLUSIVE' and not report['confirmation_eligible']
    assert report['phase']=='development' and report['promotion'] is False
    assert set(catalog.reads)=={'2026-01-01','2026-01-03','2026-01-06'}
    assert store.get_candidate(candidate['id'])['status']=='PROPOSED'
    with store.connect() as db:
        assert not db.execute('SELECT 1 FROM holdouts').fetchone()
        assert not db.execute("SELECT 1 FROM data_exposures WHERE phase!='development'").fetchone()


def test_zero_trade_screen_reuses_frozen_dates_after_preparation_expands(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    calls=replay_fixture(evaluator,monkeypatch)
    first=evaluator.screen(cid,candidate)
    catalog.prepared.update({6,7})
    resumed=evaluator.screen(cid,store.get_candidate(candidate['id']))
    assert resumed['already_measured'] and resumed['selection']['indices']==[0,2,5]
    assert first['status']=='DEVELOPMENT_INACTIVE' and len(calls)==2
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='DEVELOPMENT_SCREEN_PLANNED'").fetchone()[0]==1


def test_positive_screen_never_promotes_and_requires_sufficient_confirmation_plan(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    replay_fixture(evaluator,monkeypatch,(2,-1,3),trades=3)
    report=evaluator.screen(cid,candidate)
    assert report['status']=='DEVELOPMENT_SCREEN_PASSED'
    assert not report['confirmation_eligible'] and report['confirmation_reasons']
    assert store.get_candidate(candidate['id'])['status']=='PROPOSED'
    with pytest.raises(ValueError,match='No candidates'):
        evaluator.seal(cid)


def test_operational_failure_retains_partial_evidence_and_resumes_without_replay(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    calls=replay_fixture(evaluator,monkeypatch,fail_baseline=True)
    failed=evaluator.screen(cid,candidate)
    assert failed['status']=='EVALUATION_ERROR' and 'operational' in failed['failure_kind']
    assert store.get_candidate(candidate['id'])['status']=='PROPOSED'
    repaired_calls=replay_fixture(evaluator,monkeypatch)
    resumed=evaluator.screen(cid,store.get_candidate(candidate['id']))
    assert resumed['status']=='DEVELOPMENT_INACTIVE'
    assert len(calls)==2 and len(repaired_calls)==1
    assert repaired_calls[0][0]!=candidate['id']


@pytest.mark.parametrize('indices',[[8],[True],[0,0],[-1],[0,1,2,3]])
def test_invalid_indices_cannot_reach_validation_or_create_a_plan(tmp_path,indices):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    with pytest.raises(ValueError,match='development indices'):
        evaluator.screen(cid,candidate,indices)
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM events WHERE kind='DEVELOPMENT_SCREEN_PLANNED'").fetchone()


def test_grounding_and_compact_cohort_keep_development_honest(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    replay_fixture(evaluator,monkeypatch)
    report=evaluator.screen(cid,candidate)
    tools=ResearchTools(cid,store,catalog,evaluator,evaluator.settings)
    tools.call('historian','record_lesson',{'candidate_id':candidate['id'],'lesson':'No activity in this configuration',
        'evidence':'Observed screen','next_experiment':'Test a distinct mechanism against costs'})
    with store.connect() as db:
        lesson=json.loads(db.execute("SELECT payload FROM memories WHERE kind='LESSON'").fetchone()[0])
    grounding=lesson['grounding']
    assert grounding['assessment_phase']=='development' and not grounding['validation_reasons']
    assert grounding['development_experiment']['screen_key']==report['screen_key']
    compact=compact_cohort([{'candidate_id':candidate['id'],'assessment':report}])[0]
    assert compact['evidence_phase']=='development' and 'validation' not in compact
    assert compact['assessment']['development']['evaluated_dates']==report['selection']['dates']


def test_matched_baseline_does_not_inherit_candidate_input_restriction(tmp_path,monkeypatch):
    store,cid,candidate,catalog,evaluator=setup(tmp_path)
    restricted=dict(candidate['spec'],input_features=['qi_5'])
    ident,_=store.candidate(cid,restricted,CODE); candidate=store.get_candidate(ident)
    replay_fixture(evaluator,monkeypatch)
    evaluator.screen(cid,candidate)
    baseline=next(c for c in store.candidates(cid) if c['title'].startswith('REFERENCE'))
    assert 'input_features' not in baseline['spec']


def test_fitted_screen_preserves_earlier_training_and_ablation(tmp_path,monkeypatch):
    from tests.test_fitted_models import desk
    store,cid,candidate,catalog,evaluator,settings=desk(tmp_path,monkeypatch)
    report=evaluator.screen(cid,candidate,[6,8,11])
    development=report['development']
    assert len(development['model_artifacts'])==3
    assert 'without_custom_alpha' in development['comparisons']
    assert report['status']=='DEVELOPMENT_INCONCLUSIVE'
    for day in development['daily_results']:
        assert all(date<day['date'] for date in day['model']['training_dates'])
    assert not any(date>='2026-01-13' for date in catalog.reads)


def test_equal_or_better_ablation_blocks_automatic_confirmation():
    evidence={'daily_results':[{'duration_seconds':86400,'pnl':1}]*3,
        'pnl':3,'trades':6,'comparisons':{'without_custom_alpha':{'pnl':4}}}
    status,reasons=classify(evidence,'taker')
    assert status=='DEVELOPMENT_NO_INCREMENT'
    assert 'matched' in reasons[0]
