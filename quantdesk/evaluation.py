"""Chronological development, reusable validation, once-only sealed holdout.

Statistical confidence is an explicit, conservative daily block bootstrap with
family-wise correction. No annualized trade Sharpe is passed off as DSR or PBO.
"""
import json
import shutil
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import numpy as np
import polars as pl
from .execution import replay
from .strategy import run_signal,run_feature_program,validate_spec,VALIDATOR_VERSION
from .store import digest
from .control import ResearchStopped

REFERENCE_CODE = '''import polars as pl
import numpy as np
def compute_signal(df):
    return pl.Series('signal', np.tanh(df['qi_5'].to_numpy() + 0.25 * df['ofi_norm'].clip(-3.0,3.0).to_numpy()))
'''

def daily_statistics(returns,trials=1):
    x=np.asarray(returns,dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        return {'days':0,'mean_daily_return':0.0,'daily_sharpe':0.0,'annualized_sharpe':None,'positive_days_fraction':0.0,'raw_p_value':1.0,'adjusted_p_value':1.0,'lower_mean_95':0.0}
    mean=float(x.mean()); std=float(x.std(ddof=1)) if len(x)>1 else 0.0
    raw=1.0; lower=mean
    if len(x)>=5 and std>1e-12:
        rng=np.random.default_rng(731)
        block=max(2,min(5,int(np.sqrt(len(x)))))
        draws=max(4000,min(200000,80*max(1,int(trials))))
        batches=[]
        for start in range(0,draws,500):
            count=min(500,draws-start)
            starts=rng.integers(0,len(x),size=(count,int(np.ceil(len(x)/block))))
            indices=(starts[:,:,None]+np.arange(block))%len(x)
            batches.append(x[indices.reshape(count,-1)[:,:len(x)]].mean(axis=1))
        boot=np.concatenate(batches)
        centered=boot-mean
        raw=float((1+np.sum(centered>=mean))/(len(boot)+1)) if mean>0 else 1.0
        lower=float(np.quantile(boot,0.05))
    ds=mean/std if std>1e-12 else 0.0
    return {'days':len(x),'mean_daily_return':mean,'daily_sharpe':ds,'annualized_sharpe':ds*np.sqrt(365) if len(x)>=20 else None,'positive_days_fraction':float((x>0).mean()),'raw_p_value':raw,'adjusted_p_value':min(1.0,raw*max(trials,1)),'lower_mean_95':lower,'bootstrap_draws':len(boot) if len(x)>=5 and std>1e-12 else 0,'method':'moving_daily_block_bootstrap; one_sided; Bonferroni across evaluated candidates and experiments','assumptions':'Approximate stationarity within the tested period; not a probability of future profitability'}

class Evaluator:
    def __init__(self,store,catalog,settings):
        self.store,self.catalog,self.settings=store,catalog,settings

    def trial_count(self,plan_hash):
        with self.store.connect() as db:
            rows=db.execute("SELECT payload FROM events WHERE kind='EXPERIMENT_STARTED'").fetchall()
        # Conservative lab-wide family: changing campaign names, sampled dates,
        # or code labels cannot reset the multiple-testing penalty.
        return max(1,sum(max(1,int(json.loads(r[0]).get('trials',1))) for r in rows))

    def diagnose_prediction(self,cid,candidate,index):
        from .models import ModelService,prediction_frontier
        plan=self.store.get_campaign(cid)['plan']
        if candidate['campaign']!=cid or 'model' not in candidate['spec'] or not 0<=index<len(plan['development']):
            raise ValueError('Prediction diagnostics require this campaign fitted candidate and a development index')
        record=plan['development'][index]
        self.store.expose_dates(cid,[record],'development')
        frame,_=self.catalog.load(record,plan.get('max_rows_per_day'))
        earlier=[r for r in plan['development'] if r['date']<record['date']]
        service=ModelService(self.store,self.catalog,self.settings)
        artifact,path=service.fit(cid,candidate,earlier,int(frame['ts_ms'][0]))
        alpha=service.alpha(candidate,record,frame,plan.get('max_rows_per_day')) if artifact['recipe']['include_alpha'] else np.zeros(len(frame))
        result=dict(prediction_frontier(frame,alpha,artifact,candidate['spec']),
            candidate_id=candidate['id'],date=record['date'],model_hash=artifact['artifact_hash'],
            model_path=str(path),phase='development',duration_seconds=float(frame['ts_ms'][-1]-frame['ts_ms'][0])/1000)
        self.store.memory(cid,candidate['id'],'PREDICTION_DIAGNOSTIC',result)
        return result

    def evaluate(self,cid,candidate,phase='validation',limit_days=None,stress=False,development_indices=None):
        if phase not in {'development','validation','holdout'}:
            raise ValueError('Unknown evaluation phase')
        campaign=self.store.get_campaign(cid); plan=campaign['plan']
        spec=validate_spec(candidate['spec'])
        records=plan[phase][:limit_days] if limit_days else plan[phase]
        if development_indices is not None:
            if phase!='development' or not 1<=len(development_indices)<=3 or len(set(development_indices))!=len(development_indices) or any(not 0<=i<len(plan['development']) for i in development_indices):
                raise ValueError('Explicit experiment indices must name 1..3 distinct development dates')
            records=[plan['development'][i] for i in development_indices]
        self.store.event(cid,'validator','EXPERIMENT_STARTED',{'candidate':candidate['id'],'phase':phase,'stress':stress,'plan_hash':plan['hash']})
        summaries=[]; execution=self.settings.execution_dict()
        model_artifact = None
        if 'model' in spec and phase != 'development':
            from .models import ModelService, verify_artifact
            if phase == 'holdout':
                reference = (candidate.get('report') or {}).get('base',{}).get('model_artifact')
                if not reference:
                    raise ValueError('Holdout needs the exact fitted artifact frozen during validation')
                model_artifact = json.loads(Path(reference['path']).read_text(encoding='utf-8'))
                verify_artifact(model_artifact)
                if model_artifact['artifact_hash'] != reference['hash'] or model_artifact['spec_hash'] != digest(spec) or model_artifact['code_hash'] != digest(candidate['code']):
                    raise ValueError('Frozen model changed after validation')
                model_path = Path(reference['path'])
            else:
                cutoff = int(datetime.fromisoformat(plan['validation'][0]['date']).replace(tzinfo=timezone.utc).timestamp()*1000)
                train = [r for r in plan['development'] if r['date'] < plan['validation'][0]['date']]
                model_artifact, model_path = ModelService(self.store,self.catalog,self.settings).fit(cid,candidate,train,cutoff)
        if stress:
            execution=dict(execution,taker_fee_bps=execution['taker_fee_bps']*1.5,maker_fee_bps=execution['maker_fee_bps']*1.5,slippage_bps=max(1.0,execution['slippage_bps']*2),latency_ms=execution['latency_ms']*2)
        output=self.store.home/'replays'/candidate['id']/phase/('stress' if stress else 'base')
        output.mkdir(parents=True,exist_ok=True)
        if phase in {'development','validation'}:
            # Record exposure before loading or running any code; crashes cannot
            # make inspected dates fresh holdouts in another campaign.
            self.store.expose_dates(cid,records,phase)
        for record in records:
            self.store.check_research_control(cid)
            frame,quality=self.catalog.load(record,plan.get('max_rows_per_day'))
            if 'model' in spec and phase == 'development':
                from .models import ModelService
                train = [r for r in plan['development'] if r['date'] < record['date']]
                cutoff = int(frame['ts_ms'][0])
                model_artifact, model_path = ModelService(self.store,self.catalog,self.settings).fit(cid,candidate,train,cutoff)
            if model_artifact and int(frame['ts_ms'][0]) < model_artifact['provenance']['prediction_cutoff_ms']:
                raise ValueError('Prediction observations precede frozen training cutoff')
            signal_directory=self.store.home/'signals'; signal_directory.mkdir(parents=True,exist_ok=True)
            signal_path=signal_directory/(digest({'record':record,'rows':plan.get('max_rows_per_day'),'features':plan['version'],'validator':VALIDATOR_VERSION,'code':candidate['code'],'input_features':spec.get('input_features'), 'model':model_artifact['artifact_hash'] if model_artifact else None})+'.npy')
            if signal_path.exists():
                signals=np.load(signal_path,allow_pickle=False)
            else:
                runner=run_feature_program if spec.get('model',{}).get('custom_features',False) else run_signal
                from .input_contract import program_input
                signals=runner(candidate['code'],program_input(frame,spec),self.settings.worker_timeout)
                if model_artifact:
                    from .models import fitted_signal
                    signals=fitted_signal(frame,signals,model_artifact)
                np.save(signal_path,signals,allow_pickle=False)
            summary,equity,trades=replay(frame,signals,spec,execution)
            trade_log=output/(record['date']+'.trades.jsonl')
            with trade_log.open('w',encoding='utf-8') as stream:
                for trade in trades:
                    stream.write(json.dumps(trade,allow_nan=False)+'\n')
            duration=(int(frame['ts_ms'][-1])-int(frame['ts_ms'][0]))/1000
            summary.update(date=record['date'],rows=len(frame),duration_seconds=duration,quality=quality,
                closed_trade_gross_mid_pnl=sum(t['gross_mid_pnl'] for t in trades),
                closed_trade_execution_drag=sum(t['execution_drag'] for t in trades),trade_log=str(trade_log))
            if model_artifact:
                summary['model']={'hash':model_artifact['artifact_hash'],'training_dates':model_artifact['provenance']['training_dates'],
                    'max_label_end_ms':model_artifact['provenance']['max_label_end_ms'],'prediction_cutoff_ms':model_artifact['provenance']['prediction_cutoff_ms'],
                    'selected_alpha':model_artifact['fitted'].get('alpha'),
                    'selected_parameters':model_artifact['fitted'].get('parameters'),
                    'family':model_artifact['fitted']['family'],
                    'inner_diagnostics':model_artifact['provenance'].get('selected_inner_diagnostics')}
            summaries.append(summary)
            equity.write_parquet(output/(record['date']+'.parquet'))
        stats=daily_statistics([r['return_fraction'] for r in summaries],self.trial_count(plan['hash']))
        report = {'phase':phase,'stress':stress,'candidate_id':candidate['id'],'plan_hash':plan['hash'],'spec_hash':digest(spec),'code_hash':digest(candidate['code']),'execution':execution,'statistics':stats,'days':summaries,'trades':sum(r['trades'] for r in summaries),'pnl':sum(r['pnl'] for r in summaries),'replay_directory':str(output),'sampled':plan['sampled'],'fill_model':spec['mode'],'limits':plan['limitations']}
        if model_artifact:
            report['model_artifact']={'path':str(model_path),'hash':model_artifact['artifact_hash'],
                'policy':'nested temporal selection on earlier development dates; frozen across validation and holdout'}
        return report

    def screen(self,cid,candidate,development_indices=None):
        from .screening import screen
        return screen(self,cid,candidate,development_indices)

    def walk_forward(self,cid,candidate,folds=5,start_index=None,end_index=None,development_indices=None,checkpoint_key=None):
        """Outer unseen development dates, each with strictly earlier inner fitting."""
        plan=self.store.get_campaign(cid)['plan']
        if 'model' not in candidate['spec']:
            raise ValueError('Walk-forward fitting requires a fitted-model recipe')
        from .models import validate_recipe
        recipe=validate_recipe(candidate['spec']['model'])
        start=recipe['min_train_days'] if start_index is None else int(start_index)
        end=len(plan['development'])-1 if end_index is None else int(end_index)
        if not 1<=folds<=20 or not recipe['min_train_days']<=start<=end<len(plan['development']):
            raise ValueError('Need 1..20 folds and sufficient earlier development dates')
        indices=np.unique(np.linspace(start,end,min(folds,end-start+1),dtype=int))
        if development_indices is not None:
            if not 1<=len(development_indices)<=3 or len(set(development_indices))!=len(development_indices) or any(type(i) is not int or not recipe['min_train_days']<=i<len(plan['development']) for i in development_indices):
                raise ValueError('Select 1..3 outer development indices with sufficient earlier training')
            indices=development_indices
        def measured(member,index):
            key=digest({'checkpoint':checkpoint_key,'candidate':member['id'],'index':int(index)}) if checkpoint_key else None
            cached=self.store.cached_tool(key) if key else None
            if cached is not None:
                self.store.event(cid,'validator','DEVELOPMENT_REPLAY_REUSED',{'candidate':member['id'],'index':int(index),'key':key})
                return cached
            result=self.evaluate(cid,member,'development',development_indices=[int(index)])
            if key:
                self.store.cache_tool(key,result)
            return result
        dates=[]; baseline_dates=[]; ablation_dates=[]; model_artifacts=[]
        # Explicit matched comparisons use identical outer dates and execution.
        # They are fixed benchmarks, never described as AI discoveries.
        reference_spec={k:v for k,v in candidate['spec'].items() if k not in {'model','input_features'}}
        reference_spec.update(title='REFERENCE: causal depth-flow baseline',
            rationale='Fixed research benchmark; not an agent discovery',
            falsification='Compare net outer-date returns with the proposed fitted system')
        baseline_id,_=self.store.candidate(cid,reference_spec,REFERENCE_CODE)
        baseline=self.store.get_candidate(baseline_id)
        ablation=None
        if recipe['include_alpha']:
            ablation_spec=dict(candidate['spec'],title='ABLATION: remove custom alpha feature',
                rationale='Measure incremental contribution of the agent-authored causal feature',
                model=dict(recipe,include_alpha=False),parent_ids=[candidate['id']])
            ablation_id,_=self.store.candidate(cid,ablation_spec,candidate['code'])
            ablation=self.store.get_candidate(ablation_id)
        for i in indices:
            report=measured(candidate,i)
            dates.extend(report['days'])
            model_artifacts.append(dict(report['model_artifact'],date=report['days'][0]['date']))
            baseline_dates.extend(measured(baseline,i)['days'])
            if ablation:
                ablation_dates.extend(measured(ablation,i)['days'])
        def comparison(other,ident):
            delta=[a['return_fraction']-b['return_fraction'] for a,b in zip(dates,other)]
            return {'candidate_id':ident,'daily_net_returns':[d['return_fraction'] for d in other],
                'pnl':sum(d['pnl'] for d in other),'trades':sum(d['trades'] for d in other),
                'paired_daily_improvement':daily_statistics(delta,self.trial_count(plan['hash'])),
                'caveat':'Matched outer development dates; descriptive research comparison, not independent approval'}
        result={'candidate_id':candidate['id'],'status':'DEVELOPMENT_WALK_FORWARD_ONLY',
            'folds':len(indices),'outer_dates':[d['date'] for d in dates],
            'statistics':daily_statistics([d['return_fraction'] for d in dates],self.trial_count(plan['hash'])),
            'pnl':sum(d['pnl'] for d in dates),'trades':sum(d['trades'] for d in dates),
            'daily_results':dates,'sampled':plan['sampled'],
            'model_artifacts':model_artifacts,'execution':self.settings.execution_dict(),'plan_hash':plan['hash'],
            'comparisons':{'cash':{'pnl':0.0,'trades':0},'depth_flow_baseline':comparison(baseline_dates,baseline_id),
                **({'without_custom_alpha':comparison(ablation_dates,ablation['id'])} if ablation else {})},
            'closed_trade_attribution':{'gross_mid_pnl':sum(d['closed_trade_gross_mid_pnl'] for d in dates),
                'execution_drag':sum(d['closed_trade_execution_drag'] for d in dates),'fees':sum(d['fees'] for d in dates)},
            'caveat':'Outer development evidence guides research; it does not replace validation, sealed holdout or forward venue testing'}
        self.store.memory(cid,candidate['id'],'WALK_FORWARD_EXPERIMENT',result)
        return result

    def gate(self,report,phase='validation'):
        reasons=[]; s=report['statistics']
        minimum=self.settings.min_validation_days if phase=='validation' else self.settings.min_holdout_days
        if s['days']<minimum:
            reasons.append(f'Need {minimum} distinct {phase} days; observed {s["days"]}')
        if report['sampled']:
            reasons.append('Partial-day research samples cannot qualify for paper promotion')
        if any(d['duration_seconds']<18*3600 for d in report['days']):
            reasons.append('Require at least 18 hours of observations per evaluated date')
        if any(d['quality'].get('sequence_gaps',0) for d in report['days']):
            reasons.append('Archive sequence gaps require independent data repair/review')
        if any(d['quality'].get('timestamp_gaps',0) or d['quality'].get('crossed_books',0) for d in report['days']):
            reasons.append('Archive time gaps or crossed books require independent data quality review')
        if report['fill_model']=='maker':
            reasons.append('Maker results use an unverified proxy; collect trades and queue evidence before promotion')
        if report['trades']<self.settings.min_trades:
            reasons.append(f'Only {report["trades"]} trades; need {self.settings.min_trades}')
        if s['mean_daily_return']<=0 or s['lower_mean_95']<=0:
            reasons.append('Daily return mean and its block bootstrap lower bound must both exceed zero')
        if s['adjusted_p_value']>0.05:
            reasons.append('Evidence does not survive correction for the research family')
        if s['positive_days_fraction']<0.55:
            reasons.append('Fewer than 55% of dates are profitable')
        if any(d['halted'] for d in report['days']):
            reasons.append('Drawdown kill switch triggered')
        if any(d.get('funding_boundary_crossings',0) for d in report['days']):
            reasons.append('Unmodeled hourly funding was crossed; supply funding evidence before promotion')
        return reasons

    def validate(self,cid,ident):
        candidate=self.store.get_candidate(ident)
        if candidate['campaign']!=cid:
            raise ValueError('Candidate belongs to a different campaign')
        if candidate['status'] not in {'PROPOSED','EVALUATION_ERROR'}:
            return candidate['report']
        try:
            base=self.evaluate(cid,candidate)
            stress=self.evaluate(cid,candidate,stress=True)
            reasons=self.gate(base)+['Stress: '+r for r in self.gate(stress)]
            report={'base':base,'stress':stress,'reasons':reasons,'status':'VALIDATED' if not reasons else 'REJECTED'}
        except ResearchStopped:
            raise
        except Exception as error:
            report={'status':'EVALUATION_ERROR','reasons':[f'{type(error).__name__}: {error}'], 'failure_kind':'operational; hypothesis not falsified'}
        self.store.record_report(ident,report['status'],report)
        self.store.memory(cid,ident,'EVIDENCE',{'title':candidate['title'],'spec':candidate['spec'],'status':report['status'],'reasons':report['reasons'],'statistics':report.get('base',{}).get('statistics',{})})
        self.store.event(cid,'validator',report['status'],{'id':ident,'reasons':report['reasons']})
        self.dossier(candidate,report)
        return report

    def seal(self,cid):
        campaign=self.store.get_campaign(cid); plan=campaign['plan']
        if plan['sampled']:
            raise ValueError('Cannot seal a partial-day sampled campaign')
        if campaign['status']=='SEALED':
            raise ValueError('Campaign is already sealed')
        cohort=self.store.candidates(cid,status='VALIDATED')
        if not cohort:
            raise ValueError('No candidates have passed validation')
        # Family size is finalized BEFORE anyone sees the holdout.
        count=self.trial_count(plan['hash'])
        eligible=[]
        for candidate in cohort:
            report=candidate['report']
            if report['base']['execution']!=self.settings.execution_dict():
                raise ValueError('Execution assumptions changed after validation. Start a new campaign with precommitted costs and risk limits.')
            for key in ('base','stress'):
                stats=report[key]['statistics']
                stats['adjusted_p_value']=min(1,stats['raw_p_value']*count)
            reasons=self.gate(report['base'])+self.gate(report['stress'])
            if reasons:
                report.update(status='REJECTED',reasons=reasons)
                self.store.record_report(candidate['id'],'REJECTED',report)
            else:
                eligible.append(candidate)
        if not eligible:
            raise ValueError('No candidates survive final family-wise correction')
        self.store.reserve_holdout(plan['hash'],[c['id'] for c in eligible],plan['holdout'])
        self.store.set_status(cid,'SEALED')
        results=[]
        for candidate in eligible:
            report=candidate['report']
            try:
                holdout=self.evaluate(cid,candidate,'holdout')
                # Include every frozen cohort member in the holdout family.
                stats=holdout['statistics']; stats['adjusted_p_value']=min(1,stats['raw_p_value']*len(eligible))
                reasons=self.gate(holdout,'holdout')
                report.update(holdout=holdout,status='PAPER_READY' if not reasons else 'HOLDOUT_REJECTED',reasons=reasons)
            except Exception as error:
                report.update(status='HOLDOUT_REJECTED',reasons=[str(error)])
            self.store.record_report(candidate['id'],report['status'],report)
            self.dossier(candidate,report)
            if report['status']=='PAPER_READY':
                self.export(candidate,report)
            results.append({'id':candidate['id'],'status':report['status'],'reasons':report['reasons']})
        return results

    def export(self,candidate,report,exploratory=False):
        report=dict(report)
        development=None
        if exploratory and 'model' in candidate['spec'] and not report.get('base',{}).get('model_artifact'):
            with self.store.connect() as db:
                records=db.execute("SELECT payload FROM memories WHERE candidate=? AND kind IN ('WALK_FORWARD_EXPERIMENT','DEVELOPMENT_EXPERIMENT') ORDER BY id DESC",(candidate['id'],)).fetchall()
            for row in records:
                observed=json.loads(row[0])
                references=observed.get('model_artifacts') or ([observed['model_artifact']] if observed.get('model_artifact') else [])
                if references:
                    development=observed
                    report['base']={'model_artifact':references[-1],
                        'execution':observed['execution'],'plan_hash':observed['plan_hash'],'phase':'development'}
                    break
        directory=self.store.home/'bundles'/candidate['id']
        directory.mkdir(parents=True,exist_ok=True)
        spec=validate_spec(candidate['spec'])
        from .input_contract import input_contract
        manifest={'format':'quantdesk-v3','id':candidate['id'],'title':candidate['title'],'spec':spec,'code_hash':digest(candidate['code']),'spec_hash':digest(spec),'plan_hash':report.get('base',{}).get('plan_hash'),'status':'EXPLORATORY' if exploratory else report['status'],'execution':report.get('base',{}).get('execution',self.settings.execution_dict()),'funding_included':False,'venue':'OKX research -> Hyperliquid shadow','validation_summary':report.get('base',{}).get('statistics',{}),'holdout_summary':report.get('holdout',{}).get('statistics',{}),'input_contract':input_contract(candidate['code'],spec)}
        if development:
            manifest['development_summary']=development.get('statistics',{})
            manifest['model_evidence_phase']='development'
            manifest['validation_summary']={}
            manifest['holdout_summary']={}
        elif 'model' in spec:
            manifest['model_evidence_phase']=report.get('base',{}).get('phase','validation')
            if manifest['model_evidence_phase']=='development':
                manifest['development_summary']=manifest['validation_summary']
                manifest['validation_summary']={}
        manifest['execution_hash']=digest(manifest['execution'])
        manifest['shadow_execution']=self.settings.shadow_dict()
        manifest['shadow_execution_hash']=digest(manifest['shadow_execution'])
        if 'model' in spec:
            from .models import verify_artifact
            reference=report.get('base',{}).get('model_artifact')
            if not reference:
                raise ValueError('Fitted bundle requires an evaluated frozen model artifact')
            source=Path(reference['path'])
            artifact=json.loads(source.read_text(encoding='utf-8'))
            verify_artifact(artifact)
            if artifact['artifact_hash'] != reference['hash'] or artifact['code_hash'] != digest(candidate['code']) or artifact['spec_hash'] != digest(spec):
                raise ValueError('Fitted export disagrees with frozen research evidence')
            shutil.copyfile(source,directory/'model.json')
            manifest['model_hash']=artifact['artifact_hash']
            manifest['model_training_dates']=artifact['provenance']['training_dates']
        (directory/'strategy.py').write_text(candidate['code'],encoding='utf-8')
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
        return directory

    def dossier(self,candidate,report):
        directory=self.store.home/'dossiers'; directory.mkdir(parents=True,exist_ok=True)
        path=directory/(candidate['id']+'.md')
        text=f'# {candidate["title"]}\n\nStatus: **{report["status"]}**\n\n{candidate["spec"]["rationale"]}\n\nFalsification: {candidate["spec"]["falsification"]}\n\n'
        text+='## Implementation\n\n```python\n'+candidate['code']+'\n```\n\n## Evidence\n\n```json\n'+json.dumps(report,indent=2,allow_nan=False)+'\n```\n'
        path.write_text(text,encoding='utf-8')
