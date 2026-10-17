"""MiMo agents with bounded multi-step tools, retained context, and real usage."""
import json
import math
import time
import queue
import threading
from contextlib import nullcontext
import numpy as np
from scipy.stats import spearmanr
from openai import OpenAI
from .data import FEATURES
from .strategy import validate_code,validate_spec
from .strategy import VALIDATOR_VERSION,NUMPY,METHODS
from .store import digest,compact_memory
from .control import ResearchStopped,ResearchTimeLimit

class AgentTaskFailure(RuntimeError):
    def __init__(self,role,error):
        status=getattr(error,'status_code',None)
        name=type(error).__name__
        self.retryable=isinstance(error,(TimeoutError,ConnectionError)) or name in {'APITimeoutError','APIConnectionError','RateLimitError'} or (isinstance(status,int) and status>=500)
        self.role=role
        self.error_type=name
        super().__init__(f'{role} API/task failed ({name}); no synthetic replacement was produced')

class RepeatedToolFailure(RuntimeError):
    """Three identical deterministic failures cannot justify more paid turns."""
    pass

ROLES={
    'director':'You direct a small quantitative research desk. Allocate the next experiment according to evidence, novelty, information gain, execution costs, and prior failures. Select one specific falsifiable mission and measurable stop condition. Prefer investigating uncertainty over cosmetic formula mutations. Your decisions never override statistical promotion gates. Final handoff: at most 1000 words covering mechanism, prior evidence, candidate references, one decisive test, execution assumptions and stop condition. The engineer owns numerical code.',
    'scientist':'You are a market microstructure researcher. Investigate causal mechanisms using the data tools. Design specific falsifiable hypotheses, baselines, ablations, units, regimes, and execution assumptions. Run small development experiments when useful. Treat book depth reductions as a mix of trades and cancellations. Propose realistic edge after fees. Never assert a universally optimal horizon.',
    'engineer':'You are a quantitative software engineer. Build a novel numerical implementation of the research mission, inspect prior implementations, test and repair it using tools, then register the candidate. Implement compute_signal(df) returning a finite Polars Series in [-1,1], or a bounded causal feature DataFrame when model.custom_features=true. Use causal rolling or positive shifts only, no backward fill or full-sample normalization. Publish code as the actual tool argument. Preserve the experiment rationale and falsification condition.',
    'falsifier':'You are an independent falsification researcher. Inspect exact code, experimental evidence, costs, causal invariance, alternative explanations, and statistical uncertainty. Find targeted failure modes and suggest a decisive next experiment. Cite candidate IDs and observed metrics. You cannot grant deployment approval. Distinguish unsupported claims from measured failures.',
    'historian':'You maintain an evidence-based organizational memory. Translate this experiment into reusable lessons, preserving candidate IDs, observed statistics, assumptions, and counterevidence. Do not prescribe a particular horizon or factor without observed support. Distinguish operational failures from disproven economics. Record structured memory with the tool.',
}

BASE='''You are part of Quant Lab, an evidence-driven research company. Use tools to inspect and build, not merely describe. Each tool result is empirical evidence or an explicit error. API failures never create simulated discoveries. Only development dates are exposed to exploratory tools; the validator owns validation and sealed holdout. For ordinary report tasks, your final answer must be one focused JSON object describing your work, evidence, decisions, and next action. A scientist task with proposal_workflow=true must instead commit exactly one decision using submit_proposal or stop_mission. A historian must commit required lessons using the corresponding recording tool; prose alone cannot complete these artifact tasks. Do not output private reasoning. Avoid repeating profiles and requests that have already supplied the needed evidence.

NUMERICAL TOOL CONTRACT (applies to every role using experiment or register_candidate):
- Define one entry point: def compute_signal(df: pl.DataFrame) -> pl.Series. Public numerical helper functions are allowed, with simple arguments and no decorators or defaults.
- Imports may only be import numpy as np, import polars as pl, and import math.
- No file/network I/O or top-level execution. Finite for-loops over range(...) are allowed for causal online estimators; execution has a subprocess timeout. No while-loops or comprehensions.
- Return a Polars Series of identical length, finite and clipped to [-1,1]. For spec.model.custom_features=true, return a Polars DataFrame of identical height with 1..8 named finite numerical columns, each in [-1,1]. Names are public identifiers up to 40 characters and their order must be stable across every prefix, segment, date and paper observation. Fill warmup with zero.
- Use causal rolling statistics and positive shifts. Full-data means/percentiles, backward filling, and future shifts fail the prefix tests.
- Keep feature construction inside compute_signal. The simulator receives only its bounded signal and preserves the real prices.
- Fitted models are available through spec.model and walk_forward. Your compute_signal becomes one custom alpha feature or, with custom_features=true, an entire engineered causal feature matrix alongside selected stationary raw features. Build nonlinear state, temporal pressure, persistence, volatility interactions and filters in code. The trusted service owns labels, temporal fitting, train-only scaling, inner parameter selection and frozen exports. Choose ridge, quadratic_ridge or xgboost. The latter supports bounded shallow tree interactions and threshold regimes through tree_search; no random cross-validation or callbacks. Never fit using validation or holdout dates.
- Prefer build_candidate(spec, code) once, then check_candidate, candidate_feasibility and walk_forward_candidate/experiment_candidate using its immutable ID. revise_candidate creates a new artifact by applying only your explicit changes; it never edits prior code/evidence. Building alone is not submission or measured economics. Finish with register_existing_candidate after actual development assessment. Full-code tools remain available.
- Supported holding and diagnostic horizons are 1..300 seconds. State and labels reset at reconstruction segments. Before proposing a long-history fitted design, inspect segment coverage and use data_feasibility with its actual spec. Warmup plus delayed entry/holding/exit must fit inside observed segments; increasing training dates cannot repair an impossible window. Lowering warmup does not create fully mature long-window features: revise the mechanism or explicitly measure the shorter effective history.
- Inspect inner prediction diagnostics against the zero predictor and the execution cost hurdle. More complex models are not automatically better. walk_forward compares cash, a fixed trading baseline and removal of ALL custom features on identical outer dates.
- model.min_edge_bps is an ADDITIONAL margin above 2*(taker_fee_bps+slippage_bps)+observed_spread_bps, not the total cost hurdle. A value of 10.5 adds another 10.5 bps after those costs. prediction_scale_bps scales tanh(predicted_bps/scale); it neither changes the fitted target units nor substitutes for causal feature scaling. Explain any extra margin using measured uncertainty.
- Optional spec.input_features declares which raw features your generated program may read. This restriction is enforced in training, replay and paper. It makes depth dependencies auditable. A five-level stream cannot supply qi_20 or depth_total; undeclared dynamic column access conservatively requires twenty-level data. Model.features independently controls trusted raw model inputs.
- For fitted experiments select a development day_index with at least model.min_train_days earlier dates (default 10), or use walk_forward. Include a formula-only baseline and feature ablation before submission. Inspect observed costs: estimated midpoint prediction is not net alpha.
- The supplied execution parameters are authoritative. Taker roundtrip pays two taker fees plus spread and slippage; maker proxy entry pays maker fee and its exit pays taker fee. The simulator already applies latency. Do not invent another fee tier or call the proxy FIFO evidence.
- Prefer taker mode for the first transferable systems. Maker research is allowed but cannot qualify for promotion without observed trades and queue evidence.
- A failed short sample falsifies that experiment under its observed conditions. It does not kill an entire strategy class, establish an optimal horizon, or justify switching to an invented fee tier. Distinguish facts, uncertain interpretations, and proposed tests in every handoff.
Available features are: '''+', '.join(FEATURES)+'''\nAllowed pl module operations: Series, DataFrame, col, lit, when, Float64, Int64, Float32. pl.arange is unavailable; use np.arange when a causal row counter is needed. NumPy operations: '''+', '.join(sorted(NUMPY))+'''\nNumerical methods: '''+', '.join(sorted(METHODS))+'''\nA static or runtime error must be repaired using these supported operations. End an engineer task by committing actual code through register_candidate/register_existing_candidate, not by describing a future implementation.'''

REPORT_BASE=BASE.split('\n\nNUMERICAL TOOL CONTRACT',1)[0]+'''
Research constraints: supplied costs and measured data contracts are authoritative. Read prior artifacts before allocating repeated work. A development experiment, useful allocation stop, validation result and forward record are different evidence. Statistical approval belongs to the deterministic gates. Preserve source IDs, counterevidence and uncertainties; never turn a numerical check or stored lesson into proof of alpha or beneficial learning. Funding, maker fills and venue transfer require actual evidence. Declared development dates are research commitments, not untouched confirmation dates.'''

SCIENTIFIC_HANDOFF_BASE=REPORT_BASE+'''
This is an artifact-only scientific handoff. Commit exactly one submit_proposal or stop_mission decision using the advertised schema and supplied public evidence. The engineer owns deployable code. A proposal needs supported fully mature history, an exact strategy recipe, declared development indices, baseline/ablation, implementation design and a measurable stop condition. Warmup and delayed entry/holding/exit must fit inside measured reconstruction segments. Lowering declared warmup does not create mature unsupported features. Fitted outer dates require enough earlier training dates; min_train_days >= inner_days + 3 and train_days >= min_train_days. min_edge_bps is an EXTRA margin above roundtrip fees, slippage and observed spread, not their total. A compatible recipe does not establish fit sufficiency or net performance. An evidenced mission stop concerns this allocation only; no entire feature family or optimum horizon is established. If evidence is insufficient for either artifact, report that limitation; the task will remain uncompleted rather than fabricate a result.'''

def task_system_prompt(role,decision_only=False):
    if role=='scientist' and decision_only:
        base=SCIENTIFIC_HANDOFF_BASE
    elif role in {'director','historian'}:
        base=REPORT_BASE
    else:
        base=BASE
    return base+'\n'+ROLES[role]

def schema(name,description,properties=None,required=None):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':properties or {},'required':required or [],'additionalProperties':False}}}

def compact_cohort(cohort):
    result=[]
    for member in cohort:
        field='assessment' if 'assessment' in member else 'validation'
        report=member[field]
        result.append({'candidate_id':member['candidate_id'],'title':member.get('title'),
            'evidence_phase':report.get('phase','validation'),
            field:{'status':report['status'],'reasons':report['reasons'],
                **({k:compact_memory(report[k]) for k in ('selection','development','prediction_diagnostic','confirmation_eligible','confirmation_reasons','interpretation') if k in report}),
                **{key:{field:report[key][field] for field in ('statistics','trades','pnl','sampled','fill_model','execution') if field in report[key]}
                   for key in ('base','stress') if key in report}}})
    return result


def handoff_context(task,observations,partial=None):
    """Restart from public evidence without replaying a large tool conversation."""
    keys=('objective','mission','proposal','execution','costs','cost_hurdles','research_policy',
        'research_capabilities','prepared_development_indices','development_dates','validation_dates',
        'sampled','candidate_checkpoints','unfinished_program','cohort','measured_data_feasibility','data_contract','require_strategy_spec','proposal_workflow','mission_stop','required_mission_decision','artifact_workflow','task_token_limit','venue_review','replay_ids','mission_decision_id','verified_venue_evidence','critique')
    result={k:task[k] if k in {'candidate_checkpoints','unfinished_program'} else compact_memory(task[k]) for k in keys if k in task}
    result['measured_task_evidence']=[compact_memory(value) for value in observations[-4:]]
    if not observations:
        result['initial_memory']=compact_memory(task.get('memory',[])[:4])
    if partial:
        result['incomplete_public_output']={'content':(partial.get('content') or '')[:12000],
            'tool_calls':[{'name':c['name'],'arguments':c['arguments'][:12000]} for c in partial.get('tool_calls',[])[:2]],
            'executable':False}
    result['already_committed_candidates']=task.get('already_committed_candidates',[])
    result['required_deliverable']='Finish the actual existing task using these measurements. Do not repeat retrieval or experiments. Keep the public handoff concise. Partial public output is incomplete and non-executable; actual code and lessons must still be committed through their tools.'
    return result


def report_observation(observation):
    """Focused public facts for a fresh report conversation; full results persist."""
    result=observation.get('result',{})
    tool=observation.get('tool')
    if tool=='development_coverage' and isinstance(result,dict):
        summary={'records':[{k:r[k] for k in ('development_index','date','globally_exposed','full_feature_cache_available') if k in r}
            for r in result.get('records',[])],
            'interpretation':'Exact observed exposure/cache flags; first-exposure provenance remains in the durable tool result'}
    elif tool=='recall' and isinstance(result,list):
        summary=[]
        keys={'lesson','evidence','grounding','status','phase','statistics','pnl','trades','sampled','outer_dates','date',
            'study_id','capture_id','effective_source_state','depth_observations','book_window','segments','quality_counts',
            'replay_id','model_key','model_hash','quote_arm','depth_arm','active_signals','forward_eligible','strategy_promoted',
            'coverage_verified','strategy_pnl_computed','limits',
            'feature','horizon_seconds','observations','pearson_ic','spearman_ic','mean_abs_forward_bps','source','interpretation',
            'reason','cycle_id','summary','development_reasons','validation_reasons'}
        for memory in result[:4]:
            payload=memory.get('payload',{})
            selected={k:compact_memory(v) for k,v in payload.items() if k in keys} if isinstance(payload,dict) else payload
            summary.append({k:memory[k] for k in ('id','campaign','candidate','kind') if k in memory}|{'payload':selected})
        summary={'memories':summary,'retrieved_count':len(result),'included_count':len(summary),
            'interpretation':'Focused public evidence for finalization; full memory and omitted recipe/model tables remain in durable retrieval artifacts. No new economic claim.'}
    else:
        summary=compact_memory(result)
    return {'tool':tool,'result_digest':digest(result),'result':summary}

TEXT={'type':'string'}
from .models import MODEL_FEATURES
TREE_SPEC={'type':'object','properties':{'max_depth':{'type':'integer','minimum':1,'maximum':4},
    'min_child_weight':{'type':'number','minimum':5,'maximum':2000},'eta':{'type':'number','minimum':0.01,'maximum':0.2},
    'reg_lambda':{'type':'number','minimum':0.01,'maximum':10000},'rounds':{'type':'integer','minimum':10,'maximum':200}},'additionalProperties':False}
MODEL_SPEC={'type':'object','properties':{'family':{'type':'string','enum':['ridge','quadratic_ridge','xgboost']},
    'custom_features':{'type':'boolean'},'tree_search':{'type':'array','items':TREE_SPEC,'minItems':1,'maxItems':4},
    'include_alpha':{'type':'boolean'},
    'features':{'type':'array','items':{'type':'string','enum':list(MODEL_FEATURES)},'minItems':1,'maxItems':8},
    'alphas':{'type':'array','items':{'type':'number','minimum':0.0001,'maximum':100000},'minItems':1,'maxItems':8},
    'train_days':{'type':'integer','minimum':6,'maximum':120},'min_train_days':{'type':'integer','minimum':6,'maximum':60},
    'inner_days':{'type':'integer','minimum':1,'maximum':20},'stride_seconds':{'type':'integer','minimum':1,'maximum':300},
    'max_train_rows':{'type':'integer','minimum':200,'maximum':200000},'embargo_seconds':{'type':'integer','minimum':0,'maximum':3600},
    'prediction_scale_bps':{'type':'number','minimum':0.1,'maximum':1000,'description':'Scale in tanh(predicted midpoint return bps / scale); does not change model labels or feature scaling.'},
    'min_edge_bps':{'type':'number','minimum':0,'maximum':100,'description':'Extra predicted-return margin ABOVE modeled roundtrip fees, slippage and observed spread; never put total transaction costs here.'}},
    'additionalProperties':False}
SPEC={'type':'object','properties':{'title':TEXT,'rationale':TEXT,'falsification':TEXT,'mode':{'type':'string','enum':['taker','maker']},'holding_seconds':{'type':'number','minimum':1,'maximum':300},'threshold':{'type':'number','minimum':0.01,'maximum':1},'warmup_rows':{'type':'integer','minimum':0,'maximum':5000},'parent_ids':{'type':'array','items':TEXT},'input_features':{'type':'array','items':{'type':'string','enum':list(FEATURES)},'minItems':1,'uniqueItems':True},'model':MODEL_SPEC},'required':['title','rationale','falsification','mode','holding_seconds','threshold'],'additionalProperties':False}
SPEC_PATCH={'type':'object','properties':SPEC['properties'],'additionalProperties':False}

class ResearchTools:
    def __init__(self,cid,store,catalog,evaluator,settings,cycle_id=None):
        self.cid,self.store,self.catalog,self.evaluator,self.settings=cid,store,catalog,evaluator,settings
        self.plan=store.get_campaign(cid)['plan']
        self.cycle_id=cycle_id
        self.active_task_id=None
        self.tool_specs={
            'read_venue_replay':schema('read_venue_replay','Read the exact immutable retrospective venue replay, bound journals, frozen assumptions and economic diagnostics by ID. No forward acceptance or promotion.',{'replay_id':TEXT},['replay_id']),
            'list_frozen_models':schema('list_frozen_models','List up to ten exact existing models matching this candidate code/spec and execution. No fitting, forecasts or approval.',{'candidate_id':TEXT},['candidate_id']),
            'replay_venue_candidate':schema('replay_venue_candidate','Retrospective research on an exact frozen public study and immutable taker candidate. Compare quote and displayed-depth fill assumptions using identical causal signals. Reuse an exact earlier frozen model for fitted candidates; no target-venue fitting, forward acceptance or promotion.',
                {'candidate_id':TEXT,'study_id':TEXT,'model_key':TEXT},['candidate_id','study_id']),
            'list_venue_captures':schema('list_venue_captures','List local public Hyperliquid captures with process-aware liveness and source inventory. Reported counts/status are not measured coverage or strategy evidence.',{'limit':{'type':'integer','minimum':1,'maximum':10}}),
            'study_venue_capture':schema('study_venue_capture','Measure frozen public capture cadence, depth, spread, unique prints, causal previous-quote distances and hypothetical depth sweeps. Never computes strategy PnL or calibrated own fills. Rejects live sources; missing tails/unknown coverage remain explicit.',
                {'capture_id':TEXT,'notionals_usd':{'type':'array','items':{'type':'number','minimum':50,'maximum':100000},'minItems':1,'maxItems':6,'uniqueItems':True}},['capture_id']),
            'read_venue_study':schema('read_venue_study','Read an exact hashed public capture study and frozen sweep assumptions by ID. This observation cannot grant alpha, maker, forward or portfolio approval.',{'study_id':TEXT},['study_id']),
            'submit_proposal':schema('submit_proposal','Commit an immutable scientific design with exact spec, supported mature history, declared development test indices and actual metadata feasibility. No code, replay or candidate submission. This completes the scientist handoff.',
                {'strategy_spec':SPEC,'effective_history_rows':{'type':'integer','minimum':1,'maximum':5001},
                 'development_indices':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'maxItems':3,'uniqueItems':True},
                 **{k:TEXT for k in ('hypothesis','implementation_design','baseline','ablation','stop_condition','limitations')}},
                ['strategy_spec','effective_history_rows','development_indices','hypothesis','implementation_design','baseline','ablation','stop_condition','limitations']),
            'stop_mission':schema('stop_mission','Stop/redesign this research allocation using verified data incompatibility or exact prior development evidence. Never rejects a whole strategy class or grants approval. This completes the scientist handoff for independent mission review.',
                {'basis':{'type':'string','enum':['DATA_INCOMPATIBLE','MEASURED_DEVELOPMENT','MEASURED_VENUE']},'proposed_spec':SPEC,
                 'day_index':{'type':'integer','minimum':0},'candidate_ids':{'type':'array','items':TEXT,'minItems':1,'maxItems':3,'uniqueItems':True},
                 'replay_ids':{'type':'array','items':TEXT,'minItems':1,'maxItems':3,'uniqueItems':True},
                 **{k:TEXT for k in ('reason','next_experiment','limitations')}},['basis','reason','next_experiment','limitations']),
            'read_decision':schema('read_decision','Read the exact immutable scientific design/stop artifact in this campaign and its trusted development grounding. No holdout.',{'decision_id':TEXT},['decision_id']),
            'record_mission_lesson':schema('record_mission_lesson','Record one immutable lesson from this cycle\'s actual stop decision after independent falsifier review. No candidate or economic approval.',
                {'decision_id':TEXT,'lesson':TEXT,'next_experiment':TEXT},['decision_id','lesson','next_experiment']),
            'build_candidate':schema('build_candidate','Persist validated immutable authored code/recipe as a PROPOSED draft, without replay or submission. Use its ID for checks and replay.',{'spec':SPEC,'code':TEXT},['spec','code']),
            'revise_candidate':schema('revise_candidate','Create a new immutable draft from a campaign artifact and only your explicit spec/code changes. Partial model fields merge; replace_model=true explicitly starts the new recipe from defaults (useful when changing families). No fit, replay or submission; prior evidence is retained.',{'candidate_id':TEXT,'spec_changes':SPEC_PATCH,'code':TEXT,'replace_model':{'type':'boolean'}},['candidate_id']),
            'check_candidate':schema('check_candidate','Check the exact stored code/recipe using the bounded numerical worker; returns causal/schema/bounds and bounded label diagnostics. No PnL or submission.',{'candidate_id':TEXT,'day_index':{'type':'integer','minimum':0}},['candidate_id']),
            'candidate_feasibility':schema('candidate_feasibility','Measure actual segment/label compatibility of the stored fitted recipe without retyping it; development only, no forecasts or fitting.',{'candidate_id':TEXT,'day_index':{'type':'integer','minimum':0}},['candidate_id']),
            'walk_forward_candidate':schema('walk_forward_candidate','Nested temporal DEVELOPMENT fit/replay of the exact stored artifact with matched comparisons. No retyping, validation, holdout or submission.',{'candidate_id':TEXT,'folds':{'type':'integer','minimum':1,'maximum':20},'start_index':{'type':'integer','minimum':6},'end_index':{'type':'integer','minimum':6}},['candidate_id']),
            'experiment_candidate':schema('experiment_candidate','Replay the exact stored artifact on up to three DEVELOPMENT dates; earlier-only fitting when modeled. No submission or independent approval.',{'candidate_id':TEXT,'days':{'type':'integer','minimum':1,'maximum':3},'day_indices':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'maxItems':3,'uniqueItems':True}},['candidate_id']),
            'data_feasibility':schema('data_feasibility','Check fitted recipe label counts and segment history before numerical code or fitting. Uses a full prepared DEVELOPMENT frame when available, otherwise a bounded prefix. Metadata only; no forecasts, PnL, validation or holdout.',{'spec':SPEC,'day_index':{'type':'integer','minimum':0}},['spec']),
            'screen_candidate':schema('screen_candidate','Precommit and replay up to three DEVELOPMENT dates with cash, baseline and fitted feature-removal comparisons. Triage only; cannot validate, seal or promote. Defaults prefer prepared dates and freeze their selection across restart.',{'candidate_id':TEXT,'day_indices':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'maxItems':3,'uniqueItems':True}},['candidate_id']),
            'development_coverage':schema('development_coverage','Check frozen development indices against global recorded data exposure BEFORE calling dates fresh. Lists dates and first recorded exposure only; no holdout data or metrics.',{'start_index':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':50}}),
            'diagnose_prediction':schema('diagnose_prediction','Separate weak fitted predictions, modeled costs and an EXTRA margin; inspect full DEVELOPMENT prediction quantiles, zero-predictor comparison and rows above each gate. Fits only earlier development dates; no validation/holdout access.',{'candidate_id':TEXT,'day_index':{'type':'integer','minimum':0}},['candidate_id','day_index']),
            'check_program':schema('check_program','Run actual numerical code on a bounded earlier DEVELOPMENT frame with causal prefix, schema, bounds and declared-input checks. No fitting or PnL; use before an expensive replay.',{'spec':SPEC,'code':TEXT,'day_index':{'type':'integer','minimum':0}},['spec','code']),
            'research_map':schema('research_map','Inspect measured coverage, candidate/program/recipe variety, actual role token usage, failures and grounded next-experiment suggestions. Use to allocate work; these proxies do not establish alpha or improved learning.'),
            'walk_forward':schema('walk_forward','Fit and test a predictor on 1..20 outer DEVELOPMENT dates; each uses earlier development training with inner temporal parameter selection. No validation or holdout access. Returns chronology, costs and daily net replay evidence.',{'spec':SPEC,'code':TEXT,'folds':{'type':'integer','minimum':1,'maximum':20},'start_index':{'type':'integer','minimum':6},'end_index':{'type':'integer','minimum':6}},['spec','code']),
            'inspect_data':schema('inspect_data','Profile an actual development date and available causal features.',{'day_index':{'type':'integer','minimum':0}}),
            'measure_feature':schema('measure_feature','Measure a causal feature against actual time-based future returns on DEVELOPMENT only. Default reads the first 10000 rows, with nonoverlapping observations and segment isolation. This bounded descriptive diagnostic does not characterize a whole day or establish executable alpha.',{'day_index':{'type':'integer','minimum':0},'feature':{'type':'string','enum':list(FEATURES)},'horizon_seconds':{'type':'integer','minimum':1,'maximum':300}},['feature','horizon_seconds']),
            'recall':schema('recall','Retrieve focused persistent evidence, failures and lessons. Default 4 records; read exact artifacts separately.',{'limit':{'type':'integer','minimum':1,'maximum':15}}),
            'list_candidates':schema('list_candidates','List compact registered candidate summaries. Default 8; use read_candidate for exact code and recipe.',{'limit':{'type':'integer','minimum':1,'maximum':20}}),
            'read_candidate':schema('read_candidate','Read exact numerical code, specification, and report.',{'candidate_id':TEXT},['candidate_id']),
            'diagnose_execution':schema('diagnose_execution','Inspect recorded trade-level gross midpoint capture, spread/slippage drag, fees and net PnL. Development only for scientist/engineer; falsifier may inspect validation. Never exposes holdout.',{'candidate_id':TEXT,'phase':{'type':'string','enum':['development','validation']},'day_index':{'type':'integer','minimum':0},'stress':{'type':'boolean'},'limit':{'type':'integer','minimum':1,'maximum':1000}},['candidate_id','day_index']),
            'experiment':schema('experiment','Build and run a bounded causal numerical strategy on up to three DEVELOPMENT dates. Select indices to examine distinct regimes instead of reusing the first morning. Every replay is counted. No validation or holdout is available.',{'spec':SPEC,'code':TEXT,'days':{'type':'integer','minimum':1,'maximum':3},'day_indices':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'maxItems':3,'uniqueItems':True}},['spec','code']),
            'register_candidate':schema('register_candidate','Persist immutable implementable code and its falsifiable specification for independent validation.',{'spec':SPEC,'code':TEXT},['spec','code']),
            'register_existing_candidate':schema('register_existing_candidate','Select an exact, already implemented campaign artifact for independent validation, without retyping its code. Use after inspecting its development evidence.',{'candidate_id':TEXT},['candidate_id']),
            'record_lesson':schema('record_lesson','Persist a grounded reusable lesson or uncertain claim for future agents.',{'candidate_id':TEXT,'lesson':TEXT,'evidence':TEXT,'next_experiment':TEXT},['candidate_id','lesson','evidence','next_experiment']),
        }

    def available(self,role):
        names=['inspect_data','recall','research_map','development_coverage','list_candidates','read_candidate','read_decision','data_feasibility','candidate_feasibility','list_venue_captures','study_venue_capture','read_venue_study','read_venue_replay']
        if role in {'scientist','engineer'}:
            names+=['experiment','walk_forward','measure_feature','diagnose_execution','diagnose_prediction','check_program','screen_candidate','list_frozen_models','replay_venue_candidate']
        if role=='falsifier':
            names+=['diagnose_execution','diagnose_prediction','screen_candidate','list_frozen_models','replay_venue_candidate']
        if role=='engineer':
            names+=['register_candidate','register_existing_candidate','build_candidate','revise_candidate','check_candidate','walk_forward_candidate','experiment_candidate']
        if role=='historian':
            names+=['record_lesson','record_mission_lesson']
        if role=='scientist':
            names+=['submit_proposal','stop_mission']
        return [self.tool_specs[n] for n in names]

    def call(self,role,name,args):
        self.store.check_research_control(self.cid)
        if name not in {s['function']['name'] for s in self.available(role)}:
            raise ValueError('This role does not own that tool')
        cacheable=name in {'inspect_data','measure_feature','experiment','walk_forward','check_program','diagnose_prediction','check_candidate','walk_forward_candidate','experiment_candidate'}
        key=digest({'campaign':self.cid,'plan':self.plan.get('hash'),'tool':name,'args':args,'validator':VALIDATOR_VERSION,'tool_contract':2,'trade_log_format':1,'execution':self.settings.execution_dict()})
        if cacheable:
            cached=self.store.cached_tool(key)
            if cached is not None:
                self.store.event(self.cid,role,'TOOL_CACHE_REUSED',{'tool':name,'key':key})
                return dict(cached,already_measured=True,instruction='Reuse this evidence; repeating the same request adds no new observation')
        result=self._call(role,name,args)
        if cacheable and isinstance(result,dict) and 'error' not in result:
            self.store.cache_tool(key,result)
        return result

    def _call(self,role,name,args):
        self.store.check_research_control(self.cid)
        if name=='read_venue_replay':
            from .venue_replay import load
            return load(self.store,args['replay_id'])
        if name in {'list_frozen_models','replay_venue_candidate'}:
            from . import venue_replay
            if self.store.get_candidate(args['candidate_id'])['campaign']!=self.cid: raise ValueError('Candidate belongs to another campaign')
            if name=='list_frozen_models': return venue_replay.list_models(self.store,args['candidate_id'],self.settings)
            return venue_replay.replay(self.store,self.settings,self.cid,args['candidate_id'],args['study_id'],args.get('model_key'))
        if name in {'list_venue_captures','study_venue_capture','read_venue_study'}:
            from . import venue_study
            if name=='list_venue_captures': return venue_study.inventory(self.store.home,args.get('limit',8))
            if name=='read_venue_study': return {'study_id':args['study_id'],'report':venue_study.load(self.store,args['study_id'])}
            if self.store.get_campaign(self.cid)['status']=='SEALED': raise ValueError('Sealed campaigns cannot create new venue studies')
            result=venue_study.study(self.store,self.settings,args['capture_id'],args.get('notionals_usd'),lambda:self.store.check_research_control(self.cid))
            self.store.event(self.cid,role,'VENUE_STUDY_REUSED' if result['already_measured'] else 'VENUE_STUDY_MEASURED',
                {'study_id':result['study_id'],'capture_id':args['capture_id']})
            if not result['already_measured']:
                report=result['report']
                self.store.memory(self.cid,'','VENUE_EXECUTION_STUDY',{'study_id':result['study_id'],'capture_id':args['capture_id'],
                    **{k:report[k] for k in ('mainnet','effective_source_state','quality_counts','depth_observations','book_window','segments','observations','hypothetical_sweeps','coverage_verified','strategy_pnl_computed','limits')}})
            return result
        if name in {'submit_proposal','stop_mission','read_decision','record_mission_lesson'}:
            from . import proposals
            if name=='read_decision':
                return proposals.load(self.store,args['decision_id'],self.cid)
            return {'submit_proposal':proposals.submit,'stop_mission':proposals.stop,
                'record_mission_lesson':proposals.record_lesson}[name](self,args)
        if name in {'build_candidate','revise_candidate','check_candidate','candidate_feasibility','walk_forward_candidate','experiment_candidate'}:
            if self.store.get_campaign(self.cid)['status']=='SEALED':
                raise ValueError('Campaign is sealed; no further research is allowed')
            if name=='build_candidate':
                spec=validate_spec(args['spec']); code=args['code']; validate_code(code)
            else:
                parent=self.store.get_candidate(args['candidate_id'])
                if parent['campaign']!=self.cid:
                    raise ValueError('Artifact tools require a candidate in this campaign; build an explicit local draft first')
                if name=='revise_candidate':
                    changes=args.get('spec_changes',{})
                    if not isinstance(changes,dict):
                        raise ValueError('spec_changes must be an explicit object')
                    spec=dict(parent['spec'],**changes)
                    if 'model' in changes:
                        if not isinstance(changes['model'],dict):
                            raise ValueError('Model revision must be an explicit recipe object')
                        spec['model']=dict({} if args.get('replace_model',False) else parent['spec'].get('model',{}),**changes['model'])
                    elif args.get('replace_model'):
                        raise ValueError('replace_model requires an explicit model recipe in spec_changes')
                    spec['parent_ids']=list(dict.fromkeys(spec.get('parent_ids',[])+[parent['id']]))
                    spec=validate_spec(spec); code=args.get('code',parent['code']); validate_code(code)
                else:
                    parameters={k:v for k,v in args.items() if k!='candidate_id'}
                    parameters.update(spec=parent['spec'],code=parent['code'])
                    tool={'check_candidate':'check_program','candidate_feasibility':'data_feasibility','walk_forward_candidate':'walk_forward','experiment_candidate':'experiment'}[name]
                    result=self._call(role,tool,parameters)
                    return dict(result,candidate_id=parent['id'],artifact_identity={'code_hash':digest(parent['code']),'spec_hash':digest(parent['spec'])})
            ident,created=self.store.candidate(self.cid,spec,code)
            stored=self.store.get_candidate(ident)
            self.store.event(self.cid,role,'DRAFT_BUILT',{'candidate_id':ident,'created':created,'parent_id':parent['id'] if name=='revise_candidate' else None,'cycle':self.cycle_id})
            return {'candidate_id':ident,'created':created,'status':stored['status'],
                'submitted':False,'spec':stored['spec'],'code_hash':digest(stored['code']),'spec_hash':digest(stored['spec']),
                'interpretation':'Immutable implementation draft only; check feasibility/code and actual development economics before submission'}
        if name=='data_feasibility':
            from .models import validate_recipe
            if not isinstance(args.get('spec'),dict) or 'model' not in args['spec']:
                raise ValueError('Label feasibility requires a fitted-model recipe')
            spec=validate_spec({k:v for k,v in args['spec'].items() if k!='model'})
            recipe_error=None
            try:
                recipe=validate_recipe(args['spec']['model'])
            except ValueError as error:
                recipe_error=str(error)
                # Structural timestamp/history checks remain useful even when
                # the requested fitting recipe is invalid. Never silently fit
                # this diagnostic recipe or modify the actual authored spec.
                recipe=validate_recipe({'stride_seconds':1})
            index=int(args.get('day_index',0))
            if not 0<=index<len(self.plan['development']):
                raise ValueError('Data feasibility requires a development date')
            record=self.plan['development'][index]
            requested=self.plan.get('max_rows_per_day')
            prepared=bool(getattr(self.catalog,'cached',lambda *a:False)(record,requested))
            rows=requested if prepared else min(requested or 10000,10000)
            self.store.expose_dates(self.cid,[record],'development')
            frame,quality=self.catalog.load(record,rows)
            from .models import label_feasibility
            result=dict(label_feasibility(frame,spec,self.settings.execution_dict(),recipe),
                date=record['date'],rows=len(frame),bounded_rows_requested=rows,
                full_prepared_frame_used=prepared and requested is None,quality=quality)
            if recipe_error:
                result.update(status='INVALID_RECIPE',label_window_status=result['status'],recipe_error=recipe_error,
                    diagnostic_basis='Structural label windows at 1-second thinning only. The requested recipe is invalid; no fitting is allowed and its parameters are unchanged.')
            self.store.event(self.cid,role,'DATA_FEASIBILITY',result)
            return result
        if name=='screen_candidate':
            return compact_memory(self.evaluator.screen(self.cid,self.store.get_candidate(args['candidate_id']),args.get('day_indices')))
        if name=='development_coverage':
            start=int(args.get('start_index',0)); limit=int(args.get('limit',20))
            if not 0<=start<len(self.plan['development']) or not 1<=limit<=50:
                raise ValueError('Use a development start index and limit 1..50')
            with self.store.connect() as db:
                exposed={r['date']:dict(r) for r in db.execute('SELECT date,phase,campaign,created FROM data_exposures')}
            return {'records':[{'development_index':index,'date':record['date'],
                'globally_exposed':record['date'] in exposed,
                'full_feature_cache_available':bool(getattr(self.catalog,'cached',lambda *a:False)(record,self.plan.get('max_rows_per_day'))),
                'first_recorded_exposure':exposed.get(record['date'])}
                for index,record in enumerate(self.plan['development'][start:start+limit],start)],
                'interpretation':'Recorded lab exposure at query time, including preparation and training; absence is not a claim about external human/provider knowledge. This tool does not access holdout data.'}
        if name=='diagnose_prediction':
            return self.evaluator.diagnose_prediction(self.cid,self.store.get_candidate(args['candidate_id']),int(args['day_index']))
        if name=='check_program':
            spec=validate_spec(args['spec']); validate_code(args['code'])
            index=int(args.get('day_index',0))
            if not 0<=index<len(self.plan['development']):
                raise ValueError('Program checks require an earlier development date')
            record=self.plan['development'][index]
            self.store.expose_dates(self.cid,[record],'development')
            frame,_=self.catalog.load(record,3000)
            from .input_contract import program_input
            from .strategy import run_signal,run_feature_program
            custom=spec.get('model',{}).get('custom_features',False)
            runner=run_feature_program if custom else run_signal
            output=runner(args['code'],program_input(frame,spec),self.settings.worker_timeout)
            result={'status':'BOUNDED_PROGRAM_CHECK_PASSED','date':record['date'],'rows':len(frame),
                'code_hash':digest(args['code']),'spec_hash':digest(spec),
                'output_columns':output.columns if custom else ['signal'],
                'interpretation':'Actual worker prefix/schema/bounds check on a bounded development frame; no economic replay, model fit or promotion'}
            if 'model' in spec:
                from .models import label_feasibility
                result['bounded_label_feasibility']=label_feasibility(frame,spec,self.settings.execution_dict(),spec['model'])
                result['fitting_compatibility_proven']=False
            self.store.event(self.cid,role,'PROGRAM_CHECK',result)
            return result
        if name=='research_map':
            from .research_map import research_map
            return research_map(self.store,self.cid)
        if name=='inspect_data':
            index=int(args.get('day_index',0))
            if not 0<=index<len(self.plan['development']):
                raise ValueError('Invalid development day index')
            self.store.expose_dates(self.cid,[self.plan['development'][index]],'development')
            return self.catalog.profile(self.plan['development'][index],self.plan.get('max_rows_per_day') or 3000)
        if name=='measure_feature':
            index=int(args.get('day_index',0)); feature=args['feature']; horizon=int(args['horizon_seconds'])
            if feature not in FEATURES or not 1<=horizon<=300 or not 0<=index<len(self.plan['development']):
                raise ValueError('Invalid causal feature, horizon, or development day')
            record=self.plan['development'][index]
            self.store.expose_dates(self.cid,[record],'development')
            frame,_=self.catalog.load(record,self.plan.get('max_rows_per_day') or 10000)
            ts=frame['ts_ms'].to_numpy(); mids=frame['mid_price'].to_numpy(); segment=frame['segment'].to_numpy()
            future=np.searchsorted(ts,ts+horizon*1000,side='left')
            valid=future<len(ts); valid[valid]&=segment[valid]==segment[future[valid]]
            indices=np.flatnonzero(valid)
            thinned=[]; last=-np.inf
            for i in indices:
                if ts[i]-last>=horizon*1000:
                    thinned.append(i); last=ts[i]
            indices=np.array(thinned,dtype=int)
            if len(indices)<20:
                return {'observations':len(indices),'error':'Insufficient nonoverlapping development observations; choose a shorter horizon or more rows'}
            x=frame[feature].to_numpy()[indices]
            y=(mids[future[indices]]/mids[indices]-1)*1e4
            if np.std(x)<1e-12 or np.std(y)<1e-12:
                return {'observations':len(x),'error':'Feature or forward price has zero variance'}
            low,high=np.quantile(x,[0.2,0.8])
            result={'date':record['date'],'rows':len(frame),'duration_seconds':float(ts[-1]-ts[0])/1000,
                'first_exchange_ts':int(ts[0]),'last_exchange_ts':int(ts[-1]),
                'bounded_rows_requested':self.plan.get('max_rows_per_day') or 10000,
                'feature':feature,'horizon_seconds':horizon,'observations':len(x),'pearson_ic':float(np.corrcoef(x,y)[0,1]),'spearman_ic':float(spearmanr(x,y).statistic),'bottom_quintile_mean_forward_bps':float(y[x<=low].mean()),'top_quintile_mean_forward_bps':float(y[x>=high].mean()),'mean_abs_forward_bps':float(np.abs(y).mean()),'taker_roundtrip_fee_bps':2*self.settings.taker_fee_bps,'status':'DESCRIPTIVE_DEVELOPMENT_ONLY','caveat':'Bounded earlier-day diagnostic; IC and quintile means do not establish whole-day behavior or executable profits; no p-value claimed'}
            self.store.event(self.cid,role,'EXPERIMENT_STARTED',{'kind':'feature_diagnostic','feature':feature,'plan_hash':self.plan['hash']})
            self.store.memory(self.cid,'','FEATURE_DIAGNOSTIC',result)
            return result
        if name=='recall':
            return self.store.recall(max(1,min(15,int(args.get('limit',4)))))
        if name=='list_candidates':
            return [{'id':c['id'],'campaign':c['campaign'],'title':c['title'],'status':c['status'],
                'mechanism':c['spec']['rationale'],'family':c['spec'].get('model',{}).get('family','formula'),
                'mode':c['spec']['mode'],'holding_seconds':c['spec']['holding_seconds'],
                'assessment_phase':(c['report'] or {}).get('phase','validation' if c['report'] else None),
                'assessment_status':(c['report'] or {}).get('status'),
                'statistics':(c['report'] or {}).get('base',(c['report'] or {}).get('development',{})).get('statistics',{}),'reasons':(c['report'] or {}).get('reasons',[])}
                for c in self.store.candidates()[-max(1,min(20,int(args.get('limit',8)))):]]
        if name=='read_candidate':
            candidate=self.store.get_candidate(args['candidate_id'])
            # Sealed test metrics never re-enter an active research conversation.
            if candidate['report']:
                candidate['report']=compact_memory(candidate['report'])
                candidate['report'].pop('holdout',None)
                for phase in ('base','stress'):
                    if phase in candidate['report']:
                        full=candidate['report'][phase]
                        candidate['report'][phase]={k:v for k,v in full.items() if k!='days'}
                        candidate['report'][phase]['evaluated_dates']=full.get('evaluated_dates',[d['date'] for d in full.get('days',[])])
            return candidate
        if name=='register_existing_candidate':
            candidate=self.store.get_candidate(args['candidate_id'])
            if candidate['campaign']!=self.cid or self.store.get_campaign(self.cid)['status']=='SEALED':
                raise ValueError('Submission must belong to this unsealed campaign')
            validate_spec(candidate['spec']); validate_code(candidate['code'])
            self.store.event(self.cid,role,'SUBMITTED',{'id':candidate['id'],'cycle':self.cycle_id,'existing_artifact':True})
            return {'candidate_id':candidate['id'],'created':False,'status':candidate['status']}
        if name=='diagnose_execution':
            candidate=self.store.get_candidate(args['candidate_id'])
            phase=args.get('phase','development'); index=int(args['day_index']); limit=int(args.get('limit',20))
            if candidate['campaign']!=self.cid or phase not in {'development','validation'} or (phase=='validation' and role!='falsifier'):
                raise ValueError('This role cannot inspect that campaign/phase trade log')
            if not 0<=index<len(self.plan[phase]) or not 1<=limit<=1000:
                raise ValueError('Invalid trade-log day index or row limit')
            date=self.plan[phase][index]['date']
            path=self.store.home/'replays'/candidate['id']/phase/('stress' if args.get('stress',False) else 'base')/(date+'.trades.jsonl')
            if not path.exists():
                raise ValueError('No trade log yet; run the corresponding experiment or validator. Older reports need a fresh replay for attribution.')
            trades=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            return {'candidate_id':candidate['id'],'date':date,'phase':phase,'trades':len(trades),
                'gross_mid_pnl':sum(t['gross_mid_pnl'] for t in trades),'execution_drag':sum(t['execution_drag'] for t in trades),
                'fees':sum(t['fees'] for t in trades),'net_pnl':sum(t['pnl'] for t in trades),
                'mean_per_trade_gross_bps':float(np.mean([t['gross_mid_bps'] for t in trades])) if trades else None,
                'trade_rows':trades[:limit],'log':str(path),'caveat':'Midpoint capture at actual entry/exit fills; includes elapsed execution latency, excludes funding. Small samples do not identify an edge.'}
        if name in {'experiment','register_candidate','walk_forward'}:
            if self.store.get_campaign(self.cid)['status']=='SEALED':
                raise ValueError('Campaign is sealed; no further research is allowed')
            spec=validate_spec(args['spec']); validate_code(args['code'])
            ident,created=self.store.candidate(self.cid,spec,args['code'])
            candidate=self.store.get_candidate(ident)
            if name=='walk_forward':
                return self.evaluator.walk_forward(self.cid,candidate,int(args.get('folds',5)),args.get('start_index'),args.get('end_index'))
            if name=='register_candidate':
                self.store.event(self.cid,role,'SUBMITTED',{'id':ident,'cycle':self.cycle_id})
                return {'candidate_id':ident,'created':created,'status':candidate['status']}
            days=int(args.get('days',1))
            if not 1<=days<=3:
                raise ValueError('Development experiments may use 1..3 dates')
            report=self.evaluator.evaluate(self.cid,candidate,'development',days,development_indices=args.get('day_indices'))
            summary={k:v for k,v in report.items() if k not in {'days','replay_directory'}}
            summary['daily_results']=[{k:v for k,v in d.items() if k!='quality'} for d in report['days']]
            self.store.memory(self.cid,ident,'DEVELOPMENT_EXPERIMENT',summary)
            return summary
        if name=='record_lesson':
            candidate=self.store.get_candidate(args['candidate_id'])
            report=candidate['report'] or {}
            with self.store.connect() as db:
                observed=db.execute("SELECT kind,payload FROM memories WHERE candidate=? AND kind IN ('DEVELOPMENT_EXPERIMENT','WALK_FORWARD_EXPERIMENT','DEVELOPMENT_SCREEN') ORDER BY id DESC LIMIT 1",(candidate['id'],)).fetchone()
            if not report and not observed:
                raise ValueError('A lesson requires an actual experiment or validator report for this candidate')
            development=report.get('phase')=='development'
            grounding={'candidate_status':candidate['status'],'assessment_phase':'development' if development else 'validation' if report else None,
                'validation_statistics':report.get('base',{}).get('statistics',{}),'validation_reasons':[] if development else report.get('reasons',[]),
                'development_reasons':report.get('reasons',[]) if development else [],
                'development_experiment':report if development else json.loads(observed['payload']) if observed else None,
                'development_evidence_kind':'DEVELOPMENT_SCREEN' if development else observed['kind'] if observed else None,
                'interpretation':'Measured grounding is authoritative; narrative claims remain hypotheses unless supported by this evidence'}
            self.store.memory(self.cid,candidate['id'],'LESSON',dict(args,grounding=grounding))
            return {'recorded':True}
        raise ValueError('Unknown tool')

class AgentRuntime:
    def __init__(self,store,settings,client=None,model=None,provider=None,streaming=None):
        import config
        self.store,self.settings=store,settings
        self.model=model or config.DEFAULT_MODEL
        self.provider=provider or config.ACTIVE_PROVIDER
        self.client=client
        self.streaming=(settings.provider_stream and client is None and self.provider=='mimo') if streaming is None else bool(streaming)
        if client is None:
            if not config.ACTIVE_KEY:
                raise RuntimeError('No LLM provider credential configured. Use demo for an explicitly synthetic pipeline test.')
            self.client=OpenAI(api_key=config.ACTIVE_KEY,base_url=config.ACTIVE_BASE_URL,timeout=settings.request_timeout,max_retries=0)

    def provider_call(self,kwargs,context=None):
        if self.streaming:
            from .provider_stream import streaming_call
            context=context or {}
            cid=context.get('campaign'); role=context.get('role')
            binding={k:context.get(k) for k in ('task_id','request_id')}
            return streaming_call(self.client,kwargs,self.settings.request_timeout,
                progress=(lambda value:self.store.event(cid,role,'PROVIDER_PROGRESS',dict(binding,**value))) if cid else None,
                partial=(lambda value:self.store.event(cid,role,'INCOMPLETE_PUBLIC_OUTPUT',dict(binding,**value))) if cid else None,
                control=(lambda:self.store.check_research_control(cid)) if cid else None)
        # SDK timeouts bound socket inactivity, not total elapsed request time.
        # A provider can trickle a response indefinitely; task control still
        # needs a wall-clock deadline. Never execute tools from a late response.
        result_queue=queue.Queue(maxsize=1)
        def invoke():
            try:
                result_queue.put((True,self.client.chat.completions.create(**kwargs)))
            except BaseException as error:
                result_queue.put((False,error))
        threading.Thread(target=invoke,daemon=True).start()
        try:
            succeeded,result=result_queue.get(timeout=self.settings.request_timeout)
        except queue.Empty:
            raise TimeoutError('Provider exceeded task wall-clock deadline; late output will not be used') from None
        if not succeeded:
            raise result
        return result

    def run(self,cid,role,task,tools,task_id=None):
        started=time.time()
        try:
            return self._run(cid,role,task,tools,task_id)
        except BaseException as error:
            if task_id:
                self.store.fail_task(task_id,error,'STOPPED' if isinstance(error,ResearchStopped) else 'INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED',after=started)
            raise

    def _run(self,cid,role,task,tools,task_id=None):
        self.store.check_research_control(cid)
        if task.get('venue_review'):
            if not task_id or not isinstance(tools,ResearchTools):
                raise ValueError('Venue review requires durable research tools and task ownership')
            from .mission_review import prepare
            task=prepare(tools,role,task,task_id)
        previous=None; recent=[]
        if task.get('task_token_limit') is not None and not task_id:
            raise ValueError('A bounded task requires a durable task_id')
        if task.get('proposal_workflow') and not task_id:
            raise ValueError('A scientific decision requires a durable task_id')
        if task_id:
            with self.store.connect() as db:
                previous=db.execute('SELECT status,updated,started FROM tasks WHERE id=?',(task_id,)).fetchone()
                recent=db.execute("SELECT payload FROM events WHERE campaign=? AND agent=? AND kind='TOOL_RESULT' AND created>=? ORDER BY id DESC LIMIT 8",(cid,role,previous['started'] or previous['updated'])).fetchall() if previous and previous['status']!='COMPLETE' else []
                recent=[r for r in recent if json.loads(r[0]).get('task_id',task_id)==task_id]
                drafts=db.execute("SELECT id,payload FROM events WHERE campaign=? AND agent=? AND kind='PROGRAM_ATTEMPT' ORDER BY id DESC LIMIT 10",(cid,role)).fetchall() if previous and previous['status']!='COMPLETE' else []
                draft=next((dict(json.loads(r['payload']),event_id=r['id']) for r in drafts if json.loads(r['payload']).get('task_id')==task_id),None)
                partials=db.execute("SELECT payload FROM events WHERE campaign=? AND agent=? AND kind='INCOMPLETE_PUBLIC_OUTPUT' ORDER BY id DESC LIMIT 10",(cid,role)).fetchall() if previous and previous['status']!='COMPLETE' else []
                partial=next((json.loads(r[0]) for r in partials if json.loads(r[0]).get('task_id')==task_id),None)
                required={m['candidate_id'] for m in task.get('cohort',[])} if role=='historian' else set()
                recorded={r[0] for r in db.execute("SELECT candidate FROM memories WHERE campaign=? AND kind='LESSON' AND created>=?",(cid,previous['started'] or previous['updated']))} if required and previous and previous['status']!='COMPLETE' else set()
            if required and required<=recorded:
                receipt={'artifact_receipt':True,'lessons_recorded_for':sorted(required),'recovered_from':'durable lesson artifacts'}
                self.store.finish_task(task_id,receipt)
                self.store.event(cid,role,'CHECKPOINT_REUSED',{'task_id':task_id,'artifact_receipt':True})
                return receipt
            if partial and not recent and (partial.get('content') or partial.get('tool_calls')):
                task=dict(handoff_context(task,[],partial),resume_public_draft=True)
                if role=='director':
                    task['resume_report_only']=True
                if role=='scientist' and task.get('proposal_workflow'):
                    task['resume_scientific_commit']=True
                    task['required_deliverable']='A public draft survived the interrupted request, but no scientific artifact was committed. Repair it against the supplied actual data contract, then call submit_proposal or stop_mission. The partial draft is not evidence or executable code; use trusted checks and preserve uncertainty.'
            if recent:
                task=dict(task,resume_evidence=[compact_memory(json.loads(r[0])) for r in recent],resume_instruction='Resume from these measured results. Reuse completed work and available registered implementations. Avoid repeating experiments or profile requests.')
                if role in {'director','scientist'}:
                    task=dict(handoff_context(task,[report_observation(json.loads(r[0])) for r in reversed(recent)],partial),resume_report_only=not task.get('proposal_workflow',False))
                    if role=='scientist' and task.get('proposal_workflow'):
                        task['required_deliverable']='Commit one explicit submit_proposal or stop_mission using the supplied public evidence. Do not repeat retrieval or invent economic success.'
                    elif role=='scientist' and task.get('require_strategy_spec'):
                        task['required_deliverable']='Return exactly one JSON handoff with a structurally valid strategy_spec, explicit supported effective history, decisive baseline/ablation, test dates and stop conditions. Use the supplied measured constraints and retrieved evidence. Do not repeat retrieval. Feasibility is not prediction skill, net economics or class-level falsification.'
                if role=='engineer':
                    measured_tools={'experiment','walk_forward','experiment_candidate','walk_forward_candidate'}
                    checkpoint_ids={json.loads(r[0]).get('result',{}).get('candidate_id') for r in recent if json.loads(r[0]).get('tool') in measured_tools and isinstance(json.loads(r[0]).get('result'),dict) and 'error' not in json.loads(r[0])['result']}
                    candidates=[c for c in self.store.candidates(cid) if c['id'] in checkpoint_ids][-2:]
                    if candidates:
                        measurements=[]
                        for row in recent:
                            observed=json.loads(row[0])
                            if observed.get('tool') in measured_tools:
                                measurements.append({k:v for k,v in observed['result'].items() if k in {'candidate_id','statistics','pnl','trades','sampled','fill_model','daily_results','error'}})
                        task={'execution':task.get('execution',self.settings.execution_dict()),'cost_hurdles':task.get('cost_hurdles'),
                            'candidate_checkpoints':[{k:c[k] for k in ('id','spec','code','status')} for c in candidates],
                            'measured_development_results':measurements,'resume_finalization':True,
                            'required_deliverable':'Development work already exists. Select a representative exact artifact using register_existing_candidate(candidate_id), then finish the JSON handoff. Negative results are useful. This task is finalizing prior work, so do not start a new hypothesis or repeat profiling.'}
                    else:
                        task=dict(task,resume_implementation=True,
                            required_deliverable='Complete and register the actual implementation using the measured resume evidence. Repair the latest error. Do not repeat retrieval or diagnostics. No successful submission exists yet.')
                        if draft:
                            task['unfinished_program']=draft
                            draft_spec=draft.get('args',{}).get('spec')
                            artifact_id=draft.get('args',{}).get('candidate_id')
                            if artifact_id:
                                try:
                                    existing=self.store.get_candidate(artifact_id)
                                except ValueError:
                                    existing=None
                                if existing and existing['campaign']==cid:
                                    task['candidate_checkpoints']=[{k:existing[k] for k in ('id','spec','code','status')}]
                                    draft_spec=existing['spec']
                            if tools.catalog is not None and isinstance(draft_spec,dict) and 'model' in draft_spec:
                                try:
                                    prepared=[i for i,r in enumerate(tools.plan['development']) if getattr(tools.catalog,'cached',lambda *a:False)(r,tools.plan.get('max_rows_per_day'))]
                                    check=tools._call('runtime','data_feasibility',{'spec':draft_spec,'day_index':prepared[0] if prepared else 0})
                                    task['measured_data_feasibility']=check
                                    task['required_deliverable']+=' Use the supplied actual data feasibility. If history is incompatible, revise the mechanism and its supported effective history; more training dates cannot solve it.'
                                except (ValueError,OSError) as error:
                                    task['measured_data_feasibility']={'error':str(error),'status':'CHECK_FAILED'}
                        if partial and (partial.get('content') or partial.get('tool_calls')):
                            task['incomplete_public_output']=partial
                if role=='falsifier':
                    if task.get('mission_stop'):
                        task=dict(handoff_context(task,[report_observation(json.loads(r[0])) for r in reversed(recent)]),resume_report_only=True)
                        task['required_deliverable']='Return one independent JSON critique of this mission stop using actual grounding. Distinguish a blocked configuration from a disproven mechanism; give a decisive redesign if appropriate. No promotion or new retrieval.'
                    ids={m['candidate_id'] for m in task.get('cohort',[])}
                    inspected=[]
                    for row in recent:
                        observed=json.loads(row[0])
                        candidate=observed.get('result',{})
                        if observed.get('tool')=='read_candidate' and isinstance(candidate,dict):
                            inspected.append({k:candidate[k] for k in ('id','spec','status') if k in candidate})
                    if not task.get('mission_stop'):
                        task={'cohort':compact_cohort(task.get('cohort',[])),
                        'selected_code':[{k:c[k] for k in ('id','spec','code')} for c in self.store.candidates(cid) if c['id'] in ids],
                        'inspected_alternatives':inspected,'resume_report_only':True,
                        'required_deliverable':'Code and evidence have already been inspected. Return exactly one final JSON critique and decisive next experiment, with no more tools. Use measured evidence, separate operational and economic failures, and never generalize partial dates to a whole strategy class.'}
            cached=self.store.start_task(task_id,cid,role,task)
            if cached is not None:
                self.store.event(cid,role,'CHECKPOINT_REUSED',{'task_id':task_id})
                return cached
            if role=='scientist' and task.get('proposal_workflow'):
                with self.store.connect() as db:
                    decision=db.execute('SELECT id,kind FROM research_decisions WHERE task=? AND campaign=?',(task_id,cid)).fetchone()
                if decision:
                    from .proposals import load
                    receipt={'decision_id':decision['id'],'decision_kind':decision['kind'],'decision':load(self.store,decision['id'],cid),
                        'artifact_receipt':True,'recovered_from':'durable scientific decision'}
                    self.store.finish_task(task_id,receipt)
                    self.store.event(cid,role,'CHECKPOINT_REUSED',{'task_id':task_id,'decision_id':decision['id']})
                    return receipt
            if role=='historian' and task.get('required_mission_decision'):
                with self.store.connect() as db:
                    recorded=db.execute('SELECT payload FROM mission_lessons WHERE decision_id=?',(task['required_mission_decision'],)).fetchone()
                if recorded:
                    from .proposals import load
                    decision=load(self.store,task['required_mission_decision'],cid)
                    if decision['cycle']!=tools.cycle_id:
                        raise ValueError('Mission lesson checkpoint belongs to another cycle')
                    receipt={'artifact_receipt':True,'decision_id':task['required_mission_decision'],'kind':'reviewed_mission_lesson','recovered_from':'durable mission lesson'}
                    self.store.finish_task(task_id,receipt)
                    self.store.event(cid,role,'CHECKPOINT_REUSED',{'task_id':task_id,'decision_id':receipt['decision_id']})
                    return receipt
        if isinstance(tools,ResearchTools):
            tools.active_task_id=task_id
        decision_only=role=='scientist' and task.get('proposal_workflow') and (self.settings.max_agent_steps==1 or previous and recent or task.get('resume_scientific_commit'))
        if task.get('venue_review'):
            from .mission_review import prompt
            system_prompt=prompt(role)
        else:
            system_prompt=task_system_prompt(role,decision_only)
        messages=[{'role':'system','content':system_prompt},{'role':'user','content':json.dumps(task,allow_nan=False)}]
        deterministic_failures={}
        self.store.event(cid,role,'AGENT_STARTED',{'task':task})
        committed=set()
        measured=[]; completion_recoveries=0
        step_limit=min(self.settings.max_agent_steps,{'director':4,'scientist':6,'engineer':8,'falsifier':4,'historian':3}[role])
        for step in range(step_limit):
            self.store.check_research_control(cid)
            completion_limit=self.settings.max_completion_tokens if completion_recoveries else min(self.settings.max_completion_tokens,{'historian':6000}.get(role,self.settings.max_completion_tokens))
            if task.get('resume_report_only') and role=='falsifier' and not completion_recoveries:
                completion_limit=min(completion_limit,6000)
            kwargs={'model':self.model,'messages':messages,'tools':tools.available(role),'max_completion_tokens':completion_limit}
            if task.get('venue_review'):
                from .mission_review import tool_specs
                kwargs['tools']=tool_specs(role,task,kwargs['tools'])
            if role=='engineer' and task.get('artifact_workflow'):
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] not in {'experiment','walk_forward','check_program','register_candidate','data_feasibility'}]
            if task.get('resume_finalization'):
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] in {'register_existing_candidate','read_candidate'}]
            if task.get('resume_report_only'):
                kwargs['tool_choice']='none'
            if role=='scientist' and task.get('proposal_workflow') and (task.get('resume_report_only') or step==step_limit-1 or previous and recent or task.get('resume_scientific_commit')):
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] in {'submit_proposal','stop_mission'}]
                # MiMo documents only auto; non-auto values can be ignored.
                # Completion is enforced by the trusted artifact contract below.
                kwargs['tool_choice']='auto' if self.provider=='mimo' else 'required'
                messages.append({'role':'user','content':
                    'If justified, commit this allocation stop using the supplied exact native replay IDs. Otherwise state why evidence is insufficient; this task will remain incomplete. Do not fabricate a stop or claim strategy approval.' if task.get('venue_review') else
                    'Commit exactly one scientific decision now: submit_proposal for a feasible explicit design, or stop_mission with actual grounding. A prose summary is not a committed decision. Both are scientific allocation artifacts, not a candidate or approval.'})
            if role=='engineer' and not task.get('resume_report_only') and (task.get('resume_finalization') or step>=step_limit-2):
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] in {'register_candidate','register_existing_candidate','read_candidate','check_program','data_feasibility','build_candidate','revise_candidate','check_candidate','candidate_feasibility'}]
                messages.append({'role':'user','content':'Implementation phase: use collected evidence and commit the actual code. Repair a failed submission and retry within the remaining turns. Further profiling is unavailable.'})
            elif role=='engineer' and task.get('resume_implementation'):
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] in {'register_candidate','register_existing_candidate','read_candidate','check_program','data_feasibility','walk_forward','experiment','build_candidate','revise_candidate','check_candidate','candidate_feasibility','walk_forward_candidate','experiment_candidate'}]
                messages.append({'role':'user','content':'Repair phase: use the supplied measured constraints and exact draft. Revise an incompatible mechanism, check feasibility and numerical code, then test and commit it. Repeating prior retrieval or unchanged failures is unavailable.'})
            elif role=='historian' and step>=step_limit-2:
                kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name'] in {'record_lesson','record_mission_lesson','read_candidate','read_decision'}]
                if task.get('venue_review') and step==step_limit-1:
                    kwargs['tools']=[t for t in kwargs['tools'] if t['function']['name']=='record_mission_lesson']
                messages.append({'role':'user','content':'Commit the required grounded lessons with record_mission_lesson for required_mission_decision, or record_lesson for a candidate cohort. A prose handoff without those artifacts cannot complete this task.'})
            elif step==step_limit-1 and not (role=='scientist' and task.get('proposal_workflow')):
                kwargs['tool_choice']='none'
                messages.append({'role':'user','content':'This is the final task turn. Return one concise JSON handoff using the evidence already collected. Do not request another tool or invent a successful result.'})
            if self.provider=='mimo':
                kwargs['extra_body']={'thinking':{'type':'enabled'}}
            if kwargs.get('tool_choice')=='none':
                kwargs['tools']=[]
            estimate=math.ceil(len(json.dumps({'messages':messages,'tools':kwargs['tools']}).encode())/2)+completion_limit+2000
            reservation=self.store.reserve(cid,role,estimate,task_id=task_id)
            recorded=False
            try:
                response=self.provider_call(kwargs,{'campaign':cid,'role':role,'task_id':task_id,'request_id':reservation})
                usage=response.usage
                actual=usage.total_tokens if usage else None
                self.store.finish_request(reservation,actual)
                recorded=True
                self.store.check_research_control(cid)
                choice=response.choices[0]
                self.store.event(cid,role,'PROVIDER_RESPONSE',{'task_id':task_id,'request_id':reservation,
                    'configured_model':self.model,'served_model':getattr(response,'model',None),
                    'provider_response_id':getattr(response,'id',None),'system_fingerprint':getattr(response,'system_fingerprint',None),
                    'finish_reason':choice.finish_reason,'prompt_tokens':getattr(usage,'prompt_tokens',None),
                    'completion_tokens':getattr(usage,'completion_tokens',None),
                    'actual_tokens':actual,'completion_limit':completion_limit})
                if choice.finish_reason=='length':
                    partial=choice.message
                    public={'task_id':task_id,
                        'content':(partial.content or '')[:200000],
                        'tool_calls':[{'name':c.function.name,'arguments':c.function.arguments[:200000]}
                            for c in (partial.tool_calls or [])],
                        'executable':False,'reason':'Provider completion truncated; never executed'}
                    self.store.event(cid,role,'INCOMPLETE_PUBLIC_OUTPUT',public)
                    if completion_recoveries==0 and step<step_limit-1:
                        completion_recoveries+=1
                        recovery=handoff_context(dict(task,already_committed_candidates=sorted(committed)),measured,public)
                        if role in {'director','scientist','falsifier'}:
                            recovery['resume_report_only']=True
                        task=recovery
                        # This is a new conversation grounded in public artifacts,
                        # not an assistant tool turn stripped of required MiMo context.
                        messages=[{'role':'system','content':BASE+'\n'+ROLES[role]},
                            {'role':'user','content':json.dumps(recovery,allow_nan=False)}]
                        self.store.event(cid,role,'COMPLETION_RECOVERY',{'task_id':task_id,
                            'remaining_turns':step_limit-step-1,'per_call_allowance':self.settings.max_completion_tokens,
                            'campaign_budget_raised':False,'partial_executed':False})
                        continue
                    raise RuntimeError('Agent response truncated; increase per-call completion budget')
                message=choice.message
                # MiMo requires its opaque reasoning_content be passed back with tool calls.
                # It stays inside this ephemeral conversation; no reasoning is shown or persisted.
                serialized=message.model_dump(exclude_none=True)
                serialized={k:v for k,v in serialized.items() if k in {'role','content','tool_calls','reasoning_content'}}
                messages.append(serialized)
                calls=message.tool_calls or []
                if not calls:
                    content=message.content or ''
                    if content.startswith('```'):
                        content=content.split('\n',1)[1].rsplit('```',1)[0]
                    self.store.event(cid,role,'PUBLIC_HANDOFF_ATTEMPT',{'task_id':task_id,
                        'content':content[:200000],'content_truncated':len(content)>200000,
                        'interpretation':'Public response only; contract validation and task completion remain separate. Opaque reasoning is not persisted.'})
                    try:
                        result=json.loads(content)
                        if not isinstance(result,dict):
                            raise ValueError('Final response must be a JSON object')
                    except (ValueError,json.JSONDecodeError):
                        messages.append({'role':'user','content':'Return the final work summary as exactly one JSON object. Do not repeat tool work.'})
                        continue
                    if role=='scientist' and task.get('proposal_workflow'):
                        messages.append({'role':'user','content':'No scientific decision is committed. Use submit_proposal or stop_mission. An uncommitted JSON narrative cannot finish this task.'})
                        continue
                    if role=='falsifier' and task.get('venue_review'):
                        from .mission_review import validate_review
                        try:
                            validate_review(task,result)
                        except ValueError as error:
                            self.store.event(cid,role,'MISSION_REVIEW_REJECTED',{'task_id':task_id,'error':str(error)})
                            messages.append({'role':'user','content':str(error)+'; repair this critique using supplied exact observations.'})
                            continue
                    if role=='scientist' and task.get('require_strategy_spec'):
                        try:
                            proposal_spec=validate_spec(result.get('strategy_spec'))
                            if 'model' in proposal_spec:
                                profiles=task.get('data_contract',{}).get('profiles',[])
                                index=profiles[0]['development_index'] if profiles else 0
                                feasibility=tools._call('proposal_gate','data_feasibility',{'spec':proposal_spec,'day_index':index})
                                if feasibility['status']!='LABELS_AVAILABLE':
                                    raise ValueError('The proposed fitted design has no usable labels on the measured development frame: '+json.dumps(feasibility,allow_nan=False))
                            result['strategy_spec']=proposal_spec
                        except (ValueError,TypeError,KeyError) as error:
                            feedback={'error':str(error),'required_field':'strategy_spec','instruction':'Repair the explicit design before handing it to the engineer. Use observed segment history and legal recipe relationships; revise incompatible mechanisms. No code or fitting was executed.'}
                            self.store.event(cid,role,'PROPOSAL_CONTRACT_REJECTED',{'task_id':task_id,**feedback})
                            messages.append({'role':'user','content':json.dumps(feedback,allow_nan=False)})
                            continue
                    if role=='engineer' and not committed:
                        messages.append({'role':'user','content':'No implementation has been committed. A summary cannot complete this task. Submit actual code using register_candidate or select a measured existing artifact with register_existing_candidate.'})
                        continue
                    required={m['candidate_id'] for m in task.get('cohort',[])} if role=='historian' else set()
                    if required and not required<=committed:
                        messages.append({'role':'user','content':'Required grounded lessons have not all been recorded. Use record_lesson for the remaining cohort before reporting completion.'})
                        continue
                    if role=='historian' and task.get('required_mission_decision') not in committed and task.get('required_mission_decision'):
                        messages.append({'role':'user','content':'The reviewed mission lesson is not recorded. Use record_mission_lesson; an ungrounded prose summary cannot finish this task.'})
                        continue
                    self.store.event(cid,role,'AGENT_COMPLETED',result)
                    if task_id:
                        self.store.finish_task(task_id,result)
                    return result
                for call in calls:
                    started=time.monotonic()
                    args=None
                    try:
                        args=json.loads(call.function.arguments)
                        if kwargs.get('tool_choice')=='none' or call.function.name not in {t['function']['name'] for t in kwargs['tools']}:
                            raise ValueError('This turn does not advertise that tool; use the current role/phase tools and supplied artifact evidence')
                        if task.get('venue_review'):
                            from .mission_review import validate_call
                            validate_call(task,call.function.name,args)
                        if call.function.name in {'experiment','walk_forward','register_candidate','check_program','build_candidate','revise_candidate','check_candidate','walk_forward_candidate','experiment_candidate'}:
                            # Public numerical work survives even a rejected parse or
                            # worker failure. Opaque provider reasoning is never saved.
                            self.store.event(cid,role,'PROGRAM_ATTEMPT',{'task_id':task_id,'tool':call.function.name,'args':args})
                        if call.function.name in {'submit_proposal','stop_mission'}:
                            self.store.event(cid,role,'SCIENTIFIC_DECISION_ATTEMPT',{'task_id':task_id,'tool':call.function.name,'args':args})
                        result=tools.call(role,call.function.name,args)
                        if call.function.name in {'register_candidate','register_existing_candidate'} and isinstance(result,dict) and result.get('candidate_id'):
                            committed.add(result['candidate_id'])
                        if call.function.name=='record_lesson' and isinstance(result,dict) and result.get('recorded'):
                            committed.add(args['candidate_id'])
                        if call.function.name=='record_mission_lesson' and isinstance(result,dict) and result.get('recorded'):
                            committed.add(result['decision_id'])
                    except ResearchStopped:
                        raise
                    except Exception as error:
                        result={'error':str(error)}
                        if isinstance(error,ValueError):
                            signature=digest({'tool':call.function.name,'args':args if args is not None else call.function.arguments,'error':str(error)})
                            count=deterministic_failures.get(signature,0)+1
                            deterministic_failures[signature]=count
                            result['identical_deterministic_failures']=count
                            result['repair_instruction']='This exact request failed. Correct the numeric contract or revise the mechanism; repeating unchanged arguments cannot add evidence.'
                            if count>=3:
                                self.store.event(cid,role,'REPEATED_TOOL_FAILURE',{'task_id':task_id,'tool':call.function.name,'signature':signature,'count':count,'error':str(error)})
                                raise RepeatedToolFailure(f'{role} repeated the same deterministic {call.function.name} failure {count} times; checkpoint preserved, no further paid turn will start') from error
                    self.store.event(cid,role,'TOOL_RESULT',{'task_id':task_id,'tool':call.function.name,'seconds':time.monotonic()-started,'result':result})
                    measured.append({'tool':call.function.name,'args':args,'result':result})
                    if role=='scientist' and task.get('proposal_workflow') and call.function.name in {'submit_proposal','stop_mission'} and isinstance(result,dict) and result.get('decision_id'):
                        from .proposals import load
                        decision=load(self.store,result['decision_id'],cid)
                        if decision['task_id']!=task_id or decision['cycle']!=tools.cycle_id:
                            raise ValueError('Scientific decision belongs to a different task')
                        receipt=dict(result,artifact_receipt=True)
                        self.store.finish_task(task_id,receipt)
                        self.store.event(cid,role,'AGENT_COMPLETED',receipt)
                        return receipt
                    serialized_result=json.dumps(result,allow_nan=False)
                    if len(serialized_result)>50000:
                        serialized_result=json.dumps({'error':'Result too large; query a more focused artifact'})
                    messages.append({'role':'tool','tool_call_id':call.id,'content':serialized_result})
                required={m['candidate_id'] for m in task.get('cohort',[])} if role=='historian' else set()
                mission_required=task.get('required_mission_decision') if role=='historian' else None
                if (role=='engineer' and committed) or (role=='historian' and required and required<=committed) or (mission_required and mission_required in committed):
                    # The model authored/selected the actual artifacts. An
                    # operational receipt closes the task without another LLM
                    # request merely restating code or lessons already stored.
                    receipt={'artifact_receipt':True,'committed_candidate_ids':sorted(committed),'kind':'strategy_submission' if role=='engineer' else 'grounded_lessons'}
                    self.store.event(cid,role,'AGENT_COMPLETED',receipt)
                    if task_id:
                        self.store.finish_task(task_id,receipt)
                    return receipt
            except ResearchStopped as error:
                if not recorded:
                    self.store.finish_request(reservation,getattr(error,'provider_actual_tokens',None),'FAILED')
                raise
            except Exception as error:
                if not recorded:
                    self.store.finish_request(reservation,getattr(error,'provider_actual_tokens',None),'FAILED')
                self.store.event(cid,role,'AGENT_FAILED',{'error_type':type(error).__name__})
                raise AgentTaskFailure(role,error) from error
        raise RuntimeError(f'{role} exceeded the {step_limit}-step task budget')

class Desk:
    def __init__(self,store,catalog,evaluator,settings,runtime):
        self.store,self.catalog,self.evaluator,self.settings,self.runtime=store,catalog,evaluator,settings,runtime

    def run(self,cid,cycles=1,own_lease=True):
        campaign=self.store.get_campaign(cid)
        if campaign['status']=='SEALED':
            raise ValueError('Sealed campaign cannot resume research')
        plan=campaign['plan']
        outcomes=[]
        with (self.store.lease('research',max(3600,cycles*7200)) if own_lease else nullcontext()):
            self.store.set_status(cid,'RUNNING')
            try:
                for _ in range(cycles):
                    cycle_id=self.store.start_cycle(cid)
                    tools=ResearchTools(cid,self.store,self.catalog,self.evaluator,self.settings,cycle_id)
                    cost_hurdles={'taker_roundtrip_fee_and_extra_slippage_bps':2*(self.settings.taker_fee_bps+self.settings.slippage_bps),'maker_entry_taker_exit_fee_and_extra_slippage_bps':self.settings.maker_fee_bps+self.settings.taker_fee_bps+self.settings.slippage_bps,'spread_additional':True,'maker_fills_verified':False}
                    from .research_map import research_map
                    from .planning import planning_contract
                    data_contract=planning_contract(self.store,self.catalog,cid,self.settings)
                    task={'objective':'Find a distinct, executable microstructure edge, or a useful falsification. Optimize evidence gained per experiment and API token. Use the measured research map to avoid repeated programs, spend on unresolved uncertainty, and explain which observation will change the next allocation decision.','development_dates':len(plan['development']),'validation_dates':len(plan['validation']),'costs':self.settings.execution_dict(),'cost_hurdles':cost_hurdles,'memory':self.store.recall(10),'research_map':research_map(self.store,cid),'sampled':plan['sampled'],
                        'research_policy':{'assessment_order':'Precommitted full development screen, then full validation/stress only when confirmation_eligible. Screening cannot promote.',
                            'negative_results':'Activity/cost failures concern the observed configuration; propose a decisive mechanism test rather than permanently rejecting a class.',
                            'fresh_dates':'Check recorded exposure before claiming freshness. Prepared/training dates are already exposed.',
                            'confirmation_after_screen':'Three complete dates, net positive cash comparison, adequate activity/date breadth and fitted custom-feature increment are allocation heuristics, not statistical approval'},
                        'data_contract':data_contract,
                        'prepared_development_indices':[i for i,r in enumerate(plan['development']) if getattr(self.catalog,'cached',lambda *a:False)(r,plan.get('max_rows_per_day'))],
                        'research_capabilities':{'formula':'causal numerical program','fitted':['ridge','quadratic_ridge','xgboost'],
                            'custom_features':'Agent-authored causal feature matrix with stable named columns; spec.model.custom_features=true',
                            'fitting_protocol':'Earlier development dates only; inner temporal penalty selection; outer development walk-forward; frozen validation/holdout',
                            'default_min_training_dates':10,'model_use_possible':len(plan['development'])>6}}
                    print('Director: choosing the next evidence-driven research mission',flush=True)
                    mission=self.runtime.run(cid,'director',task,tools,cycle_id+':director')
                    print('Scientist: inspecting data and designing an experiment',flush=True)
                    proposal=self.runtime.run(cid,'scientist',{'mission':mission,'costs':task['costs'],'cost_hurdles':cost_hurdles,'data_contract':data_contract,'proposal_workflow':True,
                        'required_deliverable':'Commit a scientific decision with submit_proposal or stop_mission. A proposal records exact strategy_spec, supported mature history, implementation design, baseline/ablation, stop condition and 1..3 outcome-blind development indices. Check actual history and training relationships. If current data or measured development evidence does not justify this allocation, stop/redesign the mission with exact grounding; no fake strategy is required. The independent falsifier reviews a stop. The engineer owns deployable code.'},tools,cycle_id+':scientist:decisions-v1')
                    from .proposals import load
                    decision=load(self.store,proposal.get('decision_id'),cid)
                    if decision['cycle']!=cycle_id or decision['task_id']!=cycle_id+':scientist:decisions-v1':
                        raise ValueError('Scientific handoff does not belong to the current cycle')
                    if decision['kind']=='MISSION_STOP':
                        print('Falsifier: reviewing the evidenced mission stop',flush=True)
                        review_scope={'venue_review':True,'mission_decision_id':proposal['decision_id']} if decision.get('basis')=='MEASURED_VENUE' else {}
                        critique=self.runtime.run(cid,'falsifier',{'mission':mission,'mission_stop':decision,'data_contract':data_contract,**review_scope,
                            'required_deliverable':'Return an independent critique of this allocation stop. Read the exact decision when needed, distinguish observed configuration/history limitations from a strategy-class verdict, and propose a decisive supported redesign. You cannot grant economic approval.'},tools,cycle_id+':falsifier:mission')
                        self.runtime.run(cid,'historian',{'mission_stop':decision,'critique':critique,'required_mission_decision':proposal['decision_id'],**review_scope,
                            'required_deliverable':'Call record_mission_lesson for this exact decision. Preserve actual grounding and independent critique; a stop is a research allocation judgment, not a profitable strategy or permanent feature rejection.'},tools,cycle_id+':historian:mission')
                        self.store.complete_cycle(cycle_id,'MISSION_STOP')
                        outcome={'cycle_id':cycle_id,'kind':'MISSION_STOP','decision_id':proposal['decision_id'],'candidates':[]}
                        outcomes.append(outcome)
                        self.store.event(cid,'desk','MISSION_DECISION_COMPLETED',outcome)
                        continue
                    if decision['kind']!='PROPOSAL':
                        raise ValueError('Unknown scientific allocation decision')
                    proposal=dict(decision,decision_id=proposal['decision_id'])
                    print('Engineer: building and testing a numerical implementation',flush=True)
                    implementation=self.runtime.run(cid,'engineer',{'mission':mission,'proposal':proposal,'execution':task['costs'],'cost_hurdles':cost_hurdles,'research_policy':task['research_policy'],'data_contract':data_contract,'artifact_workflow':True,'required_deliverable':'Build, check and test the actual numerical implementation, then submit it. Use data feasibility before fitting; revise any proposal incompatible with observed history. Prefer immutable candidate IDs after building to avoid retyping code/recipes. The desk screens actual submissions before confirmation.'},tools,cycle_id+':engineer')
                    with self.store.connect() as db:
                        submitted=db.execute("SELECT payload FROM events WHERE campaign=? AND kind='SUBMITTED' ORDER BY id",(cid,)).fetchall()
                    ids=list(dict.fromkeys(json.loads(r[0])['id'] for r in submitted if json.loads(r[0]).get('cycle')==cycle_id))
                    candidates=[self.store.get_candidate(i) for i in ids if self.store.get_candidate(i)['campaign']==cid]
                    if not candidates:
                        raise RuntimeError('Engineer did not submit an immutable candidate; inspect agent artifacts')
                    cohort=[]
                    for candidate in candidates:
                        if candidate['status'] in {'PROPOSED','EVALUATION_ERROR'}:
                            print(f'Validator: development screening for {candidate["id"]}',flush=True)
                            report=self.evaluator.screen(cid,candidate)
                            if report['confirmation_eligible']:
                                print(f'Validator: full chronological validation and stress for {candidate["id"]}',flush=True)
                                report=self.evaluator.validate(cid,candidate['id'])
                                report=dict(report,phase='validation')
                            else:
                                self.store.event(cid,'desk','CONFIRMATION_SKIPPED',{'candidate':candidate['id'],'screen_status':report['status'],'reasons':report['reasons']+report['confirmation_reasons']})
                        else:
                            report=dict(candidate['report'] or {},phase='validation')
                        cohort.append({'candidate_id':candidate['id'],'title':candidate['title'],'assessment':report})
                    print('Falsifier: reviewing cohort evidence and decisive comparisons',flush=True)
                    critique=self.runtime.run(cid,'falsifier',{'cohort':compact_cohort(cohort),'proposal':proposal,'required_deliverable':'Compare the submitted mechanisms and ablations, inspect code when necessary, and give one grounded critique with a decisive follow-up experiment. Respect evidence_phase: a development screen is not validation or a permanent rejection. Separate operational errors from economic findings, and check development_coverage before calling dates fresh. Approval remains deterministic.'},tools,cycle_id+':falsifier:cohort')
                    self.runtime.run(cid,'historian',{'cohort':compact_cohort(cohort),'critique':critique,'required_deliverable':'Call record_lesson for each submitted candidate, preserving observed evidence and uncertainty. Do not repeat profiles or code inspection; the tool attaches full measured grounding. Return one concise organizational handoff.'},tools,cycle_id+':historian:cohort')
                    self.store.event(cid,'desk','CYCLE_COMPLETED',{'candidates':[c['id'] for c in candidates]})
                    self.store.complete_cycle(cycle_id)
                    outcomes.append({'cycle_id':cycle_id,'kind':'STRATEGY_COHORT','decision_id':proposal.get('decision_id'),
                        'candidates':[c['id'] for c in candidates]})
                self.store.set_status(cid,'READY')
            except ResearchTimeLimit:
                self.store.suspend_cycle(cid,'PAUSED','Supervisor time limit')
                self.store.set_status(cid,'STOPPED')
                self.store.event(cid,'desk','TIME_LIMIT',{'reason':'Supervisor deadline; current unit preserved'})
                raise
            except ResearchStopped:
                self.store.suspend_cycle(cid,'PAUSED','User stop request')
                self.store.set_status(cid,'STOPPED')
                self.store.event(cid,'desk','STOPPED',{'reason':'User stop request; current call completed'})
            except BaseException as error:
                self.store.suspend_cycle(cid,'FAILED',type(error).__name__)
                self.store.set_status(cid,'STOPPED')
                raise
        status=self.store.status()
        return {'campaign':cid,'status':self.store.get_campaign(cid)['status'],
            'cycle_outcomes':outcomes,
            'candidates':[{'id':c['id'],'title':c['title'],'status':c['status']} for c in self.store.candidates(cid)],
            'usage':[u for u in status['usage'] if u['campaign']==cid],
            'tasks':[t for t in status['tasks'] if t['campaign']==cid]}
