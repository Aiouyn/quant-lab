import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from .settings import Settings
from .store import Store
from .data import Catalog
from .evaluation import Evaluator

BASELINE='''import polars as pl
import numpy as np

def compute_signal(df: pl.DataFrame) -> pl.Series:
    pressure = df['qi_5'].to_numpy()
    flow = df['ofi_norm'].clip(-3.0, 3.0).to_numpy()
    return pl.Series('signal', np.tanh(pressure + 0.25 * flow))
'''

def emit(value):
    print(json.dumps(value,indent=2,allow_nan=False),flush=True)

def parser():
    p=argparse.ArgumentParser(description='Quant Lab v2: an evidence-driven research desk and Hyperliquid shadow trader')
    p.add_argument('--home',type=Path,help='Isolated evidence/working directory')
    p.add_argument('--data-dir',type=Path,help='Directory of daily OKX L2 tar.gz archives')
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('inventory',help='Inspect archive names and disk sizes without hydrating contents')
    commands.add_parser('venue-captures',help='List public target-venue capture inventory and process-aware liveness')
    venue=commands.add_parser('venue-study',help='Study a frozen public capture; observations and hypothetical depth sweeps, no strategy PnL')
    venue.add_argument('capture_id'); venue.add_argument('--notionals',type=float,nargs='+',help='1..6 independent USD sweep scenarios')
    models=commands.add_parser('venue-models',help='List exact frozen models available for an immutable candidate')
    models.add_argument('candidate')
    vr=commands.add_parser('venue-replay',help='Retrospective quote/depth sensitivity on an exact frozen capture study; no forward acceptance')
    vr.add_argument('campaign');vr.add_argument('candidate');vr.add_argument('study');vr.add_argument('--model-key')
    read=commands.add_parser('venue-replay-read',help='Read an exact immutable retrospective venue replay and verify journals')
    read.add_argument('replay_id')
    plan=commands.add_parser('plan',help='Freeze chronological development/validation/holdout dates')
    plan.add_argument('--days',type=int,help='Evenly sampled date count; omitted means all available dates')
    plan.add_argument('--rows-per-day',type=int,help='Development-only sampling mode; cannot be promoted')
    plan.add_argument('--token-budget',type=int,default=None,help='Optional local aggregate cap; omitted means no campaign cap. Provider quota still applies.')
    research=commands.add_parser('research',help='Run bounded MiMo tool-using research cycles')
    research.add_argument('campaign'); research.add_argument('--cycles',type=int,default=1)
    research.add_argument('--max-steps',type=int,default=8)
    research.add_argument('--request-timeout',type=int,default=900,help='Provider wall-clock deadline in seconds')
    research.add_argument('--max-completion-tokens',type=int,default=18000,help='Bounded allowance including provider thinking')
    autonomous=commands.add_parser('autonomous',help='Bounded research with checkpoint recovery, budget and time limits')
    autonomous.add_argument('campaign'); autonomous.add_argument('--cycles',type=int,default=10)
    autonomous.add_argument('--hours',type=float,default=8)
    autonomous.add_argument('--max-retries',type=int,default=2)
    autonomous.add_argument('--request-timeout',type=int,default=900)
    autonomous.add_argument('--max-completion-tokens',type=int,default=18000)
    commands.add_parser('status',help='Report durable campaign/task/API usage')
    mapping=commands.add_parser('research-map',help='Measured research coverage, role usage and next-experiment suggestions')
    mapping.add_argument('campaign')
    prepare=commands.add_parser('prepare-data',help='Cache and audit frozen development dates with bounded parallel IO')
    prepare.add_argument('campaign'); prepare.add_argument('--limit',type=int)
    prepare.add_argument('--workers',type=int,default=2)
    stop=commands.add_parser('stop',help='Stop research after the current provider call'); stop.add_argument('campaign')
    abandon=commands.add_parser('abandon-cycle',help='Preserve an unfinished cycle and let the next mission change direction; no budget reset')
    abandon.add_argument('campaign'); abandon.add_argument('--reason',required=True)
    validate=commands.add_parser('validate'); validate.add_argument('campaign'); validate.add_argument('candidate')
    screen=commands.add_parser('screen',help='Development-only triage before expensive independent confirmation')
    screen.add_argument('campaign'); screen.add_argument('candidate')
    screen.add_argument('--day-indices',type=int,nargs='+',help='Precommit 1..3 outer development indices; defaults prefer cached dates')
    walk=commands.add_parser('walk-forward',help='Nested temporal fitting and outer development replay; no holdout use')
    walk.add_argument('campaign'); walk.add_argument('candidate'); walk.add_argument('--folds',type=int,default=5)
    walk.add_argument('--start-index',type=int)
    walk.add_argument('--end-index',type=int)
    seal=commands.add_parser('seal',help='Freeze cohort and consume holdout exactly once'); seal.add_argument('campaign')
    export=commands.add_parser('export',help='Export a hashed strategy bundle')
    export.add_argument('candidate'); export.add_argument('--exploratory',action='store_true')
    portfolio=commands.add_parser('portfolio',help='Allocate after historical promotion and precommitted individual Hyperliquid forward acceptance'); portfolio.add_argument('campaign')
    forward_plan=commands.add_parser('forward-plan',help='Freeze the future Hyperliquid observation window for an exact PAPER_READY bundle')
    forward_plan.add_argument('candidate'); forward_plan.add_argument('--new-window',action='store_true',help='Explicit new future window; prior failed windows remain recorded')
    forward_assess=commands.add_parser('forward-assess',help='Reconcile and assess individual precommitted forward evidence before portfolio formation')
    forward_assess.add_argument('candidate')
    demo=commands.add_parser('demo',help='Explicitly synthetic, no-API integration test on real sampled data')
    demo.add_argument('--days',type=int,default=12); demo.add_argument('--rows-per-day',type=int,default=1000)
    paper=commands.add_parser('paper',help='Dry run using public Hyperliquid books and a local shadow account')
    paper.add_argument('bundle',type=Path); paper.add_argument('--samples',type=int,default=1800)
    paper.add_argument('--interval',type=float,default=1.0); paper.add_argument('--testnet',action='store_true')
    paper.add_argument('--coin',default='BTC'); paper.add_argument('--exploratory',action='store_true')
    serve=commands.add_parser('serve',help='Local evidence dashboard; research starts only when requested')
    serve.add_argument('--port',type=int,default=8888)
    commands.add_parser('check-provider',help='Check provider model availability; no agent campaign')
    collect=commands.add_parser('collect',help='Capture public Hyperliquid L2, trades and funding context; no wallet')
    collect.add_argument('--seconds',type=int,default=3600)
    collect.add_argument('--testnet',action='store_true')
    collect.add_argument('--max-reconnects',type=int,default=20)
    collect.add_argument('--fast',action='store_true',help='Fast five-level L2 feed; default is the slower twenty-level feed')
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    settings=Settings()
    if args.home:
        settings=replace(settings,home=args.home.resolve())
    if args.data_dir:
        settings=replace(settings,data_dir=args.data_dir.resolve())
    if args.command in {'research','autonomous'}:
        if not 30<=args.request_timeout<=1800:
            raise ValueError('Provider deadline must be 30..1800 seconds')
        if not 2000<=args.max_completion_tokens<=32000:
            raise ValueError('Completion allowance must be 2000..32000 tokens')
        settings=replace(settings,max_agent_steps=getattr(args,'max_steps',8),request_timeout=args.request_timeout,max_completion_tokens=args.max_completion_tokens)
    store=Store(settings.home); catalog=Catalog(settings.data_dir,settings.home); evaluator=Evaluator(store,catalog,settings)
    if args.command=='venue-captures':
        from .venue_study import inventory
        emit(inventory(store.home))
    elif args.command=='venue-models':
        from .venue_replay import list_models
        emit(list_models(store,args.candidate,settings))
    elif args.command=='venue-replay-read':
        from .venue_replay import load
        emit(load(store,args.replay_id))
    elif args.command=='venue-replay':
        from .venue_replay import replay
        with store.lease('research'): emit(replay(store,settings,args.campaign,args.candidate,args.study,args.model_key))
    elif args.command=='venue-study':
        from .venue_study import study
        with store.lease('research'):
            emit(study(store,settings,args.capture_id,args.notionals))
    elif args.command=='inventory':
        records=catalog.inventory()
        emit({'archives':len(records),'bytes':sum(r['size'] for r in records),'first_date':records[0]['date'],'last_date':records[-1]['date'],'sample':catalog.profile(records[0],1000)})
    elif args.command=='plan':
        if (args.token_budget is not None and args.token_budget<1000) or (args.rows_per_day is not None and args.rows_per_day<100):
            raise ValueError('Optional token budget >=1000; research sample >=100 rows')
        plan=catalog.plan(args.days,args.rows_per_day)
        cid=store.campaign(plan,args.token_budget)
        emit({'campaign':cid,'plan_hash':plan['hash'],'dates':{phase:len(plan[phase]) for phase in ('development','validation','holdout')},'sampled':plan['sampled'],'token_limit':args.token_budget,'next_command':f'python -m quantdesk research {cid} --cycles 1'})
    elif args.command=='research':
        if not 1<=args.cycles<=100 or not 1<=args.max_steps<=20:
            raise ValueError('Cycles 1..100; agent steps 1..20')
        from .agents import AgentRuntime,Desk
        emit(Desk(store,catalog,evaluator,settings,AgentRuntime(store,settings)).run(args.campaign,args.cycles))
    elif args.command=='autonomous':
        from .agents import AgentRuntime,Desk
        from .autonomy import Supervisor
        desk=Desk(store,catalog,evaluator,settings,AgentRuntime(store,settings))
        emit(Supervisor(desk,store).run(args.campaign,args.cycles,args.hours,args.max_retries))
    elif args.command=='status':
        emit(store.status())
    elif args.command=='abandon-cycle':
        emit(store.abandon_cycle(args.campaign,args.reason))
    elif args.command=='research-map':
        from .research_map import research_map
        emit(research_map(store,args.campaign))
    elif args.command=='prepare-data':
        from .ingestion import prepare_data
        emit(prepare_data(store,catalog,args.campaign,args.limit,args.workers))
    elif args.command=='stop':
        campaign=store.get_campaign(args.campaign)
        from .operations import job_snapshot
        active_job=any(j['running'] and j.get('campaign')==args.campaign for j in job_snapshot(store))
        if campaign['status']!='RUNNING' and not active_job:
            raise ValueError('Campaign is not currently running')
        store.set_status(args.campaign,'STOP_REQUESTED')
        emit({'campaign':args.campaign,'status':'STOP_REQUESTED','effective':'After the current provider call or archive replay'})
    elif args.command=='screen':
        with store.lease('research'):
            emit(evaluator.screen(args.campaign,store.get_candidate(args.candidate),args.day_indices))
    elif args.command=='validate':
        with store.lease('research'):
            if store.get_campaign(args.campaign)['status']=='SEALED':
                raise ValueError('Sealed campaign cannot be revalidated')
            emit(evaluator.validate(args.campaign,args.candidate))
    elif args.command=='walk-forward':
        with store.lease('research'):
            if store.get_campaign(args.campaign)['status']=='SEALED':
                raise ValueError('Sealed campaign cannot resume research')
            candidate=store.get_candidate(args.candidate)
            if candidate['campaign'] != args.campaign:
                raise ValueError('Candidate must belong to the requested campaign')
            emit(evaluator.walk_forward(args.campaign,candidate,args.folds,args.start_index,args.end_index))
    elif args.command=='seal':
        with store.lease('research'):
            emit(evaluator.seal(args.campaign))
    elif args.command=='export':
        candidate=store.get_candidate(args.candidate)
        if candidate['status']!='PAPER_READY' and not args.exploratory:
            raise ValueError('Candidate has not passed validation and holdout; use --exploratory for a labeled research shadow test')
        emit({'bundle':str(evaluator.export(candidate,candidate['report'] or {},args.exploratory))})
    elif args.command=='portfolio':
        from .portfolio import build_portfolio
        directory,manifest=build_portfolio(store,args.campaign,settings)
        emit({'portfolio':str(directory),'manifest':manifest})
    elif args.command in {'forward-plan','forward-assess'}:
        from .paper import load_bundle
        from .forward import plan_forward,assess_forward
        candidate=store.get_candidate(args.candidate)
        if candidate['status']!='PAPER_READY':
            raise ValueError('Forward portfolio acceptance requires historical PAPER_READY status')
        manifest,_=load_bundle(store.home/'bundles'/candidate['id'])
        emit(plan_forward(store,manifest,settings,args.new_window) if args.command=='forward-plan' else assess_forward(store,manifest,settings))
    elif args.command=='demo':
        # This is a labeled infrastructure smoke test, never an AI discovery or promotion.
        plan=catalog.plan(args.days,args.rows_per_day); cid=store.campaign(plan,1000)
        spec={'title':'DEMO: causal depth-flow baseline','rationale':'Known hand-written baseline used only to exercise the infrastructure.','falsification':'Reject if net returns do not survive target-venue costs and independent dates.','holding_seconds':30,'threshold':0.5,'warmup_rows':30,'mode':'taker'}
        ident,_=store.candidate(cid,spec,BASELINE)
        report=evaluator.validate(cid,ident)
        directory=evaluator.export(store.get_candidate(ident),report,exploratory=True)
        emit({'campaign':cid,'candidate':ident,'synthetic_agent':True,'real_l2_data':True,'validation_status':report['status'],'reasons':report['reasons'],'statistics':report.get('base',{}).get('statistics',{}),'bundle':str(directory)})
    elif args.command=='paper':
        from .paper import PaperSession
        if args.coin!='BTC':
            raise ValueError('These bundles were researched on BTC; coin transfer requires a new research dataset')
        session=PaperSession(args.bundle,settings.home,settings,args.exploratory)
        emit(session.run(args.samples,args.interval,args.testnet,args.coin))
    elif args.command=='serve':
        from .web import serve
        serve(store,catalog,evaluator,settings,args.port)
    elif args.command=='check-provider':
        import config
        from openai import OpenAI
        if not config.ACTIVE_KEY:
            raise ValueError('No provider credential')
        client=OpenAI(api_key=config.ACTIVE_KEY,base_url=config.ACTIVE_BASE_URL,max_retries=0,timeout=30)
        models=client.models.list()
        identifiers=[m.id for m in models.data]
        emit({'provider':config.ACTIVE_PROVIDER,'model':config.DEFAULT_MODEL,'listed':config.DEFAULT_MODEL in identifiers,'available_models':identifiers,'base_url':config.ACTIVE_BASE_URL})
    elif args.command=='collect':
        from .collector import Collector
        emit(Collector(settings.home,testnet=args.testnet,book_fast=args.fast).run(args.seconds,args.max_reconnects))
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'Quant Lab: {type(error).__name__}: {error}',file=sys.stderr)
        raise SystemExit(1)
