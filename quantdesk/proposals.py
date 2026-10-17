"""Immutable scientific allocation decisions with actual development grounding."""
import json
import os
import time
from .store import digest
from .strategy import validate_spec


def text(args,name):
    value=args.get(name)
    if not isinstance(value,str) or not value.strip() or len(value)>12000:
        raise ValueError(name+' must be a nonempty public explanation of at most 12000 characters')
    return value


def owner(tools,role):
    task_id=getattr(tools,'active_task_id',None)
    with tools.store.connect() as db:
        task=db.execute('SELECT campaign,agent,status,pid FROM tasks WHERE id=?',(task_id,)).fetchone()
    if not task or task['campaign']!=tools.cid or task['agent']!=role or task['status']!='RUNNING' or task['pid']!=os.getpid():
        raise ValueError('Decision tools require the current owning role task')
    if tools.store.get_campaign(tools.cid)['status']=='SEALED':
        raise ValueError('Sealed campaigns cannot record new scientific allocation decisions')
    return task_id


def load(store,decision_id,cid):
    with store.connect() as db:
        row=db.execute('SELECT * FROM research_decisions WHERE id=? AND campaign=?',(decision_id,cid)).fetchone()
    if row is None:
        raise ValueError('Unknown decision in this campaign')
    value=json.loads(row['payload'])
    if digest(value)!=decision_id:
        raise ValueError('Scientific decision integrity failed')
    return value


def persist(tools,task_id,kind,payload):
    value={'version':1,'campaign':tools.cid,'cycle':tools.cycle_id,'task_id':task_id,'kind':kind,**payload}
    ident=digest(value)
    with tools.store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        existing=db.execute('SELECT id FROM research_decisions WHERE task=?',(task_id,)).fetchone()
        if existing and existing['id']!=ident:
            raise ValueError('This task already has an immutable scientific decision; use a new reviewed task to change direction')
        if not existing:
            db.execute('INSERT INTO research_decisions VALUES(?,?,?,?,?,?,?)',
                (ident,tools.cid,tools.cycle_id,task_id,kind,json.dumps(value,allow_nan=False),time.time()))
    tools.store.event(tools.cid,'scientist','SCIENTIFIC_DECISION_RECORDED',{'decision_id':ident,'kind':kind,'task_id':task_id,'cycle':tools.cycle_id})
    return {'decision_id':ident,'decision_kind':kind,'decision':value,'submitted_candidate':False,
        'interpretation':'Scientific allocation artifact; no code, economic replay, promotion or learning-benefit proof'}


def assess(tools,spec,index):
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(tools.plan['development']):
        raise ValueError('Scientific feasibility requires a development index')
    if 'model' in spec:
        return tools._call('proposal_service','data_feasibility',{'spec':spec,'day_index':index})
    from .models import label_feasibility,validate_recipe
    requested=tools.plan.get('max_rows_per_day'); record=tools.plan['development'][index]
    prepared=bool(getattr(tools.catalog,'cached',lambda *a:False)(record,requested))
    rows=requested if prepared else min(requested or 10000,10000)
    tools.store.expose_dates(tools.cid,[record],'development')
    frame,quality=tools.catalog.load(record,rows)
    result=dict(label_feasibility(frame,spec,tools.settings.execution_dict(),validate_recipe({'stride_seconds':1})),
        date=record['date'],rows=len(frame),bounded_rows_requested=rows,full_prepared_frame_used=prepared and requested is None,
        quality=quality,diagnostic_basis='Formula history/holding label-window compatibility only; no fitted model, predictions or PnL')
    tools.store.event(tools.cid,'proposal_service','DATA_FEASIBILITY',result)
    return result


def submit(tools,args):
    task_id=owner(tools,'scientist'); spec=validate_spec(args.get('strategy_spec'))
    history=args.get('effective_history_rows')
    if isinstance(history,bool) or not isinstance(history,int) or not 1<=history<=5001:
        raise ValueError('effective_history_rows must be an integer in 1..5001')
    if spec['warmup_rows']<history-1:
        raise ValueError('Declared fully mature effective history requires warmup_rows >= effective_history_rows - 1')
    indices=args.get('development_indices')
    if not isinstance(indices,list) or not 1<=len(indices)<=3 or any(isinstance(i,bool) or not isinstance(i,int) for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('Choose 1..3 unique integer development indices before replay')
    for index in indices:
        if not 0<=index<len(tools.plan['development']):
            raise ValueError('Proposals cannot choose validation/holdout indices')
        if 'model' in spec and index<spec['model']['min_train_days']:
            raise ValueError('A proposed fitted outer date needs the declared minimum earlier training dates')
    explanations={k:text(args,k) for k in ('hypothesis','implementation_design','baseline','ablation','stop_condition','limitations')}
    checks=[]
    for index in indices:
        tools.store.check_research_control(tools.cid)
        measured=assess(tools,spec,index)
        if measured['status']!='LABELS_AVAILABLE':
            raise ValueError('Proposal cannot mature on the chosen development frame: '+json.dumps(measured,allow_nan=False))
        checks.append({'development_index':index,**measured})
    return persist(tools,task_id,'PROPOSAL',{'strategy_spec':spec,'effective_history_rows':history,
        'development_indices':indices,**explanations,'grounding':{'phase':'development_metadata','feasibility':checks,
            'interpretation':'Observed compatibility only; no forecast skill, net performance, statistical approval or guaranteed fit sufficiency'}})


def stop(tools,args):
    task_id=owner(tools,'scientist')
    explanations={k:text(args,k) for k in ('reason','next_experiment','limitations')}
    basis=args.get('basis')
    if basis=='DATA_INCOMPATIBLE':
        raw=args.get('proposed_spec')
        if not isinstance(raw,dict):
            raise ValueError('A data stop requires the actual proposed specification')
        # Preserve an invalid fitting recipe but validate the rest; numerical
        # structural diagnostics are available even when that recipe is invalid.
        base=validate_spec({k:v for k,v in raw.items() if k!='model'})
        if 'model' not in raw:
            raw=base
        measured=assess(tools,raw,args.get('day_index'))
        if measured['status'] not in {'NO_USABLE_LABELS','INVALID_RECIPE'}:
            raise ValueError('Observed data supports labels; data incompatibility is not established for this design')
        grounding={'phase':'development_metadata','proposed_spec':raw,'feasibility':measured}
    elif basis=='MEASURED_DEVELOPMENT':
        ids=args.get('candidate_ids')
        if not isinstance(ids,list) or not 1<=len(ids)<=3 or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('A measured allocation stop requires 1..3 exact candidate IDs')
        observations=[]
        for ident in ids:
            candidate=tools.store.get_candidate(ident)
            with tools.store.connect() as db:
                row=db.execute("SELECT id,campaign,kind,payload FROM memories WHERE candidate=? AND kind IN ('DEVELOPMENT_EXPERIMENT','WALK_FORWARD_EXPERIMENT','DEVELOPMENT_SCREEN') ORDER BY id DESC LIMIT 1",(ident,)).fetchone()
            if not row:
                raise ValueError('No measured development evidence exists for '+ident)
            evidence=json.loads(row['payload'])
            observations.append({'candidate_id':ident,'spec_hash':digest(candidate['spec']),'code_hash':digest(candidate['code']),
                'source_memory_id':row['id'],'source_campaign':row['campaign'],'source_kind':row['kind'],
                'source_digest':digest(evidence),'observed_development':evidence})
        grounding={'phase':'development','observations':observations,
            'interpretation':'Exact prior observed configurations, possibly from earlier campaigns; the choice to stop current work is an allocation judgment, not proof against a feature family'}
    elif basis=='MEASURED_VENUE':
        from .venue_replay import load as read_replay
        ids=args.get('replay_ids')
        if not isinstance(ids,list) or not 1<=len(ids)<=3 or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('A venue allocation stop requires 1..3 exact unique replay IDs')
        observations=[];seen=set()
        for ident in ids:
            tools.store.check_research_control(tools.cid)
            report=read_replay(tools.store,ident)['report']
            if report.get('format')!='quantdesk-venue-replay-v1' or report.get('forward_eligible') is not False or report.get('strategy_promoted') is not False or report.get('orders_submitted_to_exchange')!=0:
                raise ValueError('Only retrospective unpromoted venue research can ground this stop')
            candidate=tools.store.get_candidate(report['candidate_id'])
            if report['code_hash']!=digest(candidate['code']) or report['spec_hash']!=digest(candidate['spec']):
                raise ValueError('Venue replay differs from the immutable candidate')
            with tools.store.connect() as db:
                memories=db.execute("SELECT id,campaign,payload FROM memories WHERE candidate=? AND kind='VENUE_REPLAY' ORDER BY id DESC LIMIT 100",(candidate['id'],)).fetchall()
            source=next((r for r in memories if json.loads(r['payload'])=={'replay_id':ident,**report}),None)
            if source is None: raise ValueError('Venue replay lacks its registered measured evidence')
            key=(report['candidate_id'],report['capture_id'],report['model_hash'],digest(report['execution']),digest(report['shadow_execution']))
            if key in seen: raise ValueError('Repeated replay versions cannot become separate grounding observations')
            seen.add(key)
            observations.append({'replay_id':ident,'candidate_id':candidate['id'],'source_memory_id':source['id'],
                'source_campaign':source['campaign'],'source_kind':'VENUE_REPLAY','source_digest':digest(report),
                'observed_venue_replay':report})
        grounding={'phase':'venue_development','observations':observations,
            'interpretation':'Exact retrospective public captures and frozen artifacts. Fill alternatives, versions and overlapping captures are not independent confirmation. Stop concerns this allocation only; no venue calibration or strategy-class verdict.'}
    else:
        raise ValueError('Stop basis must be DATA_INCOMPATIBLE, MEASURED_DEVELOPMENT or MEASURED_VENUE')
    return persist(tools,task_id,'MISSION_STOP',{'basis':basis,**explanations,'scope':'CURRENT_RESEARCH_ALLOCATION_ONLY',
        'grounding':grounding,'interpretation':'Evidenced choice to stop/redesign this mission; no strategy-class rejection, alpha, portfolio or capital approval'})


def record_lesson(tools,args):
    owner(tools,'historian'); ident=args.get('decision_id'); decision=load(tools.store,ident,tools.cid)
    if decision['kind']!='MISSION_STOP' or decision['cycle']!=tools.cycle_id:
        raise ValueError('Mission lessons require this cycle\'s recorded stop decision')
    with tools.store.connect() as db:
        review=db.execute('SELECT status,result,campaign,agent FROM tasks WHERE id=?',(str(tools.cycle_id)+':falsifier:mission',)).fetchone()
    if not review or review['status']!='COMPLETE' or review['campaign']!=tools.cid or review['agent']!='falsifier':
        raise ValueError('Independent mission review must complete before its grounded lesson')
    payload={'decision_id':ident,'lesson':text(args,'lesson'),'next_experiment':text(args,'next_experiment'),
        'grounding':{'phase':decision['grounding']['phase'],'decision':decision,'independent_critique':json.loads(review['result']),
            'interpretation':'Trusted measurements remain authoritative; scientific stop and critique narratives are allocation judgments, not economic confirmation or learning-benefit proof'}}
    with tools.store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('SELECT payload FROM mission_lessons WHERE decision_id=?',(ident,)).fetchone()
        if old and json.loads(old['payload'])!=payload:
            raise ValueError('This decision already has an immutable reviewed lesson; conflicting text cannot replace it')
        if not old:
            db.execute('INSERT INTO mission_lessons VALUES(?,?,?)',(ident,json.dumps(payload,allow_nan=False),time.time()))
            db.execute('INSERT INTO memories(created,campaign,candidate,kind,payload) VALUES(?,?,?,?,?)',
                (time.time(),tools.cid,'','MISSION_LESSON',json.dumps(payload,allow_nan=False)))
    return {'recorded':True,'decision_id':ident,'already_recorded':bool(old),'interpretation':'Immutable reviewed mission lesson; no candidate submission or promotion'}
