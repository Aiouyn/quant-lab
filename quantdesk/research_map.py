"""Measured research coverage and resource accounting for the next allocation decision.

Code and parameter variety are explicit proxies, never labeled mechanism novelty
or evidence that organizational learning has improved performance.
"""
import ast
import json
from collections import Counter
from .store import digest


def research_map(store,cid):
    plan=store.get_campaign(cid)['plan']
    candidates=store.candidates(cid)
    with store.connect() as db:
        requests=db.execute('SELECT agent,actual,reserved,status FROM requests WHERE campaign=?',(cid,)).fetchall()
        memories=db.execute('SELECT candidate,kind,payload FROM memories WHERE campaign=? ORDER BY id',(cid,)).fetchall()
        events=db.execute("SELECT payload FROM events WHERE campaign=? AND kind='EXPERIMENT_STARTED'",(cid,)).fetchall()
        cycles=db.execute('SELECT status,COUNT(*) AS count FROM cycles WHERE campaign=? GROUP BY status',(cid,)).fetchall()
        decisions=db.execute('SELECT id,kind,payload FROM research_decisions WHERE campaign=? ORDER BY created',(cid,)).fetchall()
        reviewed=db.execute('SELECT decision_id,payload FROM mission_lessons').fetchall()
        operational=db.execute("SELECT kind,payload FROM events WHERE campaign=? AND kind IN ('CYCLE_FAILED','CYCLE_PAUSED','CYCLE_ABANDONED','REPEATED_TOOL_FAILURE') ORDER BY id DESC LIMIT 6",(cid,)).fetchall()
    roles={}
    for r in requests:
        role=roles.setdefault(r['agent'],{'calls':0,'known_actual_tokens':0,'charged_or_reserved_tokens':0,'failed_calls':0})
        role['calls']+=1
        role['known_actual_tokens']+=r['actual'] or 0
        role['charged_or_reserved_tokens']+=r['actual'] if r['actual'] is not None else r['reserved']
        role['failed_calls']+=int(r['status']=='FAILED')
    fingerprints=set(); recipes=set(); facets=Counter()
    for candidate in candidates:
        try:
            code=ast.dump(ast.parse(candidate['code']),include_attributes=False)
        except SyntaxError:
            code=candidate['code']
        fingerprints.add(digest(code))
        spec=candidate['spec']; family=spec.get('model',{}).get('family','formula')
        facets[(family,spec.get('mode'),spec.get('holding_seconds'))]+=1
        recipes.add(digest({'code':code,'spec':{k:v for k,v in spec.items() if k not in {'title','rationale','falsification','parent_ids'}}}))
    observed={}; lessons=[]; ungrounded_lessons=0; outer_dates=set(); full_dates=set(); screens=Counter()
    current_decisions={r['id']:r for r in decisions}
    mission_lessons=[]
    for record in reviewed:
        if record['decision_id'] not in current_decisions:
            continue
        payload=json.loads(record['payload']); grounding=payload.get('grounding',{})
        decision=json.loads(current_decisions[record['decision_id']]['payload'])
        if digest(decision)!=record['decision_id'] or grounding.get('decision')!=decision or not grounding.get('independent_critique'):
            continue
        mission_lessons.append({'decision_id':record['decision_id'],'kind':'REVIEWED_MISSION_STOP',
            'basis':decision['basis'],'scope':decision['scope'],'lesson':payload['lesson'],
            'next_experiment':payload['next_experiment'],
            'interpretation':'Allocation judgment grounded in development evidence; not strategy-class rejection or economic approval'})
    for record in memories:
        payload=json.loads(record['payload'])
        if record['kind']=='LESSON':
            grounding=payload.get('grounding') or {}
            if isinstance(grounding,dict) and any(grounding.get(k) for k in ('development_experiment','validation_statistics','validation_reasons')):
                lessons.append({'candidate_id':record['candidate'],'lesson':payload.get('lesson'),
                    'next_experiment':payload.get('next_experiment'),
                    'measured_status':grounding.get('candidate_status')})
            else:
                ungrounded_lessons+=1
        if record['kind']=='DEVELOPMENT_SCREEN':
            screens[payload['status']]+=1
        if record['kind'] in {'DEVELOPMENT_EXPERIMENT','WALK_FORWARD_EXPERIMENT','DEVELOPMENT_SCREEN'}:
            if record['kind']=='DEVELOPMENT_SCREEN':
                payload=payload.get('development',{})
            for day in payload.get('daily_results',[]):
                if not isinstance(day,dict) or not day.get('date'):
                    continue
                observed[(record['candidate'],day['date'])]=day
                outer_dates.add(day['date'])
                if day.get('duration_seconds',0)>=18*3600 and not payload.get('sampled',False):
                    full_dates.add(day['date'])
    for candidate in candidates:
        report=candidate.get('report') or {}
        for phase in ('base','stress'):
            for day in report.get(phase,{}).get('days',[]):
                # Confirmation coverage is summarized without exposing sealed metrics.
                if day.get('duration_seconds',0)>=18*3600 and not report[phase].get('sampled',False):
                    full_dates.add(day['date'])
    charged=sum(r['charged_or_reserved_tokens'] for r in roles.values())
    attempts=sum(max(1,int(json.loads(e['payload']).get('trials',1))) for e in events)
    from .venue_study import summaries
    venue_summaries=summaries(store)
    from .venue_replay import summaries as replay_summaries
    return {'campaign':cid,'frozen_dates':{p:len(plan.get(p,[])) for p in ('development','validation','holdout')},
        'sampled':plan.get('sampled',False),'cycles':{r['status']:r['count'] for r in cycles},
        'completed_strategy_cohorts':sum(r['count'] for r in cycles if r['status']=='COMPLETE'),
        'completed_mission_decisions':sum(r['count'] for r in cycles if r['status']=='DECISION_COMPLETE'),
        'scientific_decisions':dict(Counter(r['kind'] for r in decisions)),
        'recent_scientific_decisions':[{'decision_id':r['id'],'kind':r['kind'],
            'cycle':json.loads(r['payload']).get('cycle'),
            'public_design_or_reason':(json.loads(r['payload']).get('hypothesis') or json.loads(r['payload']).get('reason',''))[:2000],
            'reviewed_mission_lesson':r['id'] in {m['decision_id'] for m in mission_lessons},
            'interpretation':'Read the exact decision by ID; compatible design/stop does not establish code, alpha or approval'} for r in decisions[-6:]],
        'reviewed_mission_lessons':len(mission_lessons),'pending_mission_experiment_suggestions':mission_lessons[-6:],
        'available_venue_studies':venue_summaries,
        'available_venue_replays':replay_summaries(store,cid),
        'registered_artifacts':len(candidates),'candidate_statuses':dict(Counter(c['status'] for c in candidates)),
        'distinct_program_syntax':len(fingerprints),'distinct_program_and_recipe':len(recipes),
        'research_facets':[{'family':f,'mode':m,'holding_seconds':h,'artifacts':n} for (f,m,h),n in facets.items()],
        'experiment_and_parameter_attempts':attempts,
        'development_candidate_date_receipts':len(observed),'distinct_outer_development_dates':sorted(outer_dates),
        'distinct_full_day_research_dates':sorted(full_dates),
        'observed_zero_trade_candidate_dates':sum(d.get('trades',0)==0 for d in observed.values()),
        'api_usage_by_role':roles,'known_actual_tokens':sum(r['known_actual_tokens'] for r in roles.values()),
        'charged_or_reserved_tokens':charged,
        'candidate_date_receipts_per_1000_charged_tokens':len(observed)*1000/charged if charged else None,
        'grounded_lessons':len(lessons),'ungrounded_lesson_records':ungrounded_lessons,'pending_experiment_suggestions':lessons[-6:],
        'development_screen_statuses':dict(screens),
        'operational_cycle_evidence':[{'kind':r['kind'],**json.loads(r['payload'])} for r in operational],
        'interpretation':'Measured coverage and effort, including reference/ablation artifacts. Syntax and recipe variety do not establish economic novelty. Receipt volume does not establish research quality or learning gains. Held-out metrics are excluded.'}
