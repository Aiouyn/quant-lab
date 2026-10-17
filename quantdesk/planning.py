"""Measured feasibility supplied before agents allocate research work."""
from .models import label_feasibility,validate_recipe
from .data import segment_coverage
from .store import digest


def planning_contract(store,catalog,cid,settings):
    plan=store.get_campaign(cid)['plan']; records=plan['development']
    prepared=[i for i,r in enumerate(records) if getattr(catalog,'cached',lambda *a:False)(r,plan.get('max_rows_per_day'))]
    selected=list(dict.fromkeys([prepared[0],prepared[len(prepared)//2],prepared[-1]])) if prepared else [0]
    profiles=[]
    for index in selected:
        store.check_research_control(cid)
        record=records[index]
        rows=plan.get('max_rows_per_day') if prepared else min(plan.get('max_rows_per_day') or 3000,3000)
        store.expose_dates(cid,[record],'development')
        frame,quality=catalog.load(record,rows)
        bounds={}
        for horizon in (10,30,60,120,300):
            spec={'holding_seconds':horizon,'warmup_rows':0}
            report=label_feasibility(frame,spec,settings.execution_dict(),validate_recipe({'stride_seconds':300}))
            bounds[str(horizon)]=report['maximum_observed_feasible_warmup_rows']
        profiles.append({'development_index':index,'date':record['date'],'rows':len(frame),
            'bounded_rows_requested':rows,'prepared_frame_used':bool(prepared),
            'segment_coverage':segment_coverage(frame),'quality':quality,
            'maximum_observed_feasible_warmup_rows_by_holding_seconds':bounds})
    from .venue_study import summaries
    from .mission_review import current_tool_contract
    result={'version':1,'holding_seconds_range':[1,300],
        'current_tool_constraints':current_tool_contract(),
        'target_venue_observations':summaries(store),
        'model_window_rule':'train_days >= min_train_days >= inner_days + 3',
        'recipe_defaults':validate_recipe({}),'profiles':profiles,
        'interpretation':'Observed history/label feasibility on availability-selected development frames. No forecast, outcome selection, PnL or promotion. Bounds describe these frames, not all future data. Numerical state resets at segments; revise an incompatible mechanism rather than shrinking warmup while claiming mature long history.'}
    result['hash']=digest(result)
    store.event(cid,'planning_service','DATA_CONTRACT',result)
    return result
