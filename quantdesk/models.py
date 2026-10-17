"""Auditable fitted predictors: temporal inner selection and frozen JSON artifacts.

Generated code supplies a causal alpha feature; the trusted fitting service owns
labels, scaling, parameter selection and chronology. No pickle or arbitrary fit
code is executed. Development walk-forward results remain research evidence.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
from .store import digest
from .strategy import run_signal,run_feature_program
from .learners import tree_options,fit_tree,predict_tree,prediction_diagnostics

MODEL_VERSION = '1.1.0'
MODEL_FEATURES = ('qi_1', 'qi_5', 'qi_20', 'ofi_norm', 'micro_dev_bps',
                  'momentum_10s_bps', 'vol_30s_bps', 'spread_bps', 'count_skew_5',
                  'return_1s_bps')


def validate_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError('model must be a fitted-model recipe object')
    allowed = {'family', 'features', 'include_alpha', 'alphas', 'train_days', 'min_train_days',
               'inner_days', 'stride_seconds', 'max_train_rows', 'embargo_seconds',
               'prediction_scale_bps', 'min_edge_bps','custom_features','tree_search'}
    if set(recipe) - allowed:
        raise ValueError('Unknown fitted-model recipe fields')
    result = dict(family='ridge', features=list(MODEL_FEATURES[:8]), include_alpha=True,
                  alphas=[0.1, 1.0, 10.0], train_days=30, min_train_days=10,
                  inner_days=3, stride_seconds=10, max_train_rows=60000,
                  embargo_seconds=300, prediction_scale_bps=10.0, min_edge_bps=1.0)
    result.update(recipe)
    if not isinstance(result['include_alpha'], bool):
        raise ValueError('include_alpha must be a boolean')
    if result.get('custom_features',False) not in (True,False) or ('custom_features' in result and not isinstance(result['custom_features'],bool)):
        raise ValueError('custom_features must be a boolean')
    if result['family'] not in {'ridge', 'quadratic_ridge','xgboost'}:
        raise ValueError('Supported fitted families: ridge, quadratic_ridge, xgboost')
    if result['family']=='xgboost':
        result['tree_search']=tree_options(result.get('tree_search'))
    elif 'tree_search' in result:
        raise ValueError('tree_search applies only to the xgboost family')
    features = result['features']
    if not isinstance(features, list) or not 1 <= len(features) <= 8 or len(set(features)) != len(features) or any(f not in MODEL_FEATURES for f in features):
        raise ValueError('Choose 1..8 unique stationary, venue-compatible model features')
    for key, lo, hi in [('train_days', 6, 120), ('min_train_days', 6, 60),
                        ('inner_days', 1, 20), ('stride_seconds', 1, 300),
                        ('max_train_rows', 200, 200000), ('embargo_seconds', 0, 3600)]:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or not lo <= value <= hi:
            raise ValueError(f'{key} must be an integer in {lo}..{hi}')
        result[key] = int(value)
    if result['train_days'] < result['min_train_days']:
        raise ValueError(f"train_days={result['train_days']} must cover min_train_days={result['min_train_days']}")
    if result['inner_days'] > result['min_train_days'] - 3:
        raise ValueError(f"inner_days={result['inner_days']} exceeds min_train_days={result['min_train_days']} minus 3. Use inner_days <= {result['min_train_days']-3}, or min_train_days >= {result['inner_days']+3} with train_days >= min_train_days. At least three inner training dates must remain at the minimum window.")
    alphas = result['alphas']
    if not isinstance(alphas, list) or not 1 <= len(alphas) <= 8 or len(set(alphas)) != len(alphas) or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not np.isfinite(a) or not 0.0001 <= a <= 100000 for a in alphas):
        raise ValueError('Choose 1..8 unique finite ridge penalties in 0.0001..100000')
    result['alphas'] = sorted(float(a) for a in alphas)
    for key, lo, hi in [('prediction_scale_bps', 0.1, 1000), ('min_edge_bps', 0, 100)]:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be finite in {lo}..{hi}')
        result[key] = float(value)
    return result


def feature_names(alpha,recipe):
    return ['raw:'+f for f in recipe['features']]+(
        ['custom:'+name for name in alpha.columns] if isinstance(alpha,pl.DataFrame) else ['custom:alpha']
    )*int(recipe['include_alpha'])

def feature_matrix(frame, alpha, recipe):
    custom=alpha.to_numpy() if isinstance(alpha,pl.DataFrame) else alpha
    values = np.column_stack([frame[f].to_numpy() for f in recipe['features']] + ([custom] if recipe['include_alpha'] else []))
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite fitted-model features')
    return values


def labelled_rows(frame, spec, execution, recipe, before_ms=None):
    """Labels start at delayed entry; reject segment/gap/funding boundary windows."""
    ts = frame['ts_ms'].to_numpy()
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError('Fitting requires strictly increasing book timestamps')
    latency = execution['latency_ms']
    horizon = int(spec['holding_seconds'] * 1000)
    entry = np.searchsorted(ts, ts + latency, side='left')
    safe_entry = np.minimum(entry, len(ts) - 1)
    expiry = np.searchsorted(ts, ts[safe_entry] + horizon, side='left')
    # Two separate book arrivals, matching the broker on irregular timestamps.
    safe_expiry = np.minimum(expiry, len(ts) - 1)
    exit_ = np.searchsorted(ts, ts[safe_expiry] + latency, side='left')
    good = (entry < len(ts)) & (exit_ < len(ts))
    good &= expiry < len(ts)
    safe_exit = np.minimum(exit_, len(ts) - 1)
    segment = frame['segment'].to_numpy()
    good &= (segment == segment[safe_exit]) & (segment == segment[safe_entry])
    breaks = np.r_[0, np.cumsum(np.diff(ts) > execution['max_gap_ms'])]
    good &= breaks == breaks[safe_exit]
    good &= ts[safe_exit] // 3600000 == ts[safe_entry] // 3600000
    good &= 3600000 - ts % 3600000 > horizon + 2 * latency + execution['max_gap_ms']
    good &= ts[safe_entry] - ts - latency <= execution['max_gap_ms']
    good &= ts[safe_exit] - ts - 2 * latency - horizon <= execution['max_gap_ms']
    if before_ms is not None:
        good &= ts[safe_exit] < before_ms - recipe['embargo_seconds'] * 1000
    warm = np.zeros(len(ts), dtype=bool)
    starts = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1]
    ends = np.r_[starts[1:], len(ts)]
    for start, end in zip(starts, ends):
        warm[start + spec.get('warmup_rows', 300):end] = True
    good &= warm
    indices = []
    last = -np.inf
    for i in np.flatnonzero(good):
        if ts[i] - last >= recipe['stride_seconds'] * 1000:
            indices.append(i)
            last = ts[i]
    indices = np.asarray(indices, dtype=int)
    mid = frame['mid_price'].to_numpy()
    target = (mid[exit_[indices]] / mid[entry[indices]] - 1) * 10000
    return indices, target, ts[exit_[indices]]


def design(x, mean, scale, family):
    z = np.clip((x - mean) / scale, -10, 10)
    if family == 'quadratic_ridge':
        z = np.column_stack([z] + [z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i, z.shape[1])])
    return z


def fit_ridge(x, y, alpha, family):
    if len(x) < 100 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError('Need at least 100 finite training labels')
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    z = design(x, mean, scale, family)
    zmean = z.mean(axis=0)
    centered = z - zmean
    intercept = float(y.mean())
    # The penalty is normalized by observations for stable temporal windows.
    gram = centered.T @ centered / len(y)
    coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), centered.T @ (y - intercept) / len(y))
    return {'family': family, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'design_mean': zmean.tolist(), 'coefficients': coefficients.tolist(),
            'intercept': intercept, 'alpha': alpha, 'training_rows': len(y)}


def predict(x, fitted):
    if fitted['family']=='xgboost':
        return predict_tree(x,fitted)
    z = design(x, np.asarray(fitted['mean']), np.asarray(fitted['scale']), fitted['family'])
    return (z - np.asarray(fitted['design_mean'])) @ np.asarray(fitted['coefficients']) + fitted['intercept']


def fitted_signal(frame, alpha, artifact):
    recipe = artifact['recipe']
    if artifact['fitted'].get('feature_names') is not None and feature_names(alpha,recipe)!=artifact['fitted']['feature_names']:
        raise ValueError('Custom feature names/order differ from the frozen model')
    prediction = predict(feature_matrix(frame, alpha, recipe), artifact['fitted'])
    execution = artifact['execution']
    costs = 2 * (execution['taker_fee_bps'] + execution['slippage_bps']) + frame['spread_bps'].to_numpy()
    # Prediction is an estimated conditional midpoint move, not a guaranteed edge.
    signal = np.where(np.abs(prediction) > costs + recipe['min_edge_bps'],
                      np.tanh(prediction / recipe['prediction_scale_bps']), 0.0)
    if not np.all(np.isfinite(signal)):
        raise ValueError('Nonfinite fitted predictions')
    return signal


def prediction_frontier(frame,alpha,artifact,spec):
    """Separate predictive weakness, modeled costs and an extra decision margin."""
    recipe=artifact['recipe']; execution=artifact['execution']
    estimates=predict(feature_matrix(frame,alpha,recipe),artifact['fitted'])
    absolute=np.abs(estimates)
    costs=2*(execution['taker_fee_bps']+execution['slippage_bps'])+frame['spread_bps'].to_numpy()
    indices,target,_=labelled_rows(frame,spec,execution,recipe)
    eligible=(absolute>costs+recipe['min_edge_bps']) & (
        np.abs(np.tanh(estimates/recipe['prediction_scale_bps']))>spec['threshold'])
    ts=frame['ts_ms'].to_numpy(); segment=frame['segment'].to_numpy()
    starts=np.r_[0,np.flatnonzero(segment[1:]!=segment[:-1])+1]
    age=np.arange(len(frame))-np.maximum.accumulate(np.where(np.isin(np.arange(len(frame)),starts),np.arange(len(frame)),0))
    warm=age>=spec.get('warmup_rows',300)
    funding=(3600000-ts%3600000)>spec['holding_seconds']*1000+2*execution['latency_ms']+execution['max_gap_ms']
    spread=frame['spread_bps'].to_numpy()<=execution['max_spread_bps']
    outer=prediction_diagnostics(target,estimates[indices]) if len(target)>=50 else {'observations':len(target)}
    outer['interpretation']='Outer development prediction observations; not net trading or independent confirmation'
    return {'rows':len(frame),'extra_margin_bps':recipe['min_edge_bps'],
        'absolute_prediction_bps':dict(zip(('median','p90','p99','maximum'),map(float,np.quantile(absolute,[0.5,0.9,0.99,1])))),
        'modeled_cost_bps':{'minimum':float(costs.min()),'median':float(np.median(costs))},
        'rows_above_modeled_costs':int(np.sum(absolute>costs)),
        'rows_above_costs_plus_margin':int(np.sum(absolute>costs+recipe['min_edge_bps'])),
        'rows_above_margin_and_signal_threshold':int(np.sum(eligible)),
        'rows_also_passing_warmup_funding_spread':int(np.sum(eligible & warm & funding & spread)),
        'eligible_row_examples':[{'exchange_ts':int(ts[i]),'prediction_bps':float(estimates[i]),
            'modeled_cost_bps':float(costs[i]),'segment_age_rows':int(age[i]),
            'warmup_passed':bool(warm[i]),'funding_entry_passed':bool(funding[i]),
            'spread_passed':bool(spread[i])} for i in np.flatnonzero(eligible)[:20]],
        'outer_prediction_diagnostics':outer,
        'interpretation':'Outer DEVELOPMENT prediction and gate diagnostics, not trades or net alpha. The extra margin is added after costs; warmup, broker state, funding and feed gates can further prevent execution.'}


def label_feasibility(frame,spec,execution,recipe,before_ms=None):
    from .data import segment_coverage
    coverage=segment_coverage(frame)
    indices,_,_=labelled_rows(frame,spec,execution,recipe,before_ms)
    # Determine actual available history at viable label starts without the
    # requested warmup or thinning. This uses timestamps and segment boundaries,
    # never target magnitudes, and does not change the fit recipe.
    structural,_,_=labelled_rows(frame,dict(spec,warmup_rows=0),execution,dict(recipe,stride_seconds=0),before_ms)
    segments=frame['segment'].to_numpy()
    starts=np.r_[0,np.flatnonzero(segments[1:]!=segments[:-1])+1]
    age=np.arange(len(frame))-np.maximum.accumulate(np.where(np.isin(np.arange(len(frame)),starts),np.arange(len(frame)),0))
    maximum=int(age[structural].max()) if len(structural) else None
    reasons=[]
    if spec.get('warmup_rows',300)>=coverage['maximum_rows']:
        reasons.append('Warmup consumes every observed reconstruction segment')
    elif maximum is not None and spec.get('warmup_rows',300)>maximum:
        reasons.append('Warmup plus delayed holding window cannot fit inside an observed reconstruction segment')
    if not len(structural):
        reasons.append('No window survives segment, gap, funding and cutoff exclusions even without warmup')
    return {'status':'LABELS_AVAILABLE' if len(indices) else 'NO_USABLE_LABELS',
        'eligible_label_rows':len(indices),'warmup_rows':spec.get('warmup_rows',300),
        'holding_seconds':spec['holding_seconds'],'stride_seconds':recipe['stride_seconds'],
        'segment_coverage':coverage,'maximum_observed_feasible_warmup_rows':maximum,
        'reasons':reasons,'interpretation':'Observed data/recipe compatibility only; no forecast skill or economic evidence. Increasing training dates cannot repair an impossible window on these segments.'}


class ModelService:
    def __init__(self, store, catalog, settings):
        self.store, self.catalog, self.settings = store, catalog, settings

    def alpha(self, candidate, record, frame, rows):
        custom=candidate['spec']['model'].get('custom_features',False)
        path = self.store.home / 'signals' / (digest({'model_alpha': MODEL_VERSION, 'record': record,
            'rows': rows, 'code': candidate['code'],'custom_features':custom,
            'input_features':candidate['spec'].get('input_features')}) + ('.parquet' if custom else '.npy'))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            values = pl.read_parquet(path) if custom else np.load(path, allow_pickle=False)
            if len(values) == len(frame):
                return values
        runner=run_feature_program if custom else run_signal
        from .input_contract import program_input
        values = runner(candidate['code'], program_input(frame,candidate['spec']), self.settings.worker_timeout)
        if custom:
            values.write_parquet(path)
        else:
            np.save(path, values, allow_pickle=False)
        return values

    def fit(self, cid, candidate, records, cutoff_ms):
        self.store.check_research_control(cid)
        recipe = validate_recipe(candidate['spec']['model'])
        records = sorted(records, key=lambda r: r['date'])[-recipe['train_days']:]
        if len(records) < recipe['min_train_days']:
            raise ValueError(f"Need {recipe['min_train_days']} earlier development dates for model fitting")
        campaign = self.store.get_campaign(cid)
        plan = campaign['plan']
        allowed = {digest(r) for r in plan['development']}
        if any(digest(r) not in allowed for r in records):
            raise ValueError('Model fitting may use development dates only')
        execution = self.settings.execution_dict()
        key = digest({'version': MODEL_VERSION, 'code': candidate['code'], 'spec': candidate['spec'],
                      'records': records, 'rows': plan.get('max_rows_per_day'),
                      'execution': execution, 'cutoff_ms': int(cutoff_ms)})
        path = self.store.home / 'models' / (key + '.json')
        if path.exists():
            artifact = json.loads(path.read_text(encoding='utf-8'))
            verify_artifact(artifact)
            return artifact, path
        self.store.expose_dates(cid, records, 'development')
        options=recipe['tree_search'] if recipe['family']=='xgboost' else recipe['alphas']
        self.store.event(cid, 'model_service', 'EXPERIMENT_STARTED', {'kind': 'temporal_parameter_search',
            'plan_hash': plan['hash'], 'family':recipe['family'],'options':options,'trials':len(options)})
        prepared = []; feasibility=[]
        for record in records:
            from .control import ResearchStopped
            self.store.check_research_control(cid)
            frame, quality = self.catalog.load(record, plan.get('max_rows_per_day'))
            if any(quality.get(k, 0) for k in ('sequence_gaps', 'timestamp_gaps', 'crossed_books')):
                raise ValueError('Training archive has unresolved book quality failures')
            if int(frame['ts_ms'][-1]) >= cutoff_ms:
                raise ValueError('Training observations must precede the prediction date')
            indices, target, label_end = labelled_rows(frame, candidate['spec'], execution, recipe, cutoff_ms)
            feasibility.append(dict(label_feasibility(frame,candidate['spec'],execution,recipe,cutoff_ms),date=record['date']))
            # Bound each date equally; deterministic selection never examines label values.
            bound = max(1, recipe['max_train_rows'] // len(records))
            if len(indices) > bound:
                selected = np.unique(np.linspace(0, len(indices) - 1, bound, dtype=int))
                indices, target, label_end = indices[selected], target[selected], label_end[selected]
            prepared.append((record,indices,target,label_end,int(frame['ts_ms'][0])))
        split = len(prepared) - recipe['inner_days']
        inner_cutoff = min(d[4] for d in prepared[split:]) - recipe['embargo_seconds'] * 1000
        inner_rows=sum(int(np.sum(d[3]<inner_cutoff)) for d in prepared[:split])
        selection_rows=sum(len(d[2]) for d in prepared[split:])
        diagnostic={'status':'MODEL_DATA_FEASIBILITY_ONLY','candidate_id':candidate['id'],
            'inner_training_rows':inner_rows,'inner_selection_rows':selection_rows,'dates':feasibility,
            'prediction_cutoff_ms':int(cutoff_ms),'interpretation':'Label and segment coverage, not fitting or economic failure'}
        if selection_rows<50 or inner_rows<100:
            self.store.memory(cid,candidate['id'],'MODEL_DATA_FEASIBILITY',diagnostic)
            maxima=[d['segment_coverage']['maximum_rows'] for d in feasibility]
            why=sorted(set(reason for d in feasibility for reason in d['reasons']))
            raise ValueError(f'Insufficient temporal labels: inner training={inner_rows} (need 100), selection={selection_rows} (need 50); warmup={candidate["spec"].get("warmup_rows",300)} rows, maximum observed segment={max(maxima)} rows, holding={candidate["spec"]["holding_seconds"]}s. '+ '; '.join(why))
        datasets=[];schema=None
        for record,indices,target,label_end,first_ts in prepared:
            self.store.check_research_control(cid)
            frame,_=self.catalog.load(record,plan.get('max_rows_per_day'))
            values=self.alpha(candidate,record,frame,plan.get('max_rows_per_day')) if recipe['include_alpha'] else np.zeros(len(frame))
            current_schema=feature_names(values,recipe)
            if schema is not None and current_schema!=schema:
                raise ValueError('Custom feature names/order changed across training dates')
            schema=current_schema
            datasets.append((record,feature_matrix(frame,values,recipe)[indices],target,label_end,first_ts))
        inner_x = np.concatenate([d[1][d[3] < inner_cutoff] for d in datasets[:split]])
        inner_y = np.concatenate([d[2][d[3] < inner_cutoff] for d in datasets[:split]])
        test_x = np.concatenate([d[1] for d in datasets[split:]])
        test_y = np.concatenate([d[2] for d in datasets[split:]])
        search = []
        def fit_option(x,y,option):
            fit=fit_tree(x,y,option) if recipe['family']=='xgboost' else fit_ridge(x,y,option,recipe['family'])
            fit['feature_names']=schema
            return fit
        for option in options:
            self.store.check_research_control(cid)
            fit = fit_option(inner_x, inner_y, option)
            estimates = predict(test_x, fit)
            search.append({('parameters' if recipe['family']=='xgboost' else 'alpha'):option,
                **prediction_diagnostics(test_y,estimates)})
        # Stable ties prefer stronger ridge regularization or the earlier tree config.
        selected_record=min(search,key=lambda v:(v['mse_bps_squared'],-v.get('alpha',0)))
        selected=selected_record['parameters' if recipe['family']=='xgboost' else 'alpha']
        x = np.concatenate([d[1] for d in datasets])
        y = np.concatenate([d[2] for d in datasets])
        ends = np.concatenate([d[3] for d in datasets])
        if len(ends) == 0 or int(ends.max()) >= cutoff_ms - recipe['embargo_seconds'] * 1000:
            raise ValueError('Label cutoff or embargo violated')
        artifact = {'format': 'quantdesk-fitted-v1', 'version': MODEL_VERSION,
            'candidate_id': candidate['id'], 'code_hash': digest(candidate['code']),
            'spec_hash': digest(candidate['spec']), 'recipe': recipe, 'execution': execution,
            'fitted': fit_option(x, y, selected),
            'provenance': {'training_dates': [r['date'] for r in records], 'training_records_hash': digest(records),
                'prediction_cutoff_ms': int(cutoff_ms), 'max_label_end_ms': int(ends.max()),
                'inner_training_dates': [d[0]['date'] for d in datasets[:split]],
                'inner_selection_dates': [d[0]['date'] for d in datasets[split:]],
                'inner_training_rows': len(inner_y), 'inner_selection_rows': len(test_y),
                'inner_max_label_end_ms': int(max(d[3][d[3] < inner_cutoff].max() for d in datasets[:split] if np.any(d[3] < inner_cutoff))),
                'inner_prediction_cutoff_ms': int(inner_cutoff + recipe['embargo_seconds'] * 1000),
                'parameter_search': search, 'selected_inner_diagnostics':selected_record,
                'selection_objective': 'temporal prediction MSE; no profitability claim'}}
        artifact['artifact_hash'] = digest(artifact)
        verify_artifact(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.partial')
        partial.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
        partial.replace(path)
        return artifact, path


def verify_artifact(artifact):
    if artifact.get('format') != 'quantdesk-fitted-v1' or digest({k: v for k, v in artifact.items() if k != 'artifact_hash'}) != artifact.get('artifact_hash'):
        raise ValueError('Fitted model artifact integrity check failed')
    recipe = validate_recipe(artifact['recipe'])
    p = artifact['provenance']
    if p['max_label_end_ms'] >= p['prediction_cutoff_ms'] - recipe['embargo_seconds'] * 1000:
        raise ValueError('Fitted model label cutoff violated')
    if p['inner_max_label_end_ms'] >= p['inner_prediction_cutoff_ms'] - recipe['embargo_seconds'] * 1000:
        raise ValueError('Inner model selection label cutoff violated')
    if set(p['inner_training_dates']) & set(p['inner_selection_dates']):
        raise ValueError('Inner training and selection dates overlap')
    return artifact
