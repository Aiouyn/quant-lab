"""Allocate validated strategy sleeves using aligned net returns and shrinkage."""
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from .store import digest

def allocate(return_matrix,cap=0.35,correlation_limit=0.90):
    matrix=np.asarray(return_matrix,dtype=float)
    if matrix.ndim!=2 or matrix.shape[0]<5 or matrix.shape[1]<1 or not np.all(np.isfinite(matrix)):
        raise ValueError('Portfolio requires finite aligned daily return observations')
    if not 0<cap<=1:
        raise ValueError('Allocation cap must be in (0,1]')
    keep=[]; dropped=[]
    for i in range(matrix.shape[1]):
        if np.std(matrix[:,i])<1e-12:
            dropped.append({'index':i,'reason':'Zero return variance'})
            continue
        if any(abs(np.corrcoef(matrix[:,i],matrix[:,j])[0,1])>=correlation_limit for j in keep):
            dropped.append({'index':i,'reason':'Near-duplicate net return stream'})
        else:
            keep.append(i)
    if not keep:
        raise ValueError('No nondegenerate diversified strategy sleeves')
    selected=matrix[:,keep]
    covariance=np.atleast_2d(np.cov(selected,rowvar=False))
    shrink=0.25
    covariance=(1-shrink)*covariance+shrink*np.diag(np.diag(covariance))
    scale=max(float(np.max(np.diag(covariance))),1e-16)
    gross=min(1.0,len(keep)*cap)
    initial=np.full(len(keep),gross/len(keep))
    result=minimize(lambda w:float(w@(covariance/scale)@w),initial,method='SLSQP',bounds=[(0,cap)]*len(keep),constraints=[{'type':'eq','fun':lambda w:float(w.sum()-gross)}],options={'ftol':1e-12,'maxiter':1000})
    if not result.success:
        raise ValueError('Constrained portfolio optimizer failed: '+result.message)
    weights=np.zeros(matrix.shape[1]); weights[keep]=result.x
    return {'weights':weights.tolist(),'cash_weight':float(1-weights.sum()),'gross_allocation':float(weights.sum()),'shrinkage':shrink,'dropped':dropped,'covariance':covariance.tolist(),'allocation_method':'long-only minimum variance of validated sleeve returns; fixed cap; cash allowed'}

def build_portfolio(store,cid,settings):
    candidates=store.candidates(cid,status='PAPER_READY')
    if not candidates:
        raise ValueError('No PAPER_READY strategies. Complete validation and sealed holdout first.')
    from .paper import load_bundle
    from .forward import assess_forward
    eligible=[]; assessments=[]
    for candidate in candidates:
        manifest,_=load_bundle(store.home/'bundles'/candidate['id'])
        if manifest['id']!=candidate['id'] or manifest['code_hash']!=digest(candidate['code']) or manifest['spec_hash']!=digest(candidate['spec']):
            raise ValueError('Portfolio bundle differs from the registered candidate')
        report=assess_forward(store,manifest,settings)
        assessments.append(report)
        if report['eligible']:
            eligible.append((candidate,report))
    if not eligible:
        raise ValueError('No strategies passed the precommitted Hyperliquid forward allocation gate. Inspect forward-assess; short/exploratory sessions cannot qualify.')
    dates=sorted(set.intersection(*[set(d['date'] for d in report['days']) for _,report in eligible]))
    if len(dates)<max(5,settings.min_forward_days):
        raise ValueError('Insufficient aligned qualifying forward dates; precommit common future windows for the intended sleeves')
    candidates=[candidate for candidate,_ in eligible]
    matrix=np.array([[next(d['return_fraction'] for d in report['days'] if d['date']==date) for _,report in eligible] for date in dates])
    allocation=allocate(matrix,settings.max_strategy_weight)
    members=[{'candidate_id':c['id'],'bundle':str(store.home/'bundles'/c['id']),'weight':w} for c,w in zip(candidates,allocation['weights']) if w>1e-10]
    manifest={'format':'quantdesk-portfolio-v2','campaign':cid,'members':members,'allocation':allocation,
        'fitted_on':'Aligned net returns from precommitted individual Hyperliquid forward windows',
        'forward_assessments':assessments,'dates':dates,'status':'PAPER_ONLY','portfolio_holdout_tested':False,
        'fresh_portfolio_observations_after_ms':int(time.time()*1000),
        'execution':settings.execution_dict(),'limitation':'Individual historical and forward acceptance does not validate this newly fitted portfolio. Collect a fresh portfolio shadow record after formation; no live order route or capital approval.'}
    ident=digest(manifest)[:16]; directory=store.home/'portfolios'/ident
    manifest['manifest_hash']=digest(manifest)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return directory,manifest
