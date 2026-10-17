"""Trusted learner registry. Agents choose bounded configurations, never fit code."""
import json
from functools import lru_cache
import numpy as np


def tree_options(options=None):
    options=options if options is not None else [
        {'max_depth':2,'min_child_weight':50,'eta':0.05,'reg_lambda':10,'rounds':80},
        {'max_depth':3,'min_child_weight':100,'eta':0.05,'reg_lambda':20,'rounds':80}]
    if not isinstance(options,list) or not 1<=len(options)<=4:
        raise ValueError('tree_search needs 1..4 bounded tree configurations')
    result=[]
    bounds={'max_depth':(1,4,True),'min_child_weight':(5,2000,False),
            'eta':(0.01,0.2,False),'reg_lambda':(0.01,10000,False),'rounds':(10,200,True)}
    for option in options:
        if not isinstance(option,dict) or set(option)-set(bounds):
            raise ValueError('Unsupported tree parameter; callbacks, custom objectives and external data are unavailable')
        normalized={'max_depth':2,'min_child_weight':50,'eta':0.05,'reg_lambda':10,'rounds':80}
        normalized.update(option)
        for key,(low,high,integer) in bounds.items():
            v=normalized[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not low<=v<=high or (integer and int(v)!=v):
                raise ValueError(f'{key} is outside the bounded tree contract')
            normalized[key]=int(v) if integer else float(v)
        if normalized in result:
            raise ValueError('Duplicate tree configurations add no new trial')
        result.append(normalized)
    return result


def fit_tree(x,y,parameters):
    import xgboost as xgb
    if len(x)<100 or len(x)!=len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Need at least 100 matching finite training rows for a fitted tree')
    params={k:v for k,v in parameters.items() if k!='rounds'}
    params.update(objective='reg:squarederror',tree_method='hist',max_bin=64,
                  nthread=2,seed=731,subsample=1.0,colsample_bytree=1.0,
                  base_score=float(np.mean(y)))
    booster=xgb.train(params,xgb.DMatrix(np.asarray(x,dtype=np.float32),label=y,nthread=2),
                      num_boost_round=parameters['rounds'])
    # Native model IO, not a memory snapshot or an executable pickle.
    document=booster.save_raw(raw_format='json').decode('utf-8')
    if len(document)>8_000_000:
        raise ValueError('Fitted tree artifact exceeds the portable model size bound')
    return {'family':'xgboost','parameters':parameters,'training_rows':len(y),
            'input_columns':x.shape[1],'library':'xgboost','library_version':xgb.__version__,
            'model_json':document}


@lru_cache(maxsize=8)
def _loaded_tree(document):
    import xgboost as xgb
    if not isinstance(document,str) or len(document)>8_000_000:
        raise ValueError('Invalid tree model document')
    # A model must have been produced by our trusted fitting service and its
    # enclosing artifact verified before inference. No external model imports.
    booster=xgb.Booster()
    booster.load_model(bytearray(document.encode('utf-8')))
    booster.set_param({'nthread':2})
    return booster


def predict_tree(x,fitted):
    if x.ndim!=2 or x.shape[1]!=fitted['input_columns'] or not np.all(np.isfinite(x)):
        raise ValueError('Tree input schema disagrees with the frozen fit')
    return _loaded_tree(fitted['model_json']).inplace_predict(np.asarray(x,dtype=np.float32))


def prediction_diagnostics(actual,predicted):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float)
    if not len(actual) or len(actual)!=len(predicted) or not np.all(np.isfinite(actual+predicted)):
        raise ValueError('Invalid temporal prediction diagnostics')
    zero=float(np.mean(actual**2)); mse=float(np.mean((predicted-actual)**2))
    correlation=float(np.corrcoef(actual,predicted)[0,1]) if actual.std()>1e-12 and predicted.std()>1e-12 else None
    return {'observations':len(actual),'mse_bps_squared':mse,'zero_prediction_mse_bps_squared':zero,
            'improvement_over_zero_mse':zero-mse,'pearson_ic':correlation,
            'predicted_abs_bps_quantiles':np.quantile(np.abs(predicted),[0.5,0.9,0.99,1.0]).tolist(),
            'actual_abs_bps_quantiles':np.quantile(np.abs(actual),[0.5,0.9,0.99,1.0]).tolist(),
            'interpretation':'Inner temporal diagnostic; not net trading or held-out performance'}
