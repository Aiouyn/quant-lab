"""Restricted numerical strategy contract and reproducible artifact hashes."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import re
from pathlib import Path
import numpy as np
import polars as pl
from .data import FEATURES

VALIDATOR_VERSION='2.1.0'

NUMPY = {'tanh','sign','abs','clip','where','minimum','maximum','sqrt','log','log1p','exp','expm1','isfinite','nan_to_num','zeros','ones','full','array','asarray','arange','diff','cumsum','concatenate','mean','std','sum','median','percentile','pi','nan'}
METHODS = {'alias','clip','fill_null','fill_nan','shift','diff','pct_change','rolling_mean','rolling_std','rolling_sum','rolling_min','rolling_max','rolling_quantile','ewm_mean','abs','sqrt','log','log1p','exp','sign','cast','to_numpy','to_list','with_columns','select','get_column','is_null','is_nan','is_finite','then','otherwise','when','over','sum','mean','std','min','max','median','quantile','len','head','slice','clone','columns','height'}
NUMPY |= {'float64','float32','int64','int32','zeros_like','ones_like','full_like','column_stack','stack','vstack','hstack','quantile','nanmean','nanstd','nansum','var','nanvar','divide','multiply','add','subtract','power','sin','cos','sinh','cosh','logical_and','logical_or','logical_not','all','any','isnan','isinf','convolve','correlate','argsort','sort','round','linspace','cumulative_sum'}
METHODS |= {'astype','reshape','ravel','flatten','copy','shape','size','ndim','round','transpose','rolling_var','rolling_median','cum_sum','cum_max','cum_min','rename','to_series','is_between','pow','floor','ceil','var','ewm_std','ewm_var'}

def validate_spec(spec):
    if isinstance(spec,str):
        # Some OpenAI-compatible tool providers JSON-encode nested objects.
        # Decode the same submitted specification; never substitute a default.
        spec=json.loads(spec)
    if not isinstance(spec,dict):
        raise ValueError('Strategy specification must be a JSON object')
    required={'title','rationale','falsification','holding_seconds','threshold','mode'}
    if not required<=spec.keys():
        raise ValueError(f'Missing strategy fields: {sorted(required-spec.keys())}')
    if not all(isinstance(spec[k],str) and spec[k].strip() for k in ('title','rationale','falsification')):
        raise ValueError('Title, rationale, and falsification must be nonempty strings')
    if 'input_features' in spec:
        inputs=spec['input_features']
        if not isinstance(inputs,list) or not inputs or any(not isinstance(c,str) or c not in FEATURES for c in inputs) or len(set(inputs))!=len(inputs):
            raise ValueError('input_features must be unique available feature names')
    if spec['mode'] not in {'taker','maker'}:
        raise ValueError('Mode must be taker or maker')
    if not 1<=float(spec['holding_seconds'])<=300 or not 0.01<=float(spec['threshold'])<=1:
        raise ValueError('Holding horizon must be 1..300 seconds; threshold 0.01..1')
    warmup=int(spec.get('warmup_rows',300))
    if not 0<=warmup<=5000:
        raise ValueError('Warmup rows must be 0..5000')
    if 'model' in spec:
        from .models import validate_recipe
        if spec['mode'] != 'taker':
            raise ValueError('Fitted predictors currently require taker execution')
        spec = dict(spec, model=validate_recipe(spec['model']))
    return dict(spec,holding_seconds=float(spec['holding_seconds']),threshold=float(spec['threshold']),warmup_rows=warmup,parent_ids=spec.get('parent_ids',[]))

def validate_code(code):
    if not isinstance(code,str):
        raise ValueError('Numerical code must be a Python string')
    tree=ast.parse(code)
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    main=[n for n in definitions if n.name=='compute_signal']
    if len(main)!=1 or len({n.name for n in definitions})!=len(definitions):
        raise ValueError('Define exactly one entry point compute_signal(df); numerical helpers may have unique names')
    function=main[0]
    if len(function.args.args)!=1 or function.args.args[0].arg!='df' or function.decorator_list:
        raise ValueError('Function must take only df, without decorators')
    if any(n.name.startswith('_') or n.decorator_list or n.args.defaults or n.args.kw_defaults or n.args.vararg or n.args.kwarg for n in definitions):
        raise ValueError('Numerical helpers must have public names, simple arguments, and no decorators or defaults')
    if len(code)>200000 or sum(1 for _ in ast.walk(tree))>20000:
        raise ValueError('Numerical implementation exceeds code size limit')
    for item in tree.body:
        if not isinstance(item,(ast.Import,ast.FunctionDef)) and not (isinstance(item,ast.Expr) and isinstance(item.value,ast.Constant) and isinstance(item.value.value,str)):
            raise ValueError('Top level may contain imports, a docstring, and compute_signal only')
    allowed_names={'pl','np','math','df','len','float','int','min','max','abs','range','True','False','None'}|{n.name for n in definitions}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for item in node.names:
                expected={'numpy':'np','polars':'pl','math':'math'}
                if item.name not in expected or (item.asname or item.name)!=expected[item.name]:
                    raise ValueError('Allowed imports: numpy as np, polars as pl, math')
        if isinstance(node,ast.FunctionDef) and node not in definitions:
            raise ValueError('Define helpers at module level')
        if isinstance(node,ast.For) and not (isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='range'):
            raise ValueError('Numerical loops may iterate only over finite range(...)')
        if isinstance(node,(ast.ImportFrom,ast.While,ast.AsyncFor,ast.AsyncFunctionDef,ast.With,ast.AsyncWith,ast.Try,ast.Global,ast.Nonlocal,ast.ClassDef,ast.Lambda,ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp,ast.Delete,ast.Raise)):
            raise ValueError(f'Disallowed numerical strategy syntax: {type(node).__name__}')
        if isinstance(node,ast.Name) and node.id.startswith('_'):
            raise ValueError('Private names are disallowed')
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
            allowed_names.add(node.id)
        if isinstance(node,ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError('Private attribute access is disallowed')
            if isinstance(node.value,ast.Name) and node.value.id=='np':
                if node.attr not in NUMPY:
                    raise ValueError(f'Unsupported NumPy operation: {node.attr}')
            elif isinstance(node.value,ast.Name) and node.value.id=='pl':
                if node.attr not in {'Series','DataFrame','col','lit','when','Float64','Int64','Float32'}:
                    raise ValueError(f'Unsupported Polars operation: {node.attr}')
            elif isinstance(node.value,ast.Name) and node.value.id=='math':
                if node.attr not in {'tanh','sqrt','log','log1p','exp','pi','isfinite'}:
                    raise ValueError('Unsupported math operation')
            elif node.attr not in METHODS:
                raise ValueError(f'Unsupported numerical method: {node.attr}')
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='df' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
            if node.slice.value not in FEATURES:
                raise ValueError(f'Feature is unavailable: {node.slice.value}')
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='shift':
                offset=node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg=='n'),None)
                if isinstance(offset,ast.UnaryOp) and isinstance(offset.op,ast.USub):
                    raise ValueError('Causality failure: negative shifts expose future observations')
            if node.func.attr in {'fill_null','fill_nan'} and any(kw.arg=='strategy' and isinstance(kw.value,ast.Constant) and kw.value.value=='backward' for kw in node.keywords):
                raise ValueError('Causality failure: backward filling exposes future observations')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id not in allowed_names|{'compute_signal'}:
            raise ValueError(f'Unsupported function: {node.func.id}')
    return tree

def compile_signal(code):
    validate_code(code)
    import math
    safe={'len':len,'float':float,'int':int,'min':min,'max':max,'abs':abs,'range':range}
    # Imports were parsed and checked. Bind modules without exposing __import__.
    tree=ast.parse(code)
    tree.body=[n for n in tree.body if not isinstance(n,ast.Import)]
    scope={'__builtins__':safe,'np':np,'pl':pl,'math':math}
    exec(compile(tree,'<numerical_strategy>','exec'),scope)
    return scope['compute_signal']

def signal_values(fn,frame):
    before=frame.hash_rows().to_numpy().copy()
    output=fn(frame)
    if not isinstance(output,pl.Series) or len(output)!=len(frame):
        raise ValueError('Return a Polars Series with exactly one float per input row')
    values=output.cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(values)) or np.any(np.abs(values)>1.000001):
        raise ValueError('Signals must be finite and bounded in [-1,1]; fill warmup with zero')
    if not np.array_equal(before,frame.hash_rows().to_numpy()):
        raise ValueError('Input mutation detected')
    return values

def feature_values(fn,frame):
    before=frame.hash_rows().to_numpy().copy()
    output=fn(frame)
    if not isinstance(output,pl.DataFrame) or len(output)!=len(frame) or not 1<=output.width<=8:
        raise ValueError('Feature program must return a Polars DataFrame with 1..8 columns and one row per input')
    if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}',name) for name in output.columns):
        raise ValueError('Feature names must be short public identifiers')
    if any(dtype not in (pl.Float32,pl.Float64,pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.UInt8,pl.UInt16,pl.UInt32,pl.UInt64) for dtype in output.dtypes):
        raise ValueError('Feature program columns must be numerical')
    output=output.cast({name:pl.Float64 for name in output.columns})
    values=output.to_numpy()
    if not np.all(np.isfinite(values)) or np.any(np.abs(values)>1.000001):
        raise ValueError('Custom features must be finite and bounded in [-1,1]; fill warmup with zero')
    if not np.array_equal(before,frame.hash_rows().to_numpy()):
        raise ValueError('Input mutation detected')
    return output

def evaluate_in_process(code,frame,check_causality=True,custom_features=False):
    frame=frame.select([c for c in frame.columns if c in FEATURES])
    if 'segment' in frame.columns and frame['segment'].n_unique()>1:
        # All strategy state resets with reconstruction state, in both replay
        # and paper execution. Causal rolling windows cannot span a broken feed.
        pieces=[evaluate_in_process(code,piece,check_causality,custom_features)
            for piece in frame.partition_by('segment',maintain_order=True)]
        if custom_features and any(piece.columns!=pieces[0].columns for piece in pieces):
            raise ValueError('Feature schema must remain identical across reconstruction segments')
        return pl.concat(pieces) if custom_features else np.concatenate(pieces)
    fn=compile_signal(code)
    extract=feature_values if custom_features else signal_values
    values=extract(fn,frame.clone())
    if check_causality:
        cuts={max(1,len(frame)//4),max(1,len(frame)//2),max(1,3*len(frame)//4),max(1,len(frame)-1)}
        cuts.update(c for c in (1,2,4,8,16,32,64,128,256,512,1024) if c<len(frame))
        cuts.update(int(c) for c in np.random.default_rng(713).integers(1,max(2,len(frame)),size=8))
        for cut in sorted(cuts):
            prefix=extract(fn,frame.head(cut).clone())
            if custom_features and prefix.columns!=values.columns:
                raise ValueError('Causality failure: feature schema changed with future rows')
            full_values=values.to_numpy() if custom_features else values
            prefix_values=prefix.to_numpy() if custom_features else prefix
            if not np.allclose(prefix_values,full_values[:cut],rtol=1e-8,atol=1e-8):
                raise ValueError(f'Causality failure: future rows changed historical signals at prefix {cut}')
    return values

def run_signal(code,frame,timeout=30,check_causality=True):
    return run_program(code,frame,timeout,check_causality,False)

def run_feature_program(code,frame,timeout=30,check_causality=True):
    return run_program(code,frame,timeout,check_causality,True)

def run_program(code,frame,timeout,check_causality,custom_features):
    validate_code(code)
    with tempfile.TemporaryDirectory(prefix='quantdesk_worker_') as directory:
        directory=Path(directory)
        frame.select([c for c in frame.columns if c in FEATURES]).write_parquet(directory/'input.parquet')
        (directory/'strategy.py').write_text(code,encoding='utf-8')
        # No provider or wallet credentials in the child environment.
        env={k:v for k,v in os.environ.items() if not any(word in k.upper() for word in ('KEY','TOKEN','SECRET','PASSWORD'))}
        env.update({'PYTHONPATH':str(Path(__file__).resolve().parent.parent),'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        args=[sys.executable,'-m','quantdesk.worker',str(directory),'1' if check_causality else '0','features' if custom_features else 'signal']
        try:
            result=subprocess.run(args,cwd=directory,env=env,capture_output=True,text=True,timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError('Strategy exceeded execution time limit') from None
        if result.returncode:
            error=(directory/'error.json')
            raise ValueError(json.loads(error.read_text())['error'] if error.exists() else 'Numerical worker failed')
        return pl.read_parquet(directory/'features.parquet') if custom_features else np.load(directory/'signal.npy',allow_pickle=False)
