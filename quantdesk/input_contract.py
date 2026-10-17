"""Depth compatibility derived from immutable code and model feature inputs."""
import ast


def input_contract(code,spec):
    tree=ast.parse(code)
    references=set(spec['input_features']) if 'input_features' in spec else {n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    # Dynamic indexing/aliases can hide dependencies from literal inspection.
    # Without an enforced declared input list, fail conservatively at 20 levels.
    ambiguous='input_features' not in spec and any(
        (isinstance(n,ast.Subscript) and not isinstance(n.slice,ast.Constant)) or
        (isinstance(n,ast.Attribute) and n.attr in {'columns','get_column','select'})
        for n in ast.walk(tree))
    references.update(spec.get('model',{}).get('features',[]))
    minimum=20 if ambiguous or references & {'qi_20','depth_total'} else 5 if references & {'qi_5','bid_v5','ask_v5','count_skew_5'} else 1
    return {'instrument':'BTC','research_venue':'OKX','target_venue':'Hyperliquid',
        'minimum_book_levels':minimum,
        'transfer_status':'UNVERIFIED: adequate depth does not establish matching venue distributions or execution',
        'quantity_calibration':'Raw OKX contract quantities require explicit calibration before transfer; dimensionless ratios do not establish venue equivalence'}


def program_input(frame,spec):
    if 'input_features' not in spec:
        return frame
    columns=list(spec['input_features'])
    if 'segment' in frame.columns and 'segment' not in columns:
        columns.append('segment')
    return frame.select(columns)
