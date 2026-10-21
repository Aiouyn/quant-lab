"""Charter R4 evaluation identity (docs/GATE_CHARTER.md, validator/gate-charter b8d2cd0).

    identity = sha256(canonical_json({"ast": A, "deps": D, "settings": S, "data": V}))
    canonical_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

A  normalized AST: ast.parse; every docstring removed (first-statement Expr(Constant(str)) of module/class/function
   bodies); comments vanish in the parser; ast.dump(tree, include_attributes=False). Constants are kept.
D  sorted [module, sha256(A(module))] for every in-repo `desk.*` module imported transitively by the strategy, plus
   the evaluation engine modules (desk.lab.<desk>, desk.lab.stats, desk.lab.harness) and their closure.
S  every evaluation setting the caller passes (job fields that change the result, gate settings, cost overrides),
   sorted keys; ephemeral/reporting keys are dropped.
V  {"splits_sha256": sha256(<desk>/splits.json bytes), "store_build_id": provenance build id or None}.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

EPHEMERAL_KEYS = frozenset({'_home', '_store', 'values_dir', 'return_daily', 'code', 'factor_id'})
DESK_SPLITS = {'l2': 'splits.json', 'macro': 'macro/splits.json', 'perps': 'perps/splits.json',
               'pred': 'pred/splits.json', 'carry': 'carry/splits.json'}
ENGINES = {'l2': 'desk.lab.backtest', 'macro': 'desk.lab.macro', 'perps': 'desk.lab.perps', 'pred': 'desk.lab.pred',
           'carry': 'desk.lab.carry'}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def _sha(s) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


def normalized_ast(source: str) -> str:
    return ast.dump(_strip_docstrings(ast.parse(source)), include_attributes=False)


def _desk_imports(tree, package=None) -> set:
    """desk.* modules imported by `tree` (absolute, or relative to `package` for in-repo modules)."""
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names if a.name.split('.')[0] == 'desk'}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not package:
                    continue
                try:
                    base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                except ImportError:
                    continue
            else:
                base = node.module or ''
            if base.split('.')[0] != 'desk':
                continue
            mods.add(base)
            mods |= {f'{base}.{a.name}' for a in node.names}               # `from desk.lib.contrib import mod`
    return mods


def _module_source(name):
    """(source, package) of an in-repo module, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None, None
    if spec is None or not spec.origin or not spec.origin.endswith('.py'):
        return None, None
    pkg = name if spec.submodule_search_locations else name.rpartition('.')[0]
    return Path(spec.origin).read_text(encoding='utf-8'), pkg


def dependency_closure(source: str, roots=()) -> list:
    out, todo = {}, sorted(_desk_imports(ast.parse(source)) | set(roots))
    while todo:
        m = todo.pop()
        if m in out:
            continue
        src, pkg = _module_source(m)
        if src is None:
            continue
        out[m] = _sha(normalized_ast(src))
        todo += sorted(_desk_imports(ast.parse(src), pkg) - set(out))
    return [[k, out[k]] for k in sorted(out)]


def data_version(store, desk='l2', store_build_id=None) -> dict:
    f = Path(store) / DESK_SPLITS.get(desk, 'splits.json')
    sha = _sha(f.read_bytes()) if f.is_file() and not f.is_symlink() else None
    return {'splits_sha256': sha, 'store_build_id': store_build_id}


def strategy_desk(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'DESK' for t in node.targets) \
                and isinstance(node.value, ast.Constant):
            return str(node.value.value)
    return 'l2'


def evaluation_settings(job: dict, extra=None) -> dict:
    s = {k: job[k] for k in job if k not in EPHEMERAL_KEYS}
    if extra:
        s['_gates'] = extra
    return s


def evaluation_identity(source: str, settings: dict, store, desk=None, store_build_id=None):
    """(identity hex, components) per charter R4."""
    desk = desk or strategy_desk(source)
    comp = {
        'ast': _sha(normalized_ast(source)),
        'deps': dependency_closure(source, roots=(ENGINES.get(desk, 'desk.lab.backtest'), 'desk.lab.stats',
                                                  'desk.lab.harness')),
        'settings': settings,
        'data': data_version(store, desk, store_build_id),
    }
    return _sha(canonical_json(comp)), comp
