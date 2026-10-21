"""Evaluation identity for one-shot holdout access.

identity = sha256 over
  * the normalized AST of the strategy (comments, whitespace, quoting, formatting and docstrings do not matter),
  * its dependency closure: the normalized AST of every `desk.*` module it imports, recursively,
  * the evaluation settings (job fields that change the result; ephemeral/reporting keys dropped),
  * the data version (content of every frozen splits.json the job could read).
A cosmetically edited resubmission has the same identity, so it cannot buy a second look at the holdout.
NOTE: no charter "R4" definition exists in the repo yet (searched main and every open branch); this module is that
definition until one lands, and should be reused by any other holdout guard.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

EPHEMERAL_KEYS = {'_home', '_store', 'values_dir', 'return_daily', 'stress', 'robustness', 'certify', 'factor_id',
                  'code'}
SPLIT_FILES = ('splits.json', 'macro/splits.json', 'perps/splits.json', 'pred/splits.json', 'carry/splits.json')


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def normalized_ast(code: str) -> str:
    return ast.dump(_strip_docstrings(ast.parse(code)), annotate_fields=True, include_attributes=False)


def _desk_imports(tree) -> set:
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names if a.name.split('.')[0] == 'desk'}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split('.')[0] == 'desk':
            mods.add(node.module)
            mods |= {f'{node.module}.{a.name}' for a in node.names}       # `from desk.lib import mod`
    return mods


def dependency_closure(code: str) -> dict:
    """{module: sha256(normalized AST)} for every desk.* module reachable from `code` (unresolvable names skipped)."""
    out, todo = {}, list(_desk_imports(ast.parse(code)))
    while todo:
        m = todo.pop()
        if m in out:
            continue
        try:
            spec = importlib.util.find_spec(m)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is None or not spec.origin or not spec.origin.endswith('.py'):
            continue
        src = Path(spec.origin).read_text(encoding='utf-8')
        out[m] = hashlib.sha256(normalized_ast(src).encode()).hexdigest()
        todo += list(_desk_imports(ast.parse(src)))
    return dict(sorted(out.items()))


def data_version(store) -> str:
    h = hashlib.sha256()
    for rel in SPLIT_FILES:
        p = Path(store) / rel
        if p.is_file() and not p.is_symlink():
            sp = json.loads(p.read_text())
            sp.pop('view', None)
            h.update(rel.encode() + json.dumps(sp, sort_keys=True).encode())
    return h.hexdigest()


def eval_settings(job: dict) -> dict:
    return {k: job[k] for k in sorted(job) if k not in EPHEMERAL_KEYS}


def evaluation_identity(job: dict, store) -> tuple[str, dict]:
    code = job.get('code') or ''
    comp = {
        'ast': hashlib.sha256(normalized_ast(code).encode()).hexdigest(),
        'deps': dependency_closure(code),
        'settings': eval_settings(job),
        'data_version': data_version(store),
    }
    return hashlib.sha256(json.dumps(comp, sort_keys=True, default=str).encode()).hexdigest(), comp
