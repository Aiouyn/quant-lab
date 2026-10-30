"""One-shot holdout lock: forward-only windows, burned history, and strategy FAMILY keys.

Rules (owner-approved, cycle 1b):
- A holdout window is usable only if it starts strictly after HOLDOUT_BURNED_THROUGH (2026-10-29). Every holdout
  window that existed when this lock landed (BURNED) has been looked at and is permanently burned. New holdouts are
  forward-only: data that did not exist yet when the lock landed.
- A holdout shot is keyed by strategy FAMILY, not by code hash. A family is the strategy's structural skeleton
  (normalized AST: docstrings dropped, every literal and every local/argument name canonicalized, PARAMS values
  ignored) plus its DESK, plus any declared `FAMILY = '...'` name, plus the families of its lineage parents.
  Changing parameters, constants, names, comments or docstrings does not buy a new shot; neither does a near copy
  (token similarity >= FAMILY_SIMILARITY with an already-used skeleton on the same desk).
- Each family gets exactly one shot, ever. Shots are recorded in insert-only ledger tables (see ledger.py).
"""
from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
import re
from datetime import date

HOLDOUT_BURNED_THROUGH = date(2026, 10, 29)
FAMILY_SIMILARITY = 0.85

# Every holdout window in the repo when the lock landed (documentation + belt and braces; the date rule covers them).
BURNED = {
    'macro': [('2020-01-01', '2025-06-30')],
    'perps': [('2026-04-02', '2026-10-01')],
    'pred': [('2026-06-01', '2026-07-31')],
    # carry and l2 holdouts are computed from their own panels (chronological 15% tail); all of them end before the
    # lock date and are caught by the date rule.
}

_KEEP_NAMES = set(dir(builtins)) | {'np', 'pl', 'pd', 'math', 'numpy', 'polars', 'pandas', 'signal', 'fit',
                                    'PARAMS', 'DESK', 'FAMILY', 'self'}


class HoldoutLocked(PermissionError):
    """The requested holdout window is burned or not forward-only."""


class HoldoutAlreadyUsed(PermissionError):
    """This strategy family already had its one holdout shot."""


def _as_date(x) -> date:
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def window_bounds(window) -> tuple[date, date]:
    """[start, end] pairs (macro/perps/pred/carry) or a list of day strings (l2)."""
    if not window:
        raise HoldoutLocked('empty holdout window')
    ds = [_as_date(w) for w in window]
    return min(ds), max(ds)


def assert_holdout_open(desk: str, window) -> tuple[date, date]:
    a, b = window_bounds(window)
    if (str(a), str(b)) in {(x, y) for x, y in BURNED.get(desk, [])}:
        raise HoldoutLocked(f'{desk} holdout {a}..{b} is burned (pre-lock window, already looked at)')
    if a <= HOLDOUT_BURNED_THROUGH:
        raise HoldoutLocked(f'{desk} holdout {a}..{b} starts on/before {HOLDOUT_BURNED_THROUGH}: burned. New holdouts '
                            f'must be forward-only (first day after {HOLDOUT_BURNED_THROUGH}).')
    return a, b


def desk_holdout_window(settings, desk: str):
    """The desk's frozen holdout window, read through its store (which checks the splits manifest)."""
    if desk == 'macro':
        from .macro import MacroStore
        return MacroStore(settings).splits()['holdout']
    if desk == 'perps':
        from .perps import PerpsStore
        return PerpsStore(settings).splits()['holdout']
    if desk == 'pred':
        from .pred import PredStore
        return PredStore(settings).splits()['holdout']
    if desk == 'carry':
        from .carry import CarryStore
        return CarryStore(settings).splits()['holdout']
    from ..data.store import FeatureStore
    return FeatureStore(settings).splits()['holdout']


# ---------------------------------------------------------------------------------------------- family identity
class _Canon(ast.NodeTransformer):
    def __init__(self):
        self.names: dict[str, str] = {}

    def _n(self, name):
        if name in _KEEP_NAMES:
            return name
        return self.names.setdefault(name, f'v{len(self.names)}')

    def visit_Constant(self, node):
        v = node.value
        tag = 'num' if isinstance(v, (int, float, complex)) and not isinstance(v, bool) else type(v).__name__
        return ast.copy_location(ast.Constant(value=tag), node)

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._n(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self._n(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = self._n(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):          # keep library attributes (pl.col, np.sign): they are the structure
        self.generic_visit(node)
        return node


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _declared(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _drop_meta(tree):
    """DESK/FAMILY are identity, not structure; PARAMS values are already canonicalized as literals."""
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ('DESK', 'FAMILY') for t in n.targets))]
    return tree


def skeleton(code: str) -> str:
    tree = _drop_meta(_strip_docstrings(ast.parse(code)))
    tree = _Canon().visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def code_desk(code: str) -> str:
    return _declared(ast.parse(code), 'DESK') or 'l2'


def structural_family(code: str) -> str:
    return 'sk:' + hashlib.sha256(f'{code_desk(code)}|{skeleton(code)}'.encode()).hexdigest()[:24]


def family_keys(code: str, parent_codes=()) -> list[str]:
    """All keys that identify this strategy's family; a shot is refused if ANY of them was used."""
    keys = [structural_family(code)]
    declared = _declared(ast.parse(code), 'FAMILY')
    if declared:
        keys.append(f'decl:{code_desk(code)}:{declared.strip().lower()}')
    for pc in parent_codes:
        try:
            keys.extend(k for k in family_keys(pc) if k not in keys)
        except SyntaxError:
            continue
    return keys


def skeleton_tokens(code: str) -> list[str]:
    return re.findall(r'\w+', skeleton(code))


def similar(a_tokens, b_tokens) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    return difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False).ratio()
