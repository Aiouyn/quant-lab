"""Look-ahead checks shared by the panel desks (perps, macro, pred, carry).

A signal is causal when the weight at time t depends only on rows at or before t. Three checks, each one closing a
way the others can be fooled:

1. prefix invariance at several cut points: drop every row after the cut; earlier weights must not move. Catches
   full-sample statistics and reads of later rows (shift(-k), rolling windows centred on t, ...).
2. future perturbation: keep the panel's length and dates but scramble every numeric value after a cut; earlier
   weights must not move. Catches code that behaves honestly only when it detects a truncated panel (for example
   "if the last date is before 2025, compute normally"), which a prefix test alone cannot see.
3. date-literal lint: calendar dates or epoch timestamps written into the code are how a strategy (or a model that
   memorised market history) can encode known future events. They are refused; use data columns instead.

Every call runs in the strategy jail (desk/lab/jail.py), so a strategy cannot remember the full panel between calls.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import polars as pl

DEFAULT_CUTS = (0.3, 0.5, 0.7, 0.85)
_DATE_STR = re.compile(r'(?<!\d)(19[89]\d|20\d\d)[-/.](0[1-9]|1[0-2])([-/.](0[1-9]|[12]\d|3[01]))?(?!\d)')
_MS_LO, _MS_HI = 631_152_000_000, 2_524_608_000_000          # 1990-01-01 .. 2050-01-01 in epoch milliseconds
_S_LO, _S_HI = 631_152_000, 2_524_608_000                    # same range in epoch seconds
_DATE_CALLS = {'datetime', 'date', 'Timestamp', 'to_datetime', 'datetime64', 'strptime', 'fromisoformat', 'lit_date'}


def date_literals(code: str) -> list[str]:
    """Calendar dates / epoch timestamps written into strategy code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, str) and _DATE_STR.search(v):
                found.append(repr(v)[:40])
            elif isinstance(v, int) and not isinstance(v, bool) and (_MS_LO <= v <= _MS_HI or _S_LO <= v <= _S_HI):
                found.append(str(v))
        elif isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if fn in _DATE_CALLS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, int) and 1990 <= node.args[0].value <= 2050:
                found.append(f'{fn}({node.args[0].value}, ...)')
    return sorted(set(found))


def _perturb_after(panel: pl.DataFrame, tcol: str, cut, keep_cols, seed: int) -> pl.DataFrame:
    """Same rows, same dates; every numeric value after `cut` replaced by a random draw of other future values."""
    future = panel[tcol].to_numpy() > cut
    idx = np.flatnonzero(future)
    if len(idx) < 2:
        return panel
    rng = np.random.default_rng(seed)
    cols = []
    for c in panel.columns:
        s = panel[c]
        if c in keep_cols or not s.dtype.is_numeric():
            cols.append(s)
            continue
        vals = s.to_numpy().copy()
        vals[idx] = vals[rng.permutation(idx)]
        cols.append(pl.Series(c, vals, dtype=s.dtype))
    return pl.DataFrame(cols)


def panel_causality(signal_fn, mod, panel: pl.DataFrame, tcol: str, noun: str = 'weights', cuts=DEFAULT_CUTS,
                    keep_cols=(), seed: int = 17, code: str | None = None) -> dict:
    """signal_fn(mod, panel) -> np.ndarray aligned to panel rows. The panel must be sorted by `tcol`."""
    issues = []
    if code:
        lits = date_literals(code)
        if lits:
            issues.append(f'date literals in code {lits[:5]}: calendar knowledge is not allowed in signals; '
                          'derive timing from data columns')
    full = signal_fn(mod, panel)
    t = panel[tcol].to_numpy()
    ut = np.unique(t)
    if len(ut) < 4:
        return {'ok': not issues, 'issues': issues, 'rows': len(panel), 'checks': 0}
    checks = 0
    for c in cuts:
        cut = ut[min(len(ut) - 2, int(len(ut) * c))]
        k = int(np.sum(t <= cut))
        pre = signal_fn(mod, panel.head(k))
        bad = int(np.sum(np.abs(pre - full[:k]) > 1e-6))
        checks += 1
        if bad:
            issues.append(f'prefix cut at {cut}: {bad} {noun} changed when later rows were removed (future data use)')
    cut = ut[len(ut) // 2]
    k = int(np.sum(t <= cut))
    pert = signal_fn(mod, _perturb_after(panel, tcol, cut, set(keep_cols) | {tcol}, seed))
    bad = int(np.sum(np.abs(pert[:k] - full[:k]) > 1e-6))
    checks += 1
    if bad:
        issues.append(f'future perturbation at {cut}: {bad} earlier {noun} changed when only later values were '
                      'scrambled (the signal reads the future, or behaves differently on a full panel)')
    return {'ok': not issues, 'issues': issues, 'rows': len(panel), 'checks': checks}
