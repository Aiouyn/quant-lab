"""Targeted fix hints appended to failed run_python results (the commonest agent mistakes, measured 2026-10-06:
32% of sandbox runs failed, mostly pandas-vs-polars API confusion and guessed file paths)."""
from __future__ import annotations

import re

RULES = [
    (r"No module named 'desk\.|cannot import name .* from 'desk\.", "do not import desk internals: everything is already loaded by "
     "`from desk.research import *` (macro_panel, perps_panel, pred_panel, *_backtest, event_study, xs_factor_report, "
     "walk_forward, example_strategy, example_factor, source)."),
    (r"has no attribute 'n_unique'.*Did you mean: 'nunique'", "that object is pandas, not polars: use .nunique() - or keep data in polars."),
    (r"has no attribute 'items'", "polars DataFrame has no .items(): iterate df.iter_rows(named=True) or use df.to_dicts()."),
    (r"has no attribute 'cummax'|has no attribute 'cumsum'", "polars: cum_max(), cum_sum() (with .over('symbol') per asset)."),
    (r"'Expr' object has no attribute 'corr'", "polars correlation: pl.corr('a', 'b') inside .agg()/.select()."),
    (r"'Expr' object has no attribute 'reciprocal'", "polars: 1 / pl.col('x')."),
    (r"'list' object has no attribute '(to_list|tolist)'", "that is already a Python list."),
    (r"'Series' object has no attribute 'over'", "factor(panel, params) must build an EXPRESSION on the panel: "
     "panel.select(<expr>.over('symbol').alias('v'))['v'].to_numpy() - see example_factor(desk)."),
    (r"cannot parse input of type 'str' into Polars data type", "polars dtypes are objects: pl.Float64, not 'float64'."),
    (r'unable to find column "(ret|adj_close|ts|val)"', "no such column: macro has date/symbol/adjclose (no ret/ts); compute returns "
     "with pl.col('adjclose').pct_change().over('symbol'); print(panel.columns) first."),
    (r"has no attribute 'nunique'", "polars: use .n_unique() (not .nunique())."),
    (r"has no attribute 'unique'\b.*Series|value_counts\(\)\.to_dict", "polars Series: .unique(), .value_counts() returns a DataFrame."),
    (r"'DataFrame' object has no attribute 'sort'\b", "a pandas DataFrame has no .sort: use .sort_values(...) - or stay in polars (panels are polars: df.sort(...))."),
    (r"has no attribute 'sort_values'", "polars: use df.sort('col') / df.sort(['a','b'], descending=...) (not sort_values)."),
    (r"has no attribute 'groupby'", "polars: use df.group_by(...).agg(...) (not groupby)."),
    (r"has no attribute '(iloc|loc|apply|reset_index|merge|rolling)'", "polars has no iloc/loc/apply/reset_index/merge: use row slicing df[i], filter(), "
     "with_columns(), join(); rolling stats are pl.col(x).rolling_mean(n).over('symbol')."),
    (r"has no attribute 'shift'.*groupby|over\(\)", "per-asset shifts in polars: pl.col('x').shift(1).over('symbol') (or 'coin' / 'market')."),
    (r"unexpected keyword argument 'columns'", "pl.scan_parquet has no columns=: use pl.scan_parquet(path).select([...]).collect() - "
     "or simply use macro_panel() / perps_panel() / pred_panel()."),
    (r"No such file or directory.*desk", "do not guess repo paths: use source('macro') / source('perps') / source('pred') / source('research') "
     "to read desk code, and example_strategy(desk) for a working template."),
    (r"ColumnNotFoundError", "check panel.columns first; every desk documents its columns in help: print(macro_panel().columns)."),
    (r"length .* != panel rows|signal length", "signal() must return exactly one weight per panel row, in the panel's row order "
     "(compute with with_columns/over on the panel itself; do not filter or re-sort before returning)."),
    (r"SplitViolation", "research sandboxes only see the DEV period; validation/holdout/forward are sealed for the truth machine."),
    (r"datatypes of join keys|SchemaError", "polars join/concat needs identical dtypes: cast first, e.g. pl.col('market').cast(pl.Int64)."),
    (r"killed:system_memory", "the MACHINE was short of RAM (not your code's fault): just rerun the same code in a minute."),
    (r"MemoryError|killed:memory", "too much data in memory: restrict symbols/coins/markets or columns, or aggregate first."),
    (r"killed:timeout", "the run hit the time limit: sample fewer days/markets, vectorize instead of Python loops."),
    (r"rank `method` must be one of", "polars rank has no 'pct': cross-sectional percentile = "
     "(pl.col('x').rank() / pl.col('x').count()).over('date')."),
    (r"cannot compare 'date/datetime/time' to a string|str\.strptime|str\.slice.*date", "the date column is already pl.Date: compare with "
     "datetime.date(2024, 1, 31) (import datetime), use pl.col('date').dt.month() / .dt.year() / .dt.truncate('1mo')."),
    (r"filter predicate must be of type `Boolean`, got `List", "is_in against a per-group list: compute a rank instead, e.g. "
     "pl.col('x').rank(descending=True).over('g') <= 5."),
    (r"DuplicateError: projections contained duplicate output name", "two expressions produce the same column name: add .alias('new_name')."),
    (r"has no attribute 'memory_usage'", "polars: df.estimated_size('mb')."),
    (r"run_python.*scripts/|No such file or directory: 'scripts", "each run_python is a fresh process: files you did not write in this "
     "workspace do not exist; recompute or save results explicitly (df.write_parquet('name.parquet') in the workspace)."),
    (r"No such file or directory: 'results", "results/ is not readable from the sandbox: save your own intermediate files in the workspace "
     "(the current directory) and reload them by bare filename."),
    (r"ImportError|ModuleNotFoundError","available: numpy, polars, scipy, sklearn, lightgbm, statsmodels + `from desk.research import *` "
     "(already imported). No internet access."),
]


def hints(stderr: str, status: str = '') -> list[str]:
    text = (stderr or '') + ' ' + (status or '')
    out = []
    for pat, hint in RULES:
        if re.search(pat, text, re.S) and hint not in out:
            out.append(hint)
    return out[:4]
