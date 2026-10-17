import pytest

from desk.company.hints import hints
from desk.lab.strategy import load_strategy
from desk.research_tools import example_strategy, source


@pytest.mark.parametrize('desk', ['macro', 'perps', 'pred'])
def test_example_strategies_load(desk):
    mod = load_strategy(example_strategy(desk))
    assert mod.DESK == desk and callable(mod.signal)


def test_source_helper():
    assert 'def run_macro' in source('macro') and 'def perps_panel' in source('research')
    with pytest.raises(KeyError):
        source('nope')


def test_hints_match_common_errors():
    assert any('n_unique' in h for h in hints("AttributeError: 'Series' object has no attribute 'nunique'"))
    assert any('source(' in h for h in hints("FileNotFoundError: No such file or directory: 'C:\\x\\desk\\lab\\m.py'"))
    assert any('scan_parquet' in h for h in hints("TypeError: scan_parquet() got an unexpected keyword argument 'columns'"))


@pytest.mark.parametrize('desk', ['macro', 'perps', 'pred'])
def test_example_factors_evaluate(desk):
    import types
    import numpy as np
    from desk.lab.panel_factor import load_factor, values
    from desk.research_tools import example_factor
    import polars as pl
    mod = load_factor(example_factor(desk))
    k = {'macro': 'symbol', 'perps': 'coin', 'pred': 'market'}[desk]
    t = {'macro': 'date', 'perps': 'ts', 'pred': 'ts'}[desk]
    n = 300
    panel = pl.DataFrame({t: list(range(n)) * 2, k: ['A'] * n + ['B'] * n, 'adjclose': np.linspace(1, 2, 2 * n),
                          'f': np.linspace(0, 1e-4, 2 * n), 'p': np.linspace(0.1, 0.9, 2 * n)}).sort([t, k])
    v = values(mod, panel)
    assert len(v) == len(panel)


def test_new_hints():
    assert any('do not import desk internals' in h for h in hints("ModuleNotFoundError: No module named 'desk.settings'"))
    assert any('cum_max' in h for h in hints("AttributeError: 'Expr' object has no attribute 'cummax'. Did you mean: 'cum_max'?"))


def test_sandbox_hoists_future_imports(tmp_path):
    import asyncio
    from desk.config import SETTINGS
    from desk.lab.sandbox import run_python
    r = asyncio.run(run_python('from __future__ import annotations\nprint("hi")', tmp_path, SETTINGS, timeout_s=120))
    assert r['status'] == 'ok' and 'hi' in r['stdout']
