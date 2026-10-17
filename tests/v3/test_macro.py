import types
from datetime import date

import numpy as np
import polars as pl
import pytest

from desk.lab.macro import MacroCost, MacroStore, macro_causality, run_macro, summarize_macro
from desk.lab.strategy import load_strategy


def panel():
    rows = []
    px = {'AAA': [100, 100, 110, 99, 99], 'BBB': [50, 50, 50, 55, 55], '^IDX': [10, 11, 12, 13, 14]}
    for i in range(5):
        d = date(2010, 1, 4 + i)
        for s, v in px.items():
            rows.append({'date': d, 'symbol': s, 'adjclose': float(v[i]), 'close': float(v[i]), 'cls': 'index' if s[0] == '^' else 'equity_us',
                         'tradable': 0 if s[0] == '^' else 1, 'cost_bps': 0.0 if s[0] == '^' else 2.0})
    return pl.DataFrame(rows).sort(['date', 'symbol'])


def test_one_day_delay_and_hand_pnl():
    p = panel()
    w = np.where(p['symbol'].to_numpy() == 'AAA', 1.0, 0.0)          # long AAA decided every day from day 0
    c = MacroCost(unit_notional=1.0, short_borrow=0.0)
    r = run_macro(p, w, c)
    # decided at close d0, executed at close d1 (100) -> first return d1->d2 = +10%, then -10%
    assert r['weights'][0, 0] == 0 and r['weights'][1, 0] == 1.0
    assert r['price'][2] == pytest.approx(0.10) and r['price'][3] == pytest.approx(-0.10)
    assert r['fees'][1] == pytest.approx(2e-4)


def test_feature_series_never_traded_and_borrow():
    p = panel()
    w = np.where(p['symbol'].to_numpy() == '^IDX', 1.0, np.where(p['symbol'].to_numpy() == 'BBB', -1.0, 0.0))
    r = run_macro(p, w, MacroCost(unit_notional=1.0, short_borrow=0.0252))
    assert '^IDX' not in r['symbols']
    assert r['borrow'][2] == pytest.approx(1.0 * 0.0252 / 252)
    assert r['price'][3] == pytest.approx(-0.10)                         # short BBB loses when it rises 10%


def test_causality_and_split_guard(tmp_path):
    leak = "DESK='macro'\nPARAMS={}\ndef signal(panel, params):\n    return panel.select((pl.col('adjclose').shift(-1).over('symbol') / pl.col('adjclose') - 1).sign().fill_null(0).alias('w'))['w'].to_numpy()\n"
    clean = "DESK='macro'\nPARAMS={}\ndef signal(panel, params):\n    return panel.select((pl.col('adjclose') / pl.col('adjclose').shift(1).over('symbol') - 1).sign().fill_null(0).alias('w'))['w'].to_numpy()\n"
    assert not macro_causality(load_strategy(leak), panel())['ok']
    assert macro_causality(load_strategy(clean), panel())['ok']
    st = MacroStore(types.SimpleNamespace(store=tmp_path), allowed=('dev',))
    from desk.data.store import SplitViolation
    for sp in ('val', 'holdout', 'forward'):
        with pytest.raises(SplitViolation):
            st.range_dates(sp)


def test_summary_annualizes_on_252():
    p = panel()
    s = summarize_macro(run_macro(p, np.zeros(len(p)), MacroCost(unit_notional=1.0)), MacroCost(unit_notional=1.0))
    assert s['pnl'] == 0 and 'ann_return' in s and 'mean_ex_top5_days_bps' in s
