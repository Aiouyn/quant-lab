import types

import numpy as np
import polars as pl
import pytest

from desk.lab.perps import PerpsCost, PerpsStore, perps_causality, run_perps, summarize_perps
from desk.lab.strategy import load_strategy

H = 3_600_000


def panel():
    rows = []
    for t, (pa, pb, fa, fb) in enumerate([(100, 50, 0.0, 0.0), (110, 50, 1e-4, -2e-4), (99, 55, 1e-4, -2e-4), (99, 55, 0.0, 0.0)]):
        rows.append({'ts': t * H, 'coin': 'A', 'px': pa, 'px_fill': pa, 'f': fa, 'isp': 10.0, 'vlm': 1e8})
        rows.append({'ts': t * H, 'coin': 'B', 'px': pb, 'px_fill': pb, 'f': fb, 'isp': 20.0, 'vlm': 1e8})
    return pl.DataFrame(rows).sort(['ts', 'coin'])


def test_engine_matches_hand_computation():
    p = panel()
    w = np.array([0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])       # long A / short B at t0, flat from t2
    c = PerpsCost(unit_notional=1.0)
    r = run_perps(p, w, c)
    cost_in = 0.5 * (4.5 + 5) / 1e4 + 0.5 * (4.5 + 10) / 1e4
    assert r['fees'][0] == pytest.approx(cost_in)
    # hour 1: A +10%, B 0% ; funding paid at t1: long A pays 1e-4*0.5, short B pays 2e-4*0.5 (f<0: shorts pay)
    assert r['price'][1] == pytest.approx(0.5 * 0.10)
    assert r['funding'][1] == pytest.approx(-(0.5 * 1e-4) - (-0.5 * -2e-4))
    # hour 1 rebalancing: A drifted 0.5 -> 0.55 (+10%), B unchanged -> trade 0.05 of A back to target
    assert r['fees'][1] == pytest.approx(0.05 * (4.5 + 5) / 1e4)
    # hour 2: A -10%, B +10% ; exit from the DRIFTED positions: A 0.5*0.9 = 0.45, B -0.5*1.1 = -0.55
    assert r['price'][2] == pytest.approx(0.5 * -0.10 + -0.5 * 0.10)
    assert r['fees'][2] == pytest.approx(0.45 * (4.5 + 5) / 1e4 + 0.55 * (4.5 + 10) / 1e4)
    assert r['pnl'].sum() == pytest.approx(r['price'].sum() + r['funding'].sum() - r['fees'].sum())


def test_gross_cap_and_gap_holding():
    p = panel().filter(~((pl.col('coin') == 'B') & (pl.col('ts') == 2 * H)))   # B missing at t2
    w = np.where(p['coin'].to_numpy() == 'A', 1.0, -1.0)
    r = run_perps(p, w, PerpsCost(unit_notional=1.0))
    assert np.abs(r['weights']).sum(1).max() <= 1.0 + 1e-12                    # gross capped at 1
    assert r['weights'][2, 1] == pytest.approx(-0.5)                          # B held through its gap


def test_causality_catches_future_use():
    leak = "DESK='perps'\nPARAMS={}\ndef signal(panel, params):\n    return np.sign((pl.col('px').shift(-1).over('coin') / pl.col('px') - 1).fill_null(0).alias('x').pipe(lambda e: panel.select(e)['x']).to_numpy())\n"
    clean = "DESK='perps'\nPARAMS={}\ndef signal(panel, params):\n    return np.sign(panel.select((pl.col('px') / pl.col('px').shift(1).over('coin') - 1).fill_null(0))['px'].to_numpy())\n"
    p = pl.concat([panel()] * 1).sort(['ts', 'coin'])
    assert not perps_causality(load_strategy(leak), p)['ok']
    assert perps_causality(load_strategy(clean), p)['ok']


def test_split_guard(tmp_path):
    st = PerpsStore(types.SimpleNamespace(store=tmp_path), allowed=('dev',))
    from desk.data.store import SplitViolation
    with pytest.raises(SplitViolation):
        st.range_ms('val')
    with pytest.raises(SplitViolation):
        st.range_ms('forward')


def test_summary_tail_stats():
    p = panel()
    r = run_perps(p, np.zeros(len(p)), PerpsCost(unit_notional=1.0))
    s = summarize_perps(r, PerpsCost(unit_notional=1.0))
    assert s['pnl'] == 0 and 'median_daily_bps' in s and 'mean_ex_top5_days_bps' in s


def test_maker_fills_only_on_trade_through():
    p = panel().with_columns(pl.lit(None, dtype=pl.Float64).alias('lo'), pl.lit(None, dtype=pl.Float64).alias('hi'))
    # hour 0: A trades down to 99.9 (through 100 by 10 bps) -> buy fills; B never trades above 50 -> sell does not fill
    p = p.with_columns(pl.when(pl.col('coin') == 'A').then(99.9).otherwise(50.0).alias('lo'),
                       pl.when(pl.col('coin') == 'A').then(100.0).otherwise(50.0).alias('hi'))
    w = np.where(p['coin'].to_numpy() == 'A', 0.5, -0.5)
    c = PerpsCost(unit_notional=1.0, execution='maker')
    r = run_perps(p, w, c)
    assert r['weights'][0, 0] == pytest.approx(0.5) and r['weights'][0, 1] == 0.0
    assert r['fees'][0] == pytest.approx(0.5 * 1.5 / 1e4)                 # maker fee, no spread


def test_engine_only_columns_hidden_from_strategies():
    from desk.lab.perps import perps_signal
    probe = "DESK='perps'\nPARAMS={}\ndef signal(panel, params):\n    assert 'lo' not in panel.columns and 'hi' not in panel.columns\n    return np.zeros(len(panel))\n"
    p = panel().with_columns(pl.lit(1.0).alias('lo'), pl.lit(1.0).alias('hi'))
    assert perps_signal(load_strategy(probe), p).sum() == 0
