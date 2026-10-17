import types

import numpy as np
import polars as pl
import pytest

from desk.lab.pred import PredCost, PredStore, pred_signal, run_pred, summarize_pred
from desk.lab.strategy import load_strategy

H = 3_600_000


def panel():
    rows = []
    for i, p in enumerate([0.40, 0.40, 0.50]):          # market A resolves YES at hour 3
        rows.append({'ts': i * H, 'market': 'A', 'p': p, 'end_ms': 3 * H, 'outcome': 1, 'category': 'x', 'volume': 1e5})
    for i, p in enumerate([0.80, 0.80, 0.70]):          # market B resolves NO at hour 3
        rows.append({'ts': i * H, 'market': 'B', 'p': p, 'end_ms': 3 * H, 'outcome': 0, 'category': 'y', 'volume': 1e5})
    return pl.DataFrame(rows).sort(['ts', 'market'])


def test_yes_to_resolution_pays_one_over_p():
    p = panel()
    w = np.where(p['market'].to_numpy() == 'A', 0.5, 0.0)          # decided every hour, executed one row later
    r = run_pred(p, w, PredCost(unit_notional=1.0, half_spread=0.0))
    a = r.filter(pl.col('market') == 'A')
    # share-holding: 0.5 of capital buys YES at 0.40 (hour 1) and is HELD to resolution -> 0.5 * (1/0.4 - 1)
    assert float(a['price_pnl'].sum()) == pytest.approx(0.5 * (1 / 0.4 - 1))


def test_no_side_and_spread_cost():
    p = panel()
    w = np.where(p['market'].to_numpy() == 'B', -0.5, 0.0)          # hold NO on B (YES 0.80 -> NO costs 0.20)
    r = run_pred(p, w, PredCost(unit_notional=1.0, half_spread=0.01))
    b = r.filter(pl.col('market') == 'B')
    assert float(b['price_pnl'].sum()) == pytest.approx(0.5 * (1 / 0.2 - 1))      # NO bought at 0.20, held, pays 1
    assert float(b['cost'].sum()) == pytest.approx(0.5 * 0.01 / 0.2)                # one trade: the entry


def test_outcome_and_volume_hidden_and_split_guard(tmp_path):
    probe = "DESK='pred'\nPARAMS={}\ndef signal(panel, params):\n    assert 'outcome' not in panel.columns and 'volume' not in panel.columns\n    assert 'hours_to_end' in panel.columns\n    return np.zeros(len(panel))\n"
    assert pred_signal(load_strategy(probe), panel()).sum() == 0
    st = PredStore(types.SimpleNamespace(store=tmp_path), allowed=('dev',))
    from desk.data.store import SplitViolation
    with pytest.raises(SplitViolation):
        st.range_ms('val')


def test_summary_runs():
    p = panel()
    s = summarize_pred(run_pred(p, np.zeros(len(p)), PredCost(unit_notional=1.0)), PredCost(unit_notional=1.0))
    assert s['pnl'] == 0 and 'mean_ex_top5_days_bps' in s


def test_target_change_retrades_only_the_difference():
    p = panel()
    w = np.where(p['market'].to_numpy() == 'A', np.where(p['ts'].to_numpy() == 0, 0.5, 0.25), 0.0)
    r = run_pred(p, w, PredCost(unit_notional=1.0, half_spread=0.01)).filter(pl.col('market') == 'A').sort('ts')
    # hour 1: buy 0.5 at 0.40 (cost 0.5*0.01/0.4); hour 2: target 0.25 but 0.5 drifted to 0.625 at 0.50 -> sell 0.375
    assert float(r['cost'].sum()) == pytest.approx(0.5 * 0.01 / 0.4 + 0.375 * 0.01 / 0.5)


def test_decided_market_books_no_phantom_gains():
    # YES drifts to 0.0001 and sits there for 50 hours before resolving NO: holding NO bought at 0.90 must earn
    # exactly 1/0.90 - 1 (the clipped-denominator bug used to add ~0.5% per hour here)
    prices = [0.10, 0.10] + [0.0001] * 50
    rows = [{'ts': i * H, 'market': 'Z', 'p': p, 'end_ms': len(prices) * H, 'outcome': 0, 'category': 'x', 'volume': 1e5}
            for i, p in enumerate(prices)]
    p = pl.DataFrame(rows)
    r = run_pred(p, np.full(len(p), -1.0), PredCost(unit_notional=1.0, half_spread=0.0))
    assert float(r['price_pnl'].sum()) == pytest.approx(1 / 0.90 - 1, rel=1e-6)
