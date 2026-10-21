"""Audit #6: funding is charged for EVERY hour elapsed between consecutive rows, at that hour's rate,
identically in the engine and the certifier."""
import numpy as np
import polars as pl
import pytest

from desk.lab.backtest import CostModel, backtest
from desk.lab.certify import certify, reference_backtest

H = 3_600_000
T0 = 1_762_000_000_000 // H * H            # hour-aligned
FREE = dict(taker_bps=0.0, slippage_bps=0.0, impact_mult=0.0, vol_slip_coef=0.0, deadband=0.0, latency_s=1)


def frame(ts, mid=100_000.0, rate=0.0):
    n = len(ts)
    return pl.DataFrame({'ts': np.asarray(ts, np.int64), 'valid': np.ones(n, np.int8), 'mid': np.full(n, mid),
                         'spread_bps': np.zeros(n), 'funding_rate': np.full(n, rate)})


def series(hours, rates):
    return pl.DataFrame({'hour': [T0 + h * H for h in hours], 'funding_rate': rates})


def gap_frame():
    """Rows each second for 10 s just after hour 0, then a gap to hour 5 + 10 s (hours 1..5 elapse in one step)."""
    ts = [T0 + 1000 * i for i in range(1, 11)] + [T0 + 5 * H + 1000 * i for i in range(1, 11)]
    return frame(ts, rate=0.0003)


RATES = {1: 1e-4, 2: -2e-4, 3: 3e-4, 4: 0.5e-4, 5: 2e-4}


def test_multi_hour_gap_charges_every_hour_at_its_rate():
    df = gap_frame()
    fs = series(list(RATES), list(RATES.values()))
    tgt = np.ones(len(df))                       # long 1 unit = 1 BTC at 100k, held across the gap
    cost = CostModel(**FREE)
    r = backtest(df, tgt, '1s', cost, funding_series=fs)
    expected = 1.0 * 100_000.0 * sum(RATES.values())     # pos * mid_after_gap * sum(hourly rates)
    assert r['funding_paid'] == pytest.approx(expected, rel=1e-9)
    ref = reference_backtest(df, tgt, '1s', cost, fs)
    assert ref['funding_paid'] == pytest.approx(expected, rel=1e-9)
    assert certify(df, tgt, '1s', cost, r, funding_series=fs)['ok']


def test_gap_without_series_charges_each_hour_at_row_rate():
    df = gap_frame()
    tgt = np.ones(len(df))
    r = backtest(df, tgt, '1s', CostModel(**FREE))
    assert r['funding_paid'] == pytest.approx(100_000.0 * 5 * 0.0003, rel=1e-9)   # old code: only 1 hour
    assert certify(df, tgt, '1s', CostModel(**FREE), r)['ok']


def test_missing_hour_in_series_uses_latest_earlier_print():
    df = gap_frame()
    fs = series([1, 3], [1e-4, 3e-4])            # hours 2, 4, 5 missing -> 1e-4, 3e-4, 3e-4
    r = backtest(df, np.ones(len(df)), '1s', CostModel(**FREE), funding_series=fs)
    assert r['funding_paid'] == pytest.approx(100_000.0 * (1e-4 + 1e-4 + 3e-4 + 3e-4 + 3e-4), rel=1e-9)


def test_no_gap_case_unchanged():
    n = 3 * 3600 + 10
    ts = T0 + 1000 * np.arange(1, n + 1)
    rate = np.where((ts // H) % 2 == 0, 1e-4, -0.5e-4)
    df = frame(ts).with_columns(pl.Series('funding_rate', rate))
    tgt = np.ones(n)
    cost = CostModel(**FREE)
    base = backtest(df, tgt, '1s', cost)
    fs = series([1, 2, 3], [-0.5e-4, 1e-4, -0.5e-4])
    with_series = backtest(df, tgt, '1s', cost, funding_series=fs)
    expected = 100_000.0 * (-0.5e-4 + 1e-4 - 0.5e-4)
    assert base['funding_paid'] == pytest.approx(expected, abs=1e-9)
    assert with_series['funding_paid'] == pytest.approx(expected, abs=1e-9)
    assert base['pnl'] == pytest.approx(with_series['pnl'], abs=1e-9)


@pytest.mark.parametrize('res,step', [('1s', 1000), ('1m', 60_000)])
def test_engine_certifier_parity_with_gaps(res, step):
    rng = np.random.default_rng(11)
    n = 4000
    ts = T0 + step * np.arange(1, n + 1)
    for g in rng.choice(np.arange(50, n), 6, replace=False):
        ts[g:] += int(rng.integers(1, 9)) * H + int(rng.integers(0, H))
    mid = 100_000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    df = frame(ts).with_columns(pl.Series('mid', mid), pl.Series('spread_bps', np.full(n, 0.01)),
                                pl.Series('funding_rate', rng.normal(1e-5, 5e-5, n)))
    if res != '1s':
        df = df.with_columns(pl.col('mid').alias('open'))
    hours = np.arange(0, int((ts[-1] - T0) // H) + 2)
    fs = series(hours.tolist(), rng.normal(1e-5, 8e-5, len(hours)).tolist())
    tgt = np.repeat(rng.choice([-1, -0.5, 0, 0.5, 1], size=n // 40 + 1), 40)[:n]
    for cost in (CostModel(), CostModel().stressed(extra_latency_s=2)):
        for f in (None, fs):
            r = backtest(df, tgt, res, cost, funding_series=f)
            ref = reference_backtest(df, tgt, res, cost, f)
            assert r['funding_paid'] == pytest.approx(ref['funding_paid'], rel=1e-9, abs=1e-9)
            assert certify(df, tgt, res, cost, r, funding_series=f)['ok']


def test_series_is_clipped_to_eval_window():
    """A print far before the window (or after it) must never be used: only [first ts - 1 h, last ts] counts."""
    df = gap_frame()
    cost = CostModel(**FREE)
    tgt = np.ones(len(df))
    old = series([-10], [9e-3])                        # 10 h before the data: outside the window
    fut = series([7], [9e-3])                          # after the last row (hour 5 + 10 s): outside the window
    inside = series([-1, 0, 3], [9e-3, 1e-4, 3e-4])    # hour -1 < first ts - 1 h: clipped; hours 0 and 3 kept
    for fs in (old, fut):
        r = backtest(df, tgt, '1s', cost, funding_series=fs)
        assert r['funding_paid'] == 0.0
        assert reference_backtest(df, tgt, '1s', cost, fs)['funding_paid'] == 0.0
    r = backtest(df, tgt, '1s', cost, funding_series=inside)
    expected = 100_000.0 * (1e-4 + 1e-4 + 3e-4 + 3e-4 + 3e-4)    # hours 1,2 -> print at 0; hours 3,4,5 -> print at 3
    assert r['funding_paid'] == pytest.approx(expected, rel=1e-9)
    assert certify(df, tgt, '1s', cost, r, funding_series=inside)['ok']


def test_missing_funding_file_raises_when_funding_enabled(tmp_path):
    from desk.data.store import FundingSeriesMissing, funding_series
    with pytest.raises(FundingSeriesMissing, match='funding_hl.parquet'):
        funding_series(tmp_path)
    with pytest.raises(FundingSeriesMissing):
        funding_series(None)
    assert funding_series(tmp_path, required=False) is None         # funding disabled: nothing needed
    series([1], [1e-4]).write_parquet(tmp_path / 'funding_hl.parquet')
    h, r = funding_series(tmp_path)
    assert h.tolist() == [T0 + H] and r.tolist() == [1e-4]


def test_evaluate_refuses_store_without_funding(tmp_path):
    import json
    import types
    from desk.data.store import FeatureStore, FundingSeriesMissing
    from desk.lab.strategy import evaluate
    S = types.SimpleNamespace(store=tmp_path, data_dir=tmp_path, dev_fraction=0.6, val_fraction=0.25)
    for sub in ('1s', '10s', '1m', 'meta'):
        (tmp_path / sub).mkdir()
    d = '2025-11-03'
    ts = 1_762_128_000_000 + 60_000 * np.arange(1440)
    frame(ts).write_parquet(tmp_path / '1m' / f'{d}.parquet')
    frame(ts).write_parquet(tmp_path / '1s' / f'{d}.parquet')
    (tmp_path / 'splits.json').write_text(json.dumps({'created': 0, 'dev': [d], 'val': [], 'holdout': [], 'forward': []}))
    code = "RESOLUTION='1m'\ndef signal(df, p):\n    return df['mid'] * 0 + 1\n"
    st = FeatureStore(S, root=tmp_path, allowed=('dev',))
    with pytest.raises(FundingSeriesMissing):
        evaluate(code, st, [d], CostModel(), with_causality=False)
    r = evaluate(code, st, [d], CostModel(funding=False), with_causality=False)
    assert r['funding_paid'] == 0.0
