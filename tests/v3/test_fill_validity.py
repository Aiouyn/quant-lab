"""Audit #2: the FILL row must be valid (not only the decision row), bar gap check honours stressed latency,
and engine <-> certifier parity holds. Contract: an order whose fill row is invalid/forward-filled or across a
time gap is DROPPED (no fill, no carry-over); the next valid decision row re-targets."""
import numpy as np
import polars as pl
import pytest

from desk.lab.backtest import CostModel, backtest
from desk.lab.certify import certify, reference_backtest

FREE = dict(taker_bps=0.0, slippage_bps=0.0, impact_mult=0.0, vol_slip_coef=0.0, deadband=0.0, funding=False)


def synth(n, seed=0, start=1_762_000_000_000, step_ms=1000):
    rng = np.random.default_rng(seed)
    mid = 100_000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    return pl.DataFrame({'ts': start + np.arange(n, dtype=np.int64) * step_ms, 'valid': np.ones(n, np.int8), 'mid': mid,
                         'spread_bps': np.full(n, 0.01), 'bimp_1': np.full(n, 0.4), 'aimp_1': np.full(n, 0.5),
                         'bimp_10': np.full(n, 2.0), 'aimp_10': np.full(n, 2.2), 'bimp_50': np.full(n, 6.0),
                         'aimp_50': np.full(n, 6.5), 'funding_rate': np.zeros(n)})


def _set(df, col, idx, val):
    a = df[col].to_numpy().copy()
    a[idx] = val
    return df.with_columns(pl.Series(col, a))


@pytest.mark.parametrize('latency_s', [1, 2, 5])
def test_invalid_fill_row_does_not_fill_at_stale_mid(latency_s):
    n, k = 400, 100
    cost = CostModel(latency_s=latency_s, **FREE)
    df = synth(n)
    fill = k + latency_s
    stale = 50_000.0                                    # a forward-filled, wildly stale mid on the invalid fill row
    df = _set(_set(df, 'valid', fill, 0), 'mid', fill, stale)
    tgt = np.zeros(n); tgt[k:] = 1.0                    # decision row k is valid; its fill row is not
    r = backtest(df, tgt, '1s', cost, keep_series=True)
    rows, qty, px = r['_fills']
    assert fill not in rows.tolist()
    assert not np.any(np.isclose(px, stale))
    # dropped; the next decision row that is valid AND has a valid fill row re-targets
    d = next(j for j in range(k + 1, n) if j != fill and j + latency_s != fill)
    assert rows[0] == d
    assert np.isclose(px[0], df['mid'][d + latency_s] * (1 + 0.005 / 1e4))
    ref = reference_backtest(df, tgt, '1s', cost)
    assert np.isclose(ref['pnl'], r['pnl'], rtol=0, atol=1e-6)
    assert certify(df, tgt, '1s', cost, r)['ok']


def test_invalid_bar_fill_row_drops_order():
    n, k = 60, 20
    cost = CostModel(**FREE)
    df = synth(n, step_ms=60_000).with_columns(pl.col('mid').alias('open'))
    df = _set(_set(df, 'valid', k + 1, 0.2), 'open', k + 1, 50_000.0)
    tgt = np.zeros(n); tgt[k:] = 1.0
    r = backtest(df, tgt, '1m', cost, keep_series=True)
    assert not np.any(np.isclose(r['_fills'][2], 50_000.0))
    assert r['_fills'][0][0] == k + 2              # decision k+1 is itself the invalid bar
    assert certify(df, tgt, '1m', cost, r)['ok']


def test_stressed_latency_bar_gap_blocks_fill():
    """latency 61 s on 1m bars -> fill at row i+2. Row i+1 is contiguous but row i+2 is after a 10-minute gap:
    the old check only looked one row ahead and filled across the gap."""
    n, k = 40, 10
    ts = 1_762_000_000_000 + np.arange(n, dtype=np.int64) * 60_000
    ts[k + 2:] += 600_000
    df = synth(n, step_ms=60_000).with_columns(pl.Series('ts', ts), pl.col('mid').alias('open'))
    cost = CostModel(latency_s=61, **FREE)
    tgt = np.zeros(n); tgt[k:] = 1.0
    r = backtest(df, tgt, '1m', cost, keep_series=True)
    rows = r['_fills'][0].tolist()
    assert k not in rows                       # old check: filled at row k+2's open across the gap
    assert rows[0] == k + 2                    # k+1 -> k+3 also spans the gap; k+2 -> k+4 is contiguous
    assert certify(df, tgt, '1m', cost, r)['ok']


@pytest.mark.parametrize('res,step,cost', [
    ('1s', 1000, CostModel()), ('1s', 1000, CostModel().stressed(extra_latency_s=3)),
    ('1m', 60_000, CostModel()), ('1m', 60_000, CostModel().stressed(extra_latency_s=120)),
    ('10s', 10_000, CostModel(latency_s=12))])
def test_engine_certifier_parity_with_invalid_rows_and_gaps(res, step, cost):
    n = 3000
    rng = np.random.default_rng(7)
    df = synth(n, step_ms=step)
    if res != '1s':
        df = df.with_columns((pl.col('mid') * (1 + 1e-5)).alias('open'))
    bad = rng.choice(n, 150, replace=False)
    df = _set(df, 'valid', bad, 0)
    ts = df['ts'].to_numpy().copy()
    for g in rng.choice(np.arange(10, n), 5, replace=False):
        ts[g:] += 30 * step                      # day-style gaps
    df = df.with_columns(pl.Series('ts', ts))
    tgt = np.repeat(rng.choice([-1, -0.4, 0, 0.6, 1], size=n // 20 + 1), 20)[:n]
    r = backtest(df, tgt, res, cost)
    c = certify(df, tgt, res, cost, r)
    assert c['ok'], c
