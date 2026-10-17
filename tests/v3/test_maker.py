"""Queue-aware limit-order engine: behavior and parity with the independent reference implementation."""
import numpy as np
import polars as pl
import pytest

from desk.lab.backtest import CostModel
from desk.lab.certify import reference_maker
from desk.lab.maker import run_block, validate_execution, maker_summary

T0 = 1_762_000_000_000 - (1_762_000_000_000 % 3_600_000)


def frame(n, bid=100_000.0, bsz=500.0):
    return pl.DataFrame({'ts': T0 + np.arange(n, dtype=np.int64) * 1000, 'valid': np.ones(n, np.int8),
                         'mid': np.full(n, bid + 0.05), 'bid_px': np.full(n, bid), 'ask_px': np.full(n, bid + 0.1),
                         'bid_sz0': np.full(n, bsz), 'ask_sz0': np.full(n, bsz), 'spread_bps': np.full(n, 0.01),
                         'bimp_1': np.full(n, .5), 'aimp_1': np.full(n, .5), 'bimp_10': np.full(n, 2.), 'aimp_10': np.full(n, 2.),
                         'bimp_50': np.full(n, 6.), 'aimp_50': np.full(n, 6.), 'funding_rate': np.zeros(n)})


def tape(rows):
    if not rows:
        return pl.DataFrame({'ts': [], 'px': [], 'side': [], 'size': []},
                            schema={'ts': pl.Int64, 'px': pl.Int32, 'side': pl.Int8, 'size': pl.Float32})
    return pl.DataFrame(rows, schema={'ts': pl.Int64, 'px': pl.Int32, 'side': pl.Int8, 'size': pl.Float32}, orient='row')


EX = validate_execution({'mode': 'maker', 'ttl_s': 30, 'on_expiry': 'cancel', 'repeg_s': 5, 'exit': 'maker'})
COST = CostModel(deadband=0.0)
BID_T = 1_000_000                                   # 100000.0 / 0.1


def test_queue_is_consumed_before_we_fill():
    df = frame(20)
    tgt = np.zeros(20); tgt[0:] = 1.0              # want +$100k (~1 BTC) from row 0 -> order placed at row 1
    # queue ahead = 500 contracts (5 BTC). Sells AT our bid: 300 ct (row 3), 300 ct (row 4) -> 1 BTC fills at row 4
    tp = tape([(T0 + 3000, BID_T, -1, 300.0), (T0 + 4000, BID_T, -1, 300.0)])
    eq, pos, fills, st = run_block(df, tgt, tp, COST, EX)
    assert pos[3] == 0.0 and pos[4] == pytest.approx(min(1.0, (600 - 500) / 100) , rel=1e-6)
    assert fills[0, 3] == 1.0 and fills[0, 2] == pytest.approx(100_000.0)


def test_trade_through_fills_everything_and_buyer_trades_do_not():
    df = frame(10)
    tgt = np.ones(10)
    tp = tape([(T0 + 2000, BID_T, 1, 5000.0),          # buyer-initiated at our price: cannot hit a resting bid
               (T0 + 3000, BID_T - 1, -1, 1.0)])       # a sell below our bid -> our level was traded through
    eq, pos, fills, st = run_block(df, tgt, tp, COST, EX)
    assert pos[2] == 0.0 and pos[3] == pytest.approx(100_000 / 100_000.05, rel=1e-6)


def test_no_fill_without_trades_then_expiry_cross():
    df = frame(50)
    tgt = np.ones(50)
    ex = dict(EX, on_expiry='cross', ttl_s=10)
    eq, pos, fills, st = run_block(df, tgt, tape([]), COST, ex)
    assert pos[10] == 0.0 and pos[-1] > 0.99           # crossed as taker after ttl
    assert fills[0, 3] == 0.0                           # taker fill


def test_orders_never_fill_on_trades_of_their_own_placement_second():
    df = frame(6)
    tgt = np.ones(6)
    tp = tape([(T0 + 1000, BID_T - 5, -1, 1e6)])       # attached to row 1 = trades BEFORE our order arrived
    eq, pos, fills, st = run_block(df, tgt, tp, COST, EX)
    assert np.all(pos == 0.0)


def _rows_and_tape(df, tp):
    tape_by_ts = {}
    for r in tp.iter_rows():
        tape_by_ts.setdefault(r[0], []).append((r[1], r[2], r[3]))
    return df.to_dicts(), tape_by_ts


@pytest.mark.parametrize('ex', [EX, dict(EX, on_expiry='cross', ttl_s=8), dict(EX, exit='taker', repeg_s=2)])
@pytest.mark.parametrize('cost', [CostModel(), CostModel(queue_mult=2.0, latency_s=3)])
def test_parity_with_reference_on_random_market(ex, cost):
    rng = np.random.default_rng(4)
    n = 3000
    steps = rng.choice([-1, 0, 0, 0, 1], size=n)
    bid_t = BID_T + np.cumsum(steps)
    df = frame(n).with_columns(pl.Series('bid_px', bid_t * 0.1), pl.Series('ask_px', (bid_t + 1) * 0.1),
                               pl.Series('mid', (bid_t + 0.5) * 0.1), pl.Series('bid_sz0', rng.uniform(10, 400, n)),
                               pl.Series('ask_sz0', rng.uniform(10, 400, n)))
    rows = []
    for k in range(1, n):
        for _ in range(rng.integers(0, 4)):
            side = int(rng.choice([-1, 1]))
            px = int(bid_t[k - 1] + (0 if side < 0 else 1) + rng.integers(-2, 3))
            rows.append((T0 + k * 1000, px, side, float(rng.uniform(1, 200))))
    tp = tape(rows).sort(['ts', 'side', 'px'])
    tgt = np.repeat(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n // 60 + 1), 60)[:n]
    eq, pos, fills, st = run_block(df, tgt, tp, cost, validate_execution(ex))
    r, tbt = _rows_and_tape(df, tp)
    ref = reference_maker(r, tbt, tgt, cost, validate_execution(ex))
    assert len(fills) > 5
    assert eq[-1] == pytest.approx(ref['pnl'], abs=1e-6 * cost.unit_notional)


def test_parity_on_real_okx_day():
    from desk.config import SETTINGS
    from desk.lab.maker import load_tape, EXEC_COLS
    from desk.data.store import FeatureStore
    st = FeatureStore(SETTINGS, allowed=('dev',))
    if not (SETTINGS.store / 'tape').exists():
        pytest.skip('tape not built')
    d = st.days('dev')[3]
    df = st.load([d], '1s', columns=EXEC_COLS).head(20_000)
    tp = load_tape(SETTINGS.store, [d])
    tp = tp.filter(pl.col('ts') <= df['ts'][-1])
    rng = np.random.default_rng(9)
    tgt = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=len(df) // 300 + 1), 300)[:len(df)]
    eq, pos, fills, s = run_block(df, tgt, tp, CostModel(), EX)
    r, tbt = _rows_and_tape(df, tp)
    ref = reference_maker(r, tbt, tgt, CostModel(), EX)
    assert len(fills) > 10
    assert eq[-1] == pytest.approx(ref['pnl'], abs=1e-2)
