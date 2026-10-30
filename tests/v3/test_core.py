"""Desk OS v3 regression tests: reconstruction, accounting parity, causality, split guard, stats, org plumbing."""
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from desk.data.book import GridBook, TICK
from desk.lab.backtest import backtest, CostModel
from desk.lab.certify import certify
from desk.lab.strategy import load_strategy, causality_test, StrategyError
from desk.lab import stats


def synth(n=20000, seed=0, start=1_762_000_000_000, funding=0.0001):
    rng = np.random.default_rng(seed)
    mid = 100_000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    sp = np.full(n, 0.01)
    df = pl.DataFrame({'ts': start + np.arange(n, dtype=np.int64) * 1000, 'valid': np.ones(n, np.int8), 'mid': mid,
                       'bid_px': mid * (1 - 0.5e-6), 'ask_px': mid * (1 + 0.5e-6), 'spread_bps': sp,
                       'bimp_1': np.full(n, 0.4), 'aimp_1': np.full(n, 0.5), 'bimp_10': np.full(n, 2.0),
                       'aimp_10': np.full(n, 2.2), 'bimp_50': np.full(n, 6.0), 'aimp_50': np.full(n, 6.5),
                       'qi10': rng.normal(0, 0.3, n), 'funding_rate': np.full(n, funding)})
    return df


# ------------------------------------------------------------------ reconstruction
def _lv(px, sz):
    return [[f'{p:.1f}', f'{s}', '3'] for p, s in zip(px, sz)]


def test_periodic_snapshot_is_a_resync_not_a_break():
    b = GridBook(max_price=200_000)
    bids = _lv([100.0 - i * TICK for i in range(50)], [5] * 50)
    asks = _lv([100.1 + i * TICK for i in range(50)], [5] * 50)
    assert b.apply({'action': 'snapshot', 'bids': bids, 'asks': asks})
    f1 = b.features()
    assert b.apply({'action': 'update', 'bids': [['100.0', '9', '4']], 'asks': [['100.1', '0', '0']]})
    f2 = b.features()
    assert abs(f2['ask_px'] - 100.2) < 1e-9 and f2['bid_sz0'] == 9
    assert f2['ofi1'] != 0                       # OFI uses the previous second
    # an identical periodic snapshot must not reset OFI state or count as a break
    bids2 = _lv([100.0 - i * TICK for i in range(50)], [9] + [5] * 49)
    asks2 = _lv([100.2 + i * TICK for i in range(49)], [5] * 49)
    assert b.apply({'action': 'snapshot', 'bids': bids2, 'asks': asks2})
    assert b.prev_top is not None
    assert b.q['resyncs'] == 1 and b.q['resync_top20_mismatch_sum'] < 0.1
    f3 = b.features()
    assert f3['mid'] == f2['mid'] and f3['ofi1'] == 0


def test_depth_bands_and_sweep_impact():
    b = GridBook(max_price=200_000)
    bids = _lv([100_000.0 - i * TICK for i in range(3000)], [10] * 3000)
    asks = _lv([100_000.1 + i * TICK for i in range(3000)], [10] * 3000)
    b.apply({'action': 'snapshot', 'bids': bids, 'asks': asks})
    f = b.features()
    # 10 bps of 100k = $100 = 1000 ticks -> ~1000 levels x 10 contracts
    assert 9_900 <= f['bdep_10'] <= 10_100
    assert f['bimp_10'] > f['bimp_1'] > 0


# ------------------------------------------------------------------ accounting
@pytest.mark.parametrize('cost', [CostModel(), CostModel(latency_s=3, impact_mult=2.0), CostModel(unit_notional=5e6, deadband=0.0)])
def test_engine_matches_independent_reference(cost):
    df = synth(6000)
    rng = np.random.default_rng(3)
    tgt = np.repeat(rng.choice([-1, -0.4, 0, 0.6, 1], size=200), 30)[:6000]
    r = backtest(df, tgt, '1s', cost)
    c = certify(df, tgt, '1s', cost, r)
    assert c['ok'], c


def test_funding_is_charged_hourly_on_open_positions():
    df = synth(3 * 3600 + 10, funding=0.0001)
    tgt = np.ones(len(df))
    c = CostModel(taker_bps=0, slippage_bps=0, impact_mult=0, deadband=0)
    r = backtest(df.with_columns(pl.lit(0.0).alias('spread_bps')), tgt, '1s', c)
    # ~3 hour boundaries crossed with $100k long, 1 bp/hour each
    assert 25 < r['funding_paid'] < 35


def test_fees_scale_with_turnover():
    df = synth(5000)
    tgt = np.where(np.arange(5000) // 500 % 2 == 0, 1.0, -1.0)
    r = backtest(df, tgt, '1s', CostModel())
    assert abs(r['fees'] - r['turnover_units'] * 100_000 * 4.5e-4) / r['fees'] < 0.01


# ------------------------------------------------------------------ causality
CAUSAL = '''RESOLUTION='1s'
PARAMS={'w': 60}
def signal(df, params):
    m = df['qi10'].rolling_mean(params['w'], min_samples=1).fill_null(0)
    return (m > 0.05).cast(pl.Float64) - (m < -0.05).cast(pl.Float64)
'''
LEAK_SHIFT = '''RESOLUTION='1s'
def signal(df, params):
    return np.sign(df['mid'].shift(-10).fill_null(strategy='forward') - df['mid']).to_numpy()
'''
LEAK_ZSCORE = '''RESOLUTION='1s'
def signal(df, params):
    x = df['qi10']
    return ((x - x.mean()) / x.std()).clip(-1, 1)
'''


def test_causal_strategy_passes():
    mod = load_strategy(CAUSAL)
    assert causality_test(mod, synth(5000))['ok']


@pytest.mark.parametrize('code', [LEAK_SHIFT, LEAK_ZSCORE])
def test_lookahead_is_caught(code):
    mod = load_strategy(code)
    assert not causality_test(mod, synth(5000))['ok']


def test_forbidden_io_rejected():
    with pytest.raises(StrategyError):
        load_strategy("RESOLUTION='1m'\nimport os\ndef signal(df, p): return df['mid']*0")


# ------------------------------------------------------------------ split guard
def test_split_guard(tmp_path):
    from desk.data.store import FeatureStore, SplitViolation

    class S:
        store = tmp_path; data_dir = tmp_path; dev_fraction = 0.6; val_fraction = 0.25
    for sub in ('1s', '10s', '1m', 'meta'):
        (tmp_path / sub).mkdir()
    days = [f'2025-11-{d:02d}' for d in range(1, 11)]
    for d in days:
        synth(100).write_parquet(tmp_path / '1s' / f'{d}.parquet')
    from desk.data.provenance import write_splits
    write_splits(tmp_path / 'splits.json', {'created': 0, 'dev': days[:6], 'val': days[6:8], 'holdout': days[8:], 'forward': []})
    st = FeatureStore(S(), root=tmp_path, allowed=('dev',))
    st.load(days[0])
    with pytest.raises(SplitViolation):
        st.load(days[7])
    with pytest.raises(SplitViolation):
        st.days('holdout')
    view = FeatureStore(S(), root=tmp_path).build_dev_view()
    assert sorted(p.stem for p in (view / '1s').glob('*.parquet')) == days[:6]


# ------------------------------------------------------------------ statistics
def test_dsr_penalizes_many_trials():
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.01, 120)
    assert stats.dsr(x, 1) > stats.dsr(x, 1000)
    assert 0 <= stats.psr(x) <= 1


def test_bootstrap_lower_bound_below_mean():
    x = np.random.default_rng(1).normal(0.001, 0.01, 200)
    assert stats.block_bootstrap_lower(x) < x.mean()


def test_pbo_detects_pure_noise_selection():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, (200, 30))
    assert stats.pbo_cscv(M, 8) > 0.2


# ------------------------------------------------------------------ org plumbing
def test_throttle_aimd():
    from desk.company.llm import Throttle

    async def run():
        t = Throttle(start=8, cap=32)
        await t.acquire(); t.release(ok=False, throttled=True)
        assert t.limit < 8
        t.last_change = t.last_throttle = time.time() - 200
        await t.acquire(); t.release(ok=True)
        return t.limit
    assert asyncio.run(run()) > 4.8


def test_priority_queue_order():
    from desk.company.llm import Throttle

    async def run():
        t = Throttle(start=1, cap=1)
        await t.acquire()                  # occupy the only slot
        order = []

        async def w(name, p):
            await t.acquire(p); order.append(name); t.release()
        tasks = [asyncio.create_task(w('low', 0.5)), asyncio.create_task(w('high', 3.0))]
        await asyncio.sleep(0.01)
        t.release()
        await asyncio.gather(*tasks)
        return order
    assert asyncio.run(run()) == ['high', 'low']


def test_ledger_board_messages_and_broadcast(tmp_path):
    from desk.company.ledger import Ledger
    L = Ledger(tmp_path / 'l.sqlite3')
    bid = L.post('thesis', 'OFI persistence', 'test plan', 'a', pod='orderflow', assignee='b')
    L.update_board(bid, 'b', 'in_progress', 'started')
    assert L.board_item(bid)['comments'][0]['body'] == 'started'
    L.send('all', 'cio', 'memo')
    L.send('pod:orderflow', 'lead', 'pod note')
    assert len(L.inbox('b', 'orderflow')) == 2
    assert len(L.inbox('b', 'orderflow')) == 0          # read once per agent
    assert len(L.inbox('c', 'orderflow')) == 2          # broadcast still visible to others
    assert L.search('persistence')[0]['ref'] == bid


def test_gates():
    from desk.pipeline.validation import check, DEV_GATES
    rep = {'summary': {'sharpe': 2.0, 'trades': 100}, 'causality': {'ok': True}, 'certification': {'ok': True},
           'robustness': {'fees_x1.5': {'pnl': 5}, 'latency_+2s': {'pnl': 3}, 'first_half_sharpe': 1, 'second_half_sharpe': 0.5,
                          'positive_months': 0.7, 'param_neighborhood': {'fraction_profitable': 0.75}}}
    assert check(DEV_GATES, rep) == []
    rep['robustness']['fees_x1.5']['pnl'] = -1
    assert check(DEV_GATES, rep) == ['fees_x1.5_profitable']


def test_portfolio_allocation_caps_and_dedup():
    from desk.pipeline.portfolio import allocate
    rng = np.random.default_rng(5)
    base = rng.normal(0.001, 0.01, (100, 4))
    R = np.c_[base, base[:, 0] + rng.normal(0, 1e-4, 100)]      # column 4 duplicates column 0
    w = allocate(R, list('abcde'))
    assert len(w) == 4 and max(w.values()) <= 0.35 + 1e-9 and sum(w.values()) <= 1 + 1e-9


def test_roster_scales():
    from desk.company.runtime import roster
    r = roster(28)
    from desk.company.roles import PODS
    assert len(r) == 28 and sum(1 for x in r if x[1] == 'lead') == len(PODS) and any(x[1] == 'officer' for x in r)
    assert r[len([x for x in r if x[1] not in ('lead', 'researcher')])][2] == 'macro'


def test_quota_exhausted_pauses_without_consuming_retries(tmp_path):
    import types
    from desk.company.ledger import Ledger
    from desk.company.llm import LLM

    class QuotaErr(Exception):
        status_code = 429

    calls = {'n': 0}

    async def fake_stream(messages, tools, max_tokens, thinking, model=None):
        calls['n'] += 1
        if calls['n'] <= 8:                      # 8 consecutive quota errors would exceed 6 normal retries
            raise QuotaErr("Error code: 429 - {'message': 'quota exhausted'}")
        return {'content': 'ok', 'reasoning': '', 'tool_calls': [], 'finish': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}

    s = types.SimpleNamespace(api_key='x', base_url='http://localhost', request_timeout_s=5, start_concurrency=2,
                              max_concurrency=2, model='m', max_completion_tokens=10)
    llm = LLM(s, Ledger(tmp_path / 'l.sqlite3'))
    llm._stream = fake_stream

    async def no_wait(agent):
        llm.quota_paused_until = 0
    llm.quota_gate = no_wait
    res = asyncio.run(llm.chat('a', [{'role': 'user', 'content': 'hi'}]))
    assert res['content'] == 'ok' and calls['n'] == 9 and llm.throttle.inflight == 0


def test_missed_entries_drop_whole_episodes():
    from desk.lab.strategy import miss_entries
    sig = np.tile(np.r_[np.ones(10), np.zeros(5), -np.ones(10), np.zeros(5)], 200)
    out = miss_entries(sig, 0.25)
    assert set(np.unique(out)) <= {-1.0, 0.0, 1.0}
    assert np.all((out == 0) | (out == sig))                 # never creates or flips positions
    kept = np.sum(np.abs(np.diff(np.r_[0, np.abs(out)])) > 0) / 2
    total = np.sum(np.abs(np.diff(np.r_[0, np.abs(sig)])) > 0) / 2
    assert 0.6 < kept / total < 0.9                          # ~25% of episodes skipped


def test_volatility_slippage_costs_more_in_fast_markets():
    df = synth(4000)
    tgt = np.where(np.arange(4000) // 400 % 2 == 0, 1.0, -1.0)
    calm = backtest(df, tgt, '1s', CostModel(vol_slip_coef=0.0))
    fast = backtest(df, tgt, '1s', CostModel(vol_slip_coef=0.5))
    assert fast['pnl'] < calm['pnl']


def test_positive_weeks_ignores_flat_weeks():
    from desk.lab.stats import weekly_positive_fraction
    days = list(range(70))
    rets = [0.0] * 70
    for d in (0, 1, 14, 15, 28):          # three active weeks, all positive; seven flat weeks
        rets[d] = 1.0
    assert weekly_positive_fraction(days, rets) == 1.0
    rets[28] = -3.0
    assert abs(weekly_positive_fraction(days, rets) - 2 / 3) < 1e-12


def test_context_clipping_keeps_prefix_stable():
    import json
    from desk.company.agent import CLIP_BLOCK, _rolling_context, _trim_reasoning
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'brief'}]
    rendered, changes = [], 0
    for t in range(30):
        msgs.append({'role': 'assistant', 'content': 'a' * 2000, 'reasoning_content': 'r' * 500,
                     'tool_calls': [{'id': f'c{t}', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]})
        msgs.append({'role': 'tool', 'tool_call_id': f'c{t}', 'content': 't' * 3000})
        out = [json.dumps(m, sort_keys=True) for m in _rolling_context(_trim_reasoning(msgs))]
        if rendered and out[:len(rendered)] != rendered:
            changes += 1
        rendered = out
    assert changes <= 30 // CLIP_BLOCK * 3          # the prefix is rewritten only at block boundaries
    assert sum(len(m.get('content') or '') for m in _rolling_context(msgs)) < 30 * 5000 / 2   # still clipping


def test_desk_aware_sharpe_bars():
    from desk.pipeline.validation import DEV_GATES, val_gates
    from desk.config import SETTINGS
    mk = lambda desk, sh: {'desk': desk, 'summary': {'sharpe': sh}}
    assert DEV_GATES['dev_sharpe'](mk('macro', 0.75)) and not DEV_GATES['dev_sharpe'](mk('macro', 0.65))
    assert not DEV_GATES['dev_sharpe'](mk('perps', 0.75)) and DEV_GATES['dev_sharpe'](mk('perps', 1.6))
    vg = [f for k, f in val_gates(SETTINGS).items() if k.startswith('val_sharpe')][0]
    assert vg(mk('macro', 0.65)) and not vg(mk('perps', 0.65))
