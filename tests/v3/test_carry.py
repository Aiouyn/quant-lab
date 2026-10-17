"""Carry engine: accounting identities, parity with an independent reference loop, causality detection."""
import numpy as np
import polars as pl
import pytest

from desk.lab.carry import CarryCost, run_carry, summarize_carry, carry_causality, MAJORS
from desk.lab.strategy import load_strategy

H = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % H)


def panel(n=500, coins=('BTC', 'WLD', 'SOL'), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in coins:
        hp = np.cumsum(rng.normal(0, 2e-5, n)); bp = np.cumsum(rng.normal(0, 2e-5, n))
        for t in range(n):
            rows.append({'ts': T0 + t * H, 'coin': c, 'hl_funding': float(rng.normal(1e-5, 2e-5)),
                         'hl_premium': float(hp[t]), 'bn_funding': float(rng.normal(8e-5, 5e-5)) if t % 8 == 0 else 0.0,
                         'bn_rate_last': 1e-4, 'bn_interval_h': 8.0, 'bn_premium': float(bp[t]), 'bn_close': 1.0,
                         'bn_qvol': float(rng.uniform(1e6, 5e7))})
    return pl.DataFrame(rows).sort(['ts', 'coin'])


def reference(p, pos, cost):
    """Independent loop over hours and coins (written from the contract in desk/lab/carry.py)."""
    pos = np.clip(pos, -1, 1)
    if cost.execution == 'maker':                       # every coin's position becomes effective one hour later
        df0 = p.with_columns(pl.Series('pos', pos)).sort(['ts', 'coin'])
        ts_list = sorted(df0['ts'].unique().to_list())
        nxt = {a: b for a, b in zip(ts_list, ts_list[1:])}
        shifted = {(nxt[r['ts']], r['coin']): r['pos'] for r in df0.iter_rows(named=True) if r['ts'] in nxt}
        pos = np.array([shifted.get((r['ts'], r['coin']), 0.0) for r in df0.iter_rows(named=True)])
        p = df0.drop('pos')
    df = p.with_columns(pl.Series('pos', pos)).sort(['ts', 'coin'])
    coins = sorted(df['coin'].unique().to_list())
    by_ts = {}
    for r in df.iter_rows(named=True):
        by_ts.setdefault(r['ts'], {})[r['coin']] = r
    hist_qv = {c: [] for c in coins}
    prev_n = {c: 0.0 for c in coins}
    prev_row = {}
    total = 0.0
    for ts in sorted(by_ts):
        rows = by_ts[ts]
        gross = max(1.0, sum(abs(rows[c]['pos']) for c in coins if c in rows) / cost.max_gross)
        for c in coins:
            r = rows.get(c)
            if r is None:
                continue
            if c in prev_row:
                pr = prev_row[c]
                total += prev_n[c] * (r['bn_funding'] - r['hl_funding'])
                total += prev_n[c] * ((r['hl_premium'] - pr['hl_premium']) - (r['bn_premium'] - pr['bn_premium']))
            n_new = cost.unit_notional * r['pos'] / gross
            d = abs(n_new - prev_n[c])
            if d > 0 and cost.execution == 'maker':
                total -= d * (cost.hl_maker_bps + cost.bn_maker_bps + 2 * cost.maker_adverse_bps + cost.basis_noise_bps) / 1e4
            elif d > 0:
                qv = np.mean(hist_qv[c][-24:]) if hist_qv[c] else r['bn_qvol']
                slip = cost.major_slip_bps if c in MAJORS else cost.alt_slip_bps
                per_leg = slip + min(cost.impact_coef * d / max(qv, 1e5), 50.0)
                total -= d * (cost.hl_taker_bps + cost.bn_taker_bps + 2 * per_leg + cost.basis_noise_bps) / 1e4
            hist_qv[c].append(r['bn_qvol'])
            prev_n[c] = n_new
            prev_row[c] = r
    return total


@pytest.mark.parametrize('cost', [CarryCost(), CarryCost(max_gross=0.5, impact_coef=200.0), CarryCost(execution='maker')])
def test_parity_with_reference(cost):
    p = panel()
    rng = np.random.default_rng(1)
    pos = np.repeat(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=len(p) // 30 + 1), 30)[:len(p)]
    res = run_carry(p, pos, cost)
    assert res['pnl'].sum() == pytest.approx(reference(p, pos, cost), rel=1e-9, abs=1e-6)


def test_pure_funding_accounting():
    n = 49
    p = pl.DataFrame({'ts': T0 + np.arange(n) * H, 'coin': ['WLD'] * n, 'hl_funding': np.full(n, 2e-5),
                      'hl_premium': np.zeros(n), 'bn_funding': np.where(np.arange(n) % 8 == 0, 1e-4, 0.0),
                      'bn_rate_last': np.full(n, 1e-4), 'bn_interval_h': np.full(n, 8.0), 'bn_premium': np.zeros(n),
                      'bn_close': np.ones(n), 'bn_qvol': np.full(n, 1e9)})
    cost = CarryCost(hl_taker_bps=0, bn_taker_bps=0, alt_slip_bps=0, impact_coef=0)
    res = run_carry(p, np.ones(n), cost)                  # long HL pays 0.2 bps/h, short BN pays 1 bp per 8 h
    expected = 100_000 * (6 * 1e-4 - 48 * 2e-5)           # 6 Binance events after t=0, 48 HL hours
    assert res['funding'].sum() == pytest.approx(expected, rel=1e-9)
    s = summarize_carry(res, cost)
    assert s['days'] >= 2 and s['fees'] == 0


def test_carry_lookahead_is_caught():
    leak = '''DESK = 'carry'
PARAMS = {}
def signal(panel, params):
    nxt = panel.group_by('coin', maintain_order=True).agg(pl.col('hl_funding').shift(-1)).explode('hl_funding')
    return np.sign(-np.nan_to_num(panel.with_columns(pl.col('hl_funding').shift(-1).over('coin').alias('f'))['f'].to_numpy()))
'''
    ok = '''DESK = 'carry'
PARAMS = {'w': 24}
def signal(panel, params):
    s = panel.with_columns((pl.col('hl_funding') - pl.col('bn_rate_last') / 8).rolling_mean(params['w'], min_samples=1).over('coin').alias('x'))
    return np.sign(-s['x'].to_numpy())
'''
    p = panel(300)
    assert not carry_causality(load_strategy(leak), p)['ok']
    assert carry_causality(load_strategy(ok), p)['ok']
