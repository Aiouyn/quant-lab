"""Panel-desk drills: the planted effect is real and tradable, the null market predicts nothing, the answer is hidden
from agents, and scores reach the main ledger's scoreboard."""
from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from desk.lab import panel_drills as PD

HOUR = 3_600_000


def perps_panel(n_coins=12, start='2023-06-01', end='2026-10-01', seed=0):
    rng = np.random.default_rng(seed)
    t0 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    ts = np.arange(t0, t1, HOUR)
    frames = []
    for i in range(n_coins):
        r = rng.normal(0, 0.006, len(ts))
        px = 10 * np.exp(np.cumsum(r))
        prem = np.convolve(rng.normal(0, 1e-4, len(ts)), np.ones(24) / 24, mode='same')
        frames.append(pl.DataFrame({'ts': ts, 'coin': f'C{i}', 'px': px, 'px_fill': px * 1.0001, 'oracle': px,
                                    'mark': px, 'premium': prem, 'oi': rng.uniform(1e6, 2e6, len(ts)),
                                    'vlm': rng.uniform(1e6, 5e6, len(ts)), 'isp': rng.uniform(2, 10, len(ts)),
                                    'f': rng.normal(1e-5, 1e-5, len(ts))}))
    return pl.concat(frames).sort(['ts', 'coin'])


def _fwd_ret_corr(panel, score, lag=1):
    p = panel.with_columns(pl.Series('s', score))
    p = p.with_columns((pl.col('px_fill').shift(-lag).over('coin') / pl.col('px_fill') - 1).alias('fr'))
    p = p.drop_nulls(['fr'])
    return float(np.corrcoef(p['s'].to_numpy(), p['fr'].to_numpy())[0, 1])


def test_planted_effect_is_causal_and_tradable():
    panel = perps_panel(n_coins=10, start='2024-01-01', end='2024-04-01')
    secret = {'feature': 'premium', 'window': 24, 'transform': 'level', 'sign': 1, 'beta': 1e-3, 'lag': 1}
    planted = PD.make_planted(panel, 'perps', secret)
    s = PD.hidden_score(planted, 'perps', secret)
    assert np.allclose(s, PD.hidden_score(panel, 'perps', secret))       # the score uses no price column
    # it predicts the next hour, and only because it was planted (same noise on both sides of the difference)
    assert _fwd_ret_corr(planted, s) - _fwd_ret_corr(panel, s) > 0.05
    k = len(planted) // 2                                                 # causal: prefix gives the same score
    assert np.allclose(PD.hidden_score(planted.head(k), 'perps', secret), s[:k])


def test_null_market_has_no_predictability_beyond_a_block():
    panel = perps_panel(n_coins=6, start='2024-01-01', end='2024-03-01')
    secret = {'feature': 'premium', 'window': 24, 'transform': 'level', 'sign': 1, 'beta': 2e-3, 'lag': 1}
    planted = PD.make_planted(panel, 'perps', secret)
    null = PD.make_null(planted, 'perps', np.random.default_rng(1))
    s = PD.hidden_score(null, 'perps', secret)
    assert _fwd_ret_corr(planted, s) > 0.1
    assert abs(_fwd_ret_corr(null, s)) < 0.04                             # the planted effect is destroyed
    assert float(null['f'].abs().max()) == 0.0                            # no funding yield to harvest
    for c in ('px', 'px_fill', 'oracle', 'mark'):                         # prices stay positive and consistent
        assert float(null[c].min()) > 0
    ratio = (null['px'] / null['oracle']).to_numpy()
    assert np.allclose(ratio, (panel['px'] / panel['oracle']).to_numpy())


def test_build_report_and_scoreboard(tmp_path):
    from desk.config import SETTINGS
    from desk.company.ledger import Ledger
    from desk.lab.perps import PerpsStore
    s = dataclasses.replace(SETTINGS, home=tmp_path, store_dir=None)
    root = s.store / 'perps'
    root.mkdir(parents=True)
    perps_panel(n_coins=4).write_parquet(root / 'panel.parquet')
    PerpsStore(s).splits()
    out = PD.build_panel_drill(s, 'perps', 'planted', seed=3)
    home = tmp_path / 'drills' / 'perps-planted'
    secret = json.loads((home / 'drill_secret.json').read_text())
    assert secret['feature'] in PD.SPECS['perps']['features'] and out['drill'] == 'perps-planted'
    assert (home / 'store' / 'perps' / 'views' / 'dev.parquet').exists()
    sp = json.loads((home / 'store' / 'perps' / 'splits.json').read_text())
    assert sp['forward'][0] >= '2099'                                     # drills have no forward data

    # the sandbox policy for a drill run hides the secret (it lives in the drill home, outside the workspace)
    from desk.lab.sandbox import guard_policy
    ds = dataclasses.replace(s, home=home)
    pol = guard_policy(ds, home / 'workspaces' / 'a1', {})
    from desk.lab.guard import norm, under
    assert under(norm(home / 'drill_secret.json'), pol['deny_roots'])
    assert not under(norm(home / 'drill_secret.json'), pol['write_roots'] + pol['allow_roots'])

    L = Ledger(home / 'desk.sqlite3')
    oracle = ("DESK='perps'\nPARAMS={}\ndef signal(panel, p):\n"
              f"    x = pl.col('{secret['feature']}').cast(pl.Float64)\n"
              "    return panel.select(x.rank().over('ts') / pl.len().over('ts') - 0.5)[\"%s\"].fill_null(0) * 2\n"
              % secret['feature'])
    L.add_strategy('oracle', 'r1', 'perps', oracle, 'h1', '1h', 't')
    sid = L.q('SELECT id FROM strategies')[0]['id']
    L.update_strategy(sid, status='approved')
    main = Ledger(tmp_path / 'desk.sqlite3')
    rep = PD.panel_drill_report(s, 'perps', 'planted', record_in=main)
    assert rep['approved'] == 1 and rep['best_corr_with_hidden']
    board = main.get('drill_scores')
    assert 'perps-planted' in board['per_drill'] and board['desks_covered'] == ['perps']

    PD.build_panel_drill(s, 'perps', 'null', seed=4)
    nrep = PD.panel_drill_report(s, 'perps', 'null', record_in=main)
    assert nrep['false_discoveries'] == 0 and nrep['verdict'].startswith('PASS')
    board = main.get('drill_scores')
    assert board['null_false_discoveries'] == 0 and set(board['per_drill']) == {'perps-planted', 'perps-null'}
    from desk.company.rsi import compute_generation_scoreboard
    sb = compute_generation_scoreboard(main)
    assert sb['null_false_discoveries'] == 0 and sb['drill_desks_covered'] == ['perps']


def test_macro_planted_respects_execution_lag_and_feature_rows():
    rng = np.random.default_rng(0)
    days = [date(2001, 1, 1) + timedelta(days=i) for i in range(400)]
    rows = []
    for sym, tradable in (('SPY', 1), ('TLT', 1), ('GLD', 1), ('EFA', 1), ('^VIX', 0)):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        rows.append(pl.DataFrame({'date': days, 'symbol': sym, 'open': px, 'high': px * 1.01, 'low': px * 0.99,
                                  'close': px, 'adjclose': px, 'adj_open': px,
                                  'volume': rng.uniform(1e6, 2e6, len(days)), 'tradable': tradable}))
    panel = pl.concat(rows).sort(['date', 'symbol'])
    secret = {'feature': 'volume', 'window': 10, 'transform': 'log_change', 'sign': -1, 'beta': 2e-3, 'lag': 2}
    planted = PD.make_planted(panel, 'macro', secret)
    s = PD.hidden_score(planted, 'macro', secret)
    assert np.all(s[(planted['tradable'] == 0).to_numpy()] == 0)          # feature series are never planted
    vix = planted.filter(pl.col('symbol') == '^VIX')['close'].to_numpy()
    assert np.allclose(vix, panel.filter(pl.col('symbol') == '^VIX')['close'].to_numpy())
    p = planted.with_columns(pl.Series('s', s))
    p = p.with_columns((pl.col('adjclose').shift(-2).over('symbol') / pl.col('adjclose').shift(-1).over('symbol') - 1)
                       .alias('r_exec')).filter(pl.col('tradable') == 1).drop_nulls(['r_exec'])
    assert np.corrcoef(p['s'].to_numpy(), p['r_exec'].to_numpy())[0, 1] > 0.05   # tradable after the 1-day delay
