"""Synthetic macro store for harness tests (random-walk prices, sealed splits). No real data is touched."""
from datetime import date, timedelta

import numpy as np
import polars as pl

from desk.config import Settings
from desk.data.provenance import write_splits

SPLITS = {'dev': ['2010-01-01', '2011-12-31'], 'val': ['2012-01-01', '2012-12-31'],
          'holdout': ['2026-11-02', '2027-04-30'], 'forward': ['2027-05-01', '2099-12-31']}

TREND = '''DESK = 'macro'
PARAMS = {'look': 20}


def signal(panel, params):
    r = pl.col('adjclose') / pl.col('adjclose').shift(params['look']).over('symbol') - 1
    return panel.select(r.sign().fill_null(0).alias('w'))['w'].to_numpy()
'''


def macro_panel(n_days=780, n_sym=6, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    days, d = [], date(2009, 6, 1)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for k in range(n_sym):
        px = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n_days)))
        for i, day in enumerate(days):
            rows.append({'date': day, 'symbol': f'S{k}', 'adjclose': float(px[i]), 'close': float(px[i]),
                         'cls': 'equity_us', 'tradable': 1, 'cost_bps': 1.0})
    return pl.DataFrame(rows).sort(['date', 'symbol'])


def macro_settings(tmp_path, **kw):
    s = Settings(home=tmp_path / 'home', store_dir=tmp_path / 'store')
    root = s.store / 'macro'
    root.mkdir(parents=True, exist_ok=True)
    macro_panel(**kw).write_parquet(root / 'panel.parquet')
    write_splits(root / 'splits.json', SPLITS)
    return s


def job(s, code=TREND, split='dev', **kw):
    j = {'type': 'strategy_eval', 'code': code, 'split': split,
         'allowed': ['dev'] if split == 'dev' else ['dev', 'val'], '_home': str(s.home), '_store': str(s.store)}
    j.update(kw)
    return j
