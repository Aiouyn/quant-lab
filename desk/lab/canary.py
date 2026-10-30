"""Planted leaky canary (cycle 1b): a strategy that trades on the very return it will be paid (the next bar's
return after its one-close execution delay). It would sail through every statistical gate; the harness must
reject it on every run. `harness.strategy_eval` calls `canary_selfcheck()` before evaluating anything and refuses
to evaluate if the canary is not rejected."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

CANARY_MACRO = '''DESK = 'macro'
PARAMS = {}


def signal(panel, params):
    paid = pl.col('adjclose').shift(-2).over('symbol') / pl.col('adjclose').shift(-1).over('symbol') - 1
    return panel.select(paid.sign().fill_null(0).alias('w'))['w'].to_numpy()
'''


class HarnessCompromised(RuntimeError):
    """The look-ahead canary was not rejected: the harness refuses to evaluate anything."""


def synthetic_macro_panel(n_days=120, n_sym=4, seed=0):
    rng = np.random.default_rng(seed)
    days, d = [], date(2001, 1, 1)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for k in range(n_sym):
        px = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n_days)))
        rows += [{'date': day, 'symbol': f'C{k}', 'adjclose': float(px[i]), 'close': float(px[i]), 'cls': 'equity_us',
                  'tradable': 1, 'cost_bps': 1.0} for i, day in enumerate(days)]
    return pl.DataFrame(rows).sort(['date', 'symbol'])


def canary_selfcheck():
    """Run the canary through the same as-of guard every evaluation uses; it must raise LookAheadError."""
    from .asof import LookAheadError, asof_guard
    from .macro import macro_signal
    from .strategy import load_strategy
    mod = load_strategy(CANARY_MACRO)
    panel = synthetic_macro_panel()
    try:
        asof_guard(lambda v: macro_signal(mod, v), panel, 'date', n_cuts=4, label='canary')
    except LookAheadError:
        return True
    raise HarnessCompromised('look-ahead canary was NOT rejected; refusing to evaluate strategies')
