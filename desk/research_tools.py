"""Research primitives for agent sandboxes (re-exported by desk.research):

event_study(panel, mask, horizons=None, side=1)  -> per horizon: n events, mean / median / mean excluding the 5 best
                                                    events (bps), win rate, t-stat, distinct periods. Forward returns use
                                                    the desk execution price (perps px_fill; macro next close).
xs_factor_report(panel, values, desk=None)       -> cross-sectional IC by horizon + MEAN quintile spreads (also without
                                                    the 5 best periods) + cost hurdle: the report register_factor
                                                    produces, for fast iteration (does not register, no causality check).
walk_forward(panel, features, (target, h), fit, predict, refit_every=None, min_train=None)
                                                 -> out-of-sample predictions: at each refit time the model is fit ONLY
                                                    on rows whose h-period target was fully realized before that time.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .lab.panel_factor import DESK_CFG, horizon_report


def _desk(panel, desk=None):
    if desk:
        return desk
    if 'market' in panel.columns:
        return 'pred'
    return 'perps' if 'coin' in panel.columns else 'macro'


def event_study(panel, mask, horizons=None, side=1.0, desk=None):
    desk = _desk(panel, desk)
    cfg = DESK_CFG[desk]
    t, k, px, lag = cfg['t'], cfg['k'], cfg['px'], cfg['lag']
    sides = np.broadcast_to(np.asarray(side, float), (len(panel),)).copy()
    p = panel.with_columns(pl.Series('_m', np.asarray(mask, bool)), pl.Series('_s', sides))
    out = {}
    for h in horizons or cfg['horizons']:
        x = p.with_columns((pl.col(px).shift(-(lag + h)).over(k) / pl.col(px).shift(-lag).over(k) - 1).alias('fwd'))
        e = x.filter(pl.col('_m') & pl.col('fwd').is_finite())
        key = f'{h}{cfg["unit"]}'
        if len(e) < 5:
            out[key] = {'n': len(e)}
            continue
        r = (e['fwd'] * e['_s']).to_numpy() * 1e4
        srt = np.sort(r)
        out[key] = {'n': int(len(r)), 'mean_bps': round(float(r.mean()), 2), 'median_bps': round(float(np.median(r)), 2),
                    'mean_ex_best5_bps': round(float(srt[:-5].mean()), 2) if len(r) > 10 else None,
                    'win_rate': round(float(np.mean(r > 0)), 3),
                    't': round(float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(len(r))), 2),
                    'distinct_periods': int(e[t].n_unique())}
    return out


def xs_factor_report(panel, values, desk=None):
    desk = _desk(panel, desk)
    vals = np.asarray(values, float)
    if desk == 'macro' and 'tradable' in panel.columns:
        keep = (panel['tradable'] == 1).to_numpy()
        panel, vals = panel.filter(pl.col('tradable') == 1), vals[keep]
    rep = horizon_report(panel, vals, desk)
    rep.pop('_df', None)
    return rep


def walk_forward(panel, features, target, fit, predict, refit_every=None, min_train=None, desk=None):
    desk = _desk(panel, desk)
    t = DESK_CFG[desk]['t']
    col, th = target if isinstance(target, tuple) else (target, 1)
    times = np.sort(panel[t].unique().to_numpy())
    refit_every = refit_every or (24 * 7 if desk == 'perps' else 21)
    min_train = min_train or (24 * 90 if desk == 'perps' else 252)
    pred = np.full(len(panel), np.nan)
    X = panel.select(features).to_numpy().astype(float)
    y = panel[col].to_numpy().astype(float)
    tv = panel[t].to_numpy()
    okx = np.all(np.isfinite(X), axis=1)
    for j in range(min_train, len(times), refit_every):
        cut = times[j]
        train_end = times[max(j - th, 0)]                       # targets realized strictly before the refit time
        tr = (tv < train_end) & np.isfinite(y) & okx
        if tr.sum() < 100:
            continue
        model = fit(X[tr], y[tr])
        te = (tv >= cut) & ((tv < times[j + refit_every]) if j + refit_every < len(times) else True) & okx
        if te.any():
            pred[te] = predict(model, X[te])
    return pred


_EXAMPLES = {
    'macro': '''DESK = 'macro'
WARMUP_DAYS = 300
PARAMS = {'lb': 252, 'vol_lb': 60, 'target_vol': 0.10}

def signal(panel, params):
    # trend sign x inverse volatility, tradable rows only; one weight per panel row, same row order
    x = panel.with_columns((pl.col('adjclose') / pl.col('adjclose').shift(params['lb']).over('symbol') - 1).alias('mom'),
                           pl.col('adjclose').pct_change().over('symbol').alias('r'))
    x = x.with_columns((pl.col('r').rolling_std(params['vol_lb']).over('symbol') * 252 ** 0.5).alias('vol'))
    # scale by the number of assets trading THAT day (a whole-panel count would leak future listings)
    n_today = (pl.col('tradable') == 1).sum().over('date')
    w = (pl.col('mom').sign() * params['target_vol'] / pl.col('vol') / n_today.cast(pl.Float64).sqrt()).clip(-1, 1)
    return x.select(pl.when(pl.col('tradable') == 1).then(w).otherwise(0.0).fill_null(0.0).fill_nan(0.0).alias('w'))['w'].to_numpy()
''',
    'perps': '''DESK = 'perps'
WARMUP_DAYS = 30
PERPS_EXECUTION = 'taker'          # or 'maker' (limit orders filled only on trade-through)
PARAMS = {'lb_h': 24 * 14, 'min_vlm': 2e7}

def signal(panel, params):
    # 2-week trend on liquid coins, equal weight (the engine caps gross at 1)
    x = panel.with_columns((pl.col('px') / pl.col('px').shift(params['lb_h']).over('coin') - 1).alias('mom'))
    w = pl.when(pl.col('vlm') >= params['min_vlm']).then(pl.col('mom').sign() * 0.05).otherwise(0.0)
    return x.select(w.fill_null(0.0).alias('w'))['w'].to_numpy()
''',
    'pred': '''DESK = 'pred'
PARAMS = {'lo': 0.05, 'hi': 0.15, 'min_h': 72}

def signal(panel, params):
    # positions are HELD as shares while the target is unchanged: emit a STABLE target. Here: once a market first
    # trades in the band far from its end, hold NO on it until resolution (a favorite-longshot probe).
    x = panel.with_columns(((pl.col('p') >= params['lo']) & (pl.col('p') <= params['hi'])
                            & (pl.col('hours_to_end') >= params['min_h'])).alias('trig'))
    x = x.with_columns(pl.col('trig').cast(pl.Int32).cum_max().over('market').alias('held'))
    return x.select(pl.when(pl.col('held') == 1).then(-0.01).otherwise(0.0).alias('w'))['w'].to_numpy()
''',
}


def example_strategy(desk):
    """A minimal WORKING strategy module for a desk ('macro' | 'perps' | 'pred'): copy and modify it. It follows the
    contract (one weight per panel row, same row order, causal per-asset .over())."""
    return _EXAMPLES[desk]


def source(name):
    """Source code of a desk module, e.g. source('macro'), source('perps'), source('pred'), source('research'),
    source('panel_factor'), source('availability'). Use it instead of guessing repository paths."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    cand = {'research': root / 'research.py', 'research_tools': root / 'research_tools.py',
            'availability': root / 'data' / 'availability.py'}
    p = cand.get(name) or root / 'lab' / f'{name}.py'
    if not p.exists():
        p = root / 'data' / f'{name}.py'
    if not p.exists():
        raise KeyError(f'no module {name!r}; try macro, perps, pred, carry, panel_factor, research, availability')
    return p.read_text(encoding='utf-8')


_FACTORS = {
    'macro': """DESK = 'macro'
PARAMS = {'lb': 252, 'skip': 21}

def factor(panel, params):
    # 12-1 month momentum per symbol: ONE value per panel row, computed with expressions over the panel itself
    e = pl.col('adjclose').shift(params['skip']).over('symbol') / pl.col('adjclose').shift(params['lb']).over('symbol') - 1
    return panel.select(e.alias('v'))['v'].to_numpy()
""",
    'perps': """DESK = 'perps'
PARAMS = {'lb_h': 168}

def factor(panel, params):
    # 1-week funding level (crowding) per coin
    e = pl.col('f').rolling_mean(params['lb_h']).over('coin')
    return panel.select(e.alias('v'))['v'].to_numpy()
""",
    'pred': """DESK = 'pred'
PARAMS = {'lb': 24}

def factor(panel, params):
    # 24-hour YES price change per market
    e = pl.col('p') - pl.col('p').shift(params['lb']).over('market')
    return panel.select(e.alias('v'))['v'].to_numpy()
""",
}


def example_factor(desk):
    """A minimal WORKING factor module for register_factor on a panel desk ('macro' | 'perps' | 'pred')."""
    return _FACTORS[desk]
