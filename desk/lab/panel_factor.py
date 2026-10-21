"""Factor evaluation for the panel desks (PERPS hourly, MACRO daily), used by register_factor for modules with
DESK = 'perps' | 'macro' and `def factor(panel, params) -> array aligned to panel rows`.

On the desk's DEV panel only:
  * causality: prefix-invariance (values must not change when later rows are removed)
  * per horizon h: cross-sectional Spearman rank IC between the factor and the forward return over (t+lag, t+lag+h]
    (perps lag 0 with px_fill = execution price; macro lag 1 day = execution at the next close), averaged over
    non-overlapping periods with a t-stat; plus the MEAN top-minus-bottom quintile return per period, its median and
    the mean EXCLUDING the 5 best periods (lottery-skew trap: rank IC can be strong while the mean spread is not)
  * cost hurdle: spread per period vs a round-trip cost (perps ~2 x (4.5 bps + half impact spread); macro 2 x cost_bps)
  * novelty: average cross-sectional rank correlation with every active library factor of the same desk
Values are cached to artifacts/factor_values/<id>.parquet for later novelty checks and combinations.
"""
from __future__ import annotations

import numpy as np
import polars as pl

DESK_CFG = {
    'perps': {'t': 'ts', 'k': 'coin', 'px': 'px_fill', 'horizons': [1, 4, 24, 72, 168], 'unit': 'h', 'lag': 0},
    'macro': {'t': 'date', 'k': 'symbol', 'px': 'adjclose', 'horizons': [1, 5, 21, 63], 'unit': 'd', 'lag': 1},
    # prediction markets: YES price returns (a NO position earns the mirror; event_study side=-1 approximates it)
    'pred': {'t': 'ts', 'k': 'market', 'px': 'p', 'horizons': [1, 4, 24, 72], 'unit': 'h', 'lag': 1},
}


def load_factor(code):
    """Agent-authored module: executed in the OS jail (factor_jail), never exec'd in this process."""
    from .factor_jail import load
    return load(code)


def values(mod, panel):
    v = mod.factor(panel, dict(mod.PARAMS))
    if isinstance(v, pl.Series):
        v = v.to_numpy()
    v = np.asarray(v, dtype=float).reshape(-1)
    if len(v) != len(panel):
        raise ValueError(f'factor length {len(v)} != panel rows {len(panel)}')
    return v


def causality(mod, panel, tcol, cuts=(0.5, 0.75)):
    full = values(mod, panel)
    ts = panel[tcol].to_numpy()
    ut = np.unique(ts)
    for c in cuts:
        cut = ut[int(len(ut) * c)]
        k = int(np.sum(ts <= cut))
        pre = values(mod, panel.head(k))
        a, b = pre, full[:k]
        bad = np.sum(~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9 * (1 + np.abs(b)))))
        if bad:
            return {'ok': False, 'issue': f'{int(bad)} values changed when data after {cut} was removed (future data use)'}
    return {'ok': True}


def horizon_report(panel, v, desk):
    """Cross-sectional IC / mean quintile spreads / cost hurdle for values v aligned to `panel` rows."""
    cfg = DESK_CFG[desk]
    t, k, px = cfg['t'], cfg['k'], cfg['px']
    df = panel.select(t, k, px, *([pl.col('isp')] if 'isp' in panel.columns else []),
                      *([pl.col('cost_bps')] if 'cost_bps' in panel.columns else [])).with_columns(pl.Series('v', np.asarray(v, float)))
    df = df.sort([t, k])
    if desk == 'perps':
        cost_bps = 2 * (4.5 + 0.5 * float(np.nanmedian(np.minimum(df['isp'].to_numpy(), 50))))
    elif desk == 'pred':                                  # 1-cent half-spread on a typical TRADABLE price, both ways
        mid = df.filter((pl.col(px) > 0.05) & (pl.col(px) < 0.95))[px]
        cost_bps = 2 * 0.01 / (float(mid.median()) if len(mid) else 0.5) * 1e4
    else:
        cost_bps = 2 * float(df['cost_bps'].median())
    out = {'desk': desk, 'round_trip_cost_bps': round(cost_bps, 2)}
    hz = {}
    best = None
    for h in cfg['horizons']:
        lag = cfg['lag']
        d = df.with_columns((pl.col(px).shift(-(lag + h)).over(k) / pl.col(px).shift(-lag).over(k) - 1).alias('fwd'))
        d = d.filter(pl.col('v').is_finite() & pl.col('fwd').is_finite())
        times = np.sort(d[t].unique().to_numpy())
        keep = set(times[::h].tolist())                                  # non-overlapping periods
        d = d.filter(pl.col(t).is_in(list(keep)))
        g = d.with_columns(pl.col('v').rank().over(t).alias('rv'), pl.col('fwd').rank().over(t).alias('rf'),
                           pl.len().over(t).alias('n'), (pl.col('fwd') - pl.col('fwd').mean().over(t)).alias('xf'))
        g = g.filter(pl.col('n') >= 8)
        g = g.with_columns((pl.col('rv') / pl.col('n')).alias('q'))
        per = g.group_by(t).agg(pl.corr('rv', 'rf').alias('ic'),
                                (pl.col('xf').filter(pl.col('q') > 0.8).mean() - pl.col('xf').filter(pl.col('q') <= 0.2).mean()).alias('spread'))
        ic = per['ic'].drop_nulls().drop_nans().to_numpy()
        sp = per['spread'].drop_nulls().drop_nans().to_numpy()
        if len(ic) < 20:
            continue
        srt = np.sort(sp)
        r = {'periods': int(len(ic)), 'rank_ic': round(float(ic.mean()), 4),
             'ic_t': round(float(ic.mean() / (ic.std(ddof=1) + 1e-12) * np.sqrt(len(ic))), 2),
             'pct_periods_ic_positive': round(float(np.mean(ic > 0)), 3),
             'q5_q1_mean_bps': round(float(sp.mean() * 1e4), 2), 'q5_q1_median_bps': round(float(np.median(sp) * 1e4), 2),
             'q5_q1_mean_ex_top5_bps': round(float((srt[:-5] if np.mean(sp) >= 0 else srt[5:]).mean() * 1e4), 2),
             'spread_vs_cost': round(float(abs(sp.mean()) * 1e4 / max(cost_bps, 1e-9)), 3)}
        r['mean_agrees_with_ic'] = bool(np.sign(r['q5_q1_mean_bps']) == np.sign(r['rank_ic']) and
                                        np.sign(r['q5_q1_mean_ex_top5_bps']) == np.sign(r['rank_ic']))
        hz[f'{h}{cfg["unit"]}'] = r
        if r['mean_agrees_with_ic'] and (best is None or abs(r['ic_t']) > abs(best[1]['ic_t'])):
            best = (f'{h}{cfg["unit"]}', r)
    out['horizons'] = hz
    out['best_horizon'] = best[0] if best else None
    out['best_t'] = best[1]['ic_t'] if best else 0.0
    out['best_spread_vs_cost'] = best[1]['spread_vs_cost'] if best else 0.0
    out['economic_value'] = round(float(abs(out['best_t']) * min(out['best_spread_vs_cost'], 3.0)), 3)
    out['_df'] = df
    return out


def evaluate(code, panel, desk, library=None):
    cfg = DESK_CFG[desk]
    t, k = cfg['t'], cfg['k']
    if desk == 'macro':
        panel = panel.filter(pl.col('tradable') == 1)
    panel = panel.sort([t, k])
    mod = load_factor(code)
    v = values(mod, panel)
    rep = horizon_report(panel, v, desk)
    df = rep.pop('_df')
    out = {'desk': desk, 'rows': len(panel), 'coverage': float(np.mean(np.isfinite(v))), 'causality': causality(mod, panel, t)}
    out['causal'] = out['causality']['ok']
    out.update(rep)
    nov = {}
    for fid, lv in (library or {}).items():
        j = df.select(t, k, 'v').join(lv, on=[t, k], how='inner').filter(pl.col('v').is_finite() & pl.col('lv').is_finite())
        if len(j) < 1000:
            continue
        c = j.with_columns(pl.col('v').rank().over(t).alias('a'), pl.col('lv').rank().over(t).alias('b')) \
             .group_by(t).agg(pl.corr('a', 'b').alias('c'))['c'].drop_nulls().drop_nans()
        if len(c):
            nov[fid] = round(float(c.mean()), 3)
    out['novelty'] = {'max_abs_corr': max((abs(x) for x in nov.values()), default=0.0),
                      'most_similar': max(nov, key=lambda f: abs(nov[f])) if nov else None, 'corr': nov}
    out['_values'] = df.select(t, k, pl.col('v').alias('lv'))
    return out
