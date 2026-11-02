"""Drill markets for the panel desks (perps hourly, macro daily): measure the research firm itself.

    planted : the desk's real panel, with a hidden, causal, tradable effect injected into prices. Every asset's
              next-period return gets beta * s_{t-lag}, where s is a cross-sectional score built from ONE visible
              non-price column. The column, window, transform and sign are drawn at random when the drill is built and
              stored only in the drill home (agent sandboxes cannot read it), so knowing this file does not reveal the
              answer. A capable firm finds the effect and gets a strategy on it through validation and holdout.
    null    : the desk's real panel with every asset's return series shuffled in blocks (a day of hours for perps, a
              week for macro), so nothing predicts returns beyond one block. Funding is set to zero on perps (otherwise
              its real persistence is a genuine yield). Every approval here is a false discovery.

Build:  python -m desk drill planted --desk perps      Run: python -m desk run --drill perps-planted
Score:  python -m desk drill planted --desk perps --report   (also recorded in the main ledger's drill scoreboard)
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

SPECS = {
    'perps': {
        't': 'ts', 'k': 'coin', 'ret_px': 'px_fill', 'lag': 1, 'block': 24, 'beta': 2.0e-4,
        'price_cols': ('px', 'px_fill', 'oracle', 'mark', 'lo', 'hi'),
        'features': ('premium', 'oi', 'vlm', 'isp', 'bn_oi_value', 'bn_ls_accounts', 'bn_ls_top_positions',
                     'bn_taker_ratio'),
        'windows': (12, 24, 48, 96), 'zero_cols': ('f',),
    },
    'macro': {
        't': 'date', 'k': 'symbol', 'ret_px': 'adjclose', 'lag': 2, 'block': 5, 'beta': 8.0e-4,
        'price_cols': ('open', 'high', 'low', 'close', 'adjclose', 'adj_open'),
        'features': ('volume', '_range', '_gap'), 'windows': (5, 10, 20, 60), 'zero_cols': (),
    },
}
DRILL_DESKS = tuple(SPECS)


def drill_name(desk, kind):
    return kind if desk == 'l2' else f'{desk}-{kind}'


def _store_cls(desk):
    if desk == 'perps':
        from .perps import PerpsStore, write_dev_view
        return PerpsStore, write_dev_view
    from .macro import MacroStore, write_dev_view
    return MacroStore, write_dev_view


def _with_derived(panel, desk):
    if desk != 'macro':
        return panel
    k = 'symbol'
    return panel.with_columns(((pl.col('high') - pl.col('low')) / pl.col('close')).alias('_range'),
                              (pl.col('open') / pl.col('close').shift(1).over(k) - 1).alias('_gap'))


def hidden_score(panel: pl.DataFrame, desk: str, secret: dict) -> np.ndarray:
    """Cross-sectional score s in (-1, 1) per row, causal (uses rows <= t of the chosen feature only)."""
    sp = SPECS[desk]
    t, k, col, w = sp['t'], sp['k'], secret['feature'], int(secret['window'])
    p = _with_derived(panel, desk).with_row_index('_i')
    x = pl.col(col).cast(pl.Float64)
    if secret['transform'] == 'log_change':
        x = (x.abs() + 1e-9).log() - (x.abs() + 1e-9).log().shift(w).over(k)
    feat = x.rolling_mean(w, min_samples=max(2, w // 2)).over(k) if secret['transform'] == 'level' else x
    p = p.with_columns(feat.alias('_f'))
    p = p.with_columns(((pl.col('_f') - pl.col('_f').mean().over(t)) / (pl.col('_f').std().over(t) + 1e-12))
                       .fill_null(0).fill_nan(0).alias('_z'))
    if desk == 'macro' and 'tradable' in p.columns:
        p = p.with_columns(pl.when(pl.col('tradable') == 1).then(pl.col('_z')).otherwise(0.0).alias('_z'))
    s = np.tanh(p.sort('_i')['_z'].to_numpy()) * float(secret['sign'])
    return s


def _draw_secret(panel, desk, rng):
    sp = SPECS[desk]
    p = _with_derived(panel, desk)
    usable = [c for c in sp['features'] if c in p.columns and p[c].null_count() < 0.5 * len(p)
              and p[c].cast(pl.Float64).std() not in (None, 0.0)]
    if not usable:
        raise SystemExit(f'no usable feature column for a {desk} planted drill')
    return {'feature': str(rng.choice(usable)), 'window': int(rng.choice(sp['windows'])),
            'transform': str(rng.choice(['level', 'log_change'])), 'sign': int(rng.choice([-1, 1])),
            'beta': sp['beta'], 'lag': sp['lag']}


def _reprice(panel, desk, new_log_ret):
    """Replace each asset's return path by `new_log_ret` (per row, same order), scaling every price column."""
    sp = SPECS[desk]
    k, px = sp['k'], sp['ret_px']
    p = panel.with_columns(pl.Series('_nr', new_log_ret).fill_nan(0.0))
    p = p.with_columns(pl.col(px).forward_fill().over(k).alias('_p0'))
    first = pl.col('_p0').drop_nulls().first().over(k)
    p = p.with_columns((first * pl.col('_nr').cum_sum().over(k).exp() / pl.col('_p0')).alias('_ratio'))
    cols = [(pl.col(c) * pl.col('_ratio')).alias(c) for c in sp['price_cols'] if c in p.columns]
    return p.with_columns(cols).drop('_nr', '_p0', '_ratio')


def _log_ret(panel, desk):
    sp = SPECS[desk]
    k, px = sp['k'], sp['ret_px']
    r = (pl.col(px).forward_fill().over(k).log() - pl.col(px).forward_fill().over(k).log().shift(1).over(k))
    return panel.select(r.fill_null(0.0).fill_nan(0.0).alias('r'))['r'].to_numpy()


def make_planted(panel, desk, secret):
    s = hidden_score(panel, desk, secret)
    k = SPECS[desk]['k']
    lagged = (pl.DataFrame({'k': panel[k], 's': s}).select(pl.col('s').shift(int(secret['lag'])).over('k'))['s']
              .fill_null(0.0).to_numpy())
    return _reprice(panel, desk, _log_ret(panel, desk) + float(secret['beta']) * lagged)


def make_null(panel, desk, rng):
    sp = SPECS[desk]
    k, B = sp['k'], sp['block']
    r = _log_ret(panel, desk)
    keys = panel[k].to_numpy()
    out = np.zeros_like(r)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        rr = r[idx]
        n = len(rr) // B * B
        if n >= 2 * B:
            blocks = rr[:n].reshape(-1, B)
            rr = np.r_[blocks[rng.permutation(len(blocks))].ravel(), rr[n:]]
        out[idx] = rr
    p = _reprice(panel, desk, out)
    zero = [pl.lit(0.0).alias(c) for c in sp['zero_cols'] if c in p.columns]
    return p.with_columns(zero) if zero else p


def build_panel_drill(settings, desk, kind, seed=None):
    import dataclasses
    if desk not in SPECS:
        raise SystemExit(f'panel drills exist for {DRILL_DESKS}; l2 uses `python -m desk drill {kind}`')
    Store, write_view = _store_cls(desk)
    src = Store(settings, allowed=('dev', 'val', 'holdout'))
    sp = src.splits()
    hi = max(src.range_ms(x)[1] for x in ('holdout',)) if desk == 'perps' else src.range_dates('holdout')[1]
    t = SPECS[desk]['t']
    panel = src.panel().filter(pl.col(t) < hi if desk == 'perps' else pl.col(t) <= hi).sort([t, SPECS[desk]['k']])
    rng = np.random.default_rng(seed if seed is not None else int(time.time()))
    home = settings.home / 'drills' / drill_name(desk, kind)
    if home.exists():
        shutil.rmtree(home)
    root = home / 'store' / desk
    root.mkdir(parents=True)
    if kind == 'planted':
        secret = _draw_secret(panel, desk, rng)
        out = make_planted(panel, desk, secret)
    else:
        secret = {'kind': 'null', 'block': SPECS[desk]['block']}
        out = make_null(panel, desk, rng)
    out.write_parquet(root / 'panel.parquet', compression='zstd')
    (root / 'splits.json').write_text(json.dumps({**{x: sp[x] for x in ('dev', 'val', 'holdout')},
                                                  'forward': ['2099-01-01', '2099-12-31'], 'drill': kind}))
    (home / 'drill_secret.json').write_text(json.dumps({'desk': desk, 'kind': kind, 'built': time.time(), **secret}))
    write_view(dataclasses.replace(settings, home=home, store_dir=None))
    return {'drill': drill_name(desk, kind), 'home': str(home), 'rows': len(out),
            'run': f'python -m desk run --drill {drill_name(desk, kind)} --agents 12 --hours 2'}


def _desk_strategies(L, desk):
    import re
    pat = re.compile(r"^DESK\s*=\s*['\"]%s['\"]" % desk, re.M)
    return [r for r in L.q('SELECT id, status, code FROM strategies') if pat.search(r['code'] or '')]


def panel_drill_report(settings, desk, kind, record_in=None):
    """Score a finished panel drill run. planted: did approved strategies trade the hidden score? null: how many
    false discoveries? With record_in (the MAIN ledger), the verdict goes into its drill scoreboard."""
    import dataclasses
    from ..company.ledger import Ledger
    home = settings.home / 'drills' / drill_name(desk, kind)
    L = Ledger(home / 'desk.sqlite3')
    secret = json.loads((home / 'drill_secret.json').read_text())
    strats = _desk_strategies(L, desk)
    st: dict = {}
    for r in strats:
        st[r['status']] = st.get(r['status'], 0) + 1
    approved = [r for r in strats if r['status'] in ('approved', 'paper')]
    out = {'drill': drill_name(desk, kind), 'desk': desk, 'kind': kind, 'strategies': st, 'approved': len(approved),
           'scored_at': time.time()}
    if kind == 'null':
        out['false_discoveries'] = len(approved)
        out['passed_validation_falsely'] = len(approved) + st.get('rejected_holdout', 0)
        out['verdict'] = 'PASS (no false discoveries)' if not approved else 'FAIL (false discoveries approved)'
    else:
        Store, _ = _store_cls(desk)
        s2 = dataclasses.replace(settings, home=home, store_dir=None)
        panel, a, b = Store(s2, allowed=('dev',)).load('dev', warmup_days=0)
        hid = hidden_score(panel, desk, secret)
        from .strategy import load_strategy
        sig_fn = __import__(f'desk.lab.{desk}', fromlist=['x']).__dict__[f'{desk}_signal']
        corrs = []
        for r in (approved or [x for x in strats if x['status'] not in ('error',)])[:20]:
            try:
                w = sig_fn(load_strategy(r['code']), panel)
                c = np.corrcoef(np.nan_to_num(w), hid)[0, 1]
                corrs.append((round(abs(float(c)), 3), r['id'], r['status']))
            except Exception:
                continue
        corrs.sort(reverse=True)
        out['best_corr_with_hidden'] = corrs[:5]
        best_approved = max((c for c, _, s in corrs if s in ('approved', 'paper')), default=0.0)
        out['planted_signal_corr'] = best_approved
        out['verdict'] = ('PASS (discovered and approved)' if best_approved >= 0.3 else
                          'PARTIAL (found, not approved)' if corrs and corrs[0][0] >= 0.3 else 'FAIL (not discovered)')
    if record_in is not None:
        record_drill_score(record_in, out)
    return out


def record_drill_score(ledger, result: dict):
    """Keep the latest result per drill in the main ledger and roll them up for the RSI scoreboard."""
    scores = ledger.get('drill_scores', None) or {}
    per = dict(scores.get('per_drill') or {})
    per[result['drill']] = {k: result.get(k) for k in ('desk', 'kind', 'verdict', 'approved', 'false_discoveries',
                                                       'planted_signal_corr', 'scored_at')}
    nulls = [v for v in per.values() if v.get('kind') == 'null']
    planted = [v for v in per.values() if v.get('kind') == 'planted']
    roll = {
        'per_drill': per,
        'null_false_discoveries': sum(int(v.get('false_discoveries') or 0) for v in nulls) if nulls else None,
        'planted_signal_corr': (round(float(np.mean([v.get('planted_signal_corr') or 0.0 for v in planted])), 3)
                                if planted else None),
        'verdict': ('PASS' if per and all(str(v.get('verdict', '')).startswith('PASS') for v in per.values())
                    else 'FAIL' if any(str(v.get('verdict', '')).startswith('FAIL') for v in per.values())
                    else 'PARTIAL'),
        'desks_covered': sorted({v.get('desk') for v in per.values()}),
    }
    ledger.set('drill_scores', roll)
    return roll
