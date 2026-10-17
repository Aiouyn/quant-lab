"""Pre-flight pass-probability for the SEALED validation gates (desk.pipeline.validation.val_gates), estimated from DEV data only (never touches sealed splits). Structural facts it encodes: the val window is ~22 days (val_fraction 0.25), so (1) trades>=40 requires a dev trade rate >= 40/22 = 1.82/day (~98 dev trades per 54d for P(pass)>0.5 under Poisson) — sparse <1.5/day strategies are structurally doomed BEFORE any metric is looked at; (2) top5_day_share<0.5 is measured over 5/22 = 23% of days (vs 9% on 54 dev days), so concentration inflates mechanically; (3) positive_weeks>=55% is coarse at 22 days (only ~4 week buckets). Usage: from desk.lib.contrib.val_precheck import val_precheck; r = val_precheck(cost_eval(...)['daily_net_bps'], trades=n, dev_days=54); print(r['p_all'], r['warnings']). Inputs: daily net PnL in bps of unit (cost_eval daily_net_bps), dev trade count and dev day count; returns per-gate pass probabilities (p_sharpe, p_bootstrap_lb, p_pos_weeks, p_top5, p_trades) from a day-resample Monte Carlo + Poisson trade count, a joint p_all, and structural warnings. Run it BEFORE firing submit_strategy: a failing p_all means the validation will be burned on a known-structural failure."""
import numpy as np
import polars as pl
"""Pre-flight pass-probability for the SEALED validation gates (desk.pipeline.validation.val_gates),
estimated from DEV data only. See doc for structural rationale. Pure numpy/scipy, causal w.r.t. sealed
splits: uses nothing but the dev daily PnL series you already have."""

import numpy as np
from scipy import stats as st

VAL_DAYS_DEFAULT = 22  # val_fraction 0.25 of ~88-90 split days; observed val windows are 22d


def _top5_share(row, k=5):
    total = row.sum()
    if total <= 0:
        return 1.0                       # matches desk.lab.stats.top_day_share semantics
    return float(np.sort(row)[::-1][:k].sum() / total)


def _pos_weeks(wids, rets):
    """Fraction of positive week buckets, grouping by original week id (day//7)."""
    order = np.unique(wids)
    tot = 0.0
    pos = 0
    for w in order:
        tot += rets[wids == w].sum()
        pos += 1 if tot is not None and rets[wids == w].sum() > 0 else 0
    # NOTE: recompute cleanly per week (above accumulates; fix):
    sums = np.array([rets[wids == w].sum() for w in order])
    return float((sums > 0).mean()) if len(sums) else float('nan')


def _block_lb(x, boots_in=200, block=5, alpha=0.05, rng=None):
    """5th-pct block-bootstrap lower bound of the mean of series x (mirrors stats.block_bootstrap_lower)."""
    T = len(x)
    if T < 5:
        return float('nan')
    rng = rng or np.random.default_rng(7)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(boots_in, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(boots_in, -1)[:, :T]
    return float(np.quantile(x[idx].mean(axis=1), alpha))


def val_precheck(daily_bps, trades=None, dev_days=None, days=None, val_days=VAL_DAYS_DEFAULT,
                 min_val_sharpe=1.0, min_val_trades=40, max_top5=0.5, min_pos_weeks=0.55,
                 boots=1500, boots_in=200, seed=11):
    """Monte-Carlo pre-flight of the val gates on a resampled val-length window.

    daily_bps : array of DEV daily net PnL in bps of unit (cost_eval's daily_net_bps).
    trades    : dev trade (entry) count; dev_days: dev day count -> Poisson trade-count gate.
    days      : optional epoch-day ids aligned with daily_bps (for week grouping); default arange.
    Returns dict of per-gate pass probabilities + joint p_all + structural warnings."""
    x = np.asarray(daily_bps, float)
    N = len(x)
    if N < 10:
        raise ValueError('need >=10 dev days')
    wids = np.asarray(days, float) // 7 if days is not None else np.arange(N) // 7
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, N, size=(boots, val_days))
    X = x[idx]                                   # (boots, val_days)
    W = wids[idx]

    # val_sharpe >= min (annualized, ddof=1 like stats.sharpe)
    mu = X.mean(axis=1); sd = X.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mu / np.where(sd > 0, sd, 1) * np.sqrt(365.0), 0.0)
    p_sharpe = float((sharpe >= min_val_sharpe).mean())

    # bootstrap_lower_mean_bps > 0
    lbs = np.array([_block_lb(X[i], boots_in=boots_in, rng=rng) for i in range(boots)])
    p_lb = float(np.nanmean(lbs > 0))

    # top5_days < max_top5
    t5 = np.array([_top5_share(X[i]) for i in range(boots)])
    p_top5 = float((t5 < max_top5).mean())

    # positive_weeks >= min_pos_weeks (group resampled days by their ORIGINAL week id)
    pw = np.array([_pos_weeks(W[i], X[i]) for i in range(boots)])
    p_pw = float(np.nanmean(pw >= min_pos_weeks))

    p_day = float(((sharpe >= min_val_sharpe) & (lbs > 0) & (t5 < max_top5) & (pw >= min_pos_weeks)).mean())

    out = {
        'val_days': val_days, 'n_dev_days': N, 'boots': boots,
        'p_sharpe': round(p_sharpe, 3), 'p_bootstrap_lb': round(p_lb, 3),
        'p_top5': round(p_top5, 3), 'p_pos_weeks': round(p_pw, 3),
        'p_day_gates': round(p_day, 3),
        'est_top5_val_median': round(float(np.median(t5)), 3),
        'est_top5_dev': round(_top5_share(x), 3),
    }
    warn = []
    if trades is not None and dev_days:
        lam = trades / dev_days * val_days
        p_tr = float(st.poisson.sf(min_val_trades - 1, lam))
        out['p_trades'] = round(p_tr, 3)
        out['est_val_trades'] = round(lam, 1)
        out['min_trade_rate_per_day'] = round(min_val_trades / val_days, 2)
        out['dev_trade_rate'] = round(trades / dev_days, 2)
        if p_tr < 0.8:
            need = min(t for t in range(trades, 4000)
                       if st.poisson.sf(min_val_trades - 1, t / dev_days * val_days) >= 0.5)
            warn.append(f'TRADES GATE structural risk: dev rate {trades/dev_days:.2f}/d expects '
                        f'{lam:.0f} val trades (need >= {min_val_trades}); P(pass)={p_tr:.2f}. '
                        f'Need >= {need} dev trades over {dev_days}d (rate {min_val_trades/val_days:.2f}/d+) '
                        f'for P>0.5, ~+20% more for P>0.8.')
        out['p_all'] = round(p_day * (out.get('p_trades', 1.0)), 3)
    else:
        out['p_all'] = round(p_day, 3)
    if out['est_top5_val_median'] >= max_top5:
        warn.append(f'TOP5 GATE structural risk: dev top5 share {out["est_top5_dev"]:.2f} inflates to '
                    f'~{out["est_top5_val_median"]:.2f} at N={val_days} (top-5 = {100*5/val_days:.0f}% of window '
                    f'vs {100*5/N:.0f}% on dev); P(pass)={p_top5:.2f}. Cure = more/dispersed trades, not looser dev gate.')
    if out['p_pos_weeks'] < 0.8:
        warn.append(f'POS_WEEKS coarse at N={val_days} (~{int(np.ceil(val_days/7))} week buckets): P(pass)={p_pw:.2f}.')
    out['warnings'] = warn
    return out
