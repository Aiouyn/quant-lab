"""Statistics for honest strategy selection: PSR/DSR (Bailey & Lopez de Prado), block bootstrap, PBO."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

EULER = 0.5772156649015329


def sharpe(daily, periods=365):
    x = np.asarray(daily, float)
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return 0.0
    return float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(periods))


def max_drawdown(curve):
    c = np.asarray(curve, float)
    if len(c) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.r_[0.0, c])[1:]
    return float(np.max(peak - c))   # in units of the position unit (non-compounding)


def psr(daily, sr_benchmark=0.0):
    """Probability that the true (per-period) Sharpe exceeds the benchmark, given skew/kurtosis."""
    x = np.asarray(daily, float)
    n = len(x)
    if n < 5 or np.std(x, ddof=1) == 0:
        return 0.0
    sr = np.mean(x) / np.std(x, ddof=1)
    g3 = st.skew(x)
    g4 = st.kurtosis(x, fisher=False)
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    return float(st.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / den))


def expected_max_sharpe(n_trials, sr_std):
    """Expected maximum per-period Sharpe among n_trials unskilled strategies with Sharpe dispersion sr_std."""
    n = max(int(n_trials), 2)
    return float(sr_std * ((1 - EULER) * st.norm.ppf(1 - 1 / n) + EULER * st.norm.ppf(1 - 1 / (n * math.e))))


def dsr(daily, n_trials, trial_sr_std=None):
    """Deflated Sharpe: PSR against the expected max Sharpe of n_trials. trial_sr_std is the std of
    per-period Sharpe ratios across trials; default assumes the null std 1/sqrt(T)."""
    x = np.asarray(daily, float)
    T = max(len(x), 2)
    s = trial_sr_std if trial_sr_std is not None else 1 / math.sqrt(T)
    return psr(x, expected_max_sharpe(n_trials, s))


def block_bootstrap_lower(daily, block=5, n=2000, alpha=0.05, seed=7):
    x = np.asarray(daily, float)
    T = len(x)
    if T < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha))


def weekly_positive_fraction(days, rets):
    if not len(rets):
        return 0.0
    # Fraction of ACTIVE weeks that were positive: a week with exactly zero PnL on every day (flat by design, e.g. a
    # calendar strategy outside its window) says nothing about the edge and must not count as a losing week.
    d = np.asarray(days) // 7
    wk, act = {}, {}
    for k, r in zip(d, rets):
        wk[k] = wk.get(k, 0.0) + r
        act[k] = act.get(k, False) or r != 0
    v = np.array([wk[k] for k in wk if act[k]])
    return float(np.mean(v > 0)) if len(v) else 0.0


def top_day_share(rets, k=5):
    r = np.asarray(rets, float)
    total = r.sum()
    if total <= 0:
        return 1.0
    return float(np.sort(r)[::-1][:k].sum() / total)


def pbo_cscv(matrix, splits=10):
    """Probability of backtest overfitting via combinatorially symmetric cross-validation.
    matrix: T x N daily returns of N configurations."""
    from itertools import combinations
    M = np.asarray(matrix, float)
    T, N = M.shape
    if N < 2 or T < splits * 2:
        return float('nan')
    S = splits - splits % 2
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for c in combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in c])
        te = np.concatenate([blocks[i] for i in range(S) if i not in c])
        def sr(a):
            sd = a.std(axis=0, ddof=1)
            sd[sd == 0] = np.inf
            return a.mean(axis=0) / sd
        best = int(np.argmax(sr(M[tr])))
        rank = st.rankdata(sr(M[te]))[best] / (N + 1)
        lam.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(lam) <= 0))


def mean_ex_top(daily, k=5):
    """Mean daily return after removing the k best days (tail-aware: lumpy event payoffs fail this)."""
    import numpy as _np
    d = _np.sort(_np.asarray(daily, float))
    return float(d[:-k].mean()) if len(d) > 2 * k else 0.0


def cpcv_folds(T, n_groups=6, k_test=2, purge=5, embargo=0):
    """Yield (test_idx, train_mask) for every combination of k_test of n_groups contiguous groups; the train mask
    excludes the test bars, `purge` bars on both sides of each test group and `embargo` bars after it."""
    from itertools import combinations
    groups = np.array_split(np.arange(T), n_groups)
    for c in combinations(range(n_groups), k_test):
        test = np.concatenate([groups[i] for i in c])
        train = np.ones(T, bool)
        for i in c:
            lo, hi = groups[i][0], groups[i][-1]
            train[max(0, lo - purge):min(T, hi + 1 + purge + embargo)] = False
        yield test, train


def cpcv(matrix, n_groups=6, k_test=2, purge=5, embargo=0, periods=365):
    """Combinatorial purged cross-validation (Lopez de Prado, AFML ch.12) over a T x N matrix of daily returns of
    N configurations (column 0 = the submitted configuration). For every choice of k_test of the n_groups contiguous
    test groups: the training set drops `purge` bars on both sides of each test group and `embargo` further bars
    after it; the configuration with the best in-sample Sharpe is "selected" and scored out of sample.
    Returns PBO (share of paths whose selected config ranks at/below the OOS median) and OOS Sharpes."""
    M = np.asarray(matrix, float)
    if M.ndim == 1:
        M = M[:, None]
    T, N = M.shape
    if T < n_groups * 4 or n_groups < 2 or not 1 <= k_test < n_groups:
        return {'pbo': float('nan'), 'n_paths': 0, 'oos_sharpe_median': float('nan'),
                'base_oos_sharpe_median': float('nan'), 'n_groups': n_groups, 'k_test': k_test, 'purge': purge,
                'embargo': embargo}

    def sr(a):
        sd = a.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        return a.mean(axis=0) / sd
    lam, sel, base = [], [], []
    for test, train in cpcv_folds(T, n_groups, k_test, purge, embargo):
        if train.sum() < 8:
            continue
        oos = sr(M[test])
        best = int(np.argmax(sr(M[train])))
        sel.append(oos[best] * math.sqrt(periods))
        base.append(oos[0] * math.sqrt(periods))
        if N >= 2:
            rank = st.rankdata(oos)[best] / (N + 1)
            lam.append(math.log(rank / (1 - rank)))
    return {'pbo': float(np.mean(np.array(lam) <= 0)) if lam else float('nan'), 'n_paths': len(sel),
            'oos_sharpe_median': float(np.median(sel)) if sel else float('nan'),
            'base_oos_sharpe_median': float(np.median(base)) if base else float('nan'),
            'n_groups': n_groups, 'k_test': k_test, 'purge': purge, 'embargo': embargo}


def overfit_report(series, periods=365, n_groups=6, k_test=2, purge=5, embargo_frac=0.01, cscv_splits=10):
    """series: list of (days, rets) per configuration, the submitted one first. Aligns on the union of days (flat
    = 0), then PBO by CSCV and by CPCV with purge + embargo. pbo = the worse (max) of the two; NaN if undefined."""
    days = sorted({int(d) for ds, _ in series for d in ds})
    ix = {d: i for i, d in enumerate(days)}
    M = np.zeros((len(days), len(series)))
    for j, (ds, rs) in enumerate(series):
        for d, r in zip(ds, rs):
            M[ix[int(d)], j] = r
    T, N = M.shape
    embargo = int(math.ceil(embargo_frac * T))
    c = cpcv(M, n_groups, k_test, purge, embargo, periods)
    p1 = pbo_cscv(M, cscv_splits) if N >= 2 else float('nan')
    vals = [x for x in (p1, c['pbo']) if np.isfinite(x)]
    return {'n_configs': N, 'T': T, 'pbo_cscv': p1, 'pbo_cpcv': c['pbo'], 'pbo': max(vals) if vals else float('nan'),
            'cpcv': c}
