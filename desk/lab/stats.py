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
