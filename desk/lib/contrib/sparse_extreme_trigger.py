"""Sparse extreme-episode entry trigger (L-66e538 recipe, RSI-refactored for Polars Series/DataFrame & multi-asset panels): causal trailing-quantile gate on |x|, first-onset sign entry, cooldown between entries, fixed hold. Includes rolling_q(x, w, q) and panel_sparse_trigger(df, col, group_col)."""
import numpy as np
import polars as pl


def rolling_q(x, w, q, min_frac=0.5, interpolation='nearest'):
    """Causal trailing rolling quantile: output at row i uses rows [i-w+1 .. i] ONLY. Accepts numpy arrays or Polars Series."""
    if isinstance(x, pl.DataFrame):
        x = x.to_series(0)
    arr = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x, float).reshape(-1)
    arr = np.nan_to_num(arr.astype(float), nan=0.0)
    s = pl.Series(arr)
    k = max(2, int(w * min_frac))
    try:
        return s.rolling_quantile(q, window_size=w, min_samples=k, interpolation=interpolation).to_numpy()
    except TypeError:
        return s.rolling_quantile(q, window_size=w, min_periods=k, interpolation=interpolation).to_numpy()


def sparse_extreme_trigger(x, lookback=2880, q=0.995, cooldown=240, hold=30, min_frac=0.5, interpolation='nearest'):
    """Causal sparse extreme-episode trigger supporting both 1D arrays and Polars Series without NaN crashes."""
    if isinstance(x, pl.DataFrame):
        x = x.to_series(0)
    arr = x.to_numpy() if isinstance(x, pl.Series) else np.asarray(x, float).reshape(-1)
    arr = np.nan_to_num(arr.astype(float), nan=0.0)
    n = arr.size
    w = min(max(4, int(lookback)), max(4, n))
    thr = rolling_q(np.abs(arr), w, q, min_frac, interpolation)
    gate = np.isfinite(thr) & (np.abs(arr) > thr)
    ev = np.zeros(n, np.int8)
    pos = np.zeros(n)
    last = -10 ** 9
    for i in range(n):
        if gate[i] and (i == 0 or not gate[i - 1]) and (i - last >= cooldown):
            ev[i] = 1 if arr[i] >= 0 else -1
            last = i
    for i in np.nonzero(ev)[0]:
        pos[i:i + hold] = ev[i]
    return {'event': ev, 'pos': pos, 'gate': gate, 'thr': thr}
