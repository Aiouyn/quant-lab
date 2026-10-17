"""Sparse extreme-episode entry trigger (L-66e538 recipe): causal trailing-quantile gate on |x| (default trailing-2d q99.5 at 1m), first-onset sign entry, cooldown between entries, fixed hold. Includes helper rolling_q(x, w, q) = causal trailing quantile matching np.quantile exactly (also usable as a robust threshold / z-score baseline). Returns event/pos/gate/thr arrays; 'pos' is ready-to-use strategy signal (harness fills 1 row later). Validated: reproduces L-66e538 sparsity on 10 contiguous dev days (1.62 fires/day vs 1.3/day full sample, gross signed 30m +43.4bps, hit 85%); cooldown suppresses 54% of raw onsets (3.5/day). Use q>=0.995, lookback>=1d; first-onset+cooldown semantics drop later onsets in-window."""
import numpy as np
import polars as pl
import numpy as np
import polars as pl


def rolling_q(x, w, q, min_frac=0.5, interpolation='nearest'):
    """Causal trailing rolling quantile: output at row i uses rows [i-w+1 .. i] ONLY.
    NaN until min_frac*w samples seen (warmup / gap protection). Matches np.quantile(seg, q) exactly
    (verified to 2e-16 with interpolation='linear').
    interpolation: 'nearest'|'linear'|'lower'|'higher'|'midpoint'.
    NOTE this polars takes (quantile, interpolation, window_size) positionally - always call with keywords."""
    s = pl.Series(np.asarray(x, float))
    k = max(2, int(w * min_frac))
    try:
        return s.rolling_quantile(q, window_size=w, min_samples=k, interpolation=interpolation).to_numpy()
    except TypeError:  # older polars: min_periods
        return s.rolling_quantile(q, window_size=w, min_periods=k, interpolation=interpolation).to_numpy()


def sparse_extreme_trigger(x, lookback=2880, q=0.995, cooldown=240, hold=30, min_frac=0.5, interpolation='nearest'):
    """Sparse extreme-episode trigger = the exact entry recipe of playbook L-66e538 (net +16.8bps/trade, t=3.0).

    Gate: |x_i| > causal trailing quantile q of |x| over `lookback` rows. Entry: FIRST-ONSET only (rising edge of
    the gate), direction = sign(x_i) at onset, at most one entry per `cooldown` rows (first-come; later onsets in
    the window are DROPPED - on 10 dev days this suppresses 54% of onsets), target held `hold` rows.

    Returns dict (len-n numpy arrays):
      event : int8, 0 / +1 / -1 at accepted onset rows
      pos   : float, target position over rows i..i+hold-1 - use THIS as the strategy signal (harness fills 1 row later)
      gate  : bool, raw threshold crossing
      thr   : float, causal trailing q-quantile of |x| (reusable as a robust threshold / z baseline)

    Usage (1m strategy, L-66e538 config):
        from desk.lib.contrib.sparse_extreme_trigger import sparse_extreme_trigger
        xv = (df['ofi10'] / (df['bdep_25'] + df['adep_25'])).to_numpy()
        x15 = pl.Series(xv).rolling_sum(15, min_samples=15).to_numpy()
        return sparse_extreme_trigger(x15, lookback=2880, q=0.995, cooldown=240, hold=30)['pos']

    CAVEATS: fully causal (trailing windows only). Use q >= 0.995 with lookback >= 1 day so crossings are genuinely
    rare (~1.6 fires/day in the config above vs 3.5 raw onsets/day); at lower q, frequent noise onsets can pre-empt
    genuine spikes inside the cooldown window (first-onset semantics). Require cooldown >= hold to avoid overlaps."""
    x = np.asarray(x, float)
    n = x.size
    thr = rolling_q(np.abs(x), lookback, q, min_frac, interpolation)
    gate = np.isfinite(thr) & (np.abs(x) > thr)
    ev = np.zeros(n, np.int8)
    pos = np.zeros(n)
    last = -10 ** 9
    for i in range(n):
        if gate[i] and (i == 0 or not gate[i - 1]) and (i - last >= cooldown):
            ev[i] = 1 if x[i] >= 0 else -1
            last = i
    for i in np.nonzero(ev)[0]:
        pos[i:i + hold] = ev[i]
    return {'event': ev, 'pos': pos, 'gate': gate, 'thr': thr}