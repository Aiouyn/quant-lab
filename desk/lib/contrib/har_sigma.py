from __future__ import annotations
"""Causal HAR-style log-vol sizing helper (spec from deepbook-r1 K-74c973): trailing log-variance levels at 60/240/1440m -> NaN-skipping mean (lfc) -> trailing-2d median reference frozen with shift(1) -> mm = clip(exp(0.5*(ref-lfc)), 0.3, 2.5), non-finite -> 1.0. Downsizes in vol spikes (>trailing median), upsizes in calm, neutral during warmup. Truncation-causal (verified in test: head(k) == full[:k]). Use in signal() for entry-frozen or per-row vol targeting (remember L-a65f88: vol-targeting helps holds >=4h, HURTS 30m; L-1c5446: top-5 concentration cure only works when top days are high-vol days — check that coincidence first)."""
import numpy as np
import polars as pl
"""Causal HAR-style log-vol level + trailing-median reference + frozen sizing multiplier.

Spec per deepbook-r1 (K-74c973): input = 1m mid returns (NaN allowed).
  lv  = log(trailing variance) at windows (60, 240, 1440), min_samples = w//2, var clipped at 1e-14
  lfc = NaN-skipping mean of the three lv series (NaN only when all three missing)
  ref = lfc.rolling_median(2880, min_samples=240).shift(1)      # trailing-reference, frozen one row
  mm  = clip(exp(0.5 * (ref - lfc)), 0.3, 2.5); non-finite -> 1.0
Truncation-causal: every operation is trailing-only (rolling_*/shift(+1)), so head(k) outputs == first k
rows of the full run. mm<1 downsizes when current vol > trailing 2d median vol, mm>1 upsize in calm
markets (capped), and equals 1.0 during warmup/undefined rows. Use ONLY in signal()/sizing (not fit()).
"""

import warnings

import numpy as np
import polars as pl

DEFAULTS = dict(w=(60, 240, 1440), ref_window=2880, ref_min=240, var_floor=1e-14,
                mm_lo=0.3, mm_hi=2.5)


def har_sigma(ret, w=(60, 240, 1440), ref_window=2880, ref_min=240, var_floor=1e-14,
              mm_lo=0.3, mm_hi=2.5):
    """ret: polars Series (or 1d array) of 1m returns. Returns dict of numpy arrays:
    {'lv': (n,3) log-var levels, 'lfc': log forecast component, 'ref': trailing ref, 'mm': multiplier}"""
    if not isinstance(ret, pl.Series):
        ret = pl.Series(np.asarray(ret, float))
    cols = []
    for wi in w:
        var = ret.rolling_var(window_size=int(wi), min_samples=max(int(wi) // 2, 2), ddof=1)
        var = var.clip(lower_bound=var_floor)
        cols.append(var.log().to_numpy())                    # log var; NaN preserved
    arr = np.column_stack(cols)                              # (n, 3)

    with warnings.catch_warnings():                          # all-NaN rows -> NaN, no RuntimeWarning
        warnings.simplefilter('ignore', category=RuntimeWarning)
        lfc = np.nanmean(arr, axis=1)
    lfc[np.all(~np.isfinite(arr), axis=1)] = np.nan

    ref = pl.Series(lfc).rolling_median(window_size=int(ref_window),
                                        min_samples=int(ref_min)).shift(1).to_numpy()

    with np.errstate(invalid='ignore'):
        mm = np.clip(np.exp(0.5 * (ref - lfc)), mm_lo, mm_hi)
    mm = np.where(np.isfinite(mm), mm, 1.0)                  # warmup / undefined -> neutral 1.0
    return {'lv': arr, 'lfc': lfc, 'ref': ref, 'mm': mm}
