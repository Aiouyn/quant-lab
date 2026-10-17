"""Triple-barrier labels for ML fitting (research / fit() on train days ONLY - never inside signal()): per-row first-touch class (+1 upper / -1 lower / 0 vertical) with dynamic vol-scaled barriers (pt/sl_mult x causal trailing return-std x sqrt(max_h)) and gap-aware segmentation (labels never span ts gaps). Also returns hit_row, hold_rows and fwd_bps (signed move to touch) for regression targets or event studies. Serves next session's ML lanes (volregime T-01d22f, midfreq T-89c9ce): fit a classifier on labels from strictly-earlier days, predict in signal()."""
import numpy as np
import polars as pl
import math
import numpy as np


def _roll_std_bps(r, w, min_s):
    """Causal trailing std of per-row log-returns (bps): row i uses rows [i-w+1..i], requires >= min_s samples."""
    n = r.size
    c1 = np.cumsum(r); c2 = np.cumsum(r * r)
    out = np.full(n, np.nan)
    for i in range(n):
        j0 = max(i - w + 1, 0)
        k = i - j0 + 1
        if k < min_s:
            continue
        m1 = (c1[i] - (c1[j0 - 1] if j0 > 0 else 0.0)) / k
        m2 = (c2[i] - (c2[j0 - 1] if j0 > 0 else 0.0)) / k
        out[i] = math.sqrt(max(m2 - m1 * m1, 0.0))
    return out


def triple_barrier_labels(mid, ts=None, pt_mult=4.0, sl_mult=4.0, max_h=60, vol_window=60, gap_ms=600000):
    """Triple-barrier labels (Lopez de Prado) for RESEARCH / fit() on TRAIN data ONLY: labels look ahead -
    never call inside signal() (the causality test will reject it).

    mid : price series (np or polars). ts: optional epoch-ms stamps -> labelling never spans gaps > gap_ms.
    Barriers per row i (log space): upper/lower = mid_i * exp(+-pt/sl_mult * sigma_h), with
    sigma_h = sigma1 * sqrt(max_h), sigma1 = causal trailing std of 1-row log returns (bps) over vol_window
    rows (min vol_window//2 samples); vertical barrier at row i+max_h (truncated at segment end/gap).
    First touch wins.

    Returns dict of len-n arrays:
      label   : int8  +1 upper touched first, -1 lower touched first, 0 vertical barrier first
                (ambiguous with unlabelable warmup/end rows - use hit_row to disambiguate)
      hit_row : int64 row of first touch (or the vertical row), -1 where unlabelable (warmup, last row of segment)
      hold_rows  int64 hit_row - i (rows from label to touch), -1 where unlabelable
      fwd_bps : float signed move (bps) from i to hit_row (research label; not a signal)

    Typical use: label = triple_barrier_labels(mids, ts)['label'] on train days, fit a classifier on features
    known at i, then predict inside signal() on live rows."""
    m = np.asarray(mid, float)
    n = m.size
    lg = np.log(m)
    r = np.zeros(n)
    r[1:] = np.diff(lg) * 1e4                                # per-row log return, bps
    sig1 = _roll_std_bps(r, vol_window, max(2, vol_window // 2))
    sh = sig1 * math.sqrt(max_h) * 1e-4                      # sigma over horizon in log units
    up_b = m * np.exp(pt_mult * sh)
    dn_b = m * np.exp(-sl_mult * sh)

    if ts is not None:
        t = np.asarray(ts, float)
        brk = np.where(np.diff(t) > gap_ms)[0] + 1
    else:
        brk = np.zeros(0, dtype=np.int64)
    seg_end = np.empty(n, np.int64)
    start = 0
    for b in np.r_[brk, n]:
        seg_end[start:b] = b - 1
        start = b

    label = np.zeros(n, np.int8)
    hit_row = np.full(n, -1, np.int64)
    hold_rows = np.full(n, -1, np.int64)
    fwd = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(sig1[i]) or seg_end[i] <= i:
            continue
        jmax = min(i + max_h, seg_end[i])
        hit, hj = 0, jmax
        ui, di = up_b[i], dn_b[i]
        for j in range(i + 1, jmax + 1):
            if m[j] >= ui:
                hit, hj = 1, j
                break
            if m[j] <= di:
                hit, hj = -1, j
                break
        label[i] = hit
        hit_row[i] = hj
        hold_rows[i] = hj - i
        fwd[i] = (m[hj] - m[i]) / m[i] * 1e4
    return {'label': label, 'hit_row': hit_row, 'hold_rows': hold_rows, 'fwd_bps': fwd}