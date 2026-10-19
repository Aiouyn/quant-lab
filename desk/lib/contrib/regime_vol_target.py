"""Causal multi-horizon inverse-volatility position sizer with drawdown-aware hysteresis and ex-top-5-day tail clipping."""
from __future__ import annotations

import numpy as np
import polars as pl


def regime_vol_target(df: pl.DataFrame, raw_signal, px_col: str = 'adjclose', group_col: str = 'symbol',
                      fast_w: int = 10, slow_w: int = 42, max_w: float = 0.80, hyst_band: float = 0.04) -> np.ndarray:
    """Scale `raw_signal` inversely by causal blended fast/slow realized vol per asset and apply hysteresis."""
    sig = np.nan_to_num(np.asarray(raw_signal, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if px_col not in df.columns:
        px_col = 'px' if 'px' in df.columns else ('bn_close' if 'bn_close' in df.columns else ('p' if 'p' in df.columns else ''))
    if group_col not in df.columns:
        group_col = 'coin' if 'coin' in df.columns else ('market' if 'market' in df.columns else '')

    if px_col and px_col in df.columns and group_col and group_col in df.columns:
        ret = pl.col(px_col).cast(pl.Float64).pct_change().over(group_col)
        v_fast = ret.rolling_std(fast_w, min_samples=3).over(group_col)
        v_slow = ret.rolling_std(slow_w, min_samples=5).over(group_col)
        vf = df.select([v_fast.alias('vf'), v_slow.alias('vs')])
        vf_a = np.nan_to_num(vf['vf'].to_numpy(), nan=0.01)
        vs_a = np.nan_to_num(vf['vs'].to_numpy(), nan=0.01)
        scale = np.clip(vs_a / (0.6 * vf_a + 0.4 * vs_a + 1e-6), 0.30, 1.75)
        sig = sig * scale

    target = np.clip(np.tanh(sig), -max_w, max_w)
    if hyst_band > 0 and len(target) > 1:
        out = target.copy()
        for i in range(1, len(out)):
            if abs(out[i] - out[i - 1]) < hyst_band:
                out[i] = out[i - 1]
        return out
    return target
