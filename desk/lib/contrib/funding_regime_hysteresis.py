"""Causal cross-venue funding regime state machine with spread-compression dampening and maker step-hold lock."""
from __future__ import annotations

import numpy as np
import polars as pl


def funding_regime_hysteresis(df: pl.DataFrame, fast_h: int = 24, slow_h: int = 96,
                              step_hold_h: int = 72, min_spread_bps_h: float = 0.12) -> np.ndarray:
    """Compute regime-adaptive funding carry or crowding signal with maker step-hold lock."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)
    group_col = 'coin' if 'coin' in df.columns else ('symbol' if 'symbol' in df.columns else None)
    if 'bn_funding' in df.columns and 'hl_funding' in df.columns:
        raw_sp = (df['bn_funding'].cast(pl.Float64).fill_null(0.0) - df['hl_funding'].cast(pl.Float64).fill_null(0.0)).to_numpy()
    elif 'f' in df.columns:
        raw_sp = -df['f'].cast(pl.Float64).fill_null(0.0).to_numpy()
    else:
        return np.zeros(n, dtype=float)

    if group_col:
        wdf = df.select(pl.col(group_col)).with_columns(pl.Series('_sp', raw_sp))
        mf = wdf.select(pl.col('_sp').rolling_mean(fast_h, min_samples=4).over(group_col).alias('mf'))['mf'].to_numpy().copy()
        ms = wdf.select(pl.col('_sp').rolling_mean(slow_h, min_samples=8).over(group_col).alias('ms'))['ms'].to_numpy().copy()
    else:
        s = pl.Series(raw_sp)
        mf = s.rolling_mean(fast_h, min_samples=4).to_numpy().copy()
        ms = s.rolling_mean(slow_h, min_samples=8).to_numpy().copy()

    blended_bps = (0.65 * np.nan_to_num(mf, nan=0.0) + 0.35 * np.nan_to_num(ms, nan=0.0)) * 1e4
    active = np.where(np.abs(blended_bps) >= min_spread_bps_h, np.tanh(blended_bps / max(min_spread_bps_h * 2.5, 0.1)), 0.0)
    if 'ts' in df.columns and group_col:
        bucket = df['ts'].to_numpy() // (3600_000 * max(12, int(step_hold_h)))
        tmp = df.select([group_col]).with_columns([pl.Series('_a', active), pl.Series('_b', bucket)])
        locked = tmp.select(pl.col('_a').first().over([group_col, '_b']).alias('lk'))['lk'].to_numpy().copy()
        return np.clip(np.nan_to_num(locked, nan=0.0), -0.60, 0.60)
    return np.clip(np.nan_to_num(active, nan=0.0), -0.60, 0.60)
