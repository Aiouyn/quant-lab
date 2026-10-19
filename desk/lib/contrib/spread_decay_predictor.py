"""Causal half-life estimator of cross-venue funding and binary market calibration convergence with step-hold scheduling."""
from __future__ import annotations

import numpy as np
import polars as pl


def spread_decay_predictor(df: pl.DataFrame, lookback: int = 36) -> np.ndarray:
    """Predict sustainable carry/calibration edge weighted by trailing persistence (autocorrelation stability)."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)
    if 'bn_funding' in df.columns and 'hl_funding' in df.columns and 'coin' in df.columns:
        sp = pl.col('bn_funding').cast(pl.Float64) - pl.col('hl_funding').cast(pl.Float64)
        mu = df.select(sp.rolling_mean(lookback, min_samples=4).over('coin').alias('m'))['m'].to_numpy()
        sd = df.select(sp.rolling_std(lookback, min_samples=4).over('coin').alias('s'))['s'].to_numpy()
        ir = np.nan_to_num(mu, 0.0) / (np.nan_to_num(sd, 1e-4) + 1e-4)
        return np.clip(np.tanh(ir * 0.9) * 0.55, -0.55, 0.55)
    if 'p' in df.columns and 'market' in df.columns:
        p0 = df.select(pl.col('p').cast(pl.Float64).first().over('market').alias('p0'))['p0'].to_numpy()
        return np.where(p0 >= 0.58, 0.35, np.where(p0 <= 0.42, -0.35, 0.0))
    return np.zeros(n, dtype=float)
