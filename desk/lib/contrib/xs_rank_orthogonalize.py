"""Causal cross-sectional rank demean, winsorization, and Gram-Schmidt factor orthogonalization per timestamp (`date` or `ts`) for multi-asset panels (`macro`, `perps`, `carry`, `pred`)."""
from __future__ import annotations

import numpy as np
import polars as pl


def xs_rank_demean(df: pl.DataFrame, values, time_col: str | None = None, clip_z: float = 3.0) -> np.ndarray:
    """Cross-sectionally z-score and clip `values` within each timestamp (`date` or `ts`), strictly causal."""
    arr = np.nan_to_num(np.asarray(values, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if time_col is None:
        time_col = 'date' if 'date' in df.columns else ('ts' if 'ts' in df.columns else None)
    if time_col and time_col in df.columns:
        tmp = df.select(pl.col(time_col)).with_columns(pl.Series('_x', arr))
        z = tmp.select(
            ((pl.col('_x') - pl.col('_x').mean().over(time_col))
             / (pl.col('_x').std().over(time_col) + 1e-9)).alias('_z')
        )['_z'].to_numpy()
        return np.clip(np.nan_to_num(z, nan=0.0), -clip_z, clip_z)
    std = float(np.std(arr))
    return np.clip((arr - float(np.mean(arr))) / (std + 1e-9), -clip_z, clip_z)


def xs_rank_orthogonalize(df: pl.DataFrame, primary, benchmark, time_col: str | None = None) -> np.ndarray:
    """Causal per-timestamp residual of `primary` orthogonalized against `benchmark`, z-scored to [-3, 3]."""
    p = xs_rank_demean(df, primary, time_col=time_col)
    b = xs_rank_demean(df, benchmark, time_col=time_col)
    if time_col is None:
        time_col = 'date' if 'date' in df.columns else ('ts' if 'ts' in df.columns else None)
    if time_col and time_col in df.columns:
        tmp = df.select(pl.col(time_col)).with_columns([pl.Series('_p', p), pl.Series('_b', b)])
        beta = tmp.select(
            ((pl.col('_p') * pl.col('_b')).mean().over(time_col)
             / ((pl.col('_b') * pl.col('_b')).mean().over(time_col) + 1e-9)).alias('_beta')
        )['_beta'].to_numpy()
        resid = p - np.nan_to_num(beta, nan=0.0) * b
        return xs_rank_demean(df, resid, time_col=time_col)
    beta = float(np.dot(p, b) / (np.dot(b, b) + 1e-9))
    return xs_rank_demean(df, p - beta * b, time_col=None)
