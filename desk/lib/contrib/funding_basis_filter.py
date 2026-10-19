"""Causal cross-venue funding & perp basis carry signal with entry/exit hysteresis bands for `perps` and `carry` desks."""
from __future__ import annotations

import numpy as np
import polars as pl


def funding_basis_filter(df: pl.DataFrame, window_h: int = 24, entry_bps_h: float = 0.15,
                         exit_bps_h: float = 0.05) -> np.ndarray:
    """Compute causal smoothed net funding edge per coin with hysteresis buffer."""
    group_col = 'coin' if 'coin' in df.columns else ('symbol' if 'symbol' in df.columns else None)
    if 'hl_funding' in df.columns and 'bn_rate_last' in df.columns:
        raw_edge = (df['bn_rate_last'].fill_null(0.0).to_numpy() / 8.0) - df['hl_funding'].fill_null(0.0).to_numpy()
    elif 'f' in df.columns:
        raw_edge = -df['f'].fill_null(0.0).to_numpy()
    else:
        return np.zeros(len(df), dtype=float)

    if group_col:
        sm = (df.select(pl.col(group_col))
                .with_columns(pl.Series('_e', raw_edge))
                .select(pl.col('_e').rolling_mean(window_h, min_samples=3).over(group_col).alias('_m'))['_m'].to_numpy())
    else:
        sm = pl.Series(raw_edge).rolling_mean(window_h, min_samples=3).to_numpy()

    edge_bps = np.nan_to_num(sm, nan=0.0) * 1e4
    w = np.where(np.abs(edge_bps) >= entry_bps_h, np.tanh(edge_bps / max(entry_bps_h, 0.05)), 0.0)
    w = np.where(np.abs(edge_bps) < exit_bps_h, 0.0, w)
    return np.clip(w, -0.85, 0.85)
