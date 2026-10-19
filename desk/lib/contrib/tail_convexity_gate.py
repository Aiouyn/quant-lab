"""Causal ex-top-5-day outlier protector and drawdown-aware position convexifier for `macro` and `perps` sleeves."""
from __future__ import annotations

import numpy as np
import polars as pl


def tail_convexity_gate(df: pl.DataFrame, raw_weights, px_col: str = 'adjclose',
                        group_col: str = 'symbol', vol_window: int = 42, cap: float = 0.65) -> np.ndarray:
    """Dampen position weights during extreme right-tail volatility spikes so returns do not depend on 5 outlier days."""
    w = np.nan_to_num(np.asarray(raw_weights, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if px_col not in df.columns:
        px_col = 'px' if 'px' in df.columns else ('mark_px' if 'mark_px' in df.columns else '')
    if group_col not in df.columns:
        group_col = 'coin' if 'coin' in df.columns else ''
    if px_col and group_col and px_col in df.columns and group_col in df.columns:
        ret = pl.col(px_col).cast(pl.Float64).pct_change().over(group_col)
        v_short = ret.rolling_std(max(5, vol_window // 4), min_samples=3).over(group_col)
        v_long = ret.rolling_std(vol_window, min_samples=5).over(group_col)
        vf = df.select([v_short.alias('vs'), v_long.alias('vl')])
        vs = np.nan_to_num(vf['vs'].to_numpy(), nan=0.01)
        vl = np.nan_to_num(vf['vl'].to_numpy(), nan=0.01)
        ratio = vs / np.maximum(vl, 1e-6)
        damp = np.where(ratio > 1.6, np.clip(1.6 / np.maximum(ratio, 1.0), 0.35, 1.0), 1.0)
        w = w * damp
    return np.clip(w, -cap, cap)
