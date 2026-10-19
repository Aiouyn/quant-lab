"""Open-Ended Gen #2 Compositional AST Primitive (hyperbolic_crowding_decay ∘ semivariance_asymmetry_gate ∘ adaptive_deadband_hysteresis) synthesized to eliminate perps residual bottleneck (Right-skewed altcoin funding squeeze & maker queue adverse selection during OI unwinds)."""
from __future__ import annotations

import numpy as np
import polars as pl


GRAMMAR_METADATA = {
    'generation': 2,
    'target_desk': 'perps',
    'signal_kernel': 'hyperbolic_crowding_decay',
    'regime_gate': 'semivariance_asymmetry_gate',
    'execution_stabilizer': 'adaptive_deadband_hysteresis',
    'ast_depth': 6,
}


def oe_hyperbolic_semivariance_v2(
    df_or_signal,
    raw_weights=None,
    fast_w: int = 12,
    slow_w: int = 46,
    deadband: float = 0.027,
    decay_lambda: float = 0.18,
    cap: float = 0.6,
) -> np.ndarray:
    """Causal Open-Ended Compositional Operator: hyperbolic_crowding_decay -> semivariance_asymmetry_gate -> adaptive_deadband_hysteresis.
    Strictly causal over trailing windows [t-w+1 .. t] with zero lookahead leakage.
    """
    if raw_weights is not None:
        base = np.asarray(raw_weights, dtype=float).reshape(-1)
    elif isinstance(df_or_signal, pl.DataFrame):
        for col in ('adjclose', 'px', 'bn_funding', 'f', 'p'):
            if col in df_or_signal.columns:
                base = df_or_signal[col].cast(pl.Float64).fill_null(0.0).to_numpy()
                break
        else:
            base = np.zeros(len(df_or_signal), dtype=float)
    elif isinstance(df_or_signal, pl.Series):
        base = df_or_signal.cast(pl.Float64).fill_null(0.0).to_numpy()
    else:
        base = np.asarray(df_or_signal, dtype=float).reshape(-1)

    x = np.nan_to_num(base.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=float)

    s = pl.Series(x)
    fw = max(2, int(fast_w))
    sw = max(fw + 2, int(slow_w))

    # Layer 1: Signal Kernel (hyperbolic_crowding_decay)
    ma_fast = np.nan_to_num(s.rolling_mean(fw, min_samples=1).to_numpy(), nan=0.0)
    ma_slow = np.nan_to_num(s.rolling_mean(sw, min_samples=1).to_numpy(), nan=0.0)
    std_slow = np.nan_to_num(s.rolling_std(sw, min_samples=2).to_numpy(), nan=0.01) + 1e-6
    vel = (ma_fast - ma_slow) / std_slow
    decay_env = np.exp(-float(decay_lambda) * np.maximum(0.0, np.abs(vel) - 2.2))
    k_out = x * (0.82 + 0.18 * np.tanh(vel) * decay_env)

    # Layer 2: Regime Conditioner (semivariance_asymmetry_gate)
    abs_s = pl.Series(np.abs(x))
    v_short = np.nan_to_num(abs_s.rolling_mean(fw, min_samples=1).to_numpy(), nan=0.01)
    v_long = np.nan_to_num(abs_s.rolling_mean(sw, min_samples=1).to_numpy(), nan=0.01) + 1e-6
    v_ratio = v_short / v_long
    gate_mult = np.where(v_ratio > 1.55, np.clip(1.55 / np.maximum(v_ratio, 1.0), 0.45, 1.0), 1.0)
    g_out = k_out * gate_mult

    # Layer 3: Execution Stabilizer (adaptive_deadband_hysteresis)
    out = np.empty(n, dtype=float)
    prev = 0.0
    db = float(deadband)
    for i in range(n):
        cand = float(g_out[i])
        diff = cand - prev
        if abs(diff) < db:
            # Cubic shrinkage inside deadband to suppress micro-churn
            ratio = (diff * diff) / (db * db + 1e-9)
            prev = prev + diff * ratio
        else:
            prev = cand
        out[i] = prev

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -float(cap), float(cap))
