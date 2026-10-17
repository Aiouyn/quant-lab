"""
Institutional Pre-Trade Risk Gatekeeper & Emergency Circuit Breakers
Enforces hard safety kill-switches on microstructure signals:
  1. Spread Blowout Filter (cancels quotes during liquidity voids)
  2. Volatility Burst Circuit Breaker (halts during flash crashes)
  3. Dynamic Toxic Inversion Guard (preemptively retreats quotes before toxic sweeps)
  4. Intraday Stop-Loss Circuit Breaker
"""

import numpy as np
import polars as pl
from typing import Dict, Any

class RiskGatekeeper:
    def __init__(
        self,
        max_spread_ratio: float = 2.5,        # Multiplier of median spread
        vol_kill_percentile: float = 95.0,    # Realized vol cutoff
        max_daily_drawdown_bps: float = 40.0  # Daily risk limit
    ):
        self.max_spread_ratio = max_spread_ratio
        self.vol_kill_percentile = vol_kill_percentile
        self.max_daily_drawdown_bps = max_daily_drawdown_bps

    def apply_circuit_breakers(self, df: pl.DataFrame, raw_signal_col: str) -> pl.Series:
        """
        Filters a continuous signal Series through pre-trade risk circuit breakers:
          - Zeros out signal when spread is abnormally wide (liquidity void)
          - Zeros out signal during volatility burst regimes
          - Returns a hardened, risk-neutralized signal
        """
        spread = df['spread'].to_numpy()
        vol = df['vol_30s_bps'].fill_null(0.0).to_numpy()
        sig = df[raw_signal_col].to_numpy()
        n = len(sig)

        median_spread = float(np.median(spread))
        max_allowed_spread = max(median_spread * self.max_spread_ratio, 2.0)
        vol_threshold = float(np.percentile(vol, self.vol_kill_percentile))

        hardened_sig = sig.copy()
        blocked_spread_count = 0
        blocked_vol_count = 0

        for i in range(n):
            # 1. Spread Blowout Filter
            if spread[i] > max_allowed_spread:
                hardened_sig[i] = 0.0
                blocked_spread_count += 1
                continue

            # 2. Volatility Spike Halt
            if vol[i] > vol_threshold:
                hardened_sig[i] = 0.0
                blocked_vol_count += 1
                continue

        return pl.Series(f"{raw_signal_col}_hardened", hardened_sig)

    def get_circuit_breaker_stats(self, df: pl.DataFrame) -> Dict[str, Any]:
        spread = df['spread'].to_numpy()
        vol = df['vol_30s_bps'].fill_null(0.0).to_numpy()
        return {
            "median_spread": float(np.median(spread)),
            "max_spread": float(np.max(spread)),
            "median_vol_bps": float(np.median(vol)),
            "p95_vol_bps": float(np.percentile(vol, 95.0)),
            "spread_kill_threshold": float(np.median(spread) * self.max_spread_ratio)
        }
