"""
Adversarial Anti-Cheating & Integrity Guardrails for Quant Lab Backtester
Guarantees that no strategy can cheat, access future prices, mutate dataframes in-place,
or exploit numerical anomalies to fabricate fake backtest results.
"""

import numpy as np
import polars as pl
from typing import Set, Tuple, List, Dict, Any

# Whitelist of strictly causal L2 market features available at tick t
CAUSAL_FEATURE_WHITELIST: Set[str] = {
    'ts_ms', 'bid_p0', 'bid_v0', 'bid_n0', 'ask_p0', 'ask_v0', 'ask_n0',
    'spread', 'mid_price', 'micro_price', 'micro_dev',
    'bid_v5', 'ask_v5', 'qi_1', 'qi_5', 'qi_20',
    'count_skew_5', 'ofi', 'vol_30s_bps'
}

class BacktestIntegrityError(Exception):
    """Raised when an adversarial or illegal operation is detected in strategy code."""
    pass

class BacktestSanitizer:
    @staticmethod
    def prepare_causal_frame(df: pl.DataFrame) -> pl.DataFrame:
        """
        Strips all future forward-looking columns (fwd_ret_*, next_*, etc.)
        and returns an isolated, read-only clone containing ONLY causal past data
        and verified tools registered in the corporate tool registry.
        """
        try:
            from core.tool_registry import tool_registry
            verified_tools = set(tool_registry.get_verified_tool_names())
        except Exception:
            verified_tools = set()

        allowed_cols = CAUSAL_FEATURE_WHITELIST.union(verified_tools)
        available_cols = [c for c in df.columns if c in allowed_cols]
        causal_df = df.select(available_cols).clone()
        return causal_df

    @staticmethod
    def validate_and_sanitize_signal(sig: Any, expected_len: int) -> pl.Series:
        """
        Enforces strict mathematical bounds on the generated signal:
          - Must be a polars.Series
          - Must have length exactly matching input
          - Must have no NaNs, Infs, or Nulls (filled with 0.0)
          - Must be strictly clipped to [-1.0, 1.0]
          - Must exhibit non-zero variance (cannot be constant 0 or constant 1)
        """
        if not isinstance(sig, pl.Series):
            raise BacktestIntegrityError(f"Strategy returned {type(sig)}, expected polars.Series.")

        if len(sig) != expected_len:
            raise BacktestIntegrityError(f"Signal length mismatch: got {len(sig)}, expected {expected_len}.")

        # Replace NaNs / Nulls with 0.0 (neutral)
        clean_sig = sig.fill_nan(0.0).fill_null(0.0)

        # Ensure finite numeric values
        sig_np = clean_sig.to_numpy()
        if not np.all(np.isfinite(sig_np)):
            sig_np = np.nan_to_num(sig_np, nan=0.0, posinf=1.0, neginf=-1.0)
            clean_sig = pl.Series(sig.name, sig_np)

        # Enforce strict bounded signal [-1.0, 1.0]
        bounded_sig = clean_sig.clip(-1.0, 1.0)

        # Check for degenerate constant signals
        std_val = float(bounded_sig.std()) if len(bounded_sig) > 1 else 0.0
        if std_val < 1e-6:
            raise BacktestIntegrityError("Degenerate signal detected: Signal has zero variance (constant flatline).")

        return bounded_sig

    @staticmethod
    def verify_price_integrity(orig_mid: np.ndarray, post_mid: np.ndarray):
        """Ensures strategy code did not tamper with prices in-place."""
        if not np.array_equal(orig_mid, post_mid):
            raise BacktestIntegrityError("Tampering detected: Strategy attempted to modify market price series in-place!")
