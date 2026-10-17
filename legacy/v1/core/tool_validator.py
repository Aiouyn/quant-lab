"""
Adversarial Tool Verification & Anti-Data-Leakage Suite
Validates that autonomously invented tools are:
  1. Strictly Causal (Zero Lookahead Data Leakage)
  2. Numerically Stable (Immune to zero spreads, NaNs, infinities)
  3. High-Performance (Vectorized Polars / SIMD compliant)
  4. Memory & Price Immutable (Cannot tamper with market history)
"""

import time
import logging
import numpy as np
import polars as pl
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ToolValidationError(Exception):
    pass

class ToolValidator:
    def __init__(self, max_latency_ms: float = 150.0):
        self.max_latency_ms = max_latency_ms

    def validate_tool_candidate(
        self,
        code_str: str,
        test_df: pl.DataFrame
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs the full 5-stage adversarial audit on an agent-created tool.
        Returns: (passed: bool, report: Dict[str, Any])
        """
        report: Dict[str, Any] = {
            "syntax_valid": False,
            "causal_leakage_passed": False,
            "numerical_stability_passed": False,
            "price_immutability_passed": False,
            "latency_ms": 0.0,
            "errors": []
        }

        # Stage 1: Syntax & Execution Sandbox
        scope: Dict[str, Any] = {}
        try:
            exec(code_str, scope)
            if "calculate" not in scope or not callable(scope["calculate"]):
                report["errors"].append("Tool must define a callable function: `def calculate(df: pl.DataFrame) -> pl.Series:`")
                return False, report
            calculate_fn = scope["calculate"]
            report["syntax_valid"] = True
        except Exception as e:
            report["errors"].append(f"Syntax/compilation error: {e}")
            return False, report

        # Stage 2: Price Immutability Check
        try:
            sample_df = test_df.slice(0, min(500, len(test_df))).clone()
            orig_mid = sample_df['mid_price'].to_numpy().copy()
            res = calculate_fn(sample_df)

            if not isinstance(res, pl.Series):
                report["errors"].append(f"Function returned {type(res)}, expected polars.Series.")
                return False, report

            if len(res) != len(sample_df):
                report["errors"].append(f"Length mismatch: output has {len(res)} rows, expected {len(sample_df)}.")
                return False, report

            post_mid = sample_df['mid_price'].to_numpy()
            if not np.array_equal(orig_mid, post_mid):
                report["errors"].append("SECURITY VIOLATION: Tool modified market prices in-place!")
                return False, report

            report["price_immutability_passed"] = True
        except Exception as e:
            report["errors"].append(f"Execution failure on sample data: {e}")
            return False, report

        # Stage 3: Strict Causal Invariance (Zero Lookahead / Future Leakage Proof)
        # Mathematical Proof: Output at tick t must be strictly independent of ticks > t.
        try:
            N = len(sample_df)
            K = N // 2
            
            # Run tool on full slice
            res_full = calculate_fn(sample_df).to_numpy()
            
            # Run tool on truncated prefix [0..K]
            prefix_df = sample_df.slice(0, K).clone()
            res_prefix = calculate_fn(prefix_df).to_numpy()

            # The prefix output MUST match the first K elements of the full output
            # If future rows turned a boundary NaN into a valid float, that is severe lookahead leakage
            with np.errstate(invalid='ignore'):
                diff_arr = np.nan_to_num(np.abs(res_full[:K] - res_prefix[:K]), nan=999.0)
            diff = float(np.max(diff_arr))
            if diff > 1e-4:
                report["errors"].append(
                    f"DATA LEAKAGE DETECTED: Tool output at past ticks changed when future data was introduced (max diff = {diff:.6f}). Lookahead bias present!"
                )
                return False, report

            report["causal_leakage_passed"] = True
        except Exception as e:
            report["errors"].append(f"Causal invariance test failed: {e}")
            return False, report

        # Stage 4: Numerical Stability & NaN Immunity
        try:
            nan_count = int(res.is_nan().sum())
            null_count = int(res.is_null().sum())
            inf_count = int(np.isinf(res.to_numpy()).sum())

            if nan_count > 0 or null_count > 0 or inf_count > 0:
                report["errors"].append(
                    f"Numerical instability: Tool produced {nan_count} NaNs, {null_count} Nulls, or {inf_count} Infs. Must use `.fill_null(0.0)`."
                )
                return False, report

            # Check for non-zero variance
            std_val = float(res.std()) if len(res) > 1 else 0.0
            if std_val < 1e-7:
                report["errors"].append("Degenerate tool: Output has zero variance (constant flatline).")
                return False, report

            report["numerical_stability_passed"] = True
        except Exception as e:
            report["errors"].append(f"Stability verification error: {e}")
            return False, report

        # Stage 5: Execution Latency Benchmark
        try:
            bench_df = test_df.slice(0, min(5000, len(test_df))).clone()
            t0 = time.perf_counter()
            _ = calculate_fn(bench_df)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            report["latency_ms"] = round(elapsed_ms, 2)

            if elapsed_ms > self.max_latency_ms:
                report["errors"].append(f"Latency violation: Tool took {elapsed_ms:.1f}ms (threshold: {self.max_latency_ms}ms). Must be vectorized.")
                return False, report
        except Exception as e:
            report["errors"].append(f"Benchmark error: {e}")
            return False, report

        return True, report

tool_validator = ToolValidator()
