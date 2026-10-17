"""
Unit Test Suite: Autonomous Quantitative Tool Genesis & Corporate Swarm
Tests adversarial 5-stage validation, causal invariance (anti-lookahead leakage),
price immutability, tool registry compilation, dynamic dataframe enrichment,
and Investment Committee qualitative feedback integration.
"""

import unittest
import numpy as np
import polars as pl

from core.tool_validator import tool_validator
from core.tool_registry import QuantitativeTool, ToolRegistry
from core.integrity_guard import BacktestSanitizer
from swarm.evolution import EvolutionaryEngine
from core.knowledge_base import ResearchLedger


def create_mock_microstructure_df(n_rows: int = 500) -> pl.DataFrame:
    """Generates synthetic high-frequency orderbook ticks for testing."""
    np.random.seed(42)
    base_price = 85000.0
    returns = np.random.normal(0, 0.5, n_rows)
    mid_price = base_price + np.cumsum(returns)
    spread = np.random.uniform(0.1, 1.5, n_rows)

    bid_p0 = mid_price - spread / 2.0
    ask_p0 = mid_price + spread / 2.0
    bid_v0 = np.random.exponential(1.5, n_rows) + 0.1
    ask_v0 = np.random.exponential(1.5, n_rows) + 0.1

    bid_v5 = bid_v0 * 5.0 + np.random.uniform(0, 2, n_rows)
    ask_v5 = ask_v0 * 5.0 + np.random.uniform(0, 2, n_rows)

    qi_5 = (bid_v5 - ask_v5) / (bid_v5 + ask_v5)
    micro_dev = (bid_p0 * ask_v0 + ask_p0 * bid_v0) / (bid_v0 + ask_v0) - mid_price

    df = pl.DataFrame({
        "ts": np.arange(1000, 1000 + n_rows),
        "mid_price": mid_price,
        "spread": spread,
        "bid_p0": bid_p0,
        "ask_p0": ask_p0,
        "bid_v0": bid_v0,
        "ask_v0": ask_v0,
        "bid_n0": np.ones(n_rows, dtype=np.int32),
        "ask_n0": np.ones(n_rows, dtype=np.int32),
        "bid_v5": bid_v5,
        "ask_v5": ask_v5,
        "qi_1": (bid_v0 - ask_v0) / (bid_v0 + ask_v0),
        "qi_5": qi_5,
        "qi_20": qi_5 * 0.9,
        "count_skew_5": np.zeros(n_rows),
        "ofi": np.random.normal(0, 1.0, n_rows),
        "vol_30s_bps": np.random.uniform(5.0, 25.0, n_rows),
        "micro_dev": micro_dev
    })
    return df


class TestToolGenesisAndVerification(unittest.TestCase):
    def setUp(self):
        self.df = create_mock_microstructure_df(500)

    def test_valid_causal_tool_passes(self):
        """A properly bounded, causal tool should pass all 5 validation stages."""
        valid_code = """import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    qi = df['qi_5']
    d1 = (qi - qi.shift(1)).fill_null(0.0)
    return d1.clip(-1.0, 1.0)
"""
        passed, report = tool_validator.validate_tool_candidate(valid_code, self.df)
        self.assertTrue(passed, f"Valid tool failed validation: {report.get('errors')}")
        self.assertTrue(report["syntax_valid"])
        self.assertTrue(report["price_immutability_passed"])
        self.assertTrue(report["causal_leakage_passed"])
        self.assertTrue(report["numerical_stability_passed"])
        self.assertLessEqual(report["latency_ms"], 150.0)

    def test_lookahead_leakage_tool_rejected(self):
        """A tool peeking into the future (shift(-1)) MUST be caught by Causal Invariance!"""
        leaking_code = """import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    # Future lookahead leak!
    future_mid = df['mid_price'].shift(-1).fill_null(strategy='backward')
    return (future_mid - df['mid_price']).clip(-10.0, 10.0)
"""
        passed, report = tool_validator.validate_tool_candidate(leaking_code, self.df)
        self.assertFalse(passed, "Lookahead leaking tool was erroneously approved!")
        self.assertFalse(report["causal_leakage_passed"], "Causal invariance failed to detect shift(-1) leak!")
        self.assertTrue(any("DATA LEAKAGE DETECTED" in err for err in report["errors"]))

    def test_price_mutation_tool_rejected(self):
        """A tool attempting to alter input price series MUST be caught by Price Immutability."""
        mutating_code = """import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    # Intentionally corrupt price to cheat backtester
    df.replace('mid_price', df['mid_price'] * 2.0)
    return df['qi_5']
"""
        passed, report = tool_validator.validate_tool_candidate(mutating_code, self.df)
        self.assertFalse(passed, "Price mutator tool was erroneously approved!")
        self.assertFalse(report["price_immutability_passed"], "Price immutability guard failed to detect mutation!")

    def test_nan_producing_tool_rejected(self):
        """A tool producing unhandled NaNs or Infs must be rejected."""
        nan_code = """import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    # Division by zero without fill_null creates NaNs
    return df['qi_5'] / 0.0
"""
        passed, report = tool_validator.validate_tool_candidate(nan_code, self.df)
        self.assertFalse(passed, "NaN-producing tool was erroneously approved!")
        self.assertFalse(report["numerical_stability_passed"])

    def test_tool_registry_and_enrichment(self):
        """Tests saving a verified tool and enriching the dataframe dynamically."""
        registry = ToolRegistry(db_path=":memory:")
        verified_tool_names = registry.get_verified_tool_names()
        # Verify pre-seeded tools are present
        self.assertIn("zscore_ofi", verified_tool_names)
        self.assertIn("depth_decay_slope", verified_tool_names)
        self.assertIn("vol_regime_ratio", verified_tool_names)

        # Apply tools to dataframe
        enriched = registry.apply_tools_to_frame(self.df)
        self.assertIn("zscore_ofi", enriched.columns)
        self.assertIn("depth_decay_slope", enriched.columns)
        self.assertIn("vol_regime_ratio", enriched.columns)

        # Verify values are valid floats with no NaNs
        self.assertEqual(len(enriched), len(self.df))
        self.assertEqual(enriched["zscore_ofi"].null_count(), 0)
        self.assertEqual(enriched["depth_decay_slope"].null_count(), 0)

        # Register a new custom tool
        custom_tool = QuantitativeTool(
            name="mock_orderbook_entropy",
            author="Information_Theorist_01",
            category="FEATURE_EXTRACTOR",
            mathematical_formulation="H_t = -sum(p_i * log(p_i))",
            docstring="Orderbook distribution entropy",
            code="""import polars as pl
def calculate(df: pl.DataFrame) -> pl.Series:
    return (df['qi_5'].abs() * 0.5).fill_null(0.0)
""",
            verified=True,
            audit_report={"status": "PASSED"}
        )
        success = registry.register_tool(custom_tool)
        self.assertTrue(success)

        # Verify it can now enrich
        enriched2 = registry.apply_tools_to_frame(self.df)
        self.assertIn("mock_orderbook_entropy", enriched2.columns)

    def test_integrity_guard_whitelists_approved_tools(self):
        """Tests that IntegrityGuard allows registered tools but blocks forbidden columns."""
        registry = ToolRegistry(db_path=":memory:")
        enriched = registry.apply_tools_to_frame(self.df)
        
        # Guard should allow approved tools
        clean_df = BacktestSanitizer.prepare_causal_frame(enriched)
        self.assertIn("zscore_ofi", clean_df.columns)
        self.assertIn("depth_decay_slope", clean_df.columns)

        # Unauthorized / lookahead columns are strictly stripped from execution frame
        leaked_df = enriched.with_columns(pl.Series("future_return_5s", np.random.normal(0, 1, len(enriched))))
        sanitized_df = BacktestSanitizer.prepare_causal_frame(leaked_df)
        self.assertNotIn("future_return_5s", sanitized_df.columns)

    def test_committee_qualitative_feedback_closed_loop(self):
        """Tests that committee qualitative critiques are stored and injected into guidance."""
        kb = ResearchLedger(":memory:")
        evolution = EvolutionaryEngine(kb, query_fn=lambda p, s, a, **kw: "{}")

        evolution.record_committee_critique(
            strat_title="Naive Spread Scalper",
            critique_summary="Model overfits during low volatility and suffers adverse selection on momentum bursts.",
            key_flaws=["High adverse selection", "Unhedged spread risk"]
        )

        guidance = evolution.get_committee_guidance()
        self.assertIn("Naive Spread Scalper", guidance)
        self.assertIn("High adverse selection", guidance)


if __name__ == "__main__":
    unittest.main()
