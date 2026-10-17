"""
Unit Test Suite: Institutional Dual-Mode HFT & Avellaneda-Stoikov Engine
Tests:
  1. Avellaneda-Stoikov market making simulation with inventory penalties
  2. FIFO queue simulator with institutional fee tiers (0.05 bps maker)
  3. Red Team Auditor with dynamic holding period and threshold parameters
  4. CPCV regime stability with institutional thresholds
"""

import unittest
import numpy as np
import polars as pl

from core.avellaneda_stoikov import AvellanedaStoikovSimulator, MarketMakingConfig
from core.fifo_queue_simulator import FIFOQueueSimulator, FIFOQueueConfig
from core.execution_simulator import ExecutionSimulator, ExecutionConfig
from agents.red_team_auditor import RedTeamAuditor


def create_test_orderbook_df(n_rows: int = 500) -> pl.DataFrame:
    np.random.seed(42)
    base_price = 85000.0
    returns = np.random.normal(0, 0.4, n_rows)
    mid_price = base_price + np.cumsum(returns)
    spread = np.random.uniform(0.2, 0.8, n_rows)

    bid_p0 = mid_price - spread / 2.0
    ask_p0 = mid_price + spread / 2.0
    bid_v0 = np.random.exponential(1.5, n_rows) + 0.1
    ask_v0 = np.random.exponential(1.5, n_rows) + 0.1
    bid_v5 = bid_v0 * 5.0
    ask_v5 = ask_v0 * 5.0

    qi_5 = (bid_v5 - ask_v5) / (bid_v5 + ask_v5)
    qi_1 = (bid_v0 - ask_v0) / (bid_v0 + ask_v0)
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
        "qi_1": qi_1,
        "qi_5": qi_5,
        "qi_20": qi_5 * 0.95,
        "count_skew_5": np.zeros(n_rows),
        "ofi": np.random.normal(0, 1.0, n_rows),
        "vol_30s_bps": np.random.uniform(8.0, 15.0, n_rows),
        "micro_dev": micro_dev
    })
    return df


class TestHFTDualModeEngine(unittest.TestCase):
    def setUp(self):
        self.df = create_test_orderbook_df(600)

    def test_avellaneda_stoikov_inventory_management(self):
        """Tests that Avellaneda-Stoikov tracks inventory bounds and computes reservation prices."""
        as_sim = AvellanedaStoikovSimulator(MarketMakingConfig(
            gamma=0.05,
            max_inventory=3,
            maker_fee_bps=0.05
        ))
        res = as_sim.simulate(self.df, 'qi_5')
        self.assertIsInstance(res.trades_count, int)
        self.assertLessEqual(res.max_inventory, 4)
        self.assertGreater(len(res.equity_curve), 0)
        self.assertGreater(len(res.inventory_curve), 0)

    def test_fifo_queue_simulator_institutional_tier(self):
        """Tests FIFO queue simulation with institutional fee tiers."""
        sim = FIFOQueueSimulator(FIFOQueueConfig(
            queue_placement="MID",
            maker_fee_bps=0.05,
            taker_fee_bps=1.5,
            holding_period=20
        ))
        res = sim.simulate(self.df, 'qi_5', threshold=0.25)
        self.assertIsInstance(res.trades_count, int)
        self.assertIsInstance(res.sharpe_ratio, float)
        self.assertIsInstance(res.fill_rate_pct, float)

    def test_red_team_auditor_dynamic_parameters(self):
        """Tests that RedTeamAuditor accepts dynamic threshold and holding period."""
        auditor = RedTeamAuditor(min_dsr=0.50, min_sharpe=0.50)
        
        def signal_fn(frame):
            return (frame['qi_5'] * 0.7 + frame['qi_1'] * 0.3).clip(-1.0, 1.0)

        report = auditor.audit_implementation(
            signal_fn=signal_fn,
            df=self.df,
            total_trials=1,
            threshold=0.30,
            holding_period=20
        )
        self.assertIn("verdict", report)
        self.assertIn("dsr", report)
        self.assertIn("pbo", report)
        self.assertTrue(report["lookahead_passed"])


if __name__ == "__main__":
    unittest.main()
