"""
Unit tests for Portfolio Ensemble Engine and First-Principles Diagnostics
"""

import unittest
import numpy as np
import polars as pl
from core.portfolio_ensemble import PortfolioEnsembleEngine, PortfolioStrategy
from core.first_principles_diagnostics import FirstPrinciplesDiagnostics
from core.fifo_queue_simulator import FIFOSimulationResult


class TestPortfolioAndDiagnostics(unittest.TestCase):
    def setUp(self):
        self.engine = PortfolioEnsembleEngine(db_path=":memory:")
        self.diags = FirstPrinciplesDiagnostics()

    def test_portfolio_allocation_and_rebalance(self):
        code1 = (
            "import polars as pl\n"
            "def compute_signal(df: pl.DataFrame) -> pl.Series:\n"
            "    return df['qi_5'].fill_null(0.0)\n"
        )
        self.engine.add_approved_strategy(
            strategy_id=1,
            title="Alpha 1",
            theorist="Theorist_A",
            sharpe=2.5,
            net_pnl_bps=15.0,
            code_snippet=code1,
            holding_period=30
        )
        self.assertEqual(len(self.engine.strategies), 1)
        self.assertAlmostEqual(self.engine.strategies[0].weight, 1.0)

        code2 = (
            "import polars as pl\n"
            "def compute_signal(df: pl.DataFrame) -> pl.Series:\n"
            "    return (df['micro_dev'] / df['spread'].clip(lower_bound=0.1)).fill_null(0.0)\n"
        )
        self.engine.add_approved_strategy(
            strategy_id=2,
            title="Alpha 2",
            theorist="Theorist_B",
            sharpe=1.5,
            net_pnl_bps=8.0,
            code_snippet=code2,
            holding_period=30
        )
        self.assertEqual(len(self.engine.strategies), 2)
        # Weights should sum to 1.0
        total_w = sum(s.weight for s in self.engine.strategies)
        self.assertAlmostEqual(total_w, 1.0)
        # Higher Sharpe gets higher weight
        self.assertGreater(self.engine.strategies[0].weight, self.engine.strategies[1].weight)

    def test_first_principles_diagnostics(self):
        # 1. Adverse selection failure
        mock_fifo = FIFOSimulationResult(
            trades_count=20,
            fill_rate_pct=50.0,
            win_rate=40.0,
            gross_pnl_bps=10.0,
            fee_drag_bps=2.0,
            net_pnl_bps=-5.0,
            mean_trade_bps=-0.25,
            sharpe_ratio=-1.5,
            sortino_ratio=-1.0,
            max_drawdown_bps=8.0,
            adverse_selection_bps=1.2,
            unfilled_orders_count=10,
            equity_curve=np.zeros(20)
        )
        audit_res = {
            'verdict': 'REJECTED',
            'reasons': ['Adverse selection exceeded threshold.'],
            'fifo_res': mock_fifo,
            'baseline_res': None
        }
        decomp = self.diags.analyze_audit_failure("Test Alpha", "Test Theorist", audit_res)
        self.assertEqual(decomp.root_cause, "ADVERSE_SELECTION_TOXICITY")
        self.assertEqual(decomp.severity, "HIGH")
        self.assertIn("Order Flow Imbalance", decomp.prescriptive_directive)

        # 2. Queue starvation failure
        mock_fifo_starve = FIFOSimulationResult(
            trades_count=2,
            fill_rate_pct=10.0,
            win_rate=50.0,
            gross_pnl_bps=2.0,
            fee_drag_bps=0.2,
            net_pnl_bps=1.8,
            mean_trade_bps=0.9,
            sharpe_ratio=1.0,
            sortino_ratio=1.0,
            max_drawdown_bps=1.0,
            adverse_selection_bps=0.1,
            unfilled_orders_count=18,
            equity_curve=np.zeros(2)
        )
        audit_res_starve = {
            'verdict': 'REJECTED',
            'reasons': ['Insufficient trade count.'],
            'fifo_res': mock_fifo_starve,
            'baseline_res': None
        }
        decomp2 = self.diags.analyze_audit_failure("Starve Alpha", "Theorist 2", audit_res_starve)
        self.assertEqual(decomp2.root_cause, "QUEUE_STARVATION")
        self.assertIn("entry threshold", decomp2.prescriptive_directive)


if __name__ == '__main__':
    unittest.main()
