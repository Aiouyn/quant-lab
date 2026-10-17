"""
Unit & Integration Tests for Quant Lab Core Engine
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import polars as pl
from core.metrics import compute_ic, deflated_sharpe_ratio, minimum_track_record_length
from core.execution_simulator import ExecutionSimulator, ExecutionConfig
from core.knowledge_base import ResearchLedger

class TestCoreModules(unittest.TestCase):
    def test_metrics(self):
        x = np.linspace(1.0, 10.0, 20)
        y = x * 1.5 + np.random.randn(20) * 0.1
        ic = compute_ic(x, y)
        self.assertGreater(ic['pearson'], 0.95)

        # Deflated Sharpe test
        dsr = deflated_sharpe_ratio(observed_sr=2.5, num_trials=10, sample_len=100)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_execution_simulator(self):
        # Create synthetic orderbook slice
        n = 200
        mid = 85000.0 + np.cumsum(np.random.randn(n) * 2.0)
        df = pl.DataFrame({
            'mid_price': mid,
            'bid_p0': mid - 0.5,
            'ask_p0': mid + 0.5,
            'spread': np.ones(n),
            'test_signal': np.sin(np.linspace(0, 10, n))
        })

        sim = ExecutionSimulator(ExecutionConfig(mode="maker", latency_ticks=1))
        res = sim.simulate(df, 'test_signal', threshold=0.5)
        self.assertIsInstance(res.trades_count, int)
        self.assertIsInstance(res.sharpe_ratio, float)

    def test_knowledge_base(self):
        db_path = r"C:\Users\Mathis\.gemini\antigravity\scratch\quant_lab\database\test_ledger_mem.db"
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        kb = ResearchLedger(db_path)
        c_id = kb.create_campaign("Test Campaign", "Test Objective")
        self.assertEqual(c_id, 1)

        h_id = kb.record_hypothesis(c_id, "OFI Theorist", "Test OFI", "Ratio", "OFI / Spread", "5s")
        self.assertEqual(h_id, 1)

        trials = kb.get_total_trials_count()
        self.assertEqual(trials, 0)

if __name__ == '__main__':
    unittest.main()
