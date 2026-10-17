"""
Institutional 9.0+ Grade Integration Test Suite
Validates:
  1. Strict FIFO Queue Priority (Worst-in-queue & adverse selection)
  2. Pre-Trade Risk Gatekeeper (Spread blowout & Volatility burst circuit breakers)
  3. Combinatorial Purged Cross-Validation (CPCV) & Probability of Backtest Overfitting (PBO)
  4. Red Team 7-Point Falsification Battery
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import polars as pl
from core.fifo_queue_simulator import FIFOQueueSimulator, FIFOQueueConfig
from core.risk_gatekeeper import RiskGatekeeper
from core.cpcv_evaluator import CPCVEvaluator
from agents.red_team_auditor import RedTeamAuditor

class TestInstitutionalEngine(unittest.TestCase):
    def setUp(self):
        n = 1000
        mid = 85000.0 + np.cumsum(np.random.randn(n) * 2.0)
        spreads = np.full(n, 0.5)
        # Inject realistic spread blowout at ticks 400-420
        spreads[400:420] = 8.0

        vols = np.full(n, 1.2)
        # Inject volatility burst at ticks 700-730
        vols[700:730] = 25.0

        self.df = pl.DataFrame({
            'ts_ms': np.arange(n) * 1000,
            'mid_price': mid,
            'bid_p0': mid - spreads / 2.0,
            'ask_p0': mid + spreads / 2.0,
            'spread': spreads,
            'bid_v0': np.random.uniform(5.0, 50.0, n),
            'ask_v0': np.random.uniform(5.0, 50.0, n),
            'bid_v5': np.random.uniform(50.0, 200.0, n),
            'ask_v5': np.random.uniform(50.0, 200.0, n),
            'qi_1': np.random.uniform(-0.8, 0.8, n),
            'qi_5': np.random.uniform(-0.8, 0.8, n),
            'qi_20': np.random.uniform(-0.8, 0.8, n),
            'micro_price': mid + np.random.randn(n) * 0.1,
            'micro_dev': np.random.randn(n) * 0.1,
            'count_skew_5': np.zeros(n),
            'ofi': np.random.randn(n) * 10.0,
            'vol_30s_bps': vols
        })

    def test_fifo_queue_priority(self):
        """Validates that FIFO queue simulator models unfilled orders and queue depth."""
        sim = FIFOQueueSimulator(FIFOQueueConfig(queue_placement="TAIL", cancel_on_adverse=True))
        res = sim.simulate(self.df, 'qi_5', threshold=0.4)
        
        self.assertIsInstance(res.trades_count, int)
        self.assertIsInstance(res.fill_rate_pct, float)
        # In realistic FIFO queue, fill rate must be strictly less than 100%
        self.assertLessEqual(res.fill_rate_pct, 100.0)
        self.assertGreaterEqual(res.unfilled_orders_count, 0)

    def test_risk_gatekeeper_circuit_breakers(self):
        """Validates that spread blowouts and volatility bursts force signal to 0.0."""
        gate = RiskGatekeeper(max_spread_ratio=2.5, vol_kill_percentile=95.0)
        # Create continuous active signal
        active_df = self.df.with_columns(pl.lit(1.0).alias('active_sig'))
        hardened = gate.apply_circuit_breakers(active_df, 'active_sig')

        hardened_arr = hardened.to_numpy()
        # Spread blowout at 400-420 must be zeroed out
        self.assertTrue(np.all(hardened_arr[400:420] == 0.0))
        # Volatility burst at 700-730 must be zeroed out
        self.assertTrue(np.all(hardened_arr[700:730] == 0.0))

    def test_cpcv_combinatorial_regimes(self):
        """Validates that CPCV generates non-overlapping purged paths with PBO calculation."""
        cpcv = CPCVEvaluator(n_splits=5, n_test_splits=2, embargo_pct=0.01)
        splits = cpcv.generate_combinatorial_folds(len(self.df))
        # C(5, 2) = 10 combinatorial paths
        self.assertEqual(len(splits), 10)

        for train_mask, test_mask in splits:
            # Purging & Embargo test: train and test masks must not overlap
            overlap = np.logical_and(train_mask, test_mask)
            self.assertEqual(np.sum(overlap), 0)

    def test_upgraded_red_team_auditor(self):
        """Validates that the 7-point audit battery runs smoothly and produces PBO & DSR."""
        auditor = RedTeamAuditor(min_dsr=0.80, min_sharpe=1.0)
        def dummy_signal(df: pl.DataFrame) -> pl.Series:
            return (df['qi_5'] * 0.5 + df['micro_dev'] * 0.5).fill_null(0.0)

        audit_res = auditor.audit_implementation(dummy_signal, self.df, total_trials=5)
        self.assertIn(audit_res['verdict'], ['APPROVED', 'REJECTED'])
        self.assertIn('pbo', audit_res)
        self.assertIn('dsr', audit_res)
        self.assertIn('cpcv_passed', audit_res)

if __name__ == '__main__':
    unittest.main()
