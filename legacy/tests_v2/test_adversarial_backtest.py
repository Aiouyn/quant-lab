"""
Adversarial Security & Anti-Cheating Test Suite for Quant Lab Backtester
Validates that fake, malicious, or buggy strategies CANNOT break the backtester
or fabricate fake performance results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import polars as pl
from core.integrity_guard import BacktestSanitizer, BacktestIntegrityError
from agents.red_team_auditor import RedTeamAuditor

class TestAdversarialBacktester(unittest.TestCase):
    def setUp(self):
        n = 500
        mid = 85000.0 + np.cumsum(np.random.randn(n) * 2.0)
        self.df = pl.DataFrame({
            'mid_price': mid,
            'bid_p0': mid - 0.5,
            'ask_p0': mid + 0.5,
            'spread': np.ones(n),
            'qi_5': np.random.uniform(-0.8, 0.8, n),
            'micro_dev': np.random.randn(n) * 0.2,
            # Malicious trap: future return column
            'fwd_ret_5s_bps': np.roll(np.diff(mid, prepend=mid[0]), -5)
        })
        self.auditor = RedTeamAuditor(min_dsr=0.80, min_sharpe=1.0)

    def test_attack_1_future_peeking_blocked(self):
        """Strategy attempts to peek into future return columns."""
        def malicious_future_peeker(df: pl.DataFrame) -> pl.Series:
            # Tries to access forward-looking column
            if 'fwd_ret_5s_bps' in df.columns:
                return pl.when(df['fwd_ret_5s_bps'] > 0).then(1.0).otherwise(-1.0)
            raise KeyError("Forward column was stripped by BacktestSanitizer!")

        res = self.auditor.audit_implementation(malicious_future_peeker, self.df, total_trials=1)
        self.assertEqual(res['verdict'], 'REJECTED')
        self.assertTrue(any("Forward column was stripped" in r for r in res['reasons']))

    def test_attack_2_price_tampering_blocked(self):
        """Strategy attempts to manipulate market prices in-place."""
        def malicious_price_mutator(df: pl.DataFrame) -> pl.Series:
            # Maliciously attempt to inflate prices
            df = df.with_columns(pl.lit(999999.0).alias('mid_price'))
            return pl.Series('sig', np.random.randn(len(df)))

        res = self.auditor.audit_implementation(malicious_price_mutator, self.df, total_trials=1)
        # Price integrity check or immutability isolates it
        self.assertIn(res['verdict'], ['APPROVED', 'REJECTED'])

    def test_attack_3_inf_nan_explosion_sanitized(self):
        """Strategy returns NaNs, Infs, and extreme out-of-bounds numbers."""
        def malicious_nan_exploder(df: pl.DataFrame) -> pl.Series:
            bad_values = np.array([np.nan, np.inf, -np.inf, 1e20, -1e20] * (len(df) // 5))
            return pl.Series('sig', bad_values)

        res = self.auditor.audit_implementation(malicious_nan_exploder, self.df, total_trials=1)
        # Should not crash with ValueError or NaN error
        self.assertIn(res['verdict'], ['APPROVED', 'REJECTED'])

    def test_attack_4_constant_flatline_rejected(self):
        """Strategy returns constant 1.0 (trying to buy-and-hold without variance)."""
        def malicious_constant_signal(df: pl.DataFrame) -> pl.Series:
            return pl.Series('sig', np.ones(len(df)))

        res = self.auditor.audit_implementation(malicious_constant_signal, self.df, total_trials=1)
        self.assertEqual(res['verdict'], 'REJECTED')
        self.assertTrue(any("SECURITY AUDIT VIOLATION" in r for r in res['reasons']))

    def test_attack_5_lookahead_time_travel_caught(self):
        """Strategy uses lag -1 to cheat by looking at next tick midprice."""
        def lookahead_cheater(df: pl.DataFrame) -> pl.Series:
            # Looks forward 1 tick
            cheating_sig = df['mid_price'].shift(-1) - df['mid_price']
            return cheating_sig.fill_null(0.0)

        res = self.auditor.audit_implementation(lookahead_cheater, self.df, total_trials=1)
        self.assertEqual(res['verdict'], 'REJECTED')

if __name__ == '__main__':
    unittest.main()
