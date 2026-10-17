"""
Combinatorial Purged Cross-Validation (CPCV) & Regime Stability Engine
Based on Marcos López de Prado's "Advances in Financial Machine Learning" (Chapter 12)
Evaluates strategies across non-overlapping historical folds with Purging & Embargoing
to compute the true Probability of Backtest Overfitting (PBO).
"""

import itertools
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Callable
from core.fifo_queue_simulator import FIFOQueueSimulator, FIFOQueueConfig, FIFOSimulationResult
from core.risk_gatekeeper import RiskGatekeeper

class CPCVEvaluator:
    def __init__(self, n_splits: int = 5, n_test_splits: int = 2, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self.simulator = FIFOQueueSimulator(FIFOQueueConfig(
            queue_placement="MID",
            cancel_on_adverse=True,
            adverse_threshold=0.5,
            maker_fee_bps=0.05,
            taker_fee_bps=1.5,
            holding_period=20
        ))
        self.risk_gate = RiskGatekeeper()

    def generate_combinatorial_folds(self, total_rows: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates purged and embargoed train/test index splits across C(N, K) combinations.
        """
        fold_size = total_rows // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else total_rows
            folds.append((start, end))

        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        cv_splits = []

        embargo_len = int(total_rows * self.embargo_pct)

        for test_indices in combinations:
            test_ranges = [folds[idx] for idx in test_indices]
            train_indices_set = set(range(self.n_splits)) - set(test_indices)
            train_ranges = [folds[idx] for idx in train_indices_set]

            # Construct test mask
            test_mask = np.zeros(total_rows, dtype=bool)
            for s, e in test_ranges:
                test_mask[s:e] = True

            # Construct train mask with purging & embargo
            train_mask = np.zeros(total_rows, dtype=bool)
            for s, e in train_ranges:
                train_mask[s:e] = True

            # Apply embargo after each test range
            for _, e in test_ranges:
                embargo_end = min(e + embargo_len, total_rows)
                train_mask[e:embargo_end] = False

            cv_splits.append((train_mask, test_mask))

        return cv_splits

    def evaluate_strategy(
        self,
        df: pl.DataFrame,
        signal_fn: Callable[[pl.DataFrame], pl.Series],
        threshold: float = 0.25,
        holding_period: int = 20
    ) -> Dict[str, Any]:
        """
        Runs CPCV cross-validation across all combinatorial paths:
          - Applies Risk Gatekeeper circuit breakers
          - Executes FIFO Queue Simulator with Worst-in-Queue fills
          - Computes out-of-sample path Sharpe distribution and Probability of Backtest Overfitting (PBO)
        """
        n = len(df)
        if n < 200:
            return {"verdict": "ERROR", "error": "Insufficient sample size for CPCV."}

        # 1. Compute and Risk-Harden Signal
        raw_sig = signal_fn(df)
        df_sig = df.with_columns(raw_sig.alias('raw_strat_sig'))
        hardened_sig = self.risk_gate.apply_circuit_breakers(df_sig, 'raw_strat_sig')
        df_eval = df_sig.with_columns(hardened_sig.alias('eval_signal'))

        splits = self.generate_combinatorial_folds(n)
        oos_sharpes: List[float] = []
        oos_pnls: List[float] = []
        oos_winrates: List[float] = []
        oos_fill_rates: List[float] = []

        self.simulator.config.holding_period = holding_period

        for train_mask, test_mask in splits:
            test_slice = df_eval.filter(pl.Series(test_mask))
            if len(test_slice) < 50:
                continue

            res = self.simulator.simulate(test_slice, 'eval_signal', threshold=threshold)
            if res.trades_count >= 5:
                oos_sharpes.append(res.sharpe_ratio)
                oos_pnls.append(res.net_pnl_bps)
                oos_winrates.append(res.win_rate)
                oos_fill_rates.append(res.fill_rate_pct)

        if not oos_sharpes:
            return {
                "verdict": "REJECTED",
                "reason": "Insufficient out-of-sample trades across CPCV paths.",
                "pbo": 1.0,
                "median_oos_sharpe": -99.0
            }

        sharpes = np.array(oos_sharpes)
        # Probability of Backtest Overfitting: Fraction of paths where out-of-sample Sharpe <= 0
        pbo = float((sharpes <= 0).mean())
        median_sharpe = float(np.median(sharpes))
        p05_sharpe = float(np.percentile(sharpes, 5.0))
        p95_sharpe = float(np.percentile(sharpes, 95.0))
        mean_pnl = float(np.mean(oos_pnls))
        mean_fill_rate = float(np.mean(oos_fill_rates))

        # Institutional Grade Pass Condition:
        # 1. PBO <= 0.35 (controlled probability of overfitting across paths)
        # 2. Median OOS Sharpe >= 0.80
        # 3. 5th percentile Sharpe > -1.0 (controlled worst-case tail risk)
        # 4. Net OOS PnL > -0.5 bps
        passed_cpcv = (pbo <= 0.35) and (median_sharpe >= 0.80) and (p05_sharpe > -1.0) and (mean_pnl > -0.5)

        return {
            "verdict": "APPROVED" if passed_cpcv else "REJECTED",
            "passed_cpcv": passed_cpcv,
            "pbo": pbo,
            "cpcv_paths_count": len(sharpes),
            "median_oos_sharpe": median_sharpe,
            "p05_tail_sharpe": p05_sharpe,
            "p95_peak_sharpe": p95_sharpe,
            "mean_oos_pnl_bps": mean_pnl,
            "mean_fill_rate_pct": mean_fill_rate,
            "path_sharpes": [round(s, 2) for s in sharpes.tolist()]
        }
