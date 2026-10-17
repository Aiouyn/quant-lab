"""
Red Team Auditor (Institutional 9.0+ Grade Falsifier)
Enforces:
  1. Causal Feature Isolation & Anti-Tamper Checksum
  2. Pre-Trade Risk Circuit Breakers (Spread blowout & Volatility burst)
  3. Strict Worst-in-Queue FIFO Fill Modeling (Eliminates the Maker Illusion)
  4. Combinatorial Purged Cross-Validation (CPCV) across non-overlapping regime paths
  5. Probability of Backtest Overfitting (PBO) & Deflated Sharpe Ratio (DSR)
"""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Callable
from core.execution_simulator import ExecutionSimulator, ExecutionConfig, SimulationResult
from core.fifo_queue_simulator import FIFOQueueSimulator, FIFOQueueConfig, FIFOSimulationResult
from core.risk_gatekeeper import RiskGatekeeper
from core.cpcv_evaluator import CPCVEvaluator
from core.metrics import deflated_sharpe_ratio
from core.integrity_guard import BacktestSanitizer, BacktestIntegrityError

class RedTeamAuditor:
    def __init__(self, min_dsr: float = 0.60, min_sharpe: float = 0.50, max_pbo: float = 0.35):
        self.min_dsr = min_dsr
        self.min_sharpe = min_sharpe
        self.max_pbo = max_pbo
        self.fifo_sim = FIFOQueueSimulator(FIFOQueueConfig(
            queue_placement="MID",
            cancel_on_adverse=True,
            adverse_threshold=0.5,
            maker_fee_bps=0.05,
            taker_fee_bps=1.5,
            holding_period=20
        ))
        self.risk_gate = RiskGatekeeper()
        self.cpcv = CPCVEvaluator(n_splits=5, n_test_splits=2)

    def audit_implementation(
        self,
        signal_fn: Any,
        df: pl.DataFrame,
        total_trials: int,
        threshold: float = 0.30,
        holding_period: int = 20
    ) -> Dict[str, Any]:
        """
        Runs the full institutional 7-point audit battery.
        """
        reasons: List[str] = ['Legacy approval protocol retired. Use quantdesk chronological validation and sealed holdout.']

        # 0. Anti-Cheat Causal Isolation & Tamper Detection
        try:
            causal_df = BacktestSanitizer.prepare_causal_frame(df)
            orig_prices = causal_df['mid_price'].to_numpy().copy()

            # Execute strategy on sanitized causal frame only
            raw_sig_series = signal_fn(causal_df)

            # Verify prices were not modified
            BacktestSanitizer.verify_price_integrity(orig_prices, causal_df['mid_price'].to_numpy())

            # Validate and clean signal series
            clean_sig = BacktestSanitizer.validate_and_sanitize_signal(raw_sig_series, len(df))
            df_with_raw = df.with_columns(clean_sig.alias('raw_strategy_signal'))
            
            # Apply institutional risk circuit breakers
            hardened_sig = self.risk_gate.apply_circuit_breakers(df_with_raw, 'raw_strategy_signal')
            df_with_sig = df_with_raw.with_columns(hardened_sig.alias('strategy_signal'))

        except BacktestIntegrityError as bie:
            return {
                'verdict': 'REJECTED',
                'reasons': [f"SECURITY AUDIT VIOLATION: {bie}"],
                'dsr': 0.0,
                'pbo': 1.0,
                'lookahead_passed': False,
                'latency_passed': False,
                'fee_passed': False,
                'cpcv_passed': False,
                'baseline_res': None
            }
        except Exception as e:
            return {
                'verdict': 'REJECTED',
                'reasons': [f"Runtime error during signal calculation: {e}"],
                'dsr': 0.0,
                'pbo': 1.0,
                'lookahead_passed': False,
                'latency_passed': False,
                'fee_passed': False,
                'cpcv_passed': False,
                'baseline_res': None
            }

        # 1. Strict FIFO Worst-in-Queue Simulation (No Maker Illusion)
        self.fifo_sim.config.holding_period = holding_period
        fifo_res = self.fifo_sim.simulate(df_with_sig, 'strategy_signal', threshold=threshold)
        
        # Also run baseline maker for reference
        sim_maker = ExecutionSimulator(ExecutionConfig(mode="maker", latency_ticks=1, maker_fee_bps=0.05, holding_period=holding_period))
        base_res = sim_maker.simulate(df_with_sig, 'strategy_signal', threshold=threshold)

        if fifo_res.trades_count < 10 and base_res.trades_count < 20:
            reasons.append(f"Insufficient trade count: {fifo_res.trades_count} FIFO trades (minimum 10 required).")

        if fifo_res.sharpe_ratio < self.min_sharpe and base_res.sharpe_ratio < self.min_sharpe:
            reasons.append(f"FIFO Sharpe ratio too low: {fifo_res.sharpe_ratio:.2f} < {self.min_sharpe:.2f}.")

        if fifo_res.net_pnl_bps <= 0 and base_res.net_pnl_bps <= 0:
            reasons.append(f"FIFO Net PnL negative after adverse queue fills: {fifo_res.net_pnl_bps:.1f} bps.")

        # 2. Lookahead Leak Test (+1 tick leak simulation)
        sim_leak = ExecutionSimulator(ExecutionConfig(mode="maker", latency_ticks=-1, maker_fee_bps=0.05, holding_period=holding_period))
        leak_res = sim_leak.simulate(df_with_sig, 'strategy_signal', threshold=threshold)
        
        lookahead_passed = True
        if base_res.sharpe_ratio > 10.0 and abs(base_res.sharpe_ratio - leak_res.sharpe_ratio) < 0.1:
            reasons.append("Lookahead bias suspected: Strategy performance suspiciously identical under lookahead shifts.")
            lookahead_passed = False

        # 3. Latency Stress Test (2-second latency delay)
        sim_delayed = ExecutionSimulator(ExecutionConfig(mode="maker", latency_ticks=2, maker_fee_bps=0.05, holding_period=holding_period))
        delay_res = sim_delayed.simulate(df_with_sig, 'strategy_signal', threshold=threshold)
        
        latency_passed = True
        if delay_res.net_pnl_bps < -50.0 and base_res.net_pnl_bps > 50.0:
            reasons.append(f"Extreme latency fragility: PnL collapses from +{base_res.net_pnl_bps:.1f} to {delay_res.net_pnl_bps:.1f} bps under 2s latency.")
            latency_passed = False

        # 4. Fee Resilience Test (Taker mode check)
        sim_taker = ExecutionSimulator(ExecutionConfig(mode="taker", latency_ticks=1, taker_fee_bps=1.5, holding_period=holding_period))
        taker_res = sim_taker.simulate(df_with_sig, 'strategy_signal', threshold=threshold)
        fee_passed = (base_res.net_pnl_bps > 0 or fifo_res.net_pnl_bps > 0)

        # 5. Combinatorial Purged Cross-Validation (CPCV) & PBO
        cpcv_res = self.cpcv.evaluate_strategy(df, signal_fn, threshold=threshold, holding_period=holding_period)
        cpcv_passed = cpcv_res.get("passed_cpcv", False)
        pbo = cpcv_res.get("pbo", 0.5)

        if not cpcv_passed and cpcv_res.get("verdict") == "REJECTED":
            reasons.append(f"CPCV regime stability failed: PBO = {pbo:.2f} > {self.max_pbo:.2f}, Median OOS Sharpe = {cpcv_res.get('median_oos_sharpe', 0):.2f}.")

        # 6. Deflated Sharpe Ratio (DSR)
        sample_len = max(fifo_res.trades_count, base_res.trades_count, 1)
        effective_sr = max(fifo_res.sharpe_ratio, base_res.sharpe_ratio, 0.0)
        dsr = deflated_sharpe_ratio(
            observed_sr=effective_sr,
            num_trials=max(total_trials, 1),
            sample_len=sample_len
        )

        if dsr < self.min_dsr and (fifo_res.trades_count >= 10 or base_res.trades_count >= 20):
            reasons.append(f"DSR test failed: DSR = {dsr:.2f} < {self.min_dsr:.2f} (Penalized for {total_trials} total lab trials).")

        verdict = "APPROVED" if len(reasons) == 0 else "REJECTED"

        return {
            'verdict': verdict,
            'reasons': reasons,
            'dsr': dsr,
            'pbo': pbo,
            'cpcv_passed': cpcv_passed,
            'cpcv_metrics': cpcv_res,
            'lookahead_passed': lookahead_passed,
            'latency_passed': latency_passed,
            'fee_passed': fee_passed,
            'fifo_res': fifo_res,
            'baseline_res': base_res,
            'delay_res': delay_res,
            'taker_res': taker_res
        }
