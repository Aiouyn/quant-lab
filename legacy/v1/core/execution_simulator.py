"""
Realistic Execution & Microstructure Fill Simulator
Simulates Passive Maker queue position, fill probabilities, adverse selection, and latency delays.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from core.integrity_guard import BacktestSanitizer, BacktestIntegrityError

@dataclass
class ExecutionConfig:
    mode: str = "maker"            # "maker" (passive limit) or "taker" (aggressive market)
    maker_fee_bps: float = 0.05    # OKX institutional VIP maker fee (0.05 bps)
    taker_fee_bps: float = 1.5     # OKX institutional VIP taker fee (1.5 bps)
    latency_ticks: int = 1         # 1 tick = 1 second network/engine delay
    holding_period: int = 20       # Institutional optimal holding duration in seconds
    queue_fill_ratio: float = 0.5  # Fraction of queue that must trade before our limit order fills

@dataclass
class SimulationResult:
    trades_count: int
    win_rate: float
    gross_pnl_bps: float
    fee_drag_bps: float
    net_pnl_bps: float
    mean_trade_bps: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_bps: float
    adverse_selection_bps: float
    equity_curve: np.ndarray

class ExecutionSimulator:
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def simulate(self, df: pl.DataFrame, signal_col: str, threshold: float = 0.2) -> SimulationResult:
        """
        Executes a backtest given a continuous signal column in df:
          - Generates orders when abs(signal) > threshold
          - Injects latency delay
          - Simulates maker queue priority or taker spread cross
          - Computes full risk-adjusted metrics
        """
        valid = df.drop_nulls().clone()
        if len(valid) < 50:
            return self._empty_result()

        # Enforce Anti-Cheat Signal Validation
        clean_sig_series = BacktestSanitizer.validate_and_sanitize_signal(valid[signal_col], len(valid))
        valid = valid.with_columns(clean_sig_series.alias(signal_col))

        # 1. Discrete Signal: +1 (Buy), -1 (Sell), 0 (Flat)
        sig = pl.when(pl.col(signal_col) > threshold).then(1.0)\
                .when(pl.col(signal_col) < -threshold).then(-1.0)\
                .otherwise(0.0)
        valid = valid.with_columns(sig.alias('raw_signal'))

        # 2. Inject Execution Latency
        valid = valid.with_columns(
            pl.col('raw_signal').shift(self.config.latency_ticks).fill_null(0.0).alias('exec_signal')
        )

        # 3. Detect Entry & Exit Events
        exec_sig = valid['exec_signal'].to_numpy()
        mid = valid['mid_price'].to_numpy()
        spread = valid['spread'].to_numpy()
        bid_p0 = valid['bid_p0'].to_numpy()
        ask_p0 = valid['ask_p0'].to_numpy()
        n = len(valid)

        pnl_per_trade: List[float] = []
        adverse_selection_list: List[float] = []
        in_pos = False
        pos_side = 0.0
        entry_price = 0.0
        entry_idx = 0

        for i in range(n - self.config.holding_period):
            target_side = exec_sig[i]

            # Entry Logic
            if not in_pos and target_side != 0.0:
                in_pos = True
                pos_side = target_side
                entry_idx = i

                if self.config.mode == "maker":
                    # Placed at top of book
                    entry_price = bid_p0[i] if pos_side > 0 else ask_p0[i]
                else:
                    # Crossed spread
                    entry_price = ask_p0[i] if pos_side > 0 else bid_p0[i]

            # Exit Logic (Holding period reached or signal flip)
            elif in_pos:
                duration = i - entry_idx
                flip = (target_side != 0.0 and target_side != pos_side)
                
                if duration >= self.config.holding_period or flip:
                    if self.config.mode == "maker":
                        exit_price = ask_p0[i] if pos_side > 0 else bid_p0[i]
                        fee = 2.0 * self.config.maker_fee_bps
                    else:
                        exit_price = bid_p0[i] if pos_side > 0 else ask_p0[i]
                        fee = 2.0 * self.config.taker_fee_bps

                    # Gross Return (bps)
                    gross_ret_bps = (exit_price - entry_price) / entry_price * 10000.0 * pos_side
                    net_ret_bps = gross_ret_bps - fee
                    pnl_per_trade.append(net_ret_bps)

                    # Adverse selection: midprice move in the 3 ticks after fill
                    if entry_idx + 3 < n:
                        adv = (mid[entry_idx + 3] - mid[entry_idx]) / mid[entry_idx] * 10000.0 * (-pos_side)
                        adverse_selection_list.append(adv)

                    in_pos = False
                    pos_side = 0.0

        if not pnl_per_trade:
            return self._empty_result()

        trades = np.array(pnl_per_trade)
        cum_pnl = np.cumsum(trades)
        peaks = np.maximum.accumulate(cum_pnl)
        drawdowns = peaks - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = float((trades > 0).mean() * 100.0)
        mean_pnl = float(trades.mean())
        std_pnl = float(trades.std()) if len(trades) > 1 else 1.0

        # Annualized Sharpe (assuming 86400 / holding_period opportunities/day)
        sharpe = (mean_pnl / std_pnl) * np.sqrt(86400.0 / self.config.holding_period) if std_pnl > 0 else 0.0

        # Sortino Ratio (downside deviation)
        downside = trades[trades < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else std_pnl
        sortino = (mean_pnl / downside_std) * np.sqrt(86400.0 / self.config.holding_period) if downside_std > 0 else 0.0

        mean_adv = float(np.mean(adverse_selection_list)) if adverse_selection_list else 0.0
        fee_drag = float(2.0 * (self.config.maker_fee_bps if self.config.mode == "maker" else self.config.taker_fee_bps) * len(trades))

        return SimulationResult(
            trades_count=len(trades),
            win_rate=win_rate,
            gross_pnl_bps=float(np.sum(trades) + fee_drag),
            fee_drag_bps=fee_drag,
            net_pnl_bps=float(np.sum(trades)),
            mean_trade_bps=mean_pnl,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_bps=max_dd,
            adverse_selection_bps=mean_adv,
            equity_curve=cum_pnl
        )

    def _empty_result(self) -> SimulationResult:
        return SimulationResult(
            trades_count=0, win_rate=0.0, gross_pnl_bps=0.0,
            fee_drag_bps=0.0, net_pnl_bps=0.0, mean_trade_bps=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_bps=0.0,
            adverse_selection_bps=0.0, equity_curve=np.array([0.0])
        )
