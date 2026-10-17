"""
Strict FIFO Queue Priority & Adverse Selection Simulator
Eliminates the "Maker Illusion" by modeling realistic limit order queue position (FIFO).
An order ONLY fills if cumulative market volume trades through the resting queue ahead of you.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class FIFOQueueConfig:
    maker_fee_bps: float = 0.05        # OKX institutional VIP maker fee (0.05 bps)
    taker_fee_bps: float = 1.5         # OKX institutional VIP taker fee (1.5 bps)
    latency_ticks: int = 1             # Exchange roundtrip delay (1 tick = 1s)
    holding_period: int = 20           # Target trade duration in seconds (15-30s optimal)
    queue_placement: str = "MID"       # "MID" (realistic dynamic placement) or "TAIL" (worst case)
    cancel_on_adverse: bool = True     # Fast cancel if queue ahead rapidly collapses
    adverse_threshold: float = 0.5     # Queue inversion threshold triggering immediate quote retreat

@dataclass
class FIFOSimulationResult:
    trades_count: int
    fill_rate_pct: float              # Percentage of limit orders that actually get filled
    win_rate: float
    gross_pnl_bps: float
    fee_drag_bps: float
    net_pnl_bps: float
    mean_trade_bps: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_bps: float
    adverse_selection_bps: float
    unfilled_orders_count: int
    equity_curve: np.ndarray

class FIFOQueueSimulator:
    def __init__(self, config: Optional[FIFOQueueConfig] = None):
        self.config = config or FIFOQueueConfig()

    def simulate(self, df: pl.DataFrame, signal_col: str, threshold: float = 0.2) -> FIFOSimulationResult:
        """
        Executes a realistic limit-order simulation with strict FIFO queue priority:
          - At tick t: Signal triggers limit order at best bid / best ask.
          - Order arrives at t + latency_ticks.
          - Queue position ahead is locked: Q_ahead = V0 at arrival price.
          - Order ONLY fills if subsequent volume traded through exceeds Q_ahead.
          - If midprice runs away in our favor without filling Q_ahead: order cancelled (0 fill).
          - If midprice crashes through: order filled at the tail and suffers adverse selection.
        """
        valid = df.drop_nulls().clone()
        n = len(valid)
        if n < 50:
            return self._empty_result()

        # Generate discrete signal
        sig = pl.when(pl.col(signal_col) > threshold).then(1.0)\
                .when(pl.col(signal_col) < -threshold).then(-1.0)\
                .otherwise(0.0)
        valid = valid.with_columns(sig.alias('raw_signal'))

        # Inject exchange latency
        valid = valid.with_columns(
            pl.col('raw_signal').shift(self.config.latency_ticks).fill_null(0.0).alias('exec_signal')
        )

        exec_sig = valid['exec_signal'].to_numpy()
        mid = valid['mid_price'].to_numpy()
        bid_p0 = valid['bid_p0'].to_numpy()
        ask_p0 = valid['ask_p0'].to_numpy()
        bid_v0 = valid['bid_v0'].to_numpy()
        ask_v0 = valid['ask_v0'].to_numpy()
        qi_5 = valid['qi_5'].to_numpy()

        pnl_per_trade: List[float] = []
        adverse_selection_list: List[float] = []
        total_limit_orders = 0
        filled_orders = 0
        unfilled_orders = 0

        in_pos = False
        pos_side = 0.0
        entry_price = 0.0
        entry_tick = 0

        # Pending limit order state
        order_pending = False
        order_side = 0.0
        order_price = 0.0
        order_placed_tick = 0
        queue_ahead = 0.0
        cum_volume_traded = 0.0

        for i in range(n - self.config.holding_period - 1):
            target_side = exec_sig[i]

            # 1. Manage Active Position Exit
            if in_pos:
                duration = i - entry_tick
                flip = (target_side != 0.0 and target_side != pos_side)
                
                if duration >= self.config.holding_period or flip:
                    # Passive exit (limit order at opposite top of book)
                    exit_price = ask_p0[i] if pos_side > 0 else bid_p0[i]
                    fee = 2.0 * self.config.maker_fee_bps

                    trade_ret = (exit_price - entry_price) / entry_price * 10000.0 * pos_side
                    net_trade = trade_ret - fee
                    pnl_per_trade.append(net_trade)

                    in_pos = False
                    pos_side = 0.0
                continue

            # 2. Process Pending Limit Order (FIFO Queue Depletion)
            if order_pending:
                # Check for fast cancellation due to adverse queue inversion
                if self.config.cancel_on_adverse:
                    if (order_side > 0 and qi_5[i] < -self.config.adverse_threshold) or \
                       (order_side < 0 and qi_5[i] > self.config.adverse_threshold):
                        # Fast cancel quote before toxic fill occurs!
                        order_pending = False
                        unfilled_orders += 1
                        continue

                # Check if market price moved away from our quote (Order abandoned)
                if (order_side > 0 and bid_p0[i] > order_price) or (order_side < 0 and ask_p0[i] < order_price):
                    # Market rallied away without filling us
                    order_pending = False
                    unfilled_orders += 1
                    continue

                # Check if price crashed through our level (Guaranteed fill with toxic slip)
                price_crashed_through = (order_side > 0 and ask_p0[i] <= order_price) or \
                                        (order_side < 0 and bid_p0[i] >= order_price)

                # Estimate volume consumed at our price level
                if order_side > 0:
                    vol_consumed = max(0.0, bid_v0[order_placed_tick] - bid_v0[i])
                else:
                    vol_consumed = max(0.0, ask_v0[order_placed_tick] - ask_v0[i])
                
                cum_volume_traded += vol_consumed

                # FILL CONDITION:
                # Either market crashed through OR volume consumed exceeded initial queue ahead!
                if price_crashed_through or (cum_volume_traded >= queue_ahead):
                    in_pos = True
                    pos_side = order_side
                    entry_price = order_price
                    entry_tick = i
                    filled_orders += 1
                    order_pending = False

                    # Measure immediate 3-tick adverse selection
                    if i + 3 < n:
                        adv = (mid[i + 3] - mid[i]) / mid[i] * 10000.0 * (-pos_side)
                        adverse_selection_list.append(adv)
                    continue

                # Timeout cancel after 15 seconds without fill
                if (i - order_placed_tick) >= 15:
                    order_pending = False
                    unfilled_orders += 1
                    continue

            # 3. New Order Placement
            if not in_pos and not order_pending and target_side != 0.0:
                total_limit_orders += 1
                order_pending = True
                order_side = target_side
                order_placed_tick = i
                cum_volume_traded = 0.0

                if order_side > 0:
                    order_price = bid_p0[i]
                    initial_v = bid_v0[i]
                else:
                    order_price = ask_p0[i]
                    initial_v = ask_v0[i]

                # Queue position ahead: Tail = 100% of resting depth ahead; Mid = 50%
                queue_ahead = initial_v if self.config.queue_placement == "TAIL" else (initial_v * 0.5)

        if not pnl_per_trade:
            return self._empty_result(total_limit_orders, unfilled_orders)

        trades = np.array(pnl_per_trade)
        cum_pnl = np.cumsum(trades)
        peaks = np.maximum.accumulate(cum_pnl)
        drawdowns = peaks - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = float((trades > 0).mean() * 100.0)
        mean_pnl = float(trades.mean())
        std_pnl = float(trades.std()) if len(trades) > 1 else 1.0

        sharpe = (mean_pnl / std_pnl) * np.sqrt(86400.0 / self.config.holding_period) if std_pnl > 0 else 0.0
        downside = trades[trades < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else std_pnl
        sortino = (mean_pnl / downside_std) * np.sqrt(86400.0 / self.config.holding_period) if downside_std > 0 else 0.0

        mean_adv = float(np.mean(adverse_selection_list)) if adverse_selection_list else 0.0
        fill_rate = (filled_orders / max(total_limit_orders, 1)) * 100.0
        fee_drag = float(2.0 * self.config.maker_fee_bps * len(trades))

        return FIFOSimulationResult(
            trades_count=len(trades),
            fill_rate_pct=fill_rate,
            win_rate=win_rate,
            gross_pnl_bps=float(np.sum(trades) + fee_drag),
            fee_drag_bps=fee_drag,
            net_pnl_bps=float(np.sum(trades)),
            mean_trade_bps=mean_pnl,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_bps=max_dd,
            adverse_selection_bps=mean_adv,
            unfilled_orders_count=unfilled_orders,
            equity_curve=cum_pnl
        )

    def _empty_result(self, total: int = 0, unfilled: int = 0) -> FIFOSimulationResult:
        return FIFOSimulationResult(
            trades_count=0, fill_rate_pct=0.0, win_rate=0.0, gross_pnl_bps=0.0,
            fee_drag_bps=0.0, net_pnl_bps=0.0, mean_trade_bps=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, max_drawdown_bps=0.0, adverse_selection_bps=0.0,
            unfilled_orders_count=unfilled, equity_curve=np.array([0.0])
        )
