"""
OFI & Liquidity Dynamics Theorist
Specializes in multi-level Order Flow Imbalance, liquidity replenishment asymmetry,
and short-term price impact kernels (Cont, Bouchaud, Stoikov).
"""

import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """You are Dr. Rama Cont, a Senior Market Microstructure Theorist at an elite institutional quantitative hedge fund.
Your domain expertise is Order Flow Imbalance (OFI), high-frequency book dynamics, and Kyle's lambda.

When generating a research hypothesis:
1. Base your idea on mathematical properties of orderbook liquidity (e.g. OFI multi-level integration, cancel-to-trade ratios, volume replenishment asymmetry).
2. Avoid generic technical analysis (NO RSI, MACD, or candlestick patterns). You ONLY work with L2 orderbook features:
   - `ofi` (Order flow imbalance at top level)
   - `spread` (Best ask - best bid)
   - `bid_v0`, `ask_v0`, `bid_v5`, `ask_v5` (Top 1 and 5 depth)
   - `qi_1`, `qi_5`, `qi_20` (Queue depth imbalance)
   - `micro_dev` (Micro-price minus mid-price)
   - `count_skew_5` (Order count skew at top 5)
   - `vol_30s_bps` (Realized volatility in basis points)
3. Return output strictly in valid JSON format.
"""

class OFITheorist(BaseAgent):
    def __init__(self):
        super().__init__(
            name="OFI_Theorist",
            role="Order Flow Imbalance & Liquidity Dynamics Specialist",
            system_prompt=SYSTEM_PROMPT
        )

    def formulate_hypothesis(self, past_failures: List[str] = None) -> Dict[str, Any]:
        prompt = (
            "Formulate a novel market microstructure hypothesis based on Order Flow Imbalance (OFI) and depth replenishment.\n"
            "The hypothesis must predict midprice drift over a 5s to 30s horizon on OKX BTC-USDT SWAP.\n"
        )
        if past_failures:
            prompt += f"\nImportant: The following prior hypotheses were falsified and rejected by the Red Team:\n"
            for f in past_failures[-5:]:
                prompt += f"- {f}\n"
            prompt += "\nDo not repeat these patterns. Introduce a distinct structural mechanism.\n"

        prompt += (
            "\nProvide your output in JSON with fields:\n"
            "- 'title': Short descriptive title\n"
            "- 'rationale': Economic and microstructure logic\n"
            "- 'mathematical_formulation': Precise mathematical formula using available L2 features\n"
            "- 'target_horizon': e.g., '10s'\n"
            "- 'expected_regime': e.g., 'High liquidity, tight spread'\n"
        )

        raw = self.query(prompt, temperature=0.7, json_mode=True)
        try:
            return json.loads(raw)
        except Exception:
            return {
                "title": "Normalized Multi-Level OFI with Spread Dampening",
                "theorist": self.name,
                "rationale": "Order Flow Imbalance normalized by spread and volatility reduces false breakouts in choppy ranges.",
                "mathematical_formulation": "Signal_t = (OFI_t / max(Spread_t, 0.1)) * (QI_5_t / (1.0 + Vol_30s_bps_t))",
                "target_horizon": "10s",
                "expected_regime": "Moderate volatility, trending books"
            }
