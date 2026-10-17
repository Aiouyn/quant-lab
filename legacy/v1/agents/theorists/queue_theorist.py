"""
Queue Dynamics & Micro-price Theorist
Specializes in queue depletion hazard, volume-weighted micro-price deviations,
and non-linear depth curvature (Cartea, Jaimungal, Lehalle).
"""

import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """You are Dr. Álvaro Cartea, an Oxford Professor of Mathematical Finance and Head Microstructure Quant.
Your domain expertise is limit order book queue depletion, micro-price hazard rates, and optimal execution under queue priority.

When formulating a research hypothesis:
1. Focus on the relationship between micro-price deviation (`micro_dev = micro_p - mid_p`), depth imbalance (`qi_1`, `qi_5`, `qi_20`), and order-count skew (`count_skew_5`).
2. High-frequency edge often comes from curvature in the book: when small depth at top-of-book is backed by massive depth at levels 2-5, price is about to break through.
3. Available features:
   - `micro_dev`, `spread`, `mid_price`
   - `bid_v0`, `ask_v0`, `bid_v5`, `ask_v5`
   - `qi_1`, `qi_5`, `qi_20`
   - `count_skew_5`
   - `vol_30s_bps`
4. Return output strictly in valid JSON format.
"""

class QueueTheorist(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Queue_Theorist",
            role="Queue Depletion & Micro-price Hazard Specialist",
            system_prompt=SYSTEM_PROMPT
        )

    def formulate_hypothesis(self, past_failures: List[str] = None) -> Dict[str, Any]:
        prompt = (
            "Formulate a novel market microstructure hypothesis based on Micro-Price Curvature and Queue Depletion.\n"
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
            "- 'target_horizon': e.g., '5s' or '10s'\n"
            "- 'expected_regime': e.g., 'Normal volatility, wide spread'\n"
        )

        raw = self.query(prompt, temperature=0.7, json_mode=True)
        try:
            return json.loads(raw)
        except Exception:
            return {
                "title": "Asymmetric Micro-Price Curvature with Order Count Skew",
                "theorist": self.name,
                "rationale": "Micro-price deviation is amplified when the order count skew confirms institutional resting orders behind top of book.",
                "mathematical_formulation": "Signal_t = (micro_dev_t / spread_t) * (1.0 + 0.5 * count_skew_5_t) * (qi_5_t - qi_1_t)",
                "target_horizon": "10s",
                "expected_regime": "Normal liquidity, low toxic flow"
            }
