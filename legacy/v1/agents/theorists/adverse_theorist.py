"""
Adverse Selection & Maker Overlay Theorist
Specializes in toxic flow avoidance, passive quote skewing, and volatility regime filters.
"""

import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

SYSTEM_PROMPT = """You are Dr. David Easley, co-creator of the VPIN metric (Volume-Synchronized Probability of Toxicity) and Head of Algorithmic Market Making.
Your domain expertise is adverse selection defense, informed vs uninformed flow separation, and quote skewing under toxic flow.

When formulating a research hypothesis:
1. Focus on protecting passive limit orders from being picked off by informed traders.
2. An edge in crypto market making comes from:
   - Detecting incoming toxic flow before fills occur (sharp divergence between depth imbalance and OFI).
   - Skewing quotes: When upward toxic flow is detected, retreat ask quotes and skew bid aggressively.
   - Conditioning on volatility regimes: In extreme volatility spikes, quote wider or pull quotes entirely.
3. Available features:
   - `ofi`, `spread`, `mid_price`
   - `micro_dev`, `qi_1`, `qi_5`, `qi_20`
   - `count_skew_5`, `vol_30s_bps`
4. Return output strictly in valid JSON format.
"""

class AdverseSelectionTheorist(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AdverseSelection_Theorist",
            role="Toxic Flow Avoidance & Passive Quoting Specialist",
            system_prompt=SYSTEM_PROMPT
        )

    def formulate_hypothesis(self, past_failures: List[str] = None) -> Dict[str, Any]:
        prompt = (
            "Formulate a novel market microstructure hypothesis based on Toxic Flow Defense and Passive Quote Skewing.\n"
            "The hypothesis must produce a directional quote-skew signal to avoid adverse selection fills on OKX BTC-USDT SWAP.\n"
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
            "- 'target_horizon': e.g., '10s' or '30s'\n"
            "- 'expected_regime': e.g., 'High toxic flow, impending breakout'\n"
        )

        raw = self.query(prompt, temperature=0.7, json_mode=True)
        try:
            return json.loads(raw)
        except Exception:
            return {
                "title": "Toxicity-Aware Passive Quote Skew with Volatility Clamp",
                "theorist": self.name,
                "rationale": "Strong queue imbalance backed by aligned order flow triggers an immediate quote skew, suppressed during anomalous volatility bursts.",
                "mathematical_formulation": "Signal_t = np.sign(qi_5_t) * (abs(qi_5_t) ** 1.5) * (1.0 / (1.0 + np.exp(vol_30s_bps_t - 5.0)))",
                "target_horizon": "10s",
                "expected_regime": "Low-to-medium volatility with persistent queue depletion"
            }
