"""
Base LLM Agent Wrapper
Integrates with OpenAI-compatible APIs (MiMo, OpenRouter, OpenAI).
Includes fallback mock responses for testing without an active API key.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str, model: Optional[str] = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model or config.DEFAULT_MODEL
        self.client = None

        if config.ACTIVE_KEY:
            self.client = OpenAI(
                api_key=config.ACTIVE_KEY,
                base_url=config.ACTIVE_BASE_URL
            )

    def is_configured(self) -> bool:
        return self.client is not None

    def query(self, user_prompt: str, temperature: float = 0.7, json_mode: bool = False) -> str:
        """Queries the LLM or falls back to built-in quantitative heuristic generation if no key configured."""
        if config.ACTIVE_KEY and (self.client is None or getattr(self, '_current_key', None) != config.ACTIVE_KEY or getattr(self, '_current_url', None) != config.ACTIVE_BASE_URL):
            self.client = OpenAI(api_key=config.ACTIVE_KEY, base_url=config.ACTIVE_BASE_URL)
            self._current_key = config.ACTIVE_KEY
            self._current_url = config.ACTIVE_BASE_URL

        if not self.is_configured():
            logger.warning(f"Agent {self.name} is running in standalone mock mode (no API key detected in config).")
            raise RuntimeError('Provider unavailable; no synthetic hypothesis was substituted. Use the explicit v2 demo command for infrastructure testing.')

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt_tokens_est = len(user_prompt.split()) * 2 + len(self.system_prompt.split()) * 2
            response_format = {"type": "json_object"} if json_mode else None
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=12000,
                extra_body={"thinking": {"type": "enabled"}} if config.ACTIVE_PROVIDER == "mimo" else {},
                timeout=180,
                response_format=response_format
            )
            raw_content = response.choices[0].message.content or ""
            if not raw_content and hasattr(response.choices[0].message, "reasoning_content") and response.choices[0].message.reasoning_content:
                raw_content = response.choices[0].message.reasoning_content
            from core.token_tracker import tracker
            usage = getattr(response, "usage", None)
            if usage:
                tracker.record_usage(usage.prompt_tokens, usage.completion_tokens, self.name)
            else:
                tracker.record_usage(prompt_tokens_est, len(raw_content.split()) * 2, self.name)
            return raw_content
        except Exception as e:
            err_str = str(e)
            if "insufficient_balance" in err_str or "402" in err_str:
                logger.warning(f"Agent {self.name}: MiMo account has insufficient balance (402). Falling back to quantitative heuristic engine.")
            else:
                logger.error(f"Error querying LLM in agent {self.name}: {e}")
            # Graceful fallback to maintain research pipeline continuity
            raise RuntimeError('Provider unavailable; no synthetic hypothesis was substituted. Use the explicit v2 demo command for infrastructure testing.')

    def _fallback_response(self, user_prompt: str, json_mode: bool) -> str:
        """Deterministic fallback responses for testing the pipeline when API key is missing."""
        if "hypothesis" in user_prompt.lower() or "theorist" in self.role.lower():
            res = {
                "title": "Normalized Multi-Level OFI with Spread Dampening",
                "theorist": self.name,
                "rationale": "Order Flow Imbalance normalized by spread and volatility reduces false breakouts in choppy ranges.",
                "mathematical_formulation": "Signal_t = (OFI_t / max(Spread_t, 0.1)) * (QI_5_t / (1.0 + Vol_30s_bps_t))",
                "target_horizon": "10s",
                "expected_regime": "Moderate volatility, trending books"
            }
            return json.dumps(res) if json_mode else str(res)

        elif "code" in user_prompt.lower() or "developer" in self.role.lower():
            code = (
                "import polars as pl\n\n"
                "def compute_signal(df: pl.DataFrame) -> pl.Series:\n"
                "    spread = df['spread'].clip(lower_bound=0.1)\n"
                "    ofi_norm = df['ofi'] / spread\n"
                "    qi = df['qi_5']\n"
                "    vol = df['vol_30s_bps'].fill_null(1.0).clip(lower_bound=0.5)\n"
                "    sig = (ofi_norm * 0.3 + qi * 0.7) / (1.0 + vol * 0.05)\n"
                "    return sig.clip(-1.0, 1.0)\n"
            )
            if json_mode:
                return json.dumps({"code": code})
            return f"```python\n{code}\n```"

        return "Default agent acknowledgment."
