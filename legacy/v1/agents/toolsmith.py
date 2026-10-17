"""
Toolsmith Quant Agent
Specialized Quantitative Software Engineer that autonomously conceives,
designs, and implements reusable quantitative microstructure tools.
Incorporates self-healing reflexion to fix syntax or causal leakage issues.
"""

import re
import json
import logging
import polars as pl
from typing import Dict, Any, Tuple, Optional
from agents.base_agent import BaseAgent
from core.tool_validator import tool_validator
from core.tool_registry import QuantitativeTool, tool_registry

logger = logging.getLogger(__name__)

TOOLSMITH_SYSTEM_PROMPT = """You are an elite Quantitative Tools Engineer at a premier proprietary trading firm.
Your mission is to invent NOVEL, REUSABLE, INSTITUTIONAL-GRADE quantitative tools for market microstructure analysis on OKX BTC L2 orderbook data.

CRITICAL RULES FOR TOOL IMPLEMENTATION:
1. You MUST define a single vectorized function:
   `def calculate(df: pl.DataFrame) -> pl.Series:`
2. The returned Series MUST:
   - Have the EXACT same length as `df`.
   - Be strictly CAUSAL: output at tick t must NEVER depend on rows > t (NO forward rolling windows, NO negative shifts).
   - Use `.fill_null(0.0)` or backward/forward causal fill so there are NEVER any NaNs, Nulls, or Infs.
   - Be numerically bounded or standardized (Z-scores, bounded ratios [-1, 1], or bounded indices).
3. Do NOT use Python row iteration (for loops). Use pure vectorized Polars or NumPy operations.
4. Input DataFrame columns available:
   - `bid_p0`, `bid_v0`, `bid_n0`
   - `ask_p0`, `ask_v0`, `ask_n0`
   - `spread`, `mid_price`, `micro_price`, `micro_dev`
   - `bid_v5`, `ask_v5`, `qi_1`, `qi_5`, `qi_20`
   - `count_skew_5`, `ofi`, `vol_30s_bps`
   - Plus any previously registered tools!
5. Return JSON with:
   - "name": lowercase alphanumeric identifier with underscores (e.g. "flow_toxicity_gauge")
   - "category": "FEATURE_EXTRACTOR", "REGIME_FILTER", or "ESTIMATOR"
   - "mathematical_formulation": LaTeX or mathematical description
   - "docstring": Summary of economic rationale and microstructure theory
   - "code": Valid Python code inside a ```python ... ``` block.
"""

class ToolsmithQuant(BaseAgent):
    def __init__(self, name: str = "Chief_Toolsmith_Quant"):
        super().__init__(
            name=name,
            role="Autonomous Quantitative Tool Architect",
            system_prompt=TOOLSMITH_SYSTEM_PROMPT
        )

    def formulate_and_build_tool(
        self,
        domain_specialization: str,
        test_df: pl.DataFrame,
        negative_feedback: Optional[str] = None,
        max_retries: int = 3
    ) -> Tuple[Optional[QuantitativeTool], Dict[str, Any]]:
        """
        Conceives a new tool, generates implementation code, and validates it
        against the adversarial causal leakage test suite.
        """
        feedback_clause = f"\nAVOID PREVIOUS DEFECTS:\n{negative_feedback}\n" if negative_feedback else ""
        prompt = (
            f"Invent a novel quantitative microstructure tool.\n"
            f"Target Specialization: {domain_specialization}.\n"
            f"{feedback_clause}\n"
            f"Focus on dimensionally sound, normalized metrics (e.g. rolling Z-scores, depth decay, liquidity friction).\n"
            f"Return JSON with 'name', 'category', 'mathematical_formulation', 'docstring', and 'code'."
        )

        current_prompt = prompt
        for attempt in range(1, max_retries + 1):
            raw_resp = self.query(current_prompt, temperature=0.4, json_mode=True)
            tool_data = self._parse_tool_response(raw_resp)

            if not tool_data or "code" not in tool_data:
                current_prompt = "Failed to parse tool JSON. Please return valid JSON with 'name', 'category', 'mathematical_formulation', 'docstring', and 'code'."
                continue

            code_clean = self._extract_code(tool_data["code"])
            
            # Run adversarial verification
            passed, report = tool_validator.validate_tool_candidate(code_clean, test_df)

            if passed:
                logger.info(f"Toolsmith successfully created verified tool '{tool_data['name']}' on attempt {attempt}.")
                tool_obj = QuantitativeTool(
                    name=tool_data["name"],
                    author=self.name,
                    category=tool_data.get("category", "FEATURE_EXTRACTOR"),
                    mathematical_formulation=tool_data.get("mathematical_formulation", "N/A"),
                    code=code_clean,
                    docstring=tool_data.get("docstring", "Autonomous quantitative tool."),
                    verified=True,
                    audit_report=report
                )
                return tool_obj, report
            
            # Reflexion loop: feedback errors to self-heal
            err_msg = "; ".join(report.get("errors", ["Unknown validation failure"]))
            logger.warning(f"Tool '{tool_data.get('name')}' failed validation on attempt {attempt}: {err_msg}. Triggering reflexion...")
            current_prompt = (
                f"Your tool '{tool_data.get('name')}' failed verification with the following errors:\n"
                f"{err_msg}\n\n"
                f"Previous Code:\n```python\n{code_clean}\n```\n\n"
                f"Please fix the code to ensure strict causal invariance (zero lookahead leakage), no NaNs/Nulls, and fast Polars execution.\n"
                f"Return updated JSON."
            )

        return None, {"status": "FAILED", "reason": f"Exceeded {max_retries} attempts."}

    def _parse_tool_response(self, text: str) -> Optional[Dict[str, Any]]:
        from core.event_bus import extract_json_safe
        default = {
            "name": "micro_depth_acceleration",
            "category": "FEATURE_EXTRACTOR",
            "mathematical_formulation": "Accel_t = d2(QI_5)/dt2",
            "docstring": "Second derivative of 5-level queue imbalance.",
            "code": "import polars as pl\n\ndef calculate(df: pl.DataFrame) -> pl.Series:\n    qi = df['qi_5']\n    d1 = (qi - qi.shift(1)).fill_null(0.0)\n    d2 = (d1 - d1.shift(1)).fill_null(0.0)\n    return d2.clip(-1.0, 1.0)\n"
        }
        return extract_json_safe(text, default=default)

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

toolsmith_agent = ToolsmithQuant()
