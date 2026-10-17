"""
Quant Developer Agent
Translates theoretical mathematical formulations into robust, vectorized Polars code.
Includes a sandboxed self-correction (Reflexion) loop that tests the code before returning.
"""

import re
import json
import logging
import numpy as np
import polars as pl
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an elite High-Frequency Quantitative Developer at a top proprietary trading firm.
Your sole job is to translate market microstructure mathematical hypotheses into clean, bug-free, vectorized Polars code.

CRITICAL RULES:
1. You MUST define a single function:
   `def compute_signal(df: pl.DataFrame) -> pl.Series:`
2. The returned Series MUST:
   - Have the EXACT same length as `df`.
   - Contain float values roughly bounded in [-1.0, 1.0].
   - Positive (> 0) indicates bullish / quote-bid skew.
   - Negative (< 0) indicates bearish / quote-ask skew.
   - Have all NaNs / nulls filled with 0.0 (`.fill_null(0.0)`).
3. Do NOT use Python for-loops over rows. Use pure Polars / NumPy vectorized expressions.
4. Available columns in `df`:
   - `bid_p0`, `bid_v0`, `bid_n0`
   - `ask_p0`, `ask_v0`, `ask_n0`
   - `spread`, `mid_price`, `micro_price`, `micro_dev`
   - `bid_v5`, `ask_v5`, `qi_1`, `qi_5`, `qi_20`
   - `count_skew_5`, `ofi`, `vol_30s_bps`
5. Return ONLY executable Python code inside a ```python ... ``` markdown block.
"""

class QuantDeveloper(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Quant_Developer",
            role="High-Performance Microstructure Code Engineer",
            system_prompt=SYSTEM_PROMPT
        )

    def implement_hypothesis(self, hypothesis: Dict[str, Any], max_retries: int = 3) -> Tuple[str, bool, str]:
        """
        Generates and sandbox-verifies signal computation code.
        Returns: (code_str, is_valid, error_msg)
        """
        from core.tool_registry import tool_registry
        verified_tools = tool_registry.get_verified_tools()
        tools_clause = ""
        if verified_tools:
            tools_clause = "\nVERIFIED CORPORATE TOOLS AVAILABLE IN `df`:\n" + "\n".join(
                [f"- `df['{t['name']}']`: {t['docstring']} (Math: {t['mathematical_formulation']})" for t in verified_tools]
            ) + "\nYou can and SHOULD use these pre-computed, verified features in your calculation!\n"

        prompt = (
            f"Implement the following microstructure hypothesis in vectorized Polars code:\n\n"
            f"Title: {hypothesis.get('title')}\n"
            f"Theorist: {hypothesis.get('theorist', 'Unknown')}\n"
            f"Rationale: {hypothesis.get('rationale')}\n"
            f"Mathematical Formulation:\n{hypothesis.get('mathematical_formulation')}\n"
            f"Target Horizon: {hypothesis.get('target_horizon')}\n"
            f"{tools_clause}\n"
        )

        current_prompt = prompt
        for attempt in range(1, max_retries + 1):
            raw_response = self.query(current_prompt, temperature=0.2)
            code = self._extract_code(raw_response)
            
            # Sandbox validation
            is_valid, err = self._sandbox_test(code)
            if is_valid:
                logger.info(f"QuantDeveloper successfully validated code on attempt {attempt}.")
                return code, True, ""
            
            logger.warning(f"Sandbox error on attempt {attempt}: {err}. Prompting for self-correction...")
            current_prompt = (
                f"Your previous code produced the following error during execution:\n{err}\n\n"
                f"Previous Code:\n```python\n{code}\n```\n\n"
                f"Please fix the error and provide the corrected implementation.\n"
                f"Remember to return `def compute_signal(df: pl.DataFrame) -> pl.Series:` with length matching `df`."
            )

        # Fallback to safe default if all retries fail
        fallback_code = (
            "import polars as pl\n\n"
            "def compute_signal(df: pl.DataFrame) -> pl.Series:\n"
            "    spread = df['spread'].clip(lower_bound=0.1)\n"
            "    qi = df['qi_5']\n"
            "    micro_dev = df['micro_dev'] / spread\n"
            "    sig = (qi * 0.6 + micro_dev * 0.4).fill_null(0.0)\n"
            "    return sig.clip(-1.0, 1.0)\n"
        )
        return fallback_code, False, f"Exceeded {max_retries} attempts. Fallback used."

    def _extract_code(self, text: str) -> str:
        """Extracts python code robustly from markdown or text."""
        # 1. Match ```python or ```py or ```
        match = re.search(r"```(?:python|py)?\s*\n?(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if "def compute_signal" in extracted:
                return extracted

        # 2. Match function definition directly if markdown was omitted or malformed
        func_match = re.search(r"((?:import\s+[^\n]+\n+)*def\s+compute_signal\(.*)", text, re.DOTALL)
        if func_match:
            candidate = func_match.group(1).strip()
            candidate = re.sub(r"```.*$", "", candidate, flags=re.DOTALL).strip()
            return candidate

        return text.strip()

    def _sandbox_test(self, code_str: str) -> Tuple[bool, str]:
        """Runs the code in an isolated scope against dummy orderbook data."""
        try:
            scope: Dict[str, Any] = {}
            exec(code_str, scope)
            if 'compute_signal' not in scope:
                return False, "Function `compute_signal(df)` was not defined."

            fn = scope['compute_signal']
            # Create synthetic test frame
            n = 50
            dummy_df = pl.DataFrame({
                'bid_p0': np.full(n, 85000.0),
                'bid_v0': np.full(n, 10.0),
                'bid_n0': np.full(n, 5),
                'ask_p0': np.full(n, 85001.0),
                'ask_v0': np.full(n, 10.0),
                'ask_n0': np.full(n, 5),
                'spread': np.full(n, 1.0),
                'mid_price': np.full(n, 85000.5),
                'micro_price': np.full(n, 85000.5),
                'micro_dev': np.full(n, 0.0),
                'bid_v5': np.full(n, 50.0),
                'ask_v5': np.full(n, 50.0),
                'qi_1': np.zeros(n),
                'qi_5': np.zeros(n),
                'qi_20': np.zeros(n),
                'count_skew_5': np.zeros(n),
                'ofi': np.zeros(n),
                'vol_30s_bps': np.ones(n)
            })

            # Dynamically include any registered enterprise tools in dummy_df
            from core.tool_registry import tool_registry
            for tool_name in tool_registry.get_verified_tool_names():
                if tool_name not in dummy_df.columns:
                    dummy_df = dummy_df.with_columns(pl.Series(tool_name, np.zeros(n, dtype=np.float64)))

            res = fn(dummy_df)
            if not isinstance(res, pl.Series):
                return False, f"Expected `compute_signal` to return a `polars.Series`, got `{type(res)}`."
            if len(res) != n:
                return False, f"Output length mismatch: expected {n}, got {len(res)}."
            if res.null_count() > 0:
                return False, f"Output contains {res.null_count()} null values. Please use `.fill_null(0.0)`."

            return True, ""
        except Exception as e:
            return False, str(e)
