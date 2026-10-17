"""
First-Principles Root-Cause Diagnostic Engine (Elon Musk Starship Failure Post-Mortem)
Relentlessly deconstructs why an alpha hypothesis failed backtest or audit falsification.
Breaks failures down to fundamental market physics:
  1. Adverse Selection Drift: Toxic fills by informed counterparties.
  2. Queue Starvation: Stale quote placement and slow queue exhaustion.
  3. Friction vs Gross Capture: Tick capture insufficient to beat VIP fee drag.
  4. Regime Instability: Inversion during volatility shocks or cascades.
Generates actionable Prescriptive Directives that force next-generation adaptation.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class FailureDecomposition:
    root_cause: str
    severity: str
    metric_evidence: str
    prescriptive_directive: str


class FirstPrinciplesDiagnostics:
    def __init__(self):
        self.diagnostic_history: List[Dict[str, Any]] = []

    def analyze_audit_failure(
        self,
        hypothesis_title: str,
        theorist_name: str,
        audit_res: Dict[str, Any],
        df: Optional[pl.DataFrame] = None
    ) -> FailureDecomposition:
        """Deconstructs the audit failure into first-principles mechanics."""
        reasons = audit_res.get('reasons', [])
        reasons_str = " ".join(reasons).lower()
        base_res = audit_res.get('baseline_res')
        fifo_res = audit_res.get('fifo_res')

        # 1. Check for Adverse Selection
        adv_sel = getattr(fifo_res, 'adverse_selection_bps', 0.0) if fifo_res else 0.0
        if "adverse" in reasons_str or adv_sel > 0.5:
            decomp = FailureDecomposition(
                root_cause="ADVERSE_SELECTION_TOXICITY",
                severity="HIGH",
                metric_evidence=f"Adverse post-fill drift: +{adv_sel:.2f} bps against entry within 3 ticks.",
                prescriptive_directive=(
                    "MANDATORY FIX: Incorporate Order Flow Imbalance (OFI) or Micro-Price Deviation filter. "
                    "Do NOT quote bids when short-term momentum or OFI is aggressively selling."
                )
            )

        # 2. Check for Queue Starvation
        elif fifo_res and getattr(fifo_res, 'fill_rate_pct', 100.0) < 20.0:
            fill_pct = getattr(fifo_res, 'fill_rate_pct', 0.0)
            decomp = FailureDecomposition(
                root_cause="QUEUE_STARVATION",
                severity="MEDIUM",
                metric_evidence=f"Fill rate collapsed to {fill_pct:.1f}% under FIFO priority.",
                prescriptive_directive=(
                    "MANDATORY FIX: Widen signal entry threshold or target higher-liquidity queue levels. "
                    "Incorporate Depth Decay Slope or Queue Acceleration to identify queues about to trade."
                )
            )

        # 3. Check for Fee Drag / Insufficient Holding Horizon
        elif "fee" in reasons_str or (base_res and getattr(base_res, 'gross_pnl_bps', 0.0) > 0 and getattr(base_res, 'net_pnl_bps', 0.0) <= 0):
            decomp = FailureDecomposition(
                root_cause="SUB_OPTIMAL_HOLDING_HORIZON",
                severity="CRITICAL",
                metric_evidence="Gross alpha was positive, but wiped out by roundtrip spread friction.",
                prescriptive_directive=(
                    "MANDATORY FIX: Extend target holding period to 25s-35s (optimal is 30s). "
                    "Microstructure queue signals take 25s+ to overcome the 0.5-tick spread friction on BTC."
                )
            )

        # 4. Check for CPCV Regime Overfitting
        elif "cpcv" in reasons_str or "pbo" in reasons_str:
            pbo = audit_res.get('pbo', 0.5)
            decomp = FailureDecomposition(
                root_cause="REGIME_CASCADE_FRAGILITY",
                severity="HIGH",
                metric_evidence=f"PBO = {pbo:.2f} > 0.35. Strategy collapsed during trend cascade folds.",
                prescriptive_directive=(
                    "MANDATORY FIX: Add volatility regime conditioning. Scale down alpha magnitude "
                    "when realized 30s volatility exceeds 2.5x normal or when spread blows out."
                )
            )

        # 5. Default Fallback
        else:
            decomp = FailureDecomposition(
                root_cause="WEAK_INFORMATION_COEFFICIENT",
                severity="LOW",
                metric_evidence="Signal lacks persistent statistical correlation with forward midprice.",
                prescriptive_directive=(
                    "MANDATORY FIX: Synthesize multi-depth non-linear features (Level 1-20 cumulative imbalance) "
                    "rather than relying on single-level static depth."
                )
            )

        entry = {
            "title": hypothesis_title,
            "theorist": theorist_name,
            "root_cause": decomp.root_cause,
            "severity": decomp.severity,
            "metric_evidence": decomp.metric_evidence,
            "prescriptive_directive": decomp.prescriptive_directive
        }
        self.diagnostic_history.append(entry)
        return decomp

    def get_recent_diagnostics(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.diagnostic_history[-limit:]


# Global Singleton
diagnostics_engine = FirstPrinciplesDiagnostics()
