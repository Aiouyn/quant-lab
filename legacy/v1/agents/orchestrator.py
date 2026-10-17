"""
Lab Director & Autonomous Research Orchestrator
Coordinates the Theorist Council, Quant Developer, Execution Engine, and Red Team Auditor.
Executes iterative hypothesis-to-falsification research sprints.
"""

import os
import glob
import logging
import polars as pl
from typing import Dict, Any, List, Optional
from pathlib import Path

from config import DATA_DIR, DATABASE_PATH, RESULTS_DIR
from core.orderbook import load_and_extract_features
from core.knowledge_base import ResearchLedger
from agents.theorists.ofi_theorist import OFITheorist
from agents.theorists.queue_theorist import QueueTheorist
from agents.theorists.adverse_theorist import AdverseSelectionTheorist
from agents.quant_developer import QuantDeveloper
from agents.red_team_auditor import RedTeamAuditor

logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    def __init__(self, data_dir: Optional[str] = None, db_path: Optional[str] = None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.kb = ResearchLedger(str(db_path or DATABASE_PATH))
        self.auditor = RedTeamAuditor(min_dsr=0.80, min_sharpe=1.0)
        self.developer = QuantDeveloper()

        # The Council of Theorists
        self.theorists = [
            OFITheorist(),
            QueueTheorist(),
            AdverseSelectionTheorist()
        ]

        self._cached_dataset: Optional[pl.DataFrame] = None
        self._dataset_date: str = "Unknown"

    def load_research_dataset(self, max_rows: int = 15000) -> pl.DataFrame:
        """Loads a slice of the latest available OKX 5000-level archive."""
        if self._cached_dataset is not None:
            return self._cached_dataset

        archives = sorted(glob.glob(str(self.data_dir / "*.tar.gz")), reverse=True)
        if not archives:
            raise FileNotFoundError(f"No OKX 5000lv archives found in {self.data_dir}")

        target_archive = archives[0]
        self._dataset_date = os.path.basename(target_archive).replace(".tar.gz", "").split("-")[-1]
        logger.info(f"Loading research dataset from {target_archive} ({max_rows} ticks)...")
        
        self._cached_dataset = load_and_extract_features(target_archive, max_rows=max_rows)
        logger.info(f"Successfully loaded {len(self._cached_dataset)} ticks from {self._dataset_date}.")
        return self._cached_dataset

    def run_research_iteration(self, campaign_id: int, theorist_idx: int = 0) -> Dict[str, Any]:
        """
        Executes one full hypothesis -> code -> backtest -> audit loop:
          1. Formulates hypothesis from designated Theorist
          2. Generates and sandboxes vectorized Polars code
          3. Backtests in execution simulator
          4. Submits to Red Team 5-point audit battery
          5. Records everything to Research Ledger
        """
        theorist = self.theorists[theorist_idx % len(self.theorists)]
        total_trials = self.kb.get_total_trials_count() + 1
        
        print(f"\n{'='*75}")
        print(f"🔬 RESEARCH SPRINT: Trial #{total_trials} | Theorist: {theorist.name}")
        print(f"{'='*75}")

        # 1. Hypothesis Formulation
        hypo = theorist.formulate_hypothesis()
        h_id = self.kb.record_hypothesis(
            campaign_id=campaign_id,
            theorist=theorist.name,
            title=hypo.get('title', 'Untitled'),
            rationale=hypo.get('rationale', ''),
            math_form=hypo.get('mathematical_formulation', ''),
            target_horizon=hypo.get('target_horizon', '10s')
        )
        print(f"💡 Hypothesis Proposed: '{hypo.get('title')}'")
        print(f"   Math: {hypo.get('mathematical_formulation')}")

        # 2. Code Implementation & Sandboxing
        code_str, is_valid, err = self.developer.implement_hypothesis(hypo)
        impl_id = self.kb.record_implementation(h_id, code_str, is_valid, err)
        if not is_valid:
            print(f"❌ Implementation Failed Sandbox: {err}")
            return {'status': 'IMPLEMENTATION_FAILED', 'error': err}

        print("⚡ Code successfully generated and sandboxed.")

        # 3. Dynamic Function Compilation
        scope: Dict[str, Any] = {}
        exec(code_str, scope)
        signal_fn = scope['compute_signal']

        # 4. Load Dataset & Execute Audit
        df = self.load_research_dataset()
        audit_res = self.auditor.audit_implementation(signal_fn, df, total_trials)
        base_res = audit_res['baseline_res']

        # 5. Record Backtest & Audit to Ledger
        if base_res:
            self.kb.record_backtest(impl_id, self._dataset_date, base_res)

        self.kb.record_audit(
            impl_id=impl_id,
            verdict=audit_res['verdict'],
            reasons=audit_res['reasons'],
            dsr=audit_res['dsr'],
            lookahead=audit_res['lookahead_passed'],
            latency=audit_res['latency_passed'],
            fee_resilience=audit_res['fee_passed'],
            notes=f"Theorist: {theorist.name}"
        )

        # 6. Display Findings
        verdict = audit_res['verdict']
        symbol = "✅" if verdict == "APPROVED" else "❌"
        print(f"\n{symbol} AUDIT VERDICT: {verdict} (DSR: {audit_res['dsr']:.2f})")
        if base_res:
            print(f"   Trades: {base_res.trades_count} | Win Rate: {base_res.win_rate:.1f}%")
            print(f"   Net PnL: {base_res.net_pnl_bps:+.1f} bps | Sharpe: {base_res.sharpe_ratio:.2f}")
            print(f"   Adverse Selection: {base_res.adverse_selection_bps:.2f} bps/trade")

        if audit_res['reasons']:
            print("   Rejection Rationale:")
            for r in audit_res['reasons']:
                print(f"     - {r}")

        return {
            'trial_number': total_trials,
            'theorist': theorist.name,
            'title': hypo.get('title'),
            'verdict': verdict,
            'dsr': audit_res['dsr'],
            'sharpe': base_res.sharpe_ratio if base_res else 0.0,
            'reasons': audit_res['reasons']
        }

    def run_campaign(self, num_iterations: int = 5, title: str = "Alpha Discovery Sprint") -> List[Dict[str, Any]]:
        """Runs an autonomous multi-iteration research campaign."""
        c_id = self.kb.create_campaign(title, f"Autonomous research sprint across {num_iterations} trials")
        print(f"\n🚀 Launching Campaign #{c_id}: '{title}' ({num_iterations} iterations)")
        
        results = []
        for i in range(num_iterations):
            res = self.run_research_iteration(campaign_id=c_id, theorist_idx=i)
            results.append(res)

        print(f"\n{'='*75}")
        print("🏆 RESEARCH CAMPAIGN SUMMARY & LEADERBOARD")
        print(f"{'='*75}")
        alphas = self.kb.get_approved_alphas()
        print(f"Total Hypotheses Tested: {self.kb.get_total_trials_count()}")
        print(f"Approved Alphas: {len(alphas)}")

        for idx, a in enumerate(alphas, 1):
            print(f"\n#{idx} {a['title']} (by {a['theorist']})")
            print(f"    Sharpe: {a['sharpe_ratio']:.2f} | DSR: {a['deflated_sharpe']:.2f} | Net: {a['net_pnl_bps']:+.1f} bps | Win Rate: {a['win_rate']:.1f}%")
            print(f"    Formula: {a['mathematical_formulation']}")

        return results
