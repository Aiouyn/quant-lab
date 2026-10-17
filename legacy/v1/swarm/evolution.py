"""
Evolutionary Learning & Memetic Optimization Engine for 110-Agent Swarm
Implements Genetic Crossover, Strategy Mutation, and Reflexion-based Negative Learning.
Enables the hedge fund swarm to evolve, learn from Red Team falsifications, and improve generation-over-generation.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from core.knowledge_base import ResearchLedger
from core.event_bus import extract_json_safe
from swarm.desks import SWARM_MANIFEST, SwarmAgentDef

logger = logging.getLogger(__name__)

class EvolutionaryEngine:
    def __init__(self, kb: ResearchLedger, query_fn):
        self.kb = kb
        self.query_fn = query_fn
        self.generation = 0
        self.hall_of_fame: List[Dict[str, Any]] = []

    def get_hall_of_fame(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top-performing strategies ranked by Deflated Sharpe Ratio."""
        with self.kb._get_connection() as conn:
            cur = conn.execute("""
                SELECT h.id, h.title, h.theorist, h.mathematical_formulation, h.rationale,
                       b.sharpe_ratio, a.deflated_sharpe, b.win_rate, b.net_pnl_bps
                FROM hypotheses h
                JOIN implementations i ON i.hypothesis_id = h.id
                JOIN backtest_results b ON b.implementation_id = i.id
                JOIN audits a ON a.implementation_id = i.id
                ORDER BY a.deflated_sharpe DESC, b.sharpe_ratio DESC
                LIMIT ?
            """, (top_k,))
            rows = [dict(r) for r in cur.fetchall()]
            self.hall_of_fame = rows
            return rows

    def record_negative_memory(self, reason: str):
        """Appends a failure pattern to active evolutionary negative memory."""
        if not hasattr(self, "_cached_negative_memories"):
            self._cached_negative_memories = []
        self._cached_negative_memories.append(reason)

    def record_committee_critique(self, critique_summary: str, strat_title: Optional[str] = None, key_flaws: Optional[List[str]] = None):
        """Records qualitative architectural guidance from Investment Committee reviews."""
        if not hasattr(self, "_cached_committee_critiques"):
            self._cached_committee_critiques = []
        entry = critique_summary
        if strat_title:
            entry = f"[{strat_title}] {critique_summary}"
        if key_flaws:
            entry += f" | Key Flaws: {', '.join(key_flaws)}"
        self._cached_committee_critiques.append(entry)
        if len(self._cached_committee_critiques) > 10:
            self._cached_committee_critiques.pop(0)

    def get_committee_guidance(self, limit: int = 4) -> str:
        """Extracts synthesized qualitative guidance from Investment Committee votes."""
        guidance = list(getattr(self, "_cached_committee_critiques", []))
        try:
            with self.kb._get_connection() as conn:
                cur = conn.execute("""
                    SELECT notes FROM audits
                    WHERE notes IS NOT NULL AND notes != ''
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
                for r in cur.fetchall():
                    note = r[0]
                    if note and len(note) > 15:
                        guidance.append(note)
        except Exception:
            pass
        recent = list(dict.fromkeys(guidance))[-limit:]
        return "\n".join([f"- {g}" for g in recent]) if recent else ""

    def get_negative_memory(self, limit: int = 8) -> List[str]:
        """Extracts unique failure patterns from Red Team rejections to prevent repeating mistakes."""
        reasons = list(getattr(self, "_cached_negative_memories", []))
        try:
            with self.kb._get_connection() as conn:
                cur = conn.execute("""
                    SELECT rejection_reasons FROM audits
                    WHERE verdict = 'REJECTED'
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
                for r in cur.fetchall():
                    try:
                        r_list = json.loads(r[0])
                        reasons.extend(r_list)
                    except Exception:
                        pass
        except Exception:
            pass
        # Return unique top failure patterns
        return list(set(reasons))[:5]

    def breed_crossover_hypothesis(
        self,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        author_agent: SwarmAgentDef
    ) -> Dict[str, Any]:
        """
        Genetic Crossover Operator:
        Cross-breeds two winning parent strategies into an evolved hybrid offspring.
        """
        neg_mem = self.get_negative_memory()
        crossover_prompt = (
            f"You are {author_agent.name}, {author_agent.specialization}.\n"
            f"Your mission is to perform GENETIC CROSSOVER to breed an advanced hybrid strategy by combining two champion parent alphas:\n\n"
            f"PARENT A: '{parent_a.get('title')}'\n"
            f"Formula: {parent_a.get('mathematical_formulation')}\n"
            f"Sharpe: {parent_a.get('sharpe_ratio', 0.0):.2f} | DSR: {parent_a.get('deflated_sharpe', 0.0):.2f}\n\n"
            f"PARENT B: '{parent_b.get('title')}'\n"
            f"Formula: {parent_b.get('mathematical_formulation')}\n"
            f"Sharpe: {parent_b.get('sharpe_ratio', 0.0):.2f} | DSR: {parent_b.get('deflated_sharpe', 0.0):.2f}\n\n"
            f"KNOWN RED TEAM TRAPS TO AVOID (Hard Negative Memory):\n"
            + "\n".join([f"- {m}" for m in neg_mem]) + "\n\n"
            f"Synthesize an evolved child formula that fuses Parent A's depth dynamics with Parent B's structural damping.\n"
            f"Format response strictly in JSON: 'title', 'theorist', 'rationale', 'mathematical_formulation', 'target_horizon'."
        )

        raw = self.query_fn(
            prompt=crossover_prompt,
            system_prompt=author_agent.system_prompt,
            agent_name=author_agent.name,
            temperature=0.6,
            json_mode=True
        )

        def clean_form(f_str: Optional[str]) -> str:
            if not f_str:
                return "qi_5_t / (spread_t + 1e-4)"
            f_str = str(f_str).strip()
            if "=" in f_str:
                f_str = f_str.split("=", 1)[1].strip()
            return f_str

        clean_a = clean_form(parent_a.get('mathematical_formulation'))
        clean_b = clean_form(parent_b.get('mathematical_formulation'))
        default_child = {
            "title": f"Gen-{self.generation+1} Hybrid: {parent_a.get('title')} x {parent_b.get('title')}",
            "theorist": author_agent.name,
            "rationale": "Fused queue curvature with volatility regime dampening.",
            "mathematical_formulation": f"Signal_t = ({clean_a} + 0.5 * ({clean_b})) / 1.5",
            "target_horizon": "10s"
        }
        child = extract_json_safe(raw, default=default_child)

        child["generation"] = self.generation + 1
        child["lineage"] = [parent_a.get("id"), parent_b.get("id")]
        return child

    def mutate_hypothesis(
        self,
        parent: Dict[str, Any],
        author_agent: SwarmAgentDef
    ) -> Dict[str, Any]:
        """
        Genetic Mutation Operator:
        Mutates normalization factors, scaling exponents, or regime clamps.
        """
        neg_mem = self.get_negative_memory()
        form_p = parent.get('mathematical_formulation') or 'df["qi_5"] / (df["spread"] + 1e-4)'
        mutation_prompt = (
            f"You are {author_agent.name}, {author_agent.specialization}.\n"
            f"Apply a targeted GENETIC MUTATION to refine this high-potential alpha strategy:\n\n"
            f"Parent Strategy: '{parent.get('title')}'\n"
            f"Formula: {form_p}\n\n"
            f"Known Rejection Traps:\n"
            + "\n".join([f"- {m}" for m in neg_mem]) + "\n\n"
            f"Mutate the formulation by adding an asymmetric adverse selection filter or non-linear hyperbolic tangent scaling.\n"
            f"Format response strictly in JSON: 'title', 'theorist', 'rationale', 'mathematical_formulation', 'target_horizon'."
        )

        raw = self.query_fn(
            prompt=mutation_prompt,
            system_prompt=author_agent.system_prompt,
            agent_name=author_agent.name,
            temperature=0.7,
            json_mode=True
        )

        clean_p = str(form_p).strip()
        if "=" in clean_p:
            clean_p = clean_p.split("=", 1)[1].strip()
        default_mutant = {
            "title": f"Mutant {parent.get('title')} [Gen {self.generation+1}]",
            "theorist": author_agent.name,
            "rationale": "Added hyperbolic tangent clamp to suppress extreme tail volatility.",
            "mathematical_formulation": f"Signal_t = np.tanh({clean_p})",
            "target_horizon": "10s"
        }
        mutant = extract_json_safe(raw, default=default_mutant)

        mutant["generation"] = self.generation + 1
        mutant["lineage"] = [parent.get("id")]
        return mutant
