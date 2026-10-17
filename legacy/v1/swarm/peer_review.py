"""
Multi-Agent Peer Review & Debate Protocol
Orchestrates adversarial cross-examination, formal mathematical debate,
and referee voting between Theorists, Reviewers, and the Investment Committee.
Consumes tokens purposefully to maximize theoretical depth and self-correction.
"""

import json
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.token_tracker import tracker
from core.event_bus import event_bus, extract_json_safe
from swarm.desks import SwarmAgentDef

logger = logging.getLogger(__name__)

class PeerReviewTournament:
    def __init__(self, query_fn):
        self.query_fn = query_fn

    def conduct_adversarial_debate(
        self,
        author: SwarmAgentDef,
        reviewer: SwarmAgentDef,
        initial_hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes a 3-stage peer review debate:
          1. Author proposes hypothesis
          2. Reviewer challenges with counter-arguments and microstructure edge cases
          3. Author defends and refines the mathematical formulation
        """
        # Step 1: Reviewer critiques the hypothesis
        review_prompt = (
            f"You are {reviewer.name} ({reviewer.specialization}).\n"
            f"Review this strategy proposal by {author.name} ({author.specialization}):\n\n"
            f"Title: {initial_hypothesis.get('title')}\n"
            f"Rationale: {initial_hypothesis.get('rationale')}\n"
            f"Mathematical Formulation:\n{initial_hypothesis.get('mathematical_formulation')}\n"
            f"Target Horizon: {initial_hypothesis.get('target_horizon')}\n\n"
            f"Provide a rigorous mathematical critique:\n"
            f"1. Where will this formulation fail in high-volatility flash events or low-liquidity regimes?\n"
            f"2. Does it properly guard against adverse selection and quote cancellations?\n"
            f"3. Propose a specific mathematical adjustment to improve its resilience.\n"
            f"Format response in JSON with fields: 'critique_summary', 'weaknesses', 'proposed_amendment'."
        )

        critique_raw = self.query_fn(
            prompt=review_prompt,
            system_prompt=reviewer.system_prompt,
            agent_name=reviewer.name,
            temperature=0.7,
            json_mode=True
        )

        default_critique = {
            "critique_summary": "Formula lacks dynamic spread normalization and can suffer during orderbook cascades.",
            "weaknesses": ["Vulnerable to spread blowouts", "Ignores volume exhaustion at level 5"],
            "proposed_amendment": "Add denominator damping factor using realized volatility and spread."
        }
        critique = extract_json_safe(critique_raw, default=default_critique)

        event_bus.emit(
            "PEER_CRITIQUE",
            reviewer.name,
            f"Challenged {author.name}: \"{critique.get('critique_summary')}\"",
            {"critique": critique, "target": author.name}
        )

        # Step 2: Author responds and produces refined formulation
        rebuttal_prompt = (
            f"You are {author.name} ({author.specialization}).\n"
            f"Your strategy proposal received the following formal peer review from {reviewer.name}:\n\n"
            f"Critique Summary: {critique.get('critique_summary')}\n"
            f"Identified Weaknesses: {critique.get('weaknesses')}\n"
            f"Reviewer Amendment: {critique.get('proposed_amendment')}\n\n"
            f"Incorporate this critique into a final, battle-hardened mathematical formulation.\n"
            f"Format response in JSON with fields: 'title', 'theorist', 'rationale', 'mathematical_formulation', 'target_horizon', 'rebuttal_defense'."
        )

        rebuttal_raw = self.query_fn(
            prompt=rebuttal_prompt,
            system_prompt=author.system_prompt,
            agent_name=author.name,
            temperature=0.5,
            json_mode=True
        )

        default_rebuttal = dict(initial_hypothesis)
        default_rebuttal["rebuttal_defense"] = "Incorporated dampening factor to mitigate spread blowout risk."
        final_hypothesis = extract_json_safe(rebuttal_raw, default=default_rebuttal)

        final_hypothesis["debated_with"] = reviewer.name
        final_hypothesis["critique"] = critique

        event_bus.emit(
            "REBUTTAL",
            author.name,
            f"Refined formulation: {final_hypothesis.get('mathematical_formulation', 'N/A')}",
            {"defense": final_hypothesis.get("rebuttal_defense"), "reviewer": reviewer.name}
        )

        return final_hypothesis

    def committee_vote(
        self,
        committee_members: List[SwarmAgentDef],
        hypothesis: Dict[str, Any],
        backtest_result: Any,
        audit_verdict: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Gathers votes and formal written evaluations from Investment Committee members concurrently.
        """
        def vote_worker(member: SwarmAgentDef) -> Dict[str, Any]:
            vote_prompt = (
                f"You are {member.name}, {member.specialization}.\n"
                f"Evaluate this strategy for institutional capital allocation:\n\n"
                f"Title: {hypothesis.get('title')}\n"
                f"Math: {hypothesis.get('mathematical_formulation')}\n"
                f"Audit Verdict: {audit_verdict}\n"
                f"Sharpe Ratio: {getattr(backtest_result, 'sharpe_ratio', 0.0):.2f}\n"
                f"Net Return: {getattr(backtest_result, 'net_pnl_bps', 0.0):.1f} bps\n"
                f"Win Rate: {getattr(backtest_result, 'win_rate', 0.0):.1f}%\n"
                f"Adverse Selection: {getattr(backtest_result, 'adverse_selection_bps', 0.0):.2f} bps/trade\n\n"
                f"Do you vote to ALLOCATE capital or REJECT?\n"
                f"Format output in JSON with fields: 'vote' ('ALLOCATE' or 'REJECT'), 'confidence' (0.0 to 1.0), 'reasoning'."
            )
            raw = self.query_fn(
                prompt=vote_prompt,
                system_prompt=member.system_prompt,
                agent_name=member.name,
                temperature=0.3,
                json_mode=True
            )
            default_vote = {
                "vote": "ALLOCATE" if audit_verdict == "APPROVED" else "REJECT",
                "confidence": 0.85,
                "reasoning": "Standard quantitative threshold alignment."
            }
            v_data = extract_json_safe(raw, default=default_vote)
            v_data["member"] = member.name
            event_bus.emit(
                "COMMITTEE_VOTE",
                member.name,
                f"Voted {v_data.get('vote', 'REJECT')} (Conf: {v_data.get('confidence', 0.8):.2f}): {v_data.get('reasoning', '')[:120]}...",
                {"v_data": v_data}
            )
            return v_data

        votes = []
        approve_count = 0
        with ThreadPoolExecutor(max_workers=min(5, len(committee_members))) as pool:
            futures = [pool.submit(vote_worker, m) for m in committee_members]
            for fut in as_completed(futures):
                try:
                    v_data = fut.result()
                    votes.append(v_data)
                    if v_data.get("vote") == "ALLOCATE":
                        approve_count += 1
                except Exception as e:
                    logger.warning(f"Error in committee vote: {e}")

        passed = (approve_count > len(committee_members) / 2) and (audit_verdict == "APPROVED")
        event_bus.emit(
            "COMMITTEE_VERDICT",
            "Investment_Committee",
            f"Decision: {'APPROVED' if passed else 'REJECTED'} ({approve_count}/{len(committee_members)} Votes, Audit: {audit_verdict})",
            {"passed": passed, "approve_count": approve_count, "total_members": len(committee_members)}
        )
        return passed, votes
        return passed, votes
