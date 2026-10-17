"""
Institutional Quantitative Hedge Fund Swarm: 110 Specialized Agents
Organized across 6 Divisions mirroring an institutional mega-fund desk.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class SwarmAgentDef:
    id: str
    name: str
    division: str
    specialization: str
    color_hex: str
    system_prompt: str

DIVISIONS = {
    "MICROSTRUCTURE": "Division 1: Order Flow & Microstructure Institute",
    "QUEUE_DYNAMICS": "Division 2: Queue Dynamics & Micro-Price Bureau",
    "REGIMES": "Division 3: Statistical & Regime Detection Desk",
    "COMPILERS": "Division 4: Quantitative Engineering & Compiler Guild",
    "RED_TEAM": "Division 5: Red Team Falsification Bureau",
    "COMMITTEE": "Division 6: Executive Investment Committee & Risk Board"
}

def generate_110_agent_manifest() -> List[SwarmAgentDef]:
    agents: List[SwarmAgentDef] = []

    # Division 1: Order Flow & Microstructure Institute (20 Agents)
    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"ofi_multi_{i:02d}",
            name=f"OFI_MultiLevel_Theorist_{i:02d}",
            division="MICROSTRUCTURE",
            specialization="Multi-Level Order Flow Imbalance Integration (Levels 1-5)",
            color_hex="#00f0ff",
            system_prompt="You are a specialist in multi-level Order Flow Imbalance. Focus on cumulative volume changes at bids and asks across top 5 levels."
        ))
        agents.append(SwarmAgentDef(
            id=f"cancel_vel_{i:02d}",
            name=f"Trade_To_Cancel_Analyst_{i:02d}",
            division="MICROSTRUCTURE",
            specialization="Order Cancellation Velocity & Ghost Liquidity",
            color_hex="#00c8ff",
            system_prompt="You are a specialist in cancel-to-trade ratios and spoofing detection in high-frequency orderbooks."
        ))
        agents.append(SwarmAgentDef(
            id=f"kyle_lambda_{i:02d}",
            name=f"Kyle_Lambda_Estimator_{i:02d}",
            division="MICROSTRUCTURE",
            specialization="Instantaneous Price Impact & Kyle's Lambda",
            color_hex="#38bdf8",
            system_prompt="You model the marginal price impact per unit of aggressive volume on OKX SWAP."
        ))
        agents.append(SwarmAgentDef(
            id=f"replenish_{i:02d}",
            name=f"Cross_Spread_Replenishment_{i:02d}",
            division="MICROSTRUCTURE",
            specialization="Post-Fill Liquidity Replenishment Asymmetry",
            color_hex="#0ea5e9",
            system_prompt="You model how quickly market makers replenish quotes after a spread-crossing trade."
        ))

    # Division 2: Queue Dynamics & Micro-Price Bureau (20 Agents)
    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"micro_curv_{i:02d}",
            name=f"MicroPrice_Curvature_Theorist_{i:02d}",
            division="QUEUE_DYNAMICS",
            specialization="Non-Linear Micro-Price Curvature & Fair Value",
            color_hex="#00ff88",
            system_prompt="You specialize in volume-weighted micro-price deviations and non-linear depth weighting."
        ))
        agents.append(SwarmAgentDef(
            id=f"queue_hazard_{i:02d}",
            name=f"Queue_Depletion_Hazard_{i:02d}",
            division="QUEUE_DYNAMICS",
            specialization="Markov Jump Processes on Queue Extinction",
            color_hex="#10b981",
            system_prompt="You model the instantaneous hazard rate of the best bid/ask queue depleting to zero."
        ))
        agents.append(SwarmAgentDef(
            id=f"order_count_{i:02d}",
            name=f"Order_Count_Dispersion_{i:02d}",
            division="QUEUE_DYNAMICS",
            specialization="Order Count Dispersion & Institutional Resting Clues",
            color_hex="#059669",
            system_prompt="You analyze the ratio of total depth to number of orders at top levels to identify hidden whales."
        ))
        agents.append(SwarmAgentDef(
            id=f"depth_decay_{i:02d}",
            name=f"Level20_Depth_Decay_{i:02d}",
            division="QUEUE_DYNAMICS",
            specialization="Power-Law Depth Decay Across 20 Levels",
            color_hex="#34d399",
            system_prompt="You model the geometric slope of liquidity decay from level 1 down to level 20."
        ))

    # Division 3: Statistical & Regime Detection Desk (20 Agents)
    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"hawkes_{i:02d}",
            name=f"Hawkes_Process_Modeler_{i:02d}",
            division="REGIMES",
            specialization="Self-Exciting Point Processes & Event Clustering",
            color_hex="#ffb700",
            system_prompt="You formulate self-exciting Hawkes processes for trade arrival bursts and feedback loops."
        ))
        agents.append(SwarmAgentDef(
            id=f"jump_diff_{i:02d}",
            name=f"Jump_Diffusion_Analyst_{i:02d}",
            division="REGIMES",
            specialization="Discontinuous Jump Predictors & Volatility Bursts",
            color_hex="#f59e0b",
            system_prompt="You detect impending price jumps by analyzing spread widening and tail order cancellation."
        ))
        agents.append(SwarmAgentDef(
            id=f"spread_state_{i:02d}",
            name=f"Spread_Regime_StateMachine_{i:02d}",
            division="REGIMES",
            specialization="Hidden Markov Models on Spread Regimes",
            color_hex="#d97706",
            system_prompt="You classify market states into Tight-Stable, Choppy-Wide, and Crisis-Breakout."
        ))
        agents.append(SwarmAgentDef(
            id=f"vol_cluster_{i:02d}",
            name=f"Volatility_Clustering_Quant_{i:02d}",
            division="REGIMES",
            specialization="Ultra-High-Frequency Realized Volatility Clusters",
            color_hex="#fbbf24",
            system_prompt="You compute rolling high-frequency volatility estimators and conditional threshold scalers."
        ))

    # Division 4: Quantitative Engineering & Compiler Guild (20 Agents)
    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"polars_dev_{i:02d}",
            name=f"Vectorized_Polars_Architect_{i:02d}",
            division="COMPILERS",
            specialization="High-Throughput Vectorized Polars Execution",
            color_hex="#a855f7",
            system_prompt="You write zero-copy vectorized Polars expressions for complex mathematical microstructure signals."
        ))
        agents.append(SwarmAgentDef(
            id=f"reflex_debug_{i:02d}",
            name=f"Reflexion_Sandbox_Debugger_{i:02d}",
            division="COMPILERS",
            specialization="Self-Correcting Syntax & Tensor Shape Validation",
            color_hex="#c084fc",
            system_prompt="You catch null values, divide-by-zero, and shape mismatches in generated quantitative code."
        ))
        agents.append(SwarmAgentDef(
            id=f"latency_opt_{i:02d}",
            name=f"Latency_Profile_Optimizer_{i:02d}",
            division="COMPILERS",
            specialization="Sub-Millisecond Vector Caching & Rolling Aggregates",
            color_hex="#9333ea",
            system_prompt="You optimize signal calculation time to execute in under 5 microseconds per tick."
        ))
        agents.append(SwarmAgentDef(
            id=f"kernel_comp_{i:02d}",
            name=f"Kernel_Compiler_Specialist_{i:02d}",
            division="COMPILERS",
            specialization="Non-Linear Sigmoid & Hyperbolic Signal Clamping",
            color_hex="#7e22ce",
            system_prompt="You normalize raw mathematical outputs into bounded [-1.0, 1.0] execution signals."
        ))

    # Division 5: Red Team Falsification Bureau (20 Agents)
    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"lookahead_inq_{i:02d}",
            name=f"Lookahead_Inversion_Inquisitor_{i:02d}",
            division="RED_TEAM",
            specialization="Temporal Inversion & Lookahead Bias Auditing",
            color_hex="#ff0055",
            system_prompt="You shift signals temporally to ruthlessly expose future price leaks and data snooping."
        ))
        agents.append(SwarmAgentDef(
            id=f"latency_tort_{i:02d}",
            name=f"Latency_Stress_Torturer_{i:02d}",
            division="RED_TEAM",
            specialization="Execution Delay Degradation (1s, 2s, 5s Delays)",
            color_hex="#e11d48",
            system_prompt="You inject deliberate exchange latency delays to check if a strategy's edge instantly decays."
        ))
        agents.append(SwarmAgentDef(
            id=f"taker_exec_{i:02d}",
            name=f"Taker_Fee_Executioner_{i:02d}",
            division="RED_TEAM",
            specialization="Fee Barrier Stress Testing (Maker vs Taker Drag)",
            color_hex="#be123c",
            system_prompt="You enforce realistic transaction costs and maker-rebate vs adverse-selection tradeoffs."
        ))
        agents.append(SwarmAgentDef(
            id=f"dsr_police_{i:02d}",
            name=f"DSR_Overfit_Policeman_{i:02d}",
            division="RED_TEAM",
            specialization="Deflated Sharpe Ratio & Multiple-Testing Penalties",
            color_hex="#f43f5e",
            system_prompt="You enforce López de Prado's Deflated Sharpe Ratio penalty based on the hundreds of lab trials."
        ))

    # Division 6: Executive Investment Committee & Risk Board (10 Agents)
    committee_roles = [
        ("cio", "Chief_Investment_Officer", "Global Capital Allocation & Edge Conviction", "#eab308"),
        ("head_micro", "Head_Of_Microstructure_Research", "Academic & Economic Realism Verification", "#facc15"),
        ("head_risk", "Head_Of_Quantitative_Risk", "Maximum Drawdown & Tail-Risk Gatekeeper", "#ca8a04"),
        ("chief_audit", "Chief_Falsification_Officer", "Supreme Arbiter of Red Team Findings", "#b45309"),
        ("lead_port", "Lead_Portfolio_Architect", "Cross-Strategy Decorrelation & Pareto Frontier", "#854d0e")
    ]
    for r_id, r_name, r_spec, r_col in committee_roles:
        agents.append(SwarmAgentDef(
            id=r_id,
            name=r_name,
            division="COMMITTEE",
            specialization=r_spec,
            color_hex=r_col,
            system_prompt=f"You are the {r_name}. You hold voting veto power over whether proposed strategies receive live capital."
        ))

    for i in range(1, 6):
        agents.append(SwarmAgentDef(
            id=f"peer_referee_{i:02d}",
            name=f"Senior_Peer_Referee_{i:02d}",
            division="COMMITTEE",
            specialization="Double-Blind Strategy Whitepaper Peer Review",
            color_hex="#f59e0b",
            system_prompt="You provide rigorous mathematical peer-review critiques on strategy proposals."
        ))

    return agents

SWARM_MANIFEST = generate_110_agent_manifest()
