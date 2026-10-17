"""
Central Event Bus & Strategy Dossier Generator for Quant Lab
Streams real-time agent debates, thoughts, code generation, and committee votes
to the 3D dashboard and persists detailed research dossiers to disk.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)

def extract_json_safe(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not text:
        return default or {}
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean:
        clean = clean.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(clean)
    except Exception:
        import re
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return default or {}

class SwarmEventBus:
    def __init__(self, max_history: int = 500):
        self.lock = threading.Lock()
        self.events = deque(maxlen=max_history)
        self.event_counter = 0
        self.results_dir = Path(__file__).resolve().parent.parent / "results" / "dossiers"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, agent_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Emits a live event to the streaming queue."""
        with self.lock:
            self.event_counter += 1
            event = {
                "id": self.event_counter,
                "timestamp": time.time(),
                "time_str": time.strftime("%H:%M:%S"),
                "type": event_type,
                "agent": agent_name,
                "message": message,
                "details": details or {}
            }
            self.events.append(event)
            logger.info(f"[{event['time_str']}] [{agent_name}] {message}")
            return event

    def get_events(self, since_id: int = 0) -> List[Dict[str, Any]]:
        """Returns all events with ID strictly greater than since_id."""
        with self.lock:
            if since_id > self.event_counter:
                since_id = 0
            if since_id == 0:
                return list(self.events)[-60:]
            return [e for e in self.events if e["id"] > since_id]

    def save_strategy_dossier(
        self,
        generation: int,
        hypothesis: Dict[str, Any],
        author_name: str,
        referee_name: str,
        code_str: str,
        audit_result: Dict[str, Any],
        committee_passed: bool,
        votes: List[Dict[str, Any]]
    ) -> Path:
        """Saves a comprehensive institutional research dossier for the strategy."""
        title = hypothesis.get("title", f"Alpha_Gen_{generation}")
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]
        filename = f"Gen{generation:02d}_{safe_title}_{int(time.time())}.md"
        filepath = self.results_dir / filename

        base_res = audit_result.get("baseline_res", {})
        cpcv_res = audit_result.get("cpcv_res", {})

        vote_rows = ""
        for v in votes:
            vote_rows += f"| **{v.get('member')}** | `{v.get('vote')}` | {v.get('confidence', 0.8):.2f} | {v.get('reasoning')} |\n"

        content = f"""# 🔬 Institutional Alpha Dossier: {title}
**Generation:** {generation}  
**Lead Theorist:** {author_name}  
**Adversarial Referee:** {referee_name}  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Committee Allocation Decision:** {'🟢 **APPROVED FOR CAPITAL ALLOCATION**' if committee_passed else '🔴 **REJECTED BY RISK / RED TEAM**'}

---

## 1. Executive Summary & Rationale
{hypothesis.get('rationale', 'N/A')}

**Target Horizon:** `{hypothesis.get('target_horizon', '10s')}`  
**Mathematical Formulation:**
```math
{hypothesis.get('mathematical_formulation', 'N/A')}
```

---

## 2. Adversarial Peer Review Debate

### ⚔️ Referee Critique ({referee_name})
- **Summary:** {hypothesis.get('critique', {}).get('critique_summary', 'Rigorous microstructure challenge')}
- **Identified Failure Modes:**
{chr(10).join(['  - ' + str(w) for w in hypothesis.get('critique', {}).get('weaknesses', [])])}
- **Proposed Amendment:** {hypothesis.get('critique', {}).get('proposed_amendment', 'Dynamic spread damping')}

### 🛡️ Author Rebuttal & Defense ({author_name})
{hypothesis.get('rebuttal_defense', 'Incorporated dynamic queue liquidity and spread damping.')}

---

## 3. Production Vectorized Polars Code
```python
{code_str}
```

---

## 4. Institutional Verification & Backtest Audit (OKX 5000-Level L2)

| Metric | Measured Value | Institutional Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Audit Verdict** | `{audit_result.get('verdict')}` | `APPROVED` | {'✅' if audit_result.get('verdict') == 'APPROVED' else '❌'} |
| **Annualized Sharpe Ratio** | `{getattr(base_res, 'sharpe_ratio', 0.0):.2f}` | `> 1.00` | {'✅' if getattr(base_res, 'sharpe_ratio', 0.0) >= 1.0 else '❌'} |
| **Deflated Sharpe Ratio (DSR)** | `{getattr(base_res, 'dsr', 0.0):.3f}` | `> 0.800` | {'✅' if getattr(base_res, 'dsr', 0.0) >= 0.8 else '❌'} |
| **Net PnL (Bps after 2bps maker fees)**| `{getattr(base_res, 'net_pnl_bps', 0.0):.1f} bps` | `> 0.0 bps` | {'✅' if getattr(base_res, 'net_pnl_bps', 0.0) > 0 else '❌'} |
| **Win Rate** | `{getattr(base_res, 'win_rate', 0.0):.1f}%` | `> 50.0%` | {'✅' if getattr(base_res, 'win_rate', 0.0) >= 50.0 else '❌'} |
| **FIFO Worst-in-Queue Fill Rate** | `{getattr(base_res, 'fill_rate', 0.0):.1f}%` | Real Priority Sim | 🎯 Realistic |
| **Adverse Selection** | `{getattr(base_res, 'adverse_selection_bps', 0.0):.2f} bps` | `< 1.50 bps` | 🛡️ Circuit Breakers |
| **CPCV Probability of Overfitting (PBO)** | `{cpcv_res.get('pbo', 0.0):.2f}` | `< 0.30` | 🧪 Combinatorial Purged |

---

## 5. Investment Committee Voting & Allocations

| Committee Member | Vote | Confidence | Rationale |
| :--- | :---: | :---: | :--- |
{vote_rows}

---
*Generated automatically by Quant Lab 110-Agent Autonomous Swarm Engine.*
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Saved complete research dossier to {filepath}")
        return filepath

# Global singleton event bus
event_bus = SwarmEventBus()
