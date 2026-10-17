"""
Token & Compute Usage Tracker
Monitors real-time token throughput, prompt/completion token consumption,
and compute expenditure across the 100+ agent swarm.
"""

import time
import threading
from typing import Dict, Any

class TokenTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenTracker, cls).__new__(cls)
                cls._instance._init_tracker()
            return cls._instance

    def _init_tracker(self):
        self.start_time = time.time()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0
        self.tokens_history = []
        self.lock = threading.Lock()

    def record_usage(self, prompt_tokens: int, completion_tokens: int, agent_name: str = "Unknown"):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_api_calls += 1
            now = time.time()
            self.tokens_history.append({
                "timestamp": now,
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "agent": agent_name
            })
            # Keep history capped at 2,000 events
            if len(self.tokens_history) > 2000:
                self.tokens_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = max(time.time() - self.start_time, 1.0)
            total_tokens = self.total_prompt_tokens + self.total_completion_tokens
            tokens_per_sec = total_tokens / elapsed

            # Recent 60-second throughput
            now = time.time()
            recent_tokens = sum(
                e["prompt"] + e["completion"]
                for e in self.tokens_history
                if now - e["timestamp"] <= 60.0
            )
            recent_tps = recent_tokens / 60.0

            return {
                "total_tokens": total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "api_calls": self.total_api_calls,
                "elapsed_seconds": round(elapsed, 1),
                "avg_tokens_per_sec": round(tokens_per_sec, 1),
                "current_tokens_per_sec": round(recent_tps, 1),
                "estimated_cost_usd": round((self.total_prompt_tokens * 0.0000015) + (self.total_completion_tokens * 0.000004), 4)
            }

tracker = TokenTracker()
