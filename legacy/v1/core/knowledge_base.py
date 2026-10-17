"""
Knowledge Base & Research Ledger (SQLite)
Tracks every theoretical hypothesis, generated code, backtest, and Red Team audit.
Enforces institutional memory to prevent repeated dead-ends and calculate DSR penalties.
"""

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class ResearchLedger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            dirname = os.path.dirname(db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                objective TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER,
                theorist TEXT,
                title TEXT,
                rationale TEXT,
                mathematical_formulation TEXT,
                target_horizon TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS implementations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER,
                code_snippet TEXT,
                syntax_valid INTEGER,
                error_log TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                implementation_id INTEGER,
                dataset_date TEXT,
                trades_count INTEGER,
                win_rate REAL,
                gross_pnl_bps REAL,
                fee_drag_bps REAL,
                net_pnl_bps REAL,
                mean_trade_bps REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown_bps REAL,
                adverse_selection_bps REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                implementation_id INTEGER,
                verdict TEXT, -- 'APPROVED' or 'REJECTED'
                rejection_reasons TEXT, -- JSON array of strings
                deflated_sharpe REAL,
                lookahead_passed INTEGER,
                latency_passed INTEGER,
                fee_resilience_passed INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

    def create_campaign(self, title: str, objective: str) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO campaigns (title, objective, status) VALUES (?, ?, 'ACTIVE')",
                (title, objective)
            )
            return cur.lastrowid

    def record_hypothesis(self, campaign_id: int, theorist: str, title: str, 
                          rationale: str, math_form: str, target_horizon: str) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """INSERT INTO hypotheses 
                   (campaign_id, theorist, title, rationale, mathematical_formulation, target_horizon) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (campaign_id, theorist, title, rationale, math_form, target_horizon)
            )
            return cur.lastrowid

    def record_implementation(self, hypothesis_id: int, code: str, valid: bool, error: str = "") -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO implementations (hypothesis_id, code_snippet, syntax_valid, error_log) VALUES (?, ?, ?, ?)",
                (hypothesis_id, code, 1 if valid else 0, error)
            )
            return cur.lastrowid

    def record_backtest(self, impl_id: int, date_str: str, res: Any) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """INSERT INTO backtest_results 
                   (implementation_id, dataset_date, trades_count, win_rate, gross_pnl_bps, 
                    fee_drag_bps, net_pnl_bps, mean_trade_bps, sharpe_ratio, sortino_ratio, 
                    max_drawdown_bps, adverse_selection_bps) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (impl_id, date_str, res.trades_count, res.win_rate, res.gross_pnl_bps,
                 res.fee_drag_bps, res.net_pnl_bps, res.mean_trade_bps, res.sharpe_ratio,
                 res.sortino_ratio, res.max_drawdown_bps, res.adverse_selection_bps)
            )
            return cur.lastrowid

    def record_audit(self, impl_id: int, verdict: str, reasons: List[str], dsr: float,
                     lookahead: bool, latency: bool, fee_resilience: bool, notes: str = "") -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                """INSERT INTO audits 
                   (implementation_id, verdict, rejection_reasons, deflated_sharpe, 
                    lookahead_passed, latency_passed, fee_resilience_passed, notes) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (impl_id, verdict, json.dumps(reasons), dsr, 1 if lookahead else 0,
                 1 if latency else 0, 1 if fee_resilience else 0, notes)
            )
            return cur.lastrowid

    def get_total_trials_count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM implementations")
            return cur.fetchone()[0]

    def get_approved_alphas(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT h.title, h.theorist, h.mathematical_formulation, 
                       b.sharpe_ratio, b.net_pnl_bps, b.win_rate, a.deflated_sharpe, i.code_snippet
                FROM audits a
                JOIN implementations i ON a.implementation_id = i.id
                JOIN hypotheses h ON i.hypothesis_id = h.id
                JOIN backtest_results b ON b.implementation_id = i.id
                WHERE a.verdict = 'APPROVED'
                ORDER BY a.deflated_sharpe DESC
            """)
            return [dict(row) for row in cur.fetchall()]
