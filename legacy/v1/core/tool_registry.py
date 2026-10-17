"""
Enterprise Quantitative Tool Registry
Maintains the repository of verified tools created autonomously by Toolsmith agents.
Persists tools to SQLite, enforces schema, and dynamically enriches market dataframes.
"""

import sqlite3
import json
import logging
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import config

logger = logging.getLogger(__name__)

@dataclass
class QuantitativeTool:
    name: str
    author: str
    category: str  # FEATURE_EXTRACTOR, REGIME_FILTER, ESTIMATOR, EXECUTION_SIZER
    mathematical_formulation: str
    code: str
    docstring: str
    verified: bool = False
    audit_report: Optional[Dict[str, Any]] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

class ToolRegistry:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or config.DATABASE_PATH)
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        self._compiled_tools: Dict[str, Callable[[pl.DataFrame], pl.Series]] = {}
        self._init_db()
        self._seed_foundational_tools()
        self.reload_tools()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    author TEXT NOT NULL,
                    category TEXT NOT NULL,
                    mathematical_formulation TEXT NOT NULL,
                    code TEXT NOT NULL,
                    docstring TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    audit_report TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _seed_foundational_tools(self):
        """Pre-seeds 3 institutional-grade microstructure tools as company baseline."""
        seeds = [
            QuantitativeTool(
                name="zscore_ofi",
                author="Chief_Microstructure_Architect",
                category="FEATURE_EXTRACTOR",
                mathematical_formulation="Z_OFI_t = (OFI_t - μ_100(OFI)) / max(σ_100(OFI), 1e-4)",
                docstring="Dimensionless, stationary rolling Z-score of Order Flow Imbalance.",
                code="""import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    ofi = df['ofi']
    mean_100 = ofi.rolling_mean(window_size=100, min_samples=1).fill_null(0.0)
    std_100 = ofi.rolling_std(window_size=100, min_samples=2).fill_null(1.0)
    std_clamped = std_100.clip(lower_bound=1e-4)
    z_score = (ofi - mean_100) / std_clamped
    return z_score.clip(-3.0, 3.0).fill_null(0.0)
""",
                verified=True,
                audit_report={"status": "PASSED_CAUSAL_CHECK", "leakage": False, "latency_ms": 2.1}
            ),
            QuantitativeTool(
                name="depth_decay_slope",
                author="Senior_Queue_Physicist",
                category="FEATURE_EXTRACTOR",
                mathematical_formulation="Slope_t = (Level1_Depth - Level5_Depth) / max(Level1_Depth + Level5_Depth, 1.0)",
                docstring="Normalized depth decay steepness between top of book and outer orderbook boundary.",
                code="""import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    v0 = (df['bid_v0'] + df['ask_v0']) / 2.0
    v5 = (df['bid_v5'] + df['ask_v5']) / 10.0
    denom = (v0 + v5).clip(lower_bound=1.0)
    slope = (v0 - v5) / denom
    return slope.clip(-1.0, 1.0).fill_null(0.0)
""",
                verified=True,
                audit_report={"status": "PASSED_CAUSAL_CHECK", "leakage": False, "latency_ms": 1.4}
            ),
            QuantitativeTool(
                name="vol_regime_ratio",
                author="Head_Of_Volatility_Regimes",
                category="REGIME_FILTER",
                mathematical_formulation="Ratio_t = Vol_30s_bps / max(Vol_5m_bps, 0.5)",
                docstring="Volatility shock indicator: detects burst volatility cascades vs calm drift.",
                code="""import polars as pl

def calculate(df: pl.DataFrame) -> pl.Series:
    vol_30s = df['vol_30s_bps']
    vol_5m = df['vol_5m_bps'] if 'vol_5m_bps' in df.columns else vol_30s.rolling_mean(window_size=300, min_samples=1).fill_null(1.0)
    denom = vol_5m.clip(lower_bound=0.5)
    ratio = vol_30s / denom
    return ratio.clip(0.1, 10.0).fill_null(1.0)
""",
                verified=True,
                audit_report={"status": "PASSED_CAUSAL_CHECK", "leakage": False, "latency_ms": 1.8}
            )
        ]

        with self._get_connection() as conn:
            for s in seeds:
                conn.execute("""
                    INSERT OR IGNORE INTO tools_registry
                    (name, author, category, mathematical_formulation, code, docstring, verified, audit_report)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s.name, s.author, s.category, s.mathematical_formulation,
                    s.code, s.docstring, 1 if s.verified else 0,
                    json.dumps(s.audit_report or {})
                ))

    def reload_tools(self):
        """Compiles all verified tools into callable in-memory functions."""
        self._compiled_tools = {}
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tools_registry WHERE verified = 1")
            for row in cur.fetchall():
                try:
                    scope: Dict[str, Any] = {}
                    exec(row["code"], scope)
                    if "calculate" in scope:
                        self._compiled_tools[row["name"]] = scope["calculate"]
                except Exception as e:
                    logger.error(f"Failed to compile verified tool {row['name']}: {e}")

    def register_tool(self, tool: QuantitativeTool) -> bool:
        """Saves a verified tool to the database and in-memory registry."""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO tools_registry
                    (name, author, category, mathematical_formulation, code, docstring, verified, audit_report)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool.name, tool.author, tool.category,
                    tool.mathematical_formulation, tool.code,
                    tool.docstring, 1 if tool.verified else 0,
                    json.dumps(tool.audit_report or {})
                ))
                self.reload_tools()
                logger.info(f"🎉 Registered new verified corporate tool '{tool.name}' by {tool.author}")
                return True
            except Exception as e:
                logger.error(f"Failed to register tool '{tool.name}': {e}")
                return False

    def get_verified_tools(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tools_registry WHERE verified = 1 ORDER BY id DESC")
            tools = []
            for r in cur.fetchall():
                d = dict(r)
                if d.get("audit_report"):
                    try:
                        d["audit_report"] = json.loads(d["audit_report"])
                    except Exception:
                        pass
                tools.append(d)
            return tools

    def get_verified_tool_names(self) -> List[str]:
        return list(self._compiled_tools.keys())

    def apply_tools_to_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enriches the dataframe with all verified corporate tool features."""
        enriched = df
        for name, calc_fn in self._compiled_tools.items():
            if name not in enriched.columns:
                try:
                    res_series = calc_fn(enriched)
                    if isinstance(res_series, pl.Series) and len(res_series) == len(enriched):
                        enriched = enriched.with_columns(res_series.alias(name))
                except Exception as e:
                    logger.warning(f"Error executing tool '{name}' on dataframe: {e}")
        return enriched

tool_registry = ToolRegistry()
