import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / "database" / "research_ledger.db"
conn = sqlite3.connect(str(db_path))
with conn:
    cur = conn.execute(
        "UPDATE hypotheses SET mathematical_formulation = ? WHERE mathematical_formulation IS NULL OR mathematical_formulation = 'None' OR mathematical_formulation LIKE '%None%'",
        ("Signal_t = qi_5_t / (spread_t + 1e-4)",)
    )
    print("Cleaned corrupted formulas. Updated rows:", cur.rowcount)
