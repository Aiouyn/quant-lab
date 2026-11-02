"""Quarantine ledger rows written by the retired RSI engine (desk/company/rsi.py before commit 3e512bf).

That engine wrote into the firm's ledger, mostly without being asked:
- `ensure_seeded` ran on every dashboard refresh and every `python -m desk rsi` call. It inserted lessons, memos,
  board theses and factor reports with invented Sharpe ratios and t-stats, and fake primitive-usage rows.
- `python -m desk serve` started a background daemon that bred strategies, evaluated them on synthetic panels,
  promoted them, and wrote 'VERIFIED_LIFT' reasoning episodes, charter overlays and audit blobs.
Agents read lessons, memos, the board and factor leaderboards in every session brief, so those rows steer research.

`quarantine(ledger)` is a dry run by default: it reports what it would move. With apply=True it first backs the
database up (sqlite backup API), then MOVES matching rows into `quarantine_<table>` tables. Nothing is dropped.

Deliberately kept in place: `trials` and `holdout_uses`. They only make the multiple-testing count and the
one-shot holdout rule stricter, so keeping them is the conservative choice.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# authors the retired engine and its UI buttons wrote under (strategy authors, hint authors, event agents)
SYNTHETIC_AUTHORS = (
    'rsi-compiler', 'rsi-evolver', 'rsi-truth-machine', 'rsi-erc-allocator', 'rsi-ablation-engine',
    'rsi-forward-monitor', 'rsi-drift-rebalancer', 'rsi-engine', 'rsi-live-parity-auditor', 'toolsmith-rsi',
    'open-ended-rsi', 'readiness-certifier', 'redteam-critic', 'campaign-compiler', 'campaign-toolsmith',
    'daemon-live-appender', 'daemon-open-ended-rsi', 'execution-appender', 'auditor-lead', 'operator-ui',
    'toolsmith-ui', 'ablation-ui', 'redteam-ui', 'readiness-ui', 'open-ended-ui', 'execution-ui',
    'truth-machine-ui', 'funnel-ui', 'forward-audit-ui',
)
SEEDED_LESSON_PREFIXES = (
    'Cross-asset time-series momentum (1-6m) with inverse-vol class risk parity and VIX regime conditioning achieves',
    'Shorting high-realized-vol altcoin perps without tail clipping loses on the mean',
    'Cross-venue BN-vs-HL funding carry (`bn_funding - hl_funding`) with 24h rolling mean',
    'Binary prediction market favorite-longshot calibration held to resolution without hourly re-trading earns',
    'Harness & Provenance Auditor enforces 100% canonical strategy_eval execution',
)
SEEDED_MEMO_PREFIXES = ('DESK MEMO — Pure Backtest Truth Machine', 'KNOWLEDGE DIGEST (v3 + Pure Backtest')
SEEDED_BOARD_TITLES = (
    'Macro Cross-Asset Trend + VIX Regime Ensemble', 'Step-Hold Perps Funding Crowding Reversion',
    'Cross-Venue BN-HL Carry Maker Hysteresis v3', 'Harness & Live Flux Parity Audit',
)
SEEDED_FACTOR_NAMES = (
    'Vol-Scaled Multi-Horizon Trend (21d/126d)', 'Defensive Low-Vol Risk Parity Tilt',
    'Crowded Funding Reversion Signal (24h)', 'Multi-Horizon Perp Funding + Basis Filter',
    'BN-HL Multi-Horizon Funding Spread', 'Vol-Adjusted Cross-Venue Carry Yield',
    'Favorite-Longshot Initial Calibration Bias', 'Resolution Convergence Calibration Distance',
)
SYNTHETIC_CONTROL_KEYS = (
    'harness_audit', 'live_flux_state', 'dry_run_readiness', 'paper_step_counter', 'live_ingest_cycle',
    'execution_attribution', 'counterfactual_ablation', 'ablation_cache', 'adversarial_redteam',
    'readiness_certification', 'open_ended_rsi', 'treasury_state', 'rsi_daemon', 'toolsmith_synth_counter',
    'live_parity_dry_run',
)


def _ph(seq):
    return ','.join('?' * len(seq))


def _tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def plan(con) -> dict[str, tuple[str, tuple]]:
    """table -> (WHERE clause, params) selecting the rows to quarantine."""
    have = _tables(con)
    lineage_children = [r[0] for r in con.execute('SELECT child_id FROM lineages')] if 'lineages' in have else []
    auth = SYNTHETIC_AUTHORS
    p: dict[str, tuple[str, tuple]] = {}
    strat_where = f'author IN ({_ph(auth)})'
    strat_args: tuple = auth
    if lineage_children:
        strat_where += f' OR id IN ({_ph(lineage_children)})'
        strat_args += tuple(lineage_children)
    p['strategies'] = (strat_where, strat_args)
    fac_where = f'name IN ({_ph(SEEDED_FACTOR_NAMES)}) OR author IN ({_ph(auth)})'
    fac_args: tuple = SEEDED_FACTOR_NAMES + auth
    if lineage_children:
        fac_where += f' OR id IN ({_ph(lineage_children)})'
        fac_args += tuple(lineage_children)
    p['factors'] = (fac_where, fac_args)
    p['lessons'] = (' OR '.join(['claim LIKE ?'] * len(SEEDED_LESSON_PREFIXES)) + f' OR author IN ({_ph(auth)})',
                    tuple(x + '%' for x in SEEDED_LESSON_PREFIXES) + auth)
    p['memos'] = (' OR '.join(['body LIKE ?'] * len(SEEDED_MEMO_PREFIXES)) + f' OR author IN ({_ph(auth)})',
                  tuple(x + '%' for x in SEEDED_MEMO_PREFIXES) + auth)
    p['board'] = (f'title IN ({_ph(SEEDED_BOARD_TITLES)}) OR author IN ({_ph(auth)})', SEEDED_BOARD_TITLES + auth)
    p['primitive_usages'] = (f"ref_id LIKE 'S-seed%' OR ref_id LIKE 'F-seed%' OR agent IN ({_ph(auth)})", auth)
    p['dynamic_hints'] = (f'author IN ({_ph(auth)})', auth)
    p['portfolios'] = (f'author IN ({_ph(auth)})', auth)
    p['events'] = (f'agent IN ({_ph(auth)})', auth)
    p['control'] = (f'key IN ({_ph(SYNTHETIC_CONTROL_KEYS)})', SYNTHETIC_CONTROL_KEYS)
    for t in ('lineages', 'charter_versions', 'rsi_runs', 'rsi_reasoning'):   # only the retired engine wrote these
        p[t] = ('1=1', ())
    return {t: v for t, v in p.items() if t in have}


def quarantine(db_path, apply: bool = False, backup_dir=None) -> dict:
    db_path = Path(db_path)
    if not db_path.exists():
        return {'error': f'ledger not found: {db_path}'}
    con = sqlite3.connect(str(db_path))
    try:
        steps = plan(con)
        counts = {t: con.execute(f'SELECT COUNT(*) FROM {t} WHERE {w}', a).fetchone()[0] for t, (w, a) in steps.items()}
        out = {'ledger': str(db_path), 'apply': apply, 'rows': counts, 'total': sum(counts.values())}
        if not apply:
            out['note'] = 'dry run: nothing changed. Re-run with --apply (a backup is written first).'
            return out
        bdir = Path(backup_dir) if backup_dir else db_path.parent
        bdir.mkdir(parents=True, exist_ok=True)
        backup = bdir / f'{db_path.stem}.pre-quarantine-{time.strftime("%Y%m%d-%H%M%S")}.sqlite3'
        dst = sqlite3.connect(str(backup))
        con.backup(dst)
        dst.close()
        out['backup'] = str(backup)
        moved_lessons = [r[0] for r in con.execute(f'SELECT id FROM lessons WHERE {steps["lessons"][0]}', steps['lessons'][1])] \
            if 'lessons' in steps else []
        moved_reasoning = [r[0] for r in con.execute('SELECT id FROM rsi_reasoning')] if 'rsi_reasoning' in steps else []
        with con:
            for t, (w, a) in steps.items():
                if not counts[t]:
                    continue
                con.execute(f'CREATE TABLE IF NOT EXISTS quarantine_{t} AS SELECT * FROM {t} WHERE 0')
                con.execute(f'INSERT INTO quarantine_{t} SELECT * FROM {t} WHERE {w}', a)
                con.execute(f'DELETE FROM {t} WHERE {w}', a)
            refs = moved_lessons + moved_reasoning
            if refs and 'search' in _tables(con):
                for i in range(0, len(refs), 500):
                    chunk = refs[i:i + 500]
                    con.execute(f'DELETE FROM search WHERE ref IN ({_ph(chunk)})', tuple(chunk))
            con.execute('INSERT OR REPLACE INTO control(key, value) VALUES(?, ?)',
                        ('quarantine_last', f'{{"at": {time.time()}, "total": {out["total"]}}}'))
        out['note'] = f'moved {out["total"]} rows into quarantine_* tables; backup at {backup}'
        return out
    finally:
        con.close()
