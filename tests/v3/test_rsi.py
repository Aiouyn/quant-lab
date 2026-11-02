"""Tests for the honest RSI telemetry layer (desk/company/rsi.py) and the ledger quarantine.

The contract under test: nothing in the RSI layer invents a number. With no evidence, values are None (or 0 for
counts); hint counters only move on real events; the retired self-modification tools are not exposed to agents.
"""
from __future__ import annotations

import json
import sqlite3
import time

from desk.company.hints import hints, record_hint_resolution
from desk.company.ledger import Ledger
from desk.company.roles import charter, ROLES
from desk.company.rsi import (
    active_charter_overlay, compile_error_hints, compute_generation_scoreboard, extract_contrib_imports,
    format_primitive_brief, primitive_lift_leaderboard, record_primitive_outcome, rsi_snapshot, MIN_USES_FOR_STATS,
)
from desk.lab.drills import PROCESS_DRILL_KINDS, evaluate_process_drills, process_drill_adversarial_report
from desk.pipeline.validation import DEV_GATES, check


def test_runtime_entry_points_import_and_charter_builds(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    for role in ROLES:
        text = charter(role, pod='macro', ledger=L)
        assert 'ROLE' in text
    assert active_charter_overlay(L, 'researcher', 'macro') == ''


def test_retired_rsi_tools_are_not_exposed():
    from desk.company.tools import ROLE_TOOLS, SPECS
    retired = {'evolve_population', 'run_rsi_campaign', 'validate_and_promote', 'evolve_primitives',
               'propose_charter_patch', 'operator_bandit'}
    assert not retired & set(SPECS)
    for role, tools in ROLE_TOOLS.items():
        assert not retired & set(tools), role
        assert all(t in SPECS for t in tools), role


def test_roster_roles_all_have_tools_and_charters():
    from desk.company.runtime import roster
    from desk.company.tools import ROLE_TOOLS
    for n in (4, 12, 20, 28):
        for _, role, _ in roster(n):
            assert role in ROLE_TOOLS and role in ROLES


def test_contrib_attribution_only_counts_real_modules(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    code = ('from desk.lib.contrib.har_sigma import har_sigma\n'
            'import desk.lib.contrib.stress_eval as se\n'
            'from desk.lib.contrib.does_not_exist import x\n')
    assert extract_contrib_imports(code) == ['har_sigma', 'stress_eval']
    assert record_primitive_outcome(L, code, 'macro-r1', 'run_python', success=True) == ['har_sigma', 'stress_eval']


def test_primitive_stats_are_null_below_min_uses(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    code = 'from desk.lib.contrib.har_sigma import har_sigma\n'
    record_primitive_outcome(L, code, 'a', 'strategy', ref_id='S-1', success=True, metric_val=9.9)
    record_primitive_outcome(L, code, 'a', 'run_python', success=True)
    row = next(r for r in primitive_lift_leaderboard(L) if r['primitive'] == 'har_sigma')
    assert row['mean_strategy_dev_sharpe'] is None          # one sample is not a mean
    assert row['pass_rate'] is None
    assert row['status'] == 'INSUFFICIENT_DATA'
    for _ in range(MIN_USES_FOR_STATS):
        record_primitive_outcome(L, code, 'a', 'run_python', success=False)
    row = next(r for r in primitive_lift_leaderboard(L) if r['primitive'] == 'har_sigma')
    assert row['pass_rate'] == round(1 / (MIN_USES_FOR_STATS + 1), 3)
    assert row['status'] == 'LOW_PASS_RATE'
    unused = [r for r in primitive_lift_leaderboard(L) if r['total_uses'] == 0]
    assert all(r['pass_rate'] is None and r['status'] == 'UNUSED' for r in unused)
    assert 'har_sigma' in format_primitive_brief(L)


def test_empty_ledger_brief_and_scoreboard_are_empty_not_fabricated(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    assert format_primitive_brief(L) == ''
    board = compute_generation_scoreboard(L)
    for k in ('best_sharpe', 'mean_sharpe', 'best_val_sharpe', 'best_holdout_sharpe', 'portfolio_sharpe',
              'planted_signal_corr', 'drill_verdict', 'null_false_discoveries', 'hint_resolution_rate',
              'mean_primitive_lift', 'forward_sharpe'):
        assert board[k] is None, k
    assert board['forward_verdict'] == 'NOT_RUN'
    assert board['generation_fitness'] == 0.0
    assert rsi_snapshot(L)['kpis']['approved'] == 0


def _strategy(L, sid, status, dev=None, val=None, fwd=None):
    now = time.time()
    L.x('INSERT INTO strategies(id,name,author,pod,code,code_hash,resolution,status,dev_report,val_report,'
        'forward_report,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (sid, sid, 'tester', 'macro', "DESK='macro'", sid, '1d', status,
         json.dumps({'summary': {'sharpe': dev}}) if dev is not None else None,
         json.dumps({'summary': {'sharpe': val}}) if val is not None else None,
         json.dumps(fwd) if fwd else None, now, now))


def test_scoreboard_reads_ledger_and_fitness_counts_only_forward(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    _strategy(L, 'S-a', 'rejected_val', dev=2.0, val=0.3)
    _strategy(L, 'S-b', 'approved', dev=1.0, val=1.2, fwd={'days': 30, 'verdict': 'RUNNING', 'summary': {'sharpe': 0.4}})
    b = compute_generation_scoreboard(L)
    assert b['best_sharpe'] == 2.0 and b['mean_sharpe'] == 1.5 and b['best_val_sharpe'] == 1.2
    assert b['forward_verdict'] == 'RUNNING' and b['forward_sharpe'] == 0.4
    assert b['generation_fitness'] == 0.0                 # nothing decided on forward data yet
    _strategy(L, 'S-c', 'approved', dev=1.0, val=1.0, fwd={'days': 61, 'verdict': 'FORWARD_CONFIRMED', 'summary': {'sharpe': 0.9}})
    _strategy(L, 'S-d', 'approved', dev=1.0, val=1.0, fwd={'days': 61, 'verdict': 'FORWARD_FAILED', 'summary': {'sharpe': -0.2}})
    b = compute_generation_scoreboard(L)
    assert b['forward_confirmed'] == 1 and b['forward_decided'] == 2 and b['generation_fitness'] == 0.5


def test_hint_compiler_starts_counters_at_zero_and_needs_recurrence(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    err = 'ZeroDivisionError: float division by zero'
    L.event('macro-r1', 'RUN_PYTHON', {'status': 'error:1', 'error': err})
    assert compile_error_hints(L)['new_hints_compiled'] == 0       # one occurrence is not a pattern
    L.event('macro-r2', 'RUN_PYTHON', {'status': 'error:1', 'error': err})
    out = compile_error_hints(L)
    assert out['new_hints_compiled'] == 1
    row = L.q('SELECT hits, resolved_count FROM dynamic_hints')[0]
    assert row['hits'] == 0 and row['resolved_count'] == 0
    assert compile_error_hints(L)['new_hints_compiled'] == 0       # idempotent
    served = hints(err, 'error:1', ledger=L, agent='macro-r1')
    assert served
    record_hint_resolution(L, 'macro-r1', True)
    row = L.q('SELECT hits, resolved_count FROM dynamic_hints')[0]
    assert row['hits'] == 1 and row['resolved_count'] == 1
    assert compute_generation_scoreboard(L)['hint_resolution_rate'] == 1.0


def test_process_drills_catch_adversarial_reports():
    results = evaluate_process_drills(DEV_GATES, check)
    assert set(results) == set(PROCESS_DRILL_KINDS)
    for kind, row in results.items():
        assert row['caught'], f'{kind} must be caught by DEV_GATES, failed_gates={row["failed_gates"]}'
    junk = process_drill_adversarial_report('sycophantic_approve_junk')
    assert len(check(DEV_GATES, junk)) > 0


def test_quarantine_moves_only_retired_engine_rows(tmp_path):
    from desk.company.quarantine import quarantine
    db = tmp_path / 'desk.sqlite3'
    L = Ledger(db)
    # rows the retired engine wrote
    L.add_lesson('carry-lead', 'Cross-venue BN-vs-HL funding carry (`bn_funding - hl_funding`) with 24h rolling mean '
                 'and 96h step-hold maker hysteresis earns Dev Sharpe 5.03', 'seed', 0.96, 'carry', ['funding'])
    L.memo('cio', 'desk', 'DESK MEMO — Pure Backtest Truth Machine & Live Dry-Run Mandate: ...')
    _strategy(L, 'S-evo', 'approved', dev=1.5, val=1.1)
    L.add_lineage('S-evo', 'strategy', ['S-x'], 'vol_scale_tail_clip', 'macro', 0.2)
    L.log_primitive_usage('har_sigma', 'macro-r1', 'strategy', ref_id='S-seed1', success=True, metric_val=6.77)
    L.add_dynamic_hint('ZeroDivisionError', 'fake', author='rsi-compiler', hits=10, resolved=7)
    L.log_rsi_run('campaign_gen', 'fake', {})
    L.set('open_ended_rsi', {'qd_score': 51.0})
    # real rows that must stay
    L.add_lesson('macro-r1', 'Trend failed val after costs', 'S-real val report', 0.7, 'macro', [])
    _strategy(L, 'S-real', 'rejected_val', dev=1.6, val=-0.2)
    L.trial('rsi-evolver', 'submission', 'h1', 1.0, 100)            # trials are kept (they only make DSR stricter)

    dry = quarantine(db)
    assert dry['apply'] is False and dry['total'] >= 8
    assert L.q('SELECT COUNT(*) n FROM lessons')[0]['n'] == 2       # dry run changed nothing

    out = quarantine(db, apply=True, backup_dir=tmp_path / 'bk')
    assert (tmp_path / 'bk').exists() and out['backup']
    assert [r['id'] for r in L.q('SELECT id FROM strategies')] == ['S-real']
    assert [r['claim'] for r in L.q('SELECT claim FROM lessons')] == ['Trend failed val after costs']
    assert L.q('SELECT COUNT(*) n FROM memos')[0]['n'] == 0
    assert L.q('SELECT COUNT(*) n FROM primitive_usages')[0]['n'] == 0
    assert L.q('SELECT COUNT(*) n FROM dynamic_hints')[0]['n'] == 0
    assert L.q('SELECT COUNT(*) n FROM lineages')[0]['n'] == 0
    assert L.get('open_ended_rsi') is None
    assert L.q('SELECT COUNT(*) n FROM trials')[0]['n'] == 1
    assert L.q('SELECT COUNT(*) n FROM quarantine_strategies')[0]['n'] == 1
    assert not [r for r in L.search('step-hold') if r['kind'] == 'lesson']
    con = sqlite3.connect(out['backup'])
    assert con.execute('SELECT COUNT(*) FROM strategies').fetchone()[0] == 2     # the backup is the pre-move state
    con.close()
