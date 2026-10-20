"""Tests for Desk OS v3 Closed-Loop Recursive Self-Improvement (RSI) Engine."""
from __future__ import annotations

import json
import numpy as np
import polars as pl
import pytest

from desk.company.ledger import Ledger
from desk.company.hints import hints, record_hint_resolution
from desk.company.roles import charter
from desk.company.rsi import (
    extract_contrib_imports,
    record_primitive_outcome,
    primitive_lift_leaderboard,
    format_primitive_brief,
    compile_error_hints,
    mutate_or_repair_strategy,
    crossover_factors,
    evolve_population,
    propose_charter_patch,
    evaluate_rsi_drill,
    ensure_seeded,
    rsi_snapshot,
    compute_generation_scoreboard,
    gate_failure_leaderboard,
    record_gate_failure,
    operator_bandit_leaderboard,
    _synth_panel_for_desk,
)
from desk.lab.drills import PROCESS_DRILL_KINDS, evaluate_process_drills, process_drill_adversarial_report
from desk.pipeline.validation import DEV_GATES, check
from desk.lab.strategy import load_strategy, causality_test


def test_ast_contrib_attribution_and_empirical_lift(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    code = """
from desk.lib.contrib.har_sigma import har_sigma
import desk.lib.contrib.stress_eval as se
from desk.lib.contrib import purged_day_folds, cost_eval
"""
    prims = extract_contrib_imports(code)
    assert set(prims) == {'cost_eval', 'har_sigma', 'purged_day_folds', 'stress_eval'}

    # Record high-Sharpe outcomes for har_sigma and low pass rate for sparse_extreme_trigger
    for sr in (1.65, 1.82, 1.95):
        record_primitive_outcome(L, "from desk.lib.contrib.har_sigma import har_sigma", "macro-r1", "strategy", "S-1", True, sr)
        record_primitive_outcome(L, "from desk.lib.contrib.har_sigma import har_sigma", "macro-r1", "run_python", "", True, 1.0)
    for _ in range(5):
        record_primitive_outcome(L, "from desk.lib.contrib.sparse_extreme_trigger import rolling_q", "events-r1", "run_python", "", False, 0.0)

    lb = primitive_lift_leaderboard(L)
    by_name = {r['primitive']: r for r in lb}
    assert by_name['har_sigma']['sharpe_lift'] > 0
    assert by_name['har_sigma']['status'] == 'HIGH_LIFT'
    assert by_name['sparse_extreme_trigger']['pass_rate'] < 0.5
    assert by_name['sparse_extreme_trigger']['status'] == 'NEEDS_REFACTOR'

    brief = format_primitive_brief(L)
    assert 'har_sigma' in brief and 'NEEDS_REFACTOR' in brief


def test_error_to_hint_compiler_and_resolution_tracking(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    # Log recurring unhinted sandbox failures
    L.event('macro-r1', 'RUN_PYTHON', {
        'status': 'error', 'seconds': 0.4,
        'error': "ComputeError: division by zero in rolling_mean expression"
    })
    L.event('perps-r1', 'RUN_PYTHON', {
        'status': 'error', 'seconds': 0.3,
        'error': "CustomPanelAlignmentError: panel timestamps misaligned across coins after join"
    })
    L.event('perps-r2', 'RUN_PYTHON', {
        'status': 'error', 'seconds': 0.3,
        'error': "CustomPanelAlignmentError: panel timestamps misaligned across symbols after join"
    })

    res = compile_error_hints(L)
    assert res['active_dynamic_hints'] >= 2

    # Trigger dynamic hint matching for agent 'perps-r1'
    matched = hints("CustomPanelAlignmentError: panel timestamps misaligned across coins", "error", ledger=L, agent='perps-r1')
    assert any("CustomPanelAlignmentError" in m for m in matched)

    # Verify hit count incremented and subsequent success increments resolved_count
    row_before = L.q("SELECT hits, resolved_count FROM dynamic_hints WHERE pattern LIKE '%CustomPanelAlignmentError%'")[0]
    assert row_before['hits'] >= 1
    record_hint_resolution(L, 'perps-r1', succeeded=True)
    row_after = L.q("SELECT hits, resolved_count FROM dynamic_hints WHERE pattern LIKE '%CustomPanelAlignmentError%'")[0]
    assert row_after['resolved_count'] == row_before['resolved_count'] + 1


@pytest.mark.parametrize('operator,reasons', [
    ('vol_scale_tail_clip', ['dev_mean_ex_top5_days>0']),
    ('hysteresis_buffer', ['fees_x1.5_profitable']),
    ('regime_breadth_gate', ['both_halves_positive']),
    ('maker_execution_switch', []),
])
def test_strategy_gate_repair_operators_preserve_causality(operator, reasons):
    parent_code = """DESK = 'perps'
PARAMS = {'w': 24}
def signal(df, params):
    r = df.select(pl.col('px').cast(pl.Float64).pct_change(int(params.get('w', 24))).over('coin').alias('r'))['r'].to_numpy()
    return np.clip(np.tanh(np.nan_to_num(r, 0.0) * 15.0), -1.0, 1.0)
"""
    mut = mutate_or_repair_strategy(parent_code, operator=operator, gate_reasons=reasons)
    mod = load_strategy(mut['code'])
    panel = _synth_panel_for_desk('perps', 360)
    caus = causality_test(mod, panel)
    assert caus['ok'], caus


def test_multi_desk_factor_crossover_and_evolve_population(tmp_path):
    from desk.config import SETTINGS
    import dataclasses
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))

    facs = L.q("SELECT id, name, code, resolution, score FROM factors WHERE resolution='macro' AND status='active'")
    assert len(facs) >= 2
    crossed = crossover_factors(facs[:2], desk='macro', mode='strategy')
    mod = load_strategy(crossed['code'])
    panel = _synth_panel_for_desk('macro', 400)
    assert causality_test(mod, panel)['ok']

    # Run a full evolutionary pass and verify lineage + DSR trial logging
    trials_before = L.trial_stats()['evaluations']
    ev = evolve_population(L, settings=s, desk='macro', max_children=2)
    assert ev['evolved_count'] >= 1
    assert L.trial_stats()['evaluations'] >= trials_before + ev['evolved_count']
    snap = rsi_snapshot(L)
    assert len(snap['lineages']) >= 1


def test_drill_gated_charter_evolution(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    cand = propose_charter_patch(L, role='researcher', pod='macro')
    assert cand['status'] == 'candidate'

    drill = evaluate_rsi_drill(L, charter_id=cand['id'])
    assert drill['null_false_discoveries'] == 0
    assert drill['planted_signal_corr'] >= 0.70
    assert drill['error_rate_delta'] <= 0.0
    assert drill['charter_bars_ok'] is True
    assert drill['process_drills_ok'] is True
    assert set(drill['process_drills']) == set(PROCESS_DRILL_KINDS)
    assert all(p['caught'] for p in drill['process_drills'].values())
    assert drill['verdict'] == 'PROMOTED'

    full_charter = charter('researcher', 'macro', ledger=L)
    assert 'RSI CHARTER OVERLAY' in full_charter


def test_opencode_deepseek_v4_1_flash_provider_config(monkeypatch):
    from desk.config import Settings
    from desk.company.llm import _opencode_descending_session_id
    monkeypatch.setenv('OPENCODE_API_KEY', 'oc-test-key')
    monkeypatch.setenv('MODEL_NAME', 'mimo-v2.6-pro')
    monkeypatch.setenv('DESK_FLASH_MODEL', 'deepseek-v4.1-flash-free')
    monkeypatch.setenv('MIMO_BASE_URL', 'https://api.xiaomimimo.com/v1')
    monkeypatch.delenv('LLM_BASE_URL', raising=False)
    s = Settings()
    assert s.provider == 'opencode'
    assert s.base_url == 'https://opencode.ai/zen/v1'
    assert s.model == 'deepseek-v4.1-flash-free'
    assert s.flash_model == 'deepseek-v4.1-flash-free'
    assert s.api_key == 'oc-test-key'
    sid = _opencode_descending_session_id()
    assert sid.startswith('ses_') and len(sid) == 30


def test_store_builder_and_multi_desk_panels(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.data.store_builder import ensure_store_panels, store_panel_status
    from desk.lab.macro import MacroStore
    from desk.lab.perps import PerpsStore
    from desk.lab.carry import CarryStore
    from desk.lab.pred import PredStore

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    built = ensure_store_panels(s, force=True, try_remote=False)
    assert set(built.keys()) == {'macro', 'perps', 'carry', 'pred'}

    status = store_panel_status(s)
    assert all(p['ready'] and p['rows'] > 0 and p['assets'] > 0 for p in status)

    # Verify each desk store loads dev & val splits cleanly
    m_dev, _, _ = MacroStore(s, allowed=('dev', 'val')).load('dev', warmup_days=30)
    p_dev, _, _ = PerpsStore(s, allowed=('dev', 'val')).load('dev', warmup_days=5)
    c_dev, _, _ = CarryStore(s, allowed=('dev', 'val')).load('dev', warmup_days=5)
    pr_dev, _, _ = PredStore(s, allowed=('dev', 'val')).load('dev')
    assert len(m_dev) > 100 and len(p_dev) > 100 and len(c_dev) > 100 and len(pr_dev) > 100


def test_llm_breed_strategy_deepseek_v4_1_flash_free(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.company.rsi import llm_breed_strategy

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))

    def fake_deepseek_llm(prompt: str, desk: str = 'macro', mode: str = 'crossover') -> str:
        assert desk in prompt
        return f"""```python
DESK = {desk!r}
PARAMS = {{'w_fast': 14, 'w_slow': 42, 'clip_w': 0.75}}
def signal(df, params):
    from desk.lib.contrib.har_sigma import har_sigma
    col = 'adjclose' if 'adjclose' in df.columns else 'px'
    grp = 'symbol' if 'symbol' in df.columns else 'coin'
    r1 = df.select(pl.col(col).cast(pl.Float64).pct_change(int(params.get('w_fast', 14))).over(grp).alias('r'))['r'].to_numpy()
    r2 = df.select(pl.col(col).cast(pl.Float64).pct_change(int(params.get('w_slow', 42))).over(grp).alias('r'))['r'].to_numpy()
    v = df.select(pl.col(col).cast(pl.Float64).pct_change().rolling_std(21, min_samples=5).over(grp).alias('v'))['v'].to_numpy()
    raw = (0.5 * np.nan_to_num(r1, 0.0) + 0.5 * np.nan_to_num(r2, 0.0)) / (np.nan_to_num(v, 0.01) + 1e-4)
    w = np.tanh(raw * 0.4)
    if 'tradable' in df.columns:
        w = w * df['tradable'].to_numpy()
    return np.clip(w, -float(params.get('clip_w', 0.75)), float(params.get('clip_w', 0.75)))
```"""

    bred = llm_breed_strategy(["# parent 1"], desk='macro', mode='crossover', llm_fn=fake_deepseek_llm)
    assert bred is not None and bred['llm_generated'] is True
    assert bred['operator'] == 'llm_crossover_deepseek'

    ev = evolve_population(L, settings=s, desk='macro', max_children=1, use_llm=True, llm_fn=fake_deepseek_llm)
    assert ev['evolved_count'] >= 1
    assert any(c.get('llm_generated') for c in ev['children'])


def test_autonomous_toolsmith_evolve_primitives(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.company.rsi import evolve_primitives

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = Ledger(tmp_path / 'desk.sqlite3')
    for _ in range(4):
        record_primitive_outcome(L, "from desk.lib.contrib.sparse_extreme_trigger import rolling_q", "events-r1", "run_python", "", False, 0.0)

    res = evolve_primitives(L, settings=s, use_llm=False)
    assert res['primitives_evolved'] >= 1
    assert res['drill_gate'] == 'PROMOTED'
    names = {a['primitive'] for a in res['actions']}
    assert 'sparse_extreme_trigger' in names or 'xs_rank_orthogonalize' in names


def test_multi_generation_rsi_campaign(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.company.rsi import run_rsi_campaign

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))

    camp = run_rsi_campaign(L, settings=s, generations=2, desks=['macro', 'perps'], max_children_per_gen=1, use_llm=False)
    assert camp['generations_run'] == 2
    assert len(camp['generations']) == 2
    # Honest structure: fitness present; Sharpes may be null or real — never required ≥ 1.2
    assert 'generation_fitness' in camp['generations'][-1]
    assert 'scoreboard' in camp
    assert camp['scoreboard']['generation_fitness'] == camp['generations'][-1]['generation_fitness']
    for g in camp['generations']:
        assert 'children_bred' in g
        # children_bred is telemetry, not a success claim
        assert g.get('drill_verdict') in ('PROMOTED', 'REJECTED', None)

    snap = rsi_snapshot(L, settings=s)
    assert snap['kpis']['campaign_generations'] >= 2
    assert 'generation_fitness' in snap['kpis']
    assert len(snap['store_panels']) == 4


def test_empty_scoreboard_is_null_not_fabricated(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    board = compute_generation_scoreboard(L)
    assert board['best_sharpe'] is None
    assert board['mean_sharpe'] is None
    assert board['best_val_sharpe'] is None
    assert board['portfolio_sharpe'] is None
    assert board['planted_signal_corr'] is None
    assert board['drill_verdict'] is None
    assert board['mean_primitive_lift'] is None
    # Must not invent classic fake floors
    assert board['best_sharpe'] != 1.45
    assert board['mean_sharpe'] != 1.20
    assert board['generation_fitness'] == 0.0


def test_no_fabricated_bandit_or_drill_defaults(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    bandit = operator_bandit_leaderboard(L)
    assert len(bandit) >= 5
    for row in bandit:
        assert row['mean_dev_sharpe'] is None
        assert row['mean_val_sharpe'] is None
        assert row['mean_dev_sharpe'] != 1.25
        assert row['mean_val_sharpe'] != 1.05
    snap = rsi_snapshot(L)
    assert snap['kpis']['drill_gate_verdict'] is None
    assert snap['kpis']['planted_signal_corr'] is None
    assert snap['kpis']['best_campaign_sharpe'] is None
    assert snap['kpis']['hint_resolution_rate'] is None


def test_forward_missing_is_not_run(tmp_path):
    import dataclasses
    import time
    from desk.config import SETTINGS

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))
    now = time.time() + 1e6
    # Seed an approved strategy without forward_report
    L.x(
        "INSERT INTO strategies(id,name,author,pod,code,code_hash,resolution,status,val_report,holdout_report,created,updated) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            'S-FWD-TEST', 'fwd_missing', 'tester', 'macro',
            "DESK='macro'\nPARAMS={}\ndef signal(df, params):\n    return np.zeros(len(df))",
            'hash-fwd', '1d', 'approved',
            json.dumps({'summary': {'sharpe': 1.1}}),
            json.dumps({'summary': {'sharpe': 1.0}}),
            now, now,
        ),
    )
    L.x(
        "INSERT INTO portfolios(id,author,members,weights,report,status,created) VALUES(?,?,?,?,?,?,?)",
        ('P-FWD-TEST', 'tester', json.dumps(['S-FWD-TEST']), json.dumps({'S-FWD-TEST': 1.0}),
         json.dumps({'sharpe': 1.0}), 'active', now),
    )
    snap = rsi_snapshot(L, settings=s)
    members = (snap.get('latest_portfolio') or {}).get('members') or []
    assert members, 'expected portfolio member'
    m = next(x for x in members if x['id'] == 'S-FWD-TEST')
    assert m['forward_sharpe'] is None
    assert m['forward_verdict'] == 'NOT_RUN'
    fwd = next(x for x in snap['forward_survival'] if x['id'] == 'S-FWD-TEST')
    assert fwd['forward_sharpe'] is None
    assert fwd['verdict'] == 'NOT_RUN'


def test_gate_failure_credit_and_brief(tmp_path):
    L = Ledger(tmp_path / 'desk.sqlite3')
    record_gate_failure(
        L, strategy_id='S1', stage='dev', gates=['fees_x1.5_profitable', 'causal'],
        operator='vol_scale_tail_clip', desk='macro', primitives=['har_sigma'],
    )
    record_gate_failure(
        L, strategy_id='S2', stage='val', gates=['fees_x1.5_profitable'],
        operator='vol_scale_tail_clip', desk='perps', primitives=['har_sigma'],
    )
    lb = gate_failure_leaderboard(L)
    assert lb and lb[0]['gate'] == 'fees_x1.5_profitable'
    assert lb[0]['count'] == 2
    brief = format_primitive_brief(L)
    assert 'GATE FAILURE CREDIT' in brief
    assert 'fees_x1.5_profitable' in brief
    bandit = operator_bandit_leaderboard(L)
    by_op = {r['operator']: r for r in bandit}
    assert by_op['vol_scale_tail_clip']['gate_cluster_penalty'] > 0


def test_process_drills_catch_adversarial_reports():
    results = evaluate_process_drills(DEV_GATES, check)
    assert set(results) == set(PROCESS_DRILL_KINDS)
    for kind, row in results.items():
        assert row['caught'], f'{kind} must be caught by DEV_GATES, failed_gates={row["failed_gates"]}'
    junk = process_drill_adversarial_report('sycophantic_approve_junk')
    assert junk['summary']['sharpe'] is None
    assert len(check(DEV_GATES, junk)) > 0


def test_truth_machine_gate_and_erc_master_portfolio(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.company.rsi import validate_and_promote_population, operator_bandit_leaderboard

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))
    evolve_population(L, settings=s, max_children=3, use_llm=False)

    res = validate_and_promote_population(L, settings=s, max_candidates=6)
    assert res['total_approved'] >= 1
    assert res['portfolio'] is not None
    assert 'weights' in res['portfolio'] and len(res['portfolio']['weights']) >= 1

    # Verify Meta-Bandit operator leaderboard weights sum to ~1.0
    bandit = operator_bandit_leaderboard(L)
    assert len(bandit) >= 5
    prob_sum = sum(b['selection_prob'] for b in bandit)
    assert 0.98 <= prob_sum <= 1.02

    snap = rsi_snapshot(L, settings=s)
    assert snap['kpis']['approved_strategies'] >= 1
    assert snap['latest_portfolio'] is not None
    assert len(snap['latest_portfolio']['members']) >= 1


def test_continuous_rsi_daemon_lifecycle(tmp_path):
    import dataclasses
    import time
    from desk.config import SETTINGS
    from desk.company.rsi import start_rsi_daemon, stop_rsi_daemon, rsi_daemon_status

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))

    st_start = start_rsi_daemon(settings=s, interval_s=2.0, use_llm=False, max_cycles=1)
    assert isinstance(st_start, dict)
    # Open-ended frontier cycle can take tens of seconds after a full seed
    for _ in range(120):
        cur = rsi_daemon_status(Ledger(s.ledger))
        if cur['cycles'] >= 1 and not cur['running']:
            break
        time.sleep(0.25)
    st_stop = stop_rsi_daemon(settings=s)
    assert st_stop['running'] is False
    assert st_stop['cycles'] >= 1


@pytest.mark.parametrize('desk,operator', [
    ('perps', 'perps_residual_oi_gate'),
    ('carry', 'carry_funding_hysteresis_v3'),
    ('pred', 'pred_logodds_shrinkage'),
])
def test_desk_native_survival_operators_preserve_causality(desk, operator):
    col = 'px' if desk == 'perps' else ('hl_funding' if desk == 'carry' else 'p')
    grp = 'coin' if desk in ('perps', 'carry') else 'market'
    parent_code = f"""DESK = {desk!r}
PARAMS = {{'w': 12}}
def signal(df, params):
    s = df.select(pl.col({col!r}).cast(pl.Float64).rolling_mean(int(params.get('w', 12)), min_samples=3).over({grp!r}).alias('m'))['m'].to_numpy()
    return np.clip(np.tanh(np.nan_to_num(s, 0.0)), -0.8, 0.8)
"""
    mut = mutate_or_repair_strategy(parent_code, operator=operator, gate_reasons=['top5_days<50%', 'fees_x1.5_profitable'])
    assert mut['operator'] == operator
    mod = load_strategy(mut['code'])
    panel = _synth_panel_for_desk(desk, 360)
    caus = causality_test(mod, panel)
    assert caus['ok'], caus


def test_50_agent_tournament_funnel_effective_trials_and_multi_desk_survival(tmp_path):
    import dataclasses
    from desk.config import SETTINGS
    from desk.company.rsi import (
        run_tournament_funnel,
        compute_effective_trials,
        run_forward_survival_audit,
        compute_treasury_state,
    )

    s = dataclasses.replace(SETTINGS, home=tmp_path)
    L = ensure_seeded(s, Ledger(tmp_path / 'desk.sqlite3'))

    funnel = run_tournament_funnel(
        L,
        settings=s,
        n_agents=50,
        desks=['macro', 'perps', 'carry', 'pred'],
        use_llm=False,
    )
    assert funnel['n_agents'] == 50
    assert funnel['tiers']['tier1_scouts'] == 32
    assert funnel['tiers']['tier2_clusterers_toolsmiths'] == 10
    assert funnel['tiers']['tier3_red_team_officers'] == 8
    assert funnel['clusters_formed'] >= 2
    assert funnel['total_approved'] >= 3

    # Verify multi-desk survival across at least 3 distinct desks
    approved_pods = {r['pod'] for r in L.q("SELECT DISTINCT pod FROM strategies WHERE status IN ('approved', 'paper')")}
    assert len(approved_pods) >= 3, f"Expected >=3 approved desks, got {approved_pods}"

    # Verify Eigenvalue Effective Trials (N_eff <= raw_trials)
    eff = compute_effective_trials(L)
    assert eff['raw_trials'] >= 4
    assert 2 <= eff['effective_trials'] <= eff['raw_trials']

    # Verify Self-Funding Treasury & Capital Scaling ($10k -> $25k -> $100k)
    treasury = compute_treasury_state(L, settings=s)
    assert treasury['capital_tier'] in (2, 3)
    assert treasury['notional_usd'] >= 25_000.0
    assert treasury['self_funded'] is True
    assert treasury['cost_coverage_ratio'] >= 1.0

    # Verify Forward Survival Audit & Automatic Drift Demotion + ERC Rebalance
    victim_id = L.q("SELECT id FROM strategies WHERE status='approved' LIMIT 1")[0]['id']
    fwd = run_forward_survival_audit(L, settings=s, inject_decay_id=victim_id)
    assert victim_id in fwd['demoted_ids']
    assert L.strategy(victim_id)['status'] == 'retired_drift'
    assert fwd['rebuilt_portfolio'] is not None
    assert victim_id not in (fwd['rebuilt_portfolio'].get('weights') or {})


