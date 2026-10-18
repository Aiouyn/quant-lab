# Desk OS v3 — Phase 3: Closed-Loop Truth Machine Gate, Meta-Bandit Operator Selection, ERC Master Portfolio & Continuous RSI Daemon

Upgrade Desk OS v3 from in-sample (`dev_eval`) evolutionary search into a **closed-loop, out-of-sample Recursive Self-Improvement (RSI) system** that gates evolved candidates through the strict Truth Machine (`dev` → sealed `val` with Deflated Sharpe Ratio → one-shot `holdout` → `approved` bundle export), adapts its own mutation/crossover operators via a **Meta-Bandit weighted by out-of-sample `val` survival**, constructs an **Equal-Risk-Contribution (ERC) Master Portfolio** from approved strategies across desks, and supports both **continuous background daemon iteration** and **on-demand multi-generation campaigns**.

---

## 1. Architectural Overview & Closed-Loop RSI Dataflow

```
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │ 1. META-BANDIT OPERATOR SELECTOR (UCB1 / Thompson Sampling over Val Survival)     │
  │    Scores operators (llm_mutation, llm_crossover, vol_target_overlay,             │
  │    regime_filter_overlay, param_perturb, crossover) by empirical val pass rate    │
  │    & out-of-sample Val Sharpe lift per desk.                                      │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            ▼
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │ 2. EVOLUTIONARY BREEDING (`evolve_population` via `deepseek-v4.1-flash-free`)     │
  │    Spawns candidate children using Bandit-selected operators + high-lift          │
  │    `desk.lib.contrib` primitives; scores initial fitness on DEV panel.            │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            ▼
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │ 3. STRICT TRUTH MACHINE GATE (`validate_and_promote_population`)                  │
  │    • Gate A (Full DEV): desk-calibrated Sharpe bar, causality check,              │
  │      1.5x stressed fees > 0, both halves Sharpe > 0, positive months >= 50%,      │
  │      mean ex-top-5 days > 0.                                                      │
  │    • Gate B (Sealed VAL): desk-calibrated Val Sharpe bar, Deflated Sharpe Ratio   │
  │      (DSR >= min_dsr across all val trials), bootstrap lower bound > 0,           │
  │      1.5x stressed cost > 0, max drawdown & top-5 day concentration caps.         │
  │    • Gate C (One-Shot HOLDOUT): enforced via `holdout_uses` code_hash seal;       │
  │      requires holdout PnL > 0, Sharpe >= min_holdout_sharpe, drawdown <= cap.     │
  │    • Promotion: exports immutable strategy bundle (`results/v3/bundles/<sid>`)    │
  │      and transitions status -> `approved`.                                        │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            ▼
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │ 4. OUT-OF-SAMPLE FEEDBACK & ERC MASTER PORTFOLIO (`build_portfolio`)              │
  │    • Feeds sealed Val/Holdout pass/fail outcomes back into the Meta-Bandit and    │
  │      `primitive_usages` (penalizing primitives that overfit DEV but fail VAL).    │
  │    • Aligns out-of-sample (`val` + `holdout`) daily return streams across         │
  │      all `approved` strategies, prunes correlated duplicates (|corr| >= 0.85),    │
  │      fits Ledoit-Wolf shrunk covariance, and solves for Equal-Risk-Contribution   │
  │      (ERC) weights capped at 35% per strategy.                                    │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Planned Component Changes

### A. Strict Truth Machine Gate & Multi-Split Evaluator (`desk/company/rsi.py` & `desk/pipeline/validation.py`)
- **Multi-Split Desk Evaluation (`_evaluate_strategy_on_split(code, desk, split, settings, n_trials, stress=True)`)**:
  - Evaluates strategy code on `dev`, `val`, or `holdout` splits using the authentic desk stores (`MacroStore`, `PerpsStore`, `CarryStore`, `PredStore`) with strict split isolation (`allowed=('dev',)` for `dev`, `allowed=('dev', 'val')` for `val`, `allowed=('dev', 'val', 'holdout')` for `holdout`).
  - Computes full statistical and robustness diagnostics required by `DEV_GATES`, `val_gates(settings)`, and `holdout_gates(settings)`:
    - Causality check (prefix truncation invariance on signal weights).
    - Stressed execution simulation (`1.5x` transaction costs / half-spreads and `+1` bar delay).
    - Sub-period stability (`first_half_sharpe`, `second_half_sharpe`, `positive_months`, `positive_weeks`, `mean_ex_top5_days_bps`, `top5_day_share`).
    - **Deflated Sharpe Ratio (`dsr`)** and stationary bootstrap lower confidence bound (`bootstrap_lower_mean_bps`) using `desk.lab.stats.deflated_sharpe` and `stationary_bootstrap_mean` scaled by total validation trials `n_val`.
    - Daily return series (`daily: {'day': [...], 'ret': [...]}`) on `val` and `holdout` splits for downstream covariance and ERC portfolio allocation.
- **`validate_and_promote_population(ledger, settings, desk=None, top_k=4, author='rsi-gate')`**:
  - Selects top unvalidated `dev_eval` / `submitted` strategies per desk.
  - Runs **Gate A (`DEV_GATES`)** → transitions passing candidates to `validating` (or `rejected_dev` with specific `gate_reasons`).
  - Runs **Gate B (`val_gates(settings)`)** with desk-calibrated Sharpe bars (`SHARPE_BARS` for daily macro vs intraday/hourly crypto/prediction desks) and `deflated_sharpe >= settings.min_dsr` → transitions passing candidates to `holdout` (or `rejected_val`).
  - Runs **Gate C (`holdout_gates(settings)`)** guarded by the `holdout_uses` table (one-shot seal per `code_hash`) → promotes passing strategies to `approved` and calls `export_bundle(settings, strategy, cost)`.
  - Feeds `val` pass/fail outcomes back into `primitive_usages` (`kind='val_gate'`) so primitives that overfit `dev` but fail `val` receive negative out-of-sample lift.

### B. Meta-Bandit Operator Selection Weighted by `Val` Survival (`desk/company/rsi.py`)
- **`operator_bandit_leaderboard(ledger, desk=None)`**:
  - Joins `lineages` (`child_id`, `operator`, `desk`, `delta_score`) with `strategies` (`status`, `dev_report`, `val_report`, `holdout_report`).
  - Computes per-operator metrics across all 6 operators (`llm_mutation`, `llm_crossover`, `vol_target_overlay`, `regime_filter_overlay`, `param_perturb`, `crossover`):
    - `spawns`: total child strategies bred by the operator.
    - `dev_pass_rate`: fraction passing Gate A.
    - `val_survival_rate`: fraction of evaluated children passing sealed Gate B (`val` + DSR).
    - `approved_count`: number of children reaching `approved` status.
    - `mean_val_sharpe`: mean out-of-sample Sharpe on `val`.
    - `bandit_score` & `selection_prob`: UCB1 / Beta-Bernoulli posterior score combining out-of-sample `val_survival_rate` (primary weight `0.65`), `mean_val_sharpe` (`0.25`), and `dev_pass_rate` (`0.10`) plus an exploration bonus $\sqrt{\frac{2 \ln N}{n_i + 1}}$.
- **Bandit-Driven `evolve_population`**:
  - Replaces static operator ordering with bandit-selected operators ranked by `bandit_score` for the target desk.

### C. Automated Equal-Risk-Contribution (ERC) Master Portfolio (`desk/pipeline/portfolio.py` & `desk/company/rsi.py`)
- **Seamless Portfolio Construction after Validation**:
  - Whenever `validate_and_promote_population` produces `approved` strategies (or updates the approved pool), `run_rsi_campaign` automatically invokes `build_portfolio(settings, ledger=ledger, author='rsi-allocator')`.
  - Aligns out-of-sample daily return streams (`val_report['daily']` + `holdout_report['daily']`), filters out strategies with pairwise correlation $\ge 0.85$, estimates shrunk covariance via `LedoitWolf` (with diagonal shrinkage fallback), and solves for Equal-Risk-Contribution (ERC) weights capped at `35%` per strategy.
  - Exposes the latest Master Portfolio (`weights`, `sharpe`, `mean_daily_bps`, `max_drawdown`, `avg_pair_corr`, desk breakdown, and member strategy details) inside `rsi_snapshot()`.

### D. Continuous Background RSI Daemon (`desk/company/rsi.py`, `desk/web/server.py`, `desk/cli.py`)
- **`RsiDaemon` Background Controller**:
  - Singleton thread-safe background daemon (`start_rsi_daemon`, `stop_rsi_daemon`, `rsi_daemon_status`) that runs continuous closed-loop RSI generations at a configurable interval (default `12s` between generations) without blocking the HTTP server.
  - Each daemon tick executes one complete closed-loop generation across all desks:
    1. `compile_error_hints`
    2. `evolve_primitives` (Toolsmith refactor/synthesis)
    3. Bandit-weighted `evolve_population`
    4. `validate_and_promote_population` (Truth Machine `dev` → `val` + DSR → `holdout` → `approved`)
    5. `build_portfolio` (ERC Master Portfolio update)
    6. `propose_charter_patch` (sealed drill gate)
  - Can be started/stopped live from the Control Room UI (`start_daemon` / `stop_daemon` actions) or CLI (`python -m desk rsi daemon --generations 10`).

### E. Control Room UI Enhancements (`desk/web/index.html`)
- **Campaign & Continuous Daemon Control Bar**:
  - Add a **Continuous RSI Daemon Toggle (`Start Daemon` / `Stop Daemon`)** with live status indicator (`IDLE` / `RUNNING · Gen #N`), plus a **Run Truth Machine Gate & ERC Portfolio** button.
- **Dual In-Sample (`Dev`) vs Out-of-Sample (`Val` / `Portfolio`) Fitness Trajectory**:
  - Upgrade Section `08` SVG trajectory chart and generation table to plot both **Best Dev Sharpe** and **Best Out-of-Sample Val / ERC Portfolio Sharpe**, plus per-generation counts of `Val Passed`, `Approved`, and active top operator.
- **Meta-Bandit Operator Leaderboard & ERC Master Portfolio Book**:
  - Display the live **Meta-Bandit Operator Table** (`Operator`, `Spawns`, `Dev Pass %`, `Val Survival %`, `Mean Val SR`, `Approved`, `Bandit Weight %`) and the **ERC Master Portfolio Allocation Table** (`Strategy ID`, `Desk`, `Operator`, `Val SR`, `Holdout SR`, `ERC Weight %`).

---

## 3. Verification & Test Plan (`tests/v3/test_rsi.py`)
1. **`test_truth_machine_gate_and_promotion`**: Verify `validate_and_promote_population` runs `DEV_GATES`, `val_gates` (including DSR calculation), and `holdout_gates` (with `holdout_uses` one-shot seal), promoting valid out-of-sample strategies to `approved` and exporting bundles.
2. **`test_meta_bandit_operator_selection`**: Verify `operator_bandit_leaderboard` computes out-of-sample `val_survival_rate` and dynamically shifts selection probability toward operators whose children pass `val` gates.
3. **`test_erc_master_portfolio_construction`**: Verify `build_portfolio` constructs a covariance-shrunk ERC portfolio from `approved` strategies and surfaces it in `rsi_snapshot()['latest_portfolio']`.
4. **`test_closed_loop_campaign_and_daemon`**: Verify `run_rsi_campaign` executes the full end-to-end loop (`evolve` → `validate` → `portfolio`) and that `start_rsi_daemon` / `stop_rsi_daemon` cleanly run background iterations.
