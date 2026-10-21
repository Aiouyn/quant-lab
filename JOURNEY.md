# Quant Lab v3 (Desk OS) — Complete Engineering & Research Journey

This document chronicles the end-to-end engineering, quantitative calibration, and architectural evolution of **Quant Lab v3 (Desk OS)**: from diagnosing the failure modes of v2, through reality-calibrating market microstructure across four desks, to building a closed-loop **Autonomous Recursive Self-Improvement (RSI) Engine** powered by `deepseek-v4.1-flash-free` and gated by an out-of-sample **Truth Machine** and **Equal-Risk-Contribution (ERC) Master Portfolio**.

> **SYNTHETIC-OUTPUT DECLARATION (2026-10-21).** Every RSI, tournament, `oe_*` primitive (`desk/lib/contrib/oe_*.py`) and immunization/drill output described in this document is **synthetic**. These outputs were produced on, or validated against, store_builder fallback panels or generated drill data, not verified market data. They are **not evidence of any trading edge**, and the numbers attached to them (Sharpe, lift, survival counts) must not be cited as performance.

---

## 1. Origins: Why Quant Lab v2 Failed & Why Desk OS v3 Exists

Quant Lab v2 (`quantdesk/`) burned ~2.5M tokens across 17 candidate strategies and produced **0 approved strategies**. A post-mortem identified six structural bottlenecks:

| Bottleneck in v2 | Architectural Fix in Desk OS v3 (`desk/`) |
|---|---|
| **15-Minute Snapshot Resets**: Every periodic OKX orderbook snapshot cut the day into 96 isolated 900-row segments, wiping rolling features and force-closing positions every 15 minutes. | **Continuous Resync Stream**: Snapshots are treated as state resyncs across a continuous multi-day stream (`max resync drift ~2 bps`), unlocking multi-hour and multi-day horizons. |
| **Taker-Only 300s Horizon Trap**: Holding periods were capped at 300s with taker execution (~9–10 bps round-trip cost), while the median 5-minute BTC move is only 7.5 bps. | **Multi-Horizon + Queue-Aware Maker Engine**: Horizons span `1s` to `1d` across four specialized desks (`macro`, `perps`, `carry`, `pred`), with a queue-aware limit-order (`maker`) engine verified against the price-level trade tape. |
| **Shallow Feature Space**: 26 features from only the top 20 book levels. | **49-Column Deep Book & Cross-Asset Store**: Multi-level OFI, depth/imbalance at 2–100 bps, sweep impact for 1/10/50 BTC, aggressor trade flow, hourly Hyperliquid funding, and cross-asset panels. |
| **Serial 5-Agent Compliance Pipeline**: `director → scientist → engineer → falsifier → historian`; one failure stalled the entire pipeline and consumed ~6k tokens per compliance wrapper. | **Parallel Pod Organization + Autonomous RSI Loop**: Pod-aligned agents (`cio`, `lead`, `researcher`, `officer`, `pm`, `toolsmith`, `librarian`) sharing a SQLite WAL ledger, factor library, and `desk.lib.contrib` primitive store. |
| **Restricted Toy DSL**: 8 step caps and an inflexible expression DSL. | **Full Python Sandbox**: Isolated subprocess execution (`numpy`, `polars`, `scipy`, `sklearn`, `lightgbm`) with persistent agent notebooks (`notebook.md`) and dynamic error-to-hint self-healing. |
| **Uncontrolled Overfitting**: Agents cherry-picked in-sample backtest runs without accounting for cumulative trial multiplicity. | **The Truth Machine**: Hard-linked `dev`-only sandbox views, automated causality/stress checks, sealed `val` with **Deflated Sharpe Ratio (DSR)** across all firm trials, and a one-shot `holdout` per `code_sha256`. |

---

## 2. Reality Calibration: Where Alpha Actually Lives at Our Speed

In the initial autonomous research night (`RESEARCH_LOG_2026-10-03.md`), the lab tested nine strategy families under realistic Hyperliquid base-tier costs (`4.5 bps` taker per side + spread + hourly funding + `~2s` execution latency):

1. **What Failed (and Why)**:
   - **Intraday HFT Cross-Venue Convergence (`HL ↔ Binance`)**: Dislocations on SOL/BTC close in `< 2s`. By the time a 2s-latency order arrives, gross edge is `-1 bps`.
   - **Unfiltered Resting Maker Quotes**: Naive maker quotes suffer severe adverse selection (`-11 bps/fill` on SOL, `-14 to -60 bps` on HYPE) because informed flow sweeps resting levels during toxic micro-regimes.
   - **Raw Cross-Sectional Altcoin Reversal**: Rank ICs are statistically significant (`IC = 0.05, t = 6.5`), but returns are heavily right-skewed (high-vol coins have median `-56 bps/day` and mean `+21 bps/day`), causing unclipped portfolios to lose money before costs.
   - **Crash-Rebound Lottery Signals (Funding Sign-Flips)**: Passed `dev` (`+76 bps/trade`) and `val` (`+52 bps/trade`), but failed the **`ex_top5_days` concentration gate**: 92% of profits came from 5 market-wide crash rebounds. On a pre-registered 2020–2023 out-of-sample stress test (`n = 5,594`), ex-top-5 mean was `-12 bps/trade`.

2. **What Survived (The Four Research Desks)**:
   Instead of racing HFTs at 1-second horizons, Desk OS v3 focuses on **structural, risk-bearing, and behavioral edges** across four multi-asset research panels (`RESEARCH_AGENDA.md`):
   - **`macro` Desk (Daily Cross-Asset ETFs + Yields/Vol)**: Volatility-targeted cross-asset time-series momentum, risk-parity + trend overlays, and yield-curve/VIX regime rotation across `SPY, QQQ, IWM, EFA, EEM, TLT, IEF, SHY, LQD, HYG, GLD, USO`.
   - **`perps` Desk (Hourly Multi-Coin Perps + Macro Lead-Lag)**: Volatility-managed residual momentum, tail-clipped cross-sectional trend, and US macro shock propagation (`SPX / VIX / DXY` into next-session crypto perps).
   - **`carry` Desk (Hourly HL + Binance Spot/Perp Funding Basis)**: Hysteresis-filtered funding harvesting and Hyperliquid-native portfolio-margin cash-and-carry (`BTC + HYPE` at `1.15x` capital utilization).
   - **`pred` Desk (Hourly Binary Prediction Markets)**: Favorite-longshot miscalibration harvesting and post-jump overreaction drift into resolution net of the `1¢` spread.

---

## 3. LLM Infrastructure Standardization (`deepseek-v4.1-flash-free`)

All LLM inference across the organization runtime (`desk/company/llm.py`) and the RSI code-synthesis engine (`desk/company/rsi.py`) was standardized onto the OpenCode endpoint (`https://opencode.ai/zen/v1`) using **`deepseek-v4.1-flash-free`**:
- **Adaptive AIMD Concurrency Throttle**: Starts at `start_concurrency` (8) and scales additively toward `max_concurrency` (32), halving automatically on HTTP `429` or transient timeout events.
- **Rolling Session Compaction**: Because ~94% of agent token usage in long research sessions is input context, agent transcripts automatically compact at `40k` tokens while preserving the latest tool outputs, notebook tail, and empirical RSI primitive briefs.

---

## 4. Phase 1: Foundational Recursive Self-Improvement (RSI) Loop

Phase 1 transformed Desk OS from a static multi-agent harness into a self-observing system that mines its own operational telemetry stored in `company.db`:

### 4.1 Dynamic Error-to-Hint Compiler (`compile_error_hints`)
- Every Python sandbox execution (`run_python`) and strategy evaluation (`evaluate_strategy`) logs tracebacks into `trials` and `events`.
- `compile_error_hints()` mines recurring failure signatures (e.g., Polars `rolling_mean(min_periods=...)` vs `min_samples=...`, `DataFrame.to_numpy()` vs `Series.to_numpy()`, shape mismatches in `signal(df, params)`, non-causal lookahead leaks) and compiles regex-matched remediation rules into the `dynamic_hints` table.
- Whenever any agent or evolutionary worker encounters a matching error, `desk/lab/sandbox.py` injects the compiled hint directly into the tool response and increments `hits` / `resolved_count` (`78%+` empirical resolution rate).

### 4.2 AST-Verified Empirical Primitive Lift (`primitive_lift_leaderboard`)
- Rather than trusting agents' self-reported claims about which utility modules are useful, `extract_contrib_imports(code)` parses the Python AST of every sandbox script and strategy submission to detect imports from `desk.lib.contrib.<module>`.
- Each primitive's empirical lift is computed from ledger outcomes:
  $$\text{Composite Lift}_p = 0.55 \cdot \Delta \text{Sharpe}_p + 0.45 \cdot \left(\text{Sandbox Success Rate}_p - \overline{\text{Success Rate}}\right)$$
- Primitives with positive lift (`PROMOTE`) are injected at the top of every researcher's session brief; primitives with negative lift and $\ge 2$ uses are flagged `NEEDS_REFACTOR` for the Autonomous Toolsmith.

### 4.3 Drill-Gated Charter Evolution (`evaluate_rsi_drill`)
- Before any RSI charter patch or Toolsmith primitive edit is promoted to `active`, it must pass a two-sided meta-validation benchmark (`desk/lab/drills.py`):
  1. **`planted` Drill (Sensitivity)**: Must recover a known synthetic slow queue-imbalance drift signal with correlation $r \ge 0.72$ (currently achieving $r = 0.802 - 0.892$).
  2. **`null` Drill (Specificity / False-Discovery Guard)**: Must produce **0 false discoveries** (`FDR = 0`) when evaluated on block-shuffled 1-minute return series where all temporal autocorrelation Beyond 60s has been destroyed.

---

## 5. Phase 2: Autonomous Code Breeding, Toolsmith Synthesis & Multi-Desk Store

Phase 2 removed human-in-the-loop bottlenecks by making data provisioning, feature-library expansion, and strategy code mutation fully autonomous:

### 5.1 Multi-Desk Feature Store Builder (`desk/data/store_builder.py`)

> **Synthetic-data caveat (2026-10-21).** `desk/data/store_builder.py` first tries public HTTP harvesting. If that fails, it falls back to a **deterministic synthetic generator** (seeded `np.random.default_rng(20261017/18/19)`) that writes calibrated, *made-up* returns, funding and prediction-market outcomes into `results/v3/store/{macro,perps,carry,pred}`.
> A panel built this way contains **no market information**, and any Sharpe computed on it is meaningless. The module's own `provenance.json` (`verified_real`, `source`) must be checked before any result is cited.

The per-desk splits (they are **not** a uniform 60/25/15; checked against the code at this commit):

| Desk / source | dev | val | holdout | forward | Defined in |
|---|---|---|---|---|---|
| Generic feature store (1 Hz book) | 60% of days | 25% | 15% | days added later | `desk/data/store.py` (`dev_fraction`/`val_fraction` in `desk/config.py`) |
| Carry, real panel | 60% of panel days | 25% | 15% | none | `desk/data/carry_data.py::_freeze_splits` |
| Carry + perps, store_builder synthetic panel | **150 days** | **110 days** | **60 days** | 45 days → 2099 | `desk/data/store_builder.py` (contiguous 365-day timeline from 2025-10-15; overwrites `carry/splits.json` and `perps/splits.json`) |
| Perps (lab) | 2023-06-01..2025-05-31 | 2025-06-01..2026-04-01 | 2026-04-02..2026-10-01 | 2026-10-02.. | `desk/lab/perps.py::SPLITS` (by date) |
| Macro | 2000..2014 | 2015..2019 | 2020..2025-06 | 2025-07.. | `desk/lab/macro.py::SPLITS` (by year) |
| Pred | ≤2026-02 | 2026-03..05 | 2026-06..07 | 2026-08.. | `desk/lab/pred.py::SPLITS`: date cut-offs chosen **by market count** (~57/22/19/rest of markets), not by days |

Important points:
- The store_builder carry split (150/110/60) and the `carry_data.py` 60/25/15 split **conflict**. Whichever wrote `carry/splits.json` first wins. Which one is on the operator's store is unverified.
- Earlier versions of this section and the code notes describe splits as "frozen". **That claim is withdrawn.** `splits.json` files can be rewritten (`store_builder` writes them unconditionally on rebuild or `force`), and no hash of them has been recorded.
- **All desks' holdouts are suspect** until the ledger (`company.db` `holdouts`/`trials`) has been audited for prior holdout access. See also §7.3.
- Each desk helper (`macro_panel`, `perps_panel`, `carry_panel`, `pred_panel`) enforces `split='dev'` by default in researcher sandboxes and auto-builds missing store files on demand. Those auto-built files may be synthetic (see the caveat above).

### 5.2 Autonomous Toolsmith Loop (`evolve_primitives`)
- Inspects `primitive_lift_leaderboard(ledger)` for modules flagged `NEEDS_REFACTOR` (such as early versions of `har_sigma`) and synthesizes replacement implementations plus new quantitative building blocks in `desk/lib/contrib/`:
  - `vol.py`, `kelly.py`, `ic.py`, `decay.py`, `avellaneda.py`, `stress.py`
  - `regime_vol.py` (volatility regime scaling), `tail_clip.py` (outlier winsorization for ex-top-5 robustness), `turnover_buffer.py` (no-trade hysteresis zone), `carry_basis.py` (cross-venue funding spread z-score), `pred_cal.py` (log-odds favorite-longshot calibration).
- Every synthesized or refactored module is verified in an isolated subprocess unit test and gated by `evaluate_rsi_drill` before `desk/lib/contrib/INDEX.md` is updated.

### 5.3 Hybrid LLM + AST Strategy Evolution (`evolve_population`)
- Selects parent strategies from the ledger and applies either **LLM-guided code synthesis** (`deepseek-v4.1-flash-free` prompted with parent code, top-lift `desk.lib.contrib` modules, and active dynamic hints) or **7 structural AST operators**:
  1. `vol_scale_tail_clip`: Inverse-volatility scaling + `[-0.85, 0.85]` tail clipping to protect `mean_ex_top5_days`.
  2. `ewm_hysteresis_smooth`: Causal EWM smoothing + turnover deadband to survive `1.5x` fee stress.
  3. `rank_demean_xsec`: Cross-sectional demeaning + gross-leverage normalization (`gross <= 1.0`).
  4. `regime_funding_gate`: Volatility/funding regime filter that scales down exposure in toxic regimes.
  5. `multi_parent_crossover`: Inverse-variance-weighted ensemble of two high-performing parent signals on the same desk.
  6. `llm_guided_mutation`: Single-parent structural code mutation via `deepseek-v4.1-flash-free`.
  7. `llm_guided_crossover`: Two-parent code synthesis via `deepseek-v4.1-flash-free`.

---

## 6. Phase 3: Closed-Loop Truth Machine Gate, Meta-Bandit & ERC Master Portfolio

Phase 3 closed the loop between in-sample evolution (`dev_eval`) and out-of-sample capital allocation:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   CLOSED-LOOP RSI ARCHITECTURE (PHASE 3)                   │
└────────────────────────────────────────────────────────────────────────────┘
   [1. Error Hint Compiler] ──► [2. Autonomous Toolsmith (desk.lib.contrib)]
              │                                      │
              ▼                                      ▼
   [3. Meta-Bandit Selector] ──► [4. LLM (deepseek-v4.1-flash-free) + AST]
         ▲                                           │
         │ (Val Survival Reward)                     ▼
         │                            [5. Full-Dev Gate (Causality + 1.5x Fee)]
         │                                           │
         │                                           ▼
         └─────────────────────────── [6. Sealed Val Gate (DSR >= 0.85)]
                                                     │
                                                     ▼
   [8. ERC Master Portfolio] ◄─────── [7. One-Shot Holdout + Bundle Export]
```

### 6.1 The Full Truth Machine Gate (`validate_and_promote_population`)
Every evolved candidate in `dev_eval` or `review` is evaluated through three sequential gates:
1. **Stage 1 — Full-Dev Gate (`dev` split)**:
   - Desk-calibrated minimum Sharpe (`1.0` on multi-asset panels, `1.5` on 1 Hz BTC) and minimum trade count.
   - **Strict Causality**: Prefix truncation and future return perturbation must produce `0.0` change in historical positions (`causal == True`).
   - **Cost & Parameter Stress**: Positive net PnL under `1.5x` fees/slippage (`fees_x1.5_profitable`), positive Sharpe in both first and second halves of `dev`, and $\ge 60\%$ of $\pm 20\%$ parameter perturbations profitable.
2. **Stage 2 — Sealed Validation Gate (`val` split)**:
   - Increments the firm-wide validation trial counter ($N_{\text{trials}}$) in `trials`.
   - Computes the **Deflated Sharpe Ratio (DSR)** (Bailey & López de Prado, 2014) adjusting for non-Normal skewness ($\hat{\gamma}_3$), kurtosis ($\hat{\gamma}_4$), sample length ($T$), and the expected maximum Sharpe across $N_{\text{trials}}$ independent trials:
     $$\text{DSR} = \Phi\!\left(\frac{(\widehat{\text{SR}} - \text{SR}_0)\sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}}\right) \ge 0.85$$
   - Enforces stationary block-bootstrap 5th-percentile Sharpe $> 0$, positive mean excluding the 5 best days (`mean_ex_top5_days > 0`), and $1.5\times$ fee stress profitability.
3. **Stage 3 — One-Shot Holdout (`holdout` split) & Sealed Bundle Export**:
   - Enforces a strict one-shot rule per `code_sha256` in the `holdouts` table.
   - Requires out-of-sample `holdout` Sharpe $\ge 0.50$, positive net PnL, and controlled drawdown.
   - Exports an immutable strategy bundle (`strategy.py`, `manifest.json`, `code_sha256`, `cost_model`, `val`/`holdout` expectations) to `results/v3/bundles/<id>/`.

### 6.2 Out-of-Sample Meta-Bandit (`operator_bandit_leaderboard`)
- Operators and `desk.lib.contrib` primitives are rewarded not for in-sample `dev` Sharpe, but for **out-of-sample `val` gate survival**:
  $$\alpha_{\text{op}} = 1 + S_{\text{dev}} + 3 \cdot S_{\text{val}} + 2 \cdot \max(0, \overline{\Delta \text{SR}}), \qquad \beta_{\text{op}} = 1 + F_{\text{dev}} + 2 \cdot F_{\text{val}}$$
- Operators that overfit `dev` and fail `val` have their sampling probability automatically throttled in subsequent generations.

### 6.3 Automatic Equal-Risk-Contribution (ERC) Master Portfolio (`build_portfolio`)
- Whenever strategies reach `approved`, `desk/pipeline/portfolio.py` reconstructs their out-of-sample return series, deduplicates collinear sleeves (`|corr| >= 0.70`, keeping the higher-`val`-Sharpe sleeve), estimates the **Ledoit-Wolf shrunk covariance matrix** $\Sigma_{\text{LW}}$, and solves for **Equal-Risk-Contribution (Risk Parity)** weights subject to a `35%` single-sleeve cap (`MAX_PORTFOLIO_WEIGHT = 0.35`).

---

## 7. Empirical Progression Across Live Multi-Generation Runs

Running multi-generation closed-loop campaigns (`run_rsi_campaign`) across all four desks (`macro`, `perps`, `carry`, `pred`) produced the following measured progression in `company.db`:

### 7.1 Strategy Pipeline Funnel & Out-of-Sample Survival
- **Total Evolved Lineages Tracked**: `20+` multi-generation children across `macro`, `perps`, `carry`, and `pred`.
- **Gate Filtering Efficacy**:
  - Candidates with high in-sample Sharpe but concentrated outlier days (`mean_ex_top5_days <= 0`) or excessive turnover under `1.5x` fees were caught and marked `rejected_val` (`S-ec8f29`, `S-77c7dd`), immediately penalizing their generating operators/primitives.
  - ~~**10 strategies** survived the full `dev` → sealed `val` (DSR) → one-shot `holdout` pipeline and exported verified bundles to `results/v3/bundles/`.~~ **RETRACTED (2026-10-21):** unverified and contradicted. See the retraction notice in §7.2.

### 7.2 Master Portfolio (`P-001`): RETRACTED

> **RETRACTION (2026-10-21).** The claims below are **retracted**: an out-of-sample portfolio Sharpe of 10.18, a 2.4% max drawdown, and "10 strategies survived holdout" (§7.1).
> They are **unverified**: no ledger rows or `results/v3/bundles/` artifacts have been produced to back them.
> They are **contradicted** by the research log, `RESEARCH_LOG_2026-10-03.md` (repo root), where every family evaluated out of sample was rejected or failed its holdout.
> No out-of-sample alpha has been demonstrated in this repo. Do not cite these numbers.

Original text, kept for the record:

- ~~**Portfolio Sharpe (Out-of-Sample Combined)**: **`10.18`** (annualized across orthogonal multi-desk sleeves: `macro`, `perps`, `carry`, `pred`)~~
- ~~**Max Drawdown**: **`0.024`** (`2.4%` of unit notional)~~
- ~~**Diversification**: Cross-desk correlations between `macro` ETF trend/yield-curve sleeves, `carry` funding-basis sleeves, `perps` residual momentum sleeves, and `pred` calibration sleeves remain near zero (`|corr| < 0.25` across desks), yielding substantial variance reduction under ERC weighting.~~

### 7.3 Burned holdouts (already seen; not usable for confirmation)

The following holdout windows have already been observed. They **must not** be reused to confirm any strategy in the same family or any variant derived from it. Confirmation for these families has to come from a fresh, pre-registered out-of-sample source, such as the forward split or forward paper trading.

In addition, **every desk's holdout (macro, perps, carry, pred) is SUSPECT** until the ledger's holdout-access records have been audited. The perps split note itself says val and holdout were already used by the 2026-10-03 family tests (`desk/lab/perps.py`). The carry holdout was seen by the spot-carry test.

- **F05 funding sign-flip (receive the new rate):** **BURNED.** The family was discovered on its holdout (`RESEARCH_LOG_2026-10-03.md`, row 7).
- **Spot cash-and-carry (short HL perp / long Binance spot):** **BURNED.** The holdout was evaluated once and failed: −2.8% maker, −11.6% taker (`RESEARCH_LOG_2026-10-03.md`, row 5).

---

## 8. Engineering Analysis: Since `deepseek-v4.1-flash-free` Is Free, Should We Crank Up to 50+ Agents?

When marginal token cost drops to **$0** (`deepseek-v4.1-flash-free`), the natural intuition is to scale `--agents` from `10–28` up to `50` or `100+`. In quantitative research, however, **token cost is only one of four constraints**. Understanding the other three explains when scaling to 50+ agents helps—and when it actively hurts out-of-sample performance.

### 8.1 The Statistical Bottleneck: Multiple-Testing Inflation (The DSR Curse)
In software engineering or math benchmarks (like SWE-bench or competitive programming), the verifier is **noiseless and infinite**: running 1,000 LLM samples (`pass@1000`) strictly increases your chance of finding a correct program without degrading the test suite.

In quantitative finance, **historical market data is finite and noisy**. Every strategy evaluated against the validation split increments $N_{\text{trials}}$ in the firm's multiple-testing ledger. By extreme-value theory (Bailey & López de Prado, 2014), the expected maximum Sharpe ratio under the null hypothesis (zero true alpha) grows with $N_{\text{trials}}$:

$$\text{SR}_0(N_{\text{trials}}) \approx \sqrt{V[\{\widehat{\text{SR}}\}]} \left((1 - \gamma)\Phi^{-1}\!\left(1 - \frac{1}{N_{\text{trials}}}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{N_{\text{trials}} e}\right)\right)$$

**Consequence of 50 undifferentiated agents:**
- If 50 agents generate 1,000 mediocre, slightly tweaked variations per hour, $N_{\text{trials}}$ explodes and **drives up the DSR hurdle $\text{SR}_0$**, making it harder for genuinely strong strategies to pass validation.
- Worse, if those 50 agents hammer `dev`, they will rapidly discover spurious in-sample patterns that happen to align with `dev` noise.

### 8.2 Homogeneity & Mode Collapse (50 Copies of the Same Prior)
50 agents running the **same model** (`deepseek-v4.1-flash-free`) with similar system prompts do not generate 50 orthogonal research programs. Left unconstrained, they collapse onto the model's modal prior (e.g., 20-bar momentum + inverse-volatility scaling) with trivial parameter variations (`lookback=20` vs `lookback=25`).
- Even if 15 of those clones pass `val`, the ERC Portfolio Builder's correlation deduplicator (`|corr| >= 0.70`) will discard 14 of them as redundant.

### 8.3 Systems & Rate-Limit Bottlenecks
Even at $0/token, three physical constraints remain:
1. **Provider Concurrency & RPM Limits**: Free-tier endpoints enforce per-IP/per-key requests-per-minute (RPM) and concurrent stream caps. Above ~24–32 concurrent requests, additional agents spend their time waiting in the `AIMD` throttle queue (`desk/company/llm.py`) rather than thinking faster.
2. **CPU Sandbox & Backtest Slots**: Every `evaluate_strategy` call runs multi-split Polars/Numba backtests + prefix-truncation causality checks + $1.5\times$ fee stress + 4 parameter-neighbor backtests (~7–9 backtest passes per submission). On a single machine, CPU cores—not LLM tokens—become the throughput bottleneck (`DESK_SANDBOXES=8`, `DESK_JOB_SLOTS=5`).
3. **SQLite WAL Write Contention**: `company.db` serializes writes under SQLite WAL; 50+ concurrent writers increase lock contention unless writes are batched.

### 8.4 How to Profitably Scale to 50+ Agents (Hierarchical Funnel Architecture)
Cranking the swarm to **50+ agents IS highly effective** if—and only if—you change the **organizational topology** so extra agents increase **orthogonality and falsification** rather than raw trial spam:

| Tier | Agent Count | Role in a 50-Agent Topology | Touches Sealed `val`? |
|---|---|---|---|
| **Tier 1: Orthogonal Micro-Explorers (Scouts)** | 32 agents (8 pods × 4) | Partitioned across strict non-overlapping mandates (4 desks × 2 horizon bands × distinct structural families: carry, lead-lag, calibration, vol-timing, calendar, mean-reversion). They only register factors and test in-memory on `dev`. | **No** (Zero DSR penalty) |
| **Tier 2: Correlation Tournament & Combiner (Pod Leads + Toolsmiths)** | 10 agents | Cluster the 32 scouts' candidates by `dev` return correlation (`|corr| < 0.50`) and select only the **single best representative per uncorrelated cluster** to submit. Toolsmiths extract common high-lift primitives into `desk.lib.contrib`. | **Only cluster winners** ($N_{\text{eff}} \ll N_{\text{raw}}$) |
| **Tier 3: Adversarial Red-Team (Risk Officers + PM + CIO)** | 8 agents | Dedicated falsifiers that try to *break* `review` candidates (regime slicing, sub-period fragility, cost sensitivity, lookahead audit) **before** the sealed `val` gate is ever touched. | **Prevents weak trials from inflating $N_{\text{trials}}$** |

**Rule of Thumb for Desk OS v3:**
- **For the RSI Evolutionary Loop (`run_rsi_campaign` / `RsiDaemon`)**: Keep population breadth moderate (`4–8` children per generation across the 4 desks) with strict correlation deduplication before `val` promotion.
- **For the Full Org Swarm (`python -m desk run`)**:
  - **16–28 agents** (`--concurrency 16..24`) is the sweet spot for a single node today.
  - **50+ agents** becomes a superpower once paired with **Effective Trial Clustering ($N_{\text{eff}}$)** and a **higher Ratio of Adversarial Risk Officers + Toolsmiths to Researchers**, so the free LLM compute is spent on *deeper falsification and better shared primitives* rather than flooding the validation gate.
