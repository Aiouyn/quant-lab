# Quant Lab v3 — Desk OS

An autonomous quantitative research firm made of LLM agents. It studies 332 days of 1 Hz full-depth OKX
BTC-USDT-SWAP order books, plus every OKX trade print and hourly Hyperliquid funding. It turns what it learns into
strategies that are paper-traded on Hyperliquid and combined into portfolios.

The v1/v2 code (`quantdesk/`, `lab_runner.py`, the root `config.py`), its swarm, reports and tests were removed from
the tree on 2026-11-02; they remain in git history. v2 produced no approved strategy.

## Current status (2026-11-02)

- **No trading edge has been demonstrated out of sample.** Every family in `RESEARCH_LOG_2026-10-03.md` was rejected
  or failed its holdout. The only robust result is HL portfolio-margin cash-and-carry, which is a yield, not alpha.
- **Self-modification is drill-gated.** The previous self-improvement engine (strategy breeding, primitive synthesis,
  charter patches, auto-promotion, a background daemon) evaluated on synthetic panels and invented metrics. It was
  retired. Its replacement (Phase 3) lets agents propose changes to role charters, and a change goes live only when
  a paired trial on planted and null drill markets shows it helps. See "Process trials" below.
- **If your ledger was used with the old engine or dashboard**, it contains rows that engine wrote (seeded lessons
  with invented Sharpe ratios, fake factor reports, evolved strategies). Inspect and move them out with
  `python -m desk rsi quarantine` (dry run) and then `python -m desk rsi quarantine --apply` (backs up first).
- **The evaluator is sealed (Phase 1a), with one platform caveat.** See "Evaluator integrity" below. On Linux and
  WSL2 the kernel enforces it. On plain Windows only a Python-level guard applies, which stops accidents and casual
  cheating but not a determined adversary: run the firm under WSL2 (or a Linux container) when it runs unattended.
  Out-of-sample discipline (Phase 1b) is in place; see "Out-of-sample discipline" below.

## Evaluator integrity

The evaluator is the part of the system that decides whether a strategy is real. Agents may change strategies,
factors and their own tools; they may not change how those are judged.

- **Strategy and factor code never runs in the evaluator's process.** `load_strategy` returns a proxy; each
  `signal`, `fit` and `factor` call runs in a fresh child process (`desk/lab/jail.py`, `jail_worker.py`). The child
  gets a copy of the input frame, has no API keys or data paths, cannot read files outside the Python installation
  and `desk/lib`, cannot write, cannot open sockets or start programs, and cannot import the evaluator. Only numbers
  come back (JSON and raw float64); fitted models stay opaque bytes that are only unpickled inside a jail. Because
  nothing survives between calls, a strategy cannot remember the full sample and replay it in a causality prefix call.
- **Agent research sandboxes** (`run_python`) see code, the Python installation, the dev-only data views and their own
  workspace. The ledger, the full store, val/holdout views and other agents' workspaces are refused, as are
  sockets and subprocesses.
- **Layers** (`desk/lab/guard.py`): Landlock (kernel file allowlist, also covers polars' Rust readers), seccomp (no
  exec, fork, sockets, ptrace) and a Python audit hook. Linux/WSL2 get all three; Windows/macOS get the audit hook only.
- **Kernel manifest.** `desk/KERNEL_MANIFEST.json` pins the sha256 of every evaluator file (`desk/lab/`, validation,
  bundles, the store, config). The harness refuses to evaluate if any differs, so an unreviewed edit stops evaluation
  instead of silently changing it. After reviewing a kernel change: `python -m desk kernel seal`, commit the manifest
  with it. `.github/CODEOWNERS` requires the owner's review for these paths; CI checks the manifest.
- **Cost model.** Agents can make it stricter on their own. A change that loosens it is a proposal until a human runs
  `python -m desk costs` (list) and `python -m desk costs approve <id>` (or `reject`).
- **One-shot holdout per strategy logic.** Holdout use is keyed by a structural hash of the code: comments, formatting,
  docstrings and renamed variables do not make a new strategy.
- **Red-team tests.** `tests/v3/test_evaluator_integrity.py` holds every known attack (patching the scorer, reading
  sealed data through Python or polars, sockets, subprocesses, ctypes into libc, the unaudited `fork_exec`, memory
  across causality calls, cosmetic holdout resubmission, loosening costs, editing the kernel). All must be caught.
  On Windows, run that file once after pulling: it checks the libraries strategies use (scipy, sklearn, lightgbm) still
  work inside the jail on your machine.
- Cost: each jailed call starts a Python process (about 0.3 s on Linux, more on Windows; about 1.5 s when the strategy
  imports sklearn). Evaluations take a few seconds longer.

## Out-of-sample discipline

Sealed data is finite: every look at it spends some of its value. Phase 1b limits and counts the looks.

- **Validation budget per strategy family.** A family is the same logic with any parameter values (the structural
  hash with numbers blanked). Each family gets `val_budget_per_family` (default 2) runs on sealed validation; re-tuned
  variants after that are refused at submission and in the pipeline. Every look, including failed and errored ones,
  is recorded (`val_uses`) and counted in the deflated Sharpe.
- **Trials are not voluntary.** A sandbox run that calls a backtest or evaluation function counts as at least one
  trial even if the agent bypasses the research API or deletes its trial log.
- **Look-ahead checks for the panel desks** (`desk/lab/causality.py`): prefix invariance at four cut points, a
  future-perturbation test (scramble every value after a cut; earlier weights must not move, which catches code that
  only cheats on the full panel), and a lint that refuses calendar dates or epoch timestamps written into the code.
- **Engine certification** for panel and maker engines is granted only while the kernel manifest matches.
- **Point-in-time data.** Macro strategies see a dividend-adjusted price rebuilt from daily returns: Yahoo's
  back-adjusted `adjclose` (and `adj_open`) revealed dividends paid after each date. Agent sandboxes read the carry
  desk through a dev-only copy (`store/carry_views/dev`) built at startup and on refresh; val/holdout hours are not in
  it at all.
- **Known remaining bias.** The pred universe is resolved Polymarket markets with lifetime volume >= $250k: a
  selection on future popularity that needs per-market volume history to remove. Treat pred results as optimistic
  until confirmed on forward data.

## Why v3 exists (what was wrong with v2)

| v2 problem | v3 fix |
|---|---|
| Each 15-minute OKX periodic snapshot was treated as a break, cutting every day into 96 isolated 900-row segments. Features could not see more than 15 minutes and positions were force-closed every 15 minutes. | Snapshots are resyncs; the stream is continuous across days. The resync drift is measured as a quality metric: the max mid jump at a resync is about 2 bps. |
| Holding capped at 300 s and taker-only, against a ~10 bps round trip, while the median 5-minute move is 7.5 bps | Horizons from 1 s to days, with hourly Hyperliquid funding modeled from real history |
| 26 features from the top 20 of 5,000 levels | 49 columns: multi-level OFI; depth, imbalance and order counts at 2–100 bps; walls; sweep impact for 1/10/50 BTC; aggressor trade flow; funding |
| One serial pipeline (director → scientist → engineer → falsifier → historian); one failure stopped everything; ~6k-token compliance prompts | About 28 parallel agents in pods with ownership, a shared board, a factor library, a playbook and a yield bandit |
| A tiny code DSL and 8 tool steps | Full Python sandbox (numpy, polars, scipy, sklearn, lightgbm) and long sessions with notebook memory |
| About 2.5M tokens for 17 candidates and 0 strategies passed | Fast numba lab: a full day at 1 s backtests in under 0.5 s, and 60 dev days at 1 m in seconds |

## Run it

```powershell
Set-Location 'C:\Users\Mathis\.gemini\antigravity\scratch\quant_lab'
python -m desk data                      # one-time: OKX trades + funding + full-depth reconstruction (~1-2 h)
python -m desk splits                    # frozen dev / val / holdout (+ forward) split
python -m desk record --hours 720        # keep recording Hyperliquid books/trades AND live OKX features (paper warm-start, calibration)
python -m desk tape                      # one-time: price-level OKX trade tape (limit-order fill model)
python -m desk run --agents 16 --hours 8 # the research firm (lean default; --agents 28 for the full org)
python -m desk serve --port 8900         # read-only control room: http://127.0.0.1:8900
python -m desk status                    # KPIs, token use, funnel
python -m desk forward                   # ingest NEW Drive days -> rolling forward holdout for approved strategies (also automatic every 6 h)
python -m desk stop                      # graceful stop (agents finish their current step)
```

- **Model: Claude (recommended).** Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` (from console.anthropic.com;
  a Claude.ai subscription is not an API key). Every agent then runs on Claude Haiku 5.5 (`claude-haiku-5-5`) through
  the official `anthropic` SDK, with adaptive thinking at `DESK_EFFORT=medium` and prompt caching. Name another
  `claude-*` model in `MODEL_NAME` to change it. `DESK_USD_PER_HOUR` (e.g. `2`) makes agents wait while the last hour's
  spend at list prices is above the cap; set a monthly limit in the Console too. Running out of credit pauses the firm
  and it probes every 15 minutes, as with a plan quota.
- **Model: OpenCode.** `MODEL_NAME=auto` (the default) picks, at startup, the first model in `DESK_MODEL_PREFERENCE` that the
  OpenCode Zen gateway lists: `minimax-m3.1-flash-free`, then `deepseek-v4.1-flash-free`, then `glm-5.3-flash-free`.
  Free models on Zen come and go; the pick is printed and logged (`MODEL_SELECTED`). Name a model in `MODEL_NAME` to pin it.
- **Concurrency.** It adapts by itself (AIMD: starts at 8 and grows toward `--concurrency`, 32 by default, while the gateway does not throttle).
- **Compute caps.** `DESK_SANDBOXES` (default 8) caps concurrent agent Python processes; `DESK_JOB_SLOTS` (default 5) caps concurrent heavy evaluations.
- **Token budget.** `--tokens-per-hour N` adds a hard budget.
- **Cost controls.** All roles run on the flash model by default; set `DESK_FLASH_ROLES=researcher,toolsmith` and a stronger `MODEL_NAME` to give the judgment layer a bigger model.
  - About 94% of tokens were *input* tokens, because each step re-sent the whole session.
  - The rolling context keeps only the latest tool outputs verbatim, compacts at 40k tokens, uses smaller briefs, and lowers the staff-role cadence.
- **Resuming.** Re-running `run` resumes everything: agents rebuild context from their notebooks, and the pipeline resumes queued strategies.

### Running unattended: WSL2

The strategy jail is only enforced by the operating system on Linux (Landlock + seccomp). On Windows, run the firm in
WSL2 (Ubuntu), keeping the data on the Linux side for speed:

```powershell
wsl --install -d Ubuntu                  # once, from an admin PowerShell; reboot if asked
```
```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/Aiouyn/quant-lab.git ~/quant-lab && cd ~/quant-lab
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp /mnt/c/Users/Mathis/.gemini/antigravity/scratch/quant_lab/.env .              # your keys
rsync -a /mnt/c/Users/Mathis/.gemini/antigravity/scratch/quant_lab/results/ results/   # ledger and stores
rsync -a /mnt/c/Users/Mathis/.gemini/antigravity/scratch/quant_lab/data/ data/         # raw OKX data (or set OKX_DATA_DIR)
python -m desk kernel guard              # must report landlock+seccomp
export DESK_JAIL_REQUIRE_KERNEL=1        # refuse to evaluate strategies if the OS guard is missing
python -m desk run --agents 16 --hours 8
```

The dashboard (`python -m desk serve`) is reachable from Windows at http://localhost:8900. If `kernel guard` reports
anything but `landlock+seccomp`, update WSL (`wsl --update`): Landlock needs a 5.13+ kernel built with it.

## The organization

| Unit | Role |
|---|---|
| CIO | Sets pod budget weights (with a Thompson-sampling suggestion based on measured yield per token), reassigns researchers and publishes the Desk Memo |
| 6 pods (lead + researchers) | Order flow · deep book · trade flow · volatility/regime · medium-frequency · ML/meta. Each lead runs 2–4 open theses with test plans and does research too. Researchers own a thesis end to end: measure → `register_factor` → `evaluate_strategy` → `submit_strategy`. |
| 2 risk officers | Independent adversarial review of every submission that passes the full-dev gate |
| Portfolio manager | Builds risk-balanced portfolios from approved strategies and runs and monitors paper trading |
| Execution analyst | Calibrates the OKX→Hyperliquid cost transfer from recorder data (`set_cost_model`) |
| Librarian | Curates the playbook and writes the knowledge digest injected into every session brief |
| Toolsmith | Ships tested reusable primitives (`desk.lib.contrib`) |

How the agents stay coordinated:
- Each agent works in sessions. A session starts from a lean brief: memo, digest, pod board, factor leaderboard, inbox and the agent's own notebook.
- Colleagues coordinate through board claims, comments and messages, and build on each other's factors by ID.
- Every agent backtest is written to a firm-wide multiple-testing ledger, including the ones run inside sandboxes.

## The truth machine (gates every submission; see the status note above for its known gaps)

1. **Dev (60%).** Researchers only ever see development days. The sandbox data root is a hard-linked dev-only view.
2. **Full-dev gate.** All of these must hold:
   - Sharpe ≥ 1.5 and at least 60 trades
   - causality: prefix truncation and future perturbation must not move past outputs
   - parity of the numba engine with an independent event loop
   - still profitable with fees ×1.5 and with one extra unit of latency (+2 s on 1 s data, +1 bar on bars)
   - both halves positive and ≥ 50% of months positive
   - ≥ 60% of ±20% parameter neighbors profitable
3. **Officer review.** Approve, revise or reject.
4. **Sealed validation (25%).** All of these must hold:
   - net Sharpe ≥ 1
   - deflated Sharpe ≥ 0.95 against all trials the firm has run
   - bootstrap lower bound > 0
   - ≥ 55% positive weeks, ≥ 40 trades
   - top 5 days < 50% of PnL
   - drawdown ≤ 0.2 units
   - still profitable under cost stress

   Authors learn only which gates failed, never the sealed metrics.
5. **One-shot holdout (15%).** One shot per code hash.
6. **Approved.** A hashed bundle is exported to `results/v3/bundles/<id>`.
7. **Hyperliquid paper trading.** Decisions use live OKX features computed by the *identical* research code: the 5000-level book via REST each second plus trades over WebSocket, so there is no feature-transfer risk. Fills are simulated on Hyperliquid's live book: 1 s later, crossing the spread, with sweep impact, fees and hourly funding. `--features hl` decides on Hyperliquid book features instead. Drift bands are computed against validation returns.
8. **Portfolio.** Ledoit-Wolf risk parity, a 35% cap, correlation de-duplication, then a fresh portfolio paper run.
9. **Forward verdict** (`desk/lab/forward.py`, sealed). Approved strategies are scored on forward days, market data
   that did not exist at approval (`python -m desk forward`, also every 6 h during a run). The test is fixed in
   advance: looks only at 60, 120 and 250 forward days, each on exactly the first N days, a one-sided t-test on daily
   returns at p <= 0.05 / (3 x the number of strategies ever forward-tested), and the mean without the 5 best days
   must stay positive. A strategy is FORWARD_FAILED if its mean is not positive at the 120-day look or it has not
   confirmed by the 250-day look, and it then leaves the approved set (status `forward_failed`). Each report states
   the Sharpe the next look can detect. The previous rule (Sharpe >= 0.5 over 60 days) confirmed a strategy with no
   edge about 40% of the time.
10. **Firm verdict.** One equal-weight book of every strategy ever forward-tested (failures included, so dropping
    losers afterwards cannot help) is tested once with the same looks. `firm_forward_verdict` in
    `python -m desk rsi scoreboard` is the firm's only claim to make money.

```powershell
python -m desk paper S-xxxxxx --hours 168     # paper trade one approved strategy
python -m desk portfolio                      # build from approved strategies -> P-xxxxxx
python -m desk paper P-xxxxxx --hours 168     # paper trade the portfolio
```

## Limit orders (maker execution)

A strategy may declare
`EXECUTION = {'mode': 'maker', 'ttl_s': 60, 'on_expiry': 'cancel'|'cross', 'repeg_s': 5, 'exit': 'maker'|'taker'}`.

How the queue-aware engine fills orders (the backtest and Hyperliquid paper trading behave identically):
- An order joins the back of the queue at the touch 1 s after the decision.
- Only aggressor trades at our price consume the queue ahead.
- A trade through our price fills us.
- After `repeg_s` seconds away from the touch, the order re-pegs; after `ttl_s`, it cancels or crosses as taker.

Supporting evidence and checks:
- The backtest uses the price-level OKX tape (`store/tape`); paper trading uses Hyperliquid's live trades and queue.
- An independent reference implementation reproduces the engine's PnL exactly (`tests/v3/test_maker.py`), on synthetic data and on a real day.
- Gates add `queue_x2_profitable`.
- Reports include fill fraction, 60 s markout (adverse selection) and time to fill.

## Alpha engine (`combine_factors`)

The engine builds strategies in three steps:
1. Causal base features (multi-scale imbalance, flow, depth asymmetry, returns, vol, spread, time of day) plus chosen library factors.
2. A walk-forward LightGBM or ridge model, fit only on strictly earlier days.
3. Selective positions, opened only when the predicted move exceeds `edge_bps`, under maker or taker execution.

It generates a normal strategy module, so every gate, bundle and paper path applies unchanged. Agents call it instead of hand-writing model code.

## Measured feasibility (dev only, walk-forward; validation and holdout untouched)

On 179–199 out-of-sample BTC dev days, the generic base-feature model has out-of-sample rank IC of about 0.00–0.02 at 15–240 min. The extreme 1% of predictions realizes about ±2 bps, which is below the maker round trip plus adverse selection (about 4 bps). Every taker and maker configuration lost money (`results/v3/feasibility.json`).

Book and flow features on BTC perps at 1 Hz do not carry an obvious tradable edge. Profitable systems, if any, must come from genuinely new information or conditional/event structure that the agents discover, or from a different opportunity set.

## Carry desk (cross-venue funding)

A second opportunity set with far more history: Hyperliquid vs Binance USDT-M perps for about 30 liquid coins, hourly, from July 2023 onward.
- **Data** (`python -m desk carry-data`): Hyperliquid funding and premium; Binance funding events, premium index and klines (public archives plus API).
- **Split:** the desk has its own frozen 60/25/15 split.
- **Strategies:** a strategy sets `DESK = 'carry'` and `signal(panel, params)`, returning one position per (hour, coin). +1 means long Hyperliquid and short Binance.
- **Engine:** books the funding differential, the premium-spread (basis) change, and both legs' taker fees plus liquidity-scaled slippage, with gross exposure capped. An independent reference loop reproduces its PnL exactly (`tests/v3/test_carry.py`).
- **Paper trading:** `python -m desk paper <carry bundle>` simulates both legs at live marks. It settles real funding on each venue's own schedule and marks both legs to market at live prices.

### Measured result: one pre-registered test

`results/v3/preregistered/` holds the strategy, its hash and the result. Carry hysteresis v1 (maker execution, 14-day window, entry 0.6 / exit 0.3 bps/h) was selected on dev only and hash-registered before any sealed data was read.

| Stage | Result |
|---|---|
| Dev (731 days, 42 coins) | Sharpe 10.5; every robustness variant held; 6 of 6 parameter neighbors profitable |
| Sealed validation (2025-06 → 2026-04) | Sharpe 0.29, 9 trades, funding $3.0k vs basis −$1.8k → **rejected** |

The Hyperliquid-vs-Binance funding spreads of 2023–24 compressed as the venue matured.

Carry hysteresis v2 harvests smaller spreads: maker execution, 21-day window, entry 0.3 / exit 0.15 bps/h, about 53 coins. It was designed after the v1 failure, which is disclosed, so its validation result is contaminated evidence and the holdout is the clean test.

| Stage | Result |
|---|---|
| Dev | Sharpe 14.1; all robustness variants held |
| Validation | Sharpe 2.11, +$10.6k on a $100k unit, deflated Sharpe 0.94, stress-profitable. **Rejected** only on `top5_days<50%` (77% of PnL in 5 days). |
| Holdout (2026-04 → 2026-10; read once as a disclosed diagnostic; strategy stays rejected) | Sharpe 4.56, +$4.5k, max drawdown 0.4%, every rolling 61-day window positive, only 7 trades on 6 coins |

As of 2026-10-01, no liquid coin exceeds the 0.3 bps/h entry threshold; the widest is about 0.23 bps/h. The edge is real but currently thin.

**Generalization test (pre-registered):** the unchanged v2 rule was applied to 77 Hyperliquid-vs-Bybit pairs (`COIN@BY`, data never evaluated before) and **failed its decision rule**.

| Period | Result |
|---|---|
| Dev | Sharpe 10.4 |
| Validation | Sharpe 0.88, bootstrap lower bound < 0 |
| Holdout | Sharpe 1.0 on only 4 trades / 2 coins |

Funding spreads have compressed across venues. Bybit pairs are supported in the carry engine and live paper (`--bybit`), but they add little recent profit.

## Hyperliquid multi-asset recorder

```powershell
python -m desk record --hours 720 --coins BTC,ETH,SOL,HYPE,XRP,DOGE,SUI,ENA --archive "G:\Mon Drive\HL_L2"
```

- Records per-second research-schema features for each coin (sizes in BTC-equivalent units of 0.01 BTC, so depth and impact compare across coins), plus live OKX BTC features.
- Compacts each coin's data into one file per day and archives it to Google Drive.
- Keeps only 3 days locally, because C: is nearly full.
- This builds a venue-native dataset for research on less-efficient Hyperliquid markets.

## Reality calibration

`REALITY_CALIBRATION.md` documents how real Hyperliquid fills from the operator's earlier live bot recalibrated the simulator:
- default latency 2 s
- volatility-scaled slippage
- a missed-entries gate
- per-coin HL spreads
- a validation of the carry basis model against real cross-venue prices

## Measuring the firm itself (drills)

```powershell
python -m desk drill planted     # real books + an injected causal drift (hidden slow queue-imbalance state)
python -m desk run --drill planted --agents 12 --hours 2
python -m desk drill planted --report

python -m desk drill null        # features paired with another day's price path, scrambled in 1-min blocks: nothing is predictable
python -m desk run --drill null --agents 12 --hours 2
python -m desk drill null --report  # every approval here is a false discovery
```

The planted drill measures *discovery power*. The null drill measures the *false-discovery rate*. The causality tests and
`tests/v3` measure the plumbing (`python -m pytest`).

The perps and macro desks have their own drills (`desk/lab/panel_drills.py`), built from the desk's real panel:

```powershell
python -m desk drill planted --desk perps    # hidden causal effect from one visible column, chosen at random
python -m desk run --drill perps-planted --agents 12 --hours 2
python -m desk drill planted --desk perps --report

python -m desk drill null --desk macro       # each asset's returns shuffled in weekly blocks, nothing predictable
python -m desk run --drill macro-null --agents 12 --hours 2
python -m desk drill null --desk macro --report

python -m desk drill scoreboard              # latest verdict per drill, rolled up into the RSI scoreboard
```

The planted effect's column, window, transform and sign are drawn when the drill is built and kept in the drill home
(`drill_secret.json`), which agent sandboxes cannot read, so reading the drill code does not give the answer away.
Every `--report` is recorded in the main ledger (`drill_scores`): `null_false_discoveries`, `planted_signal_corr`,
the overall verdict and the desks covered appear in `python -m desk rsi scoreboard`. These numbers are the fitness
of the firm's PROCESS: a change to agents, prompts or tools is an improvement only if planted discovery rises while
null false discoveries stay at zero. Pred and carry drills are not built yet (pred needs synthetic resolutions, and
carry's real edge is a yield that a null market must remove explicitly).

### Process trials: how the firm changes itself

The CIO, leads, officers, librarian and toolsmith can propose a *process variant*: a short working rule appended to
one role's charter, optionally for one pod (`propose_process_change`). A proposal changes nothing until a paired
drill trial (`desk/lab/process_trials.py`, part of the sealed kernel) accepts it:

```powershell
python -m desk process list                         # live rules, proposals, past verdicts and why
python -m desk process propose --role researcher --pod xsec --patch "..." --rationale "..."   # a human can propose too
python -m desk process trial                        # trial the oldest proposal (or pass its id)
python -m desk process trial PV-1a2b3c --desk macro --pairs 3 --hours 2 --agents 8
python -m desk process retire PV-1a2b3c --rationale "superseded"
```

For each pair, one planted market is built with a fixed seed and copied for both arms; the BASE arm runs the firm
with today's live rules, the CAND arm adds the candidate. The candidate also runs one null market per pair. The rule
is fixed in advance: **ACCEPT** only if the candidate approved nothing on any null market, beat BASE on a majority of
pairs (gains under 0.02 are ties) and improved the planted score by at least 0.05 on average; otherwise **REJECT**.
A failed run makes the trial **ERROR** and the proposal stays queued. An accepted rule replaces the live rule for the
same role and pod. Every run's seed, score and verdict is stored in the main ledger (`process_trials`), and agents
read the history before proposing again. The retired engine's `charter_versions` are never read.

Cost: a trial is 3 firm runs per pair, run one after another (default 3 pairs, 2 h each, 8 agents: 18 hours of wall time and about 144 agent-hours of LLM use). With
three pairs the test can only detect large improvements; that is deliberate, since a rule that only helps a little
is not worth the risk of being noise.

## Costs and execution model

- **Fees.** Hyperliquid base tier: taker 4.5 bps plus 0.5 bps slippage per side.
- **Fills.** 1 s latency (one bar on bar strategies), crossing the half spread, plus the average depth-sweep impact measured from the deep OKX book at decision time (× `impact_mult`, which the execution analyst calibrates from Hyperliquid captures).
- **Funding.** Actual hourly Hyperliquid funding is charged at each hour boundary.
- **Maker strategies.** Not promoted. The archive has trade prints but no queue positions, so maker fills cannot be certified.

## Layout

| Path | Contents |
|---|---|
| `desk/data` | Grid order book (full depth), ingestion, OKX trades/funding, feature store and split guard |
| `desk/lab` | Numba backtester, independent certifier, strategy contract, factors, stats (PSR/DSR/PBO), sandbox, harness, drills |
| `desk/company` | Ledger (SQLite + FTS5), MiMo gateway (stream, reasoning passback, AIMD), roles, tools, agents, runtime |
| `desk/pipeline` | Validation gates, bundles, portfolio construction |
| `desk/live` | Hyperliquid recorder, live OKX feature feed (research-identical), paper engine, monitor |
| `desk/web` | Control room |
| `results/v3/store` | `1s`, `10s`, `1m`, `trades_1s`, `funding_hl.parquet`, `splits.json`, `views/dev` |
| `results/v3/desk.sqlite3` | The firm's memory |
| `results/v3/workspaces/<agent>` | Each agent's notebook and scripts |
| `results/v3/bundles`, `results/v3/portfolios`, `results/v3/paper`, `results/v3/hl_store` | Approved bundles, portfolios, paper journals and Hyperliquid captures |
