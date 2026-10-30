# Quant Lab v3 — Desk OS

An autonomous quantitative research firm made of MiMo agents. It studies 332 days of 1 Hz full-depth OKX
BTC-USDT-SWAP order books, plus every OKX trade print and hourly Hyperliquid funding. It turns what it learns into
strategies that are paper-traded on Hyperliquid and combined into portfolios.

v2 (`quantdesk/`) is kept for its ledger and its running dashboard. The legacy swarm, reports and tests are in `legacy/`.

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
python -m desk serve --port 8900         # control room: http://127.0.0.1:8900
python -m desk status                    # KPIs, token use, funnel
python -m desk forward                   # ingest NEW Drive days -> rolling forward holdout for approved strategies (also automatic every 6 h)
python -m desk stop                      # graceful stop (agents finish their current step)
```

- **Concurrency.** It adapts by itself (AIMD: starts at 8 and grows toward `--concurrency`, 32 by default, while MiMo does not throttle).
- **Compute caps.** `DESK_SANDBOXES` (default 8) caps concurrent agent Python processes; `DESK_JOB_SLOTS` (default 5) caps concurrent heavy evaluations.
- **Token budget.** `--tokens-per-hour N` adds a hard budget.
- **Cost controls.** All roles run on `mimo-v2.6-flash` by default; `DESK_FLASH_ROLES=researcher,toolsmith` restores Pro for the judgment layer.
  - About 94% of tokens were *input* tokens, because each step re-sent the whole session.
  - The rolling context keeps only the latest tool outputs verbatim, compacts at 40k tokens, uses smaller briefs, and lowers the staff-role cadence.
- **Resuming.** Re-running `run` resumes everything: agents rebuild context from their notebooks, and the pipeline resumes queued strategies.

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

## The truth machine (no agent can talk its way past it)

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
   - deflated Sharpe ≥ 0.95 against all trials the firm has run (every backtest and variant in the insert-only
     trial ledger, per strategy family and across families, with the variance of trial Sharpes)
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
