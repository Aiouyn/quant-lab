# Running the lab for the long term

## What runs, and who keeps it alive

`desk.ops.supervisor` starts at Windows logon (Startup folder: `QuantLabSupervisor.vbs`, no admin needed). Every 60 s it checks the services below.

- **Restart:** a dead service is restarted, with exponential backoff if it keeps dying.
- **Code reload:** changes under `desk/` reload the long-running services, so nothing runs stale code. The recorders wait for their hourly flush (hh:00:45). The firm stops gracefully: non-blocking, killed after 15 min if needed. `desk/lib/contrib` (the agents' own primitives) is ignored.
- **Self-update:** if `supervisor.py` itself changes, it hands over to a fresh supervisor (`--after <pid>`).
- **Disk guard:** if C: or D: falls below 3 GB free, the firm is stopped gracefully. It restarts once 6 GB are free again.
- **Daily refresh (13:10 UTC):** `desk.data.refresh` updates the forward period of every desk:
  - perps: verified live HL recordings, Binance metrics, Deribit DVOL, macro context;
  - macro: Yahoo daily bars for ~119 instruments;
  - pred: newly resolved Polymarket markets.

| Service | Command | Role |
|---|---|---|
| firm | `python -m desk run --agents 16 --hours 8760` | MiMo research firm. Pauses on quota (probes every 15 min) and resumes alone. |
| parity_recorder | `python -m desk.live.parity_recorder` | Live HL/Binance data in archive schemas. Daily parity check against the archive. |
| hl_recorder | `python -m desk record --coins ...` | 1 Hz HL book/features for 8 coins. |
| regime | `python -m desk.live.regime --loop` | Funding regime every 6 h, plus the 30-day forward review of the paper sleeves. |
| dashboard | `python -m desk serve --port 8900` | http://127.0.0.1:8900 |
| hl_carry_paper, carry_v2_paper | paper sleeves | Yield sleeve, regime-gated carry. |

Config: `results/v3/supervisor.json`. Status: `results/v3/supervisor_status.json`. Log: `results/v3/supervisor.log`. Service logs: `results/v3/services/`.

## Research desks (all share the same truth machine)

| Desk | Data | Decisions / execution | Splits (dev / val / holdout / forward) |
|---|---|---|---|
| **macro** (priority 1) | ~119 Yahoo daily series since the 1990s–2000s: US, intl, country, sector and factor equity ETFs, Treasuries, credit, TIPS, EM bonds, commodities, REITs, currency ETFs, FX spot, BTC/ETH. Feature-only: indices, VIX family, MOVE, yields, dollar index, continuous futures. | Daily. Decide at close t, execute at close t+1. Per-asset costs, 1%/yr short borrow, gross ≤ 2. | 2000–14 / 2015–19 / 2020–25H1 / 2025H2 → |
| **pred** (priority 2) | Polymarket resolved binary markets, hourly YES prices. | Hourly. YES (+) / NO (−) weights, 1¢ half-spread incl. rebalancing, pays 1/0 at resolution. Outcome and final volume are hidden from strategies. | ≤2026-02 / 2026-03–05 / 2026-06–07 / 2026-08 → (universe concentrated in 2025–26) |
| **perps** | Every HL perp (delisted incl.), funding, OI, premium, impact spread, Binance positioning, Deribit DVOL, VIX/DXY/SPX/TNX/gold context. | Hourly. Execution ~60 s after the hour. Taker, or `PERPS_EXECUTION='maker'`: a limit order filled only if the next hour trades through it. | 2023-06–25-05 / 25-06–26-04 / 26-04–26-10 / 2026-10-02 → (verified live recordings) |
| **carry** | HL vs Binance funding panel | Hourly | own splits; no forward period |
| **l2** (OKX BTC) | 1 s order book | proven null; do not invest | — |

Every desk has:
- a dev-only sandbox file (the split guard, tested);
- a causality prefix test;
- engine-only columns that strategies can never see:
  - perps: the post-decision `lo`/`hi`;
  - pred: `outcome`, `volume`;
- drift-aware rebalancing costs.

**Truth machine:**
1. dev gate, including mean excluding the 5 best days > 0;
2. officer review;
3. sealed validation, with the same tail gate and deflated Sharpe over all validation runs;
4. one-shot holdout;
5. approved: bundle exported;
6. **rolling forward holdout:** every 6 h, every approved strategy is rescored on all unseen forward days. FORWARD_CONFIRMED after ≥ 60 days with Sharpe ≥ 0.5, positive PnL and positive mean excluding the top 5 days.

**Agent tools:**
- `register_factor` on every desk: cross-sectional IC, mean quintile spreads including without the 5 best periods, cost hurdle, causality, novelty versus the library.
- In the sandbox:
  - `event_study`, `xs_factor_report`, `walk_forward` (causal ML);
  - `perps_backtest(execution='maker')`, `macro_backtest`, `pred_backtest`.

**Memory:**
- the playbook holds the 9 families rejected on 2026-10-03;
- the board holds 18 director theses across desks (`results/v3/seed_theses.py`).

## Verified end to end (2026-10-06)

`scratchpad/e2e_pipeline.py`, on a temporary ledger: submit → dev gate → verdict → validation → holdout (approved and bundle exported for perps; rejected for macro) → forward scoring per desk.

Prediction desk at full scale: 43,839 resolved markets with lifetime volume ≥ $250k, 24.7M hourly rows (dev 15M rows / 21,635 markets).

- **Truth machine:** the full sequence (dev gate, verdict, validation, holdout, forward over 67 unseen days) runs under the production 5 GB job cap: dev ~25 s, validation ~10 s.
- **Engine:** share-holding semantics. Positions are held as shares and re-traded only when the strategy's target changes; the gross cap applies at decision time.
- **Costs:** the half-spread is 1¢, bounded by the price (min 0.1¢ tick).
- **Memory:** only traded markets are simulated, and the panel is lazy-loaded with compact dtypes.

Polymarket price histories: `interval=max` returns nothing for closed markets, and windows over ~14 days are rejected, so `desk.data.polymarket` walks back from each market's end in 14-day windows. Market paging is checkpointed (`markets_parts/`, `markets_cursor.txt`) and resumes after interruptions.

## Known limits

- **Quota:** the firm only thinks with MiMo quota (exhausted as of 2026-10-06).
- **Live execution:** none exists for any desk. A confirmed strategy needs a broker or venue connector: HL for perps; a broker such as IBKR for ETFs/FX; Polymarket on-chain for pred.
- **Desk caveats:**
  - Macro FX returns exclude interest carry.
  - The pred universe is selected on lifetime volume ≥ $250k (of 1.3M+ resolved binaries, mostly tiny in-game markets).
  - The carry panel is not refreshed daily.
  - The macro and pred history before the forward split is public knowledge; only the forward split is free of hindsight.
