# Spec: research data and evaluators needed by preregs #4, #8 and #9

- **Status:** SPEC ONLY. Written 2026-10-21 from code and docs; no data was read and nothing was run.
- **Reviewers:** QL Validator, QL Safety, QL Engineer.
- **Base:** origin/main `313b185cac9e53c9d0bfcaf48774b34e79d72d34`.
- **Consumers:**
  - **#4:** `docs/prereg/2026-10-21_hl_native_carry_btc_hype_val.md` (branch `research/prereg-hl-carry`)
  - **#8:** `docs/prereg/2026-10-21_macro_xasset_tsmom_fxcarry.md` (branch `research/prereg-macro-tsmom`)
  - **#9:** `docs/prereg/2026-10-21_polymarket_favorite_longshot.md` (branch `research/prereg-pred-flb`)
- **Out of scope:** `desk/company/rsi.py`, gate logic, DSR methodology (owned by Validator / RSI Architect), and infra design. This doc only states needs.

## Rules that apply to every item
- **G1, no holdout read:**
  - Every new evaluator takes an explicit `allowed` split set. It must raise `SplitViolation` (`desk/data/store.py`) when asked for holdout.
  - Test: call the evaluator with `split='holdout'` and with `allowed=('dev','val')`, and assert it raises. Also add a monkeypatched-open test asserting that no holdout-dated rows are ever materialized (a fixture panel with a sentinel `NaN` in holdout rows; the result must be finite).
- **G2, no lookahead:** a causality test in the style of `desk/lab/carry.py::carry_causality`. Truncate the panel at cut-points 0.4 and 0.7; positions and signals before the cut must be identical.
- **G3, determinism:** two runs with the same inputs and seed give byte-identical result JSON. Its SHA-256 is recorded.
- **G4, pinning:** every output file is written with a sidecar `<file>.sha256` and a `manifest.json` containing `{git_commit, code_sha256 per file, inputs: {path: sha256}, created_utc}`.
- **G5, nothing fabricated:** when an input is missing, the code **raises**. It never imputes or substitutes synthetic data.

## P0 prerequisites

### P0-A. Synthetic-fallback fix and provenance checks (`desk/data/store_builder.py`)
- **Purpose:** #4, #8 and #9 all abort unless the store is real. Today `store_builder.py` silently falls back to seeded synthetic panels (rng seeds 20261017/18/19). It also **overwrites** `carry/splits.json` and `perps/splits.json` with a synthetic 150/110/60/45-day split, which conflicts with `carry_data.py::_freeze_splits` (60/25/15).
- **Inputs:** the existing builders in `store_builder.py`, and `provenance.json` per desk.
- **Outputs:**
  1. The synthetic generator only runs behind an explicit `--synthetic` flag, and writes to `results/v3/store_synthetic/<desk>/`, **never** to `results/v3/store/`.
  2. `results/v3/store/<desk>/provenance.json` with schema `{desk, source ∈ {yahoo, hyperliquid+binance, polymarket}, verified_real: true, fetched_utc, row_count, first_ts, last_ts, fetcher_commit}`.
  3. A shared `assert_real(desk)` helper that every evaluator calls first.
  4. The `splits.json` files are never rewritten when they already exist (see P0-B).
- **Acceptance tests:**
  - With the network mocked to fail, building into `store/` raises, and no file under `store/` changes (check by hashing the directory before and after).
  - With `--synthetic`, output lands only in `store_synthetic/`, and `assert_real` on it raises.
  - `assert_real` raises on `verified_real=false` or `source=deterministic_local_fixture`.
  - Running the builder on an existing `splits.json` leaves its SHA-256 unchanged.
- **Dependencies:** none.
- **Priority:** P0, because every result on a possibly-synthetic store is meaningless.
- **Size:** S–M (about 1–2 days).

### P0-B. `splits.json` hashing and split metadata (all desks)
- **Purpose:** #4 §3 (SHA, panel N, first/last day, val dates), #8 §1, #9 §2.
- **Inputs:** `results/v3/store/{carry,macro,pred,perps}/splits.json`. For carry: **only the `ts` column** of the coin panels (read through a column projection; no value columns).
- **Outputs:** `results/v3/store/split_manifest.json`:
  ```
  {desk: {splits_sha256, splits: {...}, generator: "carry_data._freeze_splits" | "lab SPLITS const" | "store_builder synthetic",
          panel_days_N, panel_first_day, panel_last_day, val_start, val_end, recomputed_val_matches: bool}}
  ```
  For carry, `val_start = days[floor(0.60·N)]` and `val_end = days[floor(0.85·N)−1]`, matching `_freeze_splits`. For macro and pred, the values are compared against the `SPLITS` constants in `desk/lab/macro.py` and `desk/lab/pred.py`.
- **Acceptance tests:**
  - A fixture panel with N = 100 days gives val = days[60..84].
  - A fixture whose splits.json `note` is the store_builder "contiguous 365d" one is flagged `generator = store_builder synthetic` and makes the tool exit non-zero.
  - Reads are column-projected to `ts`. A test asserts no other column is loaded (spy on `pl.scan_parquet().select`).
  - Determinism: the same inputs give the same manifest hash.
- **Dependencies:** P0-A.
- **Priority:** P0. It's cheap and unblocks every prereg's pins.
- **Size:** S (about 0.5 day).

### P0-C. Ledger trial-count query (reference only)
- **Purpose:** the n_trials floors in #4 (≥ 4), #8 (≥ 6) and #9 (≥ 15) are each combined with "ledger count if larger".
- **Observation from code:** `Ledger.trial_stats` (`desk/company/ledger.py`) counts `COUNT(DISTINCT code_hash)` over `trials WHERE kind IN ('dev_backtest','submission') AND days>=10`. That is firm-wide: the `trials` table has **no desk or split column**. Per-desk val counts therefore have to come from `strategies` (`val_report IS NOT NULL`), with the desk taken from the report JSON.
  - Proposed read-only query (the JSON path `$.desk` is an assumption that must be checked against real reports): `SELECT COUNT(DISTINCT code_hash) FROM strategies WHERE val_report IS NOT NULL AND json_extract(val_report,'$.desk') = ?`.
  - Evaluations run by ad-hoc scripts (e.g. those in `RESEARCH_LOG_2026-10-03.md`) are **not** in the ledger. That's why the preregs keep hand-counted floors.
- **Outputs:** `{desk: {val_trials, trial_sr_std}}` for the preregs to copy in.
- **Ownership:** QL Validator / RSI Architect own DSR and decide the definition. Engineering only provides the read-only query.
- **Priority:** P0 for running, but it is not an engineering build. **Size:** XS.

## Items

### (1) HL-native carry evaluator: needed by #4
- **Purpose:** backtest the always-on 1× long HL spot / 1× short HL perp position for BTC and HYPE. No such evaluator exists: `desk/lab/carry.py` models HL perp vs **Binance perp**, and the script behind the research-log result isn't in the repo.
- **Inputs:**
  - HL hourly perp funding, HL perp mark/mid, HL spot mid for BTC and HYPE (from `desk/data/carry_data.py` / `hl_ctx.py` sources), plus a schema `{ts, coin, perp_px, spot_px, hl_funding}`.
  - Val window from P0-B.
  - Costs and rules exactly as in #4 §5–§7: 10 bps per entry, exit and re-hedge notional; re-hedge monthly or when |Δprice since last hedge| > 40%; 1.15× capital; USDC interest excluded; 1.5× stress.
- **Outputs:** `results/v3/prereg/hl_carry/val_result.json` containing:
  - per coin and combined: annualized return on capital, each quarter's return, max drawdown, share of hours with negative funding, `block_bootstrap_lower`, DSR inputs
  - the HYPE coverage rule from #4 §5: a BTC-only figure always, with HYPE excluded below 90 val days
  - a manifest per G4
- **Acceptance tests:**
  - **Fixture 1:** constant funding of +1 bp/h, flat prices, no re-hedge. Expected return on capital = (8760 bp/yr − 20 bp) / 1.15, within 1e-9.
  - **Fixture 2:** a +45% price path triggers exactly one re-hedge, charged 10 bps on the re-hedged notional.
  - **Fixture 3:** negative funding gives a negative return, with the sign checked.
  - **Fixture 4:** 1.5× stress raises costs by exactly 1.5×.
  - G1 (asked for holdout, it raises), G2, G3.
- **Dependencies:** P0-A, P0-B.
- **Priority:** P1. #4's forward paper (day 30, about 2026-11-02) is the stronger evidence and doesn't need this, but the val check does.
- **Size:** M (about 2–3 days).

### (2) FX-carry engine change and rate series (10 verified, 3 pairs dropped): needed by #8
- **Purpose:** `desk/lab/macro.py` and `macro_data.py` state that FX spot returns exclude interest-rate carry. #8 §4 requires the daily carry r_base − r_quote (ACT/360) on every `fx` row, using rates known at close t and applied over (t+1, t+2].
- **Inputs:**
  - US: `^IRX`, already in the macro panel (class `rate`).
  - Non-US short rates: the series IDs are now fixed in #8 §4 (OECD MEI via FRED, monthly, lagged one month).
    - **Verified:** `IR3TIB01{EZ,GB,JP,AU,NZ,CA,CH,MX,ZA,CN}M156N`. Each FRED catalog page was checked for existence and date range covering 2014–2019; no data was downloaded.
    - **Pairs dropped in #8:** TRY=X (`IR3TIB01TRM156N` ends 2008-04), and BRL=X and INR=X (candidate IDs `IR3TIB01BRM156N` and `IR3TIB01INM156N` are **UNVERIFIED**; their FRED pages returned 404). That leaves 11 pairs.
    - **Correction to the earlier version of this spec:** `MXN=X`, `ZAR=X` and `CNY=X` are also quoted as USD/XXX, so the carry is r_USD − r_XXX for those pairs too.
  - Release timing: lag each monthly value by one month.
  - Pair mapping: from the `UNIVERSE` symbols in `macro_data.py`. Note that `JPY=X`, `CAD=X`, `CHF=X` and the other `XXX=X` symbols are quoted as USD/XXX, so the carry sign flips for them. `EURJPY=X` is a cross, so its carry is r_EUR − r_JPY.
- **Outputs:**
  - `results/v3/store/macro/rates.parquet` with schema `{date, ccy, rate_annual, source_id, available_from}`, plus a sha256 sidecar.
  - An engine change in `desk/lab/macro.py` that adds carry for `cls == 'fx'` rows only, behind a flag that defaults on once reviewed.
  - A currency with no reviewed series means that pair is **excluded** (G5). Nothing is imputed.
- **Acceptance tests:**
  - **Fixture:** a constant 5% base rate and 0% quote rate on a flat price gives a daily return of 0.05/360 per calendar day, applied on the documented lag.
  - **Fixture:** the sign flips for USD/XXX quoting.
  - **No-lookahead:** a rate value dated month m only affects returns from `available_from` onwards (G2-style truncation test).
  - Non-fx rows' P&L is byte-identical before and after the change (regression hash).
  - A missing currency raises or excludes the pair; it is never filled.
  - G1, G3.
- **Dependencies:** P0-A (macro store real), P0-B.
- **Priority:** P1. It blocks #8 completely, and the risk premium #8 targets is well documented, but #8 also needs the configs written.
- **Size:** M (about 2–4 days, mostly sourcing the series and checking quoting).

### (3) Full Polymarket re-fetch with archived/delisted coverage: needed by #9
- **Purpose:** #9 §4 needs a full, non-incremental re-page, because the cursor checkpoint in `desk/data/polymarket.py::fetch_markets` "may miss a market created long ago that closes later". It also needs an answer on whether gamma `closed=true` returns archived or delisted markets, which is unknown from the code.
- **Inputs:** gamma-api `/markets/keyset` (`closed=true`, plus an `archived=true` comparison pass if the API supports it, which needs verifying), and clob `prices-history` as now.
- **Additional stored fields (required by #9 r2):**
  - `closed_time` (gamma `closedTime`) and `resolution_time` (gamma field name **UNVERIFIED**, e.g. the UMA resolution timestamp). These feed #9's anchor A = min(endDate, closedTime, resolution) and its rule dropping markets with closedTime < endDate.
  - **Trades:** per-market trade records `{market, ts, side, price, size}` from Polymarket's public trades endpoint (data-api; endpoint and fields **UNVERIFIED**). They feed #9's floor of at least 20 trades in the 7 days before the snapshot.
- **Outputs:**
  - `results/v3/store/pred/markets_full.parquet` (same columns as now, plus `closed_time`, `resolution_time` and the item-4 fields).
  - `results/v3/store/pred/trades/<market>.parquet`, plus a sha256 manifest.
  - `results/v3/store/pred/fetch_report.json` with schema `{pages, markets_seen, kept, dropped_by_reason: {non_binary, unresolved, low_volume, parse}, archived_pass: {supported, extra_markets}, missing_price_history}`.
  - A fresh cursor that is not reused.
- **Acceptance tests:**
  - **Mocked API fixture** (pages with known markets, including one "old-created, late-closed" market placed behind a stale cursor): the full mode finds it, and incremental mode documents that it misses it.
  - The drop-reason counts equal the fixture's known counts.
  - The price-history missing rate is reported. #9 aborts if it is over 10% of eligible markets (the threshold lives in the prereg, not the fetcher).
  - Re-running against the same mocked responses gives an identical parquet hash (G3).
  - **Fixture:** a market with closedTime < endDate is flagged; A = min(endDate, closedTime, resolution) is computed correctly. Missing `closed_time` or `resolution_time` raises; it is never imputed.
  - **Fixture trades:** the trade count in [snapshot − 7 d, snapshot) is correct at the window boundaries. No trades after the snapshot are counted (no-lookahead).
- **Dependencies:** P0-A.
- **Priority:** P1. #9 can't define its universe without it.
- **Size:** M (about 2 days, plus run time).

### (4) Stored outcome labels and event/group ids: needed by #9 (clustering: event id → negRiskMarketID → (category, date of A); more than 20% on the last fallback means exploratory)
- **Purpose:** #9 §4 excludes non-Yes/No binaries, because `fetch_markets` currently treats the first outcome as "YES" when there is no "yes" label. #9 §5–§6 also cluster bootstraps by event.
- **Inputs:** gamma market fields `outcomes`, `events[].id`, `negRiskMarketID` (field names taken from #9 and the gamma API; their presence must be verified against live responses).
- **Outputs:** new columns in `markets_full.parquet` and the panel metadata: `outcome_labels: list[str]`, `is_yes_no: bool`, `event_id: str | null`, `neg_risk_group: str | null`. If `event_id` is null for some markets, `fetch_report.json` reports the share. #9 then falls back to the `end_ms` date plus `category` cluster, and discloses that.
- **Acceptance tests:**
  - A fixture with outcomes `["Yes","No"]`, `["No","Yes"]` and `["Team A","Team B"]` gives `is_yes_no` = T, T, F, and the YES index is correct for the second.
  - A fixture of a 3-candidate neg-risk event gives 3 rows with the same `event_id`.
  - The `outcome` column is unchanged for existing Yes/No markets (regression).
- **Dependencies:** item (3), which writes these columns in the same fetch.
- **Priority:** P1.
- **Size:** S (about 0.5–1 day on top of (3)).

### (5) Dispute / ambiguous-resolution source: needed by #9
- **Purpose:** #9 §4 excludes disputed or ambiguous resolutions. No field for this exists in the code.
- **Inputs:** **to be identified (REQUIRED).** Candidates are gamma market fields related to UMA resolution status, or UMA optimistic-oracle dispute events on-chain. Neither has been verified here. If no reliable source is found, the alternative is a documented waiver in #9.
- **Outputs:** `results/v3/store/pred/disputes.parquet` with schema `{market, disputed: bool, source, evidence_ref}`, plus a sha256 sidecar.
- **Acceptance tests:**
  - On a hand-labelled fixture of known disputed and clean markets (labels with cited public references), precision and recall are reported. QL Validator sets the bar.
  - Every disputed flag carries an `evidence_ref`.
  - The resolution outcome itself is never used as a feature (no outcome columns are read).
- **Dependencies:** item (3).
- **Priority:** P2. It's a refinement, and #9 permits a disclosed waiver.
- **Size:** M (research-heavy, and the size is uncertain).

### (6) Bucket and event-clustered bootstrap evaluator, plus the sleeve strategy: needed by #9
- **Purpose:** #9 §3, §5 and §6. Neither the evaluator nor the strategy is in the repo.
- **Inputs:**
  - The pred panel with columns `ts, market, p, end_ms, outcome, category, volume` (`desk/lab/pred.py`), plus `event_id` and `is_yes_no` from item 4.
  - Rules fixed by #9:
    - **Snapshot:** the last row ≤ A − 24 h, within 6 h, where A = min(endDate, closedTime, resolution); markets with closedTime < endDate are dropped and counted.
    - **Buckets:** 14 edges.
    - **Val membership:** both A and the snapshot fall in [2026-03-01, 2026-06-01) UTC. The sleeve trades only these markets.
    - **Liquidity:** at least 20 trades in the 7 days before the snapshot (from item 3's trades).
    - **Multiple testing:** Holm–Bonferroni across the 14 buckets at α = 0.05. Each bucket's p-value comes from the cluster bootstrap.
    - **Costs:** an **engine change in `desk/lab/pred.py`:** an uncapped 1.5-cent half-spread for p < 0.10 or p ≥ 0.90 (1 cent otherwise), plus a category taker fee of C × feeRate × p(1−p) (rates in #9 §7, with 0.07 for unmapped categories). Both are scaled by `fee_mult`.
    - **Minimum counts:** 30 markets and 10 events per bucket.
- **Outputs:**
  - `results/v3/prereg/pred_flb/calibration.json`: per bucket `{n_markets, n_events, mean_price, yes_rate, miscal, ci95_lo, ci95_hi, pass}`. The bootstrap resamples events, 10,000 draws, seed 7.
  - `sleeve.json`: base, 1.5× cost and +1 delay-row runs via `desk/lab/pred.py::run_pred` / `summarize_pred`, plus `block_bootstrap_lower`, `mean_ex_top`, results per half, an event-clustered lower bound, and DSR inputs.
  - The strategy file `desk/strategies/pred_flb_sleeve.py`, following the `pred.py` strategy contract (path is a proposal).
  - Manifests per G4.
- **Acceptance tests:**
  - **Planted-signal fixture:** synthetic markets where 0–2-cent contracts resolve YES at exactly half their price. The evaluator recovers miscalibration within the CI and passes. The sleeve's P&L sign is positive after costs.
  - **Null fixture:** perfectly calibrated outcomes, drawn Bernoulli(p). Over 200 seeds, the bucket false-pass rate is ≤ 5%.
  - **Clustering:** duplicating one event 50 times widens the CI compared with treating the copies as independent markets.
  - **Snapshot no-lookahead:** perturbing prices after the snapshot leaves bucket assignment unchanged.
  - **Boundary:** a market ending in val with a snapshot before 2026-03-01 is excluded and counted.
  - **Holm fixture:** with known p-values, the rejection set equals a hand-computed Holm result.
  - **Cost fixture:** at p = 0.03 the tail half-spread is 1.5 cents (not capped at 0.5 × 0.03 = 0.015, which here happens to coincide; also test p = 0.01, where the engine cap would give 0.005 and the new rule must give 0.015); the fee equals C × 0.07 × p(1−p) for crypto.
  - G1: holdout is never read. The fixture puts `NaN` outcomes in holdout rows and the result stays finite.
  - G3.
- **Dependencies:** item (4) (event ids). Item (3) is needed for the real run but not for the fixture tests.
- **Priority:** P1. It can be built in parallel against fixtures right away.
- **Size:** M (about 2–3 days).

### (7) Macro trading-day calendar and annualization fix: needed by #8
- **Purpose:** macro panel rows are calendar days whenever any symbol trades. Crypto trades on weekends, while `desk/lab/macro.py::summarize` annualizes with 252 (`sharpe(daily, periods=252)`, `years = len(daily)/252`), so a daily series that includes weekends is mis-annualized. #8 r2 drops BTC-USD and ETH-USD and evaluates on **NYSE trading days (dates where SPY has a row) at 252**.
- **Inputs:** the macro panel `date` and `symbol` columns, plus the #8 universe (crypto excluded).
- **Outputs:** an evaluator option (or wrapper) that restricts the simulation to the SPY-date calendar. Returns over a gap are close-to-close, and FX carry accrues ACT/360 over the gap. It must **assert** that every evaluation date is a SPY date and not a weekend.
- **Acceptance tests:**
  - **Fixture:** a panel with weekend crypto rows and weekday ETF rows. After the filter, no evaluation date is a weekend, and annualized Sharpe uses √252.
  - **Fixture:** a known constant daily return r on 252 trading days gives `ann_return` = 252·r.
  - Including a crypto symbol in the #8 universe raises.
  - The P&L of non-crypto symbols matches the unfiltered engine on SPY dates (regression).
  - G1 and G3.
- **Dependencies:** P0-A.
- **Priority:** P1. It blocks #8, and it is small.
- **Size:** S (about 0.5–1 day).

## Dependency graph
```mermaid
graph TD
  A[P0-A store_builder fix + provenance] --> B[P0-B splits.json hashing + metadata]
  A --> I3[(3) Polymarket full re-fetch]
  B --> I1[(1) HL-native carry evaluator]
  B --> I2[(2) FX carry engine + 10 verified rate series]
  A --> I7[(7) macro trading-day calendar]
  I7 --> P8
  I3 --> I4[(4) outcome labels + event ids]
  I3 --> I5[(5) dispute source]
  I4 --> I6[(6) bucket/cluster evaluator + sleeve]
  C[P0-C ledger trial-count query - Validator owned]
  I1 --> P4{{prereg #4 run}}
  I2 --> P8{{prereg #8 run}}
  I6 --> P9{{prereg #9 run}}
  I5 -.optional / waiver.-> P9
  C --> P4
  C --> P8
  C --> P9
  G[Eval gating fix - Validator/RSI, out of scope] --> P4
  G --> P8
  G --> P9
```

## Suggested queue order
1. **P0-A:** store_builder fix and provenance. It unblocks everything.
2. **P0-B:** split manifest. It fills #4 §3, #8 §1 and #9 §2.
3. **P0-C:** ledger query, handed to Validator. It runs in parallel.
4. **(6) against fixtures** and **(3)**, in parallel. Item (6) needs no real data to build and test.
5. **(4):** in the same fetch as (3).
6. **(1):** HL-native carry evaluator. Its timing can follow the #4 forward verdict around 2026-11-02.
7. **(7):** macro calendar fix (small; it can go earlier).
8. **(2):** FX carry and rates. The sourcing has the longest lead time, so start it early if capacity allows.
9. **(5):** dispute source, or a decision to waive it.

Nothing on this list authorises running any prereg. Each still needs its own REQUIRED-BEFORE-RUN items filled and the gating sign-off.
