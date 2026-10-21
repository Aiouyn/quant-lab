# Pre-registration: macro cross-asset time-series momentum with FX carry

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21 from code and docs only; no data files were read.
- **Desk:** macro.
- **Reviewers:** QL Validator, QL Safety.

## 0. Run conditions (binding)
- This test runs **once, on val only, and never on holdout.** The holdout is suspect anyway (`JOURNEY.md` §5.1/§7.3 in PR #3).
- It runs only after **both** of these are signed off by QL Validator:
  1. **Eval gating is fixed:** the DSR/`n_trials` accounting from the 2026-10-21 audit.
  2. **Store provenance is fixed:** `results/v3/store/macro/provenance.json` shows real Yahoo data (`desk/data/macro_data.py`), not the `store_builder.py` synthetic fallback (`source = deterministic_local_fixture`, or `verified_real = false`).
- **Abort** if any pin in §1 or §2 differs at run time, or if any REQUIRED-BEFORE-RUN item is still open.

## 1. Split (copied literally from `desk/lab/macro.py::SPLITS`)

| Split | Start | End |
|---|---|---|
| dev | 2000-01-01 | 2014-12-31 |
| **val** | **2015-01-01** | **2019-12-31** |
| holdout (never opened) | 2020-01-01 | 2025-06-30 |
| forward | 2025-07-01 | 2099-12-31 |

- **Val halves:** H1 = 2015-01-01..2017-06-30, H2 = 2017-07-01..2019-12-31, split at the calendar midpoint.
- The `macro.py` note calls this split "frozen 2026-10-06". That claim is not relied on (see PR #3). The file hash below is the pin.
- **Abort** if `results/v3/store/macro/splits.json` differs from these dates. Its SHA-256 is `REQUIRED-BEFORE-RUN`.

## 2. Code pins

| File | Pin |
|---|---|
| repo HEAD (origin/main) | `313b185cac9e53c9d0bfcaf48774b34e79d72d34` |
| `desk/lab/macro.py` (engine, SPLITS), last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `1ce90652e87e09a09ca525bc10b485f3725940658fb40f2c722862ed08ba1a22` |
| `desk/data/macro_data.py` (universe, costs) | SHA-256 `790c2cd194153126e85930105fc9ae09a6f4894e5333049c804d3838045c3eb0` |
| `desk/lab/stats.py` (`dsr`, `block_bootstrap_lower`, `mean_ex_top`) | SHA-256 `8ca1e1b0b0d94a4e2f1d26dca96a24957af8b53c9f6c7ac140519637efd7b51c` |
| FX-carry engine change (§4) | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |
| Strategy file(s) for configs A/B/C | `REQUIRED-BEFORE-RUN` (commit + SHA-256 each) |

**Re-pin rule:** if gating, the engine, stats, or the universe changes, every row must be re-pinned in a commit before the run, and the diff disclosed.

## 3. Disclosure: related test already seen on val
- Long-only crypto TS momentum (20/60/120-day) was evaluated on val already: **dev Sharpe 1.35, val Sharpe −0.22, FAILED.** Max drawdown was 11% vs 44% for buy-and-hold (`RESEARCH_LOG_2026-10-03.md`, row 6).
- How this test differs:
  - It is **long/short**, not long-only.
  - It covers **many asset classes** (equities, bonds, commodities, FX, real estate), not crypto alone.
  - It uses 1–12-month lookbacks instead of 20–120 days.
  - It weights by **equal risk per class**.
  - It runs on the **macro** split (2015–2019 val), not the crypto daily split.
- Overlap: BTC-USD and ETH-USD are in the macro universe (class `crypto`).
- Macro-desk val (2015–2019) is also public history, and the cross-asset trend literature covers it. So this is **not hindsight-free**. Only the forward split is (see the `macro.py` note).

## 4. FX carry requirement
- **Currently missing:** the `desk/lab/macro.py` docstring says "FX spot returns exclude the interest-rate carry", and `macro_data.py` says the same for the `fx` class. Trend on FX spot without carry is biased.
- **Required:** the engine must add the daily carry r_base − r_quote (ACT/360, from short rates known at close t, applied over (t+1, t+2]) to each `fx` row's return. That change has to be added, reviewed by QL Validator, and pinned (§2) **before** the run.
- **Rate-differential source:**
  - US: `^IRX` (13-week T-bill) is already in the panel as a feature series (class `rate`).
  - Non-US (EUR, GBP, JPY, AUD, NZD, CAD, CHF, MXN, BRL, ZAR, TRY, INR, CNY): **not in the repo.**
  - The proposed source, 3-month interbank or T-bill rates such as OECD/FRED `IR3TIB01xxM156N` (monthly, so lag them one month for availability), is `REQUIRED-BEFORE-RUN`. QL Validator must confirm it.
  - If a currency has no reviewed rate series, that pair is **excluded** from all configs. No imputation.
- Currency ETFs (UUP, FX*) already reflect their holdings' interest income and are left as they are.

## 5. Universe (from `desk/data/macro_data.py::UNIVERSE`, tradable == 1 only)
- `equity_us`: SPY QQQ IWM DIA MDY
- `sector`: XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC SMH KRE XBI ITB
- `equity_intl`: EFA EEM VGK EWJ EWG EWU EWC EWA EWZ FXI EWY EWT EWW INDA EWH EWS
- `factor`: IWD IWF MTUM QUAL USMV VLUE SPLV RSP
- `bond`: TLT IEF IEI SHY AGG LQD HYG JNK TIP EMB BWX MUB BIL
- `commodity`: GLD SLV USO UNG DBC DBA DBB PDBC GDX
- `real_estate`: VNQ
- `currency`: UUP FXE FXY FXB FXA FXC FXF
- `fx`: EURUSD=X GBPUSD=X JPY=X AUDUSD=X NZDUSD=X CAD=X CHF=X MXN=X EURJPY=X BRL=X ZAR=X TRY=X INR=X CNY=X
- `crypto`: BTC-USD ETH-USD

**Risk-budget classes (fixed):** these 10 groups are merged into **6 classes**: Equity (`equity_us`, `sector`, `equity_intl`, `factor`), Bonds, Commodities (`commodity`, `real_estate`), FX (`fx`, `currency`), Crypto, and Cash-like. Cash-like is BIL and SHY, which are **excluded** from signals because they are near-zero-vol cash proxies.

**Eligibility:** a symbol enters on the first day it has ≥ 252 + 21 days of history (the strategy's `WARMUP_DAYS` is 300). Inception dates per symbol (e.g. MTUM, XLC, BTC-USD) are **not derivable from code** and are `REQUIRED-BEFORE-RUN`. They must be read from metadata only, never from returns.

**Feature-only series** (indices, vol, rates, futures; `tradable == 0`) are not traded. Only `^IRX` is used, and only for FX carry.

## 6. Configs (at most 3, fixed now)
Common to all three configs:
- **Signal per asset:** s = sign(trailing total return over L) using `adjclose` up to the close of day t. For `fx` rows the return includes carry.
- **Weights:**
  - **Inverse volatility:** raw weight w_i = s_i / σ_i, where σ_i is the 63-day realized daily volatility.
  - **Equal risk per class:** each class's weights are scaled so that its ex-ante volatility (sum of |w_i| σ_i, ignoring correlation) is 1/K of the book target, where K is the number of active classes.
  - **Book volatility target:** 10%/yr ex-ante, then gross is capped at **2** (`MacroCost.max_gross`, default).
- **Rebalance:** monthly, on the last trading day, to limit turnover.
- **Execution:** the engine's next-close rule (`delay_d = 1`). A decision at close t executes at close t+1, and P&L accrues from t+1 to t+2.
- **Costs:** per-symbol `cost_bps` from `macro_data.py` (one-way, including half spread) × `fee_mult`. Short borrow is 1%/yr on short notional, not charged on FX (`MacroCost.short_borrow`).

| Config | Lookback blend |
|---|---|
| A | 12-month (252 d) only |
| B | equal blend of 1/3/12-month (21/63/252 d) signals, averaged into a score in [−1, 1] |
| C | equal blend of 3/12-month (63/252 d), skipping the most recent month |

There is no other tuning. The vol window (63 d), 10% target and monthly rebalance are fixed. They are not chosen on dev results; they are conventional.

## 7. Metrics and pass bar (per config, base costs unless stated)
Metrics: Sharpe (`stats.sharpe`, 252 trading days/yr; the engine annualization is used as given), annualized return, max drawdown, turnover, per-class contribution, and the stress Sharpe.

To **PASS**, a config must meet all of the following:
1. **Val Sharpe ≥ 0.5.**
2. **DSR > 0.95** (`stats.dsr`) with **n_trials = max(6, ledger count)**.
   - **Basis for 6:** this family's 3 configs, plus the 3 lookbacks (20/60/120 d) of the long-only crypto TS momentum already evaluated on val (`RESEARCH_LOG_2026-10-03.md`, row 6), counted as separate trials to be conservative.
   - The ledger count is the number of distinct macro-desk strategies with a val evaluation in `company.db`. It is `REQUIRED-BEFORE-RUN`, and if it is larger, that number is used.
   - `trial_sr_std` comes from the ledger, also `REQUIRED-BEFORE-RUN`.
3. **Both val halves (§1) have positive Sharpe.**
4. **Block-bootstrap 95% lower bound ≥ 0** on the daily mean (`stats.block_bootstrap_lower`, defaults: block 5, n 2000, alpha 0.05, seed 7).
5. **Mean excluding the top 5 days > 0** (`stats.mean_ex_top`, k = 5).
6. **1.5× cost stress** (`MacroCost.stressed(fee_mult=1.5)`): val Sharpe is still > 0 and net P&L is still > 0.

## 8. Reporting rule
- **All 3 configs are reported** with every metric, whether they pass or fail.
- The best of the 3 may be highlighted **only** with its DSR at the n_trials from §7. Its raw Sharpe must never be quoted alone.
- A FAIL is final. There are no added configs, re-runs, or tweaks to parameters or the universe after seeing val. Any change counts as a new trial and a new pre-registration.
- A PASS only makes the strategy a forward-paper candidate. Deployment needs a forward result and human approval.

## 9. Open items (REQUIRED-BEFORE-RUN)
1. The SHA-256 of `results/v3/store/macro/splits.json`, and a match against §1.
2. `provenance.json` showing real data (not the synthetic fallback).
3. The FX-carry engine change, its review and its pin, plus the non-US rate source and its series IDs.
4. The commits and SHA-256 of the strategy files for A, B and C.
5. Symbol inception dates (from metadata only).
6. The ledger's macro val trial count and `trial_sr_std`.
7. Sign-off from QL Validator that eval gating is fixed.
