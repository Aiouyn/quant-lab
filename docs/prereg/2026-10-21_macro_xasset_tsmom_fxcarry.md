# Pre-registration: macro cross-asset time-series momentum with FX carry

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21 from code and docs only; no data files were read.
- **Desk:** macro.
- **Reviewers:** QL Validator, QL Safety.

> **BLOCKED:** blocked until `desk/lab/stats.py` ships a G7-compliant bootstrap (assigned separately). The pin for that implementation (commit + SHA-256) is `REQUIRED-BEFORE-RUN`. Every bootstrap bound in this document follows the PR #10 spec rule G7.

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

- **Val halves (exact rule):** let D be the ordered list of evaluation dates in val (NYSE trading days, see §1a), with T = len(D).
  - H1 = D[0 : floor(T/2)], H2 = D[floor(T/2) : T].
  - Each half's Sharpe is computed on its own daily P&L, annualized with √252.
  - The calendar midpoint (2017-06-30/07-01) is roughly where this split falls, but the trading-day count rule above is what decides.

## 1a. Calendar and annualization (fixed)
- **The problem:** macro panel rows are calendar days whenever any symbol trades. BTC-USD and ETH-USD trade 7 days a week, so weekend rows appear. `summarize` in `desk/lab/macro.py` annualizes with 252 (`sharpe(daily, periods=252)`, `years = len(daily)/252`). That is wrong if weekend days are in the daily series.
- **Choice: drop crypto and evaluate on trading days only, at 252.**
  - BTC-USD and ETH-USD are **removed from the universe** for this test.
  - The evaluation calendar is **NYSE trading days**, defined as dates on which SPY has a row.
  - Rows on other dates are dropped before simulation. A return over a gap is the close-to-close return between consecutive calendar dates, and FX carry accrues ACT/360 over the gap.
  - Every annualized figure (Sharpe, return, vol) uses **252**, and that applies everywhere in this document.
- **Engineering requirement:** the evaluator must enforce this calendar and **assert** that no evaluation date falls on a weekend or a non-SPY date. This is item (7) in `docs/specs/research-data-and-evaluators.md` (PR #10). It is `REQUIRED-BEFORE-RUN`.
- The `macro.py` note calls this split "frozen 2026-10-06". That claim is not relied on (see PR #3). The file hash below is the pin.
- **Abort** if `results/v3/store/macro/splits.json` differs from these dates. Its SHA-256 is `REQUIRED-BEFORE-RUN`.

## 2. Code pins

| File | Pin |
|---|---|
| repo HEAD (origin/main) | `313b185cac9e53c9d0bfcaf48774b34e79d72d34` |
| `desk/lab/macro.py` (engine, SPLITS), last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `1ce90652e87e09a09ca525bc10b485f3725940658fb40f2c722862ed08ba1a22` |
| `desk/data/macro_data.py` (universe, costs) | SHA-256 `790c2cd194153126e85930105fc9ae09a6f4894e5333049c804d3838045c3eb0` |
| `desk/lab/stats.py` (`dsr`, `mean_ex_top`) | SHA-256 `8ca1e1b0b0d94a4e2f1d26dca96a24957af8b53c9f6c7ac140519637efd7b51c` |
| G7-compliant bootstrap in `desk/lab/stats.py` (not yet shipped) | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |
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
- **Overlap removed:** BTC-USD and ETH-USD are excluded from this test (§1a), so it has no crypto exposure.
- Macro-desk val (2015–2019) is also public history, and the cross-asset trend literature covers it. So this is **not hindsight-free**. Only the forward split is (see the `macro.py` note).

## 4. FX carry requirement
- **Currently missing:** the `desk/lab/macro.py` docstring says "FX spot returns exclude the interest-rate carry", and `macro_data.py` says the same for the `fx` class. Trend on FX spot without carry is biased.
- **Required:** the engine must add the daily carry r_base − r_quote (ACT/360, from short rates known at close t, applied over (t+1, t+2]) to each `fx` row's return. That change has to be added, reviewed by QL Validator, and pinned (§2) **before** the run.
- **Rate-differential source (series IDs fixed now).** Checked on public FRED catalog pages on 2026-10-21. "Verified" means the catalog page exists and its stated date range covers 2014-01..2019-12 (dev warm-up plus val). No observations were downloaded. All series are monthly OECD MEI 3-month interbank rates (monthly averages).

- **Availability lag (fixed now):**
  1. **Preferred:** use each observation's **first publication date** from ALFRED vintages (https://alfred.stlouisfed.org, same series IDs, `realtime_start` of the first vintage containing that month's value). The value becomes usable from the **first trading day after** that publication date.
  2. **Fallback, if ALFRED has no vintage for a series or month:** the value for month m is usable from the **first NYSE trading day of month m+2**.
     - *Why m+2:* a monthly average for m is only complete at the end of m, and OECD MEI releases typically come weeks later. Using it from the start of m+1 would leak the average of a month not yet published. The m+2 rule is conservative.
  3. **Between updates:** the last usable value is carried forward (step-wise, no interpolation). Revisions published later are never applied backwards.
  - Which rule was used for each series and month is logged. Whether ALFRED vintages exist for these series is **UNVERIFIED** (not checked; no data read).
  - `^IRX` is daily, so it is usable at the close of its own date.

| Ccy | Series | Status | Pairs using it |
|---|---|---|---|
| USD | `^IRX` (13-week T-bill; already in panel) | in code (`macro_data.py`) | all USD pairs |
| EUR | `IR3TIB01EZM156N` (Euro Area 19) | VERIFIED (FRED page; 1994-01..2026-01) | EURUSD=X, EURJPY=X |
| GBP | `IR3TIB01GBM156N` | VERIFIED (1957-01..2026-01) | GBPUSD=X |
| JPY | `IR3TIB01JPM156N` | VERIFIED (2002-04..2026-08) | JPY=X, EURJPY=X |
| AUD | `IR3TIB01AUM156N` | VERIFIED (1968-01..2026-08) | AUDUSD=X |
| NZD | `IR3TIB01NZM156N` | VERIFIED (1973-12..2026-09) | NZDUSD=X |
| CAD | `IR3TIB01CAM156N` | VERIFIED (1956-01..2026-09) | CAD=X |
| CHF | `IR3TIB01CHM156N` | VERIFIED (1999-07..2026-09) | CHF=X |
| MXN | `IR3TIB01MXM156N` | VERIFIED (1997-01..2026-09) | MXN=X |
| ZAR | `IR3TIB01ZAM156N` | VERIFIED (1980-12..2026-09) | ZAR=X |
| CNY | `IR3TIB01CNM156N` | VERIFIED (1997-06..2026-08) | CNY=X |
| TRY | `IR3TIB01TRM156N` | exists, but the series **ends 2008-04** | TRY=X → **DROPPED** |
| BRL | `IR3TIB01BRM156N` (candidate) | **UNVERIFIED**: the FRED page returned 404 | BRL=X → **DROPPED** |
| INR | `IR3TIB01INM156N` (candidate) | **UNVERIFIED**: the FRED page returned 404 | INR=X → **DROPPED** |

- **Pairs dropped now:** TRY=X, BRL=X and INR=X. That leaves **11 FX pairs**. These pairs come back only through a new pre-registration.
- **Basis caveats:**
  - `^IRX` is a T-bill yield, while the non-US series are interbank rates. The resulting basis mismatch (roughly a TED-spread size) is disclosed, not corrected.
  - The euro-area series uses a fixed 19-country composition. That is fine for 2014–2019.
- **Sign convention:** `JPY=X`, `CAD=X`, `CHF=X`, `MXN=X`, `ZAR=X` and `CNY=X` are quoted as USD/XXX (the price of 1 USD), so carry = r_USD − r_XXX. EURUSD-style pairs use carry = r_base − r_USD. EURJPY=X uses r_EUR − r_JPY.
- **No imputation:** if any monthly value is missing at run time, that pair is excluded from that date onward and the exclusion is logged.
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
- `fx` (11 pairs): EURUSD=X GBPUSD=X JPY=X AUDUSD=X NZDUSD=X CAD=X CHF=X MXN=X EURJPY=X ZAR=X CNY=X
- ~~`crypto`: BTC-USD ETH-USD~~ **excluded** (§1a)
- TRY=X, BRL=X and INR=X are not in the universe (no verified rate series; §4)

**Risk-budget classes (fixed):** these 10 groups are merged into **5 classes**: Equity (`equity_us`, `sector`, `equity_intl`, `factor`), Bonds, Commodities (`commodity`, `real_estate`), FX (`fx`, `currency`), and Cash-like. Crypto is dropped. Cash-like is BIL and SHY, which are **excluded** from signals because they are near-zero-vol cash proxies.

**Eligibility and inception dates:**
- A symbol may receive a signal on date t only if it has ≥ 273 trading days (252 + 21) of `adjclose` history before t. The strategy's `WARMUP_DAYS` is 300.
- Inception dates are **not available in code or docs**, so each symbol's first-listing date is `REQUIRED-BEFORE-RUN`.
  - It is taken from issuer or exchange metadata (fund fact sheet, or Yahoo `firstTradeDate` from the chart API's `meta`). It must never be inferred from the returns series.
  - The dates are committed in a table in this file, with the source per symbol, before the run.
- **Abort rule:** the run aborts if any symbol in the §5 universe lacks a committed inception date, **or** if a symbol's first panel row is earlier than its committed inception date. The latter indicates a backfilled or proxy series.
- A symbol with inception inside val enters only after it has 273 trading days of history (counted as panel rows, not returns).

**Feature-only series** (indices, vol, rates, futures; `tradable == 0`) are not traded. Only `^IRX` is used, and only for FX carry.

## 6. Configs (at most 3, fixed now)
Common to all three configs:
- **Lookback returns:** r_i(a, b) = adjclose_i[t−b] / adjclose_i[t−a] − 1, using trading-day offsets on the §1a calendar and data up to the close of day t. For `fx` rows the return includes accrued carry.
- **Signal score per asset (fixed per config):**
  - **A:** score = sign(r(252, 0)).
  - **B:** score = [sign(r(21, 0)) + sign(r(63, 0)) + sign(r(252, 0))] / 3. The possible values are {−1, −⅓, ⅓, 1}.
  - **C:** score = [sign(r(63, 21)) + sign(r(252, 21))] / 2, i.e. skipping the most recent 21 days. The possible values are {−1, 0, 1}.
- **Weights (the same for all configs):**
  1. **Inverse vol:** raw_i = score_i / σ_i, where σ_i is the std of the last 63 daily returns (trading days), floored at 1e-4. For B, the blend happens in the score, **before** dividing by volatility; there is one σ per asset, not one per lookback.
  2. **Equal risk per class:** for each active class k (one with at least one non-zero raw weight), scale its raw weights so that Σ_{i∈k} |w_i| σ_i √252 = 10% / K, where K is the number of active classes. Correlation is ignored.
  3. **Gross cap:** if Σ|w| > 2, scale every weight by 2/Σ|w| (`MacroCost.max_gross` = 2).
- **Rebalance:** on the last NYSE trading day of each month. Weights are held in between.
- **Execution:** next close (`delay_d = 1`).
- **Costs:** per-symbol `cost_bps` (`macro_data.py`) × `fee_mult`, plus short borrow of 1%/yr (not on FX).

| Config | Score |
|---|---|
| A | 12-month |
| B | equal blend of 1/3/12-month signs |
| C | 3/12-month signs, skipping 1 month |

The design choices fixed here are the 63-day vol window, the 10% target, monthly rebalance and the class map. They are counted as implicit trials in §7.

## 7. Metrics and pass bar (per config, base costs unless stated)
Metrics: Sharpe (`stats.sharpe(daily, periods=252)` on the §1a trading-day calendar), annualized return, max drawdown, turnover, per-class contribution, and the stress Sharpe.

To **PASS**, a config must meet all of the following:
1. **Val Sharpe ≥ 0.5.**
2. **DSR > 0.95** (`stats.dsr`) with **n_trials = max(51, ledger trials for this family)**. The ledger figure is a **hard floor**: it can only raise N.
   - **Basis for 51:** 3 lookback configs × 2 vol windows (21 / 63 d) × 2 vol targets (10% / 15%) × 2 rebalance frequencies (weekly / monthly) × 2 class maps (5 classes as in §5, or 1 class per `macro_data.py` `cls`) = **48** implied forks, plus **3** for the long-only crypto TS momentum lookbacks already run on val (`RESEARCH_LOG_2026-10-03.md`, row 6).
   - Only one fork is evaluated. The others are counted because a researcher could plausibly have chosen them, so this is a conservative upper bound.
   - The vol target barely changes Sharpe while the gross cap isn't binding. It is still counted, to be conservative.
   - **Ledger floor (R9, per Validator charter, pending):** the number of **trials** (ledger rows, i.e. evaluations with distinct code + config hash; not distinct strategies) recorded for this family (macro cross-asset TS momentum) on the macro desk, across splits. It comes from the R7a schema query in PR #10 P0-C, and it is `REQUIRED-BEFORE-RUN`.
   - `trial_sr_std` comes from the ledger, also `REQUIRED-BEFORE-RUN`.
3. **Both val halves (§1, trading-day count split) have positive Sharpe.**
4. **Block-bootstrap 95% lower bound ≥ 0** (one-sided) on the daily mean, computed per the PR #10 spec rule **G7** (studentized or BCa interval; block size max(5, ceil(T^(1/3))); ≥ 5,000 draws; seed logged in the manifest and the ledger row; passes the null-drill false-positive check).
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
3. The FX-carry engine change: review and pin. The series IDs are fixed in §4.
4. The trading-day calendar enforcement and its assertion (§1a; PR #10 item 7).
5. The commits and SHA-256 of the strategy files for A, B and C.
6. The inception-date table with a source per symbol (§5), with the abort rule.
7. The ledger's trial count for this family per R9 (hard floor) and `trial_sr_std`.
8. Sign-off from QL Validator that eval gating is fixed.

**UNVERIFIED:** `IR3TIB01BRM156N` and `IR3TIB01INM156N` (the FRED pages returned 404). Both pairs are dropped regardless.
