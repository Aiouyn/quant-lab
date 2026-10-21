# Pre-registration: HL-native cash-and-carry (BTC + HYPE), val re-test

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21.
- **Revision:** r2, addressing QL Validator's review.
- **Desk:** carry.
- **Reviewers:** QL Validator, QL Safety.

> **BLOCKED:** blocked until `desk/lab/stats.py` ships a G7-compliant bootstrap (assigned separately). The pin for that implementation (commit + SHA-256) is `REQUIRED-BEFORE-RUN`. Every bootstrap bound in this document follows the PR #10 spec rule G7.

## 0. Disclosure: this is NOT a clean out-of-sample test
- The carry **val window has already been seen.** Spot cash-and-carry (short HL perp, long Binance spot) was evaluated on carry val on 2026-10-03: +7.1% maker / +5.4% taker (`RESEARCH_LOG_2026-10-03.md`, row 5). The funding regime in val is therefore known to the researchers.
- The always-on BTC/HYPE result in the research log was computed over 2023-06..2026-10, which **includes val and holdout**.
- So a val PASS here can only be a **consistency check**, not confirmation. Confirmation can only come from the forward paper (§9).

## 1. Run conditions (binding)
- This test runs only after eval gating is fixed: the DSR/`n_trials` accounting and the integrity issues from the 2026-10-21 audit must be resolved and signed off by QL Validator.
- It runs **once, on val only, and never on the holdout split.** The carry holdout is burned (`JOURNEY.md` §7.3) and must not be opened.
- **Abort** (no result is recorded as a pass or fail) if any value in §3 or §4 differs from what is recorded here, or if any REQUIRED-BEFORE-RUN placeholder is still unfilled.

## 2. Hypothesis
The always-on, delta-neutral position (long spot, short perp of the same size, in one Hyperliquid portfolio-margin account) earns a funding yield of at least 5%/yr on capital, net of costs, in the carry val window. This is a risk-premium yield, not alpha.

## 3. Data and split pins
Source: the real carry panel built by `desk/data/carry_data.py`, with HL spot/perp prices and hourly funding for BTC and HYPE.

| Item | Value | Status |
|---|---|---|
| `splits.json` path | `results/v3/store/carry/splits.json` (operator store) | known from code |
| `splits.json` SHA-256 | `REQUIRED-BEFORE-RUN` | file not on authoring box |
| Split generator | must be `carry_data.py::_freeze_splits` (note `"carry desk: chronological 60/25/15 over its own panel"`). **Abort** if the note says `"contiguous 365d hourly panel..."`: that is the `store_builder.py` synthetic split (150/110/60), and its panel is synthetic | rule |
| `provenance.json` | must not show `source = deterministic_local_fixture`, and must show `verified_real = true` for carry | `REQUIRED-BEFORE-RUN` |
| Panel N (distinct UTC days) | `REQUIRED-BEFORE-RUN` | panel not on box |
| Panel first day / last day | `REQUIRED-BEFORE-RUN` / `REQUIRED-BEFORE-RUN` | panel not on box |
| val start (`days[floor(0.60·N)]`) | `REQUIRED-BEFORE-RUN` | derived from N |
| val end (`days[floor(0.85·N) − 1]`) | `REQUIRED-BEFORE-RUN` | derived from N |
| Consistency check | the recorded val dates must equal the `val` field in `splits.json` **and** the values recomputed from N with the formula above | rule |

How to fill these placeholders: a reviewer reads **only** the panel's timestamp column (to get N and the first/last day) and the `splits.json` metadata. They must not read any returns, funding or prices in the val or holdout rows. The values are committed to this file in a separate commit **before** the run.

## 4. Evaluator pin
- At the time of writing, no HL-native (spot plus perp, same venue) backtester exists in the repo. `desk/lab/carry.py` models HL perp vs **Binance perp**, and the HL-native result in the research log came from a script that isn't in the repo. The evaluator therefore has to be written, reviewed, and pinned here **before** the run.

| File | Pinned at | SHA-256 |
|---|---|---|
| repo HEAD (main) | `313b185cac9e53c9d0bfcaf48774b34e79d72d34` | — |
| `desk/lab/carry.py` (cost model reference) | last change `7c140052faccaf3e76833ad1c9c711038c2129dc` | `471d8dd1b83d5096240a4ec08f45ea691ef3fcf57b9b8ba46516b4b2ea097ab1` |
| `desk/data/carry_data.py` (split freezing) | same | `e0f7530f530320224877ee7573cd36b8b643330b908087ae834a98ced1be6fb9` |
| `desk/lab/stats.py` (`dsr`) | same | `8ca1e1b0b0d94a4e2f1d26dca96a24957af8b53c9f6c7ac140519637efd7b51c` |
| G7-compliant bootstrap in `desk/lab/stats.py` (not yet shipped) | `REQUIRED-BEFORE-RUN` | `REQUIRED-BEFORE-RUN` |
| HL-native evaluator script | `REQUIRED-BEFORE-RUN` | `REQUIRED-BEFORE-RUN` |


**Re-pin rule:** if gating, stats, or any pinned file changes (including the gating fix in §1), every row must be re-pinned in a commit before the run, and the diff disclosed.

## 5. Rule (fixed, no parameters)
- From the first val hour: for each coin, 1× short perp against 1× long spot of equal notional, split 50/50 BTC/HYPE. There is no regime filter, no exit and no tuning. The position closes at the last val hour.
- **Re-hedge** to equal notional monthly, or whenever |price move since the last hedge| > 40%.
  - *Justification:* this re-hedge rule is copied unchanged from the deployment plan written before any forward result (`RESEARCH_LOG_2026-10-03.md`, "Deployment sizing").
  - At a 15% buffer, the liquidation distance there is +117% (BTC) and +203% (HYPE). The worst historical 30-day run-ups are +63% and +94%.
  - Re-hedging at 40% keeps a run-up between hedges well inside the liquidation distance.
  - It was chosen from liquidation stress, not tuned on returns. Note that the run-up statistics span the full history, including val.
- **HYPE coverage rule:** HYPE spot exists for only part of history. If HYPE has fewer days than the val window, the combined book is evaluated on the days where both coins exist, and BTC alone is also evaluated on the full val window. The pass bar must hold on **both**.
  - *Justification:* there is no prior minimum-days rule in the repo. This rule avoids picking a window and avoids backfilling data. If HYPE covers less than 90 days of val, its result is reported but does not count, and the verdict is based on BTC alone.

## 6. Capital basis
- Capital = **1.15 × notional** (portfolio margin plus a 15% USDC buffer), per the research log's portfolio-margin section, which cites https://hyperliquid.gitbook.io/hyperliquid-docs/trading/portfolio-margin.
- All returns are reported on capital. Interest on idle USDC is excluded.

## 7. Costs
- **Base:** 10 bps of notional at entry and at exit, plus the same on any re-hedge notional. Funding is applied hourly with its sign, plus mark-to-market on both legs.
- *Justification for the spread assumption:* measured HL median spreads are 0.13 bps for both BTC and HYPE (`REALITY_CALIBRATION.md`, "Hyperliquid spreads", BBO 2026-08-09→10-02). Taker fees are 4.5 bps (`REALITY_CALIBRATION.md`; `CarryCost.hl_taker_bps` in `desk/lab/carry.py`). Median taker slippage is 1.3 bps, p90 3.3 bps. So 10 bps per entry or exit covers roughly the taker fee plus p90 slippage on a two-leg trade (spot and perp each ~4.5 + ~0.5 bps).
  - Caveats: the spread sample (Aug–Oct 2026) postdates val, and the portfolio-margin fee tier is assumed to equal the base tier.
- **Stress:** all costs × 1.5.

## 8. Metrics and pass bar (combined book at base cost; also BTC-only per §5)
Metrics: annualized return on capital, return for each calendar quarter in val, max drawdown on capital, share of hours with negative funding, block-bootstrap lower bound, DSR.

To **PASS**, all of the following must hold:
1. Val return on capital is **≥ 5%/yr**.
2. **Every val quarter is ≥ 0.**
3. **Max drawdown < 3%** of capital.
4. **Block-bootstrap 95% lower bound ≥ 0** (one-sided) on the mean daily return, computed per the PR #10 spec rule **G7** (studentized or BCa interval; block size max(5, ceil(T^(1/3))); ≥ 5,000 draws; seed logged in the manifest and the ledger row; passes the null-drill false-positive check).
5. **DSR > 0.95**, using `desk/lab/stats.py::dsr` with **n_trials = max(4, ledger count)**.
   - **Basis for 4:** these are the known evaluations of the funding-carry family on carry val: spot carry v1 maker, spot carry v1 taker (`RESEARCH_LOG_2026-10-03.md`, row 5), carry v2 HL-vs-Binance (it has a paper sleeve and a fixed entry threshold per the regime-monitor section; whether it was evaluated on val is unconfirmed, so it is counted to be conservative), and this test.
   - **Ledger count:** the number of distinct carry-desk strategies in `company.db` with a val evaluation, which is `REQUIRED-BEFORE-RUN`. If it is larger, that number is used.
   - `trial_sr_std` comes from the ledger (`trial_stats`), which is also `REQUIRED-BEFORE-RUN`.

The stress case (1.5× costs) is reported. If it fails conditions 1–3, that is recorded as a caveat. Any failure of conditions 1–5 is a **FAIL**, with no re-run and no variant.

## 9. Forward-paper rule (independent; the only clean evidence)
- `desk.live.hl_carry_paper` started 2026-10-03 (BTC + HYPE, $50k each).
- At **day 30 (around 2026-11-02)**, `results/v3/forward_review.json` applies the rule fixed in advance: **deploy candidate only if ≥ 5%/yr on capital** (1.15× basis).
- Deployment needs both results to pass, plus human approval. Because of §0, the forward result carries the evidential weight.

## 10. Known risks (not tested here)
Hyperliquid venue and contract risk; liquidation path (10-minute TWAP, oracle timing); beta caps; long negative-funding stretches (BTC funding was negative in about 30% of hours in 2026H1).
