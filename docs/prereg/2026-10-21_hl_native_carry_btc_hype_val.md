# Pre-registration: HL-native cash-and-carry (BTC + HYPE), val re-test

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21.
- **Desk:** carry.
- **Reviewers:** QL Validator, QL Safety.

## Run conditions (binding)
- This test runs **only after eval gating is fixed**, meaning the DSR/`n_trials` accounting and integrity issues raised in the 2026-10-21 audit have been resolved and signed off by QL Validator.
- It runs **once, on val only. It never runs on the holdout split.** The carry holdout must not be opened for this rule or any variant of it.
- This document is frozen once merged. Any change to the rule, data, costs or bar after that point counts as a new trial and must be disclosed.

## Hypothesis
The always-on, delta-neutral position (long spot, short perp of the same size, in one Hyperliquid portfolio-margin account) earns a positive funding yield net of costs. That yield is at least 5%/yr on capital out of sample. This is a risk-premium yield, not alpha.

## Data
- Hyperliquid hourly funding for BTC and HYPE perps, and Hyperliquid spot/perp prices (BBO mid where available), from the carry panel (`desk/data/carry_data.py`).
- **Exact val window:** the `val` field (`[start, end]`, UTC dates, inclusive) of the frozen carry split `results/v3/store/carry/splits.json`. That split is produced by `_freeze_splits` in `desk/data/carry_data.py`: chronological 60/25/15 over the panel's days, with val = days[⌊0.60·N⌋ … ⌊0.85·N⌋−1].
  - The file lives on the operator's store and was not available on the authoring box, so the literal dates are not copied here.
  - Before the run, the evaluator records the `val` dates and the file's SHA-256 in the run log. If the file has been re-frozen since 2026-10-03, the run is aborted.
- HYPE spot history starts later than BTC. If a coin is missing for part of val, the evaluator uses only the days where it exists and reports how many days were covered. No backfill.

## Rule (fixed, no parameters)
- From the first val hour, hold for each coin 1× short perp against 1× long spot of equal notional, split 50/50 BTC/HYPE.
- Re-hedge to equal notional monthly, or whenever |price move since the last hedge| > 40%, as in the research log's deployment plan. Each re-hedge is charged at the entry/exit cost on the traded notional.
- There is no regime filter, no exit rule and no tuning. The position is closed at the last val hour.

## Capital basis
- Capital = **1.15 × notional**: portfolio margin plus a 15% USDC buffer.
- Every return is reported **on capital**, not on gross notional.
- Interest on idle USDC is **excluded** (conservative).

## Costs
- **Base case:** 10 bps of notional at entry and 10 bps at exit, plus re-hedge costs. Funding is applied hourly with its sign; spot and perp mark-to-market are included.
- **Stress case:** all costs × 1.5.

## Metrics (reported for base and stress)
- Annualized return on capital.
- Return for each calendar quarter in val (partial quarters at the ends reported as they are).
- Max drawdown on capital.
- Share of hours with negative funding, per coin.
- Results per coin and combined.

## Pass bar (all must hold, on the combined book, base costs; stress reported)
1. Val annualized return on capital is **≥ 5%/yr**.
2. **Every val quarter is ≥ 0.**
3. **Max drawdown < 3%** of capital.
4. Stress case (1.5× costs): reported. If it fails conditions 1–3, that is recorded as a caveat in the verdict.

If any of conditions 1–3 fails, the verdict is **FAIL**. There is no re-run and no variant.

## Forward-paper rule (independent, already running)
- `desk.live.hl_carry_paper` started 2026-10-03 (BTC + HYPE, $50k each).
- At **day 30 (around 2026-11-02)**, `results/v3/forward_review.json` applies the rule fixed in advance: **deploy candidate only if ≥ 5%/yr on capital** (1.15× basis).
- Deployment needs **both** a val PASS under this document **and** a forward-paper PASS, plus human approval. Neither result alone is enough.

## Known risks (not tested here)
Hyperliquid venue and contract risk; liquidation path (10-minute TWAP, oracle timing); beta caps on portfolio margin; long stretches of negative funding (BTC funding was negative in about 30% of hours in 2026H1).
