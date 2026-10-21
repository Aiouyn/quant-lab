# Pre-registration: Polymarket favorite-longshot calibration and trading sleeve

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21 from code and docs only; no data files were read.
- **Desk:** pred.
- **Reviewers:** QL Validator, QL Safety.

## 0. Run conditions (binding)
- This test runs **once, on val only, and never on holdout.** The holdout is suspect (`JOURNEY.md` §5.1/§7.3, PR #3).
- It runs only after both of these are signed off by QL Validator:
  1. **Eval gating is fixed:** the DSR/`n_trials` accounting from the 2026-10-21 audit.
  2. **Store provenance is fixed:** `results/v3/store/pred/` must come from `desk/data/polymarket.py` (real Polymarket APIs), **not** from the `store_builder.py` synthetic fallback. That fallback generates staggered fake markets with seeded outcomes; if `provenance.json` has `verified_real = false`, the run aborts.
- **Abort** if any pin in §1 or §2 differs at run time, or if any REQUIRED-BEFORE-RUN item is still open.

## 1. Split (copied literally from `desk/lab/pred.py::SPLITS`)

| Split | Start | End (inclusive) |
|---|---|---|
| dev | 2020-01-01 | 2026-02-28 |
| **val** | **2026-03-01** | **2026-05-31** |
| holdout (never opened) | 2026-06-01 | 2026-07-31 |
| forward | 2026-08-01 | 2099-12-31 |

**What the code does.** The note in `pred.py` says the cut-offs were chosen "by market-count share (~57/22/19/rest)". The split itself is applied **by row timestamp (`ts`)**, not by market:
- `PredStore.range_ms` maps a split to [start 00:00 UTC, end + 1 day).
- `PredStore.load` keeps rows with `ts` in that range **and** `end_ms < latest`, where `latest` is the end of the latest allowed split. Only markets whose scheduled end falls before the window closes can be scored.
- `end_ms` is the market's **scheduled end date** (`endDate`, falling back to `closedTime`; see `polymarket.py::fetch_markets`). It is not the UMA resolution timestamp.
- A market whose life spans a cut-off therefore contributes price rows to two splits.

**Rule used in this test (stricter than the engine, and fixed now).** A market belongs to val if and only if **both** its `end_ms` **and** its snapshot time (§3) fall inside [2026-03-01 00:00, 2026-06-01 00:00) UTC. Markets that end in val but have their snapshot before 2026-03-01 are dropped and counted.

The `pred.py` note calls the split "frozen 2026-10-06". That claim is not relied on (PR #3). The file hash is the pin.

## 2. Code pins

| Item | Pin |
|---|---|
| repo HEAD (origin/main) | `313b185cac9e53c9d0bfcaf48774b34e79d72d34` |
| `desk/lab/pred.py`, last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `df51ddfc82ac037458e1212b77d19067ae5e473e6e6da2150cead6cf3b8498a3` |
| `desk/data/polymarket.py`, last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `ed66be282113d4d12aa1394cc023b1995345aa5184463ce7570309a923345872` |
| `desk/lab/stats.py` (`dsr`, `block_bootstrap_lower`, `mean_ex_top`) | SHA-256 `8ca1e1b0b0d94a4e2f1d26dca96a24957af8b53c9f6c7ac140519637efd7b51c` |
| `results/v3/store/pred/splits.json` | SHA-256 `REQUIRED-BEFORE-RUN` (must equal §1) |
| Calibration evaluator (bucket and event-clustered bootstrap; not in repo) | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |
| Sleeve strategy file | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |

**Re-pin rule:** if gating, the engine, stats, or the data fetcher changes, every row must be re-pinned in a commit before the run, and the diff disclosed.

## 3. Price snapshot and buckets (fixed now; not tuned)
- **Snapshot:** the YES price `p` from the last panel row with `ts ≤ end_ms − 24 h`.
  - That row must be within 6 h of `end_ms − 24 h`. Otherwise the market is dropped as stale and counted.
  - Panel prices are hourly (`fidelity=60`), and each hour keeps the last print (`polymarket.py::build`).
- **Bucket edges, in cents of YES price, half-open [lo, hi) with the last bucket closed:**
  `0–2, 2–5, 5–10, 10–20, 20–30, 30–40, 40–50, 50–60, 60–70, 70–80, 80–90, 90–95, 95–98, 98–100`. That is **14 buckets**.
- These edges are frozen by this document. Dev is used **only** to produce a sanity calibration plot, checking that the pipeline runs and that the plot looks sane. The edges, the snapshot horizon and every other parameter **do not change** after dev is looked at.

## 4. Universe
- **Source:** `polymarket.py::fetch_markets` pages through gamma-api `/markets/keyset` with `closed=true`. It keeps markets with exactly **2 outcomes, 2 prices and 2 tokens**, and a final YES price of exactly 0 or 1.
  - The YES side is the outcome labeled "yes". If there is none, the first outcome is treated as YES.
- **Exclusions already in the code:**
  - Voided, 50-50 or unresolved markets (final price not in {0, 1}).
  - Multi-outcome markets with more than 2 outcomes.
  - Lifetime volume under $250k (`MIN_VOLUME`).
- **Additional exclusions (fixed now):**
  1. Markets whose outcomes are not literally Yes/No, for example "Team A vs Team B". With those, "YES" is arbitrary, which breaks the favorite-longshot reading. The outcome labels are not stored in the panel, so this needs a re-fetch or the raw labels: `REQUIRED-BEFORE-RUN`.
  2. Markets with a disputed or ambiguous resolution (UMA dispute or clarification). No field for this exists in the code: `REQUIRED-BEFORE-RUN` (a source must be identified, or the exclusion is waived with that disclosed).
- **Multi-outcome events (neg-risk groups):** each candidate in a group like "Who will win X?" arrives as a separate binary market. They are **kept**, but they are **clustered by event** in every bootstrap (§5).
  - The event or group id (`events[].id` / `negRiskMarketID`) is **not stored** by the current fetcher: `REQUIRED-BEFORE-RUN`.
  - If it can't be obtained, the fallback is to cluster by `end_ms` date and `category`, which is a cruder proxy. The run must disclose that.
- **Survivorship and selection check (from the fetch logic):**
  - **Volume look-ahead:** `MIN_VOLUME` filters on **final** lifetime volume, which is a look-ahead selection on popularity (`pred.py` docstring). Popular markets may be better calibrated, so this biases toward finding *no* effect. It is disclosed, not corrected.
  - **Incremental-refresh gap:** the cursor checkpoint "may miss a market created long ago that closes later" (`fetch_markets` docstring). The val universe must come from a **full, non-incremental** re-page (`REQUIRED-BEFORE-RUN`: record the page count and market count).
  - **Archived or delisted markets:** whether gamma `closed=true` returns archived or delisted markets is **unknown** from the code: `REQUIRED-BEFORE-RUN` (check the API docs or compare against `archived=true`).
  - **Missing price history:** markets with no price history from `fetch_prices` are silently missing from the panel. The run must report counts (resolved in val ≥ $250k, against those with a snapshot). If more than 10% are missing, the run aborts.
- **Liquidity/volume floor (fixed now):** lifetime volume of at least $250k (as in the code), **and** at least 12 non-empty hourly prints in the 48 h before the snapshot. Volume at snapshot time is not available, so this print count is the liquidity proxy.

## 5. Calibration test (val)
- **Per bucket b:** miscalibration m_b = (realized YES rate) − (mean snapshot price), with each market weighted equally.
- **CI:** 95% percentile CI from a **bootstrap clustered by event**: resample events with replacement, 10,000 draws, seed 7.
- **Bucket passes** if |m_b| > **2 cents** (2 × the 1-cent half-spread in `PredCost.half_spread`; the uncapped value is used, which is conservative near 0 and 1) **and** the CI excludes 0.
- **Multiple testing:** each of the 14 buckets is one trial (§6). At least 30 markets **and** at least 10 events are required per bucket. Buckets below that are reported as "insufficient" and count as not passed.
- The calibration result is a **finding**, not a trading licence. The sleeve is judged separately (§6).

## 6. Trading sleeve (val)
- **Rule (fixed buckets; does not depend on which buckets pass §5):**
  - **Sell longshots:** at the snapshot, hold **NO** (weight −w) on markets priced in 0–2, 2–5 or 5–10 cents.
  - **Buy favorites:** hold **YES** (+w) on markets priced in 90–95, 95–98 or 98–100 cents.
  - Hold to resolution. Each position gets equal weight w, with gross capped at `PredCost.max_gross` = 1 across open positions (engine re-scaling).
  - Targets stay constant once entered, as required by `run_pred`.
- **Costs:** `PredCost` defaults: half-spread 1 cent (capped at half the distance to 0/1, minimum 0.1 cent), fill at the next observed price (`delay_rows = 1`), $100k unit notional.
- **Stresses (both reported; both must pass conditions 2–4 below):** `stressed(fee_mult=1.5)` for 1.5× cost, and `stressed(fee_mult=1.0, delay_rows=1)` for one extra row of delay (two rows in total).
- **Pass bar (base costs, val):**
  1. **DSR > 0.95** (`stats.dsr` on daily P&L) with **n_trials = max(15, ledger count)**.
     - **Basis for 15:** the 14 bucket tests in §5, which are looks at the same val outcomes, plus 1 sleeve configuration.
     - No other pred-desk strategy is known to have been evaluated on val (`RESEARCH_LOG_2026-10-03.md` lists none). The ledger count of distinct pred strategies with a val evaluation in `company.db` is `REQUIRED-BEFORE-RUN`, and if it is larger, that number is used.
     - `trial_sr_std` comes from the ledger: `REQUIRED-BEFORE-RUN`.
  2. **Mean excluding the top 5 days > 0** (`stats.mean_ex_top`, k = 5).
  3. **Block-bootstrap 95% lower bound ≥ 0** on daily P&L (`stats.block_bootstrap_lower`, defaults: block 5, n 2000, alpha 0.05, seed 7).
  4. **Net P&L > 0**, and both val halves (2026-03-01..04-15, 2026-04-16..05-31) have positive net P&L.

  Also required: event-clustered bootstrap of the sleeve's per-market P&L, with the 95% lower bound ≥ 0, so that a single large event can't carry the result.
- **Reporting:** all 14 bucket results and the sleeve are reported, with every metric and each stress. Nothing is selected afterwards. A FAIL is final; any change counts as a new trial and a new pre-registration. A PASS only makes the strategy a forward-paper candidate. Deployment needs forward results and human approval.

## 7. Disclosure: costs are likely optimistic
- The 1-cent half-spread (`PredCost.half_spread`) is a flat assumption. For thin markets, and for extreme prices near 0 or 1 where books are one-sided, the real spread and depth are likely worse.
- Fill at "the next observed price ~1 h later" is not a guaranteed fill.
- The 1.5× stress may not be enough. If an order-book snapshot source becomes available, re-checking costs against it should come **before** any forward paper.
- Capital is locked until resolution. Opportunity cost and the risk of resolution disputes are not modelled.

## 8. Open items (REQUIRED-BEFORE-RUN)
1. The SHA-256 of `results/v3/store/pred/splits.json`, and a match against §1.
2. `provenance.json` showing real Polymarket data (not the synthetic fallback).
3. A full, non-incremental market re-page, with page and market counts recorded.
4. Whether `closed=true` includes archived or delisted markets.
5. Outcome labels, to exclude non-Yes/No binaries.
6. A source for disputed or ambiguous resolutions, or a disclosed waiver.
7. The event or neg-risk group id per market, for clustering.
8. Evaluator and sleeve strategy commits, with SHA-256 hashes.
9. The ledger's pred val trial count and `trial_sr_std`.
10. Sign-off from QL Validator that eval gating is fixed.
