# Pre-registration: Polymarket favorite-longshot calibration and trading sleeve

- **Status:** PRE-REGISTERED, NOT RUN. Written 2026-10-21 from code and docs only; no data files were read.
- **Desk:** pred.
- **Reviewers:** QL Validator, QL Safety.

> **BLOCKED:** blocked until `desk/lab/stats.py` ships a G7-compliant bootstrap (assigned separately). The pin for that implementation (commit + SHA-256) is `REQUIRED-BEFORE-RUN`. Every bootstrap bound in this document follows the PR #10 spec rule G7.

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

**Rule used in this test (stricter than the engine, and fixed now).**
- **Anchor time:** A = min(`endDate`, `closedTime`, `umaEndDate`), all in UTC. The gamma field names are VERIFIED (Appendix V1); `umaEndDate` is the resolution timestamp, but that meaning is inferred (V2). `closedTime` must be parsed from its non-ISO format. The current fetcher **doesn't store** `closedTime` or `umaEndDate` (spec PR #10, item 3): `REQUIRED-BEFORE-RUN`.
- **Early-closed markets:** markets with closedTime < endDate are **dropped and counted**. They are typically markets resolved early, where the scheduled end isn't the real end.
- **Val membership:** a market belongs to val if and only if **both** A **and** its snapshot time (A − 24 h, §3) fall inside [2026-03-01 00:00, 2026-06-01 00:00) UTC. Markets with A in val but the snapshot before 2026-03-01 are dropped and counted.
- **Sleeve:** the trading sleeve (§6) trades **only** markets selected by this rule, from the start of val.

The `pred.py` note calls the split "frozen 2026-10-06". That claim is not relied on (PR #3). The file hash is the pin.

## 2. Code pins

| Item | Pin |
|---|---|
| repo HEAD (origin/main) | `313b185cac9e53c9d0bfcaf48774b34e79d72d34` |
| `desk/lab/pred.py`, last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `df51ddfc82ac037458e1212b77d19067ae5e473e6e6da2150cead6cf3b8498a3` |
| `desk/data/polymarket.py`, last changed `7c140052faccaf3e76833ad1c9c711038c2129dc` | SHA-256 `ed66be282113d4d12aa1394cc023b1995345aa5184463ce7570309a923345872` |
| `desk/lab/stats.py` (`dsr`, `mean_ex_top`) | SHA-256 `8ca1e1b0b0d94a4e2f1d26dca96a24957af8b53c9f6c7ac140519637efd7b51c` |
| G7-compliant bootstrap in `desk/lab/stats.py` (not yet shipped) | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |
| `results/v3/store/pred/splits.json` | SHA-256 `REQUIRED-BEFORE-RUN` (must equal §1) |
| Calibration evaluator (bucket and event-clustered bootstrap; not in repo) | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |
| Sleeve strategy file | `REQUIRED-BEFORE-RUN` (commit + SHA-256) |

**Re-pin rule:** if gating, the engine, stats, or the data fetcher changes, every row must be re-pinned in a commit before the run, and the diff disclosed.

## 3. Price snapshot and buckets (fixed now; not tuned)
- **Snapshot:** the YES price `p` from the last panel row with `ts ≤ A − 24 h`, where A is the anchor from §1.
  - That row must be within 6 h of A − 24 h. Otherwise the market is dropped as stale and counted.
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
  2. Markets with a disputed or ambiguous resolution. The source is gamma `umaResolutionStatuses` (field VERIFIED, V12): a market is excluded if its history contains more than one `proposed` entry, or any entry other than `proposed` or `resolved`. The exact dispute token is UNVERIFIED, so this rule is written to catch any extra state. The on-chain UMA source is UNVERIFIED (V13). Storing `umaResolutionStatuses` is `REQUIRED-BEFORE-RUN` (PR #10 item 5).
- **Multi-outcome events (neg-risk groups):** each candidate in a group like "Who will win X?" arrives as a separate binary market. They are **kept** and clustered as follows.
- **Clustering unit (fixed now):**
  1. **Primary:** the gamma event id (`events[0].id`, VERIFIED, V3).
  2. **Fallback 1:** if there is no event id, `negRiskMarketID`.
  3. **Fallback 2:** if neither exists, the pair (`category`, UTC date of A).
  - Neither id is stored by the current fetcher (`REQUIRED-BEFORE-RUN`; PR #10 item 4). The field names are VERIFIED (V1, V3).
  - If more than 20% of val markets need Fallback 2, the calibration result is labeled **exploratory** and can't PASS.
- **Survivorship and selection check (from the fetch logic):**
  - **Volume look-ahead:** `MIN_VOLUME` filters on **final** lifetime volume, which is a look-ahead selection on popularity (`pred.py` docstring). Popular markets may be better calibrated, so this biases toward finding *no* effect. It is disclosed, not corrected.
  - **Incremental-refresh gap:** the cursor checkpoint "may miss a market created long ago that closes later" (`fetch_markets` docstring). The val universe must come from a **full, non-incremental** re-page (`REQUIRED-BEFORE-RUN`: record the page count and market count).
  - **Archived or delisted markets:** still **UNVERIFIED**; the `archived` filter appears to be ignored (V5). The re-page must report the count of `archived=true` among returned markets. If it is 0, the doc discloses that archived coverage is unknown.
  - **Missing price history:** markets with no price history from `fetch_prices` are silently missing from the panel. The run must report counts (resolved in val ≥ $250k, against those with a snapshot). If more than 10% are missing, the run aborts.
- **Liquidity floor (fixed now):**
  - Lifetime volume of at least $250k (as in the code), **and**
  - at least **20 trades in the 7 days before the snapshot** (a trade-count floor).
  - Trades are **not stored** by the current fetcher. The source is the data-api `GET /trades`, with `market=<conditionId>`, `start`/`end` set to the window, and `takerOnly=true` (the default; it counts taker fills). The endpoint and its params are VERIFIED (V6), and it returns March 2026 trades (V7).
  - Because offsets are capped at 10,000, the count must be paged inside `start`/`end` windows. Fetching is in PR #10 item 3: `REQUIRED-BEFORE-RUN`.
  - If no trade source is verified, the test can't run as pre-registered. There is **no** fallback to price-print counts.

## 5. Calibration test (val)
- **Per bucket b:** miscalibration m_b = (realized YES rate) − (mean snapshot price), with each market weighted equally.
- **CI:** a two-sided 95% CI from a **cluster bootstrap** (clustering unit from §4) in the **G7 clustered form**: whole clusters are resampled with replacement, the interval is studentized or BCa (no percentile intervals), with ≥ 5,000 draws, the seed logged, and the clustered null-drill false-positive check (calibrated Bernoulli(p) outcomes with the cluster structure kept) passed. The two-sided p-value p_b is the bootstrap-t p-value for H0: m_b = 0, floored at 1/(number of draws).
- **Threshold:** τ_b = 2 × the base half-spread for that bucket (§7). That is 2 cents in the middle buckets (10–90 cents) and 3 cents in the tail buckets (0–10 and 90–100 cents).
- **Multiple testing: Holm–Bonferroni** across the 14 buckets at family α = 0.05.
  - Sort the p-values in ascending order as p_(1) ≤ … ≤ p_(14), and reject H0_(k) while p_(k) ≤ 0.05 / (14 − k + 1). Stop at the first failure.
  - **Bucket passes** if and only if it is Holm-rejected **and** |m_b| > τ_b **and** the 95% CI excludes 0.
  - Results that pass only without the Holm adjustment are labeled **exploratory** and never count as a PASS.
- **Minimum sample:** at least 30 markets **and** at least 10 clusters per bucket. Buckets below that are "insufficient". They count as not passed but still count in the 14.
- The calibration result is a **finding**, not a trading licence. The sleeve is judged separately (§6).

## 6. Trading sleeve (val)
- **Rule (fixed buckets; does not depend on which buckets pass §5):**
  - **Sell longshots:** at the snapshot, hold **NO** (weight −w) on markets priced in 0–2, 2–5 or 5–10 cents.
  - **Buy favorites:** hold **YES** (+w) on markets priced in 90–95, 95–98 or 98–100 cents.
  - Hold to resolution. Each position gets equal weight w, with gross capped at `PredCost.max_gross` = 1 across open positions (engine re-scaling).
  - Targets stay constant once entered, as required by `run_pred`.
- **Costs (base case; see §7):** tail-bucket half-spread of 1.5 cents (3-cent full spread), with **no** cap at half the distance to 0/1. Mid-bucket half-spread of 1 cent (`PredCost` default). Plus taker fees per §7, and a fill at the next observed price (`delay_rows = 1`). Unit notional $100k.
  - The uncapped tail spread needs an engine change (PR #10 item 6).
- **Window:** only markets selected under §1. Entries happen at the snapshot, starting no earlier than 2026-03-01. No position is opened in a market whose snapshot falls before val.
- **Stresses (both reported; both must pass conditions 2–4 below):** 1.5× all costs (spread and fees), and one extra row of delay (`delay_rows = 2`).
- **Pass bar (base costs, val):**
  1. **DSR > 0.95** (`stats.dsr` on daily P&L) with **n_trials = max(15, ledger count)**.
     - **Basis for 15:** the 14 bucket tests in §5 (each a look at the same val outcomes, counted as trials, including insufficient ones), plus 1 sleeve configuration. The ledger count is a **hard floor**.
     - No other pred-desk strategy is known to have been evaluated on val (`RESEARCH_LOG_2026-10-03.md` lists none). The ledger count of distinct pred strategies with a val evaluation in `company.db` is `REQUIRED-BEFORE-RUN`, and if it is larger, that number is used.
     - `trial_sr_std` comes from the ledger: `REQUIRED-BEFORE-RUN`.
  2. **Mean excluding the top 5 days > 0** (`stats.mean_ex_top`, k = 5).
  3. **Block-bootstrap 95% lower bound ≥ 0** (one-sided) on daily P&L, computed per the PR #10 spec rule **G7** (studentized or BCa interval; block size max(5, ceil(T^(1/3))); ≥ 5,000 draws; seed logged in the manifest and the ledger row; passes the null-drill false-positive check).
  4. **Net P&L > 0**, and both val halves (2026-03-01..04-15, 2026-04-16..05-31) have positive net P&L.

  Also required: an event-clustered bootstrap (G7 clustered form) of the sleeve's per-market P&L, with a one-sided 95% lower bound ≥ 0, so that a single large event can't carry the result.
- **Reporting:** all 14 bucket results and the sleeve are reported, with every metric and each stress. Nothing is selected afterwards. A FAIL is final; any change counts as a new trial and a new pre-registration. A PASS only makes the strategy a forward-paper candidate. Deployment needs forward results and human approval.

## 7. Costs: fees, spread, and why they're likely optimistic
- **Taker fees:**
  - The formula `fee = C × feeRate × p × (1−p)` and the current category rates are VERIFIED (V8, https://docs.polymarket.com/trading/fees).
  - **History:** fees applied to Crypto from January 2026 and Sports from 2026-02-18, and expanded to the other categories for markets deployed from **2026-03-30**. This is UNVERIFIED officially and corroborated only by secondary sources (V9). It falls **inside val**, and rates before 2026-03-30 are UNVERIFIED.
  - **Base case (conservative):** apply the current category rate to **every** val trade that isn't Geopolitics, regardless of date or `feesEnabled`. Unmapped categories use 0.07.
  - **Secondary (reported, not a gate):** apply fees only where the market's `feesEnabled` is true (V10), using the current rates.
  - Taker rebates are ignored. Polymarket US (a separate CFTC venue) is out of scope.
- **Spread:**
  - The base case models a **3-cent full spread (1.5-cent half-spread) in the tail buckets (p < 10 cents or p ≥ 90 cents)**, uncapped. That is above the engine default because books are thin and one-sided there.
  - The mid buckets use 1 cent (`PredCost.half_spread`).
  - The 1.5× stress raises these to 2.25 / 1.5 cents.
- **Why it's likely still optimistic:** depth isn't modelled, fill at "the next observed price ~1 h later" isn't guaranteed, and the spread is likely worse in thin markets than assumed. If an order-book snapshot source becomes available, costs must be re-checked against it **before** any forward paper.
- Capital is locked until resolution. Opportunity cost and the risk of resolution disputes are not modelled.

## 8. Open items (REQUIRED-BEFORE-RUN)
1. The SHA-256 of `results/v3/store/pred/splits.json`, and a match against §1.
2. `provenance.json` showing real Polymarket data (not the synthetic fallback).
3. A full, non-incremental market re-page, with page and market counts recorded.
4. Whether `closed=true` includes archived or delisted markets.
5. Stored `closedTime` and resolution timestamp per market (for the §1 anchor).
6. Stored outcome labels, to exclude non-Yes/No binaries.
7. Stored event id and `negRiskMarketID` (§4 clustering). The field names are UNVERIFIED.
8. A trade-count source and stored trades (§4 liquidity floor). The data-api endpoint is UNVERIFIED.
9. A source for disputed or ambiguous resolutions, or a disclosed waiver.
10. The engine change for the uncapped tail half-spread and category taker fees (§6–§7).
11. Evaluator (with Holm and cluster bootstrap) and sleeve strategy commits, with SHA-256 hashes.
12. The ledger's pred val trial count (hard floor) and `trial_sr_std`.
13. Sign-off from QL Validator that eval gating is fixed.

**Still UNVERIFIED (see Appendix V):** what `umaEndDate` means (V2); archived coverage (V5); the official fee-history dates and the rates before 2026-03-30 (V9); whether `feesEnabled` is historical (V10); the dispute token in `umaResolutionStatuses` (V12); the UMA on-chain source (V13).

## Appendix V. Polymarket API and fee verification (2026-10-21)
Method: public docs, plus a handful of single-record API responses (gamma market `613835`, and data-api `/trades` with `limit=1`). No datasets were downloaded or saved.

| # | Fact | Status | Source |
|---|---|---|---|
| V1 | The gamma market object has `endDate` (ISO), `closedTime` (note the non-ISO format, e.g. `2026-03-16 06:41:45+00`), `umaEndDate` (ISO), `umaResolutionStatus` (e.g. `resolved`), `umaResolutionStatuses` (a JSON-string history, e.g. `["proposed", "resolved"]`), `negRisk`, `negRiskMarketID`, `negRiskRequestID`, `outcomes` (a JSON string, e.g. `["Yes","No"]`), `conditionId`, `questionID`, `archived`, `closed`, `automaticallyResolved`, `resolvedBy`, `feesEnabled`, `feeType`, and `events[]` with `events[].id` | **VERIFIED** (field names) | https://gamma-api.polymarket.com/markets/613835 ; https://docs.polymarket.com/market-data/market-details.md |
| V2 | `umaEndDate` is the resolution timestamp. In the sample it equals `closedTime`, 2026-03-16 06:41:45 UTC, a day after `endDate` | field **VERIFIED**; meaning **inferred / UNVERIFIED** (not defined in the docs) | same |
| V3 | Event id = `events[0].id` (sample `50251`) | **VERIFIED** | same |
| V4 | `archived` means "archived and read-only: no trading, no resolution updates" | **VERIFIED** | https://docs.polymarket.com/market-data/market-details.md |
| V5 | Whether `closed=true` returns archived markets | **UNVERIFIED**. An `archived=true` query parameter appeared to be ignored (it returned markets with `archived=false`), so archived coverage can't be confirmed by filtering. Mitigation: the full re-page records the count of `archived=true` among returned markets | https://gamma-api.polymarket.com/markets?archived=true&limit=2 |
| V6 | data-api `GET /trades`: params `market` (comma-separated conditionIds) or `eventId`, `limit` ≤ 10000, `offset` ≤ 10000 (a 400 past that; page inside `start`/`end` windows to go deeper), `start`/`end` epoch seconds, `takerOnly` (default true), `side`. Market-scoped history has a floor of about 3 years. A v2 route `/v2/trades` exists | **VERIFIED** | https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets |
| V7 | Trades for a market closed in March 2026 are returned (one `limit=1` record, timestamp 2026-03-16 UTC). Offset 10001 returns `{"error":"max historical trades offset of 10000 exceeded"}` | **VERIFIED** | https://data-api.polymarket.com/trades?market=0x66af911f31ceabb5f9a3ef9203a73f0b23d18aa81bbba7888dec440c622c2528&limit=1 |
| V8 | Fee formula `fee = C × feeRate × p × (1−p)`, takers only. Current rates: Crypto 0.07; Sports, Economics, Culture, Weather and Other 0.05; Finance, Politics, Mentions and Tech 0.04; Geopolitics 0; Combos 0.05 | **VERIFIED** (current schedule) | https://docs.polymarket.com/trading/fees |
| V9 | Fee history: taker fees on Crypto from January 2026 and on Sports from 2026-02-18; expanded to Finance, Politics, Economics, Culture, Weather, Tech, Mentions and Other from **2026-03-30**, for markets **deployed** from that date | **UNVERIFIED from an official changelog**. It is corroborated only by secondary sources (Coincu, MEXC/BEN, Galaxy research, an r/Polymarket post). Rates before 2026-03-30 differed (Galaxy cites a lower crypto peak), and their exact values are UNVERIFIED. The Wayback Machine was rate-limited (HTTP 429) and couldn't be checked | https://coincu.com/news/polymarket-fee-expansion-march-30-2026 ; https://www.galaxy.com/insights/research/weekly-top-stories-03-27-26 ; https://www.reddit.com/r/Polymarket/comments/1s1xo9l/ |
| V10 | Per-market flag `feesEnabled`. The sample market (Oscars 2026, closed in March 2026) has `feesEnabled=false`, `feeType=null`, consistent with V9 | field **VERIFIED**; whether it reflects the historical flag at trade time is **UNVERIFIED** | https://gamma-api.polymarket.com/markets/613835 |
| V11 | Resolution uses the UMA Optimistic Oracle: proposal, a 2-hour challenge period, and a dispute that leads to a second proposal; a second dispute goes to a DVM vote | **VERIFIED** | https://docs.polymarket.com/concepts/resolution.md |
| V12 | A dispute is detectable from the Polymarket side via `umaResolutionStatuses` (history list) | field **VERIFIED**. The exact token used for a dispute (e.g. `"disputed"`) is **UNVERIFIED** (not documented, and not seen in the one sample) | V1 sources |
| V13 | UMA Optimistic Oracle on-chain dispute events (e.g. a `DisputePrice` event via the UMA subgraph) as an independent source | **UNVERIFIED** (not checked) | — |
