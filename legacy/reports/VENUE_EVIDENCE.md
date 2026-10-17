# Measured target-venue evidence

Agents can now inspect existing public Hyperliquid captures, compute immutable studies, read exact study IDs and retrieve their summaries through organizational memory and the research map. This provides actual constraints for research allocation. It does not establish agent acceptance, beneficial learning, profitable strategies or execution calibration.

## What the existing recordings support

Three known 30 September captures were selected by identity and acquisition mode, before measuring their results. They contain a short fast-feed probe, a long twenty-level recording and a long five-level recording. The raw journals and status files were preserved.

| Capture | Actual books | Unique public prints | Book span | Median interval | Continuous segments under the five-second gap rule |
| --- | ---: | ---: | ---: | ---: | --- |
| `hyperliquid-20260930-112558-56191c` | 110, five levels/side | 171 | 59.275 seconds | 539 ms | One segment, 110 rows |
| `hyperliquid-20260930-112115-d26336` | 5,205, twenty levels/side | 216,146 | 27,973.301 seconds | 5,393 ms | 5,182 segments; maximum two rows |
| `hyperliquid-20260930-113005-436f57` | 50,621, five levels/side | 215,192 | 27,447.104 seconds | 538 ms | Four segments; 1,730–19,905 rows |

The slow twenty-level feed exceeds the current five-second gap rule on 5,179 measured within-connection intervals. Under that policy it cannot support ordinary multirow mature-history signals in a replay of this capture. This is a constraint on this acquisition path and policy, not proof that twenty-level strategies or other collection methods are impossible. The paper runner's REST polling is a different acquisition path; these WebSocket cadence figures must not be substituted for measured REST or inference/routing timing.

The fast capture supports longer histories but provides only five observed levels. It cannot establish twenty-level features or unobserved deeper liquidity. Row-count windows also require explicit temporal interpretation when transferring between archive sampling and target-venue acquisition.

The short probe has a terminal marker. Both long recordings have no `CAPTURE_END`, their recorded PIDs are absent, and their status files still say RUNNING. The studies label them `INTERRUPTED_NO_END_MARKER`, and inventory labels their effective status INTERRUPTED. Raw counts differ slightly from stale status counters because studies scan all saved records. No missing tail or disconnected interval is reconstructed or declared complete.

## Hypothetical depth sensitivity

Each displayed-depth scenario is independent per book and direction. Frozen defaults are 10% participation per observed level, a five-basis-point sweep limit and 0.5 bps extra slippage. Requested BTC quantity is scenario USD notional divided by contemporaneous midpoint. This is the existing shadow fill model, not own-order observations or calibrated capacity. Spread-gated books are excluded from the scenario denominator, and results are weighted by received books rather than elapsed time.

| Long capture | Scenario | Fraction of eligible books fully filled, buy | Sell |
| --- | ---: | ---: | ---: |
| Twenty levels | $2,500 | 99.81% | 99.79% |
| Five levels | $2,500 | 92.08% | 94.31% |
| Twenty levels | $10,000 | 99.71% | 99.60% |
| Five levels | $10,000 | 84.09% | 88.54% |

The recordings overlap substantially but do not contain matched identical book snapshots. These are descriptive sensitivity results, not an experimental estimate of the causal effect of adding fifteen levels. They show why assuming every requested size fills at the best quote is unsupported. They do not justify increasing size or approving a strategy.

Median observed spread is approximately 0.119 bps in each long recording. Median hypothetical entry drag is approximately 0.56 bps under the frozen extra-slippage assumption; this excludes fees, exit cost, funding and any unmodeled impact. No strategy PnL is calculated.

## Causal and provenance contract

- A print joins only a previously received quote whose exchange timestamp is no later than the print, whose local receipt is no later than the print receipt, and whose age meets the frozen gap rule. The slow recording has 201,800 such prints, 13,773 stale-quote exclusions and 573 future/unavailable-quote exclusions. The fast recording has 210,125 such prints and 5,067 future/unavailable-quote exclusions.
- These price distances are not our slippage, fills or aggressor classification. Side values are preserved as reported labels only. Public user addresses and private reasoning are not included in study summaries.
- Trade identity is `(time, coin, tid)`. The long recordings contain 25 and 41 duplicate prints respectively, with zero observed conflicting duplicates. This does not certify complete market volume.
- Invalid records, disconnections and out-of-order books break quote continuity. Repeated timestamps do not restore broken continuity. A bounded previous-quote freshness fraction is descriptive and cannot grant complete-event coverage or forward eligibility.
- Sources must be frozen: live PIDs are refused; an absent recorded PID or explicit terminal marker establishes the bounded source's termination evidence. Unknown liveness without a terminal marker is refused. Exact capture IDs and resolved path guards restrict reads to local market-data sources.
- File SHA256, size and modification time are checked during fingerprinting, during scanning and again before commit. Resource limits and cooperative control apply. Study identity binds source, assumptions, measurement code, liquidity-engine code and NumPy version. Changed sources or implementations receive new immutable identities; older reports remain inspectable.
- Funding context is indicative. It is not a settlement cashflow model. Local hashes establish artifact consistency, not independently authenticated acquisition or numerical certification.

The [official Hyperliquid WebSocket specification](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions) documents fast five-level and slow twenty-level books and the composite trade identity. The studies use the actual observed levels and timestamps instead of assuming the documented service establishes capture completeness.

## Use and verification

```powershell
python -m quantdesk venue-captures
python -m quantdesk venue-study hyperliquid-20260930-113005-436f57 --notionals 250 1000 2500 10000
```

Agent tools are `list_venue_captures`, `study_venue_capture` and `read_venue_study`. Research maps expose exact available IDs; full assumptions and metrics are read from the immutable report. Sealed campaigns may read studies but cannot create new ones. Repeated identical tool measurements reuse the cached study without another memory insertion. Public studies are available across campaigns and are not candidate or holdout results.

`verify_venue_study.py` used the production agent tools on the three frozen captures. Its final receipt and report paths are in `results/desk/verification/venue-study/run.json`. Initial receipts are retained separately because the subsequent provenance refinement bound the numerical dependencies as well. Remeasuring that same data does not constitute a new market observation or additional independent evidence.

Dedicated tests cover hand-computed fills and sweep independence, causal print joins, deduplication, temporal resets, malformed records, indicative context, source liveness, path guards, cancellation/resource limits, multi-file mutation, implementation changes, immutable caching/integrity and role/map integration. Financial fixtures remain synthetic. The native directory-symlink fixture is skipped on this Windows host; resolved directory/file escape guards are exercised separately. Actual CLI inventory and exact-ID reuse are also checked. The final test counts and source hashes are in `results/desk/verification/venue-study/verification.json`.

The measured run adds no provider requests, holdout exposure, candidate promotion, cycles, forward plans or exchange orders. The original campaign allowance remains unchanged. Reliable real company completion and all strategy/portfolio acceptance remain outstanding.
