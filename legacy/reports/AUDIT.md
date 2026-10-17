# Quant Lab system audit — 2026-09-29

## Scope and observed state

Inspected the agent, swarm, theorist, tool, data, execution, validation, metrics, ledger, portfolio, dashboard, and test paths. Ran the original tests before redesign. The Drive inventory contains **331 daily BTC-USDT-SWAP archives**, **50,053,843,328 bytes**, dated **2025-11-01 through 2026-09-27**. The sampled archive exposes snapshots and book updates with millisecond timestamps and three-item price/quantity/order-count levels. Sample cadence is approximately one second, with variation. Sampled messages do not expose executed trades or sequence IDs.

The original ledger contained **72 hypotheses, 72 implementations, and 72 REJECTED audits**, plus four tools flagged verified. These historical records have been preserved. No previous strategy has been automatically adopted by v2.

## Findings

| Severity | Finding and evidence | Resulting change |
| --- | --- | --- |
| Critical | `BaseAgent.query` and `MegaSwarmEngine.llm_query` request `no_think=True`, cap responses at 1,500 tokens, and silently return canned output after API failures. Swarm fallback also counts fictional usage. | Active v2 uses thinking, sufficient completion space, multi-turn tools, real usage, finite campaign budgets, and explicit failures. |
| Critical | `ExecutionSimulator` grants immediate passive entry and exit fills without queue evidence. | V2 uses explicit delayed taker execution; maker is an unverified research proxy and cannot be promoted. |
| Critical | FIFO code repeatedly adds `initial_depth-current_depth`, double counting a single depletion; cancellations are treated as executed volume; passive exits are guaranteed. | The active engine never infers traded volume from book depletion, and closes at paid taker prices. |
| Critical | Avellaneda–Stoikov simulator invents maker fills using a 65% random rule, ignores configured queue/latency parameters, and marks next-tick fills at the preceding tick's price. | Kept as a quarantined legacy diagnostic; it is not an active promotion or execution model. |
| Critical | Auditor uses OR conditions between optimistic and FIFO trade count/PnL/Sharpe, and uses their maximum Sharpe for DSR. Optimistic maker results can rescue failed strict results. | Independent deterministic promotion, no alternate optimistic rescue, chronological full-date replay, fee and latency stress, and sealed test. |
| Critical | Seeded `zscore_ofi` and `vol_regime_ratio` backward-fill rolling warmup and are flagged verified without passing real causality checks. | Active v2 uses separate causal features and multi-prefix checks. Fresh legacy seeds also no longer backward-fill. Historical legacy registry rows are not trusted by v2. |
| Critical | CPCV calls the strategy on the original frame containing forward labels, ignores training masks, concatenates disjoint test chunks into continuous trading, and calls the fraction of losing paths PBO. | V2 isolates causal columns and replays independent dates; statistics are named according to the implemented procedure. Learned-model CPCV is not falsely claimed. |
| High | Risk thresholds use the full-frame median/percentile; this is future leakage. | V2 uses fixed precommitted spread, allocation, latency, and drawdown limits. |
| High | Forward labels and holding/latency periods use row shifts while named in seconds. | V2 uses exchange timestamps, resets, and gap handling. |
| High | Original top-of-book OFI omits disappearance of the prior bid queue on a downward move and prior ask queue on an upward move. | Corrected causal top-of-book OFI in active reconstruction. |
| High | Swarm reuses the latest 10,000-row prefix indefinitely and evolves against the same data; date cache ignores new requests. | Frozen multiday plans, distinct development/validation/test dates, content-aware feature cache, full-day gates. |
| High | Hall of fame query includes rejected strategies. Failure recommendations assert an optimal 30-second horizon without observed evidence. | Grounded durable lessons, explicit uncertainty and lineage, director decisions tied to observed reports. |
| High | Trade Sharpe is annualized with an arbitrary opportunities-per-day factor and then supplied to DSR with incompatible estimator units. | Marked account returns and daily block statistics; no misleading DSR/PBO label. |
| High | Portfolio “inverse volatility” actually weights by historical Sharpe; failed strategies can misalign signal and weight lists; holdings are hardcoded. | Aligned net returns, covariance shrinkage, capped allocations, duplicates removed, explicit cash, per-strategy specifications. |
| High | Dashboard starts autonomous API work at module import; fake activity and fixed archive-size telemetry imply more actual work than measured. | Read-focused evidence dashboard with explicit bounded starts and real ledger events. |
| Medium | Swarm shares mutable auditor/developer/config among threads, records trials after selection, uses campaign ID 1, and references undefined `hp` when adding approved strategies. | Serialized cohort orchestration, durable task states, pre-experiment accounting, atomic budget reservations, actual IDs. |
| Medium | Generated code executes directly in-process with unrestricted `exec`; input price comparison is not a full sandbox. | Numerical AST allowlist, credential-free subprocess, timeout, output validation, input mutation checks, and several causal prefixes. Explicitly not an OS sandbox. |
| Medium | Toolsmith prompt describes backward filling as causal, theorists silently substitute default mathematics after JSON errors, and original dossier fields read missing metrics as zero. | Active agents use an explicit numerical contract, real tool errors, exact report fields, and measured durable artifacts. |
| Medium | Archive loader selects `getmembers()[0]`, can select a directory, decompresses the whole archive for member inventory, and ignores all subsequent members. | Streaming member traversal without extraction; directory skip and snapshot requirement. |
| Medium | Basic/FIFO simulators omit last open positions and first-loss drawdown; assumed fee tier does not match the intended venue. | End-boundary liquidation, marked equity including initial capital, target-venue base fees, and explicit unresolved funding. |
| Medium | There was no implementable Hyperliquid paper path, bundle contract, portfolio promotion contract, or dependencies file. | Shared replay/paper broker, hashed bundles, public feed, local journals, portfolio manifests, documented CLI, requirements, and regression suite. |

## Design judgment

The main bottleneck was the trustworthiness of evidence and the specificity of agent work, rather than the number of named agents. A small set of capable agents with different authority, tools, artifacts, and durable memory is a better first implementation. The loop should earn additional complexity by producing repeatable evidence.

Each agent is allowed to work across multiple tool turns. Experiments can implement new numerical features and mechanisms, run code, inspect prior implementations, perform ablations, and revise failures. The historian records lessons with evidence rather than declaring every rejection an instruction to add a familiar factor. The director receives those lessons when choosing the next mission. This is organizational learning through memory and experiment selection, not neural-model training.

Validation remains deliberately independent of natural-language confidence. No committee vote can bypass missing data, noncausal code, high costs, underpowered samples, reused test data, unverified maker fills, or a drawdown halt.

## Verification

Before redesign: **24 original unit tests passed**. Many accepted either APPROVED or REJECTED, tested types rather than economics, or only checked a single loose invariant. Passing them did not substantiate the institutional claims.

The expanded suite currently covers corrected OFI, snapshots/sequence gaps, actual time horizons, streaming archives, several forms of lookahead, restricted numerical code, worker output, fee loss in a flat market, latency, maker depletion, liquidation/drawdown, gap cancellation, atomic budgets, durable deduplication/memory, cross-campaign holdout protection, sample/proxy promotion blocks, multiple-testing penalties, portfolio caps/duplicates/cash, bundle integrity, stale paper feeds, MiMo thinking/tool-context passthrough, API failure behavior, and tool authority.

A hand-written baseline was replayed on actual sampled Drive dates. It was rejected after costs and for inadequate validation coverage. This is an infrastructure check, not evidence of a discovered profitable strategy. Measured full-day, real-provider, and public-feed probe results are recorded in `VERIFICATION.json`. The full-day baseline is explicitly handwritten and the short live probes are infrastructure checks, not evidence of alpha.

## Remaining limitations

- All 331 archive files have been inventoried; exhaustive reconstruction and quality auditing of every message is not yet complete. Full campaigns perform that work lazily and record failures.
- The data is sampled L2, not order-level events. Executed trades, queue priority, and cancellation races are unidentified. Maker promotion stays blocked.
- Historical funding and realistic multi-level market impact are not available in the input archive. Extra slippage is a configurable conservative assumption, not a calibrated impact model. Position size is fixed conservatively; leverage/liquidation is not modeled because this is a cash-collateralized shadow account.
- The active numerical contract supports sophisticated causal transformations but does not fit or retrain arbitrary neural models. A fitted-model extension requires nested chronological training, purged labels, persisted fitted state, and fresh evaluation evidence.
- Statistical estimates have assumptions and can be underpowered. Reusable validation permits selection bias; trial accounting and a fresh sealed test reduce it but do not make financial evidence certain.
- The paper path is local shadow execution using public books. It cannot reproduce actual matching, priority, acknowledgement, or rejection. No signed exchange orders are implemented.
- Portfolio formation is based on reusable validation and individually passed test results. A formed portfolio still needs its own future paper record.
- No profitable alpha, live deployment, capital readiness, or hedge-fund-equivalent performance has been demonstrated.

## Primary references checked

- [MiMo deep thinking and multi-turn tool context](https://mimo.mi.com/docs/en-US/usage-guide/passing-back-reasoning_content): thinking parameters, response budgets, and required reasoning-content pass-through.
- [Hyperliquid info API](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint): public L2 snapshot request/response.
- [Hyperliquid perpetual fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees): current base fee assumptions used in v2.
- [OKX API documentation](https://www.okx.com/docs-v5/en/): quantities/contracts and orderbook sequencing. The downloaded archive's omissions are checked directly rather than inferred from live API capabilities.

## Final hardening and operational checks

Additional review identified cross-campaign data contamination: a date used for earlier development could be nominated as a different campaign's fresh holdout. Research exposures now have a durable date registry, and sealing rejects those dates before any test replay. Legacy campaigns lacking exact exposure logs have their planned research dates conservatively quarantined.

Strategy state now resets with reconstruction segments. Long paper runs retain the complete segment instead of dropping the first rows at 6,000 observations, reset at UTC day boundaries, and halt at an explicit history limit. Repeated stale/duplicate books count as feed failures. Exported execution assumptions and portfolio manifests are hash checked alongside numerical code and specifications. These are integrity checks, not signed attestations against hostile tampering.

The numerical contract now permits public helpers and finite range loops for causal online estimators, while retaining child-process execution, import and operation restrictions, causality tests, and timeouts. Identical tools reuse persistent evidence. Retrieved lessons survive a flood of feature probes, and historian records attach actual experiment/validation grounding automatically. A stop arriving during an API call is respected before its returned tools execute.

Hyperliquid settles funding hourly. New entries are blocked near that boundary because the OKX archive lacks target-venue funding. If a gap still carries a position across it, the replay records this and promotion fails. Bootstrap resampling resolution scales with family size, in bounded memory batches, rather than making the p-value floor an accidental permanent promotion block after 200 trials.

A new public WebSocket collector preserves BTC L2 snapshots, identified trades, public asset context, receipt times, and disconnection/gap records. It deduplicates trades by the venue's documented (time, coin, tid) identity while retaining raw packets. The first eight-second live probe received 3 books, 33 trades and 8 asset-context messages, with one observed book gap and no connection errors. The collector does not infer queue priority, certify complete capture, or bypass the maker gate.

A complete 2025-11-01 archive reconstructed 86,400 valid observations and 96 snapshots spanning 86,398.995 seconds. It had no observed crossed books or timestamp gaps; sequence continuity remained unverifiable because sequence IDs were absent. Initial cold reconstruction took about 15.7 seconds; cached loading took about 0.02 seconds. The handwritten baseline hit the 5% drawdown halt after fees and was rejected, as intended.

The real provider probe exposed long tool-call response latency. The active runtime now enforces total wall-clock deadlines separately from SDK socket-inactivity timeouts, keeps ambiguous usage charged, and ignores late responses. Engineer checkpoint finalization receives compact measured results and exact existing implementations, with a dedicated submission-by-ID tool; it does not have to regenerate tested code or restart profiling. Falsification and organizational memory are batched once per cohort, so additional ablations do not multiply review conversations.

A generated taker candidate (`6bc072b37a0b9b55`) completed a 330-observation public Hyperliquid shadow session with no feed errors. It generated no trades and no PnL during that short window. Separately, the explicit constant-signal infrastructure probe exercised two closed local trades and recorded fees, with no signed exchange orders. Neither observation establishes alpha.

Additional primary references: [Hyperliquid WebSocket subscriptions and trade identity](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions), [Hyperliquid hourly funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding).

Final regression suite: **76 passed**. This includes an integrated two-cycle fixture with actual numerical replay and deterministic rejection, verifying that the next director receives persisted empirical lessons. That fixture uses scripted cognitive decisions and is not counted as a real MiMo discovery. Real provider progression is reported separately in `VERIFICATION.json`.

The real MiMo campaign completed all five cognitive roles and the deterministic validator. Six implemented variants were stored; the engineer selected a taker fade candidate for formal validation, which was rejected on inadequate coverage, trade count, and negative net performance. Independent critique and a measured lesson persisted. It did not close an entire strategy class from a small sample, and no test cohort was consumed or candidate promoted. Provider deadlines and budget exhaustion were exercised during recovery; checkpoints and all ambiguous usage reservations were retained. Total charged/reserved usage for the verification campaign was 487,840 tokens, including failed/unknown requests; this is not a claim of actual provider billing.

The critique identified a genuine evidence gap: gross capture was unavailable separately from net PnL. Replay now writes per-date trade logs with midpoint capture, spread/slippage drag, fees, and net basis points. The tool `diagnose_execution` exposes permitted phases and blocks holdout and exploratory access to validation trade logs. A repeated three-date development replay attributed +$2.383 gross midpoint capture minus $0.761 execution drag and $6.749 fees to -$5.127 net across three trades. This verifies accounting and supports cost investigation; it does not establish strategy profitability or a general failure of the signal family. Older reports remain historical and may lack these newly introduced fields.

Stop requests are checked between archive replays and propagate as operational interruptions, preserving candidate status and checkpoints. Final regression suite: **80 passed**. Full real-provider cycle status, cold/cached replay timing, source hashes, paper probes, and collector results are recorded in `VERIFICATION.json`.
