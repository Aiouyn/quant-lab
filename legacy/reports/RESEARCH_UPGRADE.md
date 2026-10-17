# Fitted research and bounded autonomous operation

This extension gives researchers a model-building path, rather than restricting every strategy to a manually specified signal formula. It does not establish profitability or complete the proposed self-improving quant desk. Remaining work includes measured organizational learning, efficient real autonomous completion, calibrated venue execution, comprehensive data quality, independent simulation validation and fresh portfolio evidence. The dated assessment in CURRENT_EVALUATION.md records the evidence at that inspection.

## Fitted strategy contract

`compute_signal(df)` remains causal, numerical code evaluated in the existing restricted worker. For a strategy with `spec.model`, its output is a custom alpha feature. With `model.custom_features=true`, the program instead returns 1–8 named bounded causal feature columns with stable order. A trusted fitting service combines them with 1–8 selected raw features and fits `ridge`, `quadratic_ridge` or native `xgboost`. The quadratic family includes second-order interactions. XGBoost searches 1–4 bounded shallow-tree configurations. Setting `include_alpha=false` removes all custom features for an explicit ablation.

The optional model recipe controls the rolling training window, minimum earlier dates, inner selection dates, penalty candidates, sampling cadence, training-row bound, label embargo, prediction scale and cost margin. Defaults are 30 training dates, minimum 10 dates, three inner selection dates, at most 60,000 rows and a 300-second embargo. Every parameter-search penalty is counted in the lab's trial family. Generated code cannot choose label data or fit on validation/holdout dates.

Labels measure midpoint movement from the first available delayed entry quote to the first available delayed exit quote after the holding horizon. Irregular arrivals are handled through separate entry, expiry and exit timestamps. Windows crossing reconstruction segments, significant gaps, or funding boundaries are excluded; unresolved labels at a training boundary are purged. Warmup observations are excluded per segment. Scaling and coefficient fitting use training rows only. Training subsets are bounded deterministically without examining outcomes.

Inner temporal selection chooses a ridge penalty or bounded tree configuration by prediction MSE on later training dates. It records zero-predictor MSE, inner IC and prediction/target quantiles; none is a net-alpha certificate. The service then refits on that earlier development window. Outer walk-forward dates are unseen by that fit. The same outer dates, costs, holding horizon and signal threshold are used to compare cash, a fixed depth-flow trading baseline and a predictor with the custom alpha removed. Comparisons include paired net daily improvements, fees, midpoint capture and execution drag. These development comparisons guide research; they are not independent promotion evidence.

Predicted midpoint returns are filtered against configured round-trip fees, slippage, observed spread and an additional edge margin. A prediction exceeding that hurdle is an estimate, not a calibrated guarantee. MSE selection is not optimization for net trading performance. Overlapping training labels can remain correlated; temporal separation and daily economic evaluation help but do not establish statistical sufficiency.

Validation fits once using earlier development dates. The model is frozen across all validation dates, baseline/stress replay, sealed holdout and exported paper execution. Validation/holdout observations never become fit rows. This is deliberately a fixed-model confirmation policy; online retraining during the confirmation period is not implemented.

Artifacts are portable JSON containing ridge coefficients or native XGBoost model JSON, frozen feature names, parameter-selection evidence and training provenance. Hashes bind code, normalized specification, execution assumptions and model contents. No pickle loading is needed. Edited or mismatched models fail bundle integrity checks. Formula-only bundles remain supported.

## Commands

```powershell
# Existing fitted candidate, outer development tests only:
python -m quantdesk walk-forward CAMPAIGN_ID CANDIDATE_ID --folds 5 --start-index 10 --end-index 30

# Bounded unattended cycles with durable task reuse:
python -m quantdesk autonomous CAMPAIGN_ID --cycles 10 --hours 8 --max-retries 2

# Target venue recording with live status.json and raw journals:
python -m quantdesk collect --seconds 28800 --max-reconnects 20

# Prepare earlier development data before spending provider tokens:
python -m quantdesk prepare-data CAMPAIGN_ID --limit 20 --workers 2

# Separate faster five-level recording; do not mix its depth with twenty-level data:
python -m quantdesk collect --seconds 28800 --fast
```

MiMo scientist/engineer tools expose the same fitted recipe and walk-forward capability. Campaigns with too few development dates cannot fit a model. Complete daily archives are required for promotion. Date indices refer to the frozen campaign development plan, not arbitrary input paths.

## Recovery and efficiency

Submitted company cohorts now receive a development screen before full validation/stress. The screen precommits up to three outer development dates, preferring complete prepared windows without selecting on returns. Fitted candidates retain matched cash, fixed depth-flow and custom-feature-removal comparisons. Zero-trade fitted results receive a measured prediction/cost diagnostic. Inactive, negative, insufficient, operational-error and promising findings remain distinct; proposals retain PROPOSED registry status. These are allocation decisions, not permanent mechanism rejection or promotion. Only sufficiently active, positive and complete development evidence with an adequate frozen confirmation plan starts automatic full validation. Manual validation remains available.

Date selection and per-arm evidence persist across interruption. A new cache key binds candidate, frozen plan, execution assumptions and screening versions. A repeated completed screen reuses its exact dates even after more archives are prepared. Operational failures can resume unfinished arms; partial responses are never executed. CLI: `python -m quantdesk screen CAMPAIGN_ID CANDIDATE_ID`, optionally `--day-indices 10 16 22`. Scientist, engineer and falsifier tools expose the same development-only screen.

Dashboard research starts use a fresh interpreter with a complete nonsecret settings snapshot, source hashes, durable job receipts and logs. Existing dashboard processes need the current service before starting research; the updated page disables starts against an older service. The currently updated service is http://127.0.0.1:8890/. The registry labels development evidence separately and leaves its confirmation p-value blank.

Configured research calls now allow 18,000 completion tokens with a 900-second deadline, within the existing campaign reservation cap. A real unattended pilot exposed the director's old hard-coded 8,000-token cap: two successful calls consumed 72,417 actual tokens, but the second truncated before a completed mission. The director now uses the configured allowance. A truncated response can trigger one recovery within the remaining role turns, grounded in public measured artifacts; campaign budgets are never raised. Failed director handoffs can resume from their task-scoped evidence without repeating retrieval. MiMo private context remains ephemeral and is retained for actual tool conversations; recovery starts a new conversation rather than stripping required context from a tool turn. Recovery is software behavior under test, not yet proof of unattended reliability.

The supervisor persists process identity, completed-cycle count, remaining tokens, failures and stop reason under `results/desk/operations/`. It retries network timeouts, connection failures, rate limits and server errors up to a finite allowance across the run. Authentication failures and deterministic task errors stop the run. Completed roles are reused on restart; cached identical experiments are reused. Ambiguous provider failures stay charged at their reserved budget. No successful artifact is fabricated after failure.

Time limits are checked between cycles and retries. An in-flight provider call has its own wall-clock timeout; an in-flight archive replay is preserved and may exceed the run's time limit. Retry waits respond to stop requests. Sealing, allocation and signed orders are not automatic supervisor actions. This supervisor has deterministic regression coverage but still needs a substantial real unattended campaign.

Live process leases cannot be stolen solely because an estimated expiry passed. Retrieval keeps outcome statistics and date identities while omitting large replay tables; full evidence remains on disk. Formatting/comments and title changes do not create a new candidate from the same parsed numerical program. This is syntactic deduplication, not proof that economically equivalent mechanisms are distinct.

Development data preparation uses 1–4 workers and atomic cache writes, records full-day coverage and quality, and exposes no holdout dates. Missing sequence IDs remain explicitly unverifiable. Drive archive hydration can dominate cold-start time, so the preparation job publishes durable per-date progress separately from agent work.

The public [Hyperliquid subscription interface](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions) offers a faster five-level book option and a slower twenty-level option. The capture mode is stored in the journal and status. A real 60-second fast-feed probe received 110 books with no detected book gaps or feed errors. That establishes a usable capture path, not venue-transfer performance. New quantdesk-v3 bundles derive minimum book depth from code and model inputs, and halt inference on insufficient depth. Optional `input_features` restricts generated-program inputs during fitting, replay and paper. Cache identities bind that declaration. Dynamic column access without a declaration conservatively requires twenty levels. Legacy v2 bundles remain supported without this newer guarantee; venue distribution and execution calibration remain unverified.

Evaluation infrastructure failures receive `EVALUATION_ERROR`, which can be retried; they do not become economic `REJECTED` results. Rejections still follow deterministic statistical, coverage, cost and risk gates. Hyperliquid recording now publishes a status file while active; receiving books/trades does not certify complete coverage or queue priority.

## Evidence and outstanding requirements

Synthetic regression tests cover future-data isolation, train-only scaling, label boundaries, parameter-selection provenance, frozen model reuse, paper inference parity, tamper detection, selective recovery, budget accounting and live-process leases. Real complete-day upgrade evidence is written to `results/desk/verification/fitted-upgrade/`. Its verification script uses existing MiMo code and no provider calls; its benchmark/ablation artifacts are not new AI discoveries.

The system has not yet demonstrated profitable alpha, effective long-term learning, calibrated target-venue execution, or independently validated portfolio performance. A long-running process alone does not prove any of those properties.

## Implementation completion and current real probes

An engineer's final two turns retain submission tools so a rejected implementation can be repaired. Public program attempts are persisted before execution and recovered by task identity; private reasoning remains ephemeral. Failed, stopped or interrupted invocations write terminal task states and process identity prevents duplicate ownership. A prose summary without a committed strategy cannot complete an engineer task. Retrieval defaults to focused summaries and exact code remains separately available. Provider receipts record token usage and finish reasons. Truncated public responses are saved as non-executable evidence, never executed. MiMo thinking stays enabled: its documented interface has an on/off switch rather than a graded thinking budget.

The real custom-feature probe stopped first at the campaign reservation boundary and then at an 8,000-token completion limit. These were measured failures. A third attempt with 16,000 completion tokens hit its 300-second deadline; that ambiguous request stayed charged. An explicitly logged operator revision raised this test campaign from 160,000 to 250,000 tokens, with an 18,000-token per-call allowance and a 900-second deadline. The fourth attempt committed candidate 37e7854f12d976e8 in one successful 11,987-token request. This measured success does not establish a causal performance improvement from changing those limits. Independent full-day replay is recorded separately under verification/feature-replay. This is one-role integration work, not proof of a functioning autonomous company.

Exploratory fitted export can use the exact artifact referenced by a measured development walk-forward. These bundles carry `model_evidence_phase=development`, a separate development summary, and empty validation/holdout summaries. They require the exploratory flag to execute. No promotion is inferred from their availability. The `check_program` tool performs actual bounded worker checks before expensive fitting, without claiming a backtest.

The later real feature-path evidence and timing corrections are documented in [FEATURE_RESEARCH_VERIFICATION.md](FEATURE_RESEARCH_VERIFICATION.md). Paper now schedules new orders after measured signal availability plus modeled routing latency and halts on stale post-inference quotes without inventing closes. Historical replay still assumes instantaneous feature computation, so forward timing and impact remain calibration requirements. MiMo thinking controls follow the [official provider contract](https://mimo.mi.com/docs/en-US/api/chat).
