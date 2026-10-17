# Scientific proposal and evidenced mission-stop contracts

This extension makes scientific handoffs explicit and preserves measured reasons to stop an allocation. It does not establish reliable real company cohorts, profitable strategies, execution calibration or improved learning.

## Proposal

The company scientist must call `submit_proposal` with an exact validated specification, declared mature history, one to three development indices and public explanations of the hypothesis, implementation, baseline, ablation, stop condition and limitations. The service checks actual prepared development frames (or explicitly bounded prefixes), observed segment/holding compatibility and minimum earlier training-date relationships. Warmup cannot be declared shorter than the stated fully mature history.

The service records a hashed, immutable scientific decision under the owning scientist task. Compatibility is metadata evidence, not fitting success, predictive skill or PnL. It does not create or submit a numerical candidate. The engineer retains responsibility for implementation, repair and economic experiments. A strategy cohort receipt preserves the decision ID through submission and deterministic screening. Declared development indices commit future work; the service does not certify that those dates were never inspected or selected using earlier outcomes. Exposure records and untouched confirmation gates remain separate.

The declared history is a scientific commitment. The service cannot verify that future code implements that history until engineering checks and criticism inspect the implementation; arbitrary program semantics are not proven by this metadata gate.

## Stop and redesign

`stop_mission` supports three evidence bases:

- `DATA_INCOMPATIBLE`: the actual proposed specification must produce measured `NO_USABLE_LABELS` or `INVALID_RECIPE` on a development frame. A feasible recipe cannot be labeled incompatible by narrative alone.
- `MEASURED_DEVELOPMENT`: exact candidate IDs must have persisted development experiment, walk-forward or screen evidence. The immutable stop records the original source IDs, hashes, code/specification identity and full development evidence. Validation or holdout results cannot substitute for development grounding.
- `MEASURED_VENUE`: exact native replay IDs must match their immutable report, bound journals, candidate code/specification and registered measured memory. Repeated versions of the same configuration cannot become separate observations. Grounding remains retrospective venue development information, not independent confirmation. [NATIVE_REVIEW.md](NATIVE_REVIEW.md) documents focused role handoffs and their verification boundaries.

A stop concerns this allocation only. It does not permanently reject a strategy class, prove an optimal horizon or grant approval. An independent falsifier critiques the decision; the historian must then call `record_mission_lesson`. The stored lesson binds the actual stop and completed critique. Conflicting replacement text is rejected. The critic may disagree; the recorded interpretations remain scientific judgments, with measured evidence authoritative.

Stop cycles finish as `DECISION_COMPLETE`. They never become strategy `COMPLETE` cycles, trigger engineering/validation or create submissions. Supervisor receipts separately count completed strategy cohorts, mission decisions and legacy unspecified outcomes. The allocation map exposes reviewed stop suggestions without counting them as candidate economics or demonstrated learning gains.

## Runtime and recovery

Prose-only scientist or historian responses cannot complete these artifact tasks. The runtime persists public decision-tool arguments before validation, including rejected attempts. Successful scientific commits close the role without another summary request or execution of later calls in the same response. A crash after durable decision/lesson commit reuses that artifact without a new provider request.

MiMo's [official chat API documentation](https://mimo.mi.com/docs/en-US/api/chat) documents automatic tool selection and says non-auto modes may be ignored. The scientist's completion turn therefore advertises only the two decision tools with `auto`; trusted local validation enforces completion. A request parameter is not treated as a provider guarantee. Thinking remains enabled, and actual tool conversations retain ephemeral reasoning context without saving it as public evidence.

Director and historian prompts now omit generated-code syntax because those roles do not own numerical execution. A fresh artifact-only scientist handoff also receives the scientific recipe/feasibility contract without engineering syntax; exploratory scientist and engineer tasks retain the numerical contract. This changes instructions supplied to the relevant task, without truncating an active thinking/tool conversation. Serialized byte savings are measurable; provider token savings, latency and research quality require real comparison.

## Verification

The complete authored suite passed **236 tests in 80.47 seconds** after the service, identity, allocation-map, runtime and immutable lesson changes. Seventeen targeted checks passed first. One subsequent additional integration test passed in **4.57 seconds**, exercising the actual runtime, immutable build/check, economic replay, submission, screen, falsifier and historian with explicit synthetic data/provider responses. There is no claim that a 237-test full suite was run at that point.

`python verify_scientific_protocol.py` exercised the service on actual prepared November 23 data in an isolated verification ledger. It accepted the exact earlier MiMo recipe `37e7854f12d976e8` for compatibility and recorded the actual incompatible long-history draft `fb9f302b011322bd` as having zero usable labels. Probe arguments and task ownership were operator-authored; no independent LLM review, code replay or alpha was supplied by this check. Production candidate records, usage, cycles and holdout remained unchanged. Receipt: `results/desk/verification/scientific-protocol/actual-data.json`.

`python verify_scientific_protocol.py --real-provider` is a separate idempotent single-request protocol probe, under task `scientific-protocol:2b22519842f6:scientist:decisions-v1`. It has a frozen 35,000-token task limit, a 16,000 completion allowance, 300-second request deadline and no vendor retries. It uses the unchanged original campaign allowance and does not restart the exhausted previous handoff test or pilot. The receipt records terminal success or failure; rerunning the command reports the existing receipt without starting another call. Authoritative result: `results/desk/verification/scientific-protocol/real-run.json`.

The actual request **timed out after 300 seconds**, with no usable response or decision artifact. Its 31,198-token reservation remained charged; actual usage is unknown. Campaign usage is now 21 requests, 676,841 known actual tokens and 708,039 charged tokens, leaving 41,961 under the original 750,000 allowance. The original cycle, candidate statuses and holdout records remained unchanged. The process exited; no live research task or lease remained. No retry of the exhausted one-request test was launched. This did not evaluate a returned decision's validity and cannot establish that the new protocol works or fails on valid provider output.

After that failure, prompts were scoped by role and fresh artifact-only task. **55 targeted tests passed in 28.22 seconds**, including the extra complete proposal/replay integration and a new prompt/commit test. No full-suite result is claimed for this later prompt snapshot. A controlled comparison using identical current task and schemas decreased serialized context from 26,403 to 21,365 bytes, removing 5,038 bytes (19.08%). The local reservation estimate decreased from 31,202 to 28,683. This is a controlled serialization comparison, not a capture of the original request, actual provider-token savings, shorter completion time or better research. Receipt: `results/desk/verification/scientific-protocol/context-comparison.json`. The revised prompt has not had a new real-provider acceptance test.

Both verification modes leave company completion, independently confirmed economics, learning benefit and forward performance as separate unfulfilled requirements. Real scientist artifact acceptance, even when successful, is not full company acceptance.

Later transport work and accounting are recorded in [PROVIDER_STREAM.md](PROVIDER_STREAM.md). The separate engineering stream failed before delivery; a small corrected endpoint diagnostic completed. Latest campaign charge is 737,309, with 676,883 known actual tokens and 12,691 remaining. These later scopes do not change the failed scientific protocol result above.
