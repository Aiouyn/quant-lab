# Durable task allocation and real handoff acceptance

The original autonomous pilot failed. This extension makes the cost of a stalled role independently bounded and preserves useful public evidence during finalization. It does not demonstrate a completed autonomous company, effective learning, improved research decisions or profitable strategies.

## Allocation contract

A caller can include a positive integer `task_token_limit` in a role task with a durable `task_id`. The first checkpoint freezes that limit. New request reservations are attributed to the task in SQLite and must fit both its remaining allowance and the existing campaign allowance. Actual billed usage replaces a reservation; ambiguous failed calls remain charged at the reservation. The check and request attribution are atomic, including concurrent reservations.

Restarting a task cannot reset its usage, raise its limit or escape the existing cap by omitting the field. A new cap cannot be retroactively attached to an older checkpoint with incomplete request attribution. Completed checkpoints cannot cross roles or campaigns. Unbounded tasks retain the campaign limit; task limits are currently opt-in rather than an empirically calibrated automatic company allocation policy.

The reservation estimates prompt/context bytes conservatively and adds the maximum completion allowance. It can reject a request whose eventual actual bill might have fit. Conversely, unexpectedly larger actual usage is still recorded and charged; the cap cannot reverse an already completed provider bill. This is a pre-call reservation limit, not a guarantee that an inaccurate provider usage report cannot exceed an estimate.

Report-only turns no longer send unused tool schemas. A failed scientist with recorded public observations can resume in a fresh finalization conversation containing the measured planning contract, focused retrieval/coverage facts and digests of full stored results. The new conversation does not replay the prior opaque reasoning or pretend that a stripped tool conversation is complete. Live multi-turn tool conversations still preserve MiMo's required [thinking-context pass-back](https://mimo.mi.com/docs/en-US/usage-guide/passing-back-reasoning_content). Exact original results remain on disk; focused finalization summaries do not replace them.

## Real test and failure, 30 September 2026

`verify_handoff_acceptance.py` created one separately labeled scientist design-repair test in existing campaign `2b22519842f6`. It did not restart the three-attempt pilot, create another campaign or cycle, raise the original 750,000-token allowance, seal holdout or submit a strategy. Its own contract allowed at most two real provider requests, a 60,000-token task allowance and 15 minutes from the original start. The provider remained `mimo-v2.6-pro`.

The first request used **21,931 actual tokens** and retrieved memory and development coverage. Its accumulated context would have required a next reservation of **60,673 tokens**, above the task's remaining 38,069. The request was stopped before another provider call. The limit worked; no validated scientific handoff existed.

After reviewing that specific failure, one permitted finalization resumed the same task using focused public artifacts. Its next reservation was **32,704**, preserving the original task limit, two-call total and original time boundary. The real second request used **10,505 tokens** and returned final JSON, but that JSON did not contain a valid `strategy_spec`. The trusted handoff gate rejected it, and the one remaining request allowance was exhausted. The test therefore ended **FAILED**. No extra retries were launched or passing result reconstructed.

The exact rejected second response was not persisted by the then-current runtime. Its reason for omitting the specification cannot be inferred from the validation error. The current runtime now records `PUBLIC_HANDOFF_ATTEMPT` before validation, preserving the public content of rejected or malformed handoffs without opaque reasoning. That later logging fix has regression evidence, not a new real provider retest.

The original three-attempt pilot used 644,405 tokens across 18 requests. This separate test added **32,436 actual/charged tokens across two requests**. Campaign total is now **676,841**, leaving **73,159** of the unchanged original 750,000 allowance. The original cycle remains ABANDONED with zero completed company cycles. Failed test receipts and request identities remain durable.

Receipts: `results/desk/verification/handoff-acceptance/first-attempt.json`, `run.json` and `verification.json`. Source snapshots distinguish the initial attempt, compact finalization and subsequently tested public-handoff logging.

## Verification and what follows

The full authored suite passed 219 tests in 61.28 seconds after the allocation and compact-resume changes. The subsequent public-handoff logging change and its additional regression were checked with the targeted agent/task suites. Numerical and provider fixtures are often synthetic; the actual two-request test remains failed despite those green checks.

The next design work is an explicit proposal submission tool and a grounded stop/redesign path. A scientist should be able to explain that a mission is unsupported without being forced to invent a strategy specification. Rejected public work must be inspectable, and company acceptance must distinguish a useful evidenced decision from an executable strategy cohort. These capabilities, real completed cohorts, calibrated role allocations and a controlled memory-benefit comparison remain outstanding. A smaller context reservation alone does not prove cheaper or better total research.

Subsequent proposal/stop implementation, separate accounting, regression and real-data evidence are now recorded in [SCIENTIFIC_DECISIONS.md](SCIENTIFIC_DECISIONS.md). Its separate one-request real protocol test timed out without usable output; the original handoff test above was not resumed. Latest campaign charge is 708,039 across 21 requests (676,841 known actual, the newest 31,198 reservation ambiguous), leaving 41,961. No reliable real company completion or learning-benefit claim follows.
