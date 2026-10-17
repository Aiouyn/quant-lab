# Individual forward acceptance before portfolio formation

The requested sequence is now enforced in the portfolio builder: historical PAPER_READY status, then the exact strategy's precommitted Hyperliquid shadow window, then allocation from aligned accepted forward returns. The resulting portfolio remains PAPER_ONLY and requires fresh observations after formation. No real strategy currently qualifies, and no real portfolio performance is established.

## Precommitted window

`python -m quantdesk forward-plan CANDIDATE_ID` freezes the next seven full UTC days for the exact bundle and execution settings. The first qualified standalone mainnet paper run also creates this plan if none exists. Default requirements are 50 closed trades, at least 95% coverage on every frozen date, positive compounded net return, no qualifying strategy/stale-decision/funding failures, and acceptance no older than 14 days. These are allocation heuristics, not a significance test, optimal policy or capital approval.

`forward-plan CANDIDATE_ID --new-window` explicitly precommits another future window and retains the previous plan. Assessment cannot choose profitable dates retrospectively or select between overlapping full-date sessions. Changing a bundle or policy requires new future evidence. Intended portfolio members need overlapping qualifying observation dates.

## Bound and reconciled evidence

New paper sessions record their exact member identity, execution configuration, source, exploratory/testnet/portfolio status and forward plan in `session.json`. Source is labeled mainnet only for the built-in public fetch path; custom fetch functions cannot qualify. Final summaries bind the session metadata and completed journal with checksums. Manual, exploratory, testnet, portfolio and legacy observations cannot grant individual forward acceptance.

`python -m quantdesk forward-assess CANDIDATE_ID` checks the frozen calendar window and streams the raw journal. It independently reconstructs cash and positions from fills, checks modeled crossing/slippage prices and fees against observed books and frozen execution, reconciles account marks/final fees/closed trades/open exposure, and measures chronological coverage. Short or cash-only probes, missing hours, incompatible depth, stale books, nonfinite accounting, incomplete receipts and overlapping full-date records cannot qualify.

Local hashes detect edits and bind the recorded artifacts. They do not authenticate an adversarially forged history, independently certify the simulator, or establish depth-aware capacity. Fills still use the existing modeled crossing assumptions; impact, partial fills, calibrated inference/routing, venue feature transfer and settlement evidence remain acceptance work.

## Portfolio formation

The allocator now uses accepted individual Hyperliquid forward net returns rather than reusing historical validation returns for weights. It retains shrinkage, duplicate exclusion, the sleeve cap and cash. The new manifest contains the assessed evidence, frozen dates, PAPER_ONLY status and the time after which a fresh portfolio record is needed. Individual acceptance does not validate a portfolio fitted to those observations.

## Verification on 30 September 2026

The full regression suite passed **189 tests in 69.68 seconds**. The forward cases use intentionally synthetic receipt fixtures: future-window freezing, retained retries, source restrictions, positive/active full coverage, independent accounting, fee edits, nonfinite fields, overlap, missing hours, expiry, exact bundle binding, portfolio prerequisites/weights, custom-source paper recording and unrelated damaged metadata. These tests establish covered control behavior, not observed target-venue profitability.

An actual six-sample mainnet public-data smoke wrote the new metadata and final journal receipt, with matching hashes and no errors, inference, trades or exchange orders. It remained exploratory and was refused qualification. Evidence: `results/desk/verification/forward-gate/public-probe.json` and the linked paper session.

The latest failed real company pilot used 644,405 actual tokens, completed director/scientist tasks and failed the engineer, with zero completed acceptance cycles. The new precise recipe errors, structural history diagnostics even for invalid recipes, and stop after three identical deterministic tool failures are verified safeguards. Their improvement to real autonomous throughput is not yet established. No further pilot restart or automatic budget increase was performed.
