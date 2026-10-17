# Frozen strategy replay on captured Hyperliquid data

The desk can now replay immutable formula and fitted taker candidates on exact frozen public captures. It reconstructs target-venue inputs, evaluates causal code in the existing isolated worker and compares quote-based execution with the displayed-depth IOC model on the same signals. Agents can inspect the actual economic receipts through tools, memory and research maps. This is retrospective research, not qualifying forward performance or a profitable discovery.

## Actual existing-candidate result

The predeclared comparison used the long fast five-level capture `hyperliquid-20260930-113005-436f57`: 50,621 books in four continuous segments, spanning approximately 7.62 hours on 30 September. The existing custom candidate, its existing custom-feature-removal ablation and the fixed reference were chosen before their target-venue replay outcomes. The fitted pair reused the exact matching November 23 prediction-cutoff artifacts, trained on November 11–22, 2025. No fitting, parameter search, code revision or provider request occurred.

| Existing artifact | Quote arm | Displayed-depth arm |
| --- | --- | --- |
| Custom fitted candidate `37e7854f12d976e8` | Zero trades, zero modeled PnL | Zero trades, zero modeled PnL |
| Fitted ablation `4987d273959987d1` | Zero trades, zero modeled PnL | Zero trades, zero modeled PnL |
| Fixed reference `f1435f796b966260` | 221 closed trades; -$500.65 PnL; $497.25 fees | 240 closed trades; -$501.96 PnL; $458.95 fees; 223 partial fills |

The reference started with $10,000 modeled equity and 25% order allocation, and hit the configured 5% drawdown halt in both arms. It ended flat. Partial fills change exposure, subsequent orders, trading opportunities and halt timing, so the difference in final PnL is not an isolated estimate of price impact.

The custom model's maximum absolute prediction was 8.1485 bps, below minimum modeled costs of 10.1168 bps, even before its extra 10.5-bps margin. Its median prediction was 1.0270 bps. Removing the extra margin alone cannot create an eligible signal on this observed capture. Its 883 descriptive target-venue labels gave MSE 80.177 versus 77.472 for predicting zero, with Pearson IC approximately 0.0133. The ablation likewise had no prediction above modeled costs and MSE above the zero predictor. These already-observed, single-capture diagnostics are not independent statistical confirmation or a strategy-family verdict.

The explicit final-implementation recheck reproduced these results on the same source and exact models. Original receipts are retained separately. Rechecks, alternative fill arms, overlapping captures and implementation versions are not independent market samples.

## Contracts and controls

- Capture SHA256, file metadata and stopped-source provenance must match the exact study before reading, after reconstruction and before commit. Retrospective reads validate the stored report and all four bound event/trade journals. Raw captures remain unchanged.
- Book reconstruction uses continuous snapshots, resets across broken connections/invalid records/gaps and computes time features through the shared `enrich` path. Configured gap limits now reach `Book` in paper execution as well, preserving default five-second behavior while fixing custom-policy mismatches. Generated-program state resets with the resulting segments.
- Observed depth must satisfy the derived code/model input contract. Five levels cannot stand in for twenty-level features. Public depth cannot certify maker execution; this service requires taker mode.
- Fitted replay requires an exact existing model filename key. Model hash, code, full specification, recipe, execution and temporal cutoffs are checked. Both inner and outer embargo rules remain enforced, and capture observations must follow the frozen prediction cutoff. No model is retrained on the target capture.
- Both fill arms use the same signals, costs, latency and risk settings. Depth fills share each book's consumption budget across a segment-reset liquidation and the following step. Partial exits preserve exposure, including terminal residual risk. Cash, quantity and fees are reconstructed separately from fill events and reconciled with the broker state. This accounting cross-check is not independent simulator certification.
- Captured dates become globally exposed development information before forecasts are computed. Globally reserved holdout dates are blocked before the numerical worker. Experiments count the two fill assumptions conservatively in the existing research family; counting attempts does not make the arms independent.
- Cooperative stop/deadline checks, bounded raw records and a 200,000-book/one-day replay bound apply. A complete replay is cached by exact artifacts, source study, frozen model, execution/depth settings and implementation provenance. Errors leave no completed replay; exposure/attempt accounting is retained when computation has begun.
- Candidate status, company cycles, holdout records, forward plans and budgets remain unchanged. Every report explicitly carries `forward_eligible=false`, `strategy_promoted=false` and zero exchange orders.

Limitations remain material: recorded gaps and missing tails, older-model/venue distribution shift, row-history differences, assumed displayed participation, absent own routing/impact/replenishment, configured rather than measured routing latency, and missing funding settlements. The restricted funding policy is not a cashflow model. A negative or zero-trade result applies to this observed configuration and source, not all microstructure strategies.

## Agent use

Tools now include `list_frozen_models`, `replay_venue_candidate` and `read_venue_replay`. Scientist, engineer and falsifier roles may compute retrospective replays; all roles may read exact results. Directors receive actual target-venue observations in their planning contract. Research maps expose immutable replay IDs and compact economic results. Short memory handoffs preserve study/replay IDs and relevant metrics. Older candidate/mission lessons retain retrieval priority, and repeated study versions do not become separate capture entries in planning.

These are tested access and context behaviors. No new real MiMo company cohort consumed this context in the verification, and beneficial learning remains unproven. The subsequent [native-review extension](NATIVE_REVIEW.md) grounds exact replay results in scientific stops and reviewed lessons, with synthetic runtime tests and an actual-receipt/operator-authored temporary-ledger check. Real agent judgment and its effect on allocation remain acceptance work.

```powershell
python -m quantdesk venue-models 37e7854f12d976e8
python -m quantdesk venue-replay CAMPAIGN_ID CANDIDATE_ID STUDY_ID --model-key MODEL_FILENAME_KEY
python -m quantdesk venue-replay-read REPLAY_ID
```

Verification: `python -m pytest -q` passed **316 tests with one Windows native-symlink fixture skipped in 91.93 seconds**. Dedicated synthetic native-replay tests exercise formulas and frozen models through the real numerical worker, partial/residual exposure, causal resets, model/campaign/depth/cutoff guards, global holdout exclusion, cancellation/limits, source/implementation/cache/journal integrity, role permissions and context/ID retrieval. Resolved path-escape guards are separately exercised by the study tests.

Actual receipts are `results/desk/verification/venue-replay/run.json` and `verification.json`. The bounded verifier is idempotent by default; its explicit implementation-recheck flag preserves earlier receipts and does not reset any provider budget. The original MiMo allowance still has 12,691 tokens remaining. Full real company acceptance, independently calibrated execution, profitable confirmation and qualifying forward/portfolio performance remain outstanding.
