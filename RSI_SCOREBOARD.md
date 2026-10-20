# RSI Generation Scoreboard — Metric Contract

Honest Recursive Self-Improvement telemetry. **Null means unknown.** Never invent Sharpe, drill, or forward defaults.

## Command

```bash
python -m desk rsi scoreboard
python -m desk rsi campaign --desks macro,pred,carry,perps --no-llm
```

Default campaign desks: `macro,pred,carry,perps`.

## Contract

| Field | Meaning when present | When null |
|-------|----------------------|-----------|
| `best_sharpe` / `mean_sharpe` | Observed from `dev_report.summary.sharpe` | No dev samples |
| `best_val_sharpe` / `best_holdout_sharpe` | Observed from val/holdout reports | No OOS samples |
| `portfolio_sharpe` | ERC portfolio report Sharpe | No portfolio / no sharpe field |
| `planted_signal_corr` / `drill_verdict` / `null_false_discoveries` | From `evaluate_rsi_drill` | Drill not run |
| `hint_resolution_rate` | `resolved_count / hits` on dynamic hints | No hint hits |
| `mean_primitive_lift` | Mean composite lift | No primitives tracked |
| `forward_sharpe` / `forward_verdict` | From `forward_report` | Missing report → `forward_sharpe=null`, `forward_verdict=NOT_RUN` |
| `generation_fitness` | Scalar in `[0,1]` from **learning** signals only | Always computed (0 when no evidence) |

### `generation_fitness` inputs (success)

- `newly_approved` count
- Val / holdout Sharpe of **new** approvals
- Drill pass (`PROMOTED`)
- Hint resolution rate
- Primitive lift

### Explicitly **not** success

- `children_bred` — breeding volume is telemetry only

## Forbidden fabrications

Do **not** fill empty boards with defaults such as:

- Sharpe `1.45` / `1.20` / `best * 0.82`
- Drill `planted_signal_corr=0.89` or `verdict=PROMOTED`
- Forward `holdout * 0.92` or fake `FORWARD_CONFIRMED`
- Bandit mean Sharpe `1.25` / `1.05` when no samples (use neutral Beta priors + null means)

## Related

- Gate reject credit: `GATE_FAILURE` events → `gate_failure_leaderboard`
- Process drills: lookahead leakage, fee-blind, cherry-picked window, sycophantic null-junk
- Charter bars unchanged: planted discovery ≥ 0.70, null FDR = 0, error-rate delta ≤ 0
