# RSI Generation Scoreboard — Metric Contract

Honest Recursive Self-Improvement telemetry. **Null means unknown.** Never invent Sharpe, drill, or forward defaults.

## Command

```bash
python -m desk rsi scoreboard        # this contract
python -m desk rsi status            # KPIs used by the brief and the dashboard
python -m desk rsi compile-hints     # cluster recurring sandbox failures into fix hints
python -m desk rsi quarantine        # dry run: rows the retired RSI engine wrote into the ledger
python -m desk rsi quarantine --apply
```

The campaign, evolve, toolsmith, bandit, funnel and daemon commands were retired on 2026-11-02 with the engine that
backed them (it evaluated on synthetic panels and invented metrics). They come back only behind a drill gate.

## Contract

| Field | Meaning when present | When null |
|-------|----------------------|-----------|
| `best_sharpe` / `mean_sharpe` | Observed from `dev_report.summary.sharpe` | No dev samples |
| `best_val_sharpe` / `best_holdout_sharpe` | Observed from val/holdout reports | No OOS samples |
| `portfolio_sharpe` | ERC portfolio report Sharpe | No portfolio / no sharpe field |
| `planted_signal_corr` / `drill_verdict` / `null_false_discoveries` | From the `drill_scores` ledger key, written only by a real drill report | Drill not run |
| `hint_resolution_rate` | `resolved_count / hits` on dynamic hints | No hint hits |
| `mean_primitive_lift` | Causal lift from drill ablations | Always null until a drill ablation estimator exists |
| `forward_sharpe` / `forward_verdict` | From `forward_report` | Missing report → `forward_sharpe=null`, `forward_verdict=NOT_RUN` |
| `forward_confirmed` / `forward_decided` | Approved strategies with a 60-day forward verdict | 0 |
| `generation_fitness` | `forward_confirmed / forward_decided` | 0 when nothing is decided yet |

### `generation_fitness` (Phase 0 definition)

Only evidence the agents cannot move counts: confirmations on forward data nobody had when the strategy was written.
Planted / null drill results join it once drills exist for every live desk.

### Explicitly **not** success

- `children_bred`: breeding volume is telemetry only
- hint resolution rate and primitive usage: useful telemetry, but agents influence both, so they are reported and
  excluded from fitness
- validation or holdout Sharpe: rewarding them would turn the sealed splits into a training set

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
