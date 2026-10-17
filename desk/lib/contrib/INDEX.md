- `from desk.lib.contrib.cost_eval import ...` - Cost-aware evaluation + truth-machine funnel report for 1m-bar strategies. Usage (research sandbox):
    from desk.lib.contrib import cost_eval, horizon_edges, causal_threshold
    df = load(contiguous_days(30), resolution='1m', columns=['ts','mid','funding_rate'])
    sig = my_causal_signal(df)     (by toolsmith)
- `from desk.lib.contrib.har_move import ...` - Causal HAR-style expected-|move|-in-bps forecaster (vol-regime gate, T-cfce72/T-d284ad).
Fit on TRAIN days in strategy `fit()`, apply in `signal()`; both calls causal-safe. Usage:
    from desk.lib.contrib.har_move import vol_features, fit_expected_move, expected_move
    def fit(train_df, params):
 (by toolsmith)
- `from desk.lib.contrib.stress_eval import ...` - Pre-submission robustness + stability slices for strategies (pairs with cost_eval).
Runs the truth-machine stress matrix (fees x1.5, impact x2, +1-row latency, combined)
and the halves/months/long-short slices in two calls:
    from desk.lib.contrib.stress_eval import stress_eval, stability_report
  (by toolsmith)
- `from desk.lib.contrib.sparse_extreme_trigger import ...` - Sparse extreme-episode entry trigger (L-66e538 recipe): causal trailing-quantile gate on |x| (default trailing-2d q99.5 at 1m), first-onset sign entry, cooldown between entries, fixed hold. Includes helper rolling_q(x, w, q) = causal trailing quantile matching np.quantile exactly (also usable as a r (by toolsmith)
- `from desk.lib.contrib.triple_barrier_labels import ...` - Triple-barrier labels for ML fitting (research / fit() on train days ONLY - never inside signal()): per-row first-touch class (+1 upper / -1 lower / 0 vertical) with dynamic vol-scaled barriers (pt/sl_mult x causal trailing return-std x sqrt(max_h)) and gap-aware segmentation (labels never span ts g (by toolsmith)
- `from desk.lib.contrib.val_precheck import ...` - Pre-flight pass-probability for the SEALED validation gates (desk.pipeline.validation.val_gates), estimated from DEV data only (never touches sealed splits). Structural facts it encodes: the val window is ~22 days (val_fraction 0.25), so (1) trades>=40 requires a dev trade rate >= 40/22 = 1.82/day ( (by toolsmith)
- `from desk.lib.contrib.har_sigma import ...` - Causal HAR-style log-vol sizing helper (spec from deepbook-r1 K-74c973): trailing log-variance levels at 60/240/1440m -> NaN-skipping mean (lfc) -> trailing-2d median reference frozen with shift(1) -> mm = clip(exp(0.5*(ref-lfc)), 0.3, 2.5), non-finite -> 1.0. Downsizes in vol spikes (>trailing medi (by toolsmith)
- `from desk.lib.contrib.purged_day_folds import ...` - Embargoed day-level CV generator (fixes the classic leak: random/row-level splits across day edges where rolling windows and labels cross). purged_day_folds(days, n_splits, mode='expanding'|'blocked', embargo_days, min_train_days) -> [{'train': day_ids, 'test': day_ids}]. expanding = walk-forward, t (by toolsmith)
