"""Causal HAR-style expected-|move|-in-bps forecaster (vol-regime gate, T-cfce72/T-d284ad).
Fit on TRAIN days in strategy `fit()`, apply in `signal()`; both calls causal-safe. Usage:
    from desk.lib.contrib.har_move import vol_features, fit_expected_move, expected_move
    def fit(train_df, params):
        models, diag = fit_expected_move(train_df, horizons_min=(params['h'],))
        return models[params['h']]
    def signal(df, params, model):
        em = expected_move(vol_features(df), model)     # predicted |move| in bps at h
        gate = em >= params['min_move_bps']             # e.g. 20bps = 2x 10.2bps RT cost
        return np.where(gate, direction(df), 0.0)
Validated on 60 dev days (train 30 / OOS 30): OOS corr(pred, |fwd move|) = 0.32/0.25/0.17
at 15m/1h/4h; quintile calibration tight (1h top quintile pred 58bps vs realized 58bps).
Also exposes 'exp_30_480' vol-expansion ratio feature (RV_30m/RV_8h)."""
import numpy as np
import polars as pl
import numpy as np

__all__ = ['bar_returns', 'vol_features', 'fit_expected_move', 'expected_move']


def bar_returns(ts, mid, gap_ms=600000):
    """ret[t] = bps return earned during (t-1 -> t]. Zeroed across data gaps."""
    mid = np.asarray(mid, float); ts = np.asarray(ts, np.int64)
    r = np.zeros(len(mid))
    dr = np.diff(mid) / mid[:-1] * 1e4
    dr[np.diff(ts) > gap_ms] = 0.0
    r[1:] = dr
    return r


def vol_features(df, windows=(30, 480, 1440)):
    """Causal realized-vol features from 1m bars (row i uses data <= i). Returns dict
    of np arrays (len n): 'rv_<w>' = sqrt(mean ret^2) over trailing w bars in bps/bar
    (needs >= w/2 samples), 'exp_<short>_<long>' = rv_short/rv_long vol-expansion
    ratio (T-d284ad's RV_30m/RV_8h). Gap-safe: returns zeroed across ts gaps."""
    ts = df['ts'].to_numpy(); mid = df['mid'].to_numpy()
    ret = bar_returns(ts, mid)
    r2 = pl.Series(ret * ret)
    feats = {}
    rvs = {}
    for w in windows:
        rv = np.sqrt(r2.rolling_mean(w, min_samples=max(w // 2, 2)).to_numpy())
        rvs[w] = rv
        feats['rv_%d' % w] = rv
    ws = sorted(windows)
    for a, b in zip(ws[:-1], ws[1:]):
        feats['exp_%d_%d' % (a, b)] = rvs[a] / (rvs[b] + 1e-8)
    return feats


def _feat_matrix(feats, windows):
    cols = [feats['rv_%d' % w] for w in windows]
    return np.column_stack([np.ones_like(cols[0])] + cols)


def fit_expected_move(train_df, horizons_min=(15, 60, 240), windows=(30, 480, 1440)):
    """Walk-forward-safe HAR fit: call inside strategy `fit` on TRAIN days only (the
    harness passes strictly earlier days). OLS of |fwd move in bps| at each horizon
    on [1, rv_30, rv_480, rv_1440], target winsorized at train 99th pct (train-only).
    Returns (models, diag): models = {h: {'coef': (k,), 'winsor': float}} for
    expected_move(); diag = {h: {'n', 'train_corr_pred_absmove', 'mean_pred',
    'mean_absmove'}}. Forward returns here are labels on earlier days -> legal in fit;
    NEVER call this inside signal() (causality test)."""
    import polars as pl  # noqa: F401  (vol_features uses pl)
    feats = vol_features(train_df, windows)
    F = _feat_matrix(feats, windows)
    ts = train_df['ts'].to_numpy(); mid = train_df['mid'].to_numpy(); n = len(ts)
    models = {}
    diag = {}
    for h in horizons_min:
        k = int(h)
        y = np.full(n, np.nan)
        if k < n:
            dr = np.abs((mid[k:] - mid[:-k]) / mid[:-k] * 1e4)
            dr[(ts[k:] - ts[:-k]) != k * 60000] = np.nan
            y[:-k] = dr
        m = np.isfinite(y) & np.isfinite(F).all(axis=1)
        yy = y[m]; FF = F[m]
        winsor = float(np.quantile(yy, 0.99))
        yc = np.minimum(yy, winsor)
        coef, *_ = np.linalg.lstsq(FF, yc, rcond=None)
        pred = np.clip(FF @ coef, 0, None)
        ic = float(np.corrcoef(pred, yy)[0, 1])
        models[int(h)] = {'coef': coef, 'winsor': winsor}
        diag[int(h)] = {'n': int(m.sum()), 'train_corr_pred_absmove': ic,
                        'mean_pred': float(pred.mean()), 'mean_absmove': float(yy.mean())}
    return models, diag


def expected_move(feats, model, windows=(30, 480, 1440)):
    """Predicted |move| in bps at the model's horizon, from vol_features output.
    Fully causal (trailing vol state only). Use as the entry gate from T-cfce72:
    trade only when expected_move >= ~20bps (>= 2x the 10.2bps RT cost)."""
    coef = model['coef']
    F = _feat_matrix(feats, windows[:len(coef) - 1])
    return np.clip(F @ coef, 0, None)
