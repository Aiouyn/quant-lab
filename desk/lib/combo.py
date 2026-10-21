"""Factor combination engine: causal base features + library factors -> walk-forward model -> selective positions.

Used by generated strategy modules (see desk.company.tools.combine_factors). Everything here is causal: features
use only past rows; labels are built ONLY inside fit() on strictly earlier days (the harness enforces that).
"""
from __future__ import annotations

import warnings
import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import polars as pl

BASE = ('qi10_avg', 'qi5_avg', 'micro_dev_bps_avg', 'ofi10_15', 'ofi10_60', 'flow_imb_15', 'flow_imb_60',
        'depth_asym_25', 'depth_asym_100', 'ret_5', 'ret_15', 'ret_60', 'ret_240', 'vol_60', 'vol_ratio',
        'spread_avg', 'impact_asym_10', 'trade_intensity', 'tod_sin', 'tod_cos')


def _z(s: pl.Series, w=1440, minp=240):
    m = s.rolling_mean(w, min_samples=minp)
    sd = s.rolling_std(w, min_samples=minp)
    return ((s - m) / sd).fill_nan(0).fill_null(0).clip(-6, 6)


def base_features(df: pl.DataFrame) -> np.ndarray:
    """Causal, scale-free features on 1m bars (each z-scored with a trailing 1-day window)."""
    col = lambda c, d=0.0: df[c].cast(pl.Float64) if c in df.columns else pl.Series(np.full(len(df), d))
    lr = df['mid'].cast(pl.Float64).log().diff().fill_null(0)
    depth = (col('bdep_25') + col('adep_25')).clip(lower_bound=1.0)
    vol = col('buy_sz') + col('sell_sz')
    feats = {
        'qi10_avg': col('qi10_avg'), 'qi5_avg': col('qi5_avg'), 'micro_dev_bps_avg': col('micro_dev_bps_avg'),
        'ofi10_15': (col('ofi10') / depth).rolling_sum(15, min_samples=5),
        'ofi10_60': (col('ofi10') / depth).rolling_sum(60, min_samples=20),
        'flow_imb_15': (col('buy_sz') - col('sell_sz')).rolling_sum(15, min_samples=5) / (vol.rolling_sum(15, min_samples=5) + 1e-9),
        'flow_imb_60': (col('buy_sz') - col('sell_sz')).rolling_sum(60, min_samples=20) / (vol.rolling_sum(60, min_samples=20) + 1e-9),
        'depth_asym_25': (col('bdep_25') - col('adep_25')) / (col('bdep_25') + col('adep_25') + 1e-9),
        'depth_asym_100': (col('bdep_100') - col('adep_100')) / (col('bdep_100') + col('adep_100') + 1e-9),
        'ret_5': lr.rolling_sum(5, min_samples=1), 'ret_15': lr.rolling_sum(15, min_samples=1),
        'ret_60': lr.rolling_sum(60, min_samples=1), 'ret_240': lr.rolling_sum(240, min_samples=1),
        'vol_60': lr.rolling_std(60, min_samples=10),
        'vol_ratio': lr.rolling_std(15, min_samples=5) / (lr.rolling_std(240, min_samples=30) + 1e-12),
        'spread_avg': col('spread_bps_avg', 0.01), 'impact_asym_10': col('aimp_10_avg') - col('bimp_10_avg'),
        'trade_intensity': (col('buy_n') + col('sell_n')).rolling_sum(15, min_samples=5),
    }
    out = [_z(feats[k]).to_numpy() for k in BASE if k in feats]
    tod = (df['ts'].to_numpy() % 86_400_000) / 86_400_000 * 2 * np.pi
    out += [np.sin(tod), np.cos(tod)]
    return np.column_stack(out)


def factor_matrix(df, factor_codes):
    cols = []
    for code in factor_codes:
        # library factor modules are agent-authored: evaluate them in the OS jail, not exec() in this process
        from desk.lab.factor_jail import run as run_factor
        v = run_factor(code, [df])[1][0]
        cols.append(_z(pl.Series(np.nan_to_num(v))).to_numpy())
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def design(df, factor_codes, use_base=True):
    parts = []
    if use_base:
        parts.append(base_features(df))
    if factor_codes:
        parts.append(factor_matrix(df, factor_codes))
    X = np.column_stack(parts) if parts else np.zeros((len(df), 0))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def labels(df, h):
    """Forward bps return from the next bar (decision latency) to h bars later. ONLY call inside fit()."""
    m = df['mid'].to_numpy().astype(float)
    ts = df['ts'].to_numpy()
    y = np.full(len(m), np.nan)
    if len(m) > h + 1:
        y[:-h - 1] = (m[h + 1:] / m[1:-h] - 1) * 1e4
        gap = np.full(len(m), np.inf); gap[:-h - 1] = ts[h + 1:] - ts[1:-h]
        y[gap > (h * 60_000 + 120_000)] = np.nan
    return y


def fit_model(df, factor_codes, h, model='lgbm', use_base=True, stride=None):
    X = design(df, factor_codes, use_base)
    y = labels(df, h)
    ok = np.isfinite(y)
    step = stride or max(1, h // 3)                       # thin overlapping labels
    idx = np.flatnonzero(ok)[::step]
    X, y = X[idx], np.clip(y[idx], -np.nanpercentile(np.abs(y[idx]), 99), np.nanpercentile(np.abs(y[idx]), 99))
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=50.0).fit(X, y)
    else:
        import lightgbm as lgb
        mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                verbose=-1, n_jobs=2).fit(X, y)
    resid = y - mdl.predict(X)
    return {'model': mdl, 'resid_sd': float(np.std(resid))}


def positions(df, factor_codes, fitted, edge_bps, hold, use_base=True):
    """Enter when the predicted move exceeds edge_bps; hold at least `hold` bars; exit when the prediction flips."""
    pred = fitted['model'].predict(design(df, factor_codes, use_base))
    out = np.zeros(len(pred)); pos = 0.0; held = 0
    for i in range(len(pred)):
        held += 1
        if pos == 0.0:
            if pred[i] > edge_bps:
                pos, held = 1.0, 0
            elif pred[i] < -edge_bps:
                pos, held = -1.0, 0
        elif held >= hold and (np.sign(pred[i]) != pos or abs(pred[i]) < edge_bps * 0.25):
            pos = 0.0
        out[i] = pos
    return out


TEMPLATE = '''RESOLUTION = '1m'
LOOKBACK = 1500
TRAIN_DAYS = {train_days}
REFIT_DAYS = {refit_days}
PARAMS = {{'h': {h}, 'edge_bps': {edge}, 'hold': {hold}}}
{execution}FACTOR_CODES = {codes!r}
MODEL = {model!r}
USE_BASE = {use_base!r}
from desk.lib.combo import fit_model, positions


def fit(train_df, params):
    return fit_model(train_df, FACTOR_CODES, int(params['h']), MODEL, USE_BASE)


def signal(df, params, model):
    return positions(df, FACTOR_CODES, model, float(params['edge_bps']), int(params['hold']), USE_BASE)
'''


def make_strategy(factor_codes=(), h=15, edge_bps=None, hold=None, model='lgbm', use_base=True, maker=True,
                  train_days=40, refit_days=10, ttl_s=120, exit='maker'):
    """Generate a combined strategy module. Default edge = round-trip cost + margin for the chosen execution."""
    edge = edge_bps if edge_bps is not None else (5.0 if maker else 12.0)
    ex = (f"EXECUTION = {{'mode': 'maker', 'ttl_s': {int(ttl_s)}, 'on_expiry': 'cross', 'repeg_s': 10, 'exit': {exit!r}}}\n"
          if maker else '')
    return TEMPLATE.format(train_days=int(train_days), refit_days=int(refit_days), h=int(h), edge=float(edge),
                           hold=int(hold or h), execution=ex, codes=list(factor_codes), model=model, use_base=bool(use_base))
