"""When is each historical row actually KNOWN in real time? Research must join external data as-of its availability
time, never its label time, or backtests trade on information a live system would not yet have.

available_at_ms = label_time_ms + lag_ms(dataset[, column]). Lags are measured live by desk.live.parity_recorder and
the probes recorded in DATA_PARITY.md; until a lag is verified it is set conservatively (rounded up). Usage:

    feats = asof(feats, metrics_long_or_wide, 'bn_metrics', left_on='ts')
"""
from __future__ import annotations

import polars as pl

# Seconds after the archive's label time at which a live system has the value (p95-ish, rounded up).
LAG_S = {
    'hl_l2book': 2.0,              # book `time` -> WS receipt ~1.0 s median
    'hl_asset_ctxs': 5.0,          # archive row for minute T = snapshot at ~T+3 s (BTC mid 88% exact vs live, 2026-10-01)
    'bn_klines_1m': 63.0,          # label = open_time; bar closes +60 s, REST has it ~1-2 s later
    'bn_trades_1s': 1.0,           # row k covers [k-1 s, k); WS arrival latency is sub-second
    'bn_bookdepth': 2.0,
    'bn_metrics': 900.0,           # fallback when the column is unknown (above every METRIC_LAG_S)
}
# Binance /futures/data publication, relative to the ARCHIVE create_time. Verified 2026-10-02: archive create_time =
# API timestamp - 5 min for OI and long/short ratios, = API timestamp for the taker ratio. Probe (SOLUSDT, 3 periods)
# saw OI / ratios at create_time + 304..453 s and the taker ratio at + 390..512 s; rounded up with margin. The parity
# recorder keeps measuring first_seen_lag_s on 20 symbols - tighten from desk.data.parity reports.
METRIC_LAG_S = {
    'sum_open_interest': 600.0, 'sum_open_interest_value': 600.0,
    'count_toptrader_long_short_ratio': 600.0, 'count_long_short_ratio': 600.0,
    'sum_toptrader_long_short_ratio': 600.0,
    'sum_taker_long_short_vol_ratio': 660.0,
}
TIME_COL = {'hl_l2book': 'time', 'hl_asset_ctxs': 'time', 'bn_klines_1m': 'open_time', 'bn_trades_1s': 'ts',
            'bn_bookdepth': 'timestamp', 'bn_metrics': 'create_time'}


def lag_ms(dataset, column=None):
    if dataset == 'bn_metrics' and column in METRIC_LAG_S:
        return int(METRIC_LAG_S[column] * 1000)
    return int(LAG_S[dataset] * 1000)


def _epoch_ms(df, col):
    dt = df.schema[col]
    if dt == pl.Utf8:
        return pl.col(col).str.to_datetime(time_zone='UTC').dt.epoch('ms')
    if isinstance(dt, pl.Datetime):
        return pl.col(col).dt.epoch('ms')
    return pl.col(col).cast(pl.Int64)


def with_available_at(df, dataset, column=None):
    """Add `available_ms` to a dataset frame (wide: one lag per row, the max over its columns for bn_metrics)."""
    tcol = TIME_COL[dataset]
    if dataset == 'bn_metrics' and column is None:
        cols = [c for c in METRIC_LAG_S if c in df.columns]
        lag = max(lag_ms(dataset, c) for c in cols) if cols else lag_ms(dataset)
    else:
        lag = lag_ms(dataset, column)
    return df.with_columns((_epoch_ms(df, tcol) + lag).alias('available_ms'))


def asof(left, right, dataset, left_on='ts', by=None, column=None, suffix=''):
    """Backward as-of join of `right` (a raw dataset frame) onto `left` by AVAILABILITY time: each left row sees only
    rows a live system would already have received. For bn_metrics wide frames each column is joined on its own lag."""
    left = left.sort(left_on)
    if dataset == 'bn_metrics' and column is None:
        out = left
        cols = [c for c in METRIC_LAG_S if c in right.columns]
        for c in cols:
            r = with_available_at(right.select([TIME_COL[dataset], *(by and [by] or []), c]), dataset, c)
            r = r.drop(TIME_COL[dataset]).rename({c: c + suffix}).sort('available_ms')
            out = out.join_asof(r, left_on=left_on, right_on='available_ms', by=by, strategy='backward', check_sortedness=False).drop('available_ms')
        return out
    r = with_available_at(right, dataset, column).sort('available_ms')
    r = r.rename({c: c + suffix for c in r.columns if c not in ('available_ms', by) and suffix})
    return left.join_asof(r, left_on=left_on, right_on='available_ms', by=by, strategy='backward')
