import polars as pl

from desk.data.availability import METRIC_LAG_S, asof, lag_ms

T0 = 1_790_000_000_000


def test_metrics_not_visible_before_publication():
    m = pl.DataFrame({'create_time': ['2026-09-21 13:55:00', '2026-09-21 14:00:00'], 'symbol': ['SOLUSDT'] * 2,
                      'sum_open_interest': [1.0, 2.0], 'sum_taker_long_short_vol_ratio': [10.0, 20.0]})
    t = int(pl.Series(['2026-09-21 14:00:00']).str.to_datetime(time_zone='UTC').dt.epoch('ms')[0])
    oi_lag, pos_lag = lag_ms('bn_metrics', 'sum_open_interest'), lag_ms('bn_metrics', 'sum_taker_long_short_vol_ratio')
    left = pl.DataFrame({'ts': [t, t + 300_000, t + oi_lag - 1, t + oi_lag, t + pos_lag]}).with_columns(symbol=pl.lit('SOLUSDT'))
    j = asof(left, m, 'bn_metrics', by='symbol')
    # at its label time nothing is known; 13:55 becomes visible at 13:55 + lag; 14:00 exactly at 14:00 + lag, not 1 ms before
    assert j['sum_open_interest'].to_list()[:4] == [None, 1.0, 1.0, 2.0]
    assert j['sum_taker_long_short_vol_ratio'].to_list()[-1] == 20.0
    assert pos_lag > oi_lag == METRIC_LAG_S['sum_open_interest'] * 1000


def test_klines_visible_only_after_bar_close():
    k = pl.DataFrame({'open_time': [T0, T0 + 60_000], 'close': [1.0, 2.0]})
    left = pl.DataFrame({'ts': [T0 + 30_000, T0 + 60_000, T0 + lag_ms('bn_klines_1m'), T0 + 60_000 + lag_ms('bn_klines_1m')]})
    assert asof(left, k, 'bn_klines_1m')['close'].to_list() == [None, None, 1.0, 2.0]


def test_carry_bars_rekeyed_to_close():
    from desk.data.carry_data import HOUR, align_bar_close
    p = pl.DataFrame({'ts': [0, HOUR, 2 * HOUR], 'coin': ['X'] * 3, 'hl_premium': [1.0, 2.0, 3.0],
                      'bn_premium': [10.0, 20.0, 30.0], 'bn_close': [100.0, 200.0, 300.0], 'bn_qvol': [5.0, 6.0, 7.0]})
    a = align_bar_close(p)
    # the bar opened at 0 closes at HOUR: only then is its value known; HL columns are untouched
    assert a['bn_premium'].to_list() == [None, 10.0, 20.0]
    assert a['bn_qvol'].to_list() == [0.0, 5.0, 6.0]
    assert a['hl_premium'].to_list() == [1.0, 2.0, 3.0]


def test_macro_context_visible_next_day_only(tmp_path):
    import types
    from datetime import date
    from desk.data.perps_panel import market_features, HOUR
    (tmp_path / 'macro').mkdir()
    pl.DataFrame({'date': [date(2025, 1, 6), date(2025, 1, 7)], 'symbol': ['^VIX', '^VIX'], 'adjclose': [20.0, 30.0]}) \
        .write_parquet(tmp_path / 'macro' / 'panel.parquet')
    d7 = int(pl.Series([date(2025, 1, 7)]).cast(pl.Datetime('ms')).dt.epoch('ms')[0])
    panel = pl.DataFrame({'ts': [d7 + 22 * HOUR, d7 + 24 * HOUR, d7 + 25 * HOUR], 'coin': ['X'] * 3})
    import desk.data.perps_panel as pp
    orig = pp.dvol; pp.dvol = lambda s: (_ for _ in ()).throw(RuntimeError('offline'))
    try:
        out = market_features(types.SimpleNamespace(store=tmp_path), panel).sort('ts')
    finally:
        pp.dvol = orig
    # Jan-7 close (30) is unknown on Jan-7 evening and at midnight; usable from Jan-8 01:00 UTC
    assert out['vix'].to_list() == [20.0, 20.0, 30.0]
