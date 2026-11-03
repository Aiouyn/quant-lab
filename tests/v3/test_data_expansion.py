"""New research data: point-in-time Polymarket volume (pred) and FRED / CFTC COT features (macro). Offline: every
network call is replaced by a fake."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from desk.data import macro_alt, polymarket, pred_volume
from desk.lab.macro import MacroCost, run_macro
from desk.lab.pred import visible as pred_visible

H = pred_volume.HOUR


# ------------------------------------------------------------------ pred volume
def _ev(i, ts, usd, yes_is_maker=True, tok='T'):
    amt = str(int(usd * 1e6))
    if yes_is_maker:
        return {'id': f'e{i}', 'timestamp': str(ts), 'makerAssetId': tok, 'takerAssetId': '0',
                'makerAmountFilled': '5', 'takerAmountFilled': amt}
    return {'id': f'e{i}', 'timestamp': str(ts), 'makerAssetId': '0', 'takerAssetId': tok,
            'makerAmountFilled': amt, 'takerAmountFilled': '5'}


def _fake_post(events, page):
    """Mimics the subgraph's filters/ordering for the two query shapes fetch_fills sends."""
    def post(c, query, v):
        side = 'makerAssetId' if 'makerAssetId: $tok' in query else 'takerAssetId'
        rows = [e for e in events if e[side] == v['tok']]
        if 'id_gt' in query:
            rows = sorted([e for e in rows if int(e['timestamp']) == int(v['ts']) and e['id'] > v['id']], key=lambda e: e['id'])
        else:
            rows = sorted([e for e in rows if int(e['timestamp']) >= int(v['ts'])], key=lambda e: (int(e['timestamp']), e['id']))
        return rows[:page]
    return post


def test_fetch_fills_pages_through_crowded_seconds(monkeypatch):
    monkeypatch.setattr(pred_volume, 'PAGE', 3)
    ev = [_ev(f'a{i:02d}', 100, 1.0) for i in range(7)]                       # 7 fills in ONE second (> page)
    ev += [_ev(f'b{i:02d}', 200 + i, 2.0, yes_is_maker=False) for i in range(5)]
    ev += [_ev('zz', 9000, 10.0, tok='OTHER')]
    got = pred_volume.fetch_fills(None, 'T', post=_fake_post(ev, 3))
    assert len(got) == 12
    assert sum(u for _, u in got) == pytest.approx(7 * 1.0 + 5 * 2.0)


def test_fill_usd_takes_the_usdc_leg():
    assert pred_volume.fill_usd(_ev(1, 0, 12.5)) == pytest.approx(12.5)
    assert pred_volume.fill_usd(_ev(2, 0, 3.0, yes_is_maker=False)) == pytest.approx(3.0)


def _write_cache(root, tok, hours_usd):
    f = pred_volume.cache_path(root, tok)
    f.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({'ts': [h * H for h, _ in hours_usd], 'usd': [u for _, u in hours_usd], 'n': [1] * len(hours_usd)},
                 schema=pred_volume.SCHEMA).write_parquet(f)


def test_point_in_time_universe_drops_hours_before_volume_crossed(tmp_path):
    root = tmp_path
    tokA, tokB = 'A' * 30, 'B' * 30
    # A: 1M final volume, fills 10/h for hours 0..9 (total 100) -> crosses 25% after hour 2's fills, visible at hour 3
    _write_cache(root, tokA, [(h, 10.0) for h in range(10)])
    # B: no CLOB fills (empty history) -> can't date membership -> dropped
    pl.DataFrame(schema=pred_volume.SCHEMA).write_parquet(pred_volume.cache_path(root, tokB))
    meta = pl.DataFrame({'yes_token': [tokA, tokB], 'volume': [1_000_000.0, 400_000.0]})
    act, entry = pred_volume.volume_frame(root, meta, 250_000.0)
    assert entry.filter(pl.col('yes_token') == tokA)['entry_ts'].item() == 3 * H
    assert tokB not in entry['yes_token'].to_list()
    panel = pl.DataFrame({'ts': [h * H for h in range(12)] * 2, 'yes_token': [tokA] * 12 + [tokB] * 12,
                          'market': [1] * 12 + [2] * 12, 'p': [0.5] * 24})
    out = pred_volume.apply_point_in_time(panel.lazy(), act, entry).collect().sort('ts')
    assert out['market'].unique().to_list() == [1]
    assert out['ts'].min() == 3 * H
    row = out.filter(pl.col('ts') == 5 * H)
    assert row['vol_usd_1h'].item() == pytest.approx(10.0)              # fills of hour 4, seen at hour 5
    assert row['vol_usd_cum'].item() == pytest.approx(50.0)             # hours 0..4
    assert out.filter(pl.col('ts') == 11 * H)['vol_usd_cum'].item() == pytest.approx(100.0)  # carried forward
    assert out.filter(pl.col('ts') == 11 * H)['vol_usd_1h'].item() == 0.0


def test_pit_universe_waits_for_backfill_coverage(tmp_path, monkeypatch):
    toks = [f'{i:030d}' for i in range(100)]
    sel = pl.DataFrame({'yes_token': toks})
    for t in toks[:90]:
        _write_cache(tmp_path, t, [(0, 1.0)])
    assert polymarket.point_in_time_status(tmp_path, sel)['applied'] is False
    for t in toks[90:]:
        _write_cache(tmp_path, t, [(0, 1.0)])
    assert polymarket.point_in_time_status(tmp_path, sel)['applied'] is True
    monkeypatch.setenv('PRED_PIT_UNIVERSE', '0')
    assert polymarket.point_in_time_status(tmp_path, sel)['applied'] is False


def test_pred_strategies_see_activity_but_not_final_volume():
    p = pl.DataFrame({'ts': [0], 'market': [1], 'p': [0.5], 'end_ms': [H], 'outcome': [1], 'category': ['x'],
                      'volume': [1e6], 'vol_usd_1h': [3.0], 'vol_usd_cum': [9.0]})
    v = pred_visible(p)
    assert 'volume' not in v.columns and 'outcome' not in v.columns
    assert {'vol_usd_1h', 'vol_usd_cum'} <= set(v.columns)


# ------------------------------------------------------------------ macro alt data
def test_fred_csv_parse_and_two_business_day_lag():
    text = 'observation_date,DGS10\n2024-01-04,4.0\n2024-01-05,.\n2024-01-08,4.1\n'
    df = macro_alt.parse_fred_csv(text, 'DGS10')
    assert df['value'].to_list() == [4.0, 4.1]
    assert macro_alt.bday_add(date(2024, 1, 4), 2) == date(2024, 1, 8)    # Thu -> Mon (skips weekend)
    assert macro_alt.bday_add(date(2024, 1, 5), 2) == date(2024, 1, 9)


def test_cot_net_positioning_visible_one_week_later():
    rows = [{'cftc_contract_market_code': '13874A', 'report_date_as_yyyy_mm_dd': '2024-01-02T00:00:00.000',
             'open_interest_all': '1000', 'noncomm_positions_long_all': '300', 'noncomm_positions_short_all': '100',
             'comm_positions_long_all': '400', 'comm_positions_short_all': '600'}]
    long = macro_alt.cot_long(macro_alt.cot_frame(rows))
    spec = long.filter(pl.col('symbol') == 'COT:ES:spec_net_oi')
    assert spec['value'].item() == pytest.approx(0.2)
    assert spec['avail'].item() == date(2024, 1, 9)
    assert long.filter(pl.col('symbol') == 'COT:ES:comm_net_oi')['value'].item() == pytest.approx(-0.2)


def test_alt_rows_respect_availability_and_go_stale():
    days = pl.Series([date(2024, 1, 1) + timedelta(days=i) for i in range(40)])
    alt = pl.DataFrame({'symbol': ['FRED:X'] * 2, 'cls': ['alt_rates'] * 2,
                        'obs_date': [date(2024, 1, 1), date(2024, 1, 2)],
                        'avail': [date(2024, 1, 3), date(2024, 1, 4)], 'value': [1.0, 2.0]})
    rows = macro_alt.to_panel_rows(alt, days).sort('date')
    assert rows['date'].min() == date(2024, 1, 3)                       # nothing before it was published
    assert rows.filter(pl.col('date') == date(2024, 1, 3))['close'].item() == 1.0
    assert rows.filter(pl.col('date') == date(2024, 1, 10))['close'].item() == 2.0
    assert rows['date'].max() == date(2024, 1, 14)                      # stale after 10 days: series disappears
    assert set(rows['tradable'].to_list()) == {0}


def test_failed_source_falls_back_to_cache_and_never_synthesizes(tmp_path, monkeypatch):
    good = pl.DataFrame({'obs_date': [date(2020, 1, 1) + timedelta(days=i) for i in range(60)],
                         'avail': [date(2020, 1, 3) + timedelta(days=i) for i in range(60)],
                         'value': [float(i) for i in range(60)]})
    monkeypatch.setattr(macro_alt, 'FRED_DAILY', {'AAA': 'a', 'BBB': 'b'})
    monkeypatch.setattr(macro_alt, 'COT_MARKETS', {})
    monkeypatch.setattr(macro_alt, 'fred_daily', lambda c, sid: good)
    alt, rep = macro_alt.fetch_all(tmp_path)
    assert rep['FRED:AAA'] == 60 and alt['symbol'].n_unique() == 2

    def boom(c, sid):
        if sid == 'AAA':
            raise RuntimeError('down')
        return good.head(5)                                            # too short -> rejected, no cache for it
    (tmp_path / 'FRED_BBB.parquet').unlink()
    monkeypatch.setattr(macro_alt, 'fred_daily', boom)
    alt, rep = macro_alt.fetch_all(tmp_path)
    assert rep['FRED:AAA'].startswith('cached')
    assert rep['FRED:BBB'].startswith('missing')
    assert alt['symbol'].unique().to_list() == ['FRED:AAA']


def test_alt_feature_rows_do_not_change_macro_pnl():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, 30))
    base = pl.DataFrame({'date': days, 'symbol': ['SPY'] * 30, 'close': px, 'adjclose': px, 'cls': ['equity_us'] * 30,
                         'tradable': [1] * 30, 'cost_bps': [1.5] * 30})
    alt = pl.DataFrame({'symbol': ['FRED:DGS10'] * 30, 'cls': ['alt_rates'] * 30, 'obs_date': days,
                        'avail': days, 'value': rng.normal(4, 0.1, 30)})
    merged = macro_alt.merge_into_panel(base, macro_alt.to_panel_rows(alt, base['date']))
    assert len(merged) > len(base)
    w_base = np.ones(len(base)) * 0.5
    w_merged = np.where(merged['symbol'] == 'SPY', 0.5, 0.9)           # weights on feature rows must be ignored
    a = run_macro(base, w_base, MacroCost())
    b = run_macro(merged, w_merged, MacroCost())
    assert np.allclose(a['pnl'], b['pnl'])
