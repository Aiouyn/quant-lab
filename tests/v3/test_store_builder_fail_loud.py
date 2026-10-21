"""store_builder: no synthetic fill, frozen splits are never rewritten, provenance sidecars."""
import inspect
import json
import os
import types
from datetime import date, timedelta

import polars as pl
import pytest

from desk.data import carry_data, macro_data, store_builder
from desk.data.provenance import SplitsMismatch, StoreBuildError, freeze_splits, write_provenance
from desk.lab.macro import SPLITS as MACRO_SPLITS


def _settings(tmp_path):
    return types.SimpleNamespace(store=tmp_path / 'store', home=tmp_path, ensure=lambda: None)


def _yahoo(sym, n=300, start=date(2010, 1, 4)):
    d = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({'date': d, 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'adjclose': 1.0,
                         'volume': 1.0, 'adj_open': 1.0, 'symbol': sym})


# ------------------------------------------------------------------ (1) no synthetic fill
def test_no_synthetic_generators_left():
    src = inspect.getsource(store_builder)
    for needle in ('default_rng', 'rng.normal', 'np.resize', 'deterministic_local_fixture', 'calibrated'):
        assert needle not in src, needle


def test_failed_macro_download_raises_and_writes_nothing(tmp_path, monkeypatch):
    frames = {s: _yahoo(s) for s in macro_data.UNIVERSE}
    frames['SPY'] = None
    frames['QQQ'] = _yahoo('QQQ', n=12)
    monkeypatch.setattr(macro_data, 'fetch_universe', lambda *a, **k: frames)
    s = _settings(tmp_path)
    with pytest.raises(StoreBuildError) as e:
        store_builder.build_macro_store(s, force=True)
    msg = str(e.value)
    assert 'Yahoo Finance' in msg and 'SPY (0 rows)' in msg and 'QQQ (12 rows)' in msg and '1970-01-01..' in msg
    assert not (s.store / 'macro' / 'panel.parquet').exists()
    assert not (s.store / 'macro' / 'provenance.json').exists()


def test_remote_disabled_without_cache_raises(tmp_path):
    s = _settings(tmp_path)
    for fn in (store_builder.build_macro_store, store_builder.build_perps_and_carry_stores, store_builder.build_pred_store):
        with pytest.raises(StoreBuildError, match='try_remote=False'):
            fn(s, try_remote=False)
    assert not list((tmp_path / 'store').rglob('*.parquet')) if (tmp_path / 'store').exists() else True


def test_perps_without_hl_ctx_files_raises(tmp_path, monkeypatch):
    from desk.data import perps_panel
    monkeypatch.setattr(perps_panel, 'HLCTX', tmp_path / 'no_hlctx')
    with pytest.raises(StoreBuildError, match='no Hyperliquid asset-ctx files'):
        store_builder.build_perps_and_carry_stores(_settings(tmp_path), force=True)


def test_failed_carry_coin_raises_and_does_not_freeze_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(carry_data, 'universe', lambda n=30: [{'coin': 'BTC', 'binance': 'BTCUSDT'},
                                                              {'coin': 'ETH', 'binance': 'ETHUSDT'}])
    def coin(u, root, start=None):
        if u['coin'] == 'ETH':
            raise ConnectionError('binance 451')
        return {'coin': 'BTC', 'rows': 10}
    monkeypatch.setattr(carry_data, 'build_coin', coin)
    s = _settings(tmp_path)
    with pytest.raises(StoreBuildError) as e:
        carry_data.build_carry_data(s, progress=lambda *_: None)
    assert 'ETH' in str(e.value) and 'binance 451' in str(e.value) and '2023-06-01..' in str(e.value)
    assert not (s.store / 'carry' / 'splits.json').exists()


# ------------------------------------------------------------------ (2) frozen splits
def test_freeze_splits_creates_only_when_absent(tmp_path):
    f = tmp_path / 'splits.json'
    sp = freeze_splits(f, MACRO_SPLITS, 'macro')
    assert f.exists() and sp['dev'] == MACRO_SPLITS['dev'] and 'created' in sp
    before, mtime = f.read_bytes(), f.stat().st_mtime_ns
    freeze_splits(f, dict(MACRO_SPLITS, note='different note is fine'), 'macro')
    assert f.read_bytes() == before and f.stat().st_mtime_ns == mtime


def test_freeze_splits_mismatch_raises_and_keeps_file(tmp_path):
    f = tmp_path / 'splits.json'
    frozen = {'created': 0, 'dev': ['2023-06-01', '2024-01-01'], 'val': ['2024-01-02', '2024-06-01'],
              'holdout': ['2024-06-02', '2024-09-01']}
    f.write_text(json.dumps(frozen))
    before = f.read_bytes()
    bogus_150_110_60 = {'dev': ['2025-10-15', '2026-03-13'], 'val': ['2026-03-14', '2026-07-01'],
                        'holdout': ['2026-07-02', '2026-08-30']}
    with pytest.raises(SplitsMismatch, match='refusing to overwrite'):
        freeze_splits(f, bogus_150_110_60, 'carry')
    assert f.read_bytes() == before


def test_macro_build_respects_existing_frozen_splits(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    (s.store / 'macro').mkdir(parents=True)
    (s.store / 'macro' / 'splits.json').write_text(json.dumps({'dev': ['1999-01-01', '2001-01-01']}))
    monkeypatch.setattr(macro_data, 'fetch_universe', lambda *a, **k: pytest.fail('must not fetch on mismatch'))
    with pytest.raises(SplitsMismatch):
        store_builder.build_macro_store(s, force=True)


def test_carry_splits_stable_when_panel_grows():
    H = 3_600_000
    day0 = 1_700_006_400_000 // 86_400_000 * 86_400_000
    p100 = pl.DataFrame({'ts': [day0 + i * H for i in range(100 * 24)]})
    frozen = carry_data.compute_carry_splits(p100)
    p130 = pl.DataFrame({'ts': [day0 + i * H for i in range(130 * 24)]})
    again = carry_data.compute_carry_splits(p130, frozen)
    assert {k: again[k] for k in ('dev', 'val', 'holdout')} == {k: frozen[k] for k in ('dev', 'val', 'holdout')}


# ------------------------------------------------------------------ (3) provenance
def test_macro_build_writes_provenance_atomically(tmp_path, monkeypatch):
    frames = {s: _yahoo(s) for s in macro_data.UNIVERSE}
    monkeypatch.setattr(macro_data, 'fetch_universe', lambda *a, **k: frames)
    s = _settings(tmp_path)
    res = store_builder.build_macro_store(s, force=True)
    root = s.store / 'macro'
    prov = json.loads((root / 'provenance.json').read_text())
    assert res['status'] == 'built' and prov['source'] == 'yahoo_finance_v8_chart' and prov['verified_real'] is True
    assert prov['rows'] == 300 * len(frames) and prov['assets'] == len(frames)
    assert prov['rows_per_asset']['SPY'] == 300
    assert prov['first'] == '2010-01-04' and prov['last'] == str(date(2010, 1, 4) + timedelta(days=299))
    assert prov['fetched_at_utc'].endswith('+00:00')
    assert len(prov['code_commit']) == 40 or prov['code_commit'] == 'unknown'
    assert json.loads((root / 'splits.json').read_text())['dev'] == MACRO_SPLITS['dev']
    assert not [p for p in root.iterdir() if p.name.endswith('.tmp')]
    if os.name != 'nt':
        assert (root / 'provenance.json').stat().st_mode & 0o777 == 0o644
    # cached rebuild does not refetch or touch anything
    monkeypatch.setattr(macro_data, 'fetch_universe', lambda *a, **k: pytest.fail('cached build must not fetch'))
    assert store_builder.build_macro_store(s)['status'] == 'cached'


def test_provenance_refuses_empty_panel(tmp_path):
    with pytest.raises(StoreBuildError):
        write_provenance(tmp_path, 'pred', 'x', pl.DataFrame({'ts': [], 'market': []}), 'ts', 'market')
