"""Dev-view funding isolation and atomic funding/meta writes (audit #7)."""
import json
import os
from datetime import datetime, timezone

import polars as pl
import pytest

from desk.data import atomic
from desk.data.atomic import atomic_write_json, atomic_write_parquet
from desk.data.store import FeatureStore

DAYS = [f'2025-11-{d:02d}' for d in range(1, 11)]


def _ms(day, hour=0):
    return int(datetime.fromisoformat(day).replace(hour=hour, tzinfo=timezone.utc).timestamp() * 1000)


def _funding():
    hours = [_ms(d, h) for d in DAYS for h in range(24)]
    return pl.DataFrame({'ts': hours, 'funding_rate': [1e-5] * len(hours), 'premium': [0.0] * len(hours), 'hour': hours})


@pytest.fixture
def store(tmp_path):
    class S:
        store = tmp_path; data_dir = tmp_path; dev_fraction = 0.6; val_fraction = 0.25
    for sub in ('1s', '10s', '1m', 'meta'):
        (tmp_path / sub).mkdir()
    for d in DAYS:
        pl.DataFrame({'ts': [_ms(d)], 'mid': [1.0], 'valid': [True]}).write_parquet(tmp_path / '1s' / f'{d}.parquet')
    (tmp_path / 'splits.json').write_text(json.dumps({'created': 0, 'dev': DAYS[:6], 'val': DAYS[6:8], 'holdout': DAYS[8:], 'forward': []}))
    _funding().write_parquet(tmp_path / 'funding_hl.parquet')
    return FeatureStore(S(), root=tmp_path)


def test_dev_view_funding_excludes_val_holdout(store):
    view = store.build_dev_view()
    src, dst = store.root / 'funding_hl.parquet', view / 'funding_hl.parquet'
    assert not os.path.samefile(src, dst), 'dev funding must be a filtered copy, not a hard link'
    got = pl.read_parquet(dst)
    assert got['hour'].max() < _ms(DAYS[6])            # nothing from val/holdout/forward
    assert got['hour'].min() == _ms(DAYS[0])
    assert got.height == 6 * 24
    leaked = set(pl.read_parquet(src).filter(pl.col('hour') >= _ms(DAYS[6]))['hour']) & set(got['hour'])
    assert not leaked


def test_dev_view_rebuild_refreshes_filtered_copy(store):
    store.build_dev_view()
    view = store.build_dev_view()
    assert pl.read_parquet(view / 'funding_hl.parquet').height == 6 * 24


def test_atomic_write_keeps_original_on_failure(tmp_path, monkeypatch):
    f = tmp_path / 'funding_hl.parquet'
    orig = pl.DataFrame({'hour': [1, 2], 'funding_rate': [0.1, 0.2]})
    atomic_write_parquet(orig, f)

    class Boom(pl.DataFrame):
        def write_parquet(self, file, **kw):
            open(file, 'wb').write(b'partial garbage')
            raise OSError('disk full')
    with pytest.raises(OSError):
        atomic_write_parquet(Boom({'hour': [9]}), f)
    assert pl.read_parquet(f).equals(orig)
    assert [p.name for p in tmp_path.iterdir()] == ['funding_hl.parquet']   # temp cleaned up

    j = tmp_path / 'meta.json'
    atomic_write_json(j, {'a': 1})
    monkeypatch.setattr(atomic.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('nope')))
    with pytest.raises(OSError):
        atomic_write_json(j, {'a': 2})
    assert json.loads(j.read_text()) == {'a': 1}


def test_atomic_write_does_not_mutate_hard_links(tmp_path):
    f = tmp_path / 'funding_hl.parquet'
    orig = pl.DataFrame({'hour': [1, 2], 'funding_rate': [0.1, 0.2]})
    atomic_write_parquet(orig, f)
    link = tmp_path / 'linked.parquet'
    os.link(f, link)
    atomic_write_parquet(pl.DataFrame({'hour': [3], 'funding_rate': [0.3]}), f)
    assert pl.read_parquet(link).equals(orig)
    assert pl.read_parquet(f)['hour'].to_list() == [3]
    assert not os.path.samefile(f, link)
