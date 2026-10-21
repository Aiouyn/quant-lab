"""Per-split store views (dev / dev+val / one-shot holdout): no data beyond the allowed splits, pre-built splits.json,
and reading a view through every harness store under a read-only mount writes nothing."""
import json
import os
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from desk.data import views

H, D = 3_600_000, 86_400_000
DAYS = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(25)]
SPL = {'dev': DAYS[:10], 'val': DAYS[10:15], 'holdout': DAYS[15:20], 'forward': DAYS[20:]}
SENT = {'dev': 1.0, 'val': 222.0, 'holdout': 333.0, 'forward': 444.0}     # sentinel value per split


def _ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def split_of_ms(t):
    d = datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat()
    return next((k for k, v in SPL.items() if d in v), 'forward')


@pytest.fixture
def store(tmp_path):
    st = tmp_path / 'store'
    rng_ = {k: [v[0], v[-1]] for k, v in SPL.items()}
    for res, step in (('1s', 1000), ('10s', 10_000), ('1m', 60_000)):
        (st / res).mkdir(parents=True)
        for k, ds in SPL.items():
            for d in ds:
                ts = _ms(d) + step * np.arange(5)
                pl.DataFrame({'ts': ts, 'valid': np.ones(5, np.int8), 'mid': np.full(5, SENT[k])}).write_parquet(st / res / f'{d}.parquet')
    (st / 'meta').mkdir()
    (st / 'splits.json').write_text(json.dumps({'created': 0, **SPL}))
    hours = np.arange(_ms(DAYS[0]), _ms(DAYS[-1]) + D, H)
    pl.DataFrame({'hour': hours, 'funding_rate': [SENT[split_of_ms(h)] for h in hours]}).write_parquet(st / 'funding_hl.parquet')
    for desk in ('macro', 'perps', 'pred', 'carry'):
        (st / desk).mkdir()
        (st / desk / 'splits.json').write_text(json.dumps({'created': 0, **rng_}))
    dates = [date.fromisoformat(d) for d in DAYS]
    pl.DataFrame({'date': dates, 'symbol': 'SPY', 'close': [SENT[split_of_ms(_ms(d.isoformat()))] for d in dates]}
                 ).write_parquet(st / 'macro' / 'panel.parquet')
    pl.DataFrame({'ts': hours, 'coin': 'BTC', 'px': [SENT[split_of_ms(h)] for h in hours], 'f': 0.0,
                  'lo': [SENT[split_of_ms(h + H)] for h in hours], 'hi': [SENT[split_of_ms(h + H)] for h in hours]}
                 ).write_parquet(st / 'perps' / 'panel.parquet')
    rows = []
    for i, d in enumerate(DAYS[2::3]):                    # markets resolving in every split
        end = _ms(d) + 12 * H
        for t in range(_ms(DAYS[0]), end, 6 * H):
            rows.append({'ts': t, 'market': i, 'p': SENT[split_of_ms(end)] / 1000, 'end_ms': end, 'category': 'x',
                         'outcome': 1, 'volume': 1.0})
    pl.DataFrame(rows).write_parquet(st / 'pred' / 'panel.parquet')
    (st / 'carry' / 'coins').mkdir()
    pl.DataFrame({'ts': hours, 'coin': 'BTC', 'hl_funding': [SENT[split_of_ms(h)] for h in hours], 'hl_premium': 0.0,
                  'bn_funding': 0.0, 'bn_rate_last': 0.0, 'bn_interval_h': 8.0,
                  'bn_premium': [SENT[split_of_ms(h + H)] for h in hours],          # bar keyed by open: closes at h+1h
                  'bn_close': [SENT[split_of_ms(h + H)] for h in hours], 'bn_qvol': 1.0}
                 ).write_parquet(st / 'carry' / 'coins' / 'BTC.parquet')
    (st / 'carry' / 'universe.json').write_text(json.dumps([{'coin': 'BTC', 'binance': 'BTCUSDT'}]))
    return st


def _numbers(view: Path):
    seen = set()
    for f in view.rglob('*.parquet'):
        df = pl.read_parquet(f)
        for c in df.columns:
            if df[c].dtype.is_numeric():
                seen |= set(float(v) for v in df[c].drop_nulls().to_list())
    return seen


def _forbidden(allowed):
    bad = [k for k in views.ORDER if k not in allowed]
    return {SENT[k] for k in bad} | {SENT[k] / 1000 for k in bad}


def _max_date(view: Path):
    last = []
    for f in view.rglob('*.parquet'):
        df = pl.read_parquet(f)
        for c in ('ts', 'hour'):
            if c in df.columns and len(df):
                last.append(datetime.fromtimestamp(df[c].max() / 1000, timezone.utc).date().isoformat())
        if 'date' in df.columns and len(df):
            last.append(df['date'].max().isoformat())
    return max(last)


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val']])
def test_view_has_no_data_outside_allowed_splits(store, allowed):
    v = views.view_for_allowed(store, allowed)
    assert v.parent == views.views_root(store) and v.name.startswith('+'.join(allowed) + '@')
    assert not (_numbers(v) & _forbidden(allowed)), _numbers(v) & _forbidden(allowed)
    assert _max_date(v) <= SPL[allowed[-1]][-1]
    days = sorted(p.stem for p in (v / '1s').glob('*.parquet'))
    assert days == sorted(d for k in allowed for d in SPL[k])
    assert os.path.samefile(v / '1s' / f'{DAYS[0]}.parquet', store / '1s' / f'{DAYS[0]}.parquet')   # immutable day: hard link


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val']])
def test_splits_json_prebuilt_and_correct(store, allowed):
    v = views.view_for_allowed(store, allowed)
    fs = json.loads((v / 'splits.json').read_text())
    for k in views.ORDER:
        assert fs[k] == (SPL[k] if k in allowed else [])
    for desk in ('macro', 'perps', 'pred', 'carry'):
        sp = json.loads((v / desk / 'splits.json').read_text())
        assert {k for k in views.ORDER if k in sp} == set(allowed)
        assert all(sp[k] == [SPL[k][0], SPL[k][-1]] for k in allowed)
    assert all(oct(f.stat().st_mode & 0o777) == oct(0o644) for f in v.rglob('*') if f.is_file() and f.suffix == '.json')


def _make_ro(root: Path):
    for p in sorted(root.rglob('*'), reverse=True) + [root]:
        os.chmod(p, (p.stat().st_mode & ~0o222))


def _make_rw(root: Path):
    for p in [root] + sorted(root.rglob('*')):
        os.chmod(p, p.stat().st_mode | stat.S_IWUSR)


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob('*'))


@pytest.mark.skipif(os.name == 'nt' or (hasattr(os, 'geteuid') and os.geteuid() == 0), reason='needs POSIX non-root')
@pytest.mark.parametrize('allowed', [('dev',), ('dev', 'val')])
def test_read_only_view_needs_no_writes(store, allowed):
    import dataclasses
    from desk.config import SETTINGS
    from desk.data.store import FeatureStore
    from desk.lab.carry import CarryStore
    from desk.lab.macro import MacroStore
    from desk.lab.perps import PerpsStore
    from desk.lab.pred import PredStore
    v = views.view_for_allowed(store, list(allowed))
    s = dataclasses.replace(SETTINGS, store_dir=v)
    before = _tree(v)
    _make_ro(v)
    try:
        last = allowed[-1]
        fsx = FeatureStore(s, allowed=allowed)
        df = fsx.load(fsx.days(last), '1m')
        assert df['funding_rate'].max() <= max(SENT[k] for k in allowed)
        for St in (MacroStore, PerpsStore, PredStore):
            St(s, allowed=allowed).splits()
        MacroStore(s, allowed=allowed).load(last, warmup_days=0)
        PerpsStore(s, allowed=allowed).load(last, warmup_days=0)
        PredStore(s, allowed=allowed).load(last)
        CarryStore(s, allowed=allowed).load(last, warmup_days=0)
        assert views.view_for_allowed(store, list(allowed)) == v            # reuse: no rebuild, no write
    finally:
        _make_rw(v)
    assert _tree(v) == before


def test_one_shot_holdout_view_is_consumed(store):
    with views.store_for_job(store, {'allowed': ['dev', 'val', 'holdout']}, job_id='promo1') as v:
        nums = _numbers(v)
        assert SENT['holdout'] in nums and SENT['forward'] not in nums and SENT['forward'] / 1000 not in nums
        assert _max_date(v) <= SPL['holdout'][-1]
        assert json.loads((v / 'splits.json').read_text())['forward'] == []
        _make_ro(v)                                          # teardown must cope with a read-only tree
    assert not v.exists()
    assert views.holdout_consumed(store, 'promo1')
    with pytest.raises(views.ViewError, match='already consumed'):
        with views.one_shot_holdout(store, 'promo1'):
            pass


def test_holdout_view_torn_down_when_job_fails(store):
    with pytest.raises(RuntimeError):
        with views.one_shot_holdout(store, 'promo2') as v:
            raise RuntimeError('job crashed')
    assert not v.exists() and views.holdout_consumed(store, 'promo2')


def test_allowed_must_be_a_prefix_and_forward_gets_full_store(store):
    with pytest.raises(views.ViewError):
        views.view_for_allowed(store, ['val'])
    with pytest.raises(views.ViewError):
        views.view_for_allowed(store, ['dev', 'val', 'holdout'])     # holdout only via the one-shot path
    assert views.view_for_allowed(store, ['dev', 'val', 'holdout', 'forward']) == store


def test_view_rebuilt_as_new_version_when_store_changes(store):
    v1 = views.view_for_allowed(store, ['dev'])
    p = store / 'macro' / 'panel.parquet'
    pl.read_parquet(p).write_parquet(p)                     # source rewritten
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    v2 = views.view_for_allowed(store, ['dev'])
    assert v2 != v1 and v1.exists() and v2.exists()        # running jobs keep their version


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val'], ['dev', 'val', 'holdout']])
def test_run_job_hands_the_harness_only_its_view(store, tmp_path, monkeypatch, allowed):
    import asyncio
    import types
    from desk.lab import sandbox
    seen = {}

    async def fake_exec(*argv, **kw):
        job = json.loads(Path(argv[-2]).read_text())
        seen['store'] = Path(job['_store'])
        seen['exists'] = seen['store'].exists()
        Path(argv[-1]).write_text(json.dumps({'ok': True}))
        async def communicate():
            return b'', b''
        return types.SimpleNamespace(returncode=0, pid=os.getpid(), communicate=communicate)

    async def no_watch(proc, t, m):
        return None, 0
    monkeypatch.setattr(sandbox.asyncio, 'create_subprocess_exec', fake_exec)
    monkeypatch.setattr(sandbox, '_watch', no_watch)
    s = types.SimpleNamespace(home=tmp_path / 'home', store=store, job_memory_mb=100)
    res = asyncio.run(sandbox.run_job({'type': 'strategy_eval', 'allowed': allowed}, s))
    assert res == {'ok': True} and seen['exists']
    assert seen['store'] != store and seen['store'].parent == views.views_root(store)
    if 'holdout' in allowed:
        assert not seen['store'].exists()                   # one-shot: torn down after the job
        assert len((views.views_root(store) / 'holdout_consumed.jsonl').read_text().splitlines()) == 1
