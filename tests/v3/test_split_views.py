"""Per-split store views (dev / dev+val / one-shot holdout): no data beyond the allowed splits, pre-built splits.json,
and reading a view through every harness store under a read-only mount writes nothing."""
import json
import os
import time
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




def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob('*'))


@pytest.fixture
def vroot(tmp_path):
    return tmp_path / 'views'


@pytest.fixture
def ledger(tmp_path):
    from desk.company.ledger import Ledger
    return Ledger(tmp_path / 'ledger.sqlite3')


STRAT = "RESOLUTION = '1m'\ndef signal(df, p):\n    return df['mid'] * 0 + 1\n"
STRAT_COSMETIC = ('"""Same idea, new look."""\n# a comment\nRESOLUTION = "1m"\n\n\ndef signal(df,   p):\n'
                  '    """docstring"""\n    return (df["mid"] * 0) + 1   # trailing\n')


def _hjob(code=STRAT, **kw):
    return {'type': 'strategy_eval', 'code': code, 'split': 'holdout', 'allowed': ['dev', 'val', 'holdout'],
            'cost': {}, **kw}


def _store_snapshot(store):
    return sorted((str(p.relative_to(store)), p.lstat().st_mode, p.lstat().st_ino, p.read_bytes() if p.is_file() else b'')
                  for p in store.rglob('*'))


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val']])
def test_view_has_no_data_outside_allowed_splits(store, vroot, allowed):
    v = views.view_for_allowed(store, allowed, root=vroot)
    assert v.parent == vroot.resolve() and v.name.startswith('+'.join(allowed) + '@')
    assert store.resolve() not in v.resolve().parents                       # outside the store tree
    assert not (_numbers(v) & _forbidden(allowed)), _numbers(v) & _forbidden(allowed)
    assert _max_date(v) <= SPL[allowed[-1]][-1]
    days = sorted(p.stem for p in (v / '1s').glob('*.parquet'))
    assert days == sorted(d for k in allowed for d in SPL[k])
    f = v / '1s' / f'{DAYS[0]}.parquet'
    assert not os.path.samefile(f, store / '1s' / f.name) and f.stat().st_nlink == 1      # a copy, not a link
    assert f.read_bytes() == (store / '1s' / f.name).read_bytes()
    assert 'source' not in json.loads((v / '.view.json').read_text())


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val']])
def test_splits_json_prebuilt_and_correct(store, vroot, allowed):
    v = views.view_for_allowed(store, allowed, root=vroot)
    fs = json.loads((v / 'splits.json').read_text())
    for k in views.ORDER:
        assert fs[k] == (SPL[k] if k in allowed else [])
    for desk in ('macro', 'perps', 'pred', 'carry'):
        sp = json.loads((v / desk / 'splits.json').read_text())
        assert {k for k in views.ORDER if k in sp} == set(allowed)
        assert all(sp[k] == [SPL[k][0], SPL[k][-1]] for k in allowed)


def test_view_files_0444_and_store_untouched_by_build_and_teardown(store, vroot, ledger):
    before = _store_snapshot(store)
    v = views.view_for_allowed(store, ['dev', 'val'], root=vroot)
    for p in v.rglob('*'):
        mode = p.lstat().st_mode & 0o777
        assert mode == (0o555 if p.is_dir() else 0o444), (p, oct(mode))
        if p.is_file():
            assert p.stat().st_nlink == 1
    with views.one_shot_holdout(store, _hjob(), ledger, job_id='j1', root=vroot) as (h, _):
        assert (h / '1s').exists()
    views.remove_view(v)
    assert not v.exists() and not h.exists()
    assert _store_snapshot(store) == before                                  # modes, inodes and bytes unchanged


@pytest.mark.skipif(os.name == 'nt' or (hasattr(os, 'geteuid') and os.geteuid() == 0), reason='needs POSIX non-root')
@pytest.mark.parametrize('allowed', [('dev',), ('dev', 'val')])
def test_read_only_view_needs_no_writes(store, vroot, allowed):
    import dataclasses
    from desk.config import SETTINGS
    from desk.data.store import FeatureStore
    from desk.lab.carry import CarryStore
    from desk.lab.macro import MacroStore
    from desk.lab.perps import PerpsStore
    from desk.lab.pred import PredStore
    v = views.view_for_allowed(store, list(allowed), root=vroot)              # already sealed 0444/0555
    s = dataclasses.replace(SETTINGS, store_dir=v)
    before = _tree(v)
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
    assert views.view_for_allowed(store, list(allowed), root=vroot) == v      # reuse: no rebuild, no write
    assert _tree(v) == before


def test_one_shot_holdout_contents_and_teardown(store, vroot, ledger):
    with views.store_for_job(store, _hjob(), ledger=ledger, job_id='promo1', root=vroot) as (v, h):
        nums = _numbers(v)
        assert SENT['holdout'] in nums and SENT['forward'] not in nums and SENT['forward'] / 1000 not in nums
        assert _max_date(v) <= SPL['holdout'][-1]
        assert json.loads((v / 'splits.json').read_text())['forward'] == []
    assert not v.exists()


def test_cosmetic_resubmission_is_refused(store, vroot, ledger):
    with views.one_shot_holdout(store, _hjob(), ledger, job_id='a', root=vroot):
        pass
    with pytest.raises(views.HoldoutRefused, match='already consumed'):
        with views.one_shot_holdout(store, _hjob(STRAT_COSMETIC, values_dir='/x', return_daily=True), ledger,
                                    job_id='b', root=vroot):
            pass
    assert not list(vroot.glob('holdout-*'))
    # a genuinely different strategy, settings or data version is a new identity
    with views.one_shot_holdout(store, _hjob(STRAT.replace('+ 1', '+ 2')), ledger, job_id='c', root=vroot):
        pass
    with views.one_shot_holdout(store, _hjob(cost={'taker_bps': 9.0}), ledger, job_id='d', root=vroot):
        pass
    rows = ledger.q('SELECT kind FROM holdout_view_claims ORDER BY id')
    assert [r['kind'] for r in rows] == ['claim', 'done', 'claim', 'done', 'claim', 'done']


def test_failed_job_still_consumes_and_infra_kill_allows_one_retry(store, vroot, ledger):
    with pytest.raises(RuntimeError):
        with views.one_shot_holdout(store, _hjob(), ledger, job_id='p1', root=vroot) as (v, h):
            raise RuntimeError('strategy crashed')
    assert not v.exists()
    with pytest.raises(views.HoldoutRefused):
        with views.one_shot_holdout(store, _hjob(), ledger, job_id='p2', root=vroot):
            pass
    other = _hjob(STRAT.replace('+ 1', '+ 3'))
    with views.one_shot_holdout(store, other, ledger, job_id='m1', root=vroot) as (v, h):
        h.infra_failed()                                   # OOM-killed before producing a result
    with views.one_shot_holdout(store, other, ledger, job_id='m2', root=vroot) as (v, h):
        h.infra_failed()
    with pytest.raises(views.HoldoutRefused):              # capped at 2 attempts
        with views.one_shot_holdout(store, other, ledger, job_id='m3', root=vroot):
            pass


def test_identity_ignores_cosmetics_but_not_dependencies():
    from desk.lab.identity import evaluation_identity
    a = evaluation_identity(_hjob(), '/nonexistent')[0]
    assert a == evaluation_identity(_hjob(STRAT_COSMETIC, _home='/h', _store='/s'), '/nonexistent')[0]
    dep = _hjob('from desk.lib.contrib.har_sigma import har_sigma\n' + STRAT)
    ident, comp = evaluation_identity(dep, '/nonexistent')
    assert ident != a and 'desk.lib.contrib.har_sigma' in comp['deps']


def test_symlinks_are_never_followed(store, vroot, tmp_path):
    outside = tmp_path / 'secret.parquet'
    pl.DataFrame({'ts': [0], 'mid': [555.0]}).write_parquet(outside)
    os.unlink(store / '1s' / f'{DAYS[0]}.parquet')
    os.symlink(outside, store / '1s' / f'{DAYS[0]}.parquet')
    with pytest.raises(views.ViewError, match='symlink'):
        views.view_for_allowed(store, ['dev'], root=vroot)
    assert not [p for p in vroot.iterdir() if p.name.startswith('.') and p.name.endswith('.tmp')]
    # teardown removes a symlink inside a view without touching its target
    d = vroot / 'x'
    d.mkdir()
    os.symlink(outside, d / 'link.parquet')
    os.symlink(tmp_path, d / 'linkdir')
    views.remove_view(d)
    assert outside.exists() and not d.exists()


def test_concurrent_builders_tolerate_the_loser(store, vroot, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda _: views.view_for_allowed(store, ['dev'], root=vroot), range(4)))
    assert len(set(out)) == 1 and (out[0] / '.view.json').exists()
    assert not [p for p in vroot.iterdir() if p.name.endswith(('.tmp', '.old'))]


def test_views_root_must_be_outside_store(store):
    with pytest.raises(views.ViewError, match='outside the store'):
        views.view_for_allowed(store, ['dev'], root=store / 'views')
    v = views.view_for_allowed(store, ['dev'])                       # default: <store>_views next to the store
    assert v.parent == store.resolve().parent / 'store_views'


def test_allowed_must_be_a_prefix_and_forward_gets_full_store(store, vroot):
    with pytest.raises(views.ViewError):
        views.view_for_allowed(store, ['val'], root=vroot)
    with pytest.raises(views.ViewError):
        views.view_for_allowed(store, ['dev', 'val', 'holdout'], root=vroot)
    assert views.view_for_allowed(store, ['dev', 'val', 'holdout', 'forward'], root=vroot) == store


def test_prune_by_last_use_not_build_time(store, vroot):
    v1 = views.view_for_allowed(store, ['dev'], root=vroot)
    old = time.time() - 3 * 86400
    os.utime(v1, (old, old))                                         # built long ago ...
    p = store / 'macro' / 'panel.parquet'
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    v2 = views.view_for_allowed(store, ['dev'], root=vroot)
    assert v2 != v1 and v1.exists()                                  # ... but used just now: kept
    os.utime(vroot / '.last_used' / v1.name, (old, old))
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    v3 = views.view_for_allowed(store, ['dev'], root=vroot)
    assert not v1.exists() and v2.exists() and v3.exists()          # idle for 3 days: pruned


@pytest.mark.parametrize('allowed', [['dev'], ['dev', 'val'], ['dev', 'val', 'holdout']])
def test_run_job_hands_the_harness_only_its_view(store, tmp_path, monkeypatch, allowed):
    import asyncio
    import types
    from desk.lab import sandbox
    seen = {}

    async def fake_exec(*argv, **kw):
        job = json.loads(Path(argv[-2]).read_text())
        seen['job'] = job
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
    s = types.SimpleNamespace(home=tmp_path / 'home', store=store, job_memory_mb=100, views_dir=tmp_path / 'v',
                              ledger=tmp_path / 'home' / 'desk.sqlite3')
    job = dict(_hjob(), allowed=allowed)
    res = asyncio.run(sandbox.run_job(job, s))
    assert res == {'ok': True} and seen['exists'] and '_home' not in seen['job']
    assert seen['store'] != store and seen['store'].parent == (tmp_path / 'v').resolve()
    if 'holdout' in allowed:
        assert not seen['store'].exists()                   # one-shot: torn down after the job
        again = asyncio.run(sandbox.run_job(dict(job, code=STRAT_COSMETIC), s))
        assert again['error'].startswith('holdout_refused')
