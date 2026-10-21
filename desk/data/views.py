"""Per-split store views: read-only, store-shaped directories that contain ONLY data up to the end of the latest
split in a job's `allowed` set, so a jailed harness job can be given (mounted) exactly what it may see.

Views (all under <store>/views/split/):
  dev@<stamp>/      allowed = ['dev']           (versioned by a stamp of the sources; superseded versions pruned
  dev+val@<stamp>/  allowed = ['dev', 'val']     after 24 h, so a running job's view is never swapped underneath it)
  holdout-<job>/  allowed = ['dev', 'val', 'holdout']  ONE-SHOT: built for a single promotion job, torn down after
                  it and recorded in holdout_consumed.jsonl; the same job id can never get a second holdout view.
`allowed` must be a chronological prefix of (dev, val, holdout, forward). A set containing `forward` (the forward
check) is everything, so it gets the full store root.

Each view is a complete store for every desk the harness reads, pre-built so that NOTHING is written lazily inside a
read-only mount:
  1s/10s/1m/tape day files  hard links (immutable) for days of the allowed splits; copy if linking fails
  splits.json               FeatureStore split with non-allowed splits emptied (keys kept)
  funding_hl.parquet        hours < cutoff
  macro|perps|pred|carry    panel rows < that desk's cutoff + splits.json without the non-allowed splits.
                            Engine-only look-ahead columns on the last allowed hour are nulled (perps lo/hi = range of
                            the NEXT hour; carry Binance bar columns close one hour after their key); pred keeps only
                            markets that resolve before the cutoff (an outcome is future information).
Views are built in a temp dir and swapped in with a rename (atomic), files 0644; a stamp of the sources lets a
fresh view be reused. Depends on nothing outside this module (PR #7's atomic helper is not on main yet).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ORDER = ('dev', 'val', 'holdout', 'forward')
DAY_RES = ('1s', '10s', '1m', 'tape')
HOUR = 3_600_000
DAY = 86_400_000


class ViewError(RuntimeError):
    pass


def normalize_allowed(allowed) -> tuple:
    a = tuple(allowed or ('dev',))
    k = len(a)
    if set(a) != set(ORDER[:k]) or len(set(a)) != k:
        raise ViewError(f'allowed={list(a)} is not a chronological prefix of {list(ORDER)}')
    return ORDER[:k]


def _ms(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _panel_cutoff_ms(sp: dict, allowed) -> int | None:
    """Exclusive end (ms) of the latest allowed split in a panel-desk splits.json ([first, last] ISO dates)."""
    last = allowed[-1]
    if last not in sp or not sp[last]:
        return None
    return _ms(sp[last][1]) + DAY


def _view_splits(sp: dict, allowed) -> dict:
    out = {k: v for k, v in sp.items() if k not in ORDER or k in allowed}
    out['view'] = {'allowed': list(allowed)}
    return out


# ---------------------------------------------------------------------------------------------- writers
def _write_parquet(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression='zstd')
    os.chmod(path, 0o644)


def _write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8')
    os.chmod(path, 0o644)


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------------------------- per desk
def _feature_store(src: Path, dst: Path, allowed):
    f = src / 'splits.json'
    if not f.exists():
        return None
    sp = json.loads(f.read_text())
    days = sorted(d for k in allowed for d in sp.get(k, []))
    for res in DAY_RES:
        if not (src / res).exists():
            continue
        (dst / res).mkdir(parents=True, exist_ok=True)
        for d in days:
            p = src / res / f'{d}.parquet'
            if p.exists():
                _link(p, dst / res / p.name)
    (dst / 'meta').mkdir(parents=True, exist_ok=True)
    view_sp = {k: (sp.get(k, []) if k in allowed else []) for k in ORDER}
    view_sp.update(created=sp.get('created'), view={'allowed': list(allowed)})
    _write_json(view_sp, dst / 'splits.json')
    cutoff = (_ms(days[-1]) + DAY) if days else None
    fh = src / 'funding_hl.parquet'
    if fh.exists() and cutoff is not None:
        _write_parquet(pl.read_parquet(fh).filter(pl.col('hour') < cutoff), dst / 'funding_hl.parquet')
    return cutoff


def _panel_desk(src_root: Path, dst_root: Path, allowed, desk: str):
    src, dst = src_root / desk, dst_root / desk
    f = src / 'splits.json'
    if not f.exists():
        return None
    sp = json.loads(f.read_text())
    cutoff = _panel_cutoff_ms(sp, allowed)
    if cutoff is None:
        return None
    _write_json(_view_splits(sp, allowed), dst / 'splits.json')
    if desk == 'carry':
        if (src / 'universe.json').exists():
            _write_json(json.loads((src / 'universe.json').read_text()), dst / 'universe.json')
        from .carry_data import BAR_COLS
        for cf in sorted((src / 'coins').glob('*.parquet')):
            p = pl.read_parquet(cf).filter(pl.col('ts') < cutoff)
            cols = [c for c in BAR_COLS if c in p.columns]
            if cols:
                p = p.with_columns([pl.when(pl.col('ts') >= cutoff - HOUR).then(None).otherwise(pl.col(c)).alias(c)
                                    for c in cols])
            _write_parquet(p, dst / 'coins' / cf.name)
        return cutoff
    pf = src / 'panel.parquet'
    if not pf.exists():
        return cutoff
    if desk == 'macro':
        cut_date = datetime.fromtimestamp(cutoff / 1000, timezone.utc).date()
        p = pl.scan_parquet(pf).filter(pl.col('date') < cut_date).collect()
    elif desk == 'perps':
        p = pl.scan_parquet(pf).filter(pl.col('ts') < cutoff).collect()
        cols = [c for c in ('lo', 'hi') if c in p.columns]
        if cols:
            p = p.with_columns([pl.when(pl.col('ts') >= cutoff - HOUR).then(None).otherwise(pl.col(c)).alias(c)
                                for c in cols])
    else:  # pred
        p = pl.scan_parquet(pf).filter((pl.col('ts') < cutoff) & (pl.col('end_ms') < cutoff)).collect()
    _write_parquet(p, dst / 'panel.parquet')
    return cutoff


# ---------------------------------------------------------------------------------------------- build / stamp
def _stamp(store: Path, allowed) -> str:
    h = hashlib.sha1(json.dumps(list(allowed)).encode())
    for rel in ('splits.json', 'funding_hl.parquet', *(f'{d}/splits.json' for d in ('macro', 'perps', 'pred', 'carry')),
                *(f'{d}/panel.parquet' for d in ('macro', 'perps', 'pred'))):
        p = store / rel
        if p.exists():
            st = p.stat(); h.update(f'{rel}:{st.st_size}:{st.st_mtime_ns}'.encode())
    for sub in (*DAY_RES, 'carry/coins'):
        d = store / sub
        if d.exists():
            for p in sorted(d.glob('*.parquet')):
                st = p.stat(); h.update(f'{sub}/{p.name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return h.hexdigest()


def build_view(store: Path, allowed, dest: Path) -> Path:
    """Build the view for `allowed` at `dest` atomically (temp dir + rename). Returns dest."""
    store, dest = Path(store), Path(dest)
    allowed = normalize_allowed(allowed)
    stamp = _stamp(store, allowed)
    marker = dest / '.view.json'
    if marker.exists() and json.loads(marker.read_text()).get('stamp') == stamp:
        return dest
    tmp = dest.with_name(f'.{dest.name}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.mkdir(parents=True)
    try:
        cut = {'features': _feature_store(store, tmp, allowed)}
        for desk in ('macro', 'perps', 'pred', 'carry'):
            cut[desk] = _panel_desk(store, tmp, allowed, desk)
        _write_json({'allowed': list(allowed), 'stamp': stamp, 'built_utc': datetime.now(timezone.utc).isoformat(),
                     'cutoff_ms': cut, 'source': str(store)}, tmp / '.view.json')
        old = None
        if dest.exists():
            old = dest.with_name(f'.{dest.name}.{uuid.uuid4().hex[:8]}.old')
            os.replace(dest, old)
        os.replace(tmp, dest)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def views_root(store: Path) -> Path:
    return Path(store) / 'views' / 'split'


def view_for_allowed(store: Path, allowed) -> Path:
    """Reusable view for dev or dev+val; the full store for sets containing `forward`. Holdout needs one_shot_holdout."""
    allowed = normalize_allowed(allowed)
    if 'forward' in allowed:
        return Path(store)
    if 'holdout' in allowed:
        raise ViewError('holdout views are one-shot: use one_shot_holdout(store, job_id)')
    # versioned by source stamp: a running job's view is never replaced underneath it when the store changes
    name = '+'.join(allowed)
    dest = views_root(store) / f'{name}@{_stamp(store, allowed)[:12]}'
    build_view(store, allowed, dest)
    _prune(views_root(store), name, keep=dest)
    return dest


def _prune(root: Path, name: str, keep: Path, max_age_s=24 * 3600):
    """Drop superseded versions of a reusable view once they are a day old (no job runs that long)."""
    for d in root.glob(f'{name}@*'):
        if d != keep and d.is_dir() and time.time() - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


def _consumed_log(store: Path) -> Path:
    return views_root(store) / 'holdout_consumed.jsonl'


def holdout_consumed(store: Path, job_id: str) -> bool:
    f = _consumed_log(store)
    return f.exists() and any(json.loads(l).get('job_id') == job_id for l in f.read_text().splitlines() if l.strip())


@contextlib.contextmanager
def one_shot_holdout(store: Path, job_id: str):
    """Build a dev+val+holdout view for ONE promotion job, yield its path, then tear it down and mark it consumed
    (even if the job fails). A job id that already consumed its holdout view is refused."""
    store = Path(store)
    if holdout_consumed(store, job_id):
        raise ViewError(f'holdout view for job {job_id} was already consumed')
    dest = views_root(store) / f'holdout-{job_id}'
    build_view(store, ORDER[:3], dest)
    try:
        yield dest
    finally:
        for p in [dest, *sorted(dest.rglob('*'))] if dest.exists() else []:   # undo 0444/0555 left by a RO mount
            with contextlib.suppress(OSError):
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
        shutil.rmtree(dest, ignore_errors=True)
        f = _consumed_log(store)
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'job_id': job_id, 'consumed_utc': datetime.now(timezone.utc).isoformat(),
                                 'ts': time.time()}) + '\n')


@contextlib.contextmanager
def store_for_job(store: Path, job: dict, job_id: str | None = None):
    """The store root a harness job should get (and the ONLY store path to mount read-only)."""
    allowed = normalize_allowed(job.get('allowed', ('dev',)))
    if 'holdout' in allowed and 'forward' not in allowed:
        with one_shot_holdout(store, job_id or uuid.uuid4().hex[:12]) as v:
            yield v
    else:
        yield view_for_allowed(store, allowed)
