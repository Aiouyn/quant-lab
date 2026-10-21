"""Per-split store views: read-only, store-shaped directories that contain ONLY data up to the end of the latest
split in a job's `allowed` set, so a (jailed) harness job is given exactly what it may see.

Where: OUTSIDE the store tree, in a configurable views root (`views_root=` / $DESK_VIEWS_ROOT, default
`<store parent>/<store name>_views`).
  dev@<stamp>/       allowed = ['dev']          reusable, versioned by a stamp of the sources
  dev+val@<stamp>/   allowed = ['dev', 'val']   (superseded versions pruned 24 h after their LAST USE)
  holdout-<job>/     allowed = ['dev', 'val', 'holdout']   ONE-SHOT per evaluation identity (desk/lab/identity.py:
                     normalized strategy AST + desk.* dependency closure + eval settings + data version), checked and
                     recorded atomically in the ledger (holdout_view_claims, append-only). Torn down after the job.
`allowed` must be a chronological prefix of (dev, val, holdout, forward). A set containing `forward` (the forward
check) is everything, so it gets the full store root.

Contents, pre-built so that NOTHING is written lazily inside a read-only mount:
  1s/10s/1m/tape day files  independent COPIES (reflink when supported), never hard links
  splits.json               FeatureStore split with non-allowed splits emptied (keys kept)
  funding_hl.parquet        hours < cutoff
  macro|perps|pred|carry    panel rows < that desk's cutoff + splits.json without the non-allowed splits.
                            Engine-only look-ahead columns on the last allowed hour are nulled (perps lo/hi = range of
                            the NEXT hour; carry Binance bar columns close one hour after their key); pred keeps only
                            markets that resolve before the cutoff (an outcome is future information).
Files are 0444 and dirs 0555 (they are copies, so this never touches a store inode). Build = temp dir + rename; a
concurrent builder that loses the rename discards its copy and uses the winner's. Symlinks are never followed when
building (error) or tearing down (removed, not traversed).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import time
import types
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


def _write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8')


def _src(p: Path) -> Path:
    """Refuse to follow symlinks out of the store: a symlinked source file or directory is an error."""
    for q in (p, p.parent):
        if q.is_symlink():
            raise ViewError(f'refusing to follow symlink {q} while building a view')
    return p


FICLONE = 0x40049409                                       # Linux ioctl: reflink (btrfs/xfs); falls back to a copy


def _copy(src: Path, dst: Path):
    """Independent copy (reflink when the filesystem supports it). Never a hard link: view files get chmod 0444 and
    a hard link would share (and change) the store file's inode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
        with open(_src(src), 'rb') as fi, open(dst, 'wb') as fo:
            fcntl.ioctl(fo.fileno(), FICLONE, fi.fileno())
        return
    except (OSError, ImportError):
        dst.unlink(missing_ok=True)
    shutil.copyfile(_src(src), dst, follow_symlinks=False)


# ---------------------------------------------------------------------------------------------- per desk
def _feature_store(src: Path, dst: Path, allowed):
    f = src / 'splits.json'
    if not f.exists():
        return None
    sp = json.loads(_src(f).read_text())
    days = sorted(d for k in allowed for d in sp.get(k, []))
    for res in DAY_RES:
        if not (src / res).exists():
            continue
        (dst / res).mkdir(parents=True, exist_ok=True)
        for d in days:
            p = src / res / f'{d}.parquet'
            if p.exists():
                _copy(p, dst / res / p.name)
    (dst / 'meta').mkdir(parents=True, exist_ok=True)
    view_sp = {k: (sp.get(k, []) if k in allowed else []) for k in ORDER}
    view_sp.update(created=sp.get('created'), view={'allowed': list(allowed)})
    _write_json(view_sp, dst / 'splits.json')
    cutoff = (_ms(days[-1]) + DAY) if days else None
    fh = src / 'funding_hl.parquet'
    if fh.exists() and cutoff is not None:
        _write_parquet(pl.read_parquet(_src(fh)).filter(pl.col('hour') < cutoff), dst / 'funding_hl.parquet')
    return cutoff


def _panel_desk(src_root: Path, dst_root: Path, allowed, desk: str):
    src, dst = src_root / desk, dst_root / desk
    f = src / 'splits.json'
    if not f.exists():
        return None
    sp = json.loads(_src(f).read_text())
    cutoff = _panel_cutoff_ms(sp, allowed)
    if cutoff is None:
        return None
    _write_json(_view_splits(sp, allowed), dst / 'splits.json')
    if desk == 'carry':
        if (src / 'universe.json').exists():
            _write_json(json.loads((src / 'universe.json').read_text()), dst / 'universe.json')
        from .carry_data import BAR_COLS
        for cf in sorted((src / 'coins').glob('*.parquet')):
            p = pl.read_parquet(_src(cf)).filter(pl.col('ts') < cutoff)
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
        p = pl.scan_parquet(_src(pf)).filter(pl.col('date') < cut_date).collect()
    elif desk == 'perps':
        p = pl.scan_parquet(_src(pf)).filter(pl.col('ts') < cutoff).collect()
        cols = [c for c in ('lo', 'hi') if c in p.columns]
        if cols:
            p = p.with_columns([pl.when(pl.col('ts') >= cutoff - HOUR).then(None).otherwise(pl.col(c)).alias(c)
                                for c in cols])
    else:  # pred
        p = pl.scan_parquet(_src(pf)).filter((pl.col('ts') < cutoff) & (pl.col('end_ms') < cutoff)).collect()
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


def views_root(store: Path, root=None) -> Path:
    """Configurable views root, never inside the store tree."""
    store = Path(store).resolve()
    r = Path(root or os.environ.get('DESK_VIEWS_ROOT') or store.parent / f'{store.name}_views').resolve()
    if r == store or store in r.parents:
        raise ViewError(f'views root {r} must be outside the store tree {store}')
    r.mkdir(parents=True, exist_ok=True)
    return r


def _seal(root: Path):
    """0444 files / 0555 dirs. Only called on our own fresh copies (nlink == 1 is asserted)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for f in filenames:
            p = os.path.join(dirpath, f)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise ViewError(f'unexpected symlink in view {p}')
            if st.st_nlink != 1:
                raise ViewError(f'view file {p} shares an inode (nlink={st.st_nlink}); refusing to chmod it')
            os.chmod(p, 0o444, follow_symlinks=False)
        os.chmod(dirpath, 0o555)


def remove_view(path: Path):
    """Tear a view down without following symlinks and without chmodding any FILE: deleting entries only needs the
    parent directories writable, and those directories are the view's own."""
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        os.chmod(dirpath, 0o755)
        for d in list(dirnames):
            if os.path.islink(os.path.join(dirpath, d)):
                dirnames.remove(d)                      # symlinked dir: never descend; rmtree unlinks the link
    shutil.rmtree(path)


def build_view(store: Path, allowed, dest: Path) -> Path:
    """Build the view for `allowed` at `dest` (temp dir + rename). If `dest` already exists with the same stamp, or
    another builder wins the rename race, the existing view is used. Returns dest."""
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
                     'cutoff_ms': cut}, tmp / '.view.json')
        _seal(tmp)
        if dest.exists():                                   # stale same-name view (one-shot rebuild): replace it
            old = dest.with_name(f'.{dest.name}.{uuid.uuid4().hex[:8]}.old')
            os.rename(dest, old)
            remove_view(old)
        try:
            os.rename(tmp, dest)                            # fails if a concurrent builder created dest meanwhile
        except OSError:
            if not (dest / '.view.json').exists():
                raise
            remove_view(tmp)                                # lost the race: the winner's view is equivalent
    except BaseException:
        remove_view(tmp)
        raise
    return dest


def _touch_used(root: Path, name: str):
    d = root / '.last_used'
    d.mkdir(exist_ok=True)
    (d / name).touch()


def view_for_allowed(store: Path, allowed, root=None) -> Path:
    """Reusable view for dev or dev+val; the full store for sets containing `forward`. Holdout: one_shot_holdout."""
    allowed = normalize_allowed(allowed)
    if 'forward' in allowed:
        return Path(store)
    if 'holdout' in allowed:
        raise ViewError('holdout views are one-shot: use one_shot_holdout(store, job, ledger)')
    r = views_root(store, root)
    name = '+'.join(allowed)
    dest = r / f'{name}@{_stamp(Path(store), allowed)[:12]}'
    build_view(store, allowed, dest)
    _touch_used(r, dest.name)
    _prune(r, name, keep=dest)
    return dest


def _prune(root: Path, name: str, keep: Path, max_idle_s=24 * 3600):
    """Drop superseded versions of a reusable view once unused for a day (no job runs that long)."""
    for d in root.glob(f'{name}@*'):
        if d == keep or d.is_symlink() or not d.is_dir():
            continue
        used = root / '.last_used' / d.name
        last = used.stat().st_mtime if used.exists() else d.stat().st_mtime
        if time.time() - last > max_idle_s:
            remove_view(d)
            used.unlink(missing_ok=True)


class HoldoutRefused(ViewError):
    pass


@contextlib.contextmanager
def one_shot_holdout(store: Path, job: dict, ledger, job_id: str | None = None, root=None):
    """Build a dev+val+holdout view for ONE evaluation identity, yield (path, handle), then tear it down.
    The claim is checked and recorded atomically in the ledger BEFORE the view exists; a resubmission with the same
    identity (cosmetic edits included) raises HoldoutRefused. handle.infra_failed() records an infrastructure kill
    (the job produced no result), which permits exactly one retry."""
    from ..lab.identity import evaluation_identity
    if ledger is None:
        raise ViewError('a holdout view needs the ledger to record its one-shot claim')
    job_id = job_id or uuid.uuid4().hex[:12]
    ident, comp = evaluation_identity(job, store)
    ok, reason = ledger.claim_holdout_view(ident, job_id, comp)
    if not ok:
        raise HoldoutRefused(f'{reason}: {ident[:16]}')
    handle = types.SimpleNamespace(identity=ident, job_id=job_id, outcome='done')
    handle.infra_failed = lambda: setattr(handle, 'outcome', 'infra_fail')
    dest = views_root(store, root) / f'holdout-{job_id}'
    try:
        build_view(store, ORDER[:3], dest)
        yield dest, handle
    finally:
        remove_view(dest)
        ledger.mark_holdout_view(ident, job_id, handle.outcome)


@contextlib.contextmanager
def store_for_job(store: Path, job: dict, ledger=None, job_id: str | None = None, root=None):
    """(store root the harness job should get = the ONLY store path to mount read-only, holdout handle or None)."""
    allowed = normalize_allowed(job.get('allowed', ('dev',)))
    if 'holdout' in allowed and 'forward' not in allowed:
        with one_shot_holdout(store, job, ledger, job_id=job_id, root=root) as (v, h):
            yield v, h
    else:
        yield view_for_allowed(store, allowed, root=root), None
