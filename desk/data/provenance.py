"""Fail-loud helpers for store panels: atomic writes, frozen-split guard and provenance sidecars.

- `atomic_write_*`: temp file in the destination dir, fsync, `os.replace` (no torn files, originals survive failures).
- `freeze_splits`: a split file is created only when absent; an existing (frozen) file is never rewritten, and a
  rebuild whose computed split differs from the frozen one raises `SplitsMismatch` instead of silently moving data
  between dev/val/holdout.
- `write_provenance`: `provenance.json` sidecar with source, UTC fetch time, row/asset counts, date range and the
  code commit that built the panel.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

SPLIT_KEYS = ('dev', 'val', 'holdout', 'forward')
REPO_ROOT = Path(__file__).resolve().parents[2]


class StoreBuildError(RuntimeError):
    """A panel could not be built from real downloaded data. There is no synthetic fallback."""


class SplitsMismatch(StoreBuildError):
    """The split a rebuild would compute differs from the frozen splits.json on disk."""


# ---------------------------------------------------------------------------------------------- atomic writes
def _replace_from_tmp(path: Path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    tmp = Path(tmp)
    try:
        write(tmp)
        os.chmod(tmp, 0o644)                                   # mkstemp creates 0600; sandbox users must read it
        with open(tmp, 'rb') as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:                                                   # persist the rename (POSIX only)
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path, obj, indent=1):
    text = json.dumps(obj, indent=indent, default=str)
    return _replace_from_tmp(path, lambda t: t.write_text(text, encoding='utf-8'))


def atomic_write_parquet(df, path, compression='zstd'):
    return _replace_from_tmp(path, lambda t: df.write_parquet(t, compression=compression))


# ---------------------------------------------------------------------------------------------- frozen splits
def _core(sp: dict) -> dict:
    return {k: sp.get(k) for k in SPLIT_KEYS if k in sp}


def freeze_splits(path, computed: dict, desk: str) -> dict:
    """Create `path` from `computed` if absent; otherwise verify and return the frozen file untouched."""
    path = Path(path)
    if path.exists():
        frozen = json.loads(path.read_text(encoding='utf-8'))
        if _core(frozen) != _core(computed):
            raise SplitsMismatch(
                f'{desk}: frozen {path} differs from the split this build computes; refusing to overwrite.\n'
                f'  frozen:   {json.dumps(_core(frozen))}\n  computed: {json.dumps(_core(computed))}\n'
                'Investigate (or delete the file deliberately) before rebuilding.')
        return frozen
    sp = {**computed, 'created': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    atomic_write_json(path, sp)
    return sp


# ---------------------------------------------------------------------------------------------- provenance
def code_commit() -> dict:
    def git(*a):
        r = subprocess.run(['git', '-C', str(REPO_ROOT), *a], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        sha = git('rev-parse', 'HEAD')
        dirty = git('status', '--porcelain', '--untracked-files=no')
    except Exception:
        sha, dirty = None, None
    return {'code_commit': sha or 'unknown', 'code_dirty': bool(dirty) if dirty is not None else None}


def _iso(v):
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, (int, float)):                         # epoch ms
        return datetime.fromtimestamp(v / 1000, timezone.utc).isoformat(timespec='seconds')
    return str(v)


def write_provenance(root, desk: str, source: str, panel, time_col: str, asset_col: str, fetched_at=None, **extra) -> dict:
    if panel is None or len(panel) == 0:
        raise StoreBuildError(f'{desk}: refusing to record provenance for an empty panel (source {source})')
    per_asset = {str(r[asset_col]): int(r['len']) for r in panel.group_by(asset_col).len().iter_rows(named=True)}
    prov = {
        'desk': desk,
        'source': source,
        'verified_real': True,
        'fetched_at_utc': (fetched_at or datetime.now(timezone.utc)).isoformat(timespec='seconds'),
        'rows': len(panel),
        'assets': len(per_asset),
        'rows_per_asset': dict(sorted(per_asset.items())),
        'first': _iso(panel[time_col].min()),
        'last': _iso(panel[time_col].max()),
        **code_commit(),
        **extra,
    }
    atomic_write_json(Path(root) / 'provenance.json', prov, indent=2)
    return prov
