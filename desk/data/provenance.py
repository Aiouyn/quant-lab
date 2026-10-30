"""Fail-loud helpers for store panels: atomic writes, frozen-split guard and provenance sidecars.

- `atomic_write_*`: temp file in the destination dir, fsync, `os.replace` (no torn files, originals survive failures).
- `freeze_splits`: a split file is created only when absent; an existing (frozen) file is never rewritten, and a
  rebuild whose computed split differs from the frozen one raises `SplitsMismatch` instead of silently moving data
  between dev/val/holdout.
- `write_splits` / `read_splits`: every splits.json has a sha256 manifest (`splits.json.sha256`) written next to it;
  `read_splits` recomputes the hash on every load and a missing or mismatching manifest is a hard error
  (`SplitsIntegrityError`). Existing files are sealed once, deliberately, with `seal_splits` / `python -m
  desk.data.provenance seal <splits.json>`.
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


class SplitsIntegrityError(RuntimeError):
    """splits.json does not match its sha256 manifest (or has none). Never auto-repaired."""


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
        frozen = read_splits(path)
        if _core(frozen) != _core(computed):
            raise SplitsMismatch(
                f'{desk}: frozen {path} differs from the split this build computes; refusing to overwrite.\n'
                f'  frozen:   {json.dumps(_core(frozen))}\n  computed: {json.dumps(_core(computed))}\n'
                'Investigate (or delete the file deliberately) before rebuilding.')
        return frozen
    sp = {**computed, 'created': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    write_splits(path, sp)
    return sp


# ---------------------------------------------------------------------------------------------- sha256 manifest
def manifest_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + '.sha256')


def _sha256(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def seal_splits(path) -> str:
    """Write the sha256 manifest for the CURRENT bytes of `path`. Only for writers that own the file."""
    path = Path(path)
    digest = _sha256(path.read_bytes())
    _replace_from_tmp(manifest_path(path), lambda t: t.write_text(f'{digest}  {path.name}\n', encoding='utf-8'))
    return digest


def write_splits(path, obj, indent=1) -> Path:
    """The only sanctioned way to (re)write a splits.json: atomic write, then the manifest for those exact bytes."""
    path = Path(path)
    atomic_write_json(path, obj, indent=indent)
    seal_splits(path)
    return path


def read_splits(path) -> dict:
    """Load a splits.json after checking it against its manifest. Mismatch or missing manifest -> SplitsIntegrityError."""
    path = Path(path)
    raw = path.read_bytes()
    m = manifest_path(path)
    if not m.exists():
        raise SplitsIntegrityError(f'{path}: no sha256 manifest ({m.name}); refusing to use unsealed splits. '
                                   'Seal it deliberately: python -m desk.data.provenance seal <path>')
    want = m.read_text(encoding='utf-8').split()[0].strip().lower() if m.read_text(encoding='utf-8').strip() else ''
    got = _sha256(raw)
    if want != got:
        raise SplitsIntegrityError(f'{path}: sha256 {got[:16]}.. does not match manifest {want[:16]}..; '
                                   'splits.json was modified outside write_splits(). Investigate before running anything.')
    return json.loads(raw.decode('utf-8'))


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


if __name__ == '__main__':                                     # python -m desk.data.provenance seal <splits.json>...
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'seal':
        for a in sys.argv[2:]:
            print(f'{seal_splits(a)}  {a}')
    else:
        print('usage: python -m desk.data.provenance seal <splits.json> [...]')
        sys.exit(2)
