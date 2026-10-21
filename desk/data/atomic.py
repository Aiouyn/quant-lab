"""Atomic file writers: write to a temp file in the destination directory, fsync, then os.replace.

os.replace swaps the directory entry, so readers never see a half-written file, the original survives
a failed write untouched, and any hard-linked copies (e.g. store/views/dev) keep their old inode
instead of being mutated in place.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _tmp_for(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    return Path(tmp)


def _commit(tmp: Path, path: Path):
    os.chmod(tmp, 0o644)                    # mkstemp creates 0600; readers (e.g. a non-root sandbox user) need 0644
    with open(tmp, 'rb') as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(d: Path):
    """Persist the rename itself (POSIX). Directories can't be opened for fsync on Windows: skip there."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_parquet(df, path, compression='zstd', **kw):
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        df.write_parquet(tmp, compression=compression, **kw)
        _commit(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path, text: str, encoding='utf-8'):
    path = Path(path)
    tmp = _tmp_for(path)
    try:
        tmp.write_text(text, encoding=encoding)
        _commit(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path, obj, **dump_kw):
    return atomic_write_text(path, json.dumps(obj, **dump_kw))
