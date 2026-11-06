"""Read Google Drive data through the local Drive-for-desktop mirror when it has the file, else through rclone.

The rclone remote's OAuth token can expire (2026-11-06) while the mirror (G:/Mon Drive on the owner's PC) still works.
DESK_DRIVE_ROOT points at the folder that holds QUANT_DATA; by default G:/Mon Drive is used when it exists."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REMOTES = ('loopfactory:',)


def _local(remote: str) -> Path | None:
    root = os.getenv('DESK_DRIVE_ROOT') or ('G:/Mon Drive' if Path('G:/Mon Drive').exists() else '')
    if not root:
        return None
    for r in REMOTES:
        if remote.startswith(r):
            return Path(root) / remote[len(r):]
    return None


def cat(remote: str, timeout: int = 600) -> bytes:
    p = _local(remote)
    if p is not None and p.is_file():
        return p.read_bytes()
    return subprocess.run(['rclone', 'cat', remote], capture_output=True, timeout=timeout).stdout


def lsf(remote: str) -> list[str]:
    p = _local(remote)
    if p is not None and p.is_dir():
        return sorted(f.name for f in p.iterdir() if f.is_file())
    return subprocess.run(['rclone', 'lsf', remote], capture_output=True, text=True).stdout.split()
