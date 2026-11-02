"""Evaluator kernel manifest.

The kernel is the code that decides whether a strategy is real: data splits, the strategy jail and sandbox guards,
backtests and cost models, statistics, the validation gates and the harness. It may only change through a reviewed
commit (CODEOWNERS on GitHub), and it may only become stricter.

`desk/KERNEL_MANIFEST.json` holds the sha256 of every kernel file. The harness and the validation pipeline refuse to
run when a file differs from the manifest, so an edit made outside review (by an agent, a script or by hand) stops
evaluation instead of silently changing it. After reviewing a kernel change, a human runs:

    python -m desk kernel seal        # rewrite the manifest for the current files (commit it with the change)
    python -m desk kernel verify      # what the harness checks

Line endings are normalised before hashing, so a Windows checkout (CRLF) matches the manifest.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
MANIFEST = PKG / 'KERNEL_MANIFEST.json'
KERNEL_GLOBS = ('lab/*.py', 'pipeline/validation.py', 'pipeline/bundle.py', 'data/store.py', 'config.py', 'kernel.py')


def kernel_files() -> list[Path]:
    out = set()
    for g in KERNEL_GLOBS:
        out.update(p for p in PKG.glob(g) if p.is_file())
    return sorted(out)


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def current() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): _digest(p) for p in kernel_files()}


def seal() -> dict:
    files = current()
    MANIFEST.write_text(json.dumps({'sealed': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': files}, indent=1,
                                   sort_keys=True) + '\n', encoding='utf-8')
    return {'ok': True, 'files': len(files), 'manifest': str(MANIFEST)}


def verify() -> dict:
    if not MANIFEST.exists():
        return {'ok': False, 'error': 'kernel manifest missing', 'changed': [], 'added': [], 'removed': []}
    want = json.loads(MANIFEST.read_text(encoding='utf-8')).get('files', {})
    have = current()
    changed = sorted(k for k in have.keys() & want.keys() if have[k] != want[k])
    added = sorted(have.keys() - want.keys())
    removed = sorted(want.keys() - have.keys())
    ok = not (changed or added or removed)
    out = {'ok': ok, 'changed': changed, 'added': added, 'removed': removed}
    if not ok:
        out['error'] = ('evaluator kernel differs from desk/KERNEL_MANIFEST.json (changed outside a reviewed commit?): '
                        f'changed={changed} added={added} removed={removed}. Review the change, then run '
                        '`python -m desk kernel seal` and commit the manifest with it.')
    return out


def require_sealed():
    """Raise if the kernel was modified outside a reviewed commit."""
    res = verify()
    if not res['ok']:
        raise RuntimeError(res['error'])
