"""OS-level jail for agent-written Python (bubblewrap).

Every agent script and every agent-authored factor module runs inside `bwrap` with:
  * all namespaces unshared (--unshare-all: no network, private pid/ipc/uts/cgroup), --die-with-parent, --new-session
  * a non-root uid/gid (65534 'nobody' inside the namespace)
  * a minimal read-only filesystem: /usr (+ /bin /lib /lib64 /sbin symlinks), the Python prefix(es), the repo's
    `desk/` package only (no .env, no results/, no data/), and the dev data view paths passed by the caller
  * a private tmpfs /tmp; the only writable host paths are the ones the caller passes as `rw` (the agent workspace)
  * rlimits (CPU seconds, address space, file size, processes, open files) set in a preexec step on the bwrap
    process and again by an in-jail boot step (RLIMIT_NPROC must be set inside the new user namespace so it counts
    only the jail's own processes)

Fail closed: if bwrap is missing or cannot create namespaces, `JailUnavailable` is raised. For local development on
a machine without bwrap you may opt in to the old unjailed behaviour with DESK_SANDBOX_UNSAFE=1 (never in CI/prod).
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import ROOT

UNSAFE_ENV = 'DESK_SANDBOX_UNSAFE'
JAIL_UID = 65534
BOOT = Path(__file__).resolve().parent / '_jail_boot.py'
DEFAULT_LIMITS = {'cpu_s': 600, 'as_mb': 8000, 'fsize_mb': 1024, 'nproc': 64, 'nofile': 256}


class JailUnavailable(RuntimeError):
    pass


def unsafe_opt_in() -> bool:
    return os.environ.get(UNSAFE_ENV, '') == '1'


@functools.lru_cache(maxsize=1)
def _probe():
    exe = shutil.which('bwrap')
    if not exe:
        return None, 'bubblewrap (bwrap) is not installed'
    try:
        r = subprocess.run([exe, '--unshare-all', '--die-with-parent', '--ro-bind', '/', '/', 'true'],
                           capture_output=True, timeout=20)
    except Exception as e:                                # pragma: no cover
        return None, f'bwrap probe failed: {e!r}'
    if r.returncode != 0:
        return None, 'bwrap cannot create namespaces: ' + r.stderr.decode('utf-8', 'replace').strip()[:300]
    return exe, ''


def available() -> bool:
    return _probe()[0] is not None


def require():
    """Return the bwrap path or raise JailUnavailable (fail closed)."""
    exe, why = _probe()
    if exe is None:
        raise JailUnavailable(f'agent code sandbox unavailable: {why}. Install bubblewrap (apt-get install bubblewrap) '
                              f'and allow unprivileged user namespaces, or for LOCAL DEV ONLY set {UNSAFE_ENV}=1 to run '
                              f'agent code without OS isolation.')
    return exe


def limits(timeout_s=None, mem_mb=None, **over):
    lim = dict(DEFAULT_LIMITS)
    if timeout_s:
        lim['cpu_s'] = int(timeout_s) + 30
    if mem_mb:
        # address space is virtual: polars/numba reserve far more than they touch, so allow 2x the RSS cap (min 4 GB)
        lim['as_mb'] = max(int(mem_mb) * 2, 4000)
    lim.update(over)
    return lim


def preexec(lim):
    """preexec_fn for the bwrap process: rlimits inherited by everything in the jail."""
    def _fn():
        import resource
        r = resource
        os.setsid()
        for res, val in ((r.RLIMIT_CPU, lim['cpu_s']), (r.RLIMIT_AS, lim['as_mb'] << 20),
                         (r.RLIMIT_FSIZE, lim['fsize_mb'] << 20), (r.RLIMIT_NOFILE, lim['nofile'])):
            r.setrlimit(res, (val, val))
    return _fn


def _ro(args, p):
    p = Path(p)
    if p.exists():
        args += ['--ro-bind', str(p), str(p)]


def argv(cmd, *, rw=(), ro=(), cwd=None, lim=None, exe=None):
    """bwrap command line that runs `cmd` (a python argv: [python, script, ...]) through the in-jail boot step."""
    exe = exe or require()
    lim = lim or DEFAULT_LIMITS
    a = [exe, '--unshare-all', '--die-with-parent', '--new-session', '--uid', str(JAIL_UID), '--gid', str(JAIL_UID),
         '--hostname', 'jail', '--cap-drop', 'ALL',
         '--ro-bind', '/usr', '/usr', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/var/tmp']
    for link in ('bin', 'lib', 'lib64', 'sbin'):
        p = Path('/') / link
        if p.is_symlink():
            a += ['--symlink', os.readlink(p), str(p)]
        elif p.exists():
            _ro(a, p)
    for f in ('/etc/ld.so.cache', '/etc/localtime', '/etc/nsswitch.conf'):
        _ro(a, f)
    seen = set()
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix, str(Path(sys.executable).resolve().parent.parent)}:
        if prefix not in seen and not prefix.startswith('/usr'):
            seen.add(prefix)
            _ro(a, prefix)
    for sp in sys.path:                                   # site dirs outside the prefixes (e.g. a user site)
        if sp and Path(sp).is_dir() and 'site-packages' in sp and not any(sp.startswith(x) for x in seen | {'/usr'}):
            _ro(a, sp)
    a += ['--dir', str(ROOT)]
    _ro(a, ROOT / 'desk')                                 # code only: no .env, results/, data/, legacy/
    for p in ro:
        _ro(a, p)
    for p in rw:
        Path(p).mkdir(parents=True, exist_ok=True)
        a += ['--bind', str(p), str(p)]
    a += ['--remount-ro', '/']
    a += ['--ro-bind', str(BOOT), '/tmp/.jail_boot.py', '--setenv', 'HOME', '/tmp', '--setenv', 'TMPDIR', '/tmp',
          '--chdir', str(cwd or '/tmp'), '--']
    py = cmd[0]
    return a + [py, '-u', '/tmp/.jail_boot.py', str(lim['nproc']), str(lim['nofile'])] + list(cmd[1:])


def python_cmd(script_args, *, rw=(), ro=(), cwd=None, lim=None):
    """(argv, preexec_fn) for a jailed python run, or the unjailed argv when the local-dev opt-in is set."""
    lim = lim or DEFAULT_LIMITS
    if not available() and unsafe_opt_in():
        return [sys.executable, '-u', *script_args], None
    return argv([sys.executable, *script_args], rw=rw, ro=ro, cwd=cwd, lim=lim), preexec(lim)
