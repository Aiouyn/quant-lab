"""Isolated execution of agent Python code and trusted harness jobs (async subprocesses).

Sandbox: credential-free environment, dev-only data view, thread caps, wall-clock timeout, an RSS watchdog, and a
guard installed before the agent's code runs (desk/lab/guard.py): the agent can read code, the Python installation
and the dev-only data views, write only inside its own workspace, and cannot open sockets or start programs. On Linux
(including WSL2) the kernel enforces this (Landlock + seccomp); on Windows/macOS only the Python-level audit hook does,
which stops accidents and casual peeking but is not a security boundary against deliberately hostile code.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from ..config import ROOT

SAFE_ENV = ('SYSTEMROOT', 'SystemRoot', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'LOCALAPPDATA',
            'APPDATA', 'PROCESSOR_ARCHITECTURE', 'NUMBER_OF_PROCESSORS', 'COMSPEC', 'WINDIR', 'PATHEXT')

PREAMBLE = '''import sys
sys.path[0] = {root!r}                       # not the scripts folder: a planted module there must not shadow anything
from desk.lab.guard import install_sandbox_guard as _guard
_guard({policy!r})
del _guard
import warnings
warnings.filterwarnings('ignore')
from desk.research import *
'''


def guard_policy(settings, workspace: Path, data_env: dict) -> dict:
    """What an agent sandbox may touch. Everything under results/ (the ledger, the full store, other agents'
    workspaces, val/holdout views) is denied except the dev views named in data_env and the agent's own workspace."""
    from .guard import python_roots
    root = Path(ROOT).resolve()
    deny = {root / 'results', root / 'data', root / '.env', root / '.git', Path(settings.home), Path(settings.store),
            Path(settings.data_dir)}
    deny_r = {str(Path(p).resolve()) for p in deny}

    def hidden(p):
        rp = str(p.resolve())
        return any(rp == d or rp.startswith(d + os.sep) or d.startswith(rp + os.sep) for d in deny_r)
    code = [str(p) for p in root.iterdir() if not hidden(p)]
    allow = sorted({str(v) for k, v in data_env.items() if k.startswith('DESK_') and v and Path(v).exists()})
    from .guard import norm
    n = lambda xs: [norm(x) or x for x in xs]          # one spelling per path (Windows paths are case-insensitive)
    return {'read_roots': n(python_roots(exclude=[str(root)] + sorted(deny_r)) + code), 'allow_roots': n(allow),
            'write_roots': n([str(Path(workspace).resolve())]), 'deny_roots': n(sorted(deny_r))}


def _env(extra):
    env = {k: v for k, v in os.environ.items() if k in SAFE_ENV}
    env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'POLARS_MAX_THREADS': '3', 'NUMBA_NUM_THREADS': '2',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'PYTHONDONTWRITEBYTECODE': '1'})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _clip(text, head=2500, tail=6500):
    if len(text) <= head + tail:
        return text
    return text[:head] + f'\n... [{len(text) - head - tail} chars omitted] ...\n' + text[-tail:]


SYSTEM_FLOOR_MB = 1500                                   # free-RAM floor for the whole machine


async def _watch(proc, timeout_s, mem_mb):
    start = time.time()
    peak = 0
    ps = None
    if psutil is not None:
        try:
            ps = psutil.Process(proc.pid)
        except psutil.Error:
            ps = None
    while proc.returncode is None:
        await asyncio.sleep(0.5)
        if ps is not None:
            try:
                rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                peak = max(peak, rss)
                if rss > mem_mb * 1e6:
                    _kill(proc)
                    return 'memory', peak
                # the whole machine is nearly out of RAM (lab jobs + the user's own apps): stop the big lab process
                # rather than letting Windows thrash or reap unrelated programs
                if rss > 1e9 and psutil.virtual_memory().available < SYSTEM_FLOOR_MB * 1e6:
                    _kill(proc)
                    return 'system_memory', peak
            except psutil.Error:
                pass
        if time.time() - start > timeout_s:
            _kill(proc)
            return 'timeout', peak
    return None, peak


def _kill(proc):
    if psutil is None:
        try:
            proc.kill()
        except Exception:
            pass
        return
    try:
        p = psutil.Process(proc.pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except psutil.Error:
        pass


async def _wait_for_memory(mem_mb, headroom_mb=1500, typical_mb=2000, max_wait_s=900):
    """Start a subprocess only when the machine has room for a TYPICAL run (most use < 1 GB) plus headroom for the
    user's own apps. Requiring the full cap stalled the firm for 5 h on 2026-10-07 (cap 4 GB + 4 GB headroom vs ~5 GB
    free): every run waited 15 min, then finished in 1.5 s. Runs that do grow are stopped by _watch's system floor."""
    if psutil is None:
        return 0.0
    need = min(mem_mb, typical_mb) + headroom_mb
    t0 = time.time()
    while psutil.virtual_memory().available / 1e6 < need and time.time() - t0 < max_wait_s:
        await asyncio.sleep(5)
    return round(time.time() - t0, 1)


# agent run_python peaks are ~0.1-0.5 GB (measured 119 MB for a full perps_panel load); the old 2 GB + 1.5 GB gate
# blocked every call for 15 min whenever the owner's apps left < 3.5 GB free (2026-11-05). Wait briefly, then tell
# the agent instead of burning its session; _watch's system floor still stops a run that grows.
RUN_TYPICAL_MB, RUN_HEADROOM_MB, RUN_MAX_WAIT_S = 700, 800, 120


def _ram_free_mb():
    return psutil.virtual_memory().available / 1e6 if psutil is not None else float('inf')


async def run_python(code: str, workspace: Path, settings, timeout_s=None, mem_mb=None, slot=None, extra_env=None):
    """Run agent code with `from desk.research import *` preloaded. Returns stdout/stderr/status."""
    timeout_s = min(int(timeout_s or settings.sandbox_timeout_s), 1800)
    mem_mb = mem_mb or settings.sandbox_memory_mb
    scripts = Path(workspace) / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    name = time.strftime('%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
    path = scripts / f'{name}.py'
    # `from __future__` must be the first statement: hoist it above the preamble
    fut = [l for l in code.splitlines() if l.startswith('from __future__ import')]
    body = '\n'.join(l for l in code.splitlines() if not l.startswith('from __future__ import'))
    data_env = {'DESK_STORE_ROOT': settings.store / 'views' / 'dev',
                'DESK_LIBRARY': settings.artifacts / 'library.json',
                'DESK_CARRY_ROOT': settings.store / 'carry_views' / 'dev' / 'carry',   # dev-only copy (lab.carry)
                'DESK_PERPS_DEV': settings.store / 'perps' / 'views' / 'dev.parquet',
                'DESK_MACRO_DEV': settings.store / 'macro' / 'views' / 'dev.parquet',
                'DESK_PERPS_DEV_FILLS': settings.store / 'perps' / 'views' / 'dev_fills.parquet',
                'DESK_PRED_DEV': settings.store / 'pred' / 'views' / 'dev.parquet',
                'DESK_PRED_DEV_OUTCOMES': settings.store / 'pred' / 'views' / 'dev_outcomes.parquet', **(extra_env or {})}
    ws = Path(workspace).resolve()
    tmp = ws / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    policy = guard_policy(settings, ws, data_env)
    path.write_text(''.join(f + '\n' for f in fut) + PREAMBLE.format(root=str(ROOT), policy=policy) + body,
                    encoding='utf-8')
    env = _env({**data_env, 'TMP': tmp, 'TEMP': tmp, 'TMPDIR': tmp,
                'NUMBA_CACHE_DIR': ws / '.numba_cache', 'MPLCONFIGDIR': ws / '.mpl'})
    tq = time.time()
    async with (slot or _nullslot()):
        await _wait_for_memory(mem_mb, RUN_HEADROOM_MB, RUN_TYPICAL_MB, RUN_MAX_WAIT_S)
        if _ram_free_mb() < RUN_TYPICAL_MB + RUN_HEADROOM_MB:
            return {'status': 'queued_low_ram', 'seconds': 0.0, 'queued_s': round(time.time() - tq, 1), 'peak_mb': 0,
                    'stdout': '', 'stderr': f'The machine has under {(RUN_TYPICAL_MB + RUN_HEADROOM_MB) / 1000:.1f} GB '
                    'of free RAM (the owner is using it). Your code did NOT run. Do other work (read, plan, write the '
                    'notebook) and retry in a few minutes; keep runs small.', 'script': str(path.relative_to(workspace))}
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(sys.executable, '-u', str(path), cwd=str(workspace), env=env,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        watcher = asyncio.create_task(_watch(proc, timeout_s, mem_mb))
        out, err = await proc.communicate()
        killed, peak = await watcher
    status = 'ok' if proc.returncode == 0 and not killed else (f'killed:{killed}' if killed else f'error:{proc.returncode}')
    return {'status': status, 'seconds': round(time.time() - t0, 1), 'queued_s': round(t0 - tq, 1), 'peak_mb': round(peak / 1e6),
            'stdout': _clip(out.decode('utf-8', 'replace')), 'stderr': _clip(err.decode('utf-8', 'replace'), 1500, 3500),
            'script': str(path.relative_to(workspace))}


async def run_job(job: dict, settings, timeout_s=3600, mem_mb=None, slot=None):
    """Trusted harness job in a subprocess (full store access decided by the caller)."""
    jobs = settings.home / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    jid = uuid.uuid4().hex[:12]
    jp, op = jobs / f'{jid}.json', jobs / f'{jid}.out.json'
    mem_mb = mem_mb or getattr(settings, 'job_memory_mb', 5000)
    job = dict(job, _home=str(settings.home), _store=str(settings.store))   # the harness evaluates on THIS run's store
    jp.write_text(json.dumps(job, default=str))
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', POLARS_MAX_THREADS='4', NUMBA_NUM_THREADS='2')
    for secret_key in ('MIMO_API_KEY', 'OPENCODE_API_KEY', 'OPENAI_API_KEY'):
        env.pop(secret_key, None)
    async with (slot or _nullslot()):
        await _wait_for_memory(mem_mb)
        proc = await asyncio.create_subprocess_exec(sys.executable, '-m', 'desk.lab.harness', str(jp), str(op), cwd=str(ROOT),
                                                    env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        watcher = asyncio.create_task(_watch(proc, timeout_s, mem_mb))
        out, err = await proc.communicate()
        killed, _ = await watcher
    try:
        if op.exists():
            res = json.loads(op.read_text())
        else:
            res = {'error': f'harness failed ({killed or proc.returncode})', 'stderr': _clip(err.decode('utf-8', 'replace'), 800, 2500)}
    finally:
        for p in (jp, op):
            p.unlink(missing_ok=True)
    return res


class _nullslot:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False
