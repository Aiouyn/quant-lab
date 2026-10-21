"""Isolated execution of agent Python code and trusted harness jobs (async subprocesses).

Sandbox: credential-free environment, dev-only data view, thread caps, wall-clock timeout and an RSS
watchdog, and an OS-level jail (desk/lab/jail.py: bubblewrap with no network, a read-only minimal filesystem, the dev
data view read-only, a private /tmp, the workspace as the only writable host path, rlimits and a non-root uid).
If bubblewrap is unavailable run_python fails closed (JailUnavailable) unless DESK_SANDBOX_UNSAFE=1 is set for local dev.
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
from . import jail

SAFE_ENV = ('SYSTEMROOT', 'SystemRoot', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'LOCALAPPDATA',
            'APPDATA', 'PROCESSOR_ARCHITECTURE', 'NUMBER_OF_PROCESSORS', 'COMSPEC', 'WINDIR', 'PATHEXT')

PREAMBLE = '''import sys, warnings
sys.path.insert(0, {root!r})
warnings.filterwarnings('ignore')
from desk.research import *
'''


def _env(extra):
    env = {k: v for k, v in os.environ.items() if k in SAFE_ENV}
    env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'POLARS_MAX_THREADS': '3', 'NUMBA_NUM_THREADS': '2',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'PYTHONDONTWRITEBYTECODE': '1'})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _carry_dev_view(settings):
    """Dev-only copy of the carry store (splits.json with the dev split only + coin files cut at the dev end), so the
    jail never mounts the full carry store. Rebuilt when the source changes. Returns <view>/carry."""
    src = settings.store / 'carry'
    view = settings.store / 'views' / 'carry_dev' / 'carry'
    if not (src / 'splits.json').exists():
        return view
    from datetime import datetime, timezone
    import shutil
    import polars as pl
    files = sorted((src / 'coins').glob('*.parquet'))
    stamp = json.dumps([(src / 'splits.json').stat().st_mtime] + [(f.name, f.stat().st_mtime) for f in files])
    if (view / '.stamp').exists() and (view / '.stamp').read_text() == stamp:
        return view
    dev = json.loads((src / 'splits.json').read_text())['dev']
    end = int(datetime.fromisoformat(dev[1]).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
    tmp = view.with_name('carry.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'coins').mkdir(parents=True)
    (tmp / 'splits.json').write_text(json.dumps({'dev': dev}))
    for f in files:
        pl.read_parquet(f).filter(pl.col('ts') < end).write_parquet(tmp / 'coins' / f.name)
    (tmp / '.stamp').write_text(stamp)
    shutil.rmtree(view, ignore_errors=True)
    tmp.rename(view)
    return view


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


async def run_python(code: str, workspace: Path, settings, timeout_s=None, mem_mb=None, slot=None, extra_env=None):
    """Run agent code with `from desk.research import *` preloaded. Returns stdout/stderr/status."""
    jail.require() if not jail.unsafe_opt_in() else None    # fail closed before writing anything
    timeout_s = min(int(timeout_s or settings.sandbox_timeout_s), 1800)
    mem_mb = mem_mb or settings.sandbox_memory_mb
    scripts = Path(workspace) / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    name = time.strftime('%m%d_%H%M%S_') + uuid.uuid4().hex[:4]
    path = scripts / f'{name}.py'
    # `from __future__` must be the first statement: hoist it above the preamble
    fut = [l for l in code.splitlines() if l.startswith('from __future__ import')]
    body = '\n'.join(l for l in code.splitlines() if not l.startswith('from __future__ import'))
    path.write_text(''.join(f + '\n' for f in fut) + PREAMBLE.format(root=str(ROOT)) + body, encoding='utf-8')
    env = _env({'DESK_STORE_ROOT': settings.store / 'views' / 'dev',
                'DESK_LIBRARY': settings.artifacts / 'library.json',
                'DESK_CARRY_ROOT': _carry_dev_view(settings),
                'DESK_PERPS_DEV': settings.store / 'perps' / 'views' / 'dev.parquet',
                'DESK_MACRO_DEV': settings.store / 'macro' / 'views' / 'dev.parquet',
                'DESK_PERPS_DEV_FILLS': settings.store / 'perps' / 'views' / 'dev_fills.parquet',
                'DESK_PRED_DEV': settings.store / 'pred' / 'views' / 'dev.parquet',
                'DESK_PRED_DEV_OUTCOMES': settings.store / 'pred' / 'views' / 'dev_outcomes.parquet', **(extra_env or {})})
    tq = time.time()
    async with (slot or _nullslot()):
        await _wait_for_memory(mem_mb)
        t0 = time.time()
        ws = Path(workspace).resolve()
        ro = [v for k, v in env.items() if k.startswith('DESK_') and k != 'DESK_TRIALS' and Path(v).is_absolute()]
        argv, pre = jail.python_cmd([str(path.resolve())], rw=[ws], ro=ro, cwd=ws,
                                    lim=jail.limits(timeout_s=timeout_s, mem_mb=mem_mb))
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(workspace), env=env, preexec_fn=pre,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        watcher = asyncio.create_task(_watch(proc, timeout_s, mem_mb))
        out, err = await proc.communicate()
        killed, peak = await watcher
    status = 'ok' if proc.returncode == 0 and not killed else (f'killed:{killed}' if killed else f'error:{proc.returncode}')
    return {'status': status, 'seconds': round(time.time() - t0, 1), 'queued_s': round(t0 - tq, 1), 'peak_mb': round(peak / 1e6),
            'stdout': _clip(out.decode('utf-8', 'replace')), 'stderr': _clip(err.decode('utf-8', 'replace'), 1500, 3500),
            'script': str(path.relative_to(workspace))}


JOB_ENV = ('PATH', 'LANG', 'LC_ALL', 'TZ', 'RESULTS_DIR', 'DATABASE_PATH')


def _job_env():
    """Allowlist for the jailed harness: DESK_* settings + locale/path, never credentials."""
    env = {k: v for k, v in os.environ.items() if k in JOB_ENV or k.startswith('DESK_')}
    for k in list(env):
        if any(t in k.upper() for t in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            env.pop(k)
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', POLARS_MAX_THREADS='4', NUMBA_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    return env


async def run_job(job: dict, settings, timeout_s=3600, mem_mb=None, slot=None):
    """Harness job (it executes agent strategy/factor modules) in the OS jail.

    The store and artifacts are mounted READ-ONLY (split access is still enforced by the harness stores); the only
    writable host paths are this job's private dir and the job's values_dir (factor value cache). No network, no
    credentials, non-root uid, rlimits. Fails closed without bwrap unless DESK_SANDBOX_UNSAFE=1 (local dev)."""
    if not jail.unsafe_opt_in():
        jail.require()
    jid = uuid.uuid4().hex[:12]
    jdir = settings.home / 'jobs' / jid
    jdir.mkdir(parents=True, exist_ok=True)
    jp, op = jdir / 'job.json', jdir / 'out.json'
    mem_mb = mem_mb or getattr(settings, 'job_memory_mb', 5000)
    job = dict(job, _home=str(settings.home), _store=str(settings.store))   # the harness evaluates on THIS run's store
    jp.write_text(json.dumps(job, default=str))
    rw = [jdir.resolve()]
    if job.get('values_dir'):
        rw.append(Path(job['values_dir']).resolve())
    ro = [Path(settings.store).resolve(), Path(settings.artifacts).resolve()]
    argv, pre = jail.python_cmd([str(Path(__file__).resolve().parent / '_harness_entry.py'), str(jp.resolve()), str(op.resolve())],
                                rw=rw, ro=ro, cwd=ROOT, lim=jail.limits(timeout_s=timeout_s, mem_mb=mem_mb))
    if argv[0] == sys.executable:                                      # unsafe local-dev opt-in: old module invocation
        argv = [sys.executable, '-m', 'desk.lab.harness', str(jp), str(op)]
    async with (slot or _nullslot()):
        await _wait_for_memory(mem_mb)
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(ROOT), env=_job_env(), preexec_fn=pre,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        watcher = asyncio.create_task(_watch(proc, timeout_s, mem_mb))
        out, err = await proc.communicate()
        killed, _ = await watcher
    try:
        if op.exists():
            res = json.loads(op.read_text())
        else:
            res = {'error': f'harness failed ({killed or proc.returncode})', 'stderr': _clip(err.decode('utf-8', 'replace'), 800, 2500)}
    finally:
        import shutil
        shutil.rmtree(jdir, ignore_errors=True)
    return res


class _nullslot:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False
