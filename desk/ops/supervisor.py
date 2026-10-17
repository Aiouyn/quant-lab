"""Lab supervisor: keeps the quant lab alive for ultra-long runs.

Every 60 s:
  * each enabled service is running (matched by its command line, so already-running processes are adopted);
    a dead service is restarted (with exponential backoff if it keeps dying);
  * code changes under desk/ (content hash) restart the long-running services so nothing runs stale code
    (recorders restart just after their hourly flush, at hh:00:45, to lose no data);
  * disk guard: below MIN_FREE_GB on C: or D: the research firm is stopped gracefully and its restart is held until
    RESUME_FREE_GB is available again (recorders keep running: they write little);
  * daily at 13:10 UTC: desk.data.refresh (forward holdout data + panel rebuild).
Config/state: results/v3/supervisor.json (enable/disable services, firm size), status in supervisor_status.json,
log in supervisor.log. Run: python -m desk.ops.supervisor   (installed as a Windows logon task by `--install`)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / 'results' / 'v3'
PY = sys.executable.replace('pythonw.exe', 'python.exe')
MIN_FREE_GB, RESUME_FREE_GB = 3.0, 6.0

DEFAULT = {
    'firm': {'enabled': True, 'cmd': ['-m', 'desk', 'run', '--agents', '16', '--hours', '8760'], 'match': 'desk run',
             'reload': 'graceful', 'disk_guarded': True},
    'parity_recorder': {'enabled': True, 'cmd': ['-m', 'desk.live.parity_recorder', '8760'], 'match': 'desk.live.parity_recorder',
                        'reload': 'after_flush'},
    'hl_recorder': {'enabled': True, 'cmd': ['-m', 'desk', 'record', '--coins', 'BTC,ETH,SOL,HYPE,XRP,DOGE,SUI,ENA', '--hours', '8760',
                                             '--archive', 'G:\\Mon Drive\\HL_L2'], 'match': 'desk record', 'reload': 'after_flush'},
    'regime': {'enabled': True, 'cmd': ['-m', 'desk.live.regime', '--loop'], 'match': 'desk.live.regime', 'reload': 'now'},
    'dashboard': {'enabled': True, 'cmd': ['-m', 'desk', 'serve', '--port', '8900'], 'match': 'desk serve', 'reload': 'now'},
    'hl_carry_paper': {'enabled': True, 'cmd': ['-m', 'desk.live.hl_carry_paper', '2160'], 'match': 'desk.live.hl_carry_paper',
                       'reload': 'never'},
    'carry_v2_paper': {'enabled': True, 'cmd': ['-u', '-m', 'desk', 'paper', 'S-ea7c43', '--hours', '1440'],
                       'match': 'desk paper S-ea7c43', 'reload': 'never'},
}


def log(msg):
    line = f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with open(HOME / 'supervisor.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def config():
    f = HOME / 'supervisor.json'
    cfg = json.loads(f.read_text()) if f.exists() else {}
    merged = {k: {**v, **cfg.get('services', {}).get(k, {})} for k, v in DEFAULT.items()}
    for k, v in cfg.get('services', {}).items():
        if k not in merged:
            merged[k] = v
    if not f.exists():
        f.write_text(json.dumps({'services': {k: {'enabled': v['enabled']} for k, v in DEFAULT.items()},
                                 'note': 'set enabled=false to stop managing a service; edit cmd to change it'}, indent=1))
    return merged


def code_hash():
    h = hashlib.sha1()
    for p in sorted((ROOT / 'desk').rglob('*.py')):
        if '__pycache__' in p.parts or 'contrib' in p.parts:      # agents' own primitives must not restart the firm
            continue
        h.update(p.as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


_PROCS = {'t': 0.0, 'list': []}


def _python_procs(max_age=5.0):
    """Python processes with their command lines, cached a few seconds: reading every process's command line on
    Windows is slow (~400 processes), and one loop looks services up a dozen times."""
    if time.time() - _PROCS['t'] > max_age:
        procs = []
        for p in psutil.process_iter(['pid', 'name']):
            try:
                if (p.info['name'] or '').lower().startswith('python'):
                    p.info['cmdline'] = p.cmdline()
                    procs.append(p)
            except (psutil.Error, TypeError):
                continue
        _PROCS.update(t=time.time(), list=procs)
    return [p for p in _PROCS['list'] if p.is_running()]


_MEM = {'last': 0.0}


def memory_pressure_log(threshold_gb=3.0, every_s=60):
    """When free RAM is low, record which processes hold it (results/v3/memory_pressure.jsonl), at most once a minute:
    the machine is shared with the user's apps and we need evidence, not guesses, about who fills it."""
    if time.time() - _MEM['last'] < every_s:
        return
    vm = psutil.virtual_memory()
    if vm.available / 1e9 >= threshold_gb:
        return
    _MEM['last'] = time.time()
    groups = {}
    for p in psutil.process_iter(['name', 'memory_info', 'cmdline']):
        try:
            name = (p.info['name'] or '?').lower()
            cl = p.info['cmdline'] or []
            if name.startswith('python') and '-m' in cl:
                name = 'lab:' + ' '.join(cl[cl.index('-m') + 1:cl.index('-m') + 3])[:40]
            elif name.startswith('python'):
                name = 'lab:sandbox'
            g = groups.setdefault(name, [0.0, 0])
            g[0] += p.info['memory_info'].rss / 1e9
            g[1] += 1
        except (psutil.Error, TypeError, AttributeError):
            continue
    top = sorted(groups.items(), key=lambda kv: -kv[1][0])[:12]
    rec = {'ts': time.time(), 'avail_gb': round(vm.available / 1e9, 2), 'used_pct': vm.percent,
           'top': [[n, round(g, 2), c] for n, (g, c) in top]}
    with open(HOME / 'memory_pressure.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')


def find(match):
    """Processes of a service: python(w) whose ARGUMENTS contain the match tokens in order right after '-m'
    (e.g. 'desk run' -> [..., '-m', 'desk', 'run', ...]). A shell or script that merely mentions the text never matches."""
    toks = match.split()
    out = []
    for p in _python_procs():
        try:
            cl = p.info['cmdline'] or []
            if '-m' not in cl:
                continue
            i = cl.index('-m') + 1
            if cl[i:i + len(toks)] == toks:
                out.append(p)
        except (psutil.Error, TypeError, ValueError):
            continue
    return out


def start(name, svc):
    logs = HOME / 'services'
    logs.mkdir(exist_ok=True)
    out = open(logs / f'{name}.log', 'a', encoding='utf-8'); err = open(logs / f'{name}.err', 'a', encoding='utf-8')
    flags = 0x08000000 | 0x00000200                     # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen([PY] + svc['cmd'], cwd=str(ROOT), stdout=out, stderr=err, creationflags=flags,
                         env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    _PROCS['t'] = 0.0                                     # the process list changed: next lookup rescans
    log(f'started {name} pid {p.pid}')
    return p.pid


def request_graceful_stop(name):
    """Non-blocking: the firm finishes its current steps and exits; the loop kills it after 15 min if needed."""
    if name == 'firm':
        subprocess.run([PY, '-m', 'desk', 'stop'], cwd=str(ROOT), capture_output=True, timeout=120)
    log(f'graceful stop requested for {name}')


def stop(name, procs, graceful=False):
    for p in procs:
        try:
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except psutil.Error:
            pass
    _PROCS['t'] = 0.0
    log(f'stopped {name} ({len(procs)} process(es), graceful={graceful})')


def free_gb(drive):
    try:
        return shutil.disk_usage(drive).free / 1e9
    except OSError:
        return 1e9


def is_supervisor(proc_info):
    """A real supervisor: a python(w) process started as `-m desk.ops.supervisor` (not any shell whose command text
    merely mentions the module - that false match once made a handover exit with no supervisor left)."""
    cl = proc_info.get('cmdline') or []
    name = (proc_info.get('name') or '').lower()
    if not name.startswith('python'):
        return False
    return any(cl[i] == '-m' and i + 1 < len(cl) and cl[i + 1] == 'desk.ops.supervisor' for i in range(len(cl)))         and '--install' not in cl


def other_supervisor():
    me = os.getpid()
    for p in psutil.process_iter(['pid', 'cmdline', 'name']):
        try:
            if p.info['pid'] != me and is_supervisor(p.info):
                return p.info['pid']
        except (psutil.Error, TypeError):
            continue
    return None


def main():
    if '--after' in sys.argv:                                     # handover: wait for the previous supervisor to exit
        try:
            psutil.Process(int(sys.argv[sys.argv.index('--after') + 1])).wait(timeout=120)
        except (psutil.Error, ValueError, IndexError):
            pass
    if other_supervisor():
        log(f'supervisor already running (pid {other_supervisor()}): exiting')
        return
    log(f'supervisor start pid {os.getpid()}')
    h0 = code_hash()
    me0 = Path(__file__).read_bytes()
    # processes found at startup may run older code: reload them once (paper sessions excepted: reload='never')
    cold = '--after' not in sys.argv                                # after a handover the services are already fresh
    state = {k: {'restarts': [], 'pending_reload': cold, 'stopping_since': None} for k in DEFAULT}
    disk_paused = False
    last_refresh_day = None
    pending_hash, pending_since = None, 0.0
    last_kpi_hour = None
    while True:
        if Path(__file__).read_bytes() != me0:                    # the supervisor itself changed: hand over
            log('supervisor code changed: handing over to a fresh supervisor')
            subprocess.Popen([PY.replace('python.exe', 'pythonw.exe'), '-m', 'desk.ops.supervisor', '--after', str(os.getpid())],
                             cwd=str(ROOT), creationflags=0x08000000 | 0x00000008)
            return
        cfg = config()
        now = datetime.now(timezone.utc)
        h = code_hash()
        if h != h0:                                               # debounce: reload once the code is stable for 2 min
            if h != pending_hash:
                pending_hash, pending_since = h, time.time()
            elif time.time() - pending_since >= 120:
                log('code change detected (stable 2 min): services will reload')
                for k in state:
                    state[k]['pending_reload'] = True
                h0 = h
        low = min(free_gb('C:\\'), free_gb('D:\\'))
        if low < MIN_FREE_GB and not disk_paused:
            disk_paused = True
            log(f'DISK GUARD: {low:.1f} GB free -> pausing the firm')
            request_graceful_stop('firm')
            state['firm']['stopping_since'] = time.time()
        elif disk_paused and low >= RESUME_FREE_GB:
            disk_paused = False
            log(f'disk recovered ({low:.1f} GB): firm may restart')
        status = {'updated': time.time(), 'pid': os.getpid(), 'disk_free_min_gb': round(low, 1), 'disk_paused': disk_paused,
                  'services': {}}
        for name, svc in cfg.items():
            st = state.setdefault(name, {'restarts': [], 'pending_reload': False, 'stopping_since': None})
            procs = find(svc['match'])
            if not svc.get('enabled', True):
                status['services'][name] = {'enabled': False, 'running': bool(procs)}
                continue
            if procs and st.get('stopping_since'):                       # graceful stop in progress (non-blocking)
                if time.time() - st['stopping_since'] > 900:
                    stop(name, procs)
                    procs = []
                    st['stopping_since'] = None
                else:
                    status['services'][name] = {'enabled': True, 'running': True, 'stopping': True}
                    continue
            elif not procs:
                st['stopping_since'] = None
            if svc.get('disk_guarded') and disk_paused:
                status['services'][name] = {'enabled': True, 'running': bool(procs), 'held': 'disk'}
                continue
            if procs and st['pending_reload'] and svc.get('reload', 'now') != 'never':
                due = svc['reload'] in ('now', 'graceful') or (svc['reload'] == 'after_flush' and now.minute == 0 and now.second >= 45) \
                    or (svc['reload'] == 'after_flush' and now.minute == 1)
                if due and svc['reload'] == 'graceful':
                    request_graceful_stop(name)
                    st['stopping_since'] = time.time()
                    st['pending_reload'] = False
                    status['services'][name] = {'enabled': True, 'running': True, 'stopping': True}
                    continue
                if due:
                    stop(name, procs)
                    procs = []
                    st['pending_reload'] = False
                    start(name, svc)                              # a reload is not a crash: no backoff, not counted
                    procs = find(svc['match'])
            elif not procs:
                st['pending_reload'] = False
            if not procs:                                         # crashed / died: restart with exponential backoff
                st['restarts'] = [t for t in st['restarts'] if time.time() - t < 3600]
                backoff = min(3600, 60 * 2 ** max(0, len(st['restarts']) - 2))
                if not st['restarts'] or time.time() - st['restarts'][-1] >= backoff:
                    start(name, svc)
                    st['restarts'].append(time.time())
            status['services'][name] = {'enabled': True, 'running': bool(find(svc['match'])),
                                        'restarts_last_hour': len(st['restarts']), 'pending_reload': st['pending_reload']}
        memory_pressure_log()
        if now.minute >= 5 and last_kpi_hour != (now.date(), now.hour):        # hourly firm performance snapshot
            last_kpi_hour = (now.date(), now.hour)
            try:
                from .kpi import report
                from ..config import SETTINGS
                k = report(SETTINGS, 1.0)
                k.pop('per_agent', None)
                with open(HOME / 'kpi_history.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'ts': time.time(), **k}, default=str) + '\n')
            except Exception as e:
                log(f'kpi snapshot failed: {e!r}'[:300])
        if now.hour == 13 and now.minute >= 10 and last_refresh_day != now.date():
            last_refresh_day = now.date()
            log('daily refresh (forward data + perps panel)')
            subprocess.Popen([PY, '-m', 'desk.data.refresh'], cwd=str(ROOT), creationflags=0x08000000,
                             stdout=open(HOME / 'services' / 'refresh.log', 'a'), stderr=open(HOME / 'services' / 'refresh.err', 'a'))
        (HOME / 'supervisor_status.json').write_text(json.dumps(status, indent=1))
        time.sleep(60)


def install():
    """Windows logon task so the lab survives reboots."""
    cmd = HOME / 'start_supervisor.cmd'
    cmd.write_text(f'@echo off\r\ncd /d "{ROOT}"\r\n"{PY.replace("python.exe", "pythonw.exe")}" -m desk.ops.supervisor\r\n')
    r = subprocess.run(['schtasks', '/Create', '/F', '/TN', 'QuantLabSupervisor', '/SC', 'ONLOGON', '/RL', 'LIMITED',
                        '/TR', f'"{cmd}"'], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


if __name__ == '__main__':
    if '--install' in sys.argv:
        print(install())
    else:
        main()
