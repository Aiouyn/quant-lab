"""Process guards for untrusted code: the strategy jail (desk/lab/jail_worker.py) and agent sandboxes (sandbox.py).

Standard library only, and no desk imports: this module is loaded into untrusted processes before anything else.

Three layers, strongest first. Each process gets as many as its platform supports and reports which ones it has.
  1. Landlock (Linux >= 5.13, WSL2 with a recent kernel): the KERNEL refuses to open files outside an allowlist of
     directories. This also covers native code (polars' Rust readers, C extensions) that never calls Python's open().
  2. seccomp (Linux): the kernel refuses exec, fork (threads are fine), sockets, ptrace and cross-process memory.
  3. An audit hook (all platforms, Python level): refuses file opens outside the allowlist, writes outside the
     writable roots, sockets, subprocesses, ctypes, and imports of modules that bypass it. Python's documentation is
     explicit that audit hooks are not a sandbox on their own: on Windows/macOS treat this layer as a guard against
     accidents and casual cheating, not against a determined adversary (use WSL2 or a container for that).
"""
from __future__ import annotations

import os
import sys

ACTIVE = None          # 'sandbox' once install_sandbox_guard ran in this process (never set in trusted processes)

BLOCKED_EVENT_PREFIXES = (
    'socket.', 'subprocess.', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork', 'os.forkpty',
    'os.kill', 'os.killpg', 'os.startfile', 'os.chmod', 'os.chown', 'os.chflags', 'os.lchown', 'os.symlink',
    'os.link', 'os.setxattr', 'os.removexattr', 'shutil.chown', 'winreg.',
    '_winapi.', '_overlapped.', 'webbrowser.', 'urllib.', 'http.', 'ftplib.', 'smtplib.', 'imaplib.', 'poplib.',
    'nntplib.', 'telnetlib.', 'sys.remote_exec', 'cpython.PyInterpreterState_New', 'pty.', 'resource.setrlimit',
    'signal.pthread_kill', 'ctypes.dlsym/handle',
)
# events that change the file system: allowed only inside the writable roots (value = index of the path argument)
WRITE_EVENTS = {'os.remove': 0, 'os.unlink': 0, 'os.rmdir': 0, 'os.mkdir': 0, 'os.rename': 0, 'os.replace': 0,
                'os.truncate': 0, 'os.utime': 0, 'shutil.rmtree': 0, 'shutil.copyfile': 1, 'shutil.copytree': 1,
                'shutil.move': 1, 'shutil.make_archive': None, 'shutil.unpack_archive': 1}
# Importing subprocess, socket or ctypes is allowed (scipy, sklearn, lightgbm import them): USING them is what the
# hook refuses (subprocess.Popen, socket.__new__, loading a native library from outside the Python installation).
BLOCKED_MODULES = {
    'cffi', '_cffi_backend', '_interpreters', '_xxsubinterpreters', 'concurrent.interpreters', 'webbrowser',
    'requests', 'httpx', 'urllib3', 'aiohttp', 'zmq',
}


def norm(p):
    try:
        return os.path.normcase(os.path.realpath(os.fsdecode(p)))
    except Exception:
        return None


def under(path, roots):
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def python_roots(exclude=()):
    """Directories that hold the interpreter, the standard library and installed packages (read-only)."""
    ex = [norm(e) for e in exclude if e]
    roots = set()
    for p in list(sys.path) + [sys.prefix, sys.base_prefix, sys.exec_prefix]:
        rp = norm(p) if p else None
        if not rp or not os.path.exists(rp):
            continue
        if any(rp == e or e.startswith(rp + os.sep) for e in ex):
            continue                                   # never a directory that contains the repo or the data
        if any(rp.startswith(e + os.sep) for e in ex) and 'site-packages' not in rp and 'dist-packages' not in rp:
            continue                                   # repo subdirectories, except a venv inside the repo
        roots.add(rp)
    return sorted(roots)


# ------------------------------------------------------------------------------------------------------- seccomp
_SYSCALLS = {
    # audit arch: (syscalls answered with EPERM, clone, clone3)
    #   x86_64 : execve execveat fork vfork socket connect bind listen ptrace process_vm_readv/writev kill tkill
    #            io_uring_setup bpf perf_event_open userfaultfd
    0xC000003E: ({59, 322, 57, 58, 41, 42, 49, 50, 101, 310, 311, 62, 200, 425, 321, 298, 323}, 56, 435),
    #   aarch64: execve execveat socket connect bind listen ptrace process_vm_readv/writev kill tkill
    #            io_uring_setup bpf perf_event_open userfaultfd
    0xC00000B7: ({221, 281, 198, 203, 200, 201, 117, 270, 271, 129, 130, 425, 280, 241, 282}, 220, 435),
}
_ARCH = {'x86_64': 0xC000003E, 'amd64': 0xC000003E, 'aarch64': 0xC00000B7, 'arm64': 0xC00000B7}


def _libc():
    import ctypes
    return ctypes, ctypes.CDLL(None, use_errno=True)


def install_seccomp() -> bool:
    """No exec, no fork (threads allowed), no sockets, no ptrace. Linux only; returns whether it is active."""
    if not sys.platform.startswith('linux'):
        return False
    try:
        import platform
        arch = _ARCH.get(platform.machine().lower())
        if arch is None:
            return False
        ctypes, libc = _libc()
        blocked, nr_clone, nr_clone3 = _SYSCALLS[arch]
        LD, JEQ, JGE, JSET, RET = 0x20, 0x15, 0x35, 0x45, 0x06
        ALLOW, EPERM, ENOSYS, KILL = 0x7FFF0000, 0x00050001, 0x00050026, 0x80000000
        prog = [('stmt', LD, 4), ('jeq', arch, None, 'kill'), ('stmt', LD, 0)]
        if arch == 0xC000003E:
            prog.append(('jge', 0x40000000, 'eperm', None))           # x32 ABI syscalls
        prog.append(('jeq', nr_clone3, 'enosys', None))               # glibc then falls back to clone()
        prog.append(('jeq', nr_clone, 'clone', None))
        prog += [('jeq', nr, 'eperm', None) for nr in sorted(blocked)]
        prog += [('stmt', RET, ALLOW),                                # BPF jumps only go forward
                 ('label', 'clone'), ('stmt', LD, 16), ('jset', 0x00010000, 'allow', 'eperm'),   # CLONE_THREAD
                 ('label', 'allow'), ('stmt', RET, ALLOW),
                 ('label', 'eperm'), ('stmt', RET, EPERM),
                 ('label', 'enosys'), ('stmt', RET, ENOSYS),
                 ('label', 'kill'), ('stmt', RET, KILL)]
        labels, code = {}, []
        for ins in prog:
            if ins[0] == 'label':
                labels[ins[1]] = len(code)
            else:
                code.append(ins)

        class F(ctypes.Structure):
            _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte), ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]

        class P(ctypes.Structure):
            _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(F))]

        arr = (F * len(code))()
        for i, ins in enumerate(code):
            if ins[0] == 'stmt':
                arr[i] = F(ins[1], 0, 0, ins[2])
            else:
                op = {'jeq': JEQ, 'jge': JGE, 'jset': JSET}[ins[0]]
                jt = labels[ins[2]] - i - 1 if ins[2] else 0
                jf = labels[ins[3]] - i - 1 if ins[3] else 0
                arr[i] = F(op, jt, jf, ins[1])
        fprog = P(len(code), ctypes.cast(arr, ctypes.POINTER(F)))
        if libc.prctl(38, 1, 0, 0, 0) != 0:                            # PR_SET_NO_NEW_PRIVS
            return False
        return libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) == 0       # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------------ landlock
# shared libraries that compiled packages link against (libgomp for lightgbm, libstdc++ ...): readable at the kernel
# level so the dynamic loader can map them. Not added to the audit hook's roots, so ctypes cannot load libc directly.
SYSTEM_LIB_PATHS = ('/lib', '/lib64', '/usr/lib', '/usr/lib64', '/usr/local/lib', '/etc/ld.so.cache',
                    '/etc/ld.so.conf', '/etc/ld.so.conf.d')

_LL_EXECUTE, _LL_WRITE_FILE, _LL_READ_FILE, _LL_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_LL_ALL_V1 = (1 << 13) - 1                                            # execute .. make_sym
_LL_REFER, _LL_TRUNCATE, _LL_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15


def landlock_abi() -> int:
    """The Landlock ABI this kernel offers (0 = none). Does not restrict anything."""
    if not sys.platform.startswith('linux'):
        return 0
    try:
        _, libc = _libc()
        return max(0, int(libc.syscall(444, None, 0, 1)))
    except Exception:
        return 0


def install_landlock(read_paths, write_paths) -> int:
    """Kernel file-system allowlist. Returns the Landlock ABI in force, 0 when unavailable (non-Linux, old kernel)."""
    if not sys.platform.startswith('linux'):
        return 0
    try:
        ctypes, libc = _libc()
        abi = libc.syscall(444, None, 0, 1)                           # landlock_create_ruleset(.., VERSION)
        if abi < 1:
            return 0
        handled = _LL_ALL_V1 | (_LL_REFER if abi >= 2 else 0) | (_LL_TRUNCATE if abi >= 3 else 0) \
            | (_LL_IOCTL_DEV if abi >= 5 else 0)

        class Attr(ctypes.Structure):
            _fields_ = [('fs', ctypes.c_uint64)]

        class Beneath(ctypes.Structure):
            _pack_ = 1
            _fields_ = [('allowed', ctypes.c_uint64), ('fd', ctypes.c_int32)]

        attr = Attr(handled)
        rs = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
        if rs < 0:
            return 0
        read_dir = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
        read_file = _LL_EXECUTE | _LL_READ_FILE
        write_extra = handled & ~(_LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR)

        def add(path, rights):
            try:
                fd = os.open(path, os.O_PATH | getattr(os, 'O_CLOEXEC', 0))
            except OSError:
                return
            try:
                if not os.path.isdir(path):
                    rights &= _LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE | _LL_TRUNCATE | _LL_IOCTL_DEV
                b = Beneath(rights & handled, fd)
                libc.syscall(445, rs, 1, ctypes.byref(b), 0)            # LANDLOCK_RULE_PATH_BENEATH
            finally:
                os.close(fd)

        for p in list(read_paths) + [x for x in SYSTEM_LIB_PATHS if os.path.exists(x)]:
            add(p, read_dir if os.path.isdir(p) else read_file)
        for p in write_paths:
            add(p, read_dir | write_extra)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            os.close(rs)
            return 0
        ok = libc.syscall(446, rs, 0) == 0                              # landlock_restrict_self
        os.close(rs)
        return abi if ok else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------------------------------- audit hook
def install_audit_hook(read_roots, write_roots=(), deny_roots=(), allow_roots=(), allow_fd_open=False,
                       blocked_modules=BLOCKED_MODULES, what='this code'):
    """Python-level guard. Precedence for a path: writable roots, then explicit allow roots (read), then deny
    roots, then general read roots; anything else is refused. Hooks cannot be removed once installed."""
    read_roots = [r for r in (norm(p) for p in read_roots) if r]
    write_roots = [r for r in (norm(p) for p in write_roots) if r]
    deny_roots = [r for r in (norm(p) for p in deny_roots) if r]
    allow_roots = [r for r in (norm(p) for p in allow_roots) if r]
    o_write = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, 'O_TRUNC', 0)

    def readable(p):
        if under(p, write_roots) or under(p, allow_roots):
            return True
        return not under(p, deny_roots) and under(p, read_roots)

    def hook(event, args):
        if event == 'open':
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if isinstance(path, int):
                if allow_fd_open:
                    return
                raise PermissionError(f'{what} may not open file descriptors')
            p = norm(path)
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or \
                      (isinstance(flags, int) and bool(flags & o_write))
            if p is not None and (under(p, write_roots) if writing else readable(p)):
                return
            raise PermissionError(f'{what} may not {"write" if writing else "read"} {os.fsdecode(path)!r}')
        if event in ('os.listdir', 'os.scandir'):
            p = norm(args[0] if args and args[0] is not None else '.')
            if p is None or (under(p, deny_roots) and not (under(p, write_roots) or under(p, allow_roots))):
                raise PermissionError(f'{what} may not list this directory')
            return
        if event in WRITE_EVENTS:
            idx = WRITE_EVENTS[event]
            targets = [args[i] for i in ((0, 1) if event in ('os.rename', 'os.replace') else (idx,))
                       if i is not None and i < len(args)]
            for t in targets:
                p = norm(t) if not isinstance(t, int) else None
                if p is None or not under(p, write_roots):
                    raise PermissionError(f'{what} may not modify {t!r}')
            return
        if event == 'ctypes.dlopen':                    # native libraries only from the Python installation
            name = args[0] if args else None
            p = norm(name) if isinstance(name, (str, bytes)) and os.path.isabs(os.fsdecode(name)) else None
            if p is None or not under(p, read_roots) or under(p, deny_roots):
                raise PermissionError(f'{what} may not load native library {name!r}')
            return
        if event == 'ctypes.dlsym':                     # symbols only from those libraries (not libc, not python)
            lib = getattr(args[0], '_name', None) if args else None
            sym = args[1] if len(args) > 1 else None
            if lib is None and isinstance(sym, str) and sym.startswith('Py'):
                return                                  # the public Python C API (ctypes.pythonapi), used by lightgbm
            p = norm(lib) if isinstance(lib, (str, bytes)) and os.path.isabs(os.fsdecode(lib)) else None
            if p is None or not under(p, read_roots):
                raise PermissionError(f'{what} may not look up native symbol {args[1] if len(args) > 1 else "?"!r}')
            return
        if event == 'import':
            name = args[0] if args else ''
            if name in blocked_modules or any(name.startswith(m + '.') for m in blocked_modules):
                raise ImportError(f'module {name!r} is not available to {what}')
            return
        if event.startswith(BLOCKED_EVENT_PREFIXES):
            raise PermissionError(f'{what} may not perform {event}')

    sys.addaudithook(hook)
    for name in [n for n in list(sys.modules) if n in blocked_modules or n.split('.')[0] in blocked_modules]:
        sys.modules.pop(name, None)            # preloaded by libraries: a fresh `import` must now pass the hook


# ------------------------------------------------------------------------------------------------ agent sandboxes
def install_sandbox_guard(policy: dict) -> dict:
    """Called first thing in every agent sandbox script (desk/lab/sandbox.py writes the call into its preamble).

    policy: read_roots (code + Python), allow_roots (the dev-only data views), write_roots (the agent's workspace),
    deny_roots (results/, the store, the ledger, raw data, .env). Returns which layers are active."""
    read_roots, allow_roots = list(policy.get('read_roots', ())), list(policy.get('allow_roots', ()))
    write_roots, deny_roots = list(policy.get('write_roots', ())), list(policy.get('deny_roots', ()))
    import ctypes                       # noqa: F401  load ctypes.pythonapi before the hook (libraries need ctypes)
    import subprocess                   # noqa: F401  polars and scipy import it; it binds fork_exec at import
    try:
        import numpy                    # noqa: F401  polars probes the CPU through libc at import time
        import polars                   # noqa: F401
    except ImportError:
        pass
    try:                                # numba imports colorama, which on Windows binds kernel32 by bare name at import
        import numba                    # noqa: F401  (the desk.research engines need it anyway)
    except Exception:
        pass
    try:
        import _posixsubprocess         # its fork_exec starts programs without raising an audit event
        del _posixsubprocess.fork_exec
    except (ImportError, AttributeError):
        pass
    landlock = install_landlock(read_roots + allow_roots + [os.devnull], write_roots)
    seccomp = install_seccomp()
    install_audit_hook(read_roots, write_roots=write_roots, deny_roots=deny_roots, allow_roots=allow_roots,
                       allow_fd_open=True, what='sandbox code')
    global ACTIVE
    ACTIVE = 'sandbox'
    return {'landlock': landlock, 'seccomp': seccomp, 'audit_hook': True}
