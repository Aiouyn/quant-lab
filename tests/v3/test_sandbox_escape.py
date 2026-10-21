"""Escape payloads against the OS jail (desk/lab/jail.py). Every payload must be blocked."""
import asyncio
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from desk.config import ROOT, SETTINGS
from desk.lab import jail
from desk.lab.sandbox import run_python

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')


def run(code, ws, timeout_s=60, mem_mb=None):
    return asyncio.run(run_python(code, ws, SETTINGS, timeout_s=timeout_s, mem_mb=mem_mb))


def test_valid_strategy_code_runs(tmp_path):
    r = run('import numpy as np, polars as pl\nx = pl.DataFrame({"a": np.arange(10)})\nprint("SUM", int(x["a"].sum()))', tmp_path)
    assert r['status'] == 'ok', r
    assert 'SUM 45' in r['stdout']


def test_runs_as_non_root(tmp_path):
    r = run('import os\nprint("UID", os.getuid(), os.getgid())', tmp_path)
    assert 'UID 65534 65534' in r['stdout'], r


def test_network_connect_blocked(tmp_path):
    r = run('import socket\ntry:\n    socket.create_connection(("1.1.1.1", 80), timeout=3); print("ESCAPED")\n'
            'except OSError as e:\n    print("BLOCKED", e)', tmp_path)
    assert 'BLOCKED' in r['stdout'] and 'ESCAPED' not in r['stdout'], r


def test_write_outside_tmp_blocked(tmp_path):
    target = ROOT / 'desk' / 'PWNED.txt'
    r = run(f'for p in [{str(target)!r}, "/usr/pwned", "/etc/pwned"]:\n'
            '    try:\n        open(p, "w").write("x"); print("ESCAPED", p)\n'
            '    except OSError as e:\n        print("BLOCKED", p)\n'
            'open("/tmp/ok.txt", "w").write("ok"); print("TMP_OK")', tmp_path)
    assert 'ESCAPED' not in r['stdout'] and r['stdout'].count('BLOCKED') == 3 and 'TMP_OK' in r['stdout'], r
    assert not target.exists()


def test_home_and_dotenv_unreadable(tmp_path):
    home = Path.home()
    r = run(f'import os\nfor p in [{str(home)!r}, {str(ROOT / ".env")!r}, {str(ROOT / ".env.example")!r}, '
            f'{str(SETTINGS.home)!r}, "/root", "/etc/shadow"]:\n'
            '    try:\n        os.listdir(p) if os.path.isdir(p) and os.listdir(p) else open(p).read(1); print("ESCAPED", p)\n'
            '    except OSError:\n        print("BLOCKED", p)\n'
            'print("ENV", sorted(k for k in os.environ if "KEY" in k or "TOKEN" in k or "SECRET" in k))', tmp_path)
    assert 'ESCAPED' not in r['stdout'] and r['stdout'].count('BLOCKED') == 6, r
    assert 'ENV []' in r['stdout'], r


def test_fork_bomb_hits_process_limit(tmp_path):
    r = run('import os\nn = 0\ntry:\n    for _ in range(10000):\n        if os.fork() == 0:\n            import time; time.sleep(30); os._exit(0)\n'
            '        n += 1\nexcept OSError as e:\n    print("BLOCKED", n, e)\n', tmp_path, timeout_s=60)
    assert 'BLOCKED' in r['stdout'], r
    assert int(r['stdout'].split()[1]) <= jail.DEFAULT_LIMITS['nproc'], r


def test_memory_blowup_blocked(tmp_path):
    r = run('try:\n    x = bytearray(20 * 1024**3); print("ESCAPED")\nexcept MemoryError:\n    print("BLOCKED")', tmp_path, mem_mb=2000)
    assert 'BLOCKED' in r['stdout'] and 'ESCAPED' not in r['stdout'], r


def test_subclasses_trick_gains_nothing(tmp_path):
    # classic builtins escape: reach os/subprocess via object.__subclasses__(); the jail makes it pointless
    r = run('import socket\nsub = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == "Popen"]\n'
            'os_ = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == "_wrap_close"][0].__init__.__globals__\n'
            'print("RC", os_["system"]("cat /etc/shadow ~/.ssh/id_rsa 2>/dev/null; touch /usr/x 2>/dev/null"))\n'
            'try:\n    os_["listdir"](os_["environ"].get("HOME_OUTSIDE", "/home")); print("LS", os_["listdir"]("/home"))\n'
            'except OSError:\n    print("BLOCKED home")', tmp_path)
    assert 'BLOCKED home' in r['stdout'] or "LS []" in r['stdout'], r
    assert 'RC 0' not in r['stdout'], r


def test_subprocess_and_os_system_contained(tmp_path):
    r = run('import os, subprocess\n'
            'print("SYS", os.system("curl -s --max-time 3 http://1.1.1.1 >/dev/null 2>&1 || exit 7"))\n'
            'out = subprocess.run(["sh", "-c", "id -u; ls /home 2>&1; cat /proc/1/cmdline | tr \'\\\\0\' \' \'"], capture_output=True, text=True).stdout\n'
            'print("OUT", out.replace(chr(10), "|"))', tmp_path)
    out = r['stdout']
    assert 'SYS 0' not in out, r                          # no network from children either
    assert 'OUT 65534|' in out, r                         # children stay non-root
    assert 'No such file' in out, r                       # host /home not mounted
    assert 'bwrap' in out or 'jail_boot' in out, r        # pid 1 is the jail, not the host init


def test_fail_closed_without_bwrap(tmp_path, monkeypatch):
    monkeypatch.setattr(jail, '_probe', lambda: (None, 'bubblewrap (bwrap) is not installed'))
    monkeypatch.delenv(jail.UNSAFE_ENV, raising=False)
    with pytest.raises(jail.JailUnavailable):
        run('print("hi")', tmp_path)
    from desk.lab.factor_jail import run as run_factor
    with pytest.raises(jail.JailUnavailable):
        run_factor('def factor(panel, params):\n    return panel["x"]', [pl.DataFrame({'x': [1.0]})])


def test_factor_module_runs_in_jail():
    from desk.lab.panel_factor import load_factor, values
    code = ('import os, socket\nPARAMS = {"k": 2}\nDESK = "perps"\n'
            'def factor(panel, params):\n'
            '    try:\n        socket.create_connection(("1.1.1.1", 80), timeout=2); net = 1.0\n'
            '    except OSError:\n        net = 0.0\n'
            '    return panel["x"] * params["k"] + net + (os.getuid() != 65534)\n')
    mod = load_factor(code)
    assert mod.PARAMS == {'k': 2} and mod.DESK == 'perps'
    v = values(mod, pl.DataFrame({'x': [1.0, 2.0, 3.0]}))
    assert np.allclose(v, [2.0, 4.0, 6.0])                # no network and non-root inside the factor


def test_combo_factor_matrix_runs_in_jail():
    from desk.lib.combo import factor_matrix
    code = 'PARAMS = {}\ndef factor(df, params):\n    open("/usr/pwn", "w")\n    return df["x"]'
    with pytest.raises(Exception):
        factor_matrix(pl.DataFrame({'x': [1.0, 2.0]}), [code])
    ok = 'PARAMS = {}\ndef factor(df, params):\n    return df["x"] * 2'
    assert factor_matrix(pl.DataFrame({'x': [1.0, 2.0, 3.0]}), [ok]).shape == (3, 1)


def test_full_store_and_parent_paths_invisible(tmp_path):
    """QL Validator repro: Path(DESK_STORE_ROOT).parent.parent / 'funding_hl.parquet' must not be readable."""
    store = tmp_path / 'store'
    dev = store / 'views' / 'dev'
    dev.mkdir(parents=True)
    (dev / 'dev_ok.txt').write_text('dev')
    (store / 'funding_hl.parquet').write_bytes(b'FULLSTORE')
    (store / 'views' / 'val.parquet').write_bytes(b'VALIDATION')
    import dataclasses
    s = dataclasses.replace(SETTINGS, store_dir=store)
    code = ('import os\nfrom pathlib import Path\nr = Path(os.environ["DESK_STORE_ROOT"])\n'
            'print("DEV", (r / "dev_ok.txt").read_text())\n'
            'for p in [r.parent.parent / "funding_hl.parquet", r / ".." / ".." / "funding_hl.parquet", '
            'r.parent / "val.parquet"] + list(r.parents):\n'
            '    try:\n'
            '        if p.is_dir():\n            items = os.listdir(p)\n'
            '            assert not any(i in ("funding_hl.parquet", "val.parquet") for i in items)\n'
            '            print("NOLEAK", p)\n'
            '        else:\n            open(p, "rb").read(); print("ESCAPED", p)\n'
            '    except (OSError, AssertionError):\n        print("BLOCKED", p)\n')
    r = asyncio.run(run_python(code, tmp_path / 'ws', s, timeout_s=60))
    out = r['stdout']
    assert 'DEV dev' in out, r
    assert 'ESCAPED' not in out and 'FULLSTORE' not in out and 'VALIDATION' not in out, r
    assert out.count('BLOCKED') >= 3, r


def test_dev_view_0600_and_hardlinks_readable_but_readonly(tmp_path):
    """PR #7 writes view files 0600 and dev bar files are hard links into the store: the jail uid must still read
    them (bwrap maps the host uid to 65534) and the read-only bind must stop writes through the hard link."""
    store = tmp_path / 'store'
    dev = store / 'views' / 'dev'
    dev.mkdir(parents=True)
    pl.DataFrame({'ts': [1, 2], 'funding': [0.1, 0.2]}).write_parquet(dev / 'funding_hl.parquet')
    (dev / 'splits.json').write_text('{"dev": ["2024-01-01", "2024-06-30"]}')
    (store / 'bars.parquet').write_bytes(b'BARS')
    os.link(store / 'bars.parquet', dev / 'bars.parquet')
    for f in dev.iterdir():
        f.chmod(0o600)
    import dataclasses
    s = dataclasses.replace(SETTINGS, store_dir=store)
    code = ('import os, polars as pl\nfrom pathlib import Path\nr = Path(os.environ["DESK_STORE_ROOT"])\n'
            'print("FUNDING", pl.read_parquet(r / "funding_hl.parquet")["funding"].sum())\n'
            'print("SPLITS", (r / "splits.json").read_text()[:7])\n'
            'try:\n    open(r / "bars.parquet", "ab").write(b"X"); print("ESCAPED")\nexcept OSError:\n    print("RO")\n')
    r = asyncio.run(run_python(code, tmp_path / 'ws', s, timeout_s=60))
    assert 'FUNDING 0.3' in r['stdout'] and 'SPLITS {"dev"' in r['stdout'] and 'RO' in r['stdout'], r
    assert (store / 'bars.parquet').read_bytes() == b'BARS'
