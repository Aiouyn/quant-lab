"""Red-team suite for the sealed evaluator (Phase 1a).

Every test here is an attack a strategy, a factor or an agent could try. Each one must be caught. If you add a new
way in, add the attack here first.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from desk.lab import guard
from desk.lab import stats
from desk.lab.jail import JailError, JailedModel, call, load_jailed
from desk.lab.strategy import causality_test, load_strategy, run_signal, structural_hash, StrategyError

LINUX = sys.platform.startswith('linux')
REPO = Path(__file__).resolve().parents[2]


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({'ts': np.arange(n, dtype=np.int64) * 60_000, 'mid': 100 + np.cumsum(rng.normal(size=n))})


def attack(body: str, df=None):
    """Run `body` inside signal() in a jail; return the error message (or None if it ran)."""
    code = 'def signal(df, p):\n' + ''.join(f'    {line}\n' for line in body.strip().splitlines()) + \
           "    return df['mid'] * 0\n"
    try:
        call('signal', code, df if df is not None else frame(10), {})
        return None
    except JailError as e:
        return str(e)


# ------------------------------------------------------------------------------------------- the score is out of reach
def test_strategy_cannot_patch_the_scorer():
    real_dsr = stats.dsr
    err = attack("import desk.lab.stats as s\ns.dsr = lambda *a, **k: 0.999")
    assert err and 'not available' in err
    err = attack("import sys\nsys.modules['desk.lab.stats'].dsr = lambda *a, **k: 0.999")
    assert err                                       # the evaluator's modules do not exist in the jail
    assert stats.dsr is real_dsr


def test_strategy_code_never_runs_in_the_evaluator_process():
    marker = 'import builtins\nbuiltins.PWNED_EVALUATOR = True\n'
    mod = load_strategy(marker + "RESOLUTION='1m'\ndef signal(df, p):\n    return df['mid'] * 0\n")
    run_signal(mod, frame(50))
    import builtins
    assert not hasattr(builtins, 'PWNED_EVALUATOR')


def test_jail_returns_data_only():
    mod = load_strategy("RESOLUTION='1m'\nPARAMS={'n': 3}\ndef signal(df, p):\n    return df['mid'].diff().fill_null(0)\n")
    assert mod.PARAMS == {'n': 3} and mod.DESK == 'l2' and not mod.FITTED
    s = run_signal(mod, frame(50))
    assert s.dtype == np.float64 and len(s) == 50 and np.all(np.abs(s) <= 1)


def test_fitted_models_stay_opaque_bytes():
    code = ("RESOLUTION='1m'\nTRAIN_DAYS=3\ndef fit(df, p):\n    return {'mu': float(df['mid'].mean())}\n"
            "def signal(df, p, m):\n    return (df['mid'] > m['mu']).cast(pl.Float64) * 2 - 1\n")
    mod = load_strategy(code)
    model = mod.fit(frame(100), {})
    assert isinstance(model, JailedModel) and isinstance(model.blob, bytes)
    assert len(run_signal(mod, frame(30), model=model)) == 30
    with pytest.raises(StrategyError):
        mod.signal(frame(5), {}, {'mu': 1.0})        # a raw (host-made) model object is refused


# ------------------------------------------------------------------------------------------------ no files, no network
@pytest.mark.parametrize('body', [
    "open('/etc/hostname').read()",
    "import pathlib\npathlib.Path(%r).read_text()" % str(REPO / 'README.md'),
    "import io\nio.FileIO(%r).read()" % str(REPO / 'desk' / 'config.py'),
    "import os\nos.open(%r, 0)" % str(REPO / 'desk' / 'config.py'),
    "np.load('/tmp/x.npy')",
    "np.fromfile(%r, dtype='u1')" % str(REPO / 'README.md'),
    "pl.read_parquet('/tmp/x.parquet')",
    "pl.scan_csv(%r).collect()" % str(REPO / 'README.md'),
    "type(df._df).read_ipc",
    "df.write_parquet('/tmp/w.parquet')",
    "open('/tmp/jail_write_test.txt', 'w').write('x')",
    "open(0, 'rb').read()",
    "import os\nos.listdir(%r)" % str(REPO / 'results'),
])
def test_strategy_cannot_touch_files(body):
    assert attack(body), body


@pytest.mark.parametrize('body', [
    "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)",
    "import subprocess\nsubprocess.run(['true'])",
    "import os\nos.system('true')",
    "import os\nos.popen('true').read()",
    "import os\nos.fork()",
    "import ctypes\nctypes.CDLL(None).system(b'true')",
    "import ctypes\nctypes.CDLL('libc.so.6').system(b'true')",
    "import ctypes\nctypes.pythonapi.system(b'true')",
    "import ctypes\nctypes.pythonapi._PySys_ClearAuditHooks()",
])
def test_strategy_cannot_start_programs_or_connect(body):
    assert attack(body), body


@pytest.mark.skipif(not LINUX, reason='seccomp is Linux-only')
def test_unaudited_fork_exec_is_stopped_by_the_kernel(tmp_path):
    target = tmp_path / 'pwned'
    err = attack(f"""import importlib, sys, os
sys.modules.pop('_posixsubprocess', None)
ps = importlib.import_module('_posixsubprocess')
r, w = os.pipe()
ps.fork_exec([b'/bin/sh', b'-c', b'touch {target}'], [b'/bin/sh'], True, (), None, None, -1, -1, -1, -1, -1, -1, r, w, True, False, 0, None, None, None, -1, None, False)""")
    assert err and not target.exists()


def test_strategy_sees_no_secrets():
    err = attack("import os\nraise RuntimeError(sorted(os.environ))")
    for secret in ('OPENCODE_API_KEY', 'OPENAI_API_KEY', 'MIMO_API_KEY', 'DESK_STORE_ROOT', 'RESULTS_DIR'):
        assert secret not in err
    home = attack("import os\nraise RuntimeError(os.environ.get('HOME', '') + '|' + os.environ.get('USERPROFILE', ''))")
    assert 'jail_' in home and str(Path.home()) + '|' not in home     # HOME is the empty jail directory, not the user's


def test_print_does_not_corrupt_results():
    mod = load_strategy("RESOLUTION='1m'\ndef signal(df, p):\n    print('noise' * 1000)\n    return df['mid'] * 0 + 0.5\n")
    assert np.allclose(run_signal(mod, frame(20)), 0.5)


def test_runaway_and_crashing_strategies_fail_cleanly():
    with pytest.raises(JailError, match='timed out'):
        call('signal', "def signal(df, p):\n    while True: pass\n", frame(5), {}, timeout_s=3)
    with pytest.raises(JailError):
        call('signal', "def signal(df, p):\n    import os\n    os._exit(0)\n", frame(5), {})
    with pytest.raises(JailError, match='length'):
        load_strategy("RESOLUTION='1m'\ndef signal(df, p):\n    return np.zeros(3)\n").signal(frame(5), {})


@pytest.mark.skipif(not (LINUX and guard.landlock_abi()), reason='kernel guards need Linux with Landlock')
def test_jail_reports_kernel_guards():
    resp, _ = call('meta', "def signal(df, p):\n    return df['mid'] * 0\n")
    assert resp['kernel_guard'] == 'landlock+seccomp'


# ------------------------------------------------------------------------------------ causality cannot be gamed by memory
def test_strategy_cannot_carry_the_full_sample_into_the_prefix_test():
    # Peeks one row ahead, and tries to hide it: the full-sample call stores its output in a process-wide place and
    # later (shorter) prefix calls replay it. In one process this passed the prefix test; in fresh jails it cannot.
    code = """RESOLUTION='1m'
def signal(df, p):
    import builtins
    cache = getattr(builtins, '_stash', None)
    if cache is not None and len(cache) > len(df):
        return cache[:len(df)]
    out = (df['mid'].shift(-1) - df['mid']).fill_null(0).sign().to_numpy()
    builtins._stash = out
    return out
"""
    res = causality_test(load_strategy(code), frame(600))
    assert not res['ok']


# -------------------------------------------------------------------------------- libraries strategies are allowed to use
@pytest.mark.parametrize('lib,fit', [
    ('scipy', "import scipy.stats as st\n    return float(st.norm.ppf(0.5))"),
    ('sklearn', "from sklearn.linear_model import Ridge\n    return Ridge().fit(df[['mid']].to_numpy(), df['mid'].to_numpy())"),
    ('lightgbm', "import lightgbm as lgb\n    return lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(df[['mid']].to_numpy(), df['mid'].to_numpy())"),
    ('contrib', "from desk.lib.contrib.har_sigma import har_sigma\n    return 0.0"),
])
def test_allowed_libraries_still_work_in_the_jail(lib, fit):
    pytest.importorskip(lib if lib != 'contrib' else 'numpy')
    code = (f"RESOLUTION='1m'\ndef fit(df, p):\n    {fit}\n"
            "def signal(df, p, m):\n    return df['mid'] * 0\n")
    mod = load_strategy(code)
    assert len(run_signal(mod, frame(60), model=mod.fit(frame(60), {}))) == 60


# ------------------------------------------------------------------------------------------- factor code is jailed too
def test_factor_code_is_jailed():
    from desk.lab.panel_factor import load_factor
    f = load_factor("DESK='perps'\nPARAMS={'k': 2}\ndef factor(panel, p):\n    return panel['mid'] * p['k']\n")
    assert np.allclose(f.factor(frame(10), {'k': 2}), frame(10)['mid'].to_numpy() * 2)
    with pytest.raises(JailError):
        load_factor("import desk.lab.stats\ndef factor(panel, p):\n    return panel['mid']\n")


# ----------------------------------------------------------------------------------- the holdout is one-shot per logic
def test_structural_hash_ignores_cosmetics_but_not_logic():
    a = "import numpy as np\nPARAMS={'n': 5}\ndef signal(df, p):\n    x = df['mid']\n    return (x - x.rolling_mean(p['n'])).sign()\n"
    b = ('"""renamed and reformatted"""\n# comment\nimport numpy as N\nPARAMS = {"n": 5}\n\n'
         'def signal(frame, params):\n    """doc"""\n    price = frame["mid"]  # px\n'
         '    return (price - price.rolling_mean(params["n"])).sign()\n')
    assert structural_hash(a) == structural_hash(b)
    assert structural_hash(a) != structural_hash(a.replace('5', '6'))
    assert structural_hash(a) != structural_hash(a.replace("'mid'", "'mark'"))


def test_holdout_refuses_a_cosmetic_resubmission(tmp_path):
    from desk.company.ledger import Ledger
    L = Ledger(tmp_path / 'desk.sqlite3')
    a = "PARAMS={'n': 5}\ndef signal(df, p):\n    return df['mid'] * 0\n"
    b = "# v2, totally new\nPARAMS = {'n': 5}\ndef signal(frame, params):\n    return frame['mid'] * 0\n"
    L.x('INSERT INTO holdout_uses VALUES(?,?,?,?)', (structural_hash(a), 'S-1', 0.0, 0))
    used = L.q('SELECT strategy FROM holdout_uses WHERE code_hash IN (?, ?)', (structural_hash(b), 'raw'))
    assert used and used[0]['strategy'] == 'S-1'


# --------------------------------------------------------------------------------------- agents cannot loosen the costs
def _tools(tmp_path):
    from desk.company.ledger import Ledger
    from desk.company.tools import Toolbox
    L = Ledger(tmp_path / 'desk.sqlite3')

    class Org:
        def cost_params(self):
            r = L.q('SELECT params FROM cost_models ORDER BY id DESC LIMIT 1')
            return json.loads(r[0]['params']) if r else {'impact_mult': 1.0, 'slippage_bps': 0.5, 'latency_s': 1}

    class Agent:
        name, pod, role, workspace = 'exec-1', 'macro', 'execution', tmp_path

    t = Toolbox.__new__(Toolbox)
    t.L, t.org, t.agent = L, Org(), Agent()
    return t, L


def test_cost_model_can_only_tighten_without_a_human(tmp_path):
    t, L = _tools(tmp_path)
    out = asyncio.run(t.t_set_cost_model('more conservative', impact_mult=1.5))
    assert out['status'].startswith('applied') and t.org.cost_params()['impact_mult'] == 1.5
    out = asyncio.run(t.t_set_cost_model('fills look better than modelled', impact_mult=0.5, slippage_bps=0.0))
    assert out['status'] == 'pending_human_approval'
    assert t.org.cost_params()['impact_mult'] == 1.5           # unchanged until approved
    from desk.cli import main
    import desk.config as cfg
    s = dataclasses.replace(cfg.SETTINGS, home=tmp_path)
    old = cfg.SETTINGS
    cfg.SETTINGS = s
    try:
        assert main(['costs', 'approve', str(out['proposal'])]) == 0
    finally:
        cfg.SETTINGS = old
    assert t.org.cost_params()['impact_mult'] == 0.5


# ------------------------------------------------------------------------------------------------- the kernel is sealed
def test_kernel_manifest_matches_the_code():
    from desk.kernel import verify
    res = verify()
    assert res['ok'], res.get('error')


def test_kernel_verify_detects_an_edit(tmp_path, monkeypatch):
    import desk.kernel as k
    fake = tmp_path / 'desk'
    (fake / 'lab').mkdir(parents=True)
    (fake / 'lab' / 'stats.py').write_text('def dsr(*a):\n    return 0.1\n')
    (fake / 'kernel.py').write_text('# kernel\n')
    monkeypatch.setattr(k, 'PKG', fake)
    monkeypatch.setattr(k, 'ROOT', tmp_path)
    monkeypatch.setattr(k, 'MANIFEST', fake / 'KERNEL_MANIFEST.json')
    assert not k.verify()['ok']                                # no manifest yet
    k.seal()
    assert k.verify()['ok']
    (fake / 'lab' / 'stats.py').write_text('def dsr(*a):\n    return 0.999\n')
    res = k.verify()
    assert not res['ok'] and res['changed'] == ['desk/lab/stats.py']
    (fake / 'lab' / 'stats.py').write_bytes(b'def dsr(*a):\r\n    return 0.999\r\n')
    k.seal()
    (fake / 'lab' / 'stats.py').write_bytes(b'def dsr(*a):\n    return 0.999\n')
    assert k.verify()['ok']                                    # CRLF vs LF checkout is not a change


# ----------------------------------------------------------------------------------------------- the agent sandbox
def _sandbox(tmp_path):
    from desk.config import SETTINGS
    home = tmp_path / 'home'
    s = dataclasses.replace(SETTINGS, home=home, store_dir=None).ensure()
    (s.store / 'views' / 'dev').mkdir(parents=True)
    (s.store / 'views' / 'val').mkdir(parents=True)
    (s.store / 'views' / 'dev' / 'ok.txt').write_text('dev data')
    (s.store / 'views' / 'val' / 'secret.txt').write_text('VAL DATA')
    (home / 'desk.sqlite3').write_text('ledger')
    ws = s.workspaces / 'agent1'
    ws.mkdir(parents=True)
    other = s.workspaces / 'agent2'
    other.mkdir(parents=True)
    (other / 'notes.md').write_text('other agent')
    return s, home, ws, other


def test_sandbox_sees_dev_data_and_its_workspace_only(tmp_path):
    from desk.lab.sandbox import run_python
    s, home, ws, other = _sandbox(tmp_path)
    code = f'''
import os
def t(name, f):
    try:
        f(); print('ALLOWED', name)
    except Exception as e:
        print('BLOCKED', name, type(e).__name__)
dev = os.environ['DESK_STORE_ROOT']
t('read_dev', lambda: open(os.path.join(dev, 'ok.txt')).read())
t('write_ws', lambda: open('notes.txt', 'w').write('x'))
t('read_code', lambda: open({str(REPO / 'desk' / 'lab' / 'stats.py')!r}).read())
t('read_val', lambda: open({str(s.store / 'views' / 'val' / 'secret.txt')!r}).read())
t('read_ledger', lambda: open({str(home / 'desk.sqlite3')!r}).read())
t('read_other_workspace', lambda: open({str(other / 'notes.md')!r}).read())
t('write_repo', lambda: open({str(REPO / 'desk' / 'lab' / 'stats.py')!r}, 'a').write('#'))
t('socket', lambda: __import__('socket').create_connection(('1.1.1.1', 80), timeout=2))
t('subprocess', lambda: __import__('subprocess').run(['true']))
'''
    res = asyncio.run(run_python(code, ws, s, timeout_s=120))
    assert res['status'] == 'ok', res['stderr']
    got = dict(line.split()[1::-1] for line in res['stdout'].splitlines() if line.startswith(('ALLOWED', 'BLOCKED')))
    for name in ('read_dev', 'write_ws', 'read_code'):
        assert got[name] == 'ALLOWED', (name, res['stdout'])
    for name in ('read_val', 'read_ledger', 'read_other_workspace', 'write_repo', 'socket', 'subprocess'):
        assert got[name] == 'BLOCKED', (name, res['stdout'])


@pytest.mark.skipif(not (LINUX and guard.landlock_abi()), reason='native-reader blocking needs Linux with Landlock')
def test_sandbox_blocks_native_readers_on_linux(tmp_path):
    from desk.lab.sandbox import run_python
    s, home, ws, other = _sandbox(tmp_path)
    code = f"import polars as pl\npl.read_csv({str(s.store / 'views' / 'val' / 'secret.txt')!r})\n"
    res = asyncio.run(run_python(code, ws, s, timeout_s=120))
    assert res['status'] != 'ok' and 'denied' in res['stderr'].lower()
