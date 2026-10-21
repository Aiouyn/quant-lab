"""run_job (harness) executes agent strategy code inside the OS jail: escapes are blocked, results are unchanged."""
import asyncio
import dataclasses
import json
import os
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from desk.config import ROOT, SETTINGS
from desk.lab import harness, jail
from desk.lab.macro import SPLITS
from desk.lab.sandbox import run_job

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')

CLEAN = ("DESK = 'macro'\nPARAMS = {'n': 5}\n"
         "def signal(panel, params):\n"
         "    m = (pl.col('adjclose') / pl.col('adjclose').shift(params['n']).over('symbol') - 1).sign().fill_null(0)\n"
         "    return panel.select(m.alias('w'))['w'].to_numpy()\n")


def _settings(tmp_path):
    store = tmp_path / 'store'
    (store / 'macro').mkdir(parents=True)
    (store / 'macro' / 'splits.json').write_text(json.dumps(SPLITS))
    rng = np.random.default_rng(0)
    rows, d0 = [], date(2013, 1, 1)
    px = {'AAA': 100.0, 'BBB': 50.0, 'CCC': 20.0}
    for i in range(400):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for s in px:
            px[s] *= float(np.exp(rng.normal(0.0003, 0.01)))
            rows.append({'date': d, 'symbol': s, 'adjclose': px[s], 'close': px[s], 'cls': 'equity_us', 'tradable': 1,
                         'cost_bps': 2.0})
    pl.DataFrame(rows).write_parquet(store / 'macro' / 'panel.parquet')
    (store / 'funding_hl.parquet').write_bytes(b'FULLSTORE')
    home = tmp_path / 'home'
    (home / 'artifacts').mkdir(parents=True)
    return dataclasses.replace(SETTINGS, home=home, store_dir=store)


def _job(code, **kw):
    return dict({'type': 'strategy_eval', 'code': code, 'split': 'dev', 'stress': True}, **kw)


def test_run_job_results_match_in_process(tmp_path):
    s = _settings(tmp_path)
    jailed = asyncio.run(run_job(_job(CLEAN, robustness=True), s, timeout_s=300))
    assert 'error' not in jailed, jailed
    direct = harness.strategy_eval(dict(_job(CLEAN, robustness=True), _home=str(s.home), _store=str(s.store)))
    direct = json.loads(json.dumps(direct, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    for k in ('seconds',):
        jailed.pop(k, None); direct.pop(k, None)
    assert jailed == direct


def _probe(tmp_path, body):
    """Run `body` (inside signal) through run_job; it reports via the raised error message."""
    code = ("DESK = 'macro'\nPARAMS = {}\n"
            "def signal(panel, params):\n" + ''.join('    ' + l + '\n' for l in body.splitlines()) +
            "    raise RuntimeError('PROBE ' + repr(r))\n")
    out = asyncio.run(run_job(_job(code, stress=False), _settings(tmp_path), timeout_s=120))
    return out.get('error', json.dumps(out))


def test_run_job_no_network(tmp_path):
    out = _probe(tmp_path, "from socket import create_connection\ntry:\n    create_connection(('1.1.1.1', 80), timeout=3); r = 'ESCAPED'\n"
                           "except OSError as e:\n    r = 'BLOCKED'")
    assert 'BLOCKED' in out and 'ESCAPED' not in out, out


def test_run_job_os_system_and_non_root(tmp_path):
    out = _probe(tmp_path, "from os import system, getuid\nr = (getuid(), system('touch /usr/x 2>/dev/null'), "
                           "system('curl -s --max-time 3 http://1.1.1.1 >/dev/null 2>&1'))")
    assert 'PROBE (65534, ' in out and ', 0, ' not in out and ', 0)' not in out, out


def test_run_job_store_readonly_and_secrets_hidden(tmp_path):
    out = _probe(tmp_path, "from pathlib import Path\nfrom os import environ\nr = []\n"
                           f"for p in [{str(ROOT / '.env')!r}, {str(__import__('pathlib').Path.home())!r} + '/.ssh/id_rsa', '/etc/shadow']:\n"
                           "    try:\n        Path(p).read_bytes(); r.append('ESCAPED')\n    except OSError:\n        r.append('BLOCKED')\n"
                           "st = Path(environ.get('X', '')) or None\n"
                           "import desk.config as c\n"
                           "try:\n    (c.SETTINGS.store / 'macro' / 'panel.parquet').write_bytes(b'x'); r.append('ESCAPED_W')\n"
                           "except OSError:\n    r.append('RO')\n"
                           "r.append(sorted(k for k in environ if 'KEY' in k or 'TOKEN' in k))")
    assert 'ESCAPED' not in out and out.count('BLOCKED') == 3 and 'RO' in out and '[]' in out, out


def test_run_job_fork_bomb_limited(tmp_path):
    out = _probe(tmp_path, "from os import fork, _exit\nimport time\nn = 0\ntry:\n    for _ in range(5000):\n"
                           "        if fork() == 0:\n            time.sleep(20); _exit(0)\n        n += 1\n"
                           "    r = 'ESCAPED'\nexcept OSError:\n    r = n")
    assert 'ESCAPED' not in out and 'PROBE ' in out, out
    import re
    assert int(re.search(r'PROBE (\d+)', out).group(1)) <= jail.DEFAULT_LIMITS['nproc'], out


def test_run_job_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(jail, '_probe', lambda: (None, 'bubblewrap (bwrap) is not installed'))
    monkeypatch.delenv(jail.UNSAFE_ENV, raising=False)
    with pytest.raises(jail.JailUnavailable):
        asyncio.run(run_job(_job(CLEAN), _settings(tmp_path)))


def test_run_job_cannot_read_splits_outside_allowed(tmp_path):
    """With PR #16's per-split views only job['allowed'] data is mounted: val sentinels are unreachable to a dev job,
    through the harness store, the view root, or the original store path."""
    s = _settings(tmp_path)
    p = pl.read_parquet(s.store / 'macro' / 'panel.parquet')
    val = p.with_columns(pl.col('date') + pl.duration(days=365 * 3), pl.lit(777.0).alias('adjclose'), pl.lit(777.0).alias('close'))
    pl.concat([p, val]).write_parquet(s.store / 'macro' / 'panel.parquet')     # 2016-2017 rows = val split
    body = ("from pathlib import Path\nimport polars as _p\nimport desk.config as c\nseen = set()\n"
            f"for top in [Path({str(s.store)!r}), Path({str(tmp_path)!r})]:\n"
            "    for f in top.rglob('*.parquet'):\n"
            "        try:\n            d = _p.read_parquet(f)\n        except Exception:\n            continue\n"
            "        for col in d.columns:\n"
            "            if d[col].dtype.is_numeric():\n                seen |= {float(v) for v in d[col].drop_nulls()} & {777.0}\n"
            "seen |= {float(v) for v in panel['adjclose']} & {777.0}\n"
            "r = sorted(seen)")
    code = ("DESK = 'macro'\nPARAMS = {}\n"
            "def signal(panel, params):\n" + ''.join('    ' + l + '\n' for l in body.splitlines()) +
            "    raise RuntimeError('PROBE ' + repr(r))\n")
    out = asyncio.run(run_job(_job(code, stress=False, allowed=['dev']), s, timeout_s=120)).get('error', '')
    assert 'PROBE []' in out, out
    # sanity: the same probe in-process (no jail, full store) does see the val sentinel
    direct = harness.main.__globals__['strategy_eval']
    with pytest.raises(RuntimeError, match=r'PROBE \[777\.0\]'):
        direct(dict(_job(code, stress=False, allowed=['dev']), _home=str(s.home), _store=str(s.store)))
