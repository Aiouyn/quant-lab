"""Charter R7c on the host (QL Validator forgery repro): every run_python run is >= 1 trial, keyed on the R4
identity, written by the host into the insert-only ledger core. Self-reported pipe rows never count."""
import asyncio
import os
import types

import pytest

from desk.company.ledger import Ledger
from desk.company.tools import Toolbox
from desk.lab import jail

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')

FORGE = '''
import os
fd = int(os.environ["DESK_TRIALS_FD"])
for _ in range(5):
    os.write(fd, b'{"kind": "dev_backtest", "sharpe": 3.5, "days": 400, "id": "forged"}\\n')
from desk.research import _log_trial
_log_trial("dev_backtest", 3.5, 400, "forged2")
'''

UNLOGGED = '''
import numpy as np
from desk.lab.backtest import backtest as _raw     # bypasses the logging helper entirely
x = np.cumsum(np.ones(10))
print("ran", float(x[-1]))
'''


@pytest.fixture
def tb(tmp_path, monkeypatch):
    from desk.company import rsi        # main's rsi.py lacks this helper until PR #2 (unrelated to R7c)
    monkeypatch.setattr(rsi, 'record_primitive_outcome', lambda *a, **k: None, raising=False)
    s = types.SimpleNamespace(home=tmp_path / 'home', store=tmp_path / 'store', artifacts=tmp_path / 'art',
                              sandbox_timeout_s=120, sandbox_memory_mb=4000)
    s.home.mkdir()
    L = Ledger(tmp_path / 'l.sqlite3')
    org = types.SimpleNamespace(ledger=L, settings=s, sandbox_slots=None)
    agent = types.SimpleNamespace(name='alice', role='researcher', workspace=tmp_path / 'ws')
    return Toolbox(org, agent)


def _trials(tb):
    return tb.L.q('SELECT kind, code_hash, sharpe, days, identity, split FROM trials')


def test_forged_pipe_rows_get_no_credit(tb):
    asyncio.run(tb.t_run_python(FORGE))
    rows = _trials(tb)
    assert len(rows) == 1 and rows[0]['kind'] == 'sandbox_run'
    assert rows[0]['sharpe'] is None and rows[0]['days'] is None           # no self-reported metric is ever used
    assert all(r['code_hash'] not in ('forged', 'forged2') for r in rows)


def test_unlogged_backtest_still_counts(tb):
    asyncio.run(tb.t_run_python(UNLOGGED))
    asyncio.run(tb.t_run_python('raise SystemExit(4)'))                    # crashed runs count too
    assert len(_trials(tb)) == 2


def test_identity_keyed_per_r4(tb):
    a = 'X = 1\ndef f():\n    """doc"""\n    return X\nprint(f())\n'
    cosmetic = '# comment\nX = 1   \n\ndef f():\n    """other doc"""\n    return X  # c\nprint(f())\n'
    changed = a.replace('X = 1', 'X = 2')
    for c in (a, cosmetic, changed):
        asyncio.run(tb.t_run_python(c))
    ids = [r['identity'] for r in _trials(tb)]
    assert len(ids) == 3 and all(i and len(i) == 64 for i in ids)
    assert ids[0] == ids[1] != ids[2]                                       # cosmetic edit same identity, constant change new
    assert all(r['code_hash'] == r['identity'] for r in _trials(tb))


def test_trials_table_insert_only(tb):
    asyncio.run(tb.t_run_python('print(1)'))
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        tb.L.x('DELETE FROM trials')
