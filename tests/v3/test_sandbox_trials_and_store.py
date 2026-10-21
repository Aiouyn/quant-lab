"""PR #11 review (QL Validator): host-side trial logging + filesystem sentinel (QL Engineer, PR #7)."""
import asyncio
import json
import os
import types
from datetime import datetime, timezone

import polars as pl
import pytest

from desk.lab import jail
from desk.lab.sandbox import run_python

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')

H = 3_600_000
DEV, VAL, HOLD = ['2025-01-01', '2025-01-10'], ['2025-01-11', '2025-01-15'], ['2025-01-16', '2025-01-20']


def _ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _settings(tmp_path):
    return types.SimpleNamespace(store=tmp_path / 'store', artifacts=tmp_path / 'art', home=tmp_path,
                                 sandbox_timeout_s=120, sandbox_memory_mb=4000)


# ---------------------------------------------------------------- trials are written by the host
TAMPER = '''
import os, glob
from desk.research import _log_trial
_log_trial('dev_backtest', 1.5, 10, 'abc')
# try everything to reach / suppress the trial log
print('FD_ENV', 'DESK_TRIALS' in os.environ)
cands = [os.getcwd() + '/trials.jsonl']
for top in ('/tmp', '/workspace', '/home', os.path.dirname(os.getcwd()), os.path.dirname(os.path.dirname(os.getcwd()))):
    for d, _, fs in os.walk(top):
        cands += [os.path.join(d, f) for f in fs if f.endswith('.jsonl')]
for p in cands:
    for how in ('w', 'r+'):
        try:
            open(p, how).truncate(0); print('TOUCHED', p)
        except OSError:
            pass
    try:
        os.remove(p); print('REMOVED', p)
    except OSError:
        pass
os.close(int(os.environ['DESK_TRIALS_FD']))       # cannot un-send what was already written
'''


def test_trials_logged_by_host_despite_tampering(tmp_path):
    s = _settings(tmp_path)
    trials = tmp_path / 'trial_logs' / 'agent.jsonl'
    trials.parent.mkdir()
    trials.write_text('{"kind": "dev_backtest", "sharpe": 0.1, "days": 5, "id": "old"}\n')
    r = asyncio.run(run_python(TAMPER, tmp_path / 'ws', s, trials_path=trials, extra_env={'DESK_TRIALS': trials}))
    assert r['status'] == 'ok', r
    assert 'FD_ENV False' in r['stdout'] and str(trials) not in r['stdout'], r   # only workspace decoys reachable
    rows = [json.loads(l) for l in trials.read_text().splitlines()]
    assert [x['id'] for x in rows] == ['old', 'abc']
    runs = [json.loads(l) for l in (tmp_path / 'trial_logs' / 'agent.runs.jsonl').read_text().splitlines()]
    assert len(runs) == 1 and runs[0]['trials'] == 1 and runs[0]['status'] == 'ok'


def test_run_recorded_even_when_code_logs_nothing_or_crashes(tmp_path):
    s = _settings(tmp_path)
    trials = tmp_path / 'trial_logs' / 'a.jsonl'
    asyncio.run(run_python('import os\nos.close(int(os.environ["DESK_TRIALS_FD"]))\nraise SystemExit(3)', tmp_path / 'ws', s,
                           trials_path=trials))
    asyncio.run(run_python('print(1)', tmp_path / 'ws', s, trials_path=trials))
    runs = [json.loads(l) for l in (tmp_path / 'trial_logs' / 'a.runs.jsonl').read_text().splitlines()]
    assert [x['status'] for x in runs] == ['error:3', 'ok'] and all(x['trials'] == 0 for x in runs)


def test_forged_trial_lines_are_validated(tmp_path):
    s = _settings(tmp_path)
    trials = tmp_path / 'trial_logs' / 'b.jsonl'
    code = ('import os\nfd = int(os.environ["DESK_TRIALS_FD"])\n'
            'os.write(fd, b"garbage\\n{\\"kind\\": \\"x\\"}\\n")\n'
            'from desk.research import _log_trial\n_log_trial("dev_backtest", 0.5, 3, "ok1")\n')
    asyncio.run(run_python(code, tmp_path / 'ws', s, trials_path=trials))
    assert [json.loads(l)['id'] for l in trials.read_text().splitlines()] == ['ok1']


def test_trials_path_inside_workspace_rejected(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(run_python('print(1)', tmp_path / 'ws', _settings(tmp_path), trials_path=tmp_path / 'ws' / 't.jsonl'))


# ---------------------------------------------------------------- QL Engineer's PR #7 sentinel (no xfail)
PROBE_FS = '''
import os, polars as pl
from pathlib import Path
root = Path(os.environ['DESK_CARRY_ROOT'])
while root.name != 'store':
    root = root.parent
seen = set()
for f in root.rglob('*.parquet'):
    df = pl.read_parquet(f)
    for c in df.columns:
        if df[c].dtype.is_numeric():
            seen |= {float(v) for v in df[c].drop_nulls()} & {777.0, 999.0, 0.777, 0.999}
print('SEEN', sorted(seen))
'''


def test_sandbox_cannot_read_val_holdout_from_filesystem(tmp_path):
    """Copy of tests/v3/test_sandbox_dev_views.py::test_sandbox_cannot_read_val_holdout_from_filesystem (PR #7),
    self-contained so it does not depend on #7's write_dev_view helpers; passes with the OS jail."""
    s = _settings(tmp_path)
    st = s.store
    ts = list(range(_ms(DEV[0]), _ms(HOLD[1]) + 86_400_000, H))
    mark = lambda t: 999.0 if t >= _ms(HOLD[0]) else 777.0 if t >= _ms(VAL[0]) else 1.0
    sp = {'created': 0, 'dev': DEV, 'val': VAL, 'holdout': HOLD, 'forward': ['2025-01-21', '2099-12-31']}
    (st / 'carry' / 'coins').mkdir(parents=True)
    pl.DataFrame({'ts': ts, 'coin': 'BTC', 'hl_funding': [mark(t) for t in ts]}).write_parquet(st / 'carry' / 'coins' / 'BTC.parquet')
    (st / 'carry' / 'splits.json').write_text(json.dumps(sp))
    for desk in ('perps', 'macro', 'pred'):
        (st / desk).mkdir(parents=True)
        pl.DataFrame({'ts': ts, 'px': [mark(t) for t in ts], 'p': [mark(t) / 1000 for t in ts]}).write_parquet(st / desk / 'panel.parquet')
    pl.DataFrame({'ts': ts, 'funding': [mark(t) for t in ts]}).write_parquet(st / 'funding_hl.parquet')
    r = asyncio.run(run_python(PROBE_FS, tmp_path / 'ws', s))
    assert r['status'] == 'ok', r['stderr']
    assert 'SEEN []' in r['stdout'], r['stdout']
