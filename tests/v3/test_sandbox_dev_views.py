"""Sandbox data roots must only expose dev data (PR #7 review).

Planted val/holdout sentinel rows/files must never be visible to code run through `run_python`.
The env-root test passes today; the raw-filesystem test is xfail until OS-level isolation lands.
"""
import asyncio
import json
import types
from datetime import datetime, timezone

import polars as pl
import pytest

from desk.lab.sandbox import run_python

H = 3_600_000
DEV, VAL, HOLD = ['2025-01-01', '2025-01-10'], ['2025-01-11', '2025-01-15'], ['2025-01-16', '2025-01-20']
SENT_VAL, SENT_HOLD = 777.0, 999.0


def _ms(d):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _settings(tmp_path):
    store = tmp_path / 'store'
    return types.SimpleNamespace(store=store, artifacts=tmp_path / 'art', home=tmp_path,
                                 sandbox_timeout_s=120, sandbox_memory_mb=4000)


def _plant(s):
    """Full stores with sentinel values in every val/holdout row, then build the dev views."""
    from desk.lab import carry, macro, perps, pred
    st = s.store
    ts = list(range(_ms(DEV[0]), _ms(HOLD[1]) + 86_400_000, H))
    def mark(t):
        return SENT_HOLD if t >= _ms(HOLD[0]) else SENT_VAL if t >= _ms(VAL[0]) else 1.0
    sp = {'created': 0, 'dev': DEV, 'val': VAL, 'holdout': HOLD, 'forward': ['2025-01-21', '2099-12-31']}
    # carry
    (st / 'carry' / 'coins').mkdir(parents=True)
    pl.DataFrame({'ts': ts, 'coin': 'BTC', 'hl_funding': [mark(t) for t in ts], 'hl_premium': 0.0, 'bn_funding': 0.0,
                  'bn_rate_last': 0.0, 'bn_interval_h': 8.0, 'bn_premium': [mark(t) for t in ts],
                  'bn_close': [mark(t) for t in ts], 'bn_qvol': 1.0}).write_parquet(st / 'carry' / 'coins' / 'BTC.parquet')
    (st / 'carry' / 'splits.json').write_text(json.dumps(sp))
    carry.write_dev_view(s)
    # perps
    (st / 'perps').mkdir(parents=True)
    pl.DataFrame({'ts': ts, 'coin': 'BTC', 'px': [mark(t) for t in ts], 'px_fill': 1.0, 'f': 0.0,
                  'lo': 1.0, 'hi': 1.0}).write_parquet(st / 'perps' / 'panel.parquet')
    (st / 'perps' / 'splits.json').write_text(json.dumps(sp))
    perps.write_dev_view(s)
    # macro
    days = pl.date_range(datetime(2025, 1, 1), datetime(2025, 1, 20), eager=True)
    (st / 'macro').mkdir(parents=True)
    pl.DataFrame({'date': days, 'symbol': 'SPY',
                  'close': [mark(_ms(str(d))) for d in days]}).write_parquet(st / 'macro' / 'panel.parquet')
    (st / 'macro' / 'splits.json').write_text(json.dumps(sp))
    macro.write_dev_view(s)
    # pred (markets resolving in val/holdout carry the sentinel price)
    (st / 'pred').mkdir(parents=True)
    rows = [{'ts': t, 'market': i, 'p': mark(e) / 1000, 'end_ms': e, 'category': 'x', 'hours_to_end': (e - t) / H,
             'outcome': 1, 'volume': 1.0}
            for i, e in enumerate([_ms('2025-01-05'), _ms('2025-01-12'), _ms('2025-01-18')]) for t in ts if t <= e]
    pl.DataFrame(rows).write_parquet(st / 'pred' / 'panel.parquet')
    (st / 'pred' / 'splits.json').write_text(json.dumps(sp))
    pred.write_dev_view(s)


PROBE_ENV = '''
import os, polars as pl
from pathlib import Path
vals = []
for k in ('DESK_PERPS_DEV', 'DESK_MACRO_DEV', 'DESK_PRED_DEV'):
    df = pl.read_parquet(os.environ[k]); vals += [float(v) for c in df.columns if df[c].dtype.is_numeric() for v in df[c]]
for f in Path(os.environ['DESK_CARRY_ROOT'], 'coins').glob('*.parquet'):
    df = pl.read_parquet(f); vals += [float(v) for c in df.columns if df[c].dtype.is_numeric() for v in df[c].drop_nulls()]
print('MAXTS', max(pl.read_parquet(os.environ['DESK_PERPS_DEV'])['ts']))
print('SEEN', sorted({v for v in vals if v in (777.0, 999.0, 0.777, 0.999)}))
print('CARRY', carry_panel()['hl_funding'].max())
'''

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


def _run(code, s, tmp_path):
    r = asyncio.run(run_python(code, tmp_path / 'ws', s))
    assert r['status'] == 'ok', r['stderr']
    return r['stdout']


def test_sandbox_env_roots_hide_val_and_holdout(tmp_path):
    s = _settings(tmp_path)
    _plant(s)
    out = _run(PROBE_ENV, s, tmp_path)
    assert 'SEEN []' in out, out
    assert f"MAXTS {_ms('2025-01-11') - H}" in out, out
    assert 'CARRY 1.0' in out, out


@pytest.mark.xfail(strict=False, reason=(
    'run_python is not an OS-level filesystem boundary: sandboxed code can walk settings.store and read val/holdout '
    'files directly. Fixed by QL Safety\'s filesystem-isolation sandbox PR '
    'https://github.com/Aiouyn/quant-lab/pull/11; flip to a normal test once it lands.'))
def test_sandbox_cannot_read_val_holdout_from_filesystem(tmp_path):
    s = _settings(tmp_path)
    _plant(s)
    assert 'SEEN []' in _run(PROBE_FS, s, tmp_path)
