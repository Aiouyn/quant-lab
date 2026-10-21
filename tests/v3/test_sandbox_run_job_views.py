"""#14 x #16: a jailed run_job sees ONLY its per-split view; sealed-split errors are sanitized (QL Validator)."""
import asyncio
import dataclasses
import json
import os
import re

import pytest

from desk.config import SETTINGS
from desk.lab import jail
from desk.lab.sandbox import run_job
from tests.v3.test_split_views import SENT, store  # noqa: F401  (fixture: full store with per-split sentinels)

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')

PROBE = '''
from pathlib import Path
import polars as _p
want = {{{s}}}
seen, files = set(), 0
def walk(top):
    global files
    for f in Path(top).rglob('*.parquet'):
        try:
            d = _p.read_parquet(f)
        except Exception:
            continue
        files += 1
        for c in d.columns:
            if d[c].dtype.is_numeric():
                seen.update({{float(v) for v in d[c].drop_nulls()}} & want)
for top in [{store!r}, {tmp!r}, '/workspace', '/home', '/tmp']:
    walk(top)
raise RuntimeError('PROBE files=%d seen=%r' % (files, sorted(seen)))
'''


def _settings(tmp_path, store):
    home = tmp_path / 'home'
    (home / 'artifacts').mkdir(parents=True)
    return dataclasses.replace(SETTINGS, home=home, store_dir=store)


def test_dev_job_cannot_see_funding_or_carry_outside_view(tmp_path, store):  # noqa: F811
    bad = {SENT[k] for k in ('val', 'holdout', 'forward')}
    bad |= {v / 1000 for v in bad}
    code = PROBE.format(s=', '.join(map(repr, sorted(bad))), store=str(store), tmp=str(tmp_path))
    out = asyncio.run(run_job({'type': 'strategy_eval', 'code': code, 'split': 'dev', 'allowed': ['dev']},
                              _settings(tmp_path, store), timeout_s=120))
    m = re.search(r'PROBE files=(\d+) seen=(\[.*?\])', out.get('error', ''))
    assert m, out
    assert m.group(2) == '[]', out                    # no val/holdout/forward funding_hl, carry, perps, macro, pred values
    assert int(m.group(1)) > 0, out                   # ...while the dev view itself is readable


def test_artifacts_not_mounted(tmp_path, store):  # noqa: F811
    s = _settings(tmp_path, store)
    (s.artifacts / 'library.json').write_text('{"secret": 1}')
    code = ("from pathlib import Path\n"
            f"p = Path({str(s.artifacts / 'library.json')!r})\n"
            "raise RuntimeError('PROBE ' + ('SEEN' if p.exists() else 'HIDDEN'))\n")
    out = asyncio.run(run_job({'type': 'strategy_eval', 'code': code, 'split': 'dev', 'allowed': ['dev']}, s, timeout_s=120))
    assert 'PROBE HIDDEN' in out['error'], out


def test_val_job_error_is_sanitized(tmp_path, store, monkeypatch):  # noqa: F811
    monkeypatch.setenv('DESK_SEALED_QUANTUM_S', '1')
    from desk.pipeline.validation import Pipeline
    s = _settings(tmp_path, store)
    p = Pipeline.__new__(Pipeline)
    p.s, p.org = s, None
    code = ("from pathlib import Path\nimport polars as _p\n"
            "from glob import glob\nv = sorted(_p.read_parquet(glob('/tmp/**/funding_hl.parquet', recursive=True)[0])['funding_rate'].unique().to_list())\n"
            "print('STDOUT-LEAK', v)\nraise RuntimeError('LEAK %r' % v)\n")
    for split, allowed in (('val', ['dev', 'val']), ('holdout', ['dev', 'val', 'holdout'])):
        r = asyncio.run(p._job({'type': 'strategy_eval', 'code': code, 'split': split, 'allowed': allowed}, timeout=120))
        assert r == {'error': 'sealed_failure', 'sealed': True}, r
    log = [json.loads(l) for l in (s.home / 'logs' / 'sealed_errors.jsonl').read_text().splitlines()]
    assert len(log) == 2 and 'LEAK' in json.dumps(log[0]['raw'])          # raw text kept host-side only
    r = asyncio.run(p._job({'type': 'strategy_eval', 'code': code, 'split': 'dev', 'allowed': ['dev']}, timeout=120))
    assert 'LEAK' in r['error']                                           # dev errors unchanged


def _pipe(s):
    from desk.pipeline.validation import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.s, p.org = s, None
    return p


BUSY = ("import time\nt = time.time()\nwhile time.time() - t < 3:\n    pass\n"
        "raise RuntimeError('condition was true')\n")
FAST = "raise ValueError('condition was false')\n"


def test_holdout_busy_loop_vs_immediate_indistinguishable(tmp_path, store, monkeypatch):  # noqa: F811
    """1-bit-per-run leak: error class and runtime must not differ between a busy-looping and an instant job."""
    import time
    monkeypatch.setenv('DESK_SEALED_QUANTUM_S', '8')
    s = _settings(tmp_path, store)
    p = _pipe(s)
    out = []
    for i, code in enumerate((BUSY, FAST)):
        t0 = time.monotonic()
        r = asyncio.run(p._job({'type': 'strategy_eval', 'code': code, 'split': 'holdout',
                                'allowed': ['dev', 'val', 'holdout'], '_identity': f'id{i}'}, timeout=60))
        out.append((r, time.monotonic() - t0))
    (r1, t1), (r2, t2) = out
    assert r1 == r2 == {'error': 'sealed_failure', 'sealed': True}
    assert 8 <= t1 < 8.8 and 8 <= t2 < 8.8, (t1, t2)                      # both released on the same quantum
    assert abs(t1 - t2) < 0.5, (t1, t2)


def test_resubmitted_holdout_identity_refused(tmp_path, store, monkeypatch):  # noqa: F811
    """The one-shot holdout view is keyed on the R4 identity: a second job with the same identity gets no data."""
    from desk.data import views
    monkeypatch.setenv('DESK_SEALED_QUANTUM_S', '1')
    s = _settings(tmp_path, store)
    p = _pipe(s)
    ok = "DESK = 'macro'\nPARAMS = {}\nraise RuntimeError('PROBE')\n"
    job = {'type': 'strategy_eval', 'code': ok, 'split': 'holdout', 'allowed': ['dev', 'val', 'holdout'], '_identity': 'a' * 64}
    asyncio.run(p._job(dict(job), timeout=60))
    assert views.holdout_consumed(store, 'a' * 64)
    with pytest.raises(views.ViewError):                                    # raw run_job refuses the identity
        asyncio.run(run_job(dict(job), s, timeout_s=60))
    assert asyncio.run(p._job(dict(job), timeout=60)) == {'error': 'sealed_failure', 'sealed': True}


def test_sealed_retry_leaves_no_ledger_event(tmp_path, store, monkeypatch):  # noqa: F811
    from desk.pipeline import validation
    monkeypatch.setenv('DESK_SEALED_QUANTUM_S', '1')
    calls = []

    async def fake(job, s, **kw):
        calls.append(kw.get('mem_mb'))
        return {'error': 'harness failed (memory)'}
    monkeypatch.setattr(validation, '_run_job', fake)
    p = _pipe(_settings(tmp_path, store))
    p.L = types_ns = type('L', (), {'event': lambda *a, **k: (_ for _ in ()).throw(AssertionError('ledger event'))})()
    r = asyncio.run(p._job({'type': 'strategy_eval', 'code': 'x', 'split': 'val', 'allowed': ['dev', 'val']}))
    assert r == {'error': 'sealed_failure', 'sealed': True} and calls == [None, 8000]
