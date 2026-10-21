"""Shared factor-value cache: jailed factor jobs cannot overwrite or plant other factors' entries."""
import asyncio
import json
import os

import polars as pl
import pytest

from desk.lab import jail
from desk.lab.sandbox import run_job
from tests.v3.test_sandbox_run_job import _settings

pytestmark = pytest.mark.skipif(not jail.available() and os.environ.get('DESK_REQUIRE_JAIL') != '1',
                                reason='bubblewrap unavailable (set DESK_REQUIRE_JAIL=1 to make this a failure)')

HONEST = ("DESK = 'macro'\nPARAMS = {'n': 5}\n"
          "def factor(panel, params):\n"
          "    return panel.select((pl.col('adjclose') / pl.col('adjclose').shift(params['n']).over('symbol') - 1)"
          ".fill_null(0).alias('v'))['v']\n")

EVIL = HONEST.replace("def factor", '''import sys, json as _j
from pathlib import Path as _P
_job = _j.loads(_P(sys.argv[1]).read_text())
_hits = []
for _d in [_P(_job['values_dir']), _P(_job.get('_out_values', '/nonexistent'))]:
    for _n in ('victim.parquet', 'planted.parquet', '_manifest.json'):
        try:
            _d.mkdir(parents=True, exist_ok=True); (_d / _n).write_bytes(b'POISON'); _hits.append(str(_d / _n))
        except OSError:
            pass
    try:
        (_d / 'victim.parquet').unlink(); _hits.append('unlinked')
    except OSError:
        pass
def factor''')


def _job(code, fid, vdir, lib=()):
    return {'type': 'factor_eval', 'desk': 'macro', 'code': code, 'factor_id': fid, 'library_ids': list(lib),
            'values_dir': str(vdir)}


def test_honest_factor_values_are_cached(tmp_path):
    s = _settings(tmp_path)
    vdir = s.artifacts / 'factor_values'
    r = asyncio.run(run_job(_job(HONEST, 'f_honest', vdir), s, timeout_s=300))
    assert 'error' not in r, r
    df = pl.read_parquet(vdir / 'f_honest.parquet')
    assert 'lv' in df.columns and len(df) > 0
    man = json.loads((vdir / '_manifest.json').read_text())
    assert man['f_honest.parquet']['factor_id'] == 'f_honest' and len(man['f_honest.parquet']['code_sha256']) == 64
    # a later job reads the cached value for novelty (read-only mount)
    r2 = asyncio.run(run_job(_job(HONEST.replace("'n': 5", "'n': 6"), 'f_second', vdir, lib=['f_honest']), s, timeout_s=300))
    assert 'error' not in r2, r2
    assert set(json.loads((vdir / '_manifest.json').read_text())) == {'f_honest.parquet', 'f_second.parquet'}
    assert pl.read_parquet(vdir / 'f_honest.parquet').equals(df)       # first entry untouched by the second job


def test_malicious_factor_cannot_poison_other_entries(tmp_path):
    s = _settings(tmp_path)
    vdir = s.artifacts / 'factor_values'
    vdir.mkdir(parents=True)
    victim = pl.DataFrame({'date': [1], 'symbol': ['AAA'], 'lv': [0.5]})
    victim.write_parquet(vdir / 'victim.parquet')
    before = (vdir / 'victim.parquet').read_bytes()
    r = asyncio.run(run_job(_job(EVIL, 'f_evil', vdir), s, timeout_s=300))
    assert 'error' not in r, r
    assert (vdir / 'victim.parquet').read_bytes() == before          # not overwritten / not deleted
    assert not (vdir / 'planted.parquet').exists()                     # nothing planted
    man = json.loads((vdir / '_manifest.json').read_text())
    assert set(man) == {'f_evil.parquet'}                              # only its own entry, written by the host
    assert pl.read_parquet(vdir / 'f_evil.parquet')['lv'].len() > 0
