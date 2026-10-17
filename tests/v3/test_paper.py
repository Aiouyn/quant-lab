"""Paper engine timing/accounting tests with synthetic feeds (no network)."""
import dataclasses
import json

import numpy as np
import pytest

from desk.config import SETTINGS
from desk.pipeline.bundle import export_bundle, load_bundle
from desk.live.paper import PaperSession

CODE = '''RESOLUTION='10s'; LOOKBACK=2
PARAMS={}
def signal(df, params):
    return np.ones(len(df))
'''


def row(ts, mid=100_000.0, funding=0.0001):
    return {'ts': ts, 'valid': 1, 'bid_px': mid - 0.05, 'ask_px': mid + 0.05, 'mid': mid, 'spread_bps': 0.01,
            'bimp_1': 0.3, 'aimp_1': 0.3, 'bimp_10': 1.0, 'aimp_10': 1.0, 'bimp_50': 3.0, 'aimp_50': 3.0,
            'qi10': 0.1, 'funding_rate': funding, 'ofi5': 0.0, 'buy_sz': 0.0, 'sell_sz': 0.0}


@pytest.fixture()
def session(tmp_path):
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    p = export_bundle(s, {'id': 'S-t', 'name': 't', 'author': 'a', 'pod': 'p', 'code': CODE}, {'taker_bps': 4.5, 'slippage_bps': 0.5})
    return PaperSession(s, str(p), feature_source='okx')


def test_bundle_hash_detects_edits(tmp_path):
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    p = export_bundle(s, {'id': 'S-x', 'name': 'x', 'author': 'a', 'pod': 'p', 'code': CODE}, {})
    load_bundle(p)
    (p / 'strategy.py').write_text(CODE + '\n# edited', encoding='utf-8')
    with pytest.raises(ValueError):
        load_bundle(p)


def test_decide_once_per_bar_and_fill_strictly_later(session):
    t0 = 1_800_000_000_000 - (1_800_000_000_000 % 3_600_000) + 3_600_000   # an exact hour
    seconds = [s for s in range(0, 60) if s not in (19, 39)]              # bar ends at :19 and :39 are skipped
    for s in seconds:
        session.on_okx_row(row(t0 + s * 1000))
        session.on_hl_row(row(t0 + s * 1000))
    dec = [json.loads(l) for l in (session.dir / 'decisions.jsonl').read_text().splitlines()]
    bars = [(d['ts'] + 1000) // 10_000 for d in dec]
    assert len(bars) == len(set(bars))                                     # never two decisions in one bar
    fills = [json.loads(l) for l in (session.dir / 'fills.jsonl').read_text().splitlines()]
    assert fills and all(f['ts'] > f['decided'] for f in fills)            # >= 1 s latency, never same second
    assert fills[0]['price'] > fills[0]['mid']                             # buying crosses the spread


def test_funding_charged_at_hour_boundary(session):
    t0 = 1_800_000_000_000 - (1_800_000_000_000 % 3_600_000) + 3_600_000 - 30_000
    for s in range(0, 60):
        session.on_okx_row(row(t0 + s * 1000))
        session.on_hl_row(row(t0 + s * 1000))
    x = session.sleeves[0]
    assert x.pos > 0 and x.funding == pytest.approx(x.pos * 100_000.0 * 0.0001, rel=1e-6)


MAKER = CODE + "EXECUTION = {'mode': 'maker', 'ttl_s': 30, 'on_expiry': 'cancel', 'repeg_s': 5, 'exit': 'maker'}\n"


def test_paper_maker_queue_then_fill(tmp_path):
    s = dataclasses.replace(SETTINGS, home=tmp_path)
    p = export_bundle(s, {'id': 'S-m', 'name': 'm', 'author': 'a', 'pod': 'p', 'code': MAKER},
                      {'taker_bps': 4.5, 'maker_bps': 1.5, 'slippage_bps': 0.5})
    sess = PaperSession(s, str(p), feature_source='okx')
    t0 = 1_800_000_000_000 - (1_800_000_000_000 % 3_600_000) + 3_600_000
    bid = 100_000.0 - 0.05
    for k in range(40):
        ts = t0 + k * 1000
        r = row(ts)
        r['bid_sz0'], r['ask_sz0'] = 200.0, 200.0                 # 2 BTC queue ahead at the touch
        # first decision at second 19 (2 completed 10 s bars), order rests from second 20 behind 2 BTC of queue.
        # sellers hit our bid: 1.5 BTC at seconds 24-25 (still queue ahead), then 2 BTC at second 26 -> we fill
        r['_trades'] = [(-1, 75.0, bid)] if k in (24, 25) else ([(-1, 200.0, bid)] if k == 26 else [])
        sess.on_okx_row(dict(r))
        sess.on_hl_row(r)
    x = sess.sleeves[0]
    fills = [json.loads(l) for l in (sess.dir / 'fills.jsonl').read_text().splitlines()]
    assert fills and all(f.get('maker') for f in fills)
    assert fills[0]['ts'] == t0 + 26 * 1000 and fills[0]['price'] == pytest.approx(bid)
    assert x.pos > 0 and x.fees == pytest.approx(sum(f['fee'] for f in fills))
