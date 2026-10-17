"""Live-vs-archive parity check: is every historical dataset reproducible in real time?

For a day D recorded live by desk.live.parity_recorder, once the public archive for D is published (usually D+1):
compare the live rows with the archive rows on the same keys (timestamp, coin/symbol) field by field.
Verdict per dataset: VERIFIED (fields agree), PARTIAL (agree where the live source can see, e.g. depth beyond the
1000-level REST window), MISMATCH (systematic differences), or NO_DATA. Run: python -m desk.data.parity 2026-10-03
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import requests

from .harvest import COINS20, BN_SYMBOL, _hl_get, lz4_frames

BV = 'https://data.binance.vision/data/futures/um'
TOL = 1e-6


def live(settings, stream, d):
    root = settings.home / 'parity' / stream
    f = root / f'{d}.parquet'
    if f.exists():
        return pl.read_parquet(f)
    files = sorted((root / d).glob('*.parquet')) if (root / d).exists() else []
    if files:
        return pl.concat([pl.read_parquet(x) for x in files], how='diagonal_relaxed')
    raw = subprocess.run(['rclone', 'cat', f'loopfactory:QUANT_DATA/live/{stream}/{d}.parquet'], capture_output=True).stdout
    return pl.read_parquet(io.BytesIO(raw)) if raw else None


def agree(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return {'n': 0}
    rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)
    return {'n': int(ok.sum()), 'median_rel_diff': float(np.median(rel)), 'p95_rel_diff': float(np.percentile(rel, 95)),
            'exact_share': float(np.mean(rel < TOL)), 'corr': float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None}


def verdict(stats, good=0.001, partial=0.02):
    meds = [s['median_rel_diff'] for s in stats.values() if isinstance(s, dict) and s.get('n')]
    if not meds:
        return 'NO_DATA'
    m = max(meds)
    return 'VERIFIED' if m <= good else 'PARTIAL' if m <= partial else 'MISMATCH'


def check_hl_asset_ctxs(settings, d):
    lv = live(settings, 'hl_asset_ctxs', d)
    import lz4.frame
    raw = _hl_get(f'asset_ctxs/{d.replace("-", "")}.csv.lz4')
    if lv is None or raw is None:
        return {'verdict': 'NO_DATA'}
    ar = pl.read_csv(io.BytesIO(lz4_frames(raw)))
    j = lv.join(ar, on=['time', 'coin'], suffix='_arch')
    fields = ['funding', 'open_interest', 'premium', 'oracle_px', 'mark_px', 'mid_px', 'impact_bid_px', 'impact_ask_px']
    stats = {f: agree(j[f], j[f + '_arch']) for f in fields}
    lag = (lv['recv_ms'] - lv['time'].str.to_datetime(time_zone='UTC').dt.epoch('ms')).median() / 1000
    return {'verdict': verdict(stats), 'matched_rows': len(j), 'archive_rows': len(ar), 'live_rows': len(lv),
            'live_delay_s_after_minute': lag, 'fields': stats}


def check_hl_l2book(settings, d, coins=('BTC', 'SOL', 'PUMP')):
    import lz4.frame
    lv = live(settings, 'hl_l2book', d)
    if lv is None:
        return {'verdict': 'NO_DATA'}
    out = {}
    for c in coins:
        rows = []
        for h in range(24):
            raw = _hl_get(f'market_data/{d.replace("-", "")}/{h}/l2Book/{c}.lz4')
            if raw:
                for line in lz4_frames(raw).decode().splitlines():
                    x = json.loads(line)['raw']['data']
                    rows.append({'time': int(x['time']), 'a_bid': float(x['levels'][0][0]['px']), 'a_ask': float(x['levels'][1][0]['px']),
                                 'a_bdepth': sum(float(l['sz']) for l in x['levels'][0]), 'a_adepth': sum(float(l['sz']) for l in x['levels'][1])})
        if not rows:
            continue
        ar = pl.DataFrame(rows).sort('time')
        l = lv.filter(pl.col('coin') == c).sort('time').with_columns(
            pl.col('bid_px').list.first().alias('bid'), pl.col('ask_px').list.first().alias('ask'),
            pl.col('bid_sz').list.sum().alias('bdepth'), pl.col('ask_sz').list.sum().alias('adepth'))
        j = l.join(ar, on='time', how='inner')                 # identical book versions share the exchange timestamp
        out[c] = {'same_book_versions': len(j), 'live_samples': len(l), 'archive_snapshots': len(ar),
                  'bid': agree(j['bid'], j['a_bid']), 'ask': agree(j['ask'], j['a_ask']),
                  'bid_depth20': agree(j['bdepth'], j['a_bdepth']), 'ask_depth20': agree(j['adepth'], j['a_adepth']),
                  'live_delay_s': float(((l['recv_ms'] - l['time']) / 1000).median())}
    flat = {f'{c}.{k}': v for c, s in out.items() for k, v in s.items() if isinstance(v, dict)}
    return {'verdict': verdict(flat), 'coins': out}


def _bn_daily(kind, sym, d):
    r = requests.get(f'{BV}/daily/{kind}/{sym}/{sym}-{kind}-{d}.zip', timeout=120)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0])
    return pl.read_csv(io.BytesIO(raw), has_header=not raw[:1].isdigit(), infer_schema_length=None)


def check_bn_metrics(settings, d, syms=('SOLUSDT', 'BTCUSDT', '1000PEPEUSDT')):
    lg = live(settings, 'bn_metrics', d)
    if lg is None:
        return {'verdict': 'NO_DATA'}
    lags = (lg.filter(~pl.col('backfill')).group_by('column').agg(pl.col('first_seen_lag_s').median().alias('median'),
                                                                  pl.col('first_seen_lag_s').quantile(0.95).alias('p95')))
    lv = lg.pivot(on='column', index=['create_time', 'symbol'], values='value', aggregate_function='first')
    out = {}
    for s in syms:
        ar = _bn_daily('metrics', s, d)
        if ar is None:
            continue
        j = lv.filter(pl.col('symbol') == s).join(ar, on=['create_time', 'symbol'], suffix='_arch')
        out[s] = {f: agree(j[f], j[f + '_arch']) for f in ('sum_open_interest', 'sum_open_interest_value',
                  'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio', 'count_long_short_ratio',
                  'sum_taker_long_short_vol_ratio')}
    flat = {f'{s}.{k}': v for s, x in out.items() for k, v in x.items()}
    return {'verdict': verdict(flat, 0.001, 0.05), 'availability_lag_s_after_create_time': lags.to_dicts(),
            'symbols': out}


def check_bn_bookdepth(settings, d, syms=('SOLUSDT', '1000PEPEUSDT')):
    lv = live(settings, 'bn_bookdepth', d)
    if lv is None:
        return {'verdict': 'NO_DATA'}
    out = {}
    for s in syms:
        ar = _bn_daily('bookDepth', s, d)
        if ar is None:
            continue
        ar = ar.with_columns(pl.col('timestamp').str.to_datetime().dt.epoch('ms').alias('t'))
        l = lv.filter((pl.col('symbol') == s) & pl.col('complete')).with_columns(pl.col('timestamp').str.to_datetime().dt.epoch('ms').alias('t'))
        j = l.sort('t').join_asof(ar.sort('t'), on='t', by='percentage', strategy='nearest', tolerance=15_000, suffix='_arch')
        out[s] = {'depth': agree(j['depth'], j['depth_arch']), 'notional': agree(j['notional'], j['notional_arch']),
                  'incomplete_live_share': float(1 - lv.filter(pl.col('symbol') == s)['complete'].mean())}
    flat = {f'{s}.{k}': v for s, x in out.items() for k, v in x.items() if isinstance(v, dict)}
    return {'verdict': verdict(flat, 0.02, 0.10), 'symbols': out,
            'note': 'live snapshots are up to 15 s apart from archive snapshots: small differences are timing, not data'}


def check_bn_trades(settings, d, syms=('SOLUSDT', 'ZECUSDT')):
    lv = live(settings, 'bn_trades_1s', d)
    if lv is None:
        return {'verdict': 'NO_DATA'}
    out = {}
    for s in syms:
        ar = _bn_daily('aggTrades', s, d)
        if ar is None:
            continue
        cols = ar.columns
        tcol = 'transact_time' if 'transact_time' in cols else cols[5]
        qcol = 'quantity' if 'quantity' in cols else cols[2]
        mcol = 'is_buyer_maker' if 'is_buyer_maker' in cols else cols[6]
        a = (ar.with_columns(((pl.col(tcol) // 1000) * 1000 + 1000).alias('ts'), pl.col(mcol).cast(pl.Utf8).str.to_lowercase().eq('true').alias('bm'))
             .group_by('ts').agg(pl.col(qcol).filter(~pl.col('bm')).sum().alias('buy_a'), pl.col(qcol).filter(pl.col('bm')).sum().alias('sell_a')))
        j = lv.filter(pl.col('symbol') == s).join(a, on='ts')
        out[s] = {'buy_qty': agree(j['buy_qty'], j['buy_a']), 'sell_qty': agree(j['sell_qty'], j['sell_a']),
                  'arrival_latency_ms_median': float(lv.filter(pl.col('symbol') == s)['arrival_latency_ms'].median())}
    flat = {f'{s}.{k}': v for s, x in out.items() for k, v in x.items() if isinstance(v, dict)}
    return {'verdict': verdict(flat, 0.001, 0.05), 'symbols': out}


CHECKS = {'hl_asset_ctxs': check_hl_asset_ctxs, 'hl_l2book': check_hl_l2book, 'bn_metrics': check_bn_metrics,
          'bn_bookdepth': check_bn_bookdepth, 'bn_trades_1s': check_bn_trades}


def run(settings, d):
    report = {'date': d}
    for name, fn in CHECKS.items():
        try:
            report[name] = fn(settings, d)
        except Exception as e:
            report[name] = {'verdict': 'ERROR', 'error': repr(e)[:400]}
        print(name, report[name].get('verdict'), flush=True)
    out = settings.home / 'parity' / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{d}.json').write_text(json.dumps(report, indent=1, default=str))
    return report


if __name__ == '__main__':
    from ..config import SETTINGS
    run(SETTINGS, sys.argv[1] if len(sys.argv) > 1 else (date.today().isoformat()))
