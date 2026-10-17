"""Drill markets that measure the research organization itself.

planted : real order-book days whose price path carries an injected, causal, tradable drift driven by a hidden
          signal (a slow queue-imbalance state). A capable firm should discover it and get it approved.
          Hidden signal: s_t = tanh(z_1d(rolling_mean_15min(qi10))) ; drift = BETA_BPS_PER_S * s_{t-1}.
null    : each day's features are paired with ANOTHER day's price path whose 1-minute return blocks are randomly
          permuted, so neither book/flow features nor the price history predict returns beyond one minute.
          Every approval here is a false discovery.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..data.ingest import bars

BETA_BPS_PER_S = 36.0 / 1800      # |s|=1 held 30 min -> ~36 bps expected drift
PRICE_COLS = ('bid_px', 'ask_px', 'mid', 'vwap', 'last_px')


def hidden_signal(df):
    q = df['qi10'].rolling_mean(900, min_samples=300)
    z = (q - q.rolling_mean(86400, min_samples=3600)) / q.rolling_std(86400, min_samples=3600)
    return np.tanh(z.fill_null(0).fill_nan(0).to_numpy())


def _write_tape(src, store, date, df_new, df_old):
    """Shift real trade prices by the same per-second price ratio the drill applied to the book."""
    tp = src / 'tape' / f'{date}.parquet'
    if not tp.exists():
        return
    ratio = pl.DataFrame({'ts': df_new['ts'], 'ratio': (df_new['mid'] / df_old['mid']).to_numpy()})
    t = (pl.read_parquet(tp).join_asof(ratio.sort('ts'), on='ts', strategy='backward')
         .with_columns((pl.col('px') * pl.col('ratio').fill_null(1.0)).round(0).cast(pl.Int32).alias('px'))
         .drop('ratio').group_by(['ts', 'px', 'side']).agg(pl.col('size').sum()).sort(['ts', 'side', 'px']))
    (store / 'tape').mkdir(parents=True, exist_ok=True)
    t.write_parquet(store / 'tape' / f'{date}.parquet', compression='zstd')


def _write(store, date, df):
    for sub in ('1s', '10s', '1m', 'meta'):
        (store / sub).mkdir(parents=True, exist_ok=True)
    df.write_parquet(store / '1s' / f'{date}.parquet', compression='zstd')
    for res in ('10s', '1m'):
        bars(df, res).write_parquet(store / res / f'{date}.parquet', compression='zstd')
    (store / 'meta' / f'{date}.json').write_text(json.dumps({'date': date, 'drill': True}))


def build_drill(settings, kind, n_days=90, seed=7):
    src = settings.store
    days = sorted(p.stem for p in (src / '1s').glob('*.parquet'))[:n_days]
    if len(days) < 40:
        raise SystemExit(f'need >= 40 ingested days for a drill, have {len(days)}')
    home = settings.home / 'drills' / kind
    store = home / 'store'
    if home.exists():
        shutil.rmtree(home)
    store.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    a, b = int(len(days) * 0.6), int(len(days) * 0.85)
    (store / 'splits.json').write_text(json.dumps({'created': time.time(), 'dev': days[:a], 'val': days[a:b],
                                                   'holdout': days[b:], 'forward': [], 'drill': kind}))
    for f in ('funding_hl.parquet',):
        if (src / f).exists():
            shutil.copy2(src / f, store / f)
    carry = 0.0
    hist = None
    perm = rng.permutation(len(days))
    for i, d in enumerate(days):
        df = pl.read_parquet(src / '1s' / f'{d}.parquet')
        original = df.select('ts', 'mid')
        if kind == 'planted':
            # compute the hidden signal with one day of history so the 1-day z-score is warm
            full = pl.concat([hist, df], how='diagonal_relaxed') if hist is not None else df
            s = hidden_signal(full)[-len(df):]
            hist = df.select('ts', 'qi10')
            drift = np.r_[0.0, s[:-1]] * BETA_BPS_PER_S
            D = carry + np.cumsum(drift)
            carry = float(D[-1])
            mult = np.exp(D / 1e4)
            df = df.with_columns([(pl.col(c) * pl.Series(mult)).alias(c) for c in PRICE_COLS if c in df.columns])
        else:
            other = pl.read_parquet(src / '1s' / f'{days[perm[i]]}.parquet', columns=['mid'])
            r = np.diff(np.log(other['mid'].to_numpy()), prepend=np.log(other['mid'][0]))
            r[0] = 0.0
            # scramble the borrowed path in 60 s blocks: kills cross-day AND own-path predictability (momentum,
            # reversal) beyond one minute while keeping realistic intraminute returns and volatility levels
            blocks = r[: len(r) // 60 * 60].reshape(-1, 60)
            r = np.r_[blocks[rng.permutation(len(blocks))].ravel(), r[len(r) // 60 * 60:]]
            m0 = float(df['mid'][0])
            new_mid = m0 * np.exp(np.cumsum(r))
            ratio = new_mid / df['mid'].to_numpy()
            df = df.with_columns([(pl.col(c) * pl.Series(ratio)).alias(c) for c in PRICE_COLS if c in df.columns])
        _write(store, d, df)
        _write_tape(src, store, d, df.select('ts', 'mid'), original)
    return {'drill': kind, 'home': str(home), 'days': len(days), 'dev': a, 'val': b - a, 'holdout': len(days) - b,
            'run': f'python -m desk run --drill {kind} --agents 12 --hours 2'}


def drill_report(settings, kind):
    """Score a finished drill run: discovery of the hidden signal (planted) / false discoveries (null)."""
    from ..company.ledger import Ledger
    home = settings.home / 'drills' / kind
    L = Ledger(home / 'desk.sqlite3')
    st = {r['status']: r['n'] for r in L.q('SELECT status, COUNT(*) n FROM strategies GROUP BY status')}
    out = {'drill': kind, 'strategies': st, 'tokens': L.status()['tokens'], 'factors': len(L.q('SELECT id FROM factors'))}
    approved = st.get('approved', 0) + st.get('paper', 0)
    if kind == 'null':
        out['false_discoveries'] = approved
        out['passed_validation_falsely'] = approved + st.get('rejected_holdout', 0)
        out['verdict'] = 'PASS (no false discoveries)' if approved == 0 else 'FAIL (false discoveries approved)'
        return out
    # planted: how well do registered factors / approved strategies track the hidden signal?
    from ..data.store import FeatureStore
    import dataclasses
    s2 = dataclasses.replace(settings, home=home, store_dir=None)
    store = FeatureStore(s2, allowed=('dev', 'val', 'holdout'))
    days = store.days('dev')[-5:]
    df1 = store.load(days, '1s')
    hid = hidden_signal(df1)
    best = []
    for f in L.q('SELECT id, name, code, resolution FROM factors'):
        try:
            res = f['resolution']
            df = store.load(days, res)
            ns = {'np': np, 'pl': pl}
            exec(f['code'], ns)
            v = np.asarray(ns['factor'](df, dict(ns.get('PARAMS', {}))), float)
            h = pl.DataFrame({'ts': df1['ts'], 'h': hid}).join(pl.DataFrame({'ts': df['ts'], 'v': v}), on='ts', how='inner')
            c = np.corrcoef(np.nan_to_num(h['h'].to_numpy()), np.nan_to_num(h['v'].to_numpy()))[0, 1]
            best.append((abs(float(c)), f['id'], f['name']))
        except Exception:
            continue
    best.sort(reverse=True)
    out['top_factor_corr_with_hidden'] = best[:5]
    out['approved'] = approved
    out['verdict'] = ('PASS (discovered and approved)' if approved > 0 else
                      'PARTIAL (signal found, not approved)' if best and best[0][0] > 0.5 else 'FAIL (not discovered)')
    return out
