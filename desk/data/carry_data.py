"""Cross-venue funding/basis data for the carry desk: Hyperliquid (hourly funding + premium) vs Binance USDT-M
(funding events, hourly premium index and klines). Builds an hourly panel per coin:

  ts (hour, ms), coin,
  hl_funding    : HL funding rate paid for this hour (fraction; longs pay when > 0)
  hl_premium    : HL premium used for that hour's funding (mark vs oracle, fraction)
  bn_funding    : Binance funding rate PAID at this hour (0 when no funding event this hour)
  bn_rate_last  : last known Binance funding rate (forward-filled, causal)
  bn_interval_h : hours between Binance funding events (8 or 4)
  bn_premium    : Binance premium index, close of the 1h bar ENDING at ts (load_panel re-keys files stored by open time)
  bn_close      : Binance perp close of the bar ending at ts
  bn_qvol       : Binance quote volume of the bar ending at ts (1h, USDT)
Prices: the relative HL-vs-Binance price move over an hour is approximated by the change in (hl_premium -
bn_premium): both perps track the same spot index, so basis PnL of a hedged pair is the premium-spread change.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from .atomic import atomic_write_json

HL = 'https://api.hyperliquid.xyz/info'
BV = 'https://data.binance.vision/data/futures/um'
FAPI = 'https://fapi.binance.com'
HOUR = 3_600_000


def carry_root(settings):
    return settings.store / 'carry'


def universe(n=30):
    with httpx.Client(timeout=30) as c:
        meta, ctx = c.post(HL, json={'type': 'metaAndAssetCtxs'}).json()
        info = c.get(f'{FAPI}/fapi/v1/exchangeInfo').json()
    bn = {s['baseAsset']: s['symbol'] for s in info['symbols']
          if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
    rows = []
    for i, a in enumerate(meta['universe']):
        name = a['name']
        if a.get('isDelisted') or name not in bn:
            continue
        rows.append((float(ctx[i].get('dayNtlVlm', 0) or 0), name, bn[name]))
    rows.sort(reverse=True)
    return [{'coin': r[1], 'binance': r[2], 'hl_day_volume': r[0]} for r in rows[:n]]


def hl_funding(coin, start_ms, client):
    rows, cursor, end = [], start_ms, int(time.time() * 1000)
    while cursor < end:
        for attempt in range(5):
            r = client.post(HL, json={'type': 'fundingHistory', 'coin': coin, 'startTime': cursor})
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status(); break
        batch = r.json()
        if not batch:
            break
        rows += batch
        nxt = max(int(b['time']) for b in batch) + 1
        if nxt <= cursor or len(batch) < 2:
            break
        cursor = nxt
        time.sleep(0.6)
    if not rows:
        return None
    df = pl.DataFrame({'t': [int(x['time']) for x in rows], 'hl_funding': [float(x['fundingRate']) for x in rows],
                       'hl_premium': [float(x['premium']) for x in rows]})
    return (df.with_columns(((pl.col('t') + HOUR // 2) // HOUR * HOUR).alias('ts')).drop('t')
            .unique('ts', keep='last').sort('ts'))


def _months(start: Date, end: Date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f'{y}-{m:02d}'
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _zip_csv(client, url):
    r = client.get(url)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0])
    first = raw.split(b'\n', 1)[0]
    has_header = not first[:1].isdigit()
    return pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema_length=10000)


def bn_monthly(symbol, kind, start: Date, end: Date, client):
    parts = []
    for ym in _months(start, end):
        if kind == 'fundingRate':
            url = f'{BV}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip'
        else:
            url = f'{BV}/monthly/{kind}/{symbol}/1h/{symbol}-1h-{ym}.zip'
        df = _zip_csv(client, url)
        if df is not None and len(df):
            parts.append(df)
    return pl.concat(parts, how='diagonal_relaxed') if parts else None


def bn_funding_api(symbol, start_ms, client):
    rows, cursor = [], start_ms
    while True:
        r = client.get(f'{FAPI}/fapi/v1/fundingRate', params={'symbol': symbol, 'startTime': cursor, 'limit': 1000})
        if r.status_code != 200:
            break
        b = r.json()
        if not b:
            break
        rows += b
        if len(b) < 1000:
            break
        cursor = int(b[-1]['fundingTime']) + 1
        time.sleep(0.3)
    return rows


def bn_klines_api(symbol, start_ms, client, path='/fapi/v1/klines', extra=None):
    rows, cursor = [], start_ms
    while True:
        p = {'symbol': symbol, 'interval': '1h', 'startTime': cursor, 'limit': 1500}
        if extra:
            p.update(extra)
        r = client.get(f'{FAPI}{path}', params=p)
        if r.status_code != 200:
            break
        b = r.json()
        if not b:
            break
        rows += b
        if len(b) < 1500:
            break
        cursor = int(b[-1][0]) + 1
        time.sleep(0.3)
    return rows


def build_coin(u, root: Path, start=Date(2023, 6, 1)):
    coin, sym = u['coin'], u['binance']
    out = root / 'coins' / f'{coin}.parquet'
    if out.exists():
        return {'coin': coin, 'cached': True}
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        hl = hl_funding(coin, int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000), c)
        if hl is None or len(hl) < 24 * 30:
            return {'coin': coin, 'skipped': 'no HL funding history'}
        first = datetime.fromtimestamp(hl['ts'][0] / 1000, timezone.utc).date()
        today = datetime.now(timezone.utc).date()
        last_full = Date(today.year, today.month, 1)
        # Binance funding: monthly archives + API for the current month
        fr = bn_monthly(sym, 'fundingRate', first, Date(last_full.year, last_full.month, 1) if last_full.month > 1 else last_full, c)
        fund = []
        if fr is not None:
            tcols = [x for x in fr.columns if 'time' in x.lower()]
            rcols = [x for x in fr.columns if 'rate' in x.lower() and 'interval' not in x.lower()]
            tcol = tcols[0] if tcols else fr.columns[0]          # headerless archives: time first, rate last
            rcol = rcols[0] if rcols else fr.columns[-1]
            fund.append(fr.select(pl.col(tcol).cast(pl.Int64).alias('t'), pl.col(rcol).cast(pl.Float64).alias('rate')))
        api = bn_funding_api(sym, int(datetime(last_full.year, last_full.month, 1, tzinfo=timezone.utc).timestamp() * 1000) - 40 * 86_400_000, c)
        if api:
            fund.append(pl.DataFrame({'t': [int(x['fundingTime']) for x in api], 'rate': [float(x['fundingRate']) for x in api]}))
        if not fund:
            return {'coin': coin, 'skipped': 'no Binance funding'}
        bf = (pl.concat(fund).with_columns(((pl.col('t') + HOUR // 2) // HOUR * HOUR).alias('ts'))
              .group_by('ts').agg(pl.col('rate').last()).sort('ts'))
        # Binance hourly klines + premium index: monthly archives + API tail
        kl = bn_monthly(sym, 'klines', first, last_full, c)
        pi = bn_monthly(sym, 'premiumIndexKlines', first, last_full, c)
        tail_start = int(datetime(last_full.year, last_full.month, 1, tzinfo=timezone.utc).timestamp() * 1000) - 2 * 86_400_000
        kl_api = bn_klines_api(sym, tail_start, c)
        pi_api = bn_klines_api(sym, tail_start, c, '/fapi/v1/premiumIndexKlines')

    def kframe(df, api, names):
        parts = []
        if df is not None:
            cols = df.columns
            parts.append(df.select(pl.col(cols[0]).cast(pl.Int64).alias('ts'), pl.col(cols[4]).cast(pl.Float64).alias(names[0]),
                                   *( [pl.col(cols[7]).cast(pl.Float64).alias(names[1])] if len(names) > 1 else [])))
        if api:
            d = {'ts': [int(x[0]) for x in api], names[0]: [float(x[4]) for x in api]}
            if len(names) > 1:
                d[names[1]] = [float(x[7]) for x in api]
            parts.append(pl.DataFrame(d))
        if not parts:
            return None
        return pl.concat(parts, how='diagonal_relaxed').unique('ts', keep='last').sort('ts')
    k = kframe(kl, kl_api, ['bn_close', 'bn_qvol'])
    p = kframe(pi, pi_api, ['bn_premium'])
    grid = pl.DataFrame({'ts': np.arange(hl['ts'][0], hl['ts'][-1] + HOUR, HOUR, dtype=np.int64)})
    df = grid.join(hl, on='ts', how='left')
    df = df.join(bf.rename({'rate': 'bn_funding'}), on='ts', how='left')
    if k is not None:
        df = df.join(k, on='ts', how='left')
    if p is not None:
        df = df.join(p, on='ts', how='left')
    ev = df['bn_funding'].is_not_null()
    interval = df.select(pl.when(ev).then(pl.col('ts')).otherwise(None).forward_fill().alias('last_ev'))['last_ev']
    df = df.with_columns(
        pl.col('bn_funding').forward_fill().alias('bn_rate_last'),
        pl.col('bn_funding').fill_null(0.0),
        pl.col('hl_funding').fill_null(0.0),
        pl.col('hl_premium').forward_fill(),
        pl.col('bn_premium').forward_fill(),
        pl.col('bn_close').forward_fill(),
        pl.col('bn_qvol').fill_null(0.0) if 'bn_qvol' in df.columns else pl.lit(0.0).alias('bn_qvol'),
        pl.lit(coin).alias('coin'))
    evts = df.filter(ev)['ts'].to_numpy()
    gaps = np.diff(evts) / HOUR if len(evts) > 2 else np.array([8.0])
    iv = pl.Series(np.r_[gaps[:1], gaps]).alias('iv')
    df = df.join(pl.DataFrame({'ts': evts, 'bn_interval_h': iv.to_numpy()}), on='ts', how='left') \
        .with_columns(pl.col('bn_interval_h').forward_fill().fill_null(8.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression='zstd')
    return {'coin': coin, 'rows': len(df), 'first': str(first), 'bn_events': int(ev.sum())}


def build_carry_data(settings, n=30, workers=3, progress=print):
    root = carry_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    uni = universe(n)
    (root / 'universe.json').write_text(json.dumps(uni, indent=1))
    res = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(build_coin, u, root): u['coin'] for u in uni}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                r = {'coin': futs[f], 'error': repr(e)[:300]}
            res.append(r)
            progress(json.dumps(r))
    panel = load_panel(settings)
    if panel is not None:
        _freeze_splits(root, panel)
    return res


def load_panel(settings, coins=None):
    files = sorted((carry_root(settings) / 'coins').glob('*.parquet'))
    if coins:
        files = [f for f in files if f.stem in set(coins)]
    if not files:
        return None
    return align_bar_close(pl.concat([pl.read_parquet(f) for f in files], how='diagonal_relaxed').sort(['ts', 'coin']))


BAR_COLS = ('bn_premium', 'bn_close', 'bn_qvol')


def align_bar_close(panel):
    """Coin files store hourly klines/premium klines keyed by bar OPEN time (value known only at open + 1 h), while the
    HL columns at row t are known at t (funding paid at t, premium of the hour ending at t). Re-key the bar columns to
    the bar CLOSE so row t only holds what a live system has at t (verified fix 2026-10-02: 1 h look-ahead)."""
    cols = [c for c in BAR_COLS if c in panel.columns]
    if not cols or panel.is_empty():
        return panel
    bars = panel.select('ts', 'coin', *cols).with_columns(pl.col('ts') + HOUR)
    out = panel.drop(cols).join(bars, on=['ts', 'coin'], how='left')
    return out.with_columns([pl.col(c).forward_fill().over('coin') for c in ('bn_premium', 'bn_close') if c in cols]
                            + ([pl.col('bn_qvol').fill_null(0.0)] if 'bn_qvol' in cols else [])).sort(['ts', 'coin'])


def _freeze_splits(root, panel):
    f = root / 'splits.json'
    if f.exists():
        return json.loads(f.read_text())
    days = sorted({int(t) // 86_400_000 for t in panel['ts'].unique().to_list()})
    a, b = int(len(days) * 0.6), int(len(days) * 0.85)
    to = lambda d: datetime.fromtimestamp(d * 86400, timezone.utc).strftime('%Y-%m-%d')
    sp = {'created': time.time(), 'dev': [to(days[0]), to(days[a - 1])], 'val': [to(days[a]), to(days[b - 1])],
          'holdout': [to(days[b]), to(days[-1])], 'note': 'carry desk: chronological 60/25/15 over its own panel'}
    atomic_write_json(f, sp, indent=1)
    return sp


# ---------------------------------------------------------------------------------------------- Bybit hedge venue
BYBIT = 'https://api.bybit.com/v5/market'


def _by_get(client, path, params):
    for attempt in range(5):
        r = client.get(f'{BYBIT}/{path}', params=params)
        if r.status_code == 200:
            j = r.json()
            if j.get('retCode') == 0:
                return j['result']['list']
        time.sleep(2 * (attempt + 1))
    return []


def build_bybit_pair(coin, root: Path):
    """coins/<COIN>@BY.parquet: the existing HL side + Bybit as the hedge venue (mapped into the bn_* columns)."""
    src, out = root / 'coins' / f'{coin}.parquet', root / 'coins' / f'{coin}@BY.parquet'
    if out.exists():
        return {'coin': coin + '@BY', 'cached': True}
    if not src.exists():
        return {'coin': coin + '@BY', 'skipped': 'no HL panel'}
    hl = pl.read_parquet(src).select('ts', 'hl_funding', 'hl_premium')
    sym = f'{coin}USDT'
    start, end = int(hl['ts'][0]), int(hl['ts'][-1]) + HOUR
    with httpx.Client(timeout=30) as c:
        info = _by_get(c, 'instruments-info', {'category': 'linear', 'symbol': sym})
        if not info:
            return {'coin': coin + '@BY', 'skipped': 'not listed on Bybit'}
        fund, cursor = [], end
        while cursor > start:
            b = _by_get(c, 'funding/history', {'category': 'linear', 'symbol': sym, 'startTime': start, 'endTime': cursor, 'limit': 200})
            if not b:
                break
            fund += b
            m = min(int(x['fundingRateTimestamp']) for x in b)
            if m >= cursor or len(b) < 200:
                break
            cursor = m - 1
            time.sleep(0.1)
        prem, kl = [], []
        s = start
        while s < end:
            e = min(end, s + 999 * HOUR)
            prem += _by_get(c, 'premium-index-price-kline', {'category': 'linear', 'symbol': sym, 'interval': '60', 'start': s, 'end': e, 'limit': 1000})
            kl += _by_get(c, 'kline', {'category': 'linear', 'symbol': sym, 'interval': '60', 'start': s, 'end': e, 'limit': 1000})
            s = e + HOUR
            time.sleep(0.1)
    if not fund or not kl:
        return {'coin': coin + '@BY', 'skipped': 'no Bybit history'}
    bf = (pl.DataFrame({'t': [int(x['fundingRateTimestamp']) for x in fund], 'rate': [float(x['fundingRate']) for x in fund]})
          .with_columns(((pl.col('t') + HOUR // 2) // HOUR * HOUR).alias('ts')).group_by('ts').agg(pl.col('rate').last()))
    k = pl.DataFrame({'ts': [int(x[0]) for x in kl], 'bn_close': [float(x[4]) for x in kl], 'bn_qvol': [float(x[6]) for x in kl]}).unique('ts')
    p = pl.DataFrame({'ts': [int(x[0]) for x in prem], 'bn_premium': [float(x[4]) for x in prem]}).unique('ts') if prem else None
    first = int(min(bf['ts'].min(), k['ts'].min()))
    df = hl.filter(pl.col('ts') >= first).join(bf.rename({'rate': 'bn_funding'}), on='ts', how='left').join(k, on='ts', how='left')
    df = df.join(p, on='ts', how='left') if p is not None else df.with_columns(pl.lit(0.0).alias('bn_premium'))
    ev = df['bn_funding'].is_not_null()
    evts = df.filter(ev)['ts'].to_numpy()
    gaps = np.diff(evts) / HOUR if len(evts) > 2 else np.array([8.0])
    df = df.with_columns(pl.col('bn_funding').forward_fill().alias('bn_rate_last'), pl.col('bn_funding').fill_null(0.0),
                         pl.col('hl_funding').fill_null(0.0), pl.col('hl_premium').forward_fill(), pl.col('bn_premium').forward_fill(),
                         pl.col('bn_close').forward_fill(), pl.col('bn_qvol').fill_null(0.0), pl.lit(coin + '@BY').alias('coin'))
    df = df.join(pl.DataFrame({'ts': evts, 'bn_interval_h': np.r_[gaps[:1], gaps]}), on='ts', how='left') \
        .with_columns(pl.col('bn_interval_h').forward_fill().fill_null(8.0)).filter(pl.col('bn_rate_last').is_not_null())
    df.write_parquet(out, compression='zstd')
    return {'coin': coin + '@BY', 'rows': len(df), 'first': datetime.fromtimestamp(first / 1000, timezone.utc).strftime('%Y-%m-%d'),
            'events': int(ev.sum())}


def build_bybit_pairs(settings, workers=4, progress=print):
    root = carry_root(settings)
    coins = sorted(p.stem for p in (root / 'coins').glob('*.parquet') if '@' not in p.stem)
    res = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(build_bybit_pair, c, root): c for c in coins}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                r = {'coin': futs[f] + '@BY', 'error': repr(e)[:300]}
            res.append(r); progress(json.dumps(r))
    return res
