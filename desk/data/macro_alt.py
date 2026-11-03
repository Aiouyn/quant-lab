"""Alternative data for the MACRO desk: free rates/credit/inflation-expectation series (FRED) and futures positioning
(CFTC Commitments of Traders), merged into the macro panel as FEATURE-ONLY rows (tradable = 0).

Every value is dated by when it was PUBLISHED, not by the period it describes, so a strategy deciding at the close of
day t can only see what was public by then:
  * FRED daily market series (H.15 rates, Treasury real yields/breakevens, ICE BofA OAS): the value for day d is
    published after the close of the next business day, so it becomes visible at the close of d + 2 business days.
  * FRED economic releases (claims, payrolls, CPI, ...): only with FRED_API_KEY set (free key at fred.stlouisfed.org).
    We take ALFRED's INITIAL release (output_type=4), so revisions never leak, and date each value by its release day.
  * CFTC COT (legacy futures-only report): positions as of Tuesday, released Friday 15:30 ET (later in holiday
    weeks), so a report becomes visible at report date + 7 days.
A value is carried forward on later panel dates only while it is fresh (STALE_DAYS per source): a series that stops
updating disappears instead of freezing at its last level.

Panel rows: symbol 'FRED:<id>' or 'COT:<market>:<field>', cls 'alt_rates' / 'alt_econ' / 'alt_cot', tradable 0,
cost_bps 0, value in `close` (open/high/low/adjclose/adj_open/volume null). Sources that fail to download are reported
and served from the last good cache (<store>/macro/alt/*.parquet); nothing is ever synthesized.
Run alone: python -m desk.data.macro_alt
"""
from __future__ import annotations

import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

UA = {'User-Agent': 'quant-lab/1.0 (research)'}
FRED_CSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
FRED_API = 'https://api.stlouisfed.org/fred/series/observations'
COT_LEGACY = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json'     # legacy futures-only, 1986 ->

# daily market series, not revised after publication (lag: d + 2 business days)
FRED_DAILY = {
    'DGS3MO': '3m Treasury', 'DGS2': '2y Treasury', 'DGS10': '10y Treasury', 'DGS30': '30y Treasury',
    'T10Y2Y': '10y-2y slope', 'T10Y3M': '10y-3m slope', 'DFII10': '10y TIPS real yield', 'T10YIE': '10y breakeven',
    'T5YIFR': '5y5y forward inflation', 'DFF': 'fed funds effective', 'BAMLH0A0HYM2': 'US high-yield OAS',
    'BAMLC0A0CM': 'US investment-grade OAS', 'BAMLEMCBPIOAS': 'EM corporate OAS',
}
# economic releases (need FRED_API_KEY; initial release only, dated by release day)
FRED_RELEASES = {
    'ICSA': 'initial jobless claims (weekly)', 'PAYEMS': 'nonfarm payrolls', 'UNRATE': 'unemployment rate',
    'CPIAUCSL': 'CPI', 'CPILFESL': 'core CPI', 'INDPRO': 'industrial production', 'RSAFS': 'retail sales',
    'UMCSENT': 'Michigan sentiment', 'HOUST': 'housing starts', 'NFCI': 'Chicago Fed financial conditions (weekly)',
}
# CFTC contract market codes (legacy report) -> short name
COT_MARKETS = {
    '13874A': 'ES', '209742': 'NQ', '239742': 'RTY', '1170E1': 'VX', '043602': 'ZN', '042601': 'ZT', '020601': 'ZB',
    '088691': 'GC', '084691': 'SI', '085692': 'HG', '067651': 'CL', '023651': 'NG', '002602': 'ZC', '001602': 'ZW',
    '005602': 'ZS', '099741': '6E', '097741': '6J', '096742': '6B', '232741': '6A', '090741': '6C', '098662': 'DX',
    '133741': 'BTC',
}
STALE_DAYS = {'alt_rates': 10, 'alt_econ': 100, 'alt_cot': 21}
MIN_ROWS = 50


def bday_add(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _get(c, url, params=None, tries=4):
    last = None
    for attempt in range(tries):
        try:
            r = c.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f'HTTP {r.status_code}')
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'{url}: {last!r}')


# ------------------------------------------------------------------ FRED
def parse_fred_csv(text: str, sid: str) -> pl.DataFrame:
    """fredgraph.csv -> (obs_date, value). Missing values are '.' (or empty) and are dropped."""
    df = pl.read_csv(io.StringIO(text), infer_schema=False)
    dcol = df.columns[0]                                   # 'DATE' (old) or 'observation_date' (current)
    return (df.select(pl.col(dcol).str.to_date(strict=False).alias('obs_date'),
                      pl.col(sid).cast(pl.Float64, strict=False).alias('value'))
            .drop_nulls().sort('obs_date'))


def fred_daily(c, sid: str) -> pl.DataFrame:
    df = parse_fred_csv(_get(c, FRED_CSV, {'id': sid}).text, sid)
    return df.with_columns(pl.col('obs_date').map_elements(lambda d: bday_add(d, 2), return_dtype=pl.Date).alias('avail'))


def fred_initial_release(c, sid: str, key: str) -> pl.DataFrame:
    """ALFRED initial releases: each observation with the date it was first published."""
    r = _get(c, FRED_API, {'series_id': sid, 'api_key': key, 'file_type': 'json', 'output_type': 4,
                           'realtime_start': '1776-07-04', 'realtime_end': '9999-12-31', 'limit': 100000})
    obs = r.json().get('observations') or []
    df = pl.DataFrame({'obs_date': [o['date'] for o in obs], 'avail': [o['realtime_start'] for o in obs],
                       'value': [o['value'] for o in obs]}, schema={'obs_date': pl.String, 'avail': pl.String, 'value': pl.String})
    return (df.select(pl.col('obs_date').str.to_date(), pl.col('avail').str.to_date(),
                      pl.col('value').cast(pl.Float64, strict=False)).drop_nulls().sort('avail'))


# ------------------------------------------------------------------ CFTC COT
def cot_frame(rows: list[dict]) -> pl.DataFrame:
    """Socrata rows -> (code, report_date, spec_net_oi, comm_net_oi, oi). Net positions are scaled by open interest."""
    def f(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return None
    recs = []
    for r in rows:
        oi = f(r, 'open_interest_all')
        if not oi:
            continue
        nl, ns = f(r, 'noncomm_positions_long_all'), f(r, 'noncomm_positions_short_all')
        cl, cs = f(r, 'comm_positions_long_all'), f(r, 'comm_positions_short_all')
        recs.append({'code': str(r.get('cftc_contract_market_code', '')).strip(),
                     'report_date': str(r.get('report_date_as_yyyy_mm_dd', ''))[:10],
                     'spec_net_oi': (nl - ns) / oi if nl is not None and ns is not None else None,
                     'comm_net_oi': (cl - cs) / oi if cl is not None and cs is not None else None,
                     'oi': oi})
    if not recs:
        return pl.DataFrame(schema={'code': pl.String, 'report_date': pl.Date, 'spec_net_oi': pl.Float64,
                                    'comm_net_oi': pl.Float64, 'oi': pl.Float64})
    return (pl.DataFrame(recs).with_columns(pl.col('report_date').str.to_date(strict=False))
            .drop_nulls('report_date').unique(['code', 'report_date'], keep='last').sort(['code', 'report_date']))


def cot_fetch(c, code: str) -> pl.DataFrame:
    rows, off = [], 0
    while True:
        r = _get(c, COT_LEGACY, {'cftc_contract_market_code': code, '$order': 'report_date_as_yyyy_mm_dd',
                                 '$limit': 50000, '$offset': off})
        page = r.json()
        rows += page
        if len(page) < 50000:
            break
        off += 50000
    return cot_frame(rows)


def cot_long(df: pl.DataFrame) -> pl.DataFrame:
    """Wide COT frame -> long (symbol, obs_date, avail, value), visible one week after the Tuesday report date."""
    out = []
    for field in ('spec_net_oi', 'comm_net_oi', 'oi'):
        out.append(df.select(pl.concat_str([pl.lit('COT:'), pl.col('code').replace_strict(COT_MARKETS, default=pl.col('code')),
                                            pl.lit(':' + field)]).alias('symbol'),
                             pl.col('report_date').alias('obs_date'),
                             (pl.col('report_date') + pl.duration(days=7)).alias('avail'),
                             pl.col(field).alias('value')))
    return pl.concat(out).drop_nulls('value')


# ------------------------------------------------------------------ assembly
def fetch_all(cache: Path, key: str | None = None) -> tuple[pl.DataFrame, dict]:
    """Long frame (symbol, cls, obs_date, avail, value) for every source; per-source report. A failed source falls
    back to its last cached file (reported as 'cached')."""
    cache.mkdir(parents=True, exist_ok=True)
    rep, parts = {}, []
    state = {'ok': 0, 'fail': 0}

    def keep(name, cls, fn):
        f = cache / f'{name.replace(":", "_")}.parquet'
        try:
            if state['fail'] >= 3 and not state['ok']:
                raise RuntimeError('offline: first sources all failed, not retrying the rest')
            df = fn()
            state['ok'] += 1
            if len(df) < MIN_ROWS:
                raise RuntimeError(f'only {len(df)} rows')
            df.write_parquet(f)
            rep[name] = len(df)
        except Exception as e:
            state['fail'] += 1
            if f.exists():
                df = pl.read_parquet(f); rep[name] = f'cached ({repr(e)[:80]})'
            else:
                rep[name] = f'missing ({repr(e)[:120]})'; return
        parts.append(df.with_columns(pl.lit(cls).alias('cls')))

    with httpx.Client(headers=UA, follow_redirects=True) as c:
        for sid in FRED_DAILY:
            keep(f'FRED:{sid}', 'alt_rates', lambda sid=sid: fred_daily(c, sid).with_columns(pl.lit(f'FRED:{sid}').alias('symbol')))
        if key:
            for sid in FRED_RELEASES:
                keep(f'FRED:{sid}', 'alt_econ',
                     lambda sid=sid: fred_initial_release(c, sid, key).with_columns(pl.lit(f'FRED:{sid}').alias('symbol')))
        else:
            rep['FRED releases'] = 'skipped: set FRED_API_KEY (free) to add first-release economic data'
        for code, name in COT_MARKETS.items():
            keep(f'COT:{name}', 'alt_cot', lambda code=code: cot_long(cot_fetch(c, code)))
    if not parts:
        return pl.DataFrame(schema={'symbol': pl.String, 'cls': pl.String, 'obs_date': pl.Date, 'avail': pl.Date,
                                    'value': pl.Float64}), rep
    cols = ['symbol', 'cls', 'obs_date', 'avail', 'value']
    return pl.concat([p.select(cols) for p in parts]), rep


def to_panel_rows(alt: pl.DataFrame, dates: pl.Series) -> pl.DataFrame:
    """Place each series on the panel's dates: on date t the latest value whose avail <= t, if it is fresh enough.
    Several observations published the same day (e.g. a first release dump) keep the latest observation period."""
    if alt.is_empty():
        return alt
    grid = pl.DataFrame({'date': dates.unique().sort()})
    alt = (alt.sort(['symbol', 'avail', 'obs_date']).unique(['symbol', 'avail'], keep='last')
           .with_columns(pl.col('cls').replace_strict(STALE_DAYS, default=10).alias('stale')))
    out = []
    for sym, g in alt.group_by('symbol'):
        g = g.sort('avail')
        j = grid.filter(pl.col('date') >= g['avail'].min()).join_asof(g, left_on='date', right_on='avail', strategy='backward')
        j = j.filter((pl.col('date') - pl.col('avail')).dt.total_days() <= pl.col('stale'))
        out.append(j.select('date', 'symbol', 'cls', pl.col('value').alias('close')))
    if not out:
        return pl.DataFrame()
    return pl.concat(out).with_columns(pl.lit(0).alias('tradable'), pl.lit(0.0).alias('cost_bps'))


def merge_into_panel(panel: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    if rows.is_empty():
        return panel
    rows = rows.with_columns([pl.col('tradable').cast(panel.schema['tradable']) if 'tradable' in panel.columns else pl.col('tradable')])
    return (pl.concat([panel.filter(~pl.col('symbol').str.contains(':')), rows], how='diagonal_relaxed')
            .sort(['date', 'symbol']))


def build(settings, panel: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    if os.environ.get('MACRO_ALT_DATA', '1') == '0':
        return panel, {'skipped': 'MACRO_ALT_DATA=0'}
    alt, rep = fetch_all(settings.store / 'macro' / 'alt', os.environ.get('FRED_API_KEY') or None)
    rows = to_panel_rows(alt, panel['date'])
    out = merge_into_panel(panel, rows)
    return out, {'sources': rep, 'alt_symbols': int(rows['symbol'].n_unique()) if len(rows) else 0, 'alt_rows': len(rows)}


if __name__ == '__main__':
    from ..config import SETTINGS
    alt, rep = fetch_all(SETTINGS.store / 'macro' / 'alt', os.environ.get('FRED_API_KEY') or None)
    for k, v in rep.items():
        print(f'{k:24s} {v}')
