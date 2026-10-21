"""Daily data refresh for the rolling FORWARD holdout of the perps desk (run by the supervisor, ~13:00 UTC).

1. Every finished day recorded live by desk.live.parity_recorder (hl_asset_ctxs: same instant and schema as the
   archive, verified field-by-field by desk.data.parity) is converted into the minute HL ctx store.
2. Binance metrics for the current and previous month are rebuilt from data.binance.vision daily files.
3. The perps panel (and its dev-only research view) is rebuilt.
Forward days are never visible to research (PerpsStore split guard); approved strategies are scored on them by
Pipeline.forward_check. Run: python -m desk.data.refresh
"""
from __future__ import annotations

import io
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .hl_ctx import MIN_VLM, ROOT as HLCTX
from .perps_panel import build as build_panel

LIVE = 'loopfactory:QUANT_DATA/live/hl_asset_ctxs'
FORWARD_START = date(2026, 10, 2)


def live_day(settings, d: str):
    local = settings.home / 'parity' / 'hl_asset_ctxs' / f'{d}.parquet'
    if local.exists():
        return pl.read_parquet(local)
    raw = subprocess.run(['rclone', 'cat', f'{LIVE}/{d}.parquet'], capture_output=True, timeout=600).stdout
    return pl.read_parquet(io.BytesIO(raw)) if raw else None


def to_hlctx(df: pl.DataFrame):
    df = df.with_columns(pl.col('time').str.to_datetime(time_zone='UTC').dt.epoch('ms').alias('t'))
    liquid = df.group_by('coin').agg(pl.col('day_ntl_vlm').cast(pl.Float64).max()).filter(pl.col('day_ntl_vlm') >= MIN_VLM)['coin']
    df = df.filter(pl.col('coin').is_in(liquid.implode()))
    f64 = [('mid_px', 'mid'), ('oracle_px', 'oracle'), ('mark_px', 'mark'), ('impact_bid_px', 'ibid'), ('impact_ask_px', 'iask')]
    f32 = [('funding', 'funding'), ('premium', 'premium'), ('open_interest', 'oi'), ('day_ntl_vlm', 'vlm')]
    return df.select('t', 'coin', *[pl.col(a).cast(pl.Float64, strict=False).alias(b) for a, b in f64],
                     *[pl.col(a).cast(pl.Float32, strict=False).alias(b) for a, b in f32]).unique(['coin', 't']).sort(['coin', 't'])


def forward_ctx(settings):
    added = []
    today = datetime.now(timezone.utc).date()
    d = FORWARD_START
    while d < today:                                             # finished days only
        out = HLCTX / f'{d.isoformat()}.parquet'
        if not out.exists():
            df = live_day(settings, d.isoformat())
            if df is not None and len(df):
                tmp = out.with_suffix('.tmp')
                to_hlctx(df).write_parquet(tmp, compression='zstd')
                tmp.replace(out)
                added.append(d.isoformat())
        d += timedelta(days=1)
    return added


def binance_metrics_recent(settings, months_back=1):
    """Rebuild the current and previous month of Binance metrics from daily archive files."""
    from .harvest import _zip_frame
    from concurrent.futures import ThreadPoolExecutor
    root = settings.home / 'bn_metrics'
    today = datetime.now(timezone.utc).date()
    first = (today.replace(day=1) - timedelta(days=28 * months_back)).replace(day=1)
    syms = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    BV = 'https://data.binance.vision/data/futures/um/daily/metrics'

    def one(sym):
        days = [first + timedelta(days=i) for i in range((today - first).days)]
        frames = {}
        for x in days:
            f = _zip_frame(f'{BV}/{sym}/{sym}-metrics-{x.isoformat()}.zip', 'metrics')
            if f is not None and len(f):
                frames.setdefault(x.strftime('%Y-%m'), []).append(f)
        for ym, fs in frames.items():
            pl.concat(fs, how='diagonal_relaxed').write_parquet(root / sym / f'{ym}.parquet', compression='zstd')
        return sym, sum(len(v) for v in frames.values())

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(one, syms))


def refresh(settings):
    t0 = time.time()
    rep = {'forward_ctx_days_added': forward_ctx(settings)}
    try:
        got = binance_metrics_recent(settings)
        rep['binance_metric_days'] = sum(got.values())
    except Exception as e:
        rep['binance_error'] = repr(e)[:300]
    rep['panel'] = build_panel(settings)
    try:
        from .macro_data import build as build_macro
        from ..lab.macro import MacroStore, write_dev_view as macro_dev_view
        m = build_macro(settings)
        rep['macro'] = {k: m[k] for k in ('rows', 'symbols', 'missing')}
        MacroStore(settings).splits(); macro_dev_view(settings)
    except Exception as e:
        rep['macro_error'] = repr(e)[:300]
    try:
        from .polymarket import build as build_pred
        from ..lab.pred import PredStore, write_dev_view as pred_dev_view
        rep['pred'] = build_pred(settings)
        PredStore(settings).splits(); pred_dev_view(settings)
    except Exception as e:
        rep['pred_error'] = repr(e)[:300]
    try:
        from ..lab.carry import write_dev_view as carry_dev_view
        carry_dev_view(settings)
    except Exception as e:
        rep['carry_view_error'] = repr(e)[:300]
    from ..lab.perps import write_dev_view, PerpsStore
    PerpsStore(settings).splits()
    write_dev_view(settings)
    rep['seconds'] = round(time.time() - t0, 1)
    (settings.home / 'refresh_status.json').write_text(__import__('json').dumps({'updated': time.time(), **rep}, default=str, indent=1))
    return rep


if __name__ == '__main__':
    from ..config import SETTINGS
    print(refresh(SETTINGS))
