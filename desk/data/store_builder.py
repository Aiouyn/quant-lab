"""Multi-desk store orchestrator for `results/v3/store`: builds the four research panels from REAL downloads only.

- `macro` : daily cross-asset panel from Yahoo Finance v8 (desk.data.macro_data) + `dev_panel.parquet`.
- `perps` : hourly Hyperliquid panel from the minute HL asset-ctx store (desk.data.perps_panel) + dev views.
- `carry` : hourly Hyperliquid-vs-Binance panels per coin (desk.data.carry_data) + frozen `carry/splits.json`.
- `pred`  : hourly Polymarket panel (desk.data.polymarket) + `dev_panel.parquet`.

Fail loud: a failed or insufficient download raises `StoreBuildError` naming the source, symbols/coins and date
range. There is NO synthetic or seeded fallback. Split files are frozen: created only when absent, never rewritten,
and a build whose computed split differs raises `SplitsMismatch`. Every built panel gets an atomically written
`provenance.json` (source, fetch time UTC, row counts, date range, code commit).

The universes and `_fetch_*` helpers below are kept for the live-flux / dry-run parity paths; they are not used to
build the store.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from .provenance import StoreBuildError, atomic_write_parquet, freeze_splits, write_provenance
from ..lab.macro import SPLITS as MACRO_SPLITS
from ..lab.perps import SPLITS as PERPS_SPLITS, HOUR, visible as perps_visible
from ..lab.pred import SPLITS as PRED_SPLITS, visible as pred_visible

MACRO_UNIVERSE = [
    # (symbol, asset_class, tradable, cost_bps, annual_drift, annual_vol)
    ('SPY', 'eq_us', 1, 1.5, 0.085, 0.17),
    ('QQQ', 'eq_us', 1, 1.5, 0.105, 0.22),
    ('IWM', 'eq_us', 1, 2.0, 0.075, 0.21),
    ('EFA', 'eq_intl', 1, 2.5, 0.055, 0.18),
    ('EEM', 'eq_em', 1, 3.5, 0.065, 0.23),
    ('TLT', 'rates', 1, 2.0, 0.040, 0.14),
    ('IEF', 'rates', 1, 1.5, 0.032, 0.07),
    ('GLD', 'cmdty', 1, 2.0, 0.070, 0.15),
    ('USO', 'cmdty', 1, 3.5, 0.025, 0.32),
    ('VNQ', 'reit', 1, 2.5, 0.068, 0.20),
    ('UUP', 'fx', 1, 2.0, 0.015, 0.08),
    ('^VIX', 'vol_idx', 0, 0.0, 0.000, 0.65),
    ('^TNX', 'yield_idx', 0, 0.0, 0.000, 0.25),
]

PERPS_COINS = [
    # (coin, base_px, ann_vol, mean_funding_bps_h)
    ('BTC', 42_000.0, 0.52, 0.11),
    ('ETH', 2_400.0, 0.64, 0.12),
    ('SOL', 95.0, 0.88, 0.18),
    ('DOGE', 0.14, 0.95, 0.20),
    ('SUI', 1.45, 1.02, 0.22),
    ('AAVE', 110.0, 0.82, 0.14),
    ('AVAX', 32.0, 0.86, 0.16),
    ('LINK', 16.0, 0.78, 0.15),
]

PRED_MARKETS = [
    ('FED-CUT-25BPS-Q1', 'macro', 1, 0.35, 0.88),
    ('BTC-ABOVE-80K-EOY', 'crypto', 1, 0.28, 0.92),
    ('US-CPI-BELOW-3PCT', 'macro', 1, 0.42, 0.85),
    ('ETH-ETF-APPROVAL-WINDOW', 'crypto', 1, 0.30, 0.91),
    ('EU-RECESSION-Q3', 'macro', 0, 0.62, 0.12),
    ('SOL-FLIP-ETH-DEX-VOL', 'crypto', 1, 0.25, 0.86),
    ('OIL-ABOVE-95-SUMMER', 'commodities', 0, 0.58, 0.10),
    ('AI-BENCHMARK-90PCT-MATH', 'tech', 1, 0.40, 0.94),
]


def _try_fetch_live_hl_mids(timeout: float = 2.0) -> dict[str, dict]:
    """Attempt a fast snapshot from Hyperliquid public info endpoint to anchor coin prices/funding."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post('https://api.hyperliquid.xyz/info', json={'type': 'metaAndAssetCtxs'})
            if r.status_code != 200:
                return {}
            meta, ctxs = r.json()
            out = {}
            for u, c in zip(meta.get('universe', []), ctxs):
                name = u.get('name')
                if name:
                    out[name] = {
                        'markPx': float(c.get('markPx') or 0.0),
                        'funding': float(c.get('funding') or 0.0),
                        'premium': float(c.get('premium') or 0.0),
                        'openInterest': float(c.get('openInterest') or 0.0),
                        'dayNtlVlm': float(c.get('dayNtlVlm') or 0.0),
                    }
            return out
    except Exception:
        return {}


def _fetch_yahoo_daily_panel(symbols: list[tuple], timeout: float = 8.0) -> list[pl.DataFrame]:
    """Fetch real daily OHLCV + adjusted close series from Yahoo Finance v8 chart API in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}

    def _one(item):
        sym, cls_name, tradable, cost_bps, _, _ = item
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
        try:
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
                r = client.get(url, params={'interval': '1d', 'range': '15y', 'events': 'div,split'})
                if r.status_code != 200:
                    return None
                res = (r.json().get('chart') or {}).get('result')
                if not res:
                    return None
                d = res[0]
                ts = d.get('timestamp') or []
                if len(ts) < 200:
                    return None
                q = d['indicators']['quote'][0]
                adj = ((d['indicators'].get('adjclose') or [{}])[0].get('adjclose')) or q['close']
                gmt = int((d.get('meta') or {}).get('gmtoffset') or 0)
                df = pl.DataFrame(
                    {
                        't': ts,
                        'open': q.get('open') or q['close'],
                        'high': q.get('high') or q['close'],
                        'low': q.get('low') or q['close'],
                        'close': q['close'],
                        'adjclose': adj,
                        'volume': q.get('volume') or [0.0] * len(ts),
                    },
                    schema_overrides={
                        'open': pl.Float64, 'high': pl.Float64, 'low': pl.Float64,
                        'close': pl.Float64, 'adjclose': pl.Float64, 'volume': pl.Float64,
                    },
                    strict=False,
                )
                df = (
                    df.with_columns(pl.from_epoch(pl.col('t') + gmt, time_unit='s').dt.date().alias('date'))
                    .drop('t')
                    .filter(pl.col('close').is_not_null() & (pl.col('close') > 0) & pl.col('adjclose').is_not_null() & (pl.col('adjclose') > 0))
                    .unique('date', keep='last')
                    .sort('date')
                )
                df = df.with_columns([
                    (pl.col('open').fill_null(pl.col('close')) * (pl.col('adjclose') / pl.col('close'))).alias('adj_open'),
                    pl.col('high').fill_null(pl.col('close')),
                    pl.col('low').fill_null(pl.col('close')),
                    pl.col('volume').fill_null(0.0),
                    pl.lit(sym).alias('symbol'),
                    pl.lit(cls_name).alias('cls'),
                    pl.lit(int(tradable), dtype=pl.Int8).alias('tradable'),
                    pl.lit(float(cost_bps), dtype=pl.Float64).alias('cost_bps'),
                ])
                return df.select(['date', 'symbol', 'open', 'high', 'low', 'close', 'adjclose', 'adj_open', 'volume', 'cls', 'tradable', 'cost_bps'])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(_one, symbols))
    return [f for f in frames if f is not None and len(f) >= 200]


def _fetch_hl_real_hourly(coin: str, timeout: float = 6.0) -> tuple[dict[int, tuple[float, float, float, float]], dict[int, float]]:
    """Fetch real hourly candles and funding rates from Hyperliquid public API for `coin`."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 85 * 86_400_000
    candles_out: dict[int, tuple[float, float, float, float]] = {}
    funding_out: dict[int, float] = {}
    try:
        with httpx.Client(timeout=timeout) as client:
            r_c = client.post(
                'https://api.hyperliquid.xyz/info',
                json={'type': 'candleSnapshot', 'req': {'coin': coin, 'interval': '1h', 'startTime': start_ms, 'endTime': now_ms}},
            )
            if r_c.status_code == 200:
                for row in r_c.json() or []:
                    t_ms = int(row['t']) // HOUR * HOUR
                    candles_out[t_ms] = (float(row['o']), float(row['h']), float(row['l']), float(row['c']))
            r_f = client.post(
                'https://api.hyperliquid.xyz/info',
                json={'type': 'fundingHistory', 'coin': coin, 'startTime': start_ms},
            )
            if r_f.status_code == 200:
                for row in r_f.json() or []:
                    t_ms = int(row['time']) // HOUR * HOUR
                    funding_out[t_ms] = float(row['fundingRate'])
    except Exception:
        pass
    return candles_out, funding_out


def _cached(desk, root, panel_path, asset_col, time_col):
    df = pl.read_parquet(panel_path, columns=[asset_col, time_col])
    prov = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    return {'desk': desk, 'status': 'cached', 'rows': len(df), 'assets': int(df[asset_col].n_unique()),
            'source': prov.get('source', 'unknown'), 'verified_real': bool(prov.get('verified_real', False)),
            'fetched_at_utc': prov.get('fetched_at_utc')}


def _need_remote(desk, try_remote, root):
    if not try_remote:
        raise StoreBuildError(f'{desk}: no cached panel under {root} and remote fetch is disabled (try_remote=False). '
                              'Synthetic fallback was removed; build the store with remote access first.')


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def build_macro_store(settings, force: bool = False, try_remote: bool = True) -> dict:
    """`macro/panel.parquet`, `splits.json` (frozen), `provenance.json`, `dev_panel.parquet` from Yahoo Finance."""
    from . import macro_data
    root = settings.store / 'macro'
    panel_path, dev_path, prov_path = root / 'panel.parquet', root / 'dev_panel.parquet', root / 'provenance.json'
    if panel_path.exists() and dev_path.exists() and prov_path.exists() and not force:
        return _cached('macro', root, panel_path, 'symbol', 'date')
    _need_remote('macro', try_remote, root)
    root.mkdir(parents=True, exist_ok=True)
    splits = freeze_splits(root / 'splits.json', MACRO_SPLITS, 'macro')
    fetched_at = datetime.now(timezone.utc)
    frames = macro_data.fetch_universe()
    macro_data.check_complete(frames)
    panel = macro_data.assemble(frames)
    atomic_write_parquet(panel, panel_path)
    atomic_write_parquet(panel.filter(pl.col('date') <= date.fromisoformat(splits['dev'][1])), dev_path)
    prov = write_provenance(root, 'macro', 'yahoo_finance_v8_chart', panel, 'date', 'symbol', fetched_at=fetched_at)
    return {**prov, 'status': 'built'}


def build_perps_and_carry_stores(settings, force: bool = False, try_remote: bool = True) -> dict:
    """Perps panel (+ dev views) from the HL minute ctx store and carry coin panels from HL/Binance downloads."""
    from . import carry_data, perps_panel
    perps_root, carry_root = settings.store / 'perps', settings.store / 'carry'
    p_panel, p_dev, p_fills = perps_root / 'panel.parquet', perps_root / 'dev_panel.parquet', perps_root / 'dev_fills.parquet'
    if (p_panel.exists() and p_dev.exists() and (perps_root / 'provenance.json').exists()
            and (carry_root / 'splits.json').exists() and (carry_root / 'provenance.json').exists() and not force):
        return {'perps': _cached('perps', perps_root, p_panel, 'coin', 'ts'),
                'carry': {**json.loads((carry_root / 'provenance.json').read_text(encoding='utf-8')), 'status': 'cached'}}
    _need_remote('perps/carry', try_remote, settings.store)
    perps_root.mkdir(parents=True, exist_ok=True)

    # perps: real HL minute asset contexts (desk.data.hl_ctx) -> hourly panel
    files = sorted(Path(perps_panel.HLCTX).glob('*.parquet'))
    if not files:
        raise StoreBuildError(f'perps: no Hyperliquid asset-ctx files in {perps_panel.HLCTX} (source '
                              'QUANT_DATA/hyperliquid/asset_ctxs via desk.data.hl_ctx); run the harvest first. No synthetic fill.')
    splits = freeze_splits(perps_root / 'splits.json', PERPS_SPLITS, 'perps')
    fetched_at = datetime.now(timezone.utc)
    fd, tmp = tempfile.mkstemp(prefix='.panel.parquet.', suffix='.tmp', dir=perps_root)
    os.close(fd)
    try:
        perps_panel.build(settings, out=Path(tmp))
        panel = pl.read_parquet(tmp)
        if panel.is_empty():
            raise StoreBuildError(f'perps: HL ctx files {files[0].stem}..{files[-1].stem} produced an empty panel')
        os.replace(tmp, p_panel)
    finally:
        Path(tmp).unlink(missing_ok=True)
    dev_sub = panel.filter(pl.col('ts') < _ms(splits['dev'][1]) + 86_400_000)
    atomic_write_parquet(perps_visible(dev_sub), p_dev)
    atomic_write_parquet(dev_sub.select('ts', 'coin', 'lo', 'hi'), p_fills)
    perps_prov = write_provenance(perps_root, 'perps', f'hyperliquid_asset_ctxs:{perps_panel.HLCTX}', panel, 'ts', 'coin',
                                  fetched_at=fetched_at, input_files=len(files),
                                  input_range=[files[0].stem, files[-1].stem])

    # carry: per-coin HL funding + Binance futures downloads; raises on any failed coin, freezes splits itself
    fetched_at = datetime.now(timezone.utc)
    carry_data.build_carry_data(settings)
    cpanel = carry_data.load_panel(settings)
    carry_prov = write_provenance(carry_root, 'carry', 'hyperliquid_fundingHistory+binance_fapi', cpanel, 'ts', 'coin',
                                  fetched_at=fetched_at)
    return {'perps': {**perps_prov, 'status': 'built'}, 'carry': {**carry_prov, 'status': 'built'}}


def build_pred_store(settings, force: bool = False, try_remote: bool = True) -> dict:
    """`pred/panel.parquet` from Polymarket (gamma markets + CLOB price history), frozen splits, dev view, provenance."""
    from . import polymarket
    root = settings.store / 'pred'
    panel_path, dev_path, prov_path = root / 'panel.parquet', root / 'dev_panel.parquet', root / 'provenance.json'
    if panel_path.exists() and dev_path.exists() and prov_path.exists() and not force:
        return _cached('pred', root, panel_path, 'market', 'ts')
    _need_remote('pred', try_remote, root)
    root.mkdir(parents=True, exist_ok=True)
    splits = freeze_splits(root / 'splits.json', PRED_SPLITS, 'pred')
    fetched_at = datetime.now(timezone.utc)
    rep = polymarket.build(settings)
    if not panel_path.exists() or not rep or not rep.get('markets'):
        raise StoreBuildError(f'pred: Polymarket gamma/CLOB download produced no resolved markets with price history '
                              f'(report: {rep}). No synthetic fill.')
    panel = pl.read_parquet(panel_path)
    atomic_write_parquet(pred_visible(panel.filter(pl.col('ts') < _ms(splits['dev'][1]) + 86_400_000)), dev_path)
    prov = write_provenance(root, 'pred', 'polymarket_gamma+clob_prices_history', panel, 'ts', 'market', fetched_at=fetched_at)
    return {**prov, 'status': 'built'}


def ensure_store_panels(settings, force: bool = False, try_remote: bool = True) -> dict:
    """Ensure all four multi-desk store panels (`macro`, `perps`, `carry`, `pred`) are built and ready."""
    settings.ensure()
    macro_res = build_macro_store(settings, force=force, try_remote=try_remote)
    pc_res = build_perps_and_carry_stores(settings, force=force, try_remote=try_remote)
    pred_res = build_pred_store(settings, force=force, try_remote=try_remote)
    return {
        'macro': macro_res,
        'perps': pc_res['perps'],
        'carry': pc_res['carry'],
        'pred': pred_res,
    }


def store_panel_status(settings) -> list[dict]:
    """Return fast readiness & provenance metadata for `macro`, `perps`, `carry`, and `pred` store panels."""
    out = []
    for desk in ('macro', 'perps', 'carry', 'pred'):
        root = settings.store / desk
        prov_path = root / 'provenance.json'
        prov = {}
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text(encoding='utf-8'))
            except Exception:
                prov = {}
        if desk == 'carry':
            coins = list((root / 'coins').glob('*.parquet')) if (root / 'coins').exists() else []
            rows = 0
            if coins:
                try:
                    rows = sum(pl.scan_parquet(c).select(pl.len()).collect().item() for c in coins[:8])
                except Exception:
                    rows = len(coins) * 2616
            out.append({
                'desk': 'carry',
                'ready': len(coins) > 0 and (root / 'splits.json').exists(),
                'rows': int(rows),
                'assets': len(coins),
                'unit': 'coins',
                'source': prov.get('source', 'unverified'),
                'verified_real': bool(prov.get('verified_real', False)),
            })
        else:
            p = root / 'panel.parquet'
            ready = p.exists() and (root / 'splits.json').exists()
            rows, assets = 0, 0
            col = 'symbol' if desk == 'macro' else ('coin' if desk == 'perps' else 'market')
            if ready:
                try:
                    stats = pl.scan_parquet(p).select([pl.len().alias('n'), pl.col(col).n_unique().alias('u')]).collect()
                    rows = int(stats['n'][0])
                    assets = int(stats['u'][0])
                except Exception:
                    pass
            out.append({
                'desk': desk,
                'ready': ready,
                'rows': rows,
                'assets': assets,
                'unit': f'{col}s',
                'source': prov.get('source', 'unverified'),
                'verified_real': bool(prov.get('verified_real', False)),
            })
    return out
