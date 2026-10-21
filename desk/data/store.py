"""Feature store access with a frozen chronological split and a physical dev-only view.

Splits are frozen once (store/splits.json) over the full archive inventory:
  dev (60%) -> research, val (25%) -> deterministic validation, holdout (15%) -> sealed, once per strategy,
  forward -> any day added to the archive after freezing (a rolling, never-seen holdout).
Researcher sandboxes receive `store/views/dev`, a directory of hard links to dev files only.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import polars as pl

RESOLUTIONS = ('1s', '10s', '1m')
SPLITS = ('dev', 'val', 'holdout', 'forward')


class SplitViolation(PermissionError):
    pass


class FeatureStore:
    def __init__(self, settings=None, root: Path | None = None, allowed=SPLITS):
        if settings is None:
            from ..config import SETTINGS as settings
        self.settings = settings
        self.root = Path(root) if root else settings.store
        self.allowed = set(allowed)

    # ------------------------------------------------------------ splits
    def _inventory_dates(self):
        try:
            from .ingest import archive_inventory
            return [r['date'] for r in archive_inventory(self.settings.data_dir)]
        except Exception:
            return sorted(p.stem for p in (self.root / '1s').glob('*.parquet'))

    def splits(self, refresh=False):
        f = self.root / 'splits.json'
        if f.exists():
            sp = json.loads(f.read_text())
            if refresh:
                frozen = set(sp['dev'] + sp['val'] + sp['holdout'] + sp['forward'])
                last = max(sp['holdout'])
                new = [d for d in self._inventory_dates() if d not in frozen and d > last]
                if new:
                    sp['forward'] = sorted(sp['forward'] + new)
                    f.write_text(json.dumps(sp, indent=1))
        else:
            if not (self.root / 'meta').exists() and not refresh:
                raise FileNotFoundError('feature store not built yet: run `python -m desk data`')
            dates = self._inventory_dates()
            n = len(dates)
            a = int(n * self.settings.dev_fraction)
            b = int(n * (self.settings.dev_fraction + self.settings.val_fraction))
            sp = {'created': time.time(), 'dev': dates[:a], 'val': dates[a:b], 'holdout': dates[b:], 'forward': []}
            self.root.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(sp, indent=1))
        ingested = {p.stem for p in (self.root / '1s').glob('*.parquet')} if (self.root / '1s').exists() else set()
        sp['summary'] = {k: {'days': len(sp[k]), 'ingested': len([d for d in sp[k] if d in ingested]),
                             'first': sp[k][0] if sp[k] else None, 'last': sp[k][-1] if sp[k] else None} for k in SPLITS}
        return sp

    def split_of(self, date):
        sp = self.splits()
        for k in SPLITS:
            if date in sp[k]:
                return k
        return None

    def days(self, split='dev', ingested_only=True):
        if split not in self.allowed:
            raise SplitViolation(f'{split} data is not available in this context')
        sp = self.splits()
        ds = sp[split]
        if ingested_only:
            ds = [d for d in ds if (self.root / '1s' / f'{d}.parquet').exists()]
        return ds

    # ------------------------------------------------------------ loading
    def path(self, date, resolution='1s'):
        split = self.split_of(date)
        if split not in self.allowed:
            raise SplitViolation(f'{date} belongs to {split}; not available in this context')
        return self.root / resolution / f'{date}.parquet'

    def load(self, dates, resolution='1s', columns=None, with_funding=True):
        if resolution not in RESOLUTIONS:
            raise ValueError(f'resolution must be one of {RESOLUTIONS}')
        if isinstance(dates, str):
            dates = [dates]
        paths = [self.path(d, resolution) for d in dates]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError('no ingested data for the requested dates')
        lf = pl.scan_parquet([str(p) for p in paths])
        if columns:
            keep = ['ts'] + [c for c in columns if c != 'ts']
            if 'mid' not in keep:
                keep.append('mid')
            if 'valid' not in keep:
                keep.append('valid')
            lf = lf.select([c for c in keep if c in lf.collect_schema().names()])
        df = lf.collect().sort('ts')
        if with_funding:
            df = attach_funding(df, self.root)
        return df

    # ------------------------------------------------------------ dev view
    def build_dev_view(self):
        """Hard-link dev files into store/views/dev so sandboxes can't reach val/holdout files."""
        view = self.root / 'views' / 'dev'
        sp = self.splits()
        for res in RESOLUTIONS + ('tape',):
            if res == 'tape' and not (self.root / 'tape').exists():
                continue
            (view / res).mkdir(parents=True, exist_ok=True)
            for d in sp['dev']:
                src = self.root / res / f'{d}.parquet'
                dst = view / res / f'{d}.parquet'
                if src.exists() and dst.exists() and not os.path.samefile(src, dst):
                    dst.unlink()                     # source was rewritten (e.g. bar schema change): relink
                if src.exists() and not dst.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        import shutil
                        shutil.copy2(src, dst)
        for name in ('funding_hl.parquet',):
            src = self.root / name
            if src.exists():
                dst = view / name
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
        (view / 'splits.json').write_text(json.dumps({'created': sp['created'], 'dev': sp['dev'], 'val': [], 'holdout': [], 'forward': []}))
        (view / 'meta').mkdir(exist_ok=True)
        return view


def funding_series(root):
    """Hourly funding prints as (hour_ms, rate) for per-hour charging across data gaps, or None."""
    if root is None:
        return None
    f = Path(root) / 'funding_hl.parquet'
    if not f.exists():
        return None
    from ..lab.backtest import as_funding_series
    return as_funding_series(pl.read_parquet(f, columns=['hour', 'funding_rate']))


def attach_funding(df: pl.DataFrame, root: Path) -> pl.DataFrame:
    """Adds `funding_rate`: the most recent PAID hourly Hyperliquid rate (causal), and
    `funding_due`: 1 on rows that sit exactly on an hourly payment boundary."""
    f = Path(root) / 'funding_hl.parquet'
    if not f.exists() or 'ts' not in df.columns:
        return df
    fund = pl.read_parquet(f).select(pl.col('hour').alias('ts'), pl.col('funding_rate'))
    out = df.sort('ts').join_asof(fund.sort('ts'), on='ts', strategy='backward')
    return out.with_columns(pl.col('funding_rate').fill_null(0.0))
