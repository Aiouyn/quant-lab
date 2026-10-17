"""Desk OS v3 configuration. One place for paths, venue costs, splits and org scale."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


def _key():
    key = os.getenv('MIMO_API_KEY', '')
    desktop = Path.home() / 'Desktop' / 'mimo key.txt'
    if not key and desktop.exists():
        lines = [l.strip() for l in desktop.read_text(encoding='utf-8').splitlines() if l.strip()]
        key = next((l for l in lines if l.startswith(('tp-', 'ttp-', 'sk-'))), '')
    return key


@dataclass(frozen=True)
class Venue:
    """Hyperliquid perp costs (base tier). Research runs on OKX data, deploys on HL."""
    taker_bps: float = 4.5
    maker_bps: float = 1.5
    slippage_bps: float = 0.5          # extra taker slippage beyond crossing the spread
    latency_s: int = 1                 # decision -> fill delay (data is 1 Hz)
    funding_interval_s: int = 3600     # HL funding is hourly
    contract_btc: float = 0.01         # OKX BTC-USDT-SWAP contract value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv('OKX_DATA_DIR', r'G:\Mon Drive\OKX_L2_BTC'))
    home: Path = ROOT / 'results' / 'v3'
    store_dir: Path | None = None        # defaults to home/store; lets a run share the real store with its own ledger
    venue: Venue = field(default_factory=Venue)
    # chronological split of the archive frozen at first ingest
    dev_fraction: float = 0.60
    val_fraction: float = 0.25
    # backtest account
    initial_equity: float = 100_000.0
    max_leverage: float = 1.0           # |position notional| <= equity * max_leverage
    # validation gates
    min_val_trades: int = 40
    min_val_sharpe: float = 1.0         # annualized, net
    min_dsr: float = 0.80               # deflated vs. #strategies evaluated on validation; plus bootstrap > 0
    min_positive_weeks: float = 0.55
    max_drawdown: float = 0.20
    max_top5_day_share: float = 0.5
    min_holdout_sharpe: float = 0.5
    # LLM / organization
    model: str = os.getenv('MODEL_NAME', 'mimo-v2.6-pro')
    flash_model: str = os.getenv('DESK_FLASH_MODEL', 'mimo-v2.6-flash')
    # roles that run on the fast/cheap model (default: everyone). Pro for the judgment layer:
    # DESK_FLASH_ROLES=researcher,toolsmith
    flash_roles: tuple = tuple(r.strip() for r in os.getenv('DESK_FLASH_ROLES', 'cio,lead,researcher,officer,pm,execution,librarian,toolsmith').split(',') if r.strip())
    api_key: str = field(default_factory=_key)
    base_url: str = os.getenv('MIMO_BASE_URL', 'https://token-plan-ams.xiaomimimo.com/v1')
    max_concurrency: int = int(os.getenv('DESK_MAX_CONCURRENCY', '32'))
    start_concurrency: int = int(os.getenv('DESK_START_CONCURRENCY', '8'))
    request_timeout_s: int = 600
    max_completion_tokens: int = 16000
    context_budget_tokens: int = 40_000   # compaction trigger (estimated); input tokens dominate cost
    max_steps_per_session: int = 30
    # compute
    compute_workers: int = int(os.getenv('DESK_SANDBOXES', '2'))
    heavy_slots: int = int(os.getenv('DESK_JOB_SLOTS', '1'))
    sandbox_timeout_s: int = 300
    sandbox_memory_mb: int = int(os.getenv('DESK_SANDBOX_MB', '4000'))
    job_memory_mb: int = int(os.getenv('DESK_JOB_MB', '5000'))

    @property
    def store(self) -> Path:
        return Path(self.store_dir) if self.store_dir else self.home / 'store'

    @property
    def ledger(self) -> Path:
        return self.home / 'desk.sqlite3'

    @property
    def workspaces(self) -> Path:
        return self.home / 'workspaces'

    @property
    def artifacts(self) -> Path:
        return self.home / 'artifacts'

    def ensure(self):
        for p in (self.home, self.store, self.workspaces, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def public(self):
        d = asdict(self)
        d.pop('api_key', None)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}


SETTINGS = Settings()
