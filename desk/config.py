"""Desk OS v3 configuration. One place for paths, venue costs, splits and org scale."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


def _key():
    key = os.getenv('OPENCODE_API_KEY') or os.getenv('MIMO_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
    return key


def _provider() -> str:
    prov = os.getenv('LLM_PROVIDER', '').strip().lower()
    if prov:
        return prov
    if os.getenv('OPENCODE_API_KEY'):
        return 'opencode'
    if os.getenv('MIMO_API_KEY') and 'xiaomimimo' in os.getenv('MIMO_BASE_URL', 'xiaomimimo'):
        return 'mimo'
    return 'opencode'


def _base_url() -> str:
    for env_var in ('LLM_BASE_URL', 'OPENCODE_BASE_URL'):
        val = os.getenv(env_var, '').strip()
        if val:
            return val
    if _provider() == 'opencode':
        mimo_url = os.getenv('MIMO_BASE_URL', '').strip()
        if mimo_url and 'xiaomimimo.com' not in mimo_url:
            return mimo_url
        return 'https://opencode.ai/zen/v1'
    return os.getenv('MIMO_BASE_URL', 'https://token-plan-ams.xiaomimimo.com/v1').strip()


def _default_model() -> str:
    m = os.getenv('MODEL_NAME', '').strip()
    if _provider() == 'opencode':
        if m and not m.startswith('mimo-'):
            return m
        fm = os.getenv('DESK_FLASH_MODEL', '').strip()
        if fm and not fm.startswith('mimo-'):
            return fm
        return 'deepseek-v4.1-flash-free'
    return m or 'mimo-v2.6-pro'


def _default_flash_model() -> str:
    fm = os.getenv('DESK_FLASH_MODEL', '').strip()
    if _provider() == 'opencode':
        if fm and not fm.startswith('mimo-'):
            return fm
        m = os.getenv('MODEL_NAME', '').strip()
        if m and not m.startswith('mimo-'):
            return m
        return 'deepseek-v4.1-flash-free'
    return fm or 'mimo-v2.6-flash'


def _default_data_dir() -> Path:
    env_dir = os.getenv('OKX_DATA_DIR', '').strip()
    if env_dir and not (len(env_dir) > 1 and env_dir[1] == ':' and os.name != 'nt'):
        p = Path(env_dir)
        return p if p.is_absolute() else ROOT / p
    win_drive = Path(r'G:\Mon Drive\OKX_L2_BTC')
    return win_drive if (os.name == 'nt' and win_drive.exists()) else ROOT / 'data' / 'okx_l2_btc'


def _default_home() -> Path:
    env_res = os.getenv('RESULTS_DIR', '').strip()
    if env_res and env_res not in ('results', 'results/'):
        p = Path(env_res)
        return p if p.is_absolute() else ROOT / p
    return ROOT / 'results' / 'v3'


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
    data_dir: Path = field(default_factory=_default_data_dir)
    home: Path = field(default_factory=_default_home)
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
    min_dsr: float = 0.95               # deflated vs. the cumulative trial ledger (per family and global); plus bootstrap > 0
    min_positive_weeks: float = 0.55
    max_drawdown: float = 0.20
    max_top5_day_share: float = 0.5
    min_holdout_sharpe: float = 0.5
    # overfitting gates (validation): PBO = max(CSCV, CPCV) over {submitted params} + the +/-20% parameter grid
    max_pbo: float = 0.50               # reject when PBO > 0.50 (selected config does worse than the median OOS)
    pbo_fail_closed: bool = True        # strict default: no grid (no numeric params) / too few days -> PBO undefined -> reject
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    cpcv_purge: int = 5                 # bars purged on both sides of each test group
    cpcv_embargo_frac: float = 0.01     # embargo after each test group, fraction of T (ceil)
    # LLM / organization (defaults to OpenCode Zen `https://opencode.ai/zen/v1` + `deepseek-v4.1-flash`)
    provider: str = field(default_factory=_provider)
    model: str = field(default_factory=_default_model)
    flash_model: str = field(default_factory=_default_flash_model)
    # roles that run on the fast/cheap model (default: everyone). Pro for the judgment layer:
    # DESK_FLASH_ROLES=researcher,toolsmith
    flash_roles: tuple = tuple(r.strip() for r in os.getenv('DESK_FLASH_ROLES', 'cio,lead,researcher,officer,pm,execution,librarian,toolsmith').split(',') if r.strip())
    api_key: str = field(default_factory=_key)
    base_url: str = field(default_factory=_base_url)
    max_concurrency: int = int(os.getenv('DESK_MAX_CONCURRENCY', '24'))
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
        db_env = os.getenv('DATABASE_PATH', '').strip()
        if db_env and db_env.endswith('.sqlite3') and self.home == _default_home():
            p = Path(db_env)
            return p if p.is_absolute() else ROOT / p
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
