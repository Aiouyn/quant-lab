from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv('OKX_DATA_DIR', r'G:\Mon Drive\OKX_L2_BTC'))
    home: Path = ROOT / 'results' / 'desk'
    initial_equity: float = 10000.0
    allocation_fraction: float = 0.25
    maker_fee_bps: float = 1.5
    taker_fee_bps: float = 4.5
    slippage_bps: float = 0.5
    latency_ms: int = 1000
    max_gap_ms: int = 5000
    max_spread_bps: float = 5.0
    max_drawdown_fraction: float = 0.05
    min_validation_days: int = 20
    min_holdout_days: int = 10
    min_trades: int = 30
    max_agent_steps: int = 8
    max_completion_tokens: int = 18000
    max_campaign_tokens: int | None = None
    request_timeout: int = 900
    provider_stream: bool = True
    worker_timeout: int = 30
    max_strategy_weight: float = 0.35
    min_forward_days: int = 7
    min_forward_trades: int = 50
    min_forward_coverage: float = 0.95
    max_forward_age_days: int = 14
    shadow_participation_fraction: float = 0.1
    shadow_max_sweep_bps: float = 5.0

    def shadow_dict(self):
        from .liquidity import validate_policy
        return validate_policy({'model':'displayed-depth-ioc-v1',
            'participation_fraction':self.shadow_participation_fraction,
            'max_sweep_bps':self.shadow_max_sweep_bps})

    def execution_dict(self):
        keys = ('initial_equity','allocation_fraction','maker_fee_bps','taker_fee_bps','slippage_bps','latency_ms','max_gap_ms','max_spread_bps','max_drawdown_fraction')
        return {k: getattr(self,k) for k in keys}
