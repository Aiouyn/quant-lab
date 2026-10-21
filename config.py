"""
Central configuration for Quant Lab
Loads environment variables and sets up paths and model connections.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# API Keys & Endpoints
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1"))

MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")

# Automatically select endpoint based on key type:
# - Token Plan subscription keys start with 'tp-' or 'ttp-' -> https://token-plan-ams.xiaomimimo.com/v1
# - Pay-as-you-go keys start with 'sk-' -> https://api.xiaomimimo.com/v1
if MIMO_API_KEY.startswith("tp-") or MIMO_API_KEY.startswith("ttp-"):
    DEFAULT_MIMO_URL = "https://token-plan-ams.xiaomimimo.com/v1"
else:
    DEFAULT_MIMO_URL = "https://api.xiaomimimo.com/v1"

MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", DEFAULT_MIMO_URL)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Determine active provider & key (defaults to OpenCode + deepseek-v4.1-flash-free)
ACTIVE_KEY = OPENCODE_API_KEY
ACTIVE_BASE_URL = OPENCODE_BASE_URL
ACTIVE_PROVIDER = "opencode"

if OPENCODE_API_KEY or os.getenv("LLM_PROVIDER", "opencode").lower() == "opencode":
    ACTIVE_KEY = OPENCODE_API_KEY or MIMO_API_KEY
    ACTIVE_BASE_URL = OPENCODE_BASE_URL
    ACTIVE_PROVIDER = "opencode"
elif MIMO_API_KEY:
    ACTIVE_KEY = MIMO_API_KEY
    ACTIVE_BASE_URL = MIMO_BASE_URL
    ACTIVE_PROVIDER = "mimo"
elif OPENROUTER_API_KEY:
    ACTIVE_KEY = OPENROUTER_API_KEY
    ACTIVE_BASE_URL = OPENROUTER_BASE_URL
    ACTIVE_PROVIDER = "openrouter"
elif OPENAI_API_KEY:
    ACTIVE_KEY = OPENAI_API_KEY
    ACTIVE_BASE_URL = OPENAI_BASE_URL
    ACTIVE_PROVIDER = "openai"

_raw_model = os.getenv("MODEL_NAME", "").strip()
if ACTIVE_PROVIDER == "opencode" and (not _raw_model or _raw_model.startswith("mimo-")):
    DEFAULT_MODEL = os.getenv("DESK_FLASH_MODEL", "deepseek-v4.1-flash-free")
else:
    DEFAULT_MODEL = _raw_model or ("mimo-v2.6-pro" if ACTIVE_PROVIDER == "mimo" else "gpt-4o")

# Storage Paths
DATA_DIR = Path(os.getenv("OKX_DATA_DIR", str(BASE_DIR / "data" / "okx_l2_btc")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results" / "v3")))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(RESULTS_DIR / "desk.sqlite3")))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
