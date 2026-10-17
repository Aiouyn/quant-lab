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

# Check Desktop key file
desktop_key_path = Path.home() / "Desktop" / "mimo key.txt"
if desktop_key_path.exists() and not os.getenv("MIMO_API_KEY"):
    try:
        raw_lines = [l.strip() for l in desktop_key_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        tp_keys = [l for l in raw_lines if l.startswith("tp-") or l.startswith("ttp-")]
        sk_keys = [l for l in raw_lines if l.startswith("sk-")]
        if tp_keys:
            os.environ["MIMO_API_KEY"] = tp_keys[0]
        elif sk_keys:
            os.environ["MIMO_API_KEY"] = sk_keys[0]
    except Exception:
        pass

# API Keys & Endpoints
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

# Determine active provider & key
ACTIVE_KEY = ""
ACTIVE_BASE_URL = ""
ACTIVE_PROVIDER = ""

if MIMO_API_KEY:
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

DEFAULT_MODEL = os.getenv("MODEL_NAME", "mimo-v2.6-pro" if ACTIVE_PROVIDER == "mimo" else "gpt-4o")

# Storage Paths
DATA_DIR = Path(os.getenv("OKX_DATA_DIR", r"G:\Mon Drive\OKX_L2_BTC"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "database" / "research_ledger.db")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results")))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
