"""Configuration settings for the AI BI platform."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ENV_FILE = BASE_DIR / ".env"

SQLITE_URL = f"sqlite:///{BASE_DIR / 'data' / 'business_intel.db'}"

MODEL_SETTINGS = {
    "forecast_horizon_days": 30,
    "anomaly_threshold": 3.0,
}
