"""
config.py — Configuration centralisée via .env
Rétrocompatible avec l'ancien format (FRED_KEY, etc.)
"""
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# ── Rétrocompatibilité ──────────────────────────────────
FRED_KEY       = os.getenv("FRED_API_KEY", "demo")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
MAX_RISK       = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 5.0))
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", 6))
PORT           = int(os.getenv("PORT", 5001))
DEBUG          = os.getenv("DEBUG", "true").lower() == "true"


class _Settings:
    """Accès objet aux settings (sans dépendance pydantic-settings)."""

    FRED_API_KEY: str             = FRED_KEY
    XAI_API_KEY: str              = os.getenv("XAI_API_KEY", "")
    XAI_MODEL: str                = os.getenv("XAI_MODEL", "grok-3")
    XAI_LIVE_SEARCH_ENABLED: bool = os.getenv("XAI_LIVE_SEARCH_ENABLED", "true").lower() == "true"
    HELIUS_API_KEY: str           = os.getenv("HELIUS_API_KEY", "")
    BIRDEYE_API_KEY: str          = os.getenv("BIRDEYE_API_KEY", "")
    TELEGRAM_BOT_TOKEN: str       = TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID: str         = TELEGRAM_CHAT
    DRY_RUN: bool                 = os.getenv("DRY_RUN", "true").lower() == "true"
    MIN_SCORE_ALERT: int          = int(os.getenv("MIN_SCORE_ALERT", 78))
    BEEP_ON_ALERT: bool           = os.getenv("BEEP_ON_ALERT", "true").lower() == "true"
    MAX_RISK_PER_TRADE: float     = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))
    MAX_PORTFOLIO_RISK: float     = float(os.getenv("MAX_PORTFOLIO_RISK", 6.0))
    MAX_POSITIONS: int            = int(os.getenv("MAX_POSITIONS", 6))
    HARD_STOP_LOSS_PCT: float     = float(os.getenv("HARD_STOP_LOSS_PCT", 12.0))
    DAILY_LOSS_LIMIT_PCT: float   = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 4.0))
    WEEKLY_LOSS_LIMIT_PCT: float  = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", 8.0))
    SCAN_INTERVAL_MINUTES: int    = int(os.getenv("SCAN_INTERVAL_MINUTES", 10))
    MIN_VOLUME_24H: float         = float(os.getenv("MIN_VOLUME_24H", 2_000_000))
    MIN_MARKET_CAP: float         = float(os.getenv("MIN_MARKET_CAP", 20_000_000))
    MAX_MARKET_CAP: float         = float(os.getenv("MAX_MARKET_CAP", 500_000_000))
    PORT: int                     = PORT
    DEBUG: bool                   = DEBUG


settings = _Settings()
