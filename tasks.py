"""
tasks.py — Celery worker pour l'inférence lourde en background.
Lance les modèles IA (Chronos + MOIRAI + Macro) sans bloquer le dashboard.

Démarrage :
  celery -A tasks worker --loglevel=info --concurrency=1 --prefetch-multiplier=1
"""
import os
import sys
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("tasks")

# ── Celery + Redis ───────────────────────────────────────────
try:
    from celery import Celery

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    celery_app = Celery(
        "macro_alpha",
        broker=REDIS_URL,
        backend=REDIS_URL.replace("/0", "/1"),
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        task_track_started=True,
        worker_prefetch_multiplier=1,
    )
    CELERY_OK = True
except ImportError:
    CELERY_OK = False
    logger.warning("[Tasks] Celery non disponible — tâches désactivées")


def _noop_task(*args, **kwargs):
    """Placeholder si Celery non disponible."""
    logger.warning("[Tasks] Celery non dispo — tâche ignorée: %s %s", args, kwargs)
    return {"status": "skipped", "reason": "celery_not_available"}


if CELERY_OK:
    @celery_app.task(name="run_heavy_inference", bind=True,
                     max_retries=2, default_retry_delay=30)
    def run_heavy_inference(self, symbol: str, reason: str = ""):
        """
        Inférence lourde pour un token spécifique.
        Analyse technique + macro + forecast IA.
        Résultat stocké dans diskcache sous 'inference_{symbol}'.
        """
        logger.info("[Task] heavy_inference_start symbol=%s reason=%s",
                    symbol, reason)
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import httpx
            import asyncio
            import numpy as np
            import pandas as pd
            from modules.technical  import full_technical_analysis
            from modules.macro_data import compute_macro_score
            from diskcache import Cache

            # Fetch OHLCV via Binance (synchrone dans le worker)
            loop = asyncio.new_event_loop()

            async def _fetch():
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": symbol, "interval": "4h", "limit": 200},
                    )
                    return r.json()

            raw = loop.run_until_complete(_fetch())
            loop.close()

            # Parse OHLCV
            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "ct", "qav", "nt", "tbb", "tbq", "ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df = df[["open", "high", "low", "close", "volume"]]

            # Analyse
            tech  = full_technical_analysis(df, None, "4h")
            macro = compute_macro_score()

            result = {
                "symbol":   symbol,
                "reason":   reason,
                "tech":     tech,
                "macro":    macro,
                "ts":       datetime.now(timezone.utc).isoformat(),
                "status":   "complete",
            }

            # Stocke dans le cache disque partagé
            cache = Cache("./data/cache", size_limit=500_000_000)
            cache.set(f"inference_{symbol.upper()}", result, expire=600)
            logger.info("[Task] heavy_inference_complete symbol=%s", symbol)
            return result

        except Exception as e:
            logger.error("[Task] heavy_inference_failed symbol=%s error=%s",
                         symbol, e)
            raise self.retry(exc=e)

else:
    # Fallback si Celery non installé
    run_heavy_inference = _noop_task
