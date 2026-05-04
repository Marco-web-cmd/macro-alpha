"""
screener.py — Screener multi-tokens async (Top 50 Binance + anomalies)
Détecte les anomalies de volume, volatilité et variation de prix.
Worker background qui tourne toutes les 5 minutes.
"""
import httpx
import asyncio
import time
import logging
import numpy as np
from diskcache import Cache

logger = logging.getLogger("screener")

cache = Cache("./data/cache", size_limit=500_000_000)

# ── Seuils d'anomalie ────────────────────────────────────────
ANOMALY_THRESHOLDS = {
    "volume_spike_ratio": 3.0,    # volume 3x la médiane
    "price_change_pct":  10.0,    # +/-10% en 24H
    "volatility_pct":     8.0,    # range H-L > 8%
}


async def fetch_top50_binance() -> list:
    """
    Récupère le top 50 des paires USDT par volume (24H).
    Endpoint : /ticker/24hr (snapshot complet).
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get("https://api.binance.com/api/v3/ticker/24hr")
    tickers = r.json()

    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and float(t.get("quoteVolume", 0)) > 1_000_000
    ]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    logger.info("[Screener] %d paires USDT qualifiées", len(usdt))
    return usdt[:50]


async def detect_anomalies(tickers: list) -> list:
    """
    Détecte les anomalies dans le top 50.
    Utilise la médiane comme baseline (robuste aux outliers).
    Retourne la liste triée par priorité décroissante.
    """
    if not tickers:
        return []

    volumes = [float(t.get("quoteVolume", 0)) for t in tickers]
    vol_median = float(np.median(volumes)) if volumes else 1.0

    anomalies = []
    for t in tickers:
        symbol     = t["symbol"]
        vol        = float(t.get("quoteVolume", 0))
        change_pct = float(t.get("priceChangePercent", 0))
        high       = float(t.get("highPrice", 0))
        low        = float(t.get("lowPrice",  0))
        price      = float(t.get("lastPrice", 0))
        volatility = (high - low) / price * 100 if price > 0 else 0.0
        vol_spike  = vol / max(vol_median, 1.0)
        reasons    = []

        if vol_spike >= ANOMALY_THRESHOLDS["volume_spike_ratio"]:
            reasons.append(f"Volume spike {vol_spike:.1f}x médiane")
        if abs(change_pct) >= ANOMALY_THRESHOLDS["price_change_pct"]:
            reasons.append(f"Prix {change_pct:+.1f}% en 24H")
        if volatility >= ANOMALY_THRESHOLDS["volatility_pct"]:
            reasons.append(f"Volatilité {volatility:.1f}% H-L")

        if reasons:
            anomalies.append({
                "symbol":     symbol,
                "price":      price,
                "change_24h": change_pct,
                "volume_24h": vol,
                "vol_spike":  round(vol_spike, 2),
                "volatility": round(volatility, 2),
                "reasons":    reasons,
                "priority":   len(reasons),
            })

    anomalies.sort(key=lambda x: x["priority"], reverse=True)
    logger.info("[Screener] %d anomalies détectées sur %d tickers",
                len(anomalies), len(tickers))
    return anomalies


async def run_screener_once() -> dict:
    """Lance un scan unique et stocke le résultat dans le cache."""
    tickers   = await fetch_top50_binance()
    anomalies = await detect_anomalies(tickers)
    result    = {
        "tickers":   tickers[:20],
        "anomalies": anomalies,
        "ts":        time.time(),
    }
    cache.set("screener_latest", result, expire=300)

    # Enqueue les top 3 anomalies vers Celery si disponible et Redis actif
    for anomaly in anomalies[:3]:
        try:
            import redis as _redis_lib
            _r = _redis_lib.Redis(host="localhost", port=6379, socket_connect_timeout=1)
            _r.ping()   # vérifie que Redis tourne
            from tasks import run_heavy_inference
            run_heavy_inference.delay(
                symbol=anomaly["symbol"],
                reason=anomaly["reasons"][0],
            )
            logger.info("[Screener] Celery: %s → %s",
                        anomaly["symbol"], anomaly["reasons"][0])
        except Exception as e:
            logger.debug("[Screener] Celery/Redis non disponible (normal en dev local): %s",
                         type(e).__name__)

    return result


async def background_screener_loop():
    """
    Worker background — scan toutes les 5 minutes.
    Tourne tant que l'application est en vie (créé via asyncio.create_task).
    """
    logger.info("[Screener] Worker démarré (intervalle=5min)")
    while True:
        try:
            await run_screener_once()
        except Exception as e:
            logger.error("[Screener] Erreur boucle: %s", e)
        await asyncio.sleep(300)   # 5 minutes
