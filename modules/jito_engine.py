"""
jito_engine.py — Exécution anti-MEV via Jito Block Engine.

Jito garantit :
  - Protection contre les sandwich attacks (front-run/back-run)
  - Confirmation prioritaire même en période de congestion
  - Atomic bundling : toutes les txs passent ou aucune

Référence : https://jito-labs.gitbook.io/mev/searcher-resources/json-rpc-api-reference
"""
import os, random, base64
from typing import Optional
import structlog

log = structlog.get_logger()

JITO_ENDPOINT = "https://mainnet.block-engine.jito.labs.io/api/v1/bundles"

# Comptes tip Jito officiels (choisir un au hasard par bundle)
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1uw6nqDevar",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

# Tips selon l'urgence (en lamports, 1 SOL = 1_000_000_000 lamports)
TIPS = {
    "low":    5_000,    # scan régulier, marché calme
    "normal": 10_000,   # conditions normales
    "high":   50_000,   # opportunité forte / score élevé
    "urgent": 100_000,  # momentum exceptionnel, fenêtre courte
}


def get_tip_account() -> str:
    return random.choice(JITO_TIP_ACCOUNTS)


def calc_tip(score: float) -> int:
    """Tip proportionnel au score de conviction."""
    if score >= 85:
        return TIPS["urgent"]
    if score >= 75:
        return TIPS["high"]
    if score >= 65:
        return TIPS["normal"]
    return TIPS["low"]


async def send_bundle(b64_transactions: list[str], session,
                       tip_lamports: int = TIPS["normal"]) -> dict:
    """
    Soumet un bundle de transactions signées (en base64) à Jito.

    Args:
        b64_transactions : liste de txs sérialisées + signées en base64
        session          : aiohttp.ClientSession partagée
        tip_lamports     : tip envoyé au compte Jito (en lamports)

    Returns:
        {"ok": True, "bundle_id": str} ou {"ok": False, "error": str}
    """
    if not b64_transactions:
        return {"ok": False, "error": "bundle vide"}

    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "sendBundle",
        "params":  [b64_transactions],
    }

    try:
        import aiohttp as _aio
        async with session.post(
            JITO_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=_aio.ClientTimeout(total=15),
        ) as r:
            resp = await r.json()

        if "error" in resp:
            err = resp["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            log.warning("jito_bundle_error", msg=msg)
            return {"ok": False, "error": msg}

        bundle_id = resp.get("result", "")
        log.info("jito_bundle_sent", bundle_id=str(bundle_id)[:16],
                 tip_lamports=tip_lamports, n_txs=len(b64_transactions))
        return {"ok": True, "bundle_id": bundle_id}

    except Exception as e:
        log.error("jito_send_failed", error=str(e))
        return {"ok": False, "error": str(e)}


async def get_bundle_status(bundle_id: str, session) -> dict:
    """Vérifie le statut d'un bundle soumis."""
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "getBundleStatuses",
        "params":  [[bundle_id]],
    }
    try:
        async with session.post(JITO_ENDPOINT, json=payload,
                                headers={"Content-Type": "application/json"}) as r:
            resp = await r.json()
        statuses = resp.get("result", {}).get("value", [])
        if statuses:
            return statuses[0]
        return {"status": "unknown"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
