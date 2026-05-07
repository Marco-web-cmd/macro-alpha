"""
helius_enrichment.py — Enrichissement on-chain via Helius pour le scoring.

Fournit deux métriques non disponibles dans DexScreener :
  - holder_count  : nombre total de wallets détenant le token (DAS getTokenAccounts)
  - tx_5min       : nombre de transactions on-chain dans les 5 dernières minutes
                    (getSignaturesForAddress — proxy de la vitesse d'adoption)

Utilisé en phase de pré-sélection sur les top candidats seulement (score ≥ 65)
pour limiter les appels RPC aux tokens qui méritent une analyse approfondie.
"""
import os, asyncio, time
from typing import Optional
import structlog

log = structlog.get_logger()

HELIUS_KEY = os.getenv("HELIUS_API_KEY", "")
RPC_URL    = (os.getenv("SOLANA_RPC_URL")
              or f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}")


async def _rpc(method: str, params, session, timeout: int = 8) -> dict:
    import aiohttp
    try:
        async with session.post(
            RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as r:
            return await r.json()
    except Exception as e:
        log.warning("helius_rpc_failed", method=method, error=str(e))
        return {}


async def get_holder_count(mint: str, session) -> int:
    """
    Retourne le nombre total de wallets détenant le token via Helius DAS.
    La méthode getTokenAccounts avec limit=1 retourne le champ `total`
    sans transférer la liste complète — très rapide.

    Retourne 0 en cas d'erreur (pas bloquant pour le scoring).
    """
    if not HELIUS_KEY:
        return 0
    try:
        resp = await _rpc(
            "getTokenAccounts",
            {"mint": mint, "page": 1, "limit": 1},
            session,
        )
        result = resp.get("result", {})
        total  = result.get("total", 0)
        return int(total)
    except Exception as e:
        log.warning("holder_count_failed", mint=mint[:8], error=str(e))
        return 0


async def get_tx_velocity(mint: str, session,
                          window_secs: int = 300) -> int:
    """
    Compte le nombre de transactions impliquant ce token dans les
    `window_secs` dernières secondes (défaut : 5 minutes).

    Utilise getSignaturesForAddress avec limit=50 sur l'adresse du mint.
    Un token avec 20+ txs en 5 min est clairement actif/tradé.

    Retourne 0 en cas d'erreur.
    """
    try:
        resp = await _rpc(
            "getSignaturesForAddress",
            [mint, {"limit": 50, "commitment": "confirmed"}],
            session,
        )
        sigs   = resp.get("result", []) or []
        now    = int(time.time())
        cutoff = now - window_secs
        count  = sum(
            1 for s in sigs
            if s.get("blockTime") and int(s["blockTime"]) >= cutoff
        )
        return count
    except Exception as e:
        log.warning("tx_velocity_failed", mint=mint[:8], error=str(e))
        return 0


async def enrich_token(mint: str, session) -> dict:
    """
    Appelle holder_count et tx_velocity en parallèle.
    Retourne toujours un dict (pas d'exception possible).

    Format : {"holder_count": int, "tx_5min": int}
    """
    if not mint:
        return {"holder_count": 0, "tx_5min": 0}
    try:
        holder_count, tx_5min = await asyncio.gather(
            get_holder_count(mint, session),
            get_tx_velocity(mint, session, window_secs=300),
        )
        return {"holder_count": holder_count, "tx_5min": tx_5min}
    except Exception as e:
        log.warning("enrich_token_failed", mint=mint[:8], error=str(e))
        return {"holder_count": 0, "tx_5min": 0}
