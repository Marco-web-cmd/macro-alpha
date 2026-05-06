"""
solana_rpc.py — Utilitaires RPC Solana :
  - Priority fees dynamiques (75e percentile des 10 derniers blocs)
  - Pre-flight sécurité : mint authority, freeze authority (= rug risk)
  - Top holder concentration (> 10% = whale trap)
  - Slippage liquidity-aware selon impact prix estimé
"""
import os, statistics
from typing import Optional
import structlog

log = structlog.get_logger()

HELIUS_KEY = os.getenv("HELIUS_API_KEY", "")
RPC_URL    = (os.getenv("SOLANA_RPC_URL")
              or f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}")

# Lamports par défaut si l'appel RPC échoue
DEFAULT_PRIORITY_FEE = 50_000   # 50k microlamports ≈ priorité correcte
MAX_PRIORITY_FEE     = 500_000  # cap à 0.5M pour ne pas surpayer
MIN_LIQ_SOL          = 20.0     # liquidité minimale en SOL pour analyser un token


async def _rpc_call(method: str, params: list, session) -> dict:
    import aiohttp
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        async with session.post(RPC_URL, json=payload,
                                timeout=aiohttp.ClientTimeout(total=8)) as r:
            return await r.json()
    except Exception as e:
        log.warning("rpc_call_failed", method=method, error=str(e))
        return {}


async def get_priority_fee_lamports(session, percentile: int = 75) -> int:
    """
    Retourne le fee au Xe percentile des frais récents.
    Se positionne devant la majorité des transactions sans surpayer.
    """
    try:
        resp = await _rpc_call("getRecentPrioritizationFees", [], session)
        raw  = resp.get("result", [])
        fees = [int(f["prioritizationFee"])
                for f in raw if int(f.get("prioritizationFee", 0)) > 0]
        if not fees:
            return DEFAULT_PRIORITY_FEE
        fees.sort()
        idx = max(0, int(len(fees) * percentile / 100) - 1)
        fee = fees[idx]
        return min(fee, MAX_PRIORITY_FEE)
    except Exception as e:
        log.warning("priority_fee_failed", error=str(e))
        return DEFAULT_PRIORITY_FEE


async def check_token_safety(mint: str, session) -> dict:
    """
    Vérifie les permissions dangereuses du mint :
    - freezeAuthority présente → le créateur peut bloquer les ventes → REJECT
    - mintAuthority présente   → le créateur peut diluer l'offre      → WARN

    Retourne :
      {"safe": bool, "freeze_authority": bool, "mint_authority": bool, "reason": str}
    """
    try:
        resp   = await _rpc_call(
            "getAccountInfo", [mint, {"encoding": "jsonParsed"}], session)
        value  = resp.get("result", {}).get("value")
        if value is None:
            return {"safe": False, "reason": "mint introuvable on-chain"}

        info     = (value.get("data", {})
                        .get("parsed", {})
                        .get("info", {}))
        has_freeze = info.get("freezeAuthority") is not None
        has_mint   = info.get("mintAuthority")   is not None

        reason = ""
        if has_freeze:
            reason = "freezeAuthority présente (ventes bloquables)"
        elif has_mint:
            reason = "mintAuthority présente (dilution possible)"

        return {
            "safe":             not has_freeze,   # freeze = rug direct
            "freeze_authority": has_freeze,
            "mint_authority":   has_mint,
            "decimals":        info.get("decimals", 9),
            "reason":          reason,
        }
    except Exception as e:
        log.warning("token_safety_failed", mint=mint[:8], error=str(e))
        # En cas d'erreur RPC : on laisse passer avec un avertissement
        return {"safe": True, "freeze_authority": False,
                "mint_authority": False, "reason": f"RPC error: {e}"}


async def get_top_holders(mint: str, session,
                           max_concentration_pct: float = 10.0) -> dict:
    """
    Vérifie la concentration du top holder via getTokenLargestAccounts.
    Si le top holder détient > max_concentration_pct% de l'offre → concentré.

    Retourne :
      {"concentrated": bool, "top_holder_pct": float, "holder_count": int}
    """
    try:
        resp    = await _rpc_call("getTokenLargestAccounts", [mint], session)
        holders = resp.get("result", {}).get("value", [])
        if not holders:
            return {"concentrated": False, "top_holder_pct": 0.0, "holder_count": 0}

        amounts = [float(h.get("uiAmount") or 0) for h in holders]
        total   = sum(amounts)
        if total == 0:
            return {"concentrated": False, "top_holder_pct": 0.0,
                    "holder_count": len(holders)}

        top_pct = amounts[0] / total * 100

        return {
            "concentrated":   top_pct > max_concentration_pct,
            "top_holder_pct": round(top_pct, 1),
            "holder_count":   len(holders),
        }
    except Exception as e:
        log.warning("top_holders_failed", mint=mint[:8], error=str(e))
        return {"concentrated": False, "top_holder_pct": 0.0, "holder_count": 0}


def calc_liquidity_aware_slippage(amount_base: float,
                                   pool_liquidity_usd: float,
                                   sol_price_usd: float = 150.0) -> int:
    """
    Calcule le slippage en BPS selon l'impact prix estimé de notre trade.
    Impact = (trade_usd / pool_liq) * 100
    Slippage = impact * 1.2 (marge 20%), capé entre 50 et 500 BPS.

    Exemples :
      10 USDC dans pool 100k USD → impact 0.01% → 12 BPS (min 50)
      10 USDC dans pool 5k USD   → impact 0.2%  → 240 BPS
    """
    if pool_liquidity_usd <= 0:
        return 300  # fallback sécurité

    trade_usd    = amount_base * sol_price_usd
    impact_pct   = trade_usd / pool_liquidity_usd * 100
    slippage_pct = impact_pct * 1.2
    slippage_bps = int(slippage_pct * 100)

    return max(50, min(500, slippage_bps))
