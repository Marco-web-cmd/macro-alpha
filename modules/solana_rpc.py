"""
solana_rpc.py — Utilitaires RPC Solana :
  - Priority fees dynamiques (75e percentile des 10 derniers blocs)
  - Pre-flight sécurité : mint authority, freeze authority (= rug risk)
  - Top holder concentration via supply total réel (calcul corrigé)
  - Anti-sniper : détecte si les premières txs viennent du même wallet
  - Slippage liquidity-aware selon impact prix estimé
  - Solana regime : activité réseau (fees, volume DEX)
"""
import os, asyncio, statistics, time
from typing import Optional
import structlog

log = structlog.get_logger()

HELIUS_KEY = os.getenv("HELIUS_API_KEY", "")
RPC_URL    = (os.getenv("SOLANA_RPC_URL")
              or f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}")

DEFAULT_PRIORITY_FEE = 50_000
MAX_PRIORITY_FEE     = 500_000
MIN_LIQ_SOL          = 20.0


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
    try:
        resp = await _rpc_call("getRecentPrioritizationFees", [], session)
        raw  = resp.get("result", [])
        fees = [int(f["prioritizationFee"])
                for f in raw if int(f.get("prioritizationFee", 0)) > 0]
        if not fees:
            return DEFAULT_PRIORITY_FEE
        fees.sort()
        idx = max(0, int(len(fees) * percentile / 100) - 1)
        return min(fees[idx], MAX_PRIORITY_FEE)
    except Exception as e:
        log.warning("priority_fee_failed", error=str(e))
        return DEFAULT_PRIORITY_FEE


async def get_solana_regime(session) -> dict:
    """
    Évalue l'activité du réseau Solana via les fees récents.
    Fee médian élevé = réseau actif = bon pour le trading de memecoins.
    Retourne {"active": bool, "median_fee": int, "score": float 0-1}
    """
    try:
        resp = await _rpc_call("getRecentPrioritizationFees", [], session)
        raw  = resp.get("result", [])
        fees = [int(f["prioritizationFee"])
                for f in raw if int(f.get("prioritizationFee", 0)) > 0]
        if not fees:
            return {"active": True, "median_fee": 0, "score": 0.5}
        median = statistics.median(fees)
        # > 10k lamports = réseau actif, > 100k = très actif
        score  = min(1.0, median / 100_000)
        return {
            "active":     median > 5_000,
            "median_fee": int(median),
            "score":      round(score, 3),
        }
    except Exception as e:
        log.warning("solana_regime_failed", error=str(e))
        return {"active": True, "median_fee": 0, "score": 0.5}


async def check_token_safety(mint: str, session) -> dict:
    """
    Vérifie les permissions dangereuses du mint :
    - freezeAuthority présente → REJECT (le créateur peut bloquer les ventes)
    - mintAuthority présente   → WARN  (dilution possible)
    """
    try:
        resp   = await _rpc_call(
            "getAccountInfo", [mint, {"encoding": "jsonParsed"}], session)
        value  = resp.get("result", {}).get("value")
        if value is None:
            return {"safe": False, "reason": "mint introuvable on-chain"}

        info       = (value.get("data", {}).get("parsed", {}).get("info", {}))
        has_freeze = info.get("freezeAuthority") is not None
        has_mint   = info.get("mintAuthority")   is not None

        reason = ""
        if has_freeze:
            reason = "freezeAuthority présente (ventes bloquables)"
        elif has_mint:
            reason = "mintAuthority présente (dilution possible)"

        return {
            "safe":             not has_freeze,
            "freeze_authority": has_freeze,
            "mint_authority":   has_mint,
            "decimals":        info.get("decimals", 9),
            "reason":          reason,
        }
    except Exception as e:
        log.warning("token_safety_failed", mint=mint[:8], error=str(e))
        return {"safe": True, "freeze_authority": False,
                "mint_authority": False, "reason": f"RPC error: {e}"}


async def get_top_holders(mint: str, session,
                           max_concentration_pct: float = 50.0) -> dict:
    """
    Vérifie la concentration du top holder en % du supply TOTAL (corrigé).
    Utilise getTokenSupply pour le dénominateur réel — pas la somme des top-20
    qui surestimait la concentration et rejetait trop de tokens légitimes.
    """
    try:
        holders_resp, supply_resp = await asyncio.gather(
            _rpc_call("getTokenLargestAccounts", [mint], session),
            _rpc_call("getTokenSupply",          [mint], session),
        )
        holders      = holders_resp.get("result", {}).get("value", [])
        supply_info  = supply_resp.get("result",  {}).get("value", {})
        total_supply = float(supply_info.get("uiAmount") or 0)

        if not holders:
            return {"concentrated": False, "top_holder_pct": 0.0, "holder_count": 0}
        if total_supply == 0:
            # Fallback : somme des top-20 comme avant
            amounts = [float(h.get("uiAmount") or 0) for h in holders]
            total   = sum(amounts)
            top_pct = amounts[0] / total * 100 if total > 0 else 0
        else:
            top_amount = float(holders[0].get("uiAmount") or 0)
            top_pct    = top_amount / total_supply * 100

        return {
            "concentrated":   top_pct > max_concentration_pct,
            "top_holder_pct": round(top_pct, 1),
            "holder_count":   len(holders),
        }
    except Exception as e:
        log.warning("top_holders_failed", mint=mint[:8], error=str(e))
        return {"concentrated": False, "top_holder_pct": 0.0, "holder_count": 0}


async def check_sniper_concentration(mint: str, session,
                                      n_tx: int = 10,
                                      max_same_wallet_pct: float = 0.5) -> dict:
    """
    Anti-sniper : vérifie si les premières transactions du token sont
    majoritairement issues du même wallet (bot de snipe qui va dump).

    Récupère les n_tx premières signatures et compte les senders uniques.
    Si un seul wallet représente > max_same_wallet_pct des txs → suspect.

    Retourne {"sniped": bool, "top_sender_pct": float}
    """
    try:
        resp = await _rpc_call(
            "getSignaturesForAddress",
            [mint, {"limit": n_tx, "commitment": "confirmed"}],
            session,
        )
        sigs = resp.get("result", []) or []
        if len(sigs) < 3:
            # Token trop récent pour juger — on laisse passer
            return {"sniped": False, "top_sender_pct": 0.0}

        # Récupère les détails de chaque tx pour identifier le fee payer (= sender)
        tx_tasks = [
            _rpc_call("getTransaction",
                      [s["signature"], {"encoding": "json", "maxSupportedTransactionVersion": 0}],
                      session)
            for s in sigs[:n_tx]
        ]
        txs = await asyncio.gather(*tx_tasks, return_exceptions=True)

        senders: dict[str, int] = {}
        for tx in txs:
            if isinstance(tx, Exception):
                continue
            try:
                keys = (tx.get("result") or {}).get("transaction", {}) \
                          .get("message", {}).get("accountKeys", [])
                if keys:
                    sender = keys[0] if isinstance(keys[0], str) else keys[0].get("pubkey", "")
                    if sender:
                        senders[sender] = senders.get(sender, 0) + 1
            except Exception:
                continue

        if not senders:
            return {"sniped": False, "top_sender_pct": 0.0}

        total    = sum(senders.values())
        top_cnt  = max(senders.values())
        top_pct  = top_cnt / total

        return {
            "sniped":         top_pct > max_same_wallet_pct,
            "top_sender_pct": round(top_pct, 2),
        }
    except Exception as e:
        log.warning("sniper_check_failed", mint=mint[:8], error=str(e))
        return {"sniped": False, "top_sender_pct": 0.0}


def calc_liquidity_aware_slippage(amount_base: float,
                                   pool_liquidity_usd: float,
                                   sol_price_usd: float = 150.0) -> int:
    if pool_liquidity_usd <= 0:
        return 300

    trade_usd    = amount_base * sol_price_usd
    impact_pct   = trade_usd / pool_liquidity_usd * 100
    slippage_pct = impact_pct * 1.2
    slippage_bps = int(slippage_pct * 100)

    return max(50, min(500, slippage_bps))
