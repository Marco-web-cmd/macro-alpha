"""
jupiter_swap.py — DEX trading via Jupiter Aggregator (Solana).

Jupiter route tous les swaps sur le meilleur prix disponible
(Raydium, Orca, Meteora, Phoenix...).

Configuration .env :
  SOLANA_PRIVATE_KEY=clé_privée_base58  (depuis Phantom : Settings > Export Private Key)
  SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=TON_HELIUS_KEY
  DRY_RUN=true   (paper trading par défaut)

Endpoints Jupiter v6 (gratuits, sans API key) :
  GET  https://quote-api.jup.ag/v6/quote
  POST https://quote-api.jup.ag/v6/swap
"""
import os
import json
import base64
import asyncio
import httpx
import structlog
from datetime import datetime, timezone
from typing import Optional

log = structlog.get_logger()

JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"
JUPITER_PRICE_URL = "https://api.jup.ag/price/v2"
JUPITER_TOKENS_URL = "https://tokens.jup.ag/tokens?tags=verified"

# Adresses des tokens courants (pour résolution symbol → mint)
KNOWN_TOKENS: dict[str, str] = {
    "SOL":   "So11111111111111111111111111111111111111112",
    "USDC":  "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT":  "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "BONK":  "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF":   "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JUP":   "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "RAY":   "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "ORCA":  "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "PYTH":  "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "RENDER":"rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof",
    "TNSR":  "TNSRxcUxoT9xBG3de7A9MSdVDwoRkHtSbUB5TYXnWGx",
    "JITO":  "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    "MEW":   "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5",
    "POPCAT":"7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
}

USDC_MINT = KNOWN_TOKENS["USDC"]
SOL_MINT  = KNOWN_TOKENS["SOL"]

# Token de base pour les swaps (SOL ou USDC selon .env)
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "SOL").upper()
BASE_MINT     = SOL_MINT if BASE_CURRENCY == "SOL" else USDC_MINT
BASE_DECIMALS = 9 if BASE_CURRENCY == "SOL" else 6


class JupiterSwap:
    """
    Module de swap DEX via Jupiter Aggregator.
    Paper trading par défaut (DRY_RUN=true).
    """

    def __init__(self):
        self.dry_run   = os.getenv("DRY_RUN", "true").lower() == "true"
        self.rpc_url   = os.getenv(
            "SOLANA_RPC_URL",
            f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_API_KEY','')}"
        )
        self._keypair  = self._load_keypair()
        self._client   = self._init_rpc()

        mode = "PAPER" if self.dry_run else "LIVE"
        wallet = str(self._keypair.pubkey()) if self._keypair else "non configuré"
        log.info("jupiter_swap_init", mode=mode, wallet=wallet[:8] + "...")

    # ── Init ─────────────────────────────────────────────────────

    def _load_keypair(self):
        """Charge la keypair depuis SOLANA_PRIVATE_KEY (base58)."""
        pk = os.getenv("SOLANA_PRIVATE_KEY", "")
        if not pk:
            log.info("solana_no_private_key — paper trading only")
            return None
        try:
            from solders.keypair import Keypair
            import base58 as b58
            secret = b58.b58decode(pk)
            return Keypair.from_bytes(secret)
        except Exception as e:
            log.error("keypair_load_failed", error=str(e))
            return None

    def _init_rpc(self):
        """Initialise le client RPC Solana (Helius ou URL custom)."""
        helius_key = os.getenv("HELIUS_API_KEY", "")
        url = self.rpc_url or (
            f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            if helius_key else ""
        )
        if not url:
            log.warning("solana_rpc_no_url")
            return None
        try:
            from solana.rpc.api import Client
            client = Client(url)
            return client
        except Exception as e:
            log.error("rpc_init_failed", error=str(e))
            return None

    # ── Résolution token ─────────────────────────────────────────

    async def resolve_token_mint(self, symbol: str) -> Optional[str]:
        """Résout un symbole en adresse mint Solana."""
        sym = symbol.upper()

        # Check liste connue
        if sym in KNOWN_TOKENS:
            return KNOWN_TOKENS[sym]

        # Requête Jupiter token list
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(JUPITER_TOKENS_URL)
                tokens = r.json()
                for t in tokens:
                    if t.get("symbol", "").upper() == sym:
                        mint = t["address"]
                        KNOWN_TOKENS[sym] = mint
                        return mint
        except Exception as e:
            log.warning("token_resolve_failed", symbol=sym, error=str(e))

        return None

    # ── Quote ────────────────────────────────────────────────────

    async def get_quote(self,
                        input_mint: str,
                        output_mint: str,
                        amount_usdc: float,
                        slippage_bps: int = 100,
                        input_decimals: int = None) -> dict:
        """
        Obtient le meilleur prix Jupiter pour un swap.
        amount_usdc : montant dans la monnaie d'entrée
        """
        decimals   = input_decimals if input_decimals is not None else BASE_DECIMALS
        amount_raw = int(amount_usdc * (10 ** decimals))

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    JUPITER_QUOTE_URL,
                    params={
                        "inputMint":   input_mint,
                        "outputMint":  output_mint,
                        "amount":      amount_raw,
                        "slippageBps": slippage_bps,
                        "onlyDirectRoutes": False,
                    }
                )
                r.raise_for_status()
                data = r.json()

            in_amount    = int(data["inAmount"]) / (10 ** decimals)
            out_amount   = int(data["outAmount"])
            price_impact = float(data.get("priceImpactPct", 0))
            if price_impact > 1:  # si déjà en pourcentage (v1)
                pass
            else:
                price_impact *= 100  # si en fraction (v6)

            # Calcule le prix effectif (SOL = 9 décimales)
            out_decimals = 9
            out_human    = out_amount / (10 ** out_decimals)
            price        = in_amount / out_human if out_human > 0 else 0

            return {
                "ok":           True,
                "in_amount":    round(in_amount, 2),
                "out_amount":   out_amount,
                "out_human":    round(out_human, 6),
                "price":        round(price, 4),
                "price_impact": round(price_impact, 3),
                "slippage_bps": slippage_bps,
                "routes":       len(data.get("routePlan", [])),
                "raw":          data,
            }
        except Exception as e:
            log.error("jupiter_quote_failed", error=str(e))
            return {"ok": False, "error": str(e)}

    # ── Swap ─────────────────────────────────────────────────────

    async def execute_swap(self,
                           symbol: str,
                           amount_usdc: float,
                           slippage_bps: int = 100,
                           output_mint: str = None,
                           priority_fee_lamports: int = 50_000) -> dict:
        """
        Exécute un swap BASE_CURRENCY → token via Jupiter.
        output_mint : mint address direct (évite la résolution par symbole)
        """
        sym = symbol.upper()

        # Utilise le mint fourni directement, sinon résout par symbole
        if not output_mint:
            output_mint = await self.resolve_token_mint(sym)
        if not output_mint:
            return {"ok": False, "error": f"Token {sym} non trouvé sur Solana"}

        # Obtient le quote (SOL ou USDC selon BASE_CURRENCY)
        quote = await self.get_quote(
            input_mint=BASE_MINT,
            output_mint=output_mint,
            amount_usdc=amount_usdc,
            slippage_bps=slippage_bps,
        )
        if not quote["ok"]:
            return quote

        # Vérifie le price impact
        if quote["price_impact"] > 5:
            return {
                "ok":     False,
                "error":  f"Price impact trop élevé : {quote['price_impact']:.1f}% (max 5%)",
                "quote":  quote,
            }

        result_base = {
            "symbol":       sym,
            "amount_usdc":  amount_usdc,
            "out_amount":   quote["out_human"],
            "price":        quote["price"],
            "price_impact": quote["price_impact"],
            "slippage_bps": slippage_bps,
            "ts":           datetime.now(timezone.utc).isoformat(),
        }

        # Paper trading
        if self.dry_run:
            log.info("swap_simulated", symbol=sym,
                     amount_usdc=amount_usdc,
                     out=quote["out_human"],
                     price=quote["price"])
            return {
                **result_base,
                "ok":      True,
                "mode":    "PAPER",
                "tx_hash": None,
                "note":    "Simulé — DRY_RUN=true. Mettre DRY_RUN=false pour le live.",
            }

        # Live : vérifie keypair
        if not self._keypair:
            return {
                "ok":    False,
                "error": "SOLANA_PRIVATE_KEY non configuré dans .env",
            }
        if not self._client:
            return {
                "ok":    False,
                "error": "RPC Solana non disponible (vérifie HELIUS_API_KEY)",
            }

        # Construit la transaction via Jupiter
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    JUPITER_SWAP_URL,
                    json={
                        "quoteResponse":              quote["raw"],
                        "userPublicKey":              str(self._keypair.pubkey()),
                        "wrapAndUnwrapSol":           True,
                        "dynamicComputeUnitLimit":    True,
                        "prioritizationFeeLamports":  priority_fee_lamports,
                        "useSharedAccounts":          True,
                    },
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()
                swap_data = r.json()

            # Décode et signe la transaction
            from solders.transaction import VersionedTransaction
            from solana.rpc.types import TxOpts

            raw_tx   = base64.b64decode(swap_data["swapTransaction"])
            tx       = VersionedTransaction.from_bytes(raw_tx)
            signed   = self._keypair.sign_message(bytes(tx.message))

            # Envoie via RPC
            tx_response = self._client.send_raw_transaction(
                bytes(tx),
                opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"),
            )
            tx_hash = str(tx_response.value)

            log.info("swap_live_ok", symbol=sym,
                     amount_usdc=amount_usdc,
                     tx=tx_hash[:16] + "...")

            return {
                **result_base,
                "ok":      True,
                "mode":    "LIVE",
                "tx_hash": tx_hash,
                "explorer": f"https://solscan.io/tx/{tx_hash}",
            }

        except Exception as e:
            log.error("swap_live_failed", symbol=sym, error=str(e))
            return {"ok": False, "error": str(e), **result_base}

    # ── Sell ─────────────────────────────────────────────────────

    async def execute_sell(self,
                           symbol: str,
                           amount_tokens: float,
                           slippage_bps: int = 150) -> dict:
        """Vend des tokens → USDC."""
        sym         = symbol.upper()
        input_mint  = await self.resolve_token_mint(sym)
        if not input_mint:
            return {"ok": False, "error": f"Token {sym} non trouvé"}

        # Estime le montant raw (approximation si décimales inconnues)
        decimals   = 9 if sym != "USDC" else 6
        amount_raw = int(amount_tokens * (10 ** decimals))

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    JUPITER_QUOTE_URL,
                    params={
                        "inputMint":   input_mint,
                        "outputMint":  USDC_MINT,
                        "amount":      amount_raw,
                        "slippageBps": slippage_bps,
                    }
                )
                r.raise_for_status()
                quote_raw = r.json()

            usdc_out = int(quote_raw["outAmount"]) / 1_000_000
            impact   = float(quote_raw.get("priceImpactPct", 0)) * 100

            if self.dry_run:
                return {
                    "ok":          True,
                    "mode":        "PAPER",
                    "symbol":      sym,
                    "sold_tokens": amount_tokens,
                    "usdc_out":    round(usdc_out, 2),
                    "price_impact": round(impact, 3),
                    "tx_hash":     None,
                    "ts":          datetime.now(timezone.utc).isoformat(),
                }

            # Live : même logique que execute_swap
            if not self._keypair:
                return {"ok": False, "error": "SOLANA_PRIVATE_KEY manquant"}

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    JUPITER_SWAP_URL,
                    json={
                        "quoteResponse":             quote_raw,
                        "userPublicKey":             str(self._keypair.pubkey()),
                        "wrapAndUnwrapSol":          True,
                        "dynamicComputeUnitLimit":   True,
                        "prioritizationFeeLamports": priority_fee_lamports,
                        "useSharedAccounts":         True,
                    },
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()
                swap_data = r.json()

            from solders.transaction import VersionedTransaction
            from solana.rpc.types import TxOpts

            raw_tx = base64.b64decode(swap_data["swapTransaction"])
            tx     = VersionedTransaction.from_bytes(raw_tx)
            resp   = self._client.send_raw_transaction(
                bytes(tx),
                opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"),
            )
            tx_hash = str(resp.value)

            return {
                "ok":          True,
                "mode":        "LIVE",
                "symbol":      sym,
                "sold_tokens": amount_tokens,
                "usdc_out":    round(usdc_out, 2),
                "price_impact": round(impact, 3),
                "tx_hash":     tx_hash,
                "explorer":    f"https://solscan.io/tx/{tx_hash}",
                "ts":          datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            log.error("sell_failed", symbol=sym, error=str(e))
            return {"ok": False, "error": str(e)}

    # ── Portfolio ────────────────────────────────────────────────

    async def get_portfolio(self) -> dict:
        """
        Récupère le solde USDC + tokens du wallet via Helius.
        Enrichit avec les prix actuels pour calculer la valeur totale.
        """
        if not self._keypair:
            return {
                "mode":    "PAPER",
                "wallet":  None,
                "balance": {"USDC": 10000.0},
                "note":    "SOLANA_PRIVATE_KEY non configuré",
            }

        wallet = str(self._keypair.pubkey())

        try:
            # Solde via Helius getAssetsByOwner
            helius_key = os.getenv("HELIUS_API_KEY", "")
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                    json={
                        "jsonrpc": "2.0",
                        "id":      1,
                        "method":  "getTokenAccountsByOwner",
                        "params":  [
                            wallet,
                            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                            {"encoding": "jsonParsed"},
                        ],
                    }
                )
                data = r.json()

            accounts = data.get("result", {}).get("value", [])
            balances = {}
            for acc in accounts:
                info   = acc["account"]["data"]["parsed"]["info"]
                mint   = info["mint"]
                amount = float(info["tokenAmount"]["uiAmount"] or 0)
                if amount > 0:
                    # Résout le symbole
                    sym = next((k for k, v in KNOWN_TOKENS.items() if v == mint), mint[:8])
                    balances[sym] = amount

            # SOL natif
            sol_resp = self._client.get_balance(
                self._keypair.pubkey()
            )
            sol_lamports = sol_resp.value
            balances["SOL"] = round(sol_lamports / 1e9, 4)

            return {
                "mode":    "LIVE",
                "wallet":  wallet,
                "balance": balances,
                "ts":      datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            log.error("portfolio_failed", error=str(e))
            return {"ok": False, "error": str(e), "wallet": wallet}

    # ── Token info ───────────────────────────────────────────────

    async def get_token_price(self, symbol: str) -> dict:
        """Prix d'un token Solana via quote Jupiter (100 USDC → token)."""
        mint = await self.resolve_token_mint(symbol.upper())
        if not mint:
            return {"ok": False, "error": f"{symbol} non trouvé"}

        try:
            # Utilise le quote 100 USDC → token pour déduire le prix
            quote = await self.get_quote(
                input_mint=USDC_MINT,
                output_mint=mint,
                amount_usdc=100.0,
                slippage_bps=50,
            )
            if not quote["ok"]:
                return {"ok": False, "error": quote.get("error", "quote failed")}

            return {
                "ok":          True,
                "symbol":      symbol.upper(),
                "mint":        mint,
                "price":       quote["price"],
                "price_impact": quote["price_impact"],
                "ts":          datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
