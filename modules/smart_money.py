"""
smart_money.py — Copy trading intelligent sur wallets Solana rentables.

Fonctionnement :
  1. Poll Helius Enhanced Transactions API toutes les 60s par wallet
  2. Détecte les SWAP où le wallet achète un token (reçoit TOKEN, envoie SOL/USDC)
  3. Callback avec {"mint", "symbol", "wallet", "wallet_label", "source": "copy"}
  4. Le bot filtre via son scoring + preflight avant d'entrer

Configuration :
  SMART_MONEY_WALLETS=addr1:label1,addr2:label2  dans .env
  (label optionnel — sert uniquement pour les logs)

Critères "smart money" recommandés pour la sélection manuelle :
  - >= 100 trades on-chain
  - Win rate > 55%
  - Pas de multi-sig ni de bot évident (timestamp trop réguliers)
"""
import os, asyncio, time
from typing import Callable, Optional, Set
import structlog

log = structlog.get_logger()

HELIUS_KEY  = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE = f"https://api.helius.xyz/v0"

# Mints considérés comme "stablecoins / SOL" → le wallet vend ces tokens
# pour acheter autre chose. Si on les voit en sortie, c'est un achat.
_STABLE_MINTS: Set[str] = {
    "So11111111111111111111111111111111111111112",    # SOL (Wrapped)
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",  # mSOL
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # stSOL
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",  # jitoSOL
}

# Taille minimale en SOL pour considérer un achat "intentionnel"
# Évite de copier les micro-transactions de test
MIN_SOL_SPENT = 0.05


def _parse_wallets() -> list[dict]:
    """
    Parse SMART_MONEY_WALLETS depuis l'env.
    Format : addr1:label1,addr2:label2 ou simplement addr1,addr2
    """
    raw = os.getenv("SMART_MONEY_WALLETS", "")
    wallets = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            addr, label = entry.split(":", 1)
        else:
            addr, label = entry, entry[:8] + "…"
        addr = addr.strip()
        if len(addr) >= 32:
            wallets.append({"address": addr, "label": label.strip()})
    return wallets


SMART_MONEY_WALLETS = _parse_wallets()


class SmartMoneyTracker:
    """
    Surveille les wallets smart money en temps réel.

    Mode 1 — Helius Webhooks (prioritaire) :
      Helius pousse les transactions instantanément via POST /api/helius-webhook.
      L'app.py appelle handle_webhook_event() → latence < 1s.

    Mode 2 — Polling fallback (15s) :
      Si les webhooks ne sont pas configurés ou échouent, polling toutes les 15s
      (vs 60s avant = 4x plus rapide).
    """

    POLL_INTERVAL  = 15   # réduit de 60s → 15s
    WEBHOOK_SERVER = os.getenv("PUBLIC_URL", "")  # ex: http://35.225.188.104:5001

    def __init__(self):
        self._running    = False
        self._tasks: list[asyncio.Task] = []
        self._seen_sigs: dict[str, Set[str]] = {}
        self._callback: Optional[Callable] = None
        self._webhook_id: Optional[str]    = None
        self._webhook_active: bool         = False

    @property
    def wallet_count(self) -> int:
        return len(SMART_MONEY_WALLETS)

    async def handle_webhook_event(self, events: list):
        """
        Appelé par app.py quand Helius pousse un événement webhook.
        Traite la liste d'événements et déclenche le callback si c'est un achat.
        """
        if not self._callback or not self._running:
            return
        for tx in events:
            sig = tx.get("signature", "")
            if not sig:
                continue
            # Trouve quel wallet est concerné
            for w in SMART_MONEY_WALLETS:
                seen = self._seen_sigs.get(w["address"], set())
                if sig in seen:
                    continue
                seen.add(sig)
                self._seen_sigs[w["address"]] = seen
                buy = self._parse_buy(tx, w["address"])
                if buy:
                    buy["wallet"]       = w["address"]
                    buy["wallet_label"] = w["label"]
                    buy["signature"]    = sig
                    buy["source"]       = "copy"
                    log.info("smart_money_webhook_buy",
                             wallet=w["label"], mint=buy["mint"][:8])
                    asyncio.create_task(self._fire(buy))

    async def _register_webhook(self):
        """
        Enregistre (ou met à jour) un webhook Helius pour les wallets configurés.
        Nécessite PUBLIC_URL dans .env — ex: http://35.225.188.104:5001
        """
        if not self.WEBHOOK_SERVER or not HELIUS_KEY:
            return False
        webhook_url = f"{self.WEBHOOK_SERVER}/api/helius-webhook"
        addresses   = [w["address"] for w in SMART_MONEY_WALLETS]
        payload = {
            "webhookURL":       webhook_url,
            "transactionTypes": ["SWAP"],
            "accountAddresses": addresses,
            "webhookType":      "enhanced",
        }
        import aiohttp, ssl, certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ctx)
            ) as session:
                # Liste les webhooks existants
                async with session.get(
                    f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
                ) as r:
                    existing = await r.json() if r.status == 200 else []

                # Supprime les anciens webhooks sur notre URL
                for wh in (existing if isinstance(existing, list) else []):
                    if wh.get("webhookURL") == webhook_url:
                        wid = wh.get("webhookID", "")
                        if wid:
                            await session.delete(
                                f"https://api.helius.xyz/v0/webhooks/{wid}?api-key={HELIUS_KEY}"
                            )

                # Crée le nouveau webhook
                async with session.post(
                    f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}",
                    json=payload,
                ) as r:
                    if r.status in (200, 201):
                        data = await r.json()
                        self._webhook_id     = data.get("webhookID")
                        self._webhook_active = True
                        log.info("smart_money_webhook_registered",
                                 id=self._webhook_id, url=webhook_url,
                                 wallets=len(addresses))
                        return True
                    else:
                        body = await r.text()
                        log.warning("smart_money_webhook_failed",
                                    status=r.status, body=body[:200])
                        return False
        except Exception as e:
            log.warning("smart_money_webhook_error", error=str(e))
            return False

    async def start(self, callback: Callable):
        if not HELIUS_KEY:
            log.warning("smart_money_disabled", reason="HELIUS_API_KEY manquant")
            return
        if not SMART_MONEY_WALLETS:
            log.warning("smart_money_disabled",
                        reason="SMART_MONEY_WALLETS vide — configure dans .env")
            return

        self._running  = True
        self._callback = callback
        for w in SMART_MONEY_WALLETS:
            self._seen_sigs[w["address"]] = set()

        # Tente d'enregistrer les webhooks Helius (temps réel)
        webhook_ok = await self._register_webhook()
        if not webhook_ok:
            log.info("smart_money_polling_mode",
                     interval=self.POLL_INTERVAL,
                     hint="Configure PUBLIC_URL dans .env pour activer les webhooks")

        # Lance toujours le polling en parallèle (fallback ou backup)
        for w in SMART_MONEY_WALLETS:
            task = asyncio.create_task(self._poll_loop(w))
            self._tasks.append(task)

        log.info("smart_money_started",
                 wallets=[w["label"] for w in SMART_MONEY_WALLETS],
                 webhook=webhook_ok,
                 poll_interval=self.POLL_INTERVAL)

    async def stop(self):
        self._running = False
        for t in self._tasks:
            if t and not t.done():
                t.cancel()
        self._tasks = []

    # ── Boucle de polling par wallet ──────────────────────────

    async def _poll_loop(self, wallet: dict):
        addr  = wallet["address"]
        label = wallet["label"]

        # Premier poll immédiat pour charger les sigs existantes (sans callback)
        await self._fetch_and_filter(addr, label, initial=True)

        while self._running:
            await asyncio.sleep(self.POLL_INTERVAL)
            try:
                await self._fetch_and_filter(addr, label, initial=False)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("smart_money_poll_error",
                            wallet=label, error=str(e))

    async def _fetch_and_filter(self, addr: str, label: str, initial: bool):
        """
        Récupère les 10 derniers SWAPs du wallet via Helius Enhanced API.
        Si initial=True, enregistre les sigs sans appeler le callback.
        """
        import aiohttp, ssl, certifi

        if not hasattr(self, "_session") or self._session is None:
            ctx = ssl.create_default_context(cafile=certifi.where())
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ctx, limit=5),
                timeout=aiohttp.ClientTimeout(total=12),
                headers={"User-Agent": "macro_alpha/7.0"},
            )

        url = (
            f"{HELIUS_BASE}/addresses/{addr}/transactions"
            f"?api-key={HELIUS_KEY}&type=SWAP&limit=10"
        )
        try:
            async with self._session.get(url) as r:
                if r.status == 429:
                    log.warning("smart_money_rate_limited", wallet=label)
                    return
                if r.status != 200:
                    return
                txs = await r.json()
        except Exception as e:
            log.warning("smart_money_fetch_failed", wallet=label, error=str(e))
            return

        if not isinstance(txs, list):
            return

        seen = self._seen_sigs.get(addr, set())

        for tx in txs:
            sig = tx.get("signature", "")
            if not sig or sig in seen:
                continue
            seen.add(sig)

            if initial:
                continue  # Premier poll : on mémorise sans trader

            buy = self._parse_buy(tx, addr)
            if buy:
                buy["wallet"]       = addr
                buy["wallet_label"] = label
                buy["signature"]    = sig
                buy["source"]       = "copy"
                log.info("smart_money_buy_detected",
                         wallet=label, mint=buy["mint"][:8],
                         sol_spent=buy.get("sol_spent", 0))
                asyncio.create_task(self._fire(buy))

        self._seen_sigs[addr] = seen

        # Mémoire courte : évite les leaks sur wallet très actif
        if len(seen) > 5_000:
            self._seen_sigs[addr] = set(list(seen)[-2_500:])

    def _parse_buy(self, tx: dict, wallet: str) -> Optional[dict]:
        """
        Analyse une transaction SWAP pour détecter si le wallet achète un token.

        Logique :
          - tokenTransfers où toUserAccount == wallet et mint non-stable → TOKEN REÇU
          - tokenTransfers où fromUserAccount == wallet et mint stable → SOL/USDC ENVOYÉ
          → Si les deux conditions sont vraies : c'est un achat

        Retourne None si c'est une vente ou une transaction non pertinente.
        """
        transfers = tx.get("tokenTransfers", [])
        if not transfers:
            return None

        token_received: Optional[str]  = None
        symbol_received: Optional[str] = None
        sol_spent: float = 0.0

        for t in transfers:
            mint   = t.get("mint", "")
            to_acc = t.get("toUserAccount", "")
            fr_acc = t.get("fromUserAccount", "")
            amount = float(t.get("tokenAmount") or 0)

            # Le wallet REÇOIT un token non-stable → candidat "buy"
            if to_acc == wallet and mint and mint not in _STABLE_MINTS:
                token_received  = mint
                symbol_received = t.get("tokenStandard", "") or mint[:8]

            # Le wallet ENVOIE un stable (SOL/USDC) → mesure ce qu'il dépense
            if fr_acc == wallet and mint in _STABLE_MINTS:
                # SOL a 9 décimales, USDC a 6 — on approxime en SOL
                if mint == "So11111111111111111111111111111111111111112":
                    sol_spent += amount
                else:
                    sol_spent += amount / 150.0   # approx USDC→SOL

        if not token_received:
            return None

        # Ignore les micro-achats (test / dust)
        if sol_spent < MIN_SOL_SPENT:
            return None

        # Récupère le symbole depuis accountData si disponible
        for acc in tx.get("accountData", []):
            if acc.get("account") == token_received:
                sym = (acc.get("tokenBalanceChanges") or [{}])[0]
                symbol_received = sym.get("userAccount", symbol_received)

        return {
            "mint":      token_received,
            "symbol":    (symbol_received or token_received[:8]).upper(),
            "sol_spent": round(sol_spent, 4),
            "timestamp": tx.get("timestamp", int(time.time())),
        }

    async def _fire(self, event: dict):
        try:
            await self._callback(event)
        except Exception as e:
            log.error("smart_money_callback_err", error=str(e))

    async def close(self):
        if hasattr(self, "_session") and self._session:
            await self._session.close()
            self._session = None
