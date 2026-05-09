"""
smart_money.py — Copy trading intelligent sur wallets Solana rentables.

Sources de wallets (par priorité) :
  1. Birdeye /trader/gainers-losers — top traders par PnL sur 7j (auto-refresh 12h)
  2. SMART_MONEY_WALLETS dans .env — wallets manuels (complément ou fallback)

Détection des achats :
  - getSignaturesForAddress + getTransaction via RPC basique gratuit (sans quota Helius)
  - Analyse preTokenBalances vs postTokenBalances pour détecter tout DEX (Jupiter/Raydium/Orca)
  - Callback avec {"mint", "symbol", "wallet", "wallet_label", "source": "copy"}
  - Le bot filtre via son scoring + preflight avant d'entrer
"""
import os, asyncio, time
from typing import Callable, Optional, Set
import structlog

log = structlog.get_logger()

HELIUS_KEY   = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE  = f"https://api.helius.xyz/v0"
BIRDEYE_KEY  = os.getenv("BIRDEYE_API_KEY", "")
RPC_URL      = (os.getenv("SOLANA_RPC_URL")
                or f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
                or "https://api.mainnet-beta.solana.com")

# Critères de sélection des wallets Birdeye
BIRDEYE_MIN_TRADES  = 50        # ignore les wallets < 50 trades (pas assez de data)
BIRDEYE_MIN_PNL     = 5_000     # PnL minimum en USD sur 7j
BIRDEYE_MAX_WALLETS = 15        # max wallets trackés depuis Birdeye
WALLET_REFRESH_H    = 12        # refresh de la liste toutes les 12h

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


SMART_MONEY_WALLETS = _parse_wallets()  # wallets manuels depuis .env (fallback)


class SmartMoneyTracker:
    """
    Surveille les wallets smart money en temps réel.
    - Source principale : Birdeye top traders 7j (auto-refresh toutes les 12h)
    - Fallback : SMART_MONEY_WALLETS depuis .env
    - Polling via RPC basique gratuit (getSignaturesForAddress + getTransaction)
    """

    POLL_INTERVAL = 15  # secondes entre chaque poll par wallet

    def __init__(self):
        self._running  = False
        self._tasks: list[asyncio.Task] = []
        self._seen_sigs: dict[str, Set[str]] = {}
        self._callback: Optional[Callable] = None
        self._active_wallets: list[dict]  = []  # liste dynamique (Birdeye + env)
        self._polled_addrs: Set[str]      = set()  # adresses déjà en cours de poll

    @property
    def wallet_count(self) -> int:
        return len(self._active_wallets)

    # ── Découverte automatique via Birdeye ────────────────────

    async def _fetch_top_wallets_birdeye(self) -> list[dict]:
        """
        Récupère les meilleurs traders Solana sur 7j depuis Birdeye.
        Filtre : PnL > BIRDEYE_MIN_PNL, trade_count >= BIRDEYE_MIN_TRADES.
        Trie par PnL/trade (rentabilité par trade) pour favoriser l'efficacité.
        """
        if not BIRDEYE_KEY:
            return []
        session = await self._get_session()
        try:
            async with session.get(
                "https://public-api.birdeye.so/trader/gainers-losers"
                "?time_frame=7d&sort_by=PnL&sort_type=desc&limit=50",
                headers={"X-API-KEY": BIRDEYE_KEY, "x-chain": "solana"},
            ) as r:
                if r.status != 200:
                    log.warning("birdeye_traders_failed", status=r.status)
                    return []
                d = await r.json()
        except Exception as e:
            log.warning("birdeye_traders_error", error=str(e))
            return []

        items = (d.get("data") or {}).get("items") or []
        wallets = []
        for w in items:
            addr        = w.get("address", "")
            pnl         = float(w.get("pnl") or 0)
            trade_count = int(w.get("trade_count") or 0)
            if len(addr) < 32 or pnl < BIRDEYE_MIN_PNL or trade_count < BIRDEYE_MIN_TRADES:
                continue
            # PnL/trade = efficacité (évite les wallets chanceux à 2 trades)
            pnl_per_trade = pnl / max(trade_count, 1)
            wallets.append({
                "address":       addr,
                "label":         f"BE_{addr[:6]}",
                "pnl_7d":        round(pnl),
                "trade_count":   trade_count,
                "pnl_per_trade": pnl_per_trade,
            })

        # Trie par PnL/trade et garde les meilleurs
        wallets.sort(key=lambda x: x["pnl_per_trade"], reverse=True)
        return wallets[:BIRDEYE_MAX_WALLETS]

    async def _wallet_discovery_loop(self):
        """Rafraîchit la liste des wallets depuis Birdeye toutes les WALLET_REFRESH_H heures."""
        while self._running:
            try:
                discovered = await self._fetch_top_wallets_birdeye()
                if discovered:
                    await self._update_wallet_list(discovered)
                    log.info("smart_money_wallets_refreshed",
                             count=len(discovered),
                             top=discovered[0]["label"] if discovered else "")
            except Exception as e:
                log.warning("wallet_discovery_error", error=str(e))
            # Attend WALLET_REFRESH_H avant le prochain refresh
            for _ in range(WALLET_REFRESH_H * 360):  # check self._running toutes les 10s
                if not self._running:
                    return
                await asyncio.sleep(10)

    async def _update_wallet_list(self, new_wallets: list[dict]):
        """
        Ajoute les nouveaux wallets à la liste active et lance leurs poll tasks.
        Ne remet pas en cause les wallets déjà en cours de poll.
        """
        for w in new_wallets:
            addr = w["address"]
            if addr not in self._polled_addrs:
                self._active_wallets.append(w)
                self._seen_sigs[addr] = set()
                self._polled_addrs.add(addr)
                task = asyncio.create_task(self._poll_loop(w))
                self._tasks.append(task)
                log.info("smart_money_wallet_added",
                         label=w["label"], pnl_7d=w.get("pnl_7d", 0))

    async def handle_webhook_event(self, events: list):
        """Appelé par app.py si Helius pousse un événement webhook (bonus si dispo)."""
        if not self._callback or not self._running:
            return
        for tx in events:
            sig = tx.get("signature", "")
            if not sig:
                continue
            for w in self._active_wallets:
                seen = self._seen_sigs.get(w["address"], set())
                if sig in seen:
                    continue
                seen.add(sig)
                self._seen_sigs[w["address"]] = seen
                buy = self._parse_buy(tx, w["address"])
                if buy:
                    buy.update({"wallet": w["address"], "wallet_label": w["label"],
                                "signature": sig, "source": "copy"})
                    log.info("smart_money_webhook_buy",
                             wallet=w["label"], mint=buy["mint"][:8])
                    asyncio.create_task(self._fire(buy))

    async def start(self, callback: Callable):
        self._running  = True
        self._callback = callback

        # 1. Charge les wallets manuels depuis .env (disponibles immédiatement)
        for w in SMART_MONEY_WALLETS:
            self._active_wallets.append(w)
            self._seen_sigs[w["address"]] = set()
            self._polled_addrs.add(w["address"])

        # 2. Découverte Birdeye au démarrage
        discovered = await self._fetch_top_wallets_birdeye()
        if discovered:
            await self._update_wallet_list(discovered)
            log.info("smart_money_birdeye_discovery",
                     found=len(discovered),
                     top3=[w["label"] for w in discovered[:3]])
        elif not SMART_MONEY_WALLETS:
            log.warning("smart_money_no_wallets",
                        reason="Birdeye discovery vide et SMART_MONEY_WALLETS non configuré")

        # 3. Lance les polls pour les wallets manuels (.env)
        for w in SMART_MONEY_WALLETS:
            task = asyncio.create_task(self._poll_loop(w))
            self._tasks.append(task)

        # 4. Lance le refresh périodique Birdeye
        self._tasks.append(asyncio.create_task(self._wallet_discovery_loop()))

        log.info("smart_money_started",
                 manual_wallets=len(SMART_MONEY_WALLETS),
                 birdeye_wallets=len(discovered) if discovered else 0,
                 total=len(self._active_wallets),
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

    async def _get_session(self):
        import aiohttp, ssl, certifi
        if not hasattr(self, "_session") or self._session is None or self._session.closed:
            ctx = ssl.create_default_context(cafile=certifi.where())
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ctx, limit=10),
                timeout=aiohttp.ClientTimeout(total=12),
                headers={"User-Agent": "macro_alpha/7.0"},
            )
        return self._session

    async def _rpc(self, method: str, params: list) -> dict:
        """Appel RPC Solana basique (gratuit, pas de quota Helius Enhanced)."""
        session = await self._get_session()
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            async with session.post(RPC_URL, json=payload) as r:
                if r.status != 200:
                    return {}
                return await r.json()
        except Exception as e:
            log.warning("smart_money_rpc_failed", method=method, error=str(e))
            return {}

    async def _fetch_and_filter(self, addr: str, label: str, initial: bool):
        """
        Récupère les signatures récentes via getSignaturesForAddress (RPC gratuit)
        puis parse chaque nouvelle transaction pour détecter les achats de tokens.
        Remplace l'ancienne approche Helius Enhanced API (payante / quota limité).
        """
        resp = await self._rpc(
            "getSignaturesForAddress",
            [addr, {"limit": 15, "commitment": "confirmed"}],
        )
        sigs_raw = resp.get("result") or []
        if not sigs_raw:
            return

        seen = self._seen_sigs.get(addr, set())
        new_sigs = [
            s["signature"] for s in sigs_raw
            if s.get("signature") and s["signature"] not in seen and not s.get("err")
        ]

        for sig in new_sigs:
            seen.add(sig)
            if initial:
                continue

            tx_resp = await self._rpc(
                "getTransaction",
                [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            )
            tx = tx_resp.get("result")
            if not tx:
                continue

            buy = self._parse_buy_rpc(tx, addr)
            if buy:
                buy["wallet"]       = addr
                buy["wallet_label"] = label
                buy["signature"]    = sig
                buy["source"]       = "copy"
                log.info("smart_money_buy_detected",
                         wallet=label, mint=buy["mint"][:8],
                         sol_spent=buy.get("sol_spent", 0))
                asyncio.create_task(self._fire(buy))

            await asyncio.sleep(0.1)  # évite le rate-limit RPC sur les getTransaction

        self._seen_sigs[addr] = seen
        if len(seen) > 5_000:
            self._seen_sigs[addr] = set(list(seen)[-2_500:])

    def _parse_buy_rpc(self, tx: dict, wallet: str) -> Optional[dict]:
        """
        Parse une transaction RPC jsonParsed pour détecter un achat de token.
        Compare preTokenBalances vs postTokenBalances pour trouver les tokens
        reçus par le wallet, et preBalances vs postBalances pour le SOL dépensé.
        Fonctionne avec n'importe quel DEX (Jupiter, Raydium, Orca, Pump.fun).
        """
        meta = tx.get("meta") or {}
        if meta.get("err"):
            return None

        pre_tok  = {b["accountIndex"]: b for b in (meta.get("preTokenBalances")  or [])}
        post_tok = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}
        pre_nat  = meta.get("preBalances")  or []
        post_nat = meta.get("postBalances") or []

        account_keys = (tx.get("transaction") or {}).get("message", {}).get("accountKeys", [])

        def _pubkey(k):
            return k if isinstance(k, str) else k.get("pubkey", "")

        # Indices des comptes appartenant au wallet
        wallet_indices = {i for i, k in enumerate(account_keys) if _pubkey(k) == wallet}

        token_received: Optional[str] = None

        # Cherche un token non-stable dont le solde du wallet a augmenté
        all_indices = set(list(pre_tok.keys()) + list(post_tok.keys()))
        for idx in all_indices:
            pre  = pre_tok.get(idx, {})
            post = post_tok.get(idx, {})
            owner = post.get("owner") or pre.get("owner", "")
            if owner != wallet:
                continue
            mint = post.get("mint") or pre.get("mint", "")
            if not mint or mint in _STABLE_MINTS:
                continue
            pre_amt  = float((pre.get("uiTokenAmount")  or {}).get("uiAmount") or 0)
            post_amt = float((post.get("uiTokenAmount") or {}).get("uiAmount") or 0)
            if post_amt > pre_amt:
                token_received = mint
                break

        if not token_received:
            return None

        # Estime le SOL dépensé via le delta de balance native du wallet
        sol_spent = 0.0
        for i, k in enumerate(account_keys):
            if _pubkey(k) == wallet and i < len(pre_nat) and i < len(post_nat):
                delta = (pre_nat[i] - post_nat[i]) / 1e9  # lamports → SOL
                if delta > 0:
                    sol_spent += delta

        if sol_spent < MIN_SOL_SPENT:
            return None

        return {
            "mint":      token_received,
            "symbol":    token_received[:8].upper(),
            "sol_spent": round(sol_spent, 4),
            "timestamp": tx.get("blockTime", int(time.time())),
        }

    def _parse_buy(self, tx: dict, wallet: str) -> Optional[dict]:
        """Parse format Helius Enhanced (utilisé uniquement par handle_webhook_event)."""
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

            if to_acc == wallet and mint and mint not in _STABLE_MINTS:
                token_received  = mint
                symbol_received = t.get("tokenStandard", "") or mint[:8]

            if fr_acc == wallet and mint in _STABLE_MINTS:
                amount = float(t.get("tokenAmount") or 0)
                if mint == "So11111111111111111111111111111111111111112":
                    sol_spent += amount
                else:
                    sol_spent += amount / 150.0

        if not token_received or sol_spent < MIN_SOL_SPENT:
            return None

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
