"""
token_stream.py — Détection instantanée de nouveaux tokens Solana via WebSocket.
v2 — Parsing ciblé, rate-limiter, callback non-bloquant.

Corrections vs v1 :
  - Pump.fun  : regex cible "mint" dans les logs structurés CreateEvent
  - Raydium   : filtre uniquement les adresses de longueur 43-44 (vraie pubkey)
  - Rate limiter : max 5 tokens/sec par source (évite le flood event loop)
  - Callback non-bloquant : asyncio.create_task()
  - EXCLUDE_ADDRS élargi (MetaPlex, Serum, OpenBook, Token-2022, etc.)
"""
import os, json, asyncio, re, time
from typing import Callable, Optional, Set
import structlog

log = structlog.get_logger()

HELIUS_KEY = os.getenv("HELIUS_API_KEY", "")
# URLs WebSocket par priorité — bascule automatique si quota Helius épuisé
_WS_URLS = [
    f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}",
    "wss://api.mainnet-beta.solana.com",       # RPC public officiel Solana (gratuit)
    "wss://solana-mainnet.g.alchemy.com/v2/demo",  # Alchemy demo (gratuit, limité)
]

RAYDIUM_AMM  = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMPFUN      = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPFUN_AMM  = "pAMMBay6oceH9fJKBjAn43VQ1ARNJ8hEi9cNFHVxnME"  # pump.fun AMM v2 (graduation)

# ── Adresses système / programmes connus à exclure ────────────
EXCLUDE_ADDRS: Set[str] = {
    # DEX / Launchpads
    RAYDIUM_AMM, PUMPFUN,
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",  # Raydium Authority V4
    "HWy1jotHpo6UqeQxx49dpYYdQB8wj9Qk9MdxwjLvDHB8",  # Raydium fee
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Orca Whirlpool
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",  # Orca v1
    # Token Programs
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # Token Program
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",   # Token-2022
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS",  # ATA Program
    # System
    "11111111111111111111111111111111",
    "ComputeBudget111111111111111111111111111111",
    "Sysvar1111111111111111111111111111111111111",
    "SysvarRent111111111111111111111111111111111",
    "SysvarC1ock11111111111111111111111111111111",
    "SysvarRecentB1ockHashes11111111111111111111",
    "Vote111111111111111111111111111111111111111h",
    # Stablecoins / tokens de base
    "So11111111111111111111111111111111111111112",    # SOL (Wrapped)
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",  # mSOL
    # Metaplex / NFT
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s",  # Metaplex
    "p1exdMJcjVao65QdewkaZRUnU6VPSXhus9n2GzWfh98",  # Metaplex Auction
    # Serum / OpenBook
    "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",  # Serum DEX v3
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX",   # OpenBook
    # Jupiter
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter v6
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",   # Jupiter v4
    # Pump.fun infra
    "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1", # Pump fee receiver
    "CebN5WGQ4jvEPvsVU4EoHEpgznyQHeGKuL5JZRtGnMt",  # Pump bonding curve (old)
    "pAMMBay6oceH9fJKBjAn43VQ1ARNJ8hEi9cNFHVxnME",  # Pump.fun AMM v2
    "BSfD6SHZigAfDWSjzD5Q41jw8LmKwtmjskPH9XW1mrRW", # Pump global config
    # Fees / treasury communs
    "7oo7f5HQB1VTiSxPFN9m7i9PgfFKQRGjJMzqYwpAFRLz", # fee account courant
    "4wTV81szrCPnBj3wkBrDx1bNNJFdh4bHf7GEWwDnp4s1", # global fee receiver
}

# Longueur exacte d'une pubkey Solana encodée base58 : 43 ou 44 caractères
_SOLANA_ADDR_RE = re.compile(r'\b([1-9A-HJ-NP-Za-km-z]{43,44})\b')

# Patterns ciblés pour extraire le mint depuis les logs structurés
# Pump.fun émet : {"mint":"<addr>",...} ou Program log: CreateEvent { mint: <addr>
_MINT_KV_RE = re.compile(
    r'"?mint"?\s*[:=]\s*"?([1-9A-HJ-NP-Za-km-z]{43,44})"?',
    re.IGNORECASE
)

# ── Rate limiter par source ───────────────────────────────────
_RATE_WINDOW  = 1.0   # secondes
_RATE_MAX     = 5     # max tokens détectés par fenêtre par source


class _RateLimiter:
    def __init__(self, window: float = _RATE_WINDOW, max_events: int = _RATE_MAX):
        self._window = window
        self._max    = max_events
        self._hits: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        self._hits = [t for t in self._hits if now - t < self._window]
        if len(self._hits) >= self._max:
            return False
        self._hits.append(now)
        return True


class TokenStream:
    """
    Écoute Raydium et Pump.fun via WebSocket Helius.
    Callback reçoit : {"mint": str, "signature": str, "source": "raydium"|"pumpfun"}
    """

    def __init__(self):
        self._running   = False
        self._seen_sigs: Set[str] = set()
        self._seen_mints: Set[str] = set()
        self._tasks     = []
        self._limiters  = {
            "raydium":    _RateLimiter(),
            "pumpfun":    _RateLimiter(),
            "graduation": _RateLimiter(),
        }

    async def start(self, callback: Callable):
        if not HELIUS_KEY:
            log.warning("token_stream_disabled", reason="HELIUS_API_KEY manquant")
            return
        try:
            import websockets  # noqa: F401
        except ImportError:
            log.warning("token_stream_disabled",
                        reason="pip install websockets requis")
            return

        self._running  = True
        self._callback = callback
        self._tasks = [
            asyncio.create_task(
                self._subscribe(RAYDIUM_AMM, "initialize2", "raydium")),
            asyncio.create_task(
                self._subscribe(PUMPFUN, "create", "pumpfun")),
            asyncio.create_task(
                self._subscribe(PUMPFUN_AMM, "buy", "graduation")),
        ]
        log.info("token_stream_started", sources=["raydium", "pumpfun", "graduation"])

    async def stop(self):
        self._running = False
        for t in self._tasks:
            if t and not t.done():
                t.cancel()
        self._tasks = []

    async def _subscribe(self, program: str, init_kw: str, source: str):
        import websockets
        reconnect_delay = 2
        url_idx = 0  # commence par Helius, bascule si 429

        while self._running:
            ws_url = _WS_URLS[url_idx % len(_WS_URLS)]
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id":      1,
                        "method":  "logsSubscribe",
                        "params":  [
                            {"mentions": [program]},
                            {"commitment": "confirmed"},
                        ],
                    }))
                    reconnect_delay = 2
                    log.info("ws_connected", source=source, url=ws_url[:40])

                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg = json.loads(raw)
                            await self._handle(msg, init_kw, source)
                        except Exception:
                            pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._running:
                    err_str = str(e)
                    # 429 = quota Helius épuisé → bascule sur URL suivante
                    if "429" in err_str and url_idx < len(_WS_URLS) - 1:
                        url_idx += 1
                        log.warning("ws_fallback", source=source,
                                    new_url=_WS_URLS[url_idx][:40])
                        reconnect_delay = 2
                    else:
                        log.warning("ws_reconnecting", source=source,
                                    error=err_str, wait=reconnect_delay)
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 60)

    async def _handle(self, msg: dict, init_kw: str, source: str):
        params = msg.get("params")
        if not params:
            return

        value = params.get("result", {}).get("value", {})
        sig   = value.get("signature", "")
        logs  = value.get("logs", [])
        err   = value.get("err")

        if err or not sig or sig in self._seen_sigs:
            return

        # Vérifie que c'est bien une instruction de création
        if not any(init_kw.lower() in line.lower() for line in logs):
            return

        self._seen_sigs.add(sig)
        if len(self._seen_sigs) > 20_000:
            self._seen_sigs = set(list(self._seen_sigs)[-10_000:])

        mint = self._extract_mint(logs, source)
        if not mint or mint in self._seen_mints:
            return

        # Rate limit par source
        if not self._limiters[source].allow():
            return

        self._seen_mints.add(mint)
        if len(self._seen_mints) > 50_000:
            self._seen_mints = set(list(self._seen_mints)[-25_000:])

        log.info("new_token_detected", mint=mint[:8], sig=sig[:8], source=source)

        # Callback non-bloquant pour ne pas geler la boucle WS
        asyncio.create_task(self._fire({"mint": mint, "signature": sig, "source": source}))

    async def _fire(self, event: dict):
        try:
            await self._callback(event)
        except Exception as e:
            log.error("stream_callback_err", error=str(e))

    @staticmethod
    def _extract_mint(logs: list, source: str) -> Optional[str]:
        """
        Extraction ciblée selon la source :
        - pumpfun  : cherche "mint" : <addr> dans les logs structurés
        - raydium  : cherche la 1ère adresse 43-44 chars hors programmes connus
        """
        if source == "pumpfun":
            # Pump.fun émet des logs JSON-like avec "mint":"<addr>"
            for line in logs:
                m = _MINT_KV_RE.search(line)
                if m:
                    addr = m.group(1)
                    if addr not in EXCLUDE_ADDRS:
                        return addr

        # Raydium ou fallback pump.fun : cherche adresses 43-44 chars
        for line in logs:
            for addr in _SOLANA_ADDR_RE.findall(line):
                if addr not in EXCLUDE_ADDRS:
                    return addr

        return None
