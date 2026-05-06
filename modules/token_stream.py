"""
token_stream.py — Détection instantanée de nouveaux tokens Solana via WebSocket.

Mode Push : le bot reçoit les événements en temps réel (0ms) au lieu de
            poller toutes les 30 minutes.

Sources surveillées :
  - Raydium AMM V4 : nouvelles paires de liquidité (instruction initialize2)
  - Pump.fun        : nouvelles créations de memecoins
"""
import os, json, asyncio, re
from typing import Callable, Optional, Set
import structlog

log = structlog.get_logger()

HELIUS_KEY  = os.getenv("HELIUS_API_KEY", "")
WS_URL      = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

# Program IDs Solana
RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMPFUN     = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Adresses à exclure du parsing (programs systèmes, stablecoins, etc.)
EXCLUDE_ADDRS: Set[str] = {
    RAYDIUM_AMM, PUMPFUN,
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token Program
    "11111111111111111111111111111111",               # System Program
    "So11111111111111111111111111111111111111112",    # SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS",  # ATA Program
    "ComputeBudget111111111111111111111111111111",    # Compute Budget
}

_ADDR_RE = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')


class TokenStream:
    """
    Écoute les programmes Raydium et Pump.fun via WebSocket Helius.
    Pour chaque nouveau token détecté, appelle le callback avec :
      {"mint": str, "signature": str, "source": "raydium"|"pumpfun"}
    """

    def __init__(self):
        self._running    = False
        self._seen_sigs: Set[str] = set()
        self._tasks      = []

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
        ]
        log.info("token_stream_started", sources=["raydium", "pumpfun"])

    async def stop(self):
        self._running = False
        for t in self._tasks:
            if t and not t.done():
                t.cancel()
        self._tasks = []

    async def _subscribe(self, program: str, init_kw: str, source: str):
        import websockets
        reconnect_delay = 2

        while self._running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                ) as ws:
                    sub = json.dumps({
                        "jsonrpc": "2.0",
                        "id":      1,
                        "method":  "logsSubscribe",
                        "params":  [
                            {"mentions": [program]},
                            {"commitment": "confirmed"},
                        ],
                    })
                    await ws.send(sub)
                    reconnect_delay = 2  # reset après connexion réussie
                    log.info("ws_connected", source=source)

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
                    log.warning("ws_reconnecting", source=source,
                                error=str(e), wait=reconnect_delay)
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

        # Ignore les transactions échouées et les doublons
        if err or not sig or sig in self._seen_sigs:
            return

        # Filtre : ne traiter que les initialisations de pool/token
        if not any(init_kw.lower() in line.lower() for line in logs):
            return

        self._seen_sigs.add(sig)
        # Garde la mémoire courte
        if len(self._seen_sigs) > 20_000:
            self._seen_sigs = set(list(self._seen_sigs)[-10_000:])

        mint = self._extract_mint(logs)
        if not mint:
            return

        log.info("new_token_detected", mint=mint[:8], sig=sig[:8], source=source)
        try:
            await self._callback({
                "mint":      mint,
                "signature": sig,
                "source":    source,
            })
        except Exception as e:
            log.error("stream_callback_err", error=str(e))

    @staticmethod
    def _extract_mint(logs: list) -> Optional[str]:
        """
        Extrait le premier candidat mint Solana depuis les logs.
        Les logs Raydium/Pump.fun contiennent les comptes impliqués.
        """
        for line in logs:
            for addr in _ADDR_RE.findall(line):
                if addr not in EXCLUDE_ADDRS and len(addr) >= 32:
                    return addr
        return None
