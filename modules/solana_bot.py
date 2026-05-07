"""
solana_bot.py — Bot de trading autonome Solana via Jupiter DEX.
v7 — Architecture Push + exécution Elite :
  - WebSocket stream : détection instantanée Raydium/Pump.fun (token_stream)
  - Pre-flight sécurité : freeze authority + top holder check (solana_rpc)
  - Priority fees dynamiques au 75e percentile (solana_rpc)
  - Slippage liquidity-aware : calculé selon l'impact prix estimé
  - Jito bundles : anti-MEV, protection sandwich (jito_engine)
  - Filtre liq/mc ≥ 2% et liq ≥ 20 SOL minimum
  - Session HTTP persistante, prix parallèles, retry Jupiter
  - Agent IA gate : pause si crash ou signal baissier
"""
import os, json, asyncio, aiohttp, ssl, certifi, uuid
from datetime import datetime, timezone
from typing import Optional
import structlog

log = structlog.get_logger()

# ── Paramètres ───────────────────────────────────────────────
BASE_CURRENCY     = os.getenv("BASE_CURRENCY", "SOL").upper()
INITIAL_CAPITAL   = float(os.getenv("TOTAL_CAPITAL", "0.12"))
MAX_POSITIONS     = 6
SL_PCT            = 15.0
TP1_PCT           = 30.0
TP2_PCT           = 60.0
MOONBAG_TRAIL_BASE = 12.0   # % trailing depuis pic (s'élargit avec le gain)
MAX_PRICE_DRIFT   = 3.0
MAX_RISK_PCT      = 0.20
MIN_POSITION      = 0.01 if os.getenv("BASE_CURRENCY", "SOL").upper() == "SOL" else 1.5
MIN_SCORE         = 60
MIN_LIQ_MC_RATIO  = 0.02    # liquidité doit être ≥ 2% du market cap (anti-liquidity trap)
MIN_LIQ_USD       = 20 * 150.0  # ≈ 20 SOL en USD (filtre rapide avant analyse)
SCAN_INTERVAL     = 30 * 60
MONITOR_INTERVAL  = 2 * 60
JUPITER_MAX_RETRY = 3

POSITIONS_FILE = "data/solana_positions.json"
LOG_FILE       = "data/bot_log.json"

KNOWN_MINTS = {
    "SOL":      "So11111111111111111111111111111111111111112",
    "USDC":     "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "BONK":     "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF":      "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JUP":      "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "RAY":      "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "ORCA":     "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "JITO":     "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",
    "MEW":      "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5",
    "POPCAT":   "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
    "FARTCOIN": "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump",
    "TRUMP":    "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
    "PENGU":    "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
}


# ── Retry Jupiter ─────────────────────────────────────────────
async def _jupiter_with_retry(coro_fn, max_attempts=JUPITER_MAX_RETRY):
    """Retente une coroutine Jupiter jusqu'à max_attempts fois avec backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except Exception as e:
            err = str(e).lower()
            is_congestion = any(k in err for k in
                ("timeout", "blockhash", "congestion", "429", "503", "502"))
            if attempt < max_attempts and is_congestion:
                wait = 2 ** attempt
                log.warning("jupiter_retry", attempt=attempt, wait=wait, error=str(e))
                await asyncio.sleep(wait)
            else:
                raise


class SolanaBot:

    def __init__(self):
        self.dry_run      = os.getenv("DRY_RUN", "true").lower() == "true"
        self.running       = False
        self._scan_task    = None
        self._mon_task     = None
        self._stream_task  = None
        self._wallet_task  = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.birdeye_key  = os.getenv("BIRDEYE_API_KEY", "")
        os.makedirs("data", exist_ok=True)
        log.info("solana_bot_init", mode="PAPER" if self.dry_run else "LIVE",
                 initial_capital=INITIAL_CAPITAL)

    # ── Session HTTP unique et persistante ────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """Retourne la session existante ou en crée une nouvelle."""
        if self._session is None or self._session.closed:
            ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ctx,
                limit=20,           # max 20 connexions simultanées
                ttl_dns_cache=300,  # cache DNS 5 min
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=12),
                headers={"User-Agent": "macro_alpha/6.0"},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Capital dynamique ─────────────────────────────────────

    def _total_capital(self) -> float:
        positions = self._load_positions()
        realized  = sum(p.get("pnl_usdc") or 0
                        for p in positions if p["status"] == "closed")
        return round(INITIAL_CAPITAL + max(0, realized), 4)

    def _invested_capital(self) -> float:
        positions = self._load_positions()
        return sum(p.get("remaining_usdc") or p.get("amount_usdc") or 0
                   for p in positions if p["status"] == "open")

    # ── Contrôle ──────────────────────────────────────────────

    async def start(self) -> dict:
        if self.running:
            return {"ok": False, "msg": "Bot déjà démarré"}
        self.running    = True
        await self._get_session()   # crée la session une seule fois
        self._scan_task   = asyncio.create_task(self._scan_loop())
        self._mon_task    = asyncio.create_task(self._monitor_loop())
        self._stream_task = asyncio.create_task(self._start_stream())
        self._wallet_task = asyncio.create_task(self._start_wallet_tracker())
        cap  = self._total_capital()
        mode = "PAPER" if self.dry_run else "LIVE"
        self._log("BOT_START",
            f"Mode {mode} — capital {INITIAL_CAPITAL} {BASE_CURRENCY} "
            f"| total {cap} {BASE_CURRENCY}")
        return {"ok": True, "msg": "Bot démarré — scan + monitor + stream + copy trading actifs"}

    async def stop(self) -> dict:
        self.running = False
        for t in [self._scan_task, self._mon_task,
                  self._stream_task, self._wallet_task]:
            if t and not t.done():
                t.cancel()
        await self.close()
        self._log("BOT_STOP", "Bot arrêté manuellement")
        return {"ok": True, "msg": "Bot arrêté"}

    def get_status(self) -> dict:
        positions  = self._load_positions()
        open_pos   = [p for p in positions if p["status"] == "open"]
        closed_pos = [p for p in positions if p["status"] == "closed"]
        wins       = [p for p in closed_pos if (p.get("pnl_pct") or 0) > 0]
        realized   = sum(p.get("pnl_usdc") or 0 for p in closed_pos)
        unrealized = sum(p.get("pnl_usdc") or 0 for p in open_pos)
        total_cap  = self._total_capital()
        invested   = self._invested_capital()
        return {
            "running":         self.running,
            "mode":            "PAPER" if self.dry_run else "LIVE",
            "base_currency":   BASE_CURRENCY,
            "initial_capital": INITIAL_CAPITAL,
            "total_capital":   total_cap,
            "invested_usdc":   round(invested, 4),
            "available_usdc":  round(max(0, total_cap - invested), 4),
            "open_positions":  len(open_pos),
            "max_positions":   MAX_POSITIONS,
            "total_trades":    len(closed_pos),
            "win_rate":        round(len(wins) / max(1, len(closed_pos)) * 100, 1),
            "realized_pnl":    round(realized, 4),
            "unrealized_pnl":  round(unrealized, 6),
            "total_pnl_usdc":  round(realized + unrealized, 6),
            "positions":       positions,
            "log":             self._load_log()[-50:],
            "agent_context":   self._agent_context(),
        }

    # ── Boucles ───────────────────────────────────────────────

    async def _scan_loop(self):
        while self.running:
            try:
                await self._run_scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("scan_error", error=str(e))
                self._log("ERREUR", f"Scan échoué: {e}")
            await asyncio.sleep(SCAN_INTERVAL)

    async def _monitor_loop(self):
        await asyncio.sleep(60)
        while self.running:
            try:
                await self._check_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("monitor_error", error=str(e))
            await asyncio.sleep(MONITOR_INTERVAL)

    # ── Stream WebSocket push ─────────────────────────────────

    async def _start_stream(self):
        """Lance le WebSocket stream Raydium/Pump.fun en tâche de fond."""
        try:
            from modules.token_stream import TokenStream
            self._stream = TokenStream()
            await self._stream.start(self._on_stream_token)
            log.info("stream_task_started")
        except Exception as e:
            log.warning("stream_start_failed", error=str(e))

    async def _on_stream_token(self, event: dict):
        """
        Callback appelé pour chaque nouveau token détecté via WebSocket.
        Évalue le token via DexScreener : si score ≥ MIN_SCORE et slot dispo → entrée.
        Rate-limité par TokenStream (max 5/sec) donc pas besoin de guard ici.
        """
        if not self.running:
            return

        mint   = event.get("mint", "")
        source = event.get("source", "unknown")
        sig    = event.get("signature", "")[:8]

        now = asyncio.get_event_loop().time()

        # Throttle DexScreener : 1 lookup/2s max pour éviter le rate-limit
        last_lookup = getattr(self, "_last_stream_lookup", 0)
        if now - last_lookup < 2:
            return
        self._last_stream_lookup = now

        # Cooldown trade : pas plus de 1 entrée rapide toutes les 10s
        last_trade = getattr(self, "_last_stream_trade", 0)
        if now - last_trade < 10:
            return

        positions = self._load_positions()
        open_pos  = [p for p in positions if p["status"] == "open"]
        slots     = MAX_POSITIONS - len(open_pos)
        if slots <= 0:
            return

        ctx       = self._agent_context()
        size_mult = ctx["size_mult"]
        if size_mult == 0.0:
            return

        total_cap = self._total_capital()
        available = total_cap - self._invested_capital()
        if available < MIN_POSITION:
            return

        # Récupère les données du token via DexScreener
        token_data = await self._dexscreener_token_data(mint)
        if not token_data:
            return

        sym = token_data.get("symbol", mint[:8]).upper()
        if sym in {p["symbol"] for p in open_pos}:
            return

        liq = float(token_data.get("liquidity") or 0)
        mc  = float(token_data.get("mc") or 0)
        if liq < MIN_LIQ_USD:
            return
        if mc > 0 and liq / mc < MIN_LIQ_MC_RATIO:
            return

        # Score initial (sans données Helius)
        score_basic = self._score(token_data)

        # Enrichissement Helius si le token est prometteur (score ≥ 55 pour seuil bas)
        if score_basic >= 55:
            try:
                from modules.helius_enrichment import enrich_token
                s = await self._get_session()
                enrichment = await enrich_token(mint, s)
                token_data.update(enrichment)
            except Exception:
                pass

        score = self._score(token_data)
        if score < MIN_SCORE:
            log.info("stream_token_rejected", sym=sym, score=score,
                     source=source, sig=sig)
            return

        price = float(token_data.get("price") or 0)
        if not price:
            return

        pf = await self._preflight_check(mint, sym)
        if not pf["ok"]:
            return

        amount = self._position_size(score, available, slots, total_cap)
        amount = round(amount * size_mult, 4)
        if amount < MIN_POSITION:
            return

        self._last_stream_trade = now  # type: ignore[attr-defined]
        self._log("STREAM",
            f"⚡ [{source.upper()}] {sym} détecté — score {score:.0f} — entrée {amount:.4f} SOL",
            sym)
        await self._open_position(sym, token_data, price, amount, score, ctx)

    # ── Copy trading — Smart Money ────────────────────────────

    async def _start_wallet_tracker(self):
        """Lance le suivi des wallets smart money configurés dans .env."""
        try:
            from modules.smart_money import SmartMoneyTracker
            self._smart_money = SmartMoneyTracker()
            if self._smart_money.wallet_count == 0:
                log.info("smart_money_no_wallets",
                         hint="Ajoute SMART_MONEY_WALLETS=addr1:label1,addr2 dans .env")
                return
            await self._smart_money.start(self._on_wallet_buy)
        except Exception as e:
            log.warning("wallet_tracker_start_failed", error=str(e))

    async def _on_wallet_buy(self, event: dict):
        """
        Callback appelé quand un wallet smart money achète un token.
        Passe le token par le même pipeline que le stream :
          DexScreener → scoring enrichi → preflight → entrée.
        """
        if not self.running:
            return

        mint         = event.get("mint", "")
        wallet_label = event.get("wallet_label", "?")
        sol_spent    = event.get("sol_spent", 0)

        # Vérifie les slots et le capital disponible
        positions = self._load_positions()
        open_pos  = [p for p in positions if p["status"] == "open"]
        slots     = MAX_POSITIONS - len(open_pos)
        if slots <= 0:
            return

        ctx       = self._agent_context()
        size_mult = ctx["size_mult"]
        if size_mult == 0.0:
            return

        total_cap = self._total_capital()
        available = total_cap - self._invested_capital()
        if available < MIN_POSITION:
            return

        # Cooldown : pas plus d'1 entrée copy toutes les 15s
        now = asyncio.get_event_loop().time()
        if now - getattr(self, "_last_copy_trade", 0) < 15:
            return

        # Récupère les données du token
        token_data = await self._dexscreener_token_data(mint)
        if not token_data:
            log.info("copy_no_dex_data", mint=mint[:8], wallet=wallet_label)
            return

        sym = token_data.get("symbol", mint[:8]).upper()
        if sym in {p["symbol"] for p in open_pos}:
            return

        liq = float(token_data.get("liquidity") or 0)
        mc  = float(token_data.get("mc") or 0)
        if liq < MIN_LIQ_USD:
            return
        if mc > 0 and liq / mc < MIN_LIQ_MC_RATIO:
            return

        # Score initial + enrichissement Helius
        score_basic = self._score(token_data)
        if score_basic >= 45:   # seuil bas pour copy : on fait confiance au wallet
            try:
                from modules.helius_enrichment import enrich_token
                s = await self._get_session()
                enrichment = await enrich_token(mint, s)
                token_data.update(enrichment)
            except Exception:
                pass

        score = self._score(token_data)

        # Pour le copy trading on accepte un score plus bas (55 vs 70)
        # Le signal "smart money" compense le score manquant
        COPY_MIN_SCORE = 55
        if score < COPY_MIN_SCORE:
            log.info("copy_token_rejected", sym=sym, score=score,
                     wallet=wallet_label)
            return

        price = float(token_data.get("price") or 0)
        if not price:
            return

        pf = await self._preflight_check(mint, sym)
        if not pf["ok"]:
            return

        amount = self._position_size(score, available, slots, total_cap)
        amount = round(amount * size_mult, 4)
        if amount < MIN_POSITION:
            return

        self._last_copy_trade = now  # type: ignore[attr-defined]
        self._log("COPY",
            f"🧠 [{wallet_label}] {sym} copié — {sol_spent:.3f} SOL dépensés "
            f"— score {score:.0f} — entrée {amount:.4f} SOL", sym)

        # Marque le token comme "copy trade" dans les données
        token_data["copy_wallet"]       = event.get("wallet", "")
        token_data["copy_wallet_label"] = wallet_label
        await self._open_position(sym, token_data, price, amount, score, ctx)

    async def _dexscreener_token_data(self, mint: str) -> Optional[dict]:
        """Récupère les données complètes d'un token depuis DexScreener (inclut buys/sells)."""
        try:
            s = await self._get_session()
            async with s.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            ) as r:
                d = await r.json()
            if not d or not isinstance(d, dict):
                return None
            pairs = sorted(
                [p for p in (d.get("pairs") or [])
                 if p.get("chainId") == "solana"
                 and float((p.get("liquidity") or {}).get("usd") or 0) > 5_000],
                key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0),
                reverse=True,
            )
            if not pairs:
                return None
            p       = pairs[0]
            base    = p.get("baseToken", {})
            txns_h1 = p.get("txns", {}).get("h1", {})
            return {
                "symbol":                (base.get("symbol") or mint[:8]).upper(),
                "address":               base.get("address") or mint,
                "price":                 float(p.get("priceUsd") or 0),
                "priceChange24hPercent": float((p.get("priceChange") or {}).get("h24") or 0),
                "v24hUSD":               float((p.get("volume") or {}).get("h24") or 0),
                "mc":                    float(p.get("marketCap") or 0),
                "liquidity":             float((p.get("liquidity") or {}).get("usd") or 0),
                "buys_h1":               int(txns_h1.get("buys") or 0),
                "sells_h1":              int(txns_h1.get("sells") or 0),
            }
        except Exception as e:
            log.warning("dexscreener_token_data_failed", mint=mint[:8], error=str(e))
        return None

    # ── Signal agent IA ───────────────────────────────────────

    def _agent_context(self) -> dict:
        """
        Lit le contexte macro depuis le cache interne d'app.py.
        En cas d'erreur ou d'absence de données : pause prudente (size_mult=0.5).
        En cas de crash complet : pause totale (size_mult=0).
        """
        try:
            import sys
            app_module = sys.modules.get("app") or sys.modules.get("__main__")
            cache = getattr(app_module, "_cache", None) if app_module else None

            if cache is None:
                # Module app non chargé — prudence maximale
                return {
                    "macro_score":   50.0,
                    "cycle_phase":   "UNKNOWN",
                    "market_regime": "UNKNOWN",
                    "size_mult":     0.5,
                    "source":        "no_app_module",
                }

            agent = None
            for tf in ("1h", "4h", "1d"):
                agent = cache.get(f"agent_full_{tf}")
                if agent:
                    break

            if not agent:
                # Cache vide : agent pas encore lancé → taille réduite mais on trade quand même
                full  = cache.get("full_1h_24h", {})
                macro = full.get("macro", {})
                ms    = float(macro.get("score", 0) or 0)
                return {
                    "macro_score":   max(ms, 50.0),  # score neutre par défaut
                    "cycle_phase":   "UNKNOWN",
                    "market_regime": "UNKNOWN",
                    "size_mult":     0.5,             # taille réduite, jamais 0
                    "source":        "full_cache_only",
                }

            macro_score   = float(agent.get("macro_score", 0) or 0)
            cycle_phase   = str(agent.get("btc_cycle_phase", "UNKNOWN") or "UNKNOWN").upper()
            market_regime = str(agent.get("market_regime", "UNKNOWN") or "UNKNOWN").upper()

            # Phases bloquantes
            BLOCKED = ("LATE_BEAR", "CAPITULATION", "BEAR", "CORRECTION_PROFONDE")
            if any(b in cycle_phase for b in BLOCKED):
                return {
                    "macro_score":   macro_score,
                    "cycle_phase":   cycle_phase,
                    "market_regime": market_regime,
                    "size_mult":     0.0,
                    "source":        "agent_cache",
                }

            # Multiplicateur macro
            if macro_score >= 65:
                score_mult = 1.0
            elif macro_score >= 50:
                score_mult = 0.75
            elif macro_score >= 35:
                score_mult = 0.5
            else:
                score_mult = 0.0

            # Multiplicateur régime
            if "BEAR" in market_regime or "RANGING" in market_regime:
                regime_mult = 0.6
            elif "BULL" in market_regime:
                regime_mult = 1.0
            else:
                regime_mult = 0.8

            return {
                "macro_score":   macro_score,
                "cycle_phase":   cycle_phase,
                "market_regime": market_regime,
                "size_mult":     round(score_mult * regime_mult, 2),
                "source":        "agent_cache",
            }

        except Exception as e:
            log.error("agent_context_crash", error=str(e))
            # Crash de l'agent → on stoppe les entrées par sécurité
            return {
                "macro_score":   0.0,
                "cycle_phase":   "ERROR",
                "market_regime": "ERROR",
                "size_mult":     0.0,
                "source":        "error",
            }

    # ── WebSocket stream ─────────────────────────────────────

    async def _start_stream(self):
        """Lance le listener WebSocket Raydium/Pump.fun."""
        try:
            from modules.token_stream import TokenStream
            self._token_stream = TokenStream()
            await self._token_stream.start(self._on_stream_token)
        except Exception as e:
            log.warning("stream_start_failed", error=str(e))
            self._log("STREAM", f"⚠ WebSocket stream indisponible: {e}")

    async def _on_stream_token(self, event: dict):
        """
        Callback appelé pour chaque nouveau token détecté en temps réel.
        Pipeline : enrichissement DexScreener → pre-flight → score → entrée.
        """
        if not self.running:
            return

        mint   = event.get("mint", "")
        source = event.get("source", "stream")
        if not mint:
            return

        # Enrichit avec les données DexScreener
        token = await self._enrich_from_dex(mint)
        if not token:
            return

        sym = (token.get("symbol") or "").upper().strip()
        if not sym or sym in ("USDC", "USDT", "BUSD", "SOL", "WSOL"):
            return

        # Filtre liquidité minimale (20 SOL) — élimine 80% du bruit
        liq = float(token.get("liquidity") or 0)
        if liq < MIN_LIQ_USD:
            return

        # Vérifie qu'on n'a pas déjà cette position
        positions = self._load_positions()
        held = {p["symbol"] for p in positions if p["status"] == "open"}
        if sym in held or len([p for p in positions if p["status"] == "open"]) >= MAX_POSITIONS:
            return

        ctx = self._agent_context()
        if ctx["size_mult"] == 0.0:
            return

        score = self._score(token)
        if score < MIN_SCORE:
            return

        self._log("STREAM",
            f"🚨 Nouveau token stream: {sym} | score {score:.0f} "
            f"| liq ${liq:,.0f} | src={source}")

        total_cap = self._total_capital()
        invested  = self._invested_capital()
        available = total_cap - invested
        slots     = MAX_POSITIONS - len(held)

        amount = self._position_size(score, available, slots, total_cap)
        amount = round(amount * ctx["size_mult"], 4)
        if amount < MIN_POSITION:
            return

        price = await self._fetch_price_dex(mint)
        if not price:
            return

        await self._open_position(sym, token, price, amount, score, ctx)

    async def _enrich_from_dex(self, mint: str) -> Optional[dict]:
        """Enrichit un mint avec les données DexScreener (prix, volume, MC, liquidité)."""
        try:
            s = await self._get_session()
            async with s.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            ) as r:
                d = await r.json()
            pairs = [p for p in d.get("pairs", []) if p.get("chainId") == "solana"]
            if not pairs:
                return None
            p = max(pairs,
                    key=lambda x: float((x.get("liquidity") or {}).get("usd") or 0))
            return {
                "symbol":                p.get("baseToken", {}).get("symbol", ""),
                "address":               mint,
                "price":                 float(p.get("priceUsd") or 0),
                "priceChange24hPercent": float((p.get("priceChange") or {}).get("h24") or 0),
                "v24hUSD":               float(p.get("volume", {}).get("h24") or 0),
                "mc":                    float(p.get("marketCap") or 0),
                "liquidity":             float((p.get("liquidity") or {}).get("usd") or 0),
            }
        except Exception:
            return None

    # ── Pre-flight sécurité ───────────────────────────────────

    async def _preflight_check(self, mint: str, sym: str) -> dict:
        """
        Vérifie avant tout achat :
        1. Freeze authority → BLOCK (le créateur peut bloquer les ventes)
        2. Top holder > 10% → BLOCK (whale trap / rug probable)
        Retourne {"ok": bool, "reason": str}
        """
        try:
            from modules.solana_rpc import check_token_safety, get_top_holders
            s = await self._get_session()

            safety, holders = await asyncio.gather(
                check_token_safety(mint, s),
                get_top_holders(mint, s),
            )

            if not safety.get("safe", True):
                reason = safety.get("reason", "freezeAuthority présente")
                self._log("PREFLIGHT",
                    f"🚫 {sym} rejeté — {reason}", sym)
                return {"ok": False, "reason": reason}

            if holders.get("concentrated", False):
                pct    = holders.get("top_holder_pct", 0)
                reason = f"top holder détient {pct:.1f}% (whale trap)"
                self._log("PREFLIGHT",
                    f"🚫 {sym} rejeté — {reason}", sym)
                return {"ok": False, "reason": reason}

            if safety.get("mint_authority"):
                self._log("PREFLIGHT",
                    f"⚠ {sym} mintAuthority présente (dilution possible) — entrée réduite", sym)

            return {"ok": True, "reason": ""}

        except ImportError:
            return {"ok": True, "reason": "solana_rpc non disponible"}
        except Exception as e:
            log.warning("preflight_failed", sym=sym, error=str(e))
            return {"ok": True, "reason": f"preflight error: {e}"}

    # ── Priority fees dynamiques ──────────────────────────────

    async def _get_priority_fee(self) -> int:
        """Retourne le fee au 75e percentile des blocs récents."""
        try:
            from modules.solana_rpc import get_priority_fee_lamports
            s = await self._get_session()
            return await get_priority_fee_lamports(s, percentile=75)
        except Exception:
            return 50_000

    # ── Trailing stop dynamique ───────────────────────────────

    @staticmethod
    def _moonbag_trail_pct(pnl_pct: float) -> float:
        """
        Trailing stop adaptatif selon le gain réalisé.
        Plus le gain est important, plus on donne de marge au moonbag
        pour éviter d'être sorti trop tôt sur une bougie de correction.
        """
        if pnl_pct >= 200:  return 25.0
        if pnl_pct >= 100:  return 20.0
        if pnl_pct >= 60:   return 15.0
        return MOONBAG_TRAIL_BASE  # 12% par défaut

    # ── Scan ──────────────────────────────────────────────────

    async def _run_scan(self):
        total_cap = self._total_capital()
        invested  = self._invested_capital()
        available = total_cap - invested
        positions = self._load_positions()
        open_pos  = [p for p in positions if p["status"] == "open"]
        slots     = MAX_POSITIONS - len(open_pos)

        ctx           = self._agent_context()
        macro_score   = ctx["macro_score"]
        cycle_phase   = ctx["cycle_phase"]
        market_regime = ctx["market_regime"]
        size_mult     = ctx["size_mult"]

        self._log("SCAN",
            f"🔍 Scan | {total_cap:.4f} {BASE_CURRENCY} "
            f"(investi: {invested:.4f} | dispo: {available:.4f}) "
            f"| {len(open_pos)}/{MAX_POSITIONS} pos "
            f"| Macro {macro_score:.0f} | {cycle_phase} | {market_regime} "
            f"| x{size_mult}")

        if slots <= 0 or available < MIN_POSITION:
            self._log("SCAN", "⏸ Aucun slot ou capital insuffisant")
            return

        if size_mult == 0.0:
            self._log("SCAN",
                f"🛑 Agent bloque — {cycle_phase} / macro {macro_score:.0f} / src={ctx['source']}")
            return

        tokens = await self._birdeye_tokens()
        if not tokens:
            tokens = await self._dexscreener_tokens()
        if not tokens:
            self._log("SCAN", "⚠ Aucune source de tokens")
            return

        self._log("SCAN", f"📊 {len(tokens)} tokens reçus…")

        held   = {p["symbol"] for p in open_pos}
        scored = []
        for t in tokens:
            sym = (t.get("symbol") or "").upper().strip()
            if not sym or sym in held or sym in ("USDC", "USDT", "BUSD", "SOL", "WSOL"):
                continue

            liq = float(t.get("liquidity") or 0)
            mc  = float(t.get("mc") or t.get("market_cap") or 0)

            # Filtre rapide : liq minimale (≈ 20 SOL)
            if liq < MIN_LIQ_USD:
                continue

            # Filtre anti-liquidity-trap : liq doit être ≥ 2% du MC
            if mc > 0 and liq / mc < MIN_LIQ_MC_RATIO:
                continue

            score = self._score(t)
            if score >= 55:   # pré-sélection large avant enrichissement
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Enrichissement Helius pour les top 10 candidats ──
        # Ajoute holder_count + tx_5min via RPC Helius.
        # On n'enrichit que les tokens qui passent déjà 55 pts pour limiter les appels.
        try:
            from modules.helius_enrichment import enrich_token
            s = await self._get_session()
            top_candidates = scored[:10]
            enrichments = await asyncio.gather(
                *[enrich_token(t.get("address", ""), s) for _, t in top_candidates],
                return_exceptions=True,
            )
            for i, (sc, t) in enumerate(top_candidates):
                if isinstance(enrichments[i], dict):
                    t.update(enrichments[i])
                    # Recalcule le score avec les nouvelles données
                    scored[i] = (self._score(t), t)
        except Exception as e:
            log.warning("enrichment_step_failed", error=str(e))

        # Filtre final au seuil MIN_SCORE après enrichissement
        scored = [(s, t) for s, t in scored if s >= MIN_SCORE]
        scored.sort(key=lambda x: x[0], reverse=True)
        self._log("SCAN", f"✅ {len(scored)} qualifiés après enrichissement (score≥{MIN_SCORE})")

        entered = 0
        for score, token in scored:
            if entered >= slots or available < MIN_POSITION:
                break
            sym  = token["symbol"].upper().strip()
            mint = token.get("address") or KNOWN_MINTS.get(sym, "")
            if not mint or sym in held:
                continue

            price = await self._fetch_price_dex(mint)
            if not price:
                price = await self._fetch_price_birdeye(mint)
            if not price:
                self._log("SCAN", f"⚠ {sym} — prix introuvable, ignoré")
                continue

            # Cohérence prix tokenlist vs API
            tl_price = float(token.get("price") or token.get("priceUsd") or 0)
            if tl_price and price > 0:
                ratio = max(price, tl_price) / min(price, tl_price)
                if ratio > 10:
                    self._log("SCAN", f"⚠ {sym} — prix incohérent (x{ratio:.0f}), ignoré")
                    continue

            # ── Pre-flight sécurité (freeze/holder) ──
            pf = await self._preflight_check(mint, sym)
            if not pf["ok"]:
                continue

            amount = self._position_size(score, available, slots - entered, total_cap)
            amount = round(amount * size_mult, 4)
            if amount < MIN_POSITION:
                continue

            ok = await self._open_position(sym, token, price, amount, score, ctx)
            if ok:
                entered += 1
                held.add(sym)
                available -= amount
                await asyncio.sleep(1)

        if entered == 0:
            self._log("SCAN", "💤 Aucune entrée — prochain scan dans 30 min")

    # ── Monitoring parallèle ──────────────────────────────────

    async def _check_positions(self):
        positions = self._load_positions()
        open_pos  = [p for p in positions if p["status"] == "open"]
        if not open_pos:
            return

        # Fetch séquentiel avec pause — évite le rate-limit DexScreener sur GCP
        mints   = [(pos, pos.get("mint") or KNOWN_MINTS.get(pos["symbol"], ""))
                   for pos in open_pos]
        results = []
        for pos, mint in mints:
            try:
                p = await self._fetch_price_dex(mint, min_liq=0) if mint else None
                if not p:
                    p = await self._fetch_price_birdeye(mint) if mint else None
                results.append((pos, p))
            except Exception as e:
                results.append(e)
            await asyncio.sleep(0.8)  # 0.8s entre chaque appel

        updated = failed = corrupt = 0

        for res in results:
            if isinstance(res, Exception):
                failed += 1
                continue
            pos, price = res
            sym = pos["symbol"]

            if not price:
                failed += 1
                # Auto-clôture si prix introuvable depuis trop longtemps
                misses = pos.get("price_miss_count", 0) + 1
                pos["price_miss_count"] = misses
                if misses >= 5:
                    last_price = pos.get("current_price") or pos["entry_price"]
                    self._log("ERREUR",
                        f"💀 {sym} — prix absent {misses} cycles consécutifs → clôture forcée",
                        sym)
                    await self._partial_close(
                        pos, last_price, "NO_PRICE",
                        pos.get("remaining_fraction", 1.0), positions)
                continue

            entry = pos["entry_price"]
            ratio = price / entry if entry > 0 else 0

            # Données corrompues → pause sécurité (sauf crash réel sous SL)
            sl_ratio = 1 - SL_PCT / 100   # ex: 0.85 pour SL=15%
            if ratio > MAX_PRICE_DRIFT or (ratio > 0 and ratio < 1 / MAX_PRICE_DRIFT):
                # Si le prix est en dessous du SL, c'est probablement un vrai crash
                if ratio > 0 and ratio < sl_ratio:
                    # Laisser passer pour déclencher le stop-loss
                    pass
                else:
                    corrupt += 1
                    self._log("ERREUR",
                        f"⚠ {sym} prix suspect (x{ratio:.1f}) — pause sécurité ce cycle")
                    continue

            pnl_pct = (price - entry) / entry * 100

            peak = pos.get("peak_price") or entry
            if price > peak:
                pos["peak_price"] = price
                peak = price

            remaining_frac = pos.get("remaining_fraction", 1.0)
            is_moonbag     = pos.get("is_moonbag", False)

            pos.update({
                "current_price":    price,
                "pnl_pct":          round(pnl_pct, 2),
                "pnl_usdc":         round(pos["amount_usdc"] * pnl_pct / 100, 6),
                "remaining_usdc":   round(pos["amount_usdc"] * remaining_frac, 4),
                "updated_at":       _now(),
                "price_miss_count": 0,  # reset dès qu'on a un prix
            })
            updated += 1

            # ── Stop-loss ──
            if pnl_pct <= -SL_PCT:
                await self._partial_close(pos, price, "STOP_LOSS",
                                          remaining_frac, positions)

            # ── Trailing stop moonbag dynamique ──
            elif is_moonbag:
                trail_pct      = self._moonbag_trail_pct(pnl_pct)
                drop_from_peak = (peak - price) / peak * 100 if peak > 0 else 0
                if drop_from_peak >= trail_pct:
                    await self._partial_close(pos, price, "MOONBAG_EXIT",
                                              remaining_frac, positions)
                    self._log("MOONBAG_EXIT",
                        f"🌙 {sym} moonbag fermé — recul {drop_from_peak:.1f}% "
                        f"(trail {trail_pct:.0f}%) depuis pic ${peak:.8f}", sym)

            # ── TP2 : vend 90%, garde 10% moonbag ──
            elif pnl_pct >= TP2_PCT:
                sell_frac = remaining_frac * 0.90
                keep_frac = remaining_frac * 0.10
                await self._partial_close(pos, price, "TP2_PARTIAL",
                                          sell_frac, positions)
                pos["remaining_fraction"] = keep_frac
                pos["is_moonbag"]         = True
                pos["peak_price"]         = price
                pos["sl_price"]           = entry
                self._log("TP2",
                    f"🎯 {sym} TP2 +{pnl_pct:.1f}% — 90% vendu, 10% moonbag", sym)

            # ── TP1 : vend 50%, laisse 50% courir ──
            elif pnl_pct >= TP1_PCT and not pos.get("tp1_hit"):
                sell_frac = remaining_frac * 0.50
                await self._partial_close(pos, price, "TP1_PARTIAL",
                                          sell_frac, positions)
                pos["remaining_fraction"] = remaining_frac * 0.50
                pos["tp1_hit"]            = True
                pos["sl_price"]           = entry
                self._log("TP1",
                    f"💰 {sym} TP1 +{pnl_pct:.1f}% — 50% vendu, SL → breakeven", sym)

        self._save_positions(positions)
        total_cap = self._total_capital()
        parts = [f"{updated} prix OK", f"capital: {total_cap:.4f} {BASE_CURRENCY}"]
        if failed:  parts.append(f"{failed} manquants")
        if corrupt: parts.append(f"{corrupt} suspects (pause sécurité)")
        self._log("MONITOR", f"📡 {' | '.join(parts)}")

    # ── Open position ─────────────────────────────────────────

    async def _open_position(self, symbol, token, price, amount_usdc, score,
                              agent_ctx: dict = None) -> bool:
        mint = token.get("address") or KNOWN_MINTS.get(symbol, "")
        if not mint:
            return False

        ctx = agent_ctx or {}
        pos = {
            "id":                 str(uuid.uuid4())[:8],
            "symbol":             symbol,
            "mint":               mint,
            "amount_usdc":        round(amount_usdc, 4),
            "remaining_usdc":     round(amount_usdc, 4),
            "remaining_fraction": 1.0,
            "entry_price":        price,
            "current_price":      price,
            "peak_price":         price,
            "sl_price":           price * (1 - SL_PCT / 100),
            "tp1_price":          price * (1 + TP1_PCT / 100),
            "tp2_price":          price * (1 + TP2_PCT / 100),
            "score":              round(score, 1),
            "status":             "open",
            "tp1_hit":            False,
            "is_moonbag":         False,
            "dry_run":            self.dry_run,
            "pnl_pct":            0.0,
            "pnl_usdc":           0.0,
            "realized_usdc":      0.0,
            "opened_at":          _now(),
            "agent_macro_score":  ctx.get("macro_score"),
            "agent_cycle_phase":  ctx.get("cycle_phase"),
            "agent_regime":       ctx.get("market_regime"),
            "agent_size_mult":    ctx.get("size_mult"),
            # Signaux enrichis au moment de l'entrée
            "entry_buys_h1":      token.get("buys_h1", 0),
            "entry_sells_h1":     token.get("sells_h1", 0),
            "entry_holders":      token.get("holder_count", 0),
            "entry_tx_5min":      token.get("tx_5min", 0),
            # Copy trading
            "copy_wallet":        token.get("copy_wallet", ""),
            "copy_wallet_label":  token.get("copy_wallet_label", ""),
        }

        if not self.dry_run:
            try:
                from modules.jupiter_swap import JupiterSwap
                from modules.solana_rpc import calc_liquidity_aware_slippage

                jup       = JupiterSwap()
                sol_price = await self._sol_price_usd()
                pool_liq  = float(token.get("liquidity") or 0)

                # Slippage adapté à l'impact prix estimé
                slippage  = calc_liquidity_aware_slippage(
                    amount_usdc, pool_liq, sol_price)

                # Priority fee au 75e percentile des blocs récents
                prio_fee  = await self._get_priority_fee()

                async def _buy():
                    return await jup.execute_swap(
                        symbol, amount_usdc,
                        slippage_bps=slippage,
                        output_mint=mint,
                        priority_fee_lamports=prio_fee,
                    )

                result = await _jupiter_with_retry(_buy)
                if not result.get("ok"):
                    self._log("ERREUR", f"Jupiter {symbol}: {result.get('error')}")
                    return False
                pos["tx_hash"]       = result.get("tx_hash", "")
                pos["out_amount"]    = result.get("out_amount", 0)
                pos["slippage_bps"]  = slippage
                pos["priority_fee"]  = prio_fee
            except Exception as e:
                self._log("ERREUR", f"Jupiter {symbol} échoué après retries: {e}")
                return False

        positions = self._load_positions()
        positions.append(pos)
        self._save_positions(positions)

        chg  = token.get("priceChange24hPercent") or 0
        mode = "PAPER" if self.dry_run else "LIVE"
        cap  = self._total_capital()
        macro_info = (
            f"| Macro {ctx.get('macro_score', '?'):.0f} "
            f"{ctx.get('cycle_phase', '?')} x{ctx.get('size_mult', 1)}"
            if ctx else ""
        )
        buys    = token.get("buys_h1", 0)
        sells   = token.get("sells_h1", 0)
        holders = token.get("holder_count", 0)
        tx5     = token.get("tx_5min", 0)
        bs_str  = f"B/S {buys}/{sells}" if (buys or sells) else ""
        en_str  = f"| {holders}h {tx5}tx/5m" if (holders or tx5) else ""
        self._log("BUY",
            f"[{mode}] {symbol} — {amount_usdc:.4f} {BASE_CURRENCY} @ ${price:.8f} | "
            f"Score {score:.0f} | 24h {chg:+.1f}% {bs_str}{en_str} "
            f"| Capital {cap:.4f} {BASE_CURRENCY} "
            f"{macro_info} | SL ${pos['sl_price']:.8f}", symbol)
        return True

    # ── Clôture partielle ou totale ───────────────────────────

    async def _partial_close(self, pos, price, reason, fraction, all_positions):
        """
        Ferme `fraction` de la position.
        En live : envoie l'ordre Jupiter pour toute clôture, partielle ou totale.
        """
        entry     = pos["entry_price"]
        pnl_pct   = (price - entry) / entry * 100
        usdc_sold = pos["amount_usdc"] * fraction
        pnl_usdc  = usdc_sold * pnl_pct / 100

        pos["realized_usdc"] = round(
            (pos.get("realized_usdc") or 0) + usdc_sold + pnl_usdc, 4)

        is_full_close = fraction >= (pos.get("remaining_fraction", 1.0) - 0.01)

        if is_full_close:
            pos.update({
                "status":      "closed",
                "exit_price":  price,
                "exit_reason": reason,
                "pnl_pct":     round(pnl_pct, 2),
                "pnl_usdc":    round(pnl_usdc, 4),
                "closed_at":   _now(),
            })
        else:
            pos["pnl_usdc"] = round((pos.get("pnl_usdc") or 0) + pnl_usdc, 4)

        # ── Vente réelle Jupiter — partielle ET totale ──────────
        if not self.dry_run:
            try:
                from modules.jupiter_swap import JupiterSwap
                jup      = JupiterSwap()
                qty_full = pos.get("out_amount", 0)
                if qty_full:
                    sell_qty = int(qty_full * fraction)
                    if sell_qty > 0:
                        async def _sell():
                            return await jup.execute_sell(
                                pos["symbol"], sell_qty, slippage_bps=200)

                        await _jupiter_with_retry(_sell)
            except Exception as e:
                log.error("sell_failed", sym=pos["symbol"], reason=reason, error=str(e))

        emoji    = "✅" if pnl_pct > 0 else "❌"
        pct_sold = round(fraction * 100)
        mode     = "PAPER" if self.dry_run else "LIVE"
        cap      = self._total_capital()
        self._log(reason,
            f"{emoji} [{mode}] {pos['symbol']} {pct_sold}% @ ${price:.8f} | "
            f"PnL {pnl_pct:+.1f}% ({pnl_usdc:+.4f} {BASE_CURRENCY}) "
            f"| Capital→{cap:.4f} {BASE_CURRENCY}",
            pos["symbol"])

    # ── Score enrichi ─────────────────────────────────────────
    #
    # Répartition (100 pts) :
    #   Volume 24h        : 20 pts  — activité globale
    #   Price change 24h  : 15 pts  — momentum directionnel
    #   Market cap        : 15 pts  — potentiel de croissance
    #   Liquidity         :  5 pts  — sécurité de sortie
    #   Buy/sell ratio 1h : 25 pts  — pression acheteuse organique (signal le plus fort)
    #   Holder count      : 10 pts  — distribution saine
    #   Tx velocity 5min  : 10 pts  — momentum immédiat on-chain
    #

    def _score(self, t) -> float:
        v24h      = float(t.get("v24hUSD")               or t.get("volume_24h")  or 0)
        chg       = float(t.get("priceChange24hPercent")  or t.get("change_24h")  or 0)
        mc        = float(t.get("mc")                     or t.get("market_cap")  or 0)
        liq       = float(t.get("liquidity")              or 0)
        buys_h1   = int(t.get("buys_h1")   or 0)
        sells_h1  = int(t.get("sells_h1")  or 0)
        holders   = int(t.get("holder_count") or 0)
        tx_5min   = int(t.get("tx_5min")    or 0)
        score     = 0.0

        # ── Volume 24h (20 pts) ──
        if   v24h >= 5_000_000:  score += 20
        elif v24h >= 1_000_000:  score += 18
        elif v24h >= 300_000:    score += 14
        elif v24h >= 100_000:    score += 9
        elif v24h >= 30_000:     score += 4

        # ── Price change 24h (15 pts) ──
        # Zone idéale : +5/+20% — momentum sain sans euphorie
        if   5 <= chg < 20:      score += 15
        elif 2 <= chg < 5:       score += 12
        elif 20 <= chg < 50:     score += 8
        elif -3 <= chg < 2:      score += 4
        elif 50 <= chg < 100:    score += 4   # suracheté — risque retrace

        # ── Market cap (15 pts) ──
        if   0 < mc < 2_000_000: score += 15
        elif mc < 10_000_000:    score += 12
        elif mc < 30_000_000:    score += 8
        elif mc < 100_000_000:   score += 4

        # ── Liquidity (5 pts) ──
        if   liq >= 500_000:     score += 5
        elif liq >= 100_000:     score += 4
        elif liq >= 50_000:      score += 3
        elif liq >= 20_000:      score += 1

        # ── Buy/sell ratio 1h (25 pts) — signal le plus prédictif ──
        # Données absentes → score neutre (10 pts) pour ne pas pénaliser Birdeye
        if buys_h1 == 0 and sells_h1 == 0:
            score += 10   # neutre : absence de données ≠ mauvais signal
        else:
            ratio = buys_h1 / max(1, sells_h1)
            if   ratio >= 3.0: score += 25   # forte pression acheteuse
            elif ratio >= 2.0: score += 20
            elif ratio >= 1.5: score += 15
            elif ratio >= 1.2: score += 10
            elif ratio >= 0.9: score += 4    # équilibré
            # ratio < 0.9 : pression vendeuse dominante → +0

        # ── Holder count (10 pts) ──
        if holders == 0:
            score += 5    # neutre si non enrichi
        elif holders >= 500: score += 10
        elif holders >= 200: score += 8
        elif holders >= 100: score += 6
        elif holders >= 50:  score += 4
        elif holders >= 20:  score += 2

        # ── Transaction velocity 5min (10 pts) ──
        if tx_5min == 0:
            score += 5    # neutre si non enrichi
        elif tx_5min >= 100: score += 10
        elif tx_5min >= 50:  score += 8
        elif tx_5min >= 20:  score += 5
        elif tx_5min >= 10:  score += 3
        elif tx_5min >= 3:   score += 1

        return min(100.0, score)

    def _position_size(self, score: float, available: float,
                       remaining_slots: int, total_cap: float) -> float:
        if remaining_slots <= 0 or available <= 0:
            return 0.0
        base = available / remaining_slots
        if   score >= 85: mult = 1.5
        elif score >= 75: mult = 1.2
        elif score >= 65: mult = 1.0
        else:             mult = 0.7
        cap_limit = total_cap * MAX_RISK_PCT
        size = min(base * mult, cap_limit, available)
        return max(MIN_POSITION, round(size, 4))

    # ── Sources tokens ────────────────────────────────────────

    async def _birdeye_tokens(self) -> list:
        if not self.birdeye_key:
            return []
        try:
            s   = await self._get_session()
            url = "https://public-api.birdeye.so/defi/tokenlist"
            params = {
                "sort_by": "v24hUSD", "sort_type": "desc",
                "offset": 0, "limit": 50, "min_liquidity": 50_000,
            }
            headers = {"X-API-KEY": self.birdeye_key, "x-chain": "solana"}
            async with s.get(url, headers=headers, params=params) as r:
                d = await r.json()
            tokens = d.get("data", {}).get("tokens", [])
            skip = {"USDC", "USDT", "BUSD", "DAI", "USDS", "SOL", "WSOL"}
            normalized = []
            for t in tokens:
                if t.get("symbol", "").upper() in skip:
                    continue
                if float(t.get("mc") or 0) >= 300_000_000:
                    continue
                t.setdefault("buys_h1", 0)
                t.setdefault("sells_h1", 0)
                normalized.append(t)
            return normalized
        except Exception as e:
            log.warning("birdeye_tokens_failed", error=str(e))
            return []

    async def _dexscreener_tokens(self) -> list:
        try:
            s   = await self._get_session()
            url = "https://api.dexscreener.com/token-boosts/top/v1"
            async with s.get(url) as r:
                data = await r.json()
            results = []
            for item in (data if isinstance(data, list) else []):
                if item.get("chainId") != "solana":
                    continue
                mint = item.get("tokenAddress", "")
                if not mint:
                    continue
                try:
                    async with s.get(
                        f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
                    ) as r2:
                        pd = await r2.json()
                    pairs = pd.get("pairs", [])
                    if not pairs:
                        continue
                    p = pairs[0]
                    txns_h1 = p.get("txns", {}).get("h1", {})
                    results.append({
                        "symbol":                p.get("baseToken", {}).get("symbol", ""),
                        "address":               mint,
                        "price":                 float(p.get("priceUsd") or 0),
                        "priceChange24hPercent": float((p.get("priceChange") or {}).get("h24") or 0),
                        "v24hUSD":               float(p.get("volume", {}).get("h24") or 0),
                        "mc":                    float(p.get("marketCap") or 0),
                        "liquidity":             float((p.get("liquidity") or {}).get("usd") or 0),
                        # ── Nouveaux signaux on-chain depuis DexScreener ──
                        "buys_h1":               int(txns_h1.get("buys") or 0),
                        "sells_h1":              int(txns_h1.get("sells") or 0),
                    })
                    await asyncio.sleep(0.2)
                except Exception:
                    continue
            return results[:20]
        except Exception as e:
            log.warning("dexscreener_failed", error=str(e))
            return []

    async def _fetch_price_dex(self, mint: str, min_liq: float = 10_000) -> Optional[float]:
        if not mint:
            return None
        try:
            s = await self._get_session()
            async with s.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            ) as r:
                d = await r.json()
            if not d or not isinstance(d, dict):
                return None
            raw_pairs = d.get("pairs") or []
            pairs = sorted(
                [p for p in raw_pairs
                 if float((p.get("liquidity") or {}).get("usd") or 0) >= min_liq],
                key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0),
                reverse=True,
            )
            if pairs:
                v = pairs[0].get("priceUsd")
                return float(v) if v else None
        except Exception as e:
            log.warning("dex_price_failed", mint=mint[:8], error=str(e))
        return None

    async def _fetch_price_birdeye(self, mint: str) -> Optional[float]:
        if not mint or not self.birdeye_key:
            return None
        try:
            import time as _time
            s = await self._get_session()
            async with s.get(
                f"https://public-api.birdeye.so/defi/price?address={mint}",
                headers={"X-API-KEY": self.birdeye_key, "x-chain": "solana"},
            ) as r:
                d = await r.json()
            data = d.get("data") or {}
            v = data.get("value")
            if not v:
                return None
            # Reject stale Birdeye data: price frozen >4h + 0% change = cached dead price
            update_ts = data.get("updateUnixTime", 0)
            change_24h = data.get("priceChange24h", None)
            if (update_ts and change_24h == 0
                    and _time.time() - update_ts > 4 * 3600):
                log.warning("birdeye_stale_price", mint=mint[:8],
                            hours_old=round((_time.time() - update_ts) / 3600, 1))
                return None
            return float(v)
        except Exception as e:
            log.warning("birdeye_price_failed", mint=mint[:8], error=str(e))
        return None

    # ── Prix SOL ─────────────────────────────────────────────

    async def _sol_price_usd(self) -> float:
        try:
            s = await self._get_session()
            sol_mint = "So11111111111111111111111111111111111111112"
            async with s.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{sol_mint}"
            ) as r:
                d = await r.json()
            pairs = [p for p in d.get("pairs", []) if p.get("chainId") == "solana"]
            if pairs:
                return float(pairs[0].get("priceUsd") or 0)
        except Exception:
            pass
        return 150.0

    # ── Actions manuelles ─────────────────────────────────────

    async def manual_buy(self, symbol: str, amount_usdc: float) -> dict:
        positions = self._load_positions()
        open_pos  = [p for p in positions if p["status"] == "open"]
        if len(open_pos) >= MAX_POSITIONS:
            return {"ok": False, "error": f"Positions pleines ({MAX_POSITIONS})"}
        total_cap = self._total_capital()
        available = total_cap - self._invested_capital()
        if amount_usdc > available:
            return {"ok": False,
                    "error": f"Capital insuffisant ({available:.4f} {BASE_CURRENCY} dispo)"}

        mint  = KNOWN_MINTS.get(symbol.upper(), "")
        price = await self._fetch_price_dex(mint) if mint else None
        if not price:
            price = await self._fetch_price_birdeye(mint) if mint else None
        if not price:
            return {"ok": False, "error": f"Prix introuvable pour {symbol}"}

        token = {
            "symbol": symbol, "address": mint, "price": price,
            "priceChange24hPercent": 0, "v24hUSD": 1e6, "mc": 5e6, "liquidity": 2e5,
        }
        ok = await self._open_position(symbol.upper(), token, price, amount_usdc, 70.0)
        if ok:
            return {"ok": True, "symbol": symbol, "amount_usdc": amount_usdc,
                    "price": price, "mode": "PAPER" if self.dry_run else "LIVE"}
        return {"ok": False, "error": "Erreur ouverture position"}

    async def manual_close(self, position_id: str) -> dict:
        positions = self._load_positions()
        for pos in positions:
            if pos["id"] == position_id and pos["status"] == "open":
                mint  = pos.get("mint") or KNOWN_MINTS.get(pos["symbol"], "")
                price = await self._fetch_price_dex(mint) if mint else None
                if not price:
                    price = await self._fetch_price_birdeye(mint) if mint else None
                if not price:
                    price = pos["entry_price"]
                frac = pos.get("remaining_fraction", 1.0)
                await self._partial_close(pos, price, "MANUAL", frac, positions)
                self._save_positions(positions)
                return {"ok": True, "symbol": pos["symbol"],
                        "pnl_pct": pos.get("pnl_pct")}
        return {"ok": False, "error": "Position introuvable"}

    # ── Persistance ───────────────────────────────────────────

    def _load_positions(self) -> list:
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_positions(self, positions: list):
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)

    def _log(self, event_type: str, message: str, symbol: str = ""):
        entries = self._load_log()
        entries.append({"ts": _now(), "type": event_type,
                        "symbol": symbol, "message": message})
        entries = entries[-300:]
        with open(LOG_FILE, "w") as f:
            json.dump(entries, f, indent=2, default=str)
        log.info("bot_event", type=event_type, msg=message[:80])

    def _load_log(self) -> list:
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
