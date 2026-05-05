"""
whale_tracker.py — Détection des mouvements baleines via order book + trades Binance.
Inspiré du système d'analyse on-chain : ordre > 50k$ = signal d'accumulation/distribution.
"""
import asyncio
import aiohttp
import os
import time
from datetime import datetime, timezone
from diskcache import Cache
import structlog

log   = structlog.get_logger()
cache = Cache("./data/cache")

WHALE_ORDER_USD = 50_000


class WhaleTracker:

    def __init__(self):
        self._session = None
        os.makedirs("data", exist_ok=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            import ssl, certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={"User-Agent": "macro_alpha/5.0"},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Order book analysis ──────────────────────────────────────

    async def detect_whale_orders_binance(self, symbol: str) -> dict:
        """Détecte les gros ordres dans le carnet d'ordres Binance."""
        try:
            session = await self._get_session()
            async with session.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": f"{symbol}USDT", "limit": 100},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                data = await r.json()

            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

            whale_bids = [
                {"price": p, "qty": q, "usd": round(p * q), "side": "buy"}
                for p, q in bids if p * q > WHALE_ORDER_USD
            ]
            whale_asks = [
                {"price": p, "qty": q, "usd": round(p * q), "side": "sell"}
                for p, q in asks if p * q > WHALE_ORDER_USD
            ]

            total_bid = sum(w["usd"] for w in whale_bids)
            total_ask = sum(w["usd"] for w in whale_asks)
            total     = total_bid + total_ask

            buy_pressure = (total_bid / total * 100) if total > 0 else 50.0
            signal = (
                "ACCUMULATION" if buy_pressure > 65 else
                "DISTRIBUTION" if buy_pressure < 35 else
                "NEUTRE"
            )

            return {
                "symbol":        symbol,
                "whale_bids":    whale_bids[:5],
                "whale_asks":    whale_asks[:5],
                "total_bid_usd": round(total_bid),
                "total_ask_usd": round(total_ask),
                "buy_pressure":  round(buy_pressure, 1),
                "signal":        signal,
                "whale_count":   len(whale_bids) + len(whale_asks),
            }
        except Exception as e:
            log.warning("whale_binance_failed", symbol=symbol, error=str(e))
            return {"symbol": symbol, "signal": "UNKNOWN", "buy_pressure": 50, "error": str(e)}

    # ── Recent trades analysis ───────────────────────────────────

    async def analyze_recent_trades(self, symbol: str, limit: int = 500) -> dict:
        """Analyse les trades récents pour détecter les patterns whale."""
        try:
            session = await self._get_session()
            async with session.get(
                "https://api.binance.com/api/v3/trades",
                params={"symbol": f"{symbol}USDT", "limit": limit},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                trades = await r.json()

            if not trades:
                return {"symbol": symbol, "pattern": "NO_DATA"}

            amounts         = [float(t["qty"]) * float(t["price"]) for t in trades]
            avg_amount      = sum(amounts) / len(amounts)
            whale_threshold = avg_amount * 10

            whale_trades = [
                {
                    "price": float(t["price"]),
                    "qty":   float(t["qty"]),
                    "usd":   round(float(t["qty"]) * float(t["price"])),
                    "side":  "buy" if not t["isBuyerMaker"] else "sell",
                }
                for t in trades
                if float(t["qty"]) * float(t["price"]) > whale_threshold
            ]

            whale_buys  = [w for w in whale_trades if w["side"] == "buy"]
            whale_sells = [w for w in whale_trades if w["side"] == "sell"]

            if len(whale_buys) > len(whale_sells) * 2:
                pattern = "WHALE_ACCUMULATION"
            elif len(whale_sells) > len(whale_buys) * 2:
                pattern = "WHALE_DISTRIBUTION"
            else:
                pattern = "MIXED"

            return {
                "symbol":           symbol,
                "pattern":          pattern,
                "whale_trades":     len(whale_trades),
                "whale_buys":       len(whale_buys),
                "whale_sells":      len(whale_sells),
                "biggest_buy":      max((w["usd"] for w in whale_buys), default=0),
                "total_whale_vol":  sum(w["usd"] for w in whale_trades),
                "avg_trade_usd":    round(avg_amount, 2),
                "whale_threshold":  round(whale_threshold, 2),
                "recent_whales":    whale_trades[-5:],
            }
        except Exception as e:
            log.warning("trade_analysis_failed", symbol=symbol, error=str(e))
            return {"symbol": symbol, "pattern": "ERROR", "error": str(e)}

    # ── Multi-token scan ─────────────────────────────────────────

    async def scan_multiple_tokens(self, symbols: list) -> list:
        """Scan whale en parallèle sur une liste de tokens."""
        tasks = [
            asyncio.gather(
                self.detect_whale_orders_binance(sym),
                self.analyze_recent_trades(sym, limit=200),
                return_exceptions=True,
            )
            for sym in symbols[:30]
        ]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        results     = []

        for sym, raw in zip(symbols, results_raw):
            if isinstance(raw, Exception):
                continue
            order_data, trade_data = raw
            if isinstance(order_data, Exception):
                order_data = {}
            if isinstance(trade_data, Exception):
                trade_data = {}

            ob_signal     = order_data.get("signal", "NEUTRE")
            trade_pattern = trade_data.get("pattern", "MIXED")

            whale_score = 50
            if ob_signal == "ACCUMULATION":           whale_score += 25
            elif ob_signal == "DISTRIBUTION":         whale_score -= 25
            if trade_pattern == "WHALE_ACCUMULATION": whale_score += 30
            elif trade_pattern == "WHALE_DISTRIBUTION": whale_score -= 30

            whale_score = max(0, min(100, whale_score))

            results.append({
                "symbol":       sym,
                "whale_score":  whale_score,
                "ob_signal":    ob_signal,
                "trade_pattern": trade_pattern,
                "buy_pressure": order_data.get("buy_pressure", 50),
                "whale_trades": trade_data.get("whale_trades", 0),
                "biggest_buy":  trade_data.get("biggest_buy", 0),
                "signal": (
                    "SUIVRE_BALEINE" if whale_score >= 75 else
                    "SURVEILLER"     if whale_score >= 60 else
                    "NEUTRE"
                ),
            })

        results.sort(key=lambda x: x["whale_score"], reverse=True)
        return results
