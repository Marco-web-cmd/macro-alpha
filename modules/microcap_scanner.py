"""
microcap_scanner.py — Scanner multi-sources robuste pour altcoins.
Source primaire : Binance (fiable, pas de rate limit agressif)
Enrichissement : CoinGecko (market cap + ATH)
DEX : DexScreener (tokens early stage)
"""
import requests
import time
import logging
import numpy as np
from datetime import datetime, timezone

try:
    from diskcache import Cache
    _disk_cache = Cache("./data/cache")
    _CACHE_OK = True
except ImportError:
    _disk_cache = None
    _CACHE_OK = False

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("microcap_scanner")
HEADERS = {"User-Agent": "macro_alpha/5.0", "Accept": "application/json"}

_cg_last_call = 0


def _make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    r = Retry(total=3, backoff_factor=1.0,
              status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


SESSION = _make_session()


class MicrocapScanner:
    """
    Scanner multi-sources robuste pour altcoins à faible market cap.
    Méthodes publiques :
      run_scan(include_dex)  — pipeline complet (nouveau)
      scan(force)            — compatibilité ancienne interface
      get_summary()          — compatibilité ancienne interface
      get_token_detail(sym)  — compatibilité ancienne interface
    """

    def __init__(self):
        self._last_scan: dict = {}
        self._scan_ts: float = 0.0

    # ──────────────────────────────────────────────────────────────
    # SOURCES DE DONNÉES
    # ──────────────────────────────────────────────────────────────

    def fetch_binance_top(self, top_n: int = 150) -> list:
        EXCLUDED = {
            "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDTUSDT",
            "FDUSDUSDT", "EURUSDT", "GBPUSDT", "AUDUSDT",
            "BTTCUSDT", "WBETHUSDT",
        }
        try:
            r = SESSION.get(
                "https://api.binance.com/api/v3/ticker/24hr", timeout=12
            )
            r.raise_for_status()
            tickers = r.json()

            usdt = [
                t for t in tickers
                if t["symbol"].endswith("USDT")
                and t["symbol"] not in EXCLUDED
                and float(t["quoteVolume"]) > 500_000
            ]
            usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

            result = []
            for t in usdt[:top_n]:
                sym   = t["symbol"].replace("USDT", "")
                price = float(t["lastPrice"])
                high  = float(t["highPrice"])
                low   = float(t["lowPrice"])
                vol   = float(t["quoteVolume"])
                chg   = float(t["priceChangePercent"])
                volat = (high - low) / price * 100 if price > 0 else 0
                result.append({
                    "symbol":     sym,
                    "pair":       t["symbol"],
                    "price":      price,
                    "change_24h": round(chg, 2),
                    "change_abs": abs(chg),
                    "volume_24h": round(vol, 0),
                    "volatility": round(volat, 2),
                    "high_24h":   high,
                    "low_24h":    low,
                    "trades_24h": int(t.get("count", 0)),
                    "source":     "binance",
                    "exchange":   "CEX",
                    "market_cap": 0,
                    "ath_pct":    -50,
                    "liquidity":  vol * 0.1,
                })

            logger.info("binance_scan_ok count=%d", len(result))
            return result
        except Exception as e:
            logger.error("binance_scan_failed error=%s", e)
            return []

    def fetch_coingecko_enrichment(self, symbols: list) -> dict:
        global _cg_last_call
        try:
            elapsed = time.time() - _cg_last_call
            if elapsed < 2.5:
                time.sleep(2.5 - elapsed)

            r = SESSION.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": 250, "page": 1, "sparkline": False,
                    "price_change_percentage": "1h,24h,7d",
                },
                timeout=15,
            )
            _cg_last_call = time.time()

            if r.status_code == 429:
                logger.warning("coingecko_rate_limit — sleeping 65s")
                time.sleep(65)
                return {}

            r.raise_for_status()
            coins = r.json()

            enriched = {}
            for c in coins:
                sym = c["symbol"].upper()
                enriched[sym] = {
                    "market_cap":      c.get("market_cap", 0) or 0,
                    "market_cap_rank": c.get("market_cap_rank"),
                    "ath_change_pct":  c.get("ath_change_percentage") or -50,
                    "change_7d":       c.get("price_change_percentage_7d_in_currency") or 0,
                    "change_1h":       c.get("price_change_percentage_1h_in_currency") or 0,
                    "cg_volume":       c.get("total_volume", 0) or 0,
                    "cg_id":           c.get("id", ""),
                }

            logger.info("coingecko_enrichment_ok count=%d", len(enriched))
            return enriched
        except Exception as e:
            logger.warning("coingecko_enrichment_failed error=%s", e)
            return {}

    def fetch_dexscreener_gems(self) -> list:
        results = []
        chains  = ["solana", "ethereum", "bsc", "base"]

        for chain in chains[:3]:
            try:
                r = SESSION.get(
                    f"https://api.dexscreener.com/latest/dex/tokens/{chain}",
                    timeout=10,
                )
                if r.status_code != 200:
                    continue

                pairs = r.json().get("pairs", [])
                for p in pairs[:30]:
                    liq = p.get("liquidity", {}).get("usd", 0) or 0
                    vol = p.get("volume", {}).get("h24", 0) or 0
                    if liq < 50_000 or vol < 20_000:
                        continue

                    base = p.get("baseToken", {})
                    results.append({
                        "symbol":     base.get("symbol", "?").upper(),
                        "name":       base.get("name", "?"),
                        "address":    base.get("address", ""),
                        "price":      float(p.get("priceUsd", 0) or 0),
                        "volume_24h": vol,
                        "liquidity":  liq,
                        "market_cap": p.get("fdv", 0) or 0,
                        "change_24h": p.get("priceChange", {}).get("h24", 0) or 0,
                        "change_1h":  p.get("priceChange", {}).get("h1", 0) or 0,
                        "chain":      chain,
                        "dex":        p.get("dexId", "?"),
                        "source":     "dexscreener",
                        "exchange":   "DEX",
                        "ath_pct":    -70,
                        "volatility": 0,
                        "trades_24h": p.get("txns", {}).get("h24", {}).get("buys", 0),
                    })

                time.sleep(0.8)
            except Exception as e:
                logger.warning("dexscreener_chain_failed chain=%s error=%s", chain, e)

        logger.info("dexscreener_ok found=%d", len(results))
        return results

    # ──────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────

    def detect_consolidation_breakout(self, token: dict) -> dict:
        vol    = token.get("volume_24h", 0)
        chg    = token.get("change_24h", 0)
        volat  = token.get("volatility", 0)
        chg_1h = token.get("change_1h", 0)
        chg_7d = token.get("change_7d", 0)

        signals = []
        score   = 0

        if chg > 3 and chg_7d is not None:
            if chg_7d < 20:
                score += 25
                signals.append(
                    f"Momentum 24H (+{chg:.1f}%) sans surextension 7J ({chg_7d:+.1f}%)"
                )

        if vol > 5_000_000:
            vol_score = min(25, int(vol / 1_000_000) * 3)
            score += vol_score
            signals.append(f"Volume fort (${vol/1e6:.1f}M)")

        if chg > 2 and chg_1h > 0.5:
            score += 20
            signals.append(f"Confirmation 1H (+{chg_1h:.1f}%) aligne 24H")
        elif chg > 5 and chg_1h < -1:
            score -= 10
            signals.append("Divergence 1H/24H — momentum en question")

        if 3 <= volat <= 12:
            score += 15
            signals.append(f"Volatilité saine ({volat:.1f}%) — pas surextension")
        elif volat > 20:
            score -= 15
            signals.append(f"Trop volatile ({volat:.1f}%) — risque élevé")

        trades = token.get("trades_24h", 0)
        if trades > 10_000:
            score += 15
            signals.append(f"{trades:,} trades 24H — liquidité réelle")

        return {
            "breakout_score":   max(0, min(100, score)),
            "breakout_signals": signals,
            "is_breaking_out":  score >= 50 and chg > 2,
        }

    def compute_gem_score(self, token: dict, cg: dict = None) -> dict:
        cg      = cg or {}
        mc      = cg.get("market_cap", token.get("market_cap", 0)) or 0
        ath_pct = cg.get("ath_change_pct", token.get("ath_pct", -50)) or -50
        vol     = token.get("volume_24h", 0)
        chg     = token.get("change_24h", 0)
        chg_7d  = cg.get("change_7d", 0) or 0
        volat   = token.get("volatility", 0)
        breakdown = {}

        # A) Volume (0-25 pts)
        if vol >= 50_000_000:   a = 25
        elif vol >= 20_000_000: a = 22
        elif vol >= 10_000_000: a = 18
        elif vol >= 5_000_000:  a = 14
        elif vol >= 2_000_000:  a = 10
        else:                   a = 3
        breakdown["volume"] = a

        # B) Momentum (0-25 pts)
        if 5 <= chg <= 30:         b = 25
        elif 2 <= chg < 5:         b = 18
        elif 30 < chg <= 50:       b = 12
        elif chg > 50:             b = 3
        elif 0 <= chg < 2:         b = 8
        else:                       b = 2
        breakdown["momentum"] = b

        # C) Breakout (0-20 pts)
        bo = self.detect_consolidation_breakout(token)
        c  = int(bo["breakout_score"] * 0.20)
        breakdown["breakout"] = c

        # D) Upside potentiel — distance ATH (0-15 pts)
        if ath_pct < -80:   d = 15
        elif ath_pct < -60: d = 12
        elif ath_pct < -40: d = 8
        elif ath_pct < -20: d = 4
        else:                d = 1
        breakdown["upside"] = d

        # E) Risk (0-15 pts)
        if mc > 500_000_000:   e = 3
        elif mc > 100_000_000: e = 10
        elif mc > 20_000_000:  e = 14
        elif mc > 5_000_000:   e = 12
        else:                   e = 4
        if volat > 25:
            e = max(0, e - 8)
        breakdown["risk"] = e

        total = min(100, max(0, sum(breakdown.values())))

        grade = (
            "A+" if total >= 85 else
            "A"  if total >= 75 else
            "B"  if total >= 65 else
            "C"  if total >= 55 else "D"
        )
        rec = (
            "GEM_SWING"  if total >= 75 and bo["is_breaking_out"] else
            "SURVEILLER" if total >= 60 else
            "OBSERVER"   if total >= 45 else
            "IGNORER"
        )

        return {
            "total_score":      round(total, 1),
            "grade":            grade,
            "recommendation":   rec,
            "breakdown":        breakdown,
            "breakout_data":    bo,
            "market_cap":       mc,
            "ath_distance_pct": round(ath_pct, 1),
        }

    # ──────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ──────────────────────────────────────────────────────────────

    def run_scan(self, include_dex: bool = True) -> dict:
        """Pipeline complet avec fallbacks. Toujours retourne un résultat."""
        logger.info("gem_scan_start")
        start = time.time()

        # Sources
        binance_tokens = self.fetch_binance_top(top_n=150)
        if not binance_tokens:
            return {
                "error": "Binance inaccessible",
                "tokens": [], "anomalies": [], "gems": [],
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        dex_tokens = self.fetch_dexscreener_gems() if include_dex else []

        # Enrichissement CoinGecko
        all_syms = list({t["symbol"] for t in binance_tokens[:80]})
        cg_data  = self.fetch_coingecko_enrichment(all_syms)

        # Fusionne sources
        all_tokens: dict = {}
        for t in binance_tokens:
            cg = cg_data.get(t["symbol"], {})
            if cg:
                t["market_cap"] = cg.get("market_cap", 0)
                t["ath_pct"]    = cg.get("ath_change_pct", -50)
                t["change_7d"]  = cg.get("change_7d", 0)
                t["change_1h"]  = cg.get("change_1h", 0)
            all_tokens[t["symbol"]] = t

        for t in dex_tokens:
            if t["symbol"] not in all_tokens:
                all_tokens[t["symbol"]] = t

        tokens = list(all_tokens.values())

        # Score
        try:
            from config import settings
            min_mc  = settings.MIN_MARKET_CAP
            max_mc  = settings.MAX_MARKET_CAP
        except Exception:
            min_mc, max_mc = 20_000_000, 500_000_000

        scored = []
        for t in tokens:
            cg = cg_data.get(t["symbol"], {})
            mc = cg.get("market_cap", t.get("market_cap", 0)) or 0

            if mc > 0 and (mc < min_mc or mc > max_mc):
                continue

            sc = self.compute_gem_score(t, cg)
            scored.append({**t, "score": sc})

        scored.sort(key=lambda x: x["score"]["total_score"], reverse=True)

        # Anomalies volume
        volumes   = [t["volume_24h"] for t in scored if t["volume_24h"] > 0]
        vol_med   = float(np.median(volumes)) if volumes else 1
        anomalies = []
        for t in scored:
            vs = t["volume_24h"] / max(vol_med, 1)
            if vs >= 2.5 and abs(t["change_24h"]) >= 5:
                anomalies.append({
                    **t,
                    "vol_spike": round(vs, 2),
                    "reason":    f"Volume {vs:.1f}x médiane + {t['change_24h']:+.1f}% 24H",
                })

        gems = [
            t for t in scored
            if t["score"]["recommendation"] == "GEM_SWING"
            and t["score"]["total_score"] >= 65
        ][:10]

        duration = round(time.time() - start, 2)
        logger.info("gem_scan_complete total=%d gems=%d anomalies=%d duration=%.1fs",
                    len(scored), len(gems), len(anomalies), duration)

        result = {
            "tokens":    scored[:30],
            "gems":      gems,
            "anomalies": anomalies[:10],
            "stats": {
                "total_scanned": len(tokens),
                "scored":        len(scored),
                "gems_found":    len(gems),
                "duration_s":    duration,
                "sources": {
                    "binance":     bool(binance_tokens),
                    "dexscreener": bool(dex_tokens),
                    "coingecko":   bool(cg_data),
                },
            },
            "ts":       datetime.now(timezone.utc).isoformat(),
            "ts_epoch": time.time(),
        }

        # Mise en cache disque
        if _CACHE_OK and _disk_cache is not None:
            try:
                _disk_cache.set("scanner_latest", result, expire=600)
            except Exception:
                pass

        self._last_scan = result
        self._scan_ts   = time.time()
        return result

    # ──────────────────────────────────────────────────────────────
    # COMPATIBILITÉ ANCIENNE INTERFACE (app.py utilise scan/get_summary)
    # ──────────────────────────────────────────────────────────────

    def scan(self, force: bool = False) -> list:
        """Ancienne interface : retourne liste de tokens scorés."""
        if not force and self._last_scan and (time.time() - self._scan_ts) < 600:
            return self._last_scan.get("tokens", [])
        result = self.run_scan(include_dex=False)
        return result.get("tokens", [])

    def get_summary(self) -> dict:
        """Résumé du dernier scan."""
        if not self._last_scan:
            return {"status": "no_scan", "tokens": 0, "gems": 0}
        stats = self._last_scan.get("stats", {})
        return {
            "status":        "ok",
            "ts":            self._last_scan.get("ts"),
            "tokens_scored": stats.get("scored", 0),
            "gems_found":    stats.get("gems_found", 0),
            "duration_s":    stats.get("duration_s", 0),
            "sources":       stats.get("sources", {}),
        }

    def get_token_detail(self, symbol: str) -> dict:
        """Détail d'un token par symbole."""
        tokens = self._last_scan.get("tokens", [])
        t = next((x for x in tokens if x.get("symbol", "").upper() == symbol.upper()), None)
        if not t:
            return {"error": f"{symbol} non trouvé — lancer d'abord un scan"}
        return t


# Compatibilité ancienne interface (utilisé dans app.py)
class SocialMomentumAnalyzer:
    """Stub de compatibilité — analyse sociale non implémentée."""
    def analyze(self, symbol: str) -> dict:
        return {"symbol": symbol, "score": 50, "note": "non_disponible"}
