"""
altcoin_screener.py — Screener Altcoins Avancé
================================================
Scoring composite + Stratégie DCA STEVE + Signal de rotation altseason.
"""

import logging
import numpy as np
import requests
from typing import Optional

logger = logging.getLogger(__name__)

BINANCE_URL  = "https://api.binance.com/api/v3/klines"
BINANCE_TICK = "https://api.binance.com/api/v3/ticker/24hr"

# ── Segments de marché ──
SEGMENTS = {
    "large_cap":  ["ETH", "BNB", "SOL"],
    "mid_cap":    ["LINK", "AAVE", "INJ", "OP", "ARB"],
    "small_cap":  ["TAO", "HYPE", "VIRTUAL"],
    "micro_cap":  ["KAS", "ALPH"],
}

WATCHLIST = ["ETH", "SOL", "BNB", "LINK", "TAO", "ARB", "OP",
             "AAVE", "INJ", "KAS", "ALPH", "HYPE", "VIRTUAL"]

SEGMENT_COLORS = {
    "large_cap":  "#4CAF50",
    "mid_cap":    "#2196F3",
    "small_cap":  "#FF9800",
    "micro_cap":  "#F44336",
}

def _get_segment(symbol: str) -> str:
    for seg, coins in SEGMENTS.items():
        if symbol in coins:
            return seg
    return "micro_cap"


def _fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        r = requests.get(BINANCE_TICK, params={"symbol": f"{symbol}USDT"}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug(f"_fetch_ticker {symbol}: {exc}")
        return None


def _fetch_ohlcv(symbol: str, interval: str = "4h", limit: int = 100) -> Optional[list]:
    try:
        r = requests.get(BINANCE_URL, params={
            "symbol": f"{symbol}USDT", "interval": interval, "limit": limit
        }, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug(f"_fetch_ohlcv {symbol}: {exc}")
        return None


def _compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _compute_momentum(closes: list, period: int = 20) -> float:
    """Retourne le momentum en % sur N périodes."""
    if len(closes) < period + 1:
        return 0.0
    return (closes[-1] - closes[-period]) / closes[-period] * 100


def _compute_volume_trend(volumes: list, period: int = 10) -> float:
    """Ratio volume récent vs moyenne."""
    if len(volumes) < period:
        return 1.0
    recent = sum(volumes[-3:]) / 3
    avg    = sum(volumes[-period:]) / period
    return recent / avg if avg > 0 else 1.0


class AltcoinScreener:
    """
    Screener d'altcoins : scoring composite + DCA STEVE + signal de rotation.
    """

    def score_altcoin(self, symbol: str, btc_state: dict = None,
                      cycle_phase: str = "MID_BULL",
                      macro_score: float = 55.0) -> dict:
        """
        Score composite A(30%) + B(35%) + C(20%) + D(15%).
        A = Momentum & Tendance
        B = Force relative vs BTC
        C = Setup technique (RSI, volume)
        D = Contexte macro & cycle
        """
        bs = btc_state or {}
        btc_change_24h = float(bs.get("btc_change_24h", 0) or 0)
        btc_change_7d  = float(bs.get("btc_change_7d", 0) or 0)

        # ── Fetch données ──
        ticker = _fetch_ticker(symbol)
        klines_4h = _fetch_ohlcv(symbol, "4h", 100)
        klines_1d = _fetch_ohlcv(symbol, "1d", 30)

        if not ticker:
            return {"symbol": symbol, "error": "Données non disponibles", "score": 0}

        price       = float(ticker.get("lastPrice", 0) or 0)
        change_24h  = float(ticker.get("priceChangePercent", 0) or 0)
        volume_usdt = float(ticker.get("quoteVolume", 0) or 0)
        high_24h    = float(ticker.get("highPrice", 0) or 0)
        low_24h     = float(ticker.get("lowPrice", 0) or 0)

        closes_4h, volumes_4h = [], []
        if klines_4h:
            for k in klines_4h:
                try:
                    closes_4h.append(float(k[4]))
                    volumes_4h.append(float(k[5]))
                except Exception:
                    pass

        closes_1d = []
        if klines_1d:
            for k in klines_1d:
                try:
                    closes_1d.append(float(k[4]))
                except Exception:
                    pass

        # ── A : Momentum & Tendance (30%) ──
        score_a = 50.0
        change_7d  = _compute_momentum(closes_1d, 7)  if len(closes_1d) >= 8  else change_24h * 3
        change_30d = _compute_momentum(closes_1d, 20) if len(closes_1d) >= 21 else change_24h * 10

        if change_24h > 5:    score_a += 15
        elif change_24h > 2:  score_a += 8
        elif change_24h < -5: score_a -= 15
        elif change_24h < -2: score_a -= 8

        if change_7d > 15:    score_a += 20
        elif change_7d > 7:   score_a += 10
        elif change_7d < -15: score_a -= 20
        elif change_7d < -7:  score_a -= 10

        if change_30d > 30:   score_a += 15
        elif change_30d > 10: score_a += 7
        elif change_30d < -30: score_a -= 15
        elif change_30d < -10: score_a -= 7

        score_a = max(0, min(100, score_a))

        # ── B : Force relative vs BTC (35%) ──
        score_b = 50.0
        rs_24h = change_24h - btc_change_24h
        rs_7d  = change_7d  - btc_change_7d

        if rs_24h > 5:    score_b += 20
        elif rs_24h > 2:  score_b += 10
        elif rs_24h < -5: score_b -= 20
        elif rs_24h < -2: score_b -= 10

        if rs_7d > 10:    score_b += 25
        elif rs_7d > 4:   score_b += 12
        elif rs_7d < -10: score_b -= 25
        elif rs_7d < -4:  score_b -= 12

        score_b = max(0, min(100, score_b))

        # ── C : Setup technique (20%) ──
        score_c = 50.0
        if closes_4h:
            rsi = _compute_rsi(closes_4h)
            vol_trend = _compute_volume_trend(volumes_4h)

            if 45 < rsi < 65:   score_c += 20   # Zone idéale
            elif 35 < rsi < 75: score_c += 10
            elif rsi > 80:      score_c -= 20    # Suracheté
            elif rsi < 25:      score_c -= 10    # Survendu dangereux

            if vol_trend > 1.5: score_c += 20    # Volume élevé
            elif vol_trend > 1.2: score_c += 10
            elif vol_trend < 0.5: score_c -= 15

            # Proximité bas de range 24H = meilleur entry
            if price > 0 and high_24h > low_24h:
                range_pos = (price - low_24h) / (high_24h - low_24h)
                if range_pos < 0.3:  score_c += 10   # Bas de range
                elif range_pos > 0.8: score_c -= 5
        else:
            rsi = 50.0
            vol_trend = 1.0

        score_c = max(0, min(100, score_c))

        # ── D : Contexte macro & cycle (15%) ──
        score_d = 50.0
        segment = _get_segment(symbol)

        # Phase de cycle favorable aux alts
        if cycle_phase in ("MID_BULL", "LATE_BULL"):
            if segment == "large_cap":   score_d += 20
            elif segment == "mid_cap":   score_d += 25
            elif segment == "small_cap": score_d += 15
            elif segment == "micro_cap": score_d += 10

        elif cycle_phase == "EARLY_BULL":
            if segment in ("large_cap", "mid_cap"): score_d += 15
            else: score_d += 5

        elif cycle_phase in ("DISTRIBUTION", "BEAR"):
            score_d -= 25

        if macro_score > 65:   score_d += 15
        elif macro_score > 50: score_d += 5
        elif macro_score < 35: score_d -= 20
        elif macro_score < 45: score_d -= 10

        score_d = max(0, min(100, score_d))

        # ── Score composite ──
        composite = (score_a * 0.30 + score_b * 0.35 +
                     score_c * 0.20 + score_d * 0.15)

        # ── Grade ──
        if composite >= 75:   grade = "A+"
        elif composite >= 65: grade = "A"
        elif composite >= 55: grade = "B"
        elif composite >= 45: grade = "C"
        else:                 grade = "D"

        # ── Conviction ──
        if composite >= 70:   conviction = "FORTE"
        elif composite >= 55: conviction = "MODEREE"
        else:                 conviction = "FAIBLE"

        return {
            "symbol":      symbol,
            "price":       round(price, 6) if price < 1 else round(price, 2),
            "segment":     segment,
            "segment_color": SEGMENT_COLORS.get(segment, "#9E9E9E"),
            "composite_score": round(composite, 1),
            "grade":       grade,
            "conviction":  conviction,
            "breakdown": {
                "momentum_trend": round(score_a, 1),
                "relative_strength": round(score_b, 1),
                "technical_setup":   round(score_c, 1),
                "macro_cycle":       round(score_d, 1),
            },
            "metrics": {
                "change_24h":  round(change_24h, 2),
                "change_7d":   round(change_7d, 2),
                "rs_vs_btc_24h": round(rs_24h, 2),
                "rs_vs_btc_7d":  round(rs_7d, 2),
                "rsi_4h":        round(rsi, 1),
                "volume_ratio":  round(vol_trend, 2),
                "volume_usdt_24h": round(volume_usdt / 1e6, 1),  # en M$
            },
        }

    def get_dca_plan(self, symbol: str, available_capital: float,
                     conviction: str = "MODEREE",
                     current_price: float = 0.0,
                     entry_zone: dict = None) -> dict:
        """
        Stratégie STEVE : DCA progressif en 4 tranches.
        Conviction FORTE → allocations agressives
        Conviction MODEREE → allocations standard
        Conviction FAIBLE → attente / petite position
        """
        ez = entry_zone or {}
        zone_low  = float(ez.get("zone_low",  current_price * 0.97) or current_price * 0.97)
        zone_high = float(ez.get("zone_high", current_price * 1.00) or current_price)

        # Allocation totale selon conviction
        alloc_map = {
            "FORTE":   0.20,   # 20% du capital disponible
            "MODEREE": 0.12,   # 12%
            "FAIBLE":  0.05,   # 5%
        }
        total_alloc_pct = alloc_map.get(conviction, 0.10)
        total_capital   = available_capital * total_alloc_pct

        # ── 4 ordres STEVE ──
        # Ordre 1 : 35% ici maintenant (ou légèrement sous prix actuel)
        # Ordre 2 : 30% à -3%
        # Ordre 3 : 25% à -6%
        # Ordre 4 : 10% à -10% (sécurité bas de cycle)
        entry1 = min(current_price, zone_high) if current_price > 0 else zone_high
        entry2 = entry1 * 0.97
        entry3 = entry1 * 0.94
        entry4 = entry1 * 0.90

        qty1 = total_capital * 0.35 / entry1 if entry1 > 0 else 0
        qty2 = total_capital * 0.30 / entry2 if entry2 > 0 else 0
        qty3 = total_capital * 0.25 / entry3 if entry3 > 0 else 0
        qty4 = total_capital * 0.10 / entry4 if entry4 > 0 else 0

        avg_entry = (entry1 * 0.35 + entry2 * 0.30 +
                     entry3 * 0.25 + entry4 * 0.10)

        # ── TP et Stop Loss ──
        stop_loss    = entry4 * 0.93        # 7% sous ordre 4
        tp1          = avg_entry * 1.25     # +25% (1ère tranche 40%)
        tp2          = avg_entry * 1.60     # +60% (2ème tranche 35%)
        tp3          = avg_entry * 2.50     # +150% (3ème tranche 25%)

        # ── Déclencheur rotation : si BTC.D chute > 2% en 24H → accélérer ──
        rotation_trigger = "BTC.D < 54% ou chute BTC.D > 2% en 24H"

        precision = 6 if current_price < 1 else (4 if current_price < 10 else 2)

        return {
            "symbol":           symbol,
            "strategy":         "STEVE DCA",
            "conviction":       conviction,
            "total_allocation_pct": round(total_alloc_pct * 100, 1),
            "total_capital_usd":    round(total_capital, 2),
            "avg_entry_price":      round(avg_entry, precision),
            "orders": [
                {"order": 1, "price": round(entry1, precision), "allocation_pct": 35,
                 "usdt": round(total_capital * 0.35, 2), "qty": round(qty1, 4),
                 "type": "MARKET/LIMIT", "note": "Entrée immédiate"},
                {"order": 2, "price": round(entry2, precision), "allocation_pct": 30,
                 "usdt": round(total_capital * 0.30, 2), "qty": round(qty2, 4),
                 "type": "LIMIT", "note": "-3% rechargement"},
                {"order": 3, "price": round(entry3, precision), "allocation_pct": 25,
                 "usdt": round(total_capital * 0.25, 2), "qty": round(qty3, 4),
                 "type": "LIMIT", "note": "-6% accumulation"},
                {"order": 4, "price": round(entry4, precision), "allocation_pct": 10,
                 "usdt": round(total_capital * 0.10, 2), "qty": round(qty4, 4),
                 "type": "LIMIT", "note": "-10% filet sécurité"},
            ],
            "risk_management": {
                "stop_loss":     round(stop_loss, precision),
                "stop_loss_pct": round((stop_loss - avg_entry) / avg_entry * 100, 2),
                "tp1": {"price": round(tp1, precision), "allocation": 0.40, "gain_pct": 25},
                "tp2": {"price": round(tp2, precision), "allocation": 0.35, "gain_pct": 60},
                "tp3": {"price": round(tp3, precision), "allocation": 0.25, "gain_pct": 150},
                "trailing_after": "TP2 → trailing 15% sur le reste",
            },
            "rotation_trigger":  rotation_trigger,
            "max_loss_usd":      round(total_capital * abs((stop_loss - avg_entry) / avg_entry), 2),
        }

    def compute_altseason_rotation_signal(self, btc_data: dict,
                                           total3_data: dict) -> dict:
        """
        4 conditions pour signal de rotation altseason.
        Retourne un score 0-4 + signal FORT/MODERE/FAIBLE/ABSENT.
        """
        btc_d      = float(total3_data.get("btc_dominance", 60) or 60)
        total3_chg = float(total3_data.get("total3_change_24h", 0) or 0)
        btc_chg24h = float(btc_data.get("btc_change_24h", 0) or 0)
        btc_chg7d  = float(btc_data.get("btc_change_7d", 0) or 0)
        btc_cycle  = btc_data.get("btc_cycle_phase", "UNKNOWN")
        alt_dom    = float(total3_data.get("altcoin_dominance", 30) or 30)

        conditions = []
        score = 0

        # ── Condition 1 : BTC.D sous 54% ou en baisse forte ──
        btc_d_prev = btc_d / (1 + btc_chg24h / 100) if btc_chg24h != -100 else btc_d
        btc_d_change = btc_d - btc_d_prev
        if btc_d < 52:
            score += 1
            conditions.append({"condition": "BTC.D < 52%",
                                "value": f"{btc_d:.1f}%", "met": True})
        elif btc_d < 56 and btc_d_change < -0.5:
            score += 1
            conditions.append({"condition": "BTC.D baisse > 0.5pt",
                                "value": f"{btc_d:.1f}% (Δ{btc_d_change:+.2f})", "met": True})
        else:
            conditions.append({"condition": "BTC.D sous contrôle",
                                "value": f"{btc_d:.1f}%", "met": False})

        # ── Condition 2 : Total3 surperforme BTC ──
        rs_total3_vs_btc = total3_chg - btc_chg24h
        if rs_total3_vs_btc > 3:
            score += 1
            conditions.append({"condition": "Total3 surperforme BTC > 3%",
                                "value": f"RS={rs_total3_vs_btc:+.1f}%", "met": True})
        elif rs_total3_vs_btc > 1:
            score += 0.5
            conditions.append({"condition": "Total3 surperforme BTC (faible)",
                                "value": f"RS={rs_total3_vs_btc:+.1f}%", "met": True})
        else:
            conditions.append({"condition": "Total3 vs BTC",
                                "value": f"RS={rs_total3_vs_btc:+.1f}%", "met": False})

        # ── Condition 3 : BTC consolide après run (7J momentum faible) ──
        if -5 < btc_chg7d < 8:
            score += 1
            conditions.append({"condition": "BTC consolide (7J entre -5% et +8%)",
                                "value": f"BTC 7J={btc_chg7d:+.1f}%", "met": True})
        else:
            conditions.append({"condition": "BTC momentum trop fort ou faible",
                                "value": f"BTC 7J={btc_chg7d:+.1f}%", "met": False})

        # ── Condition 4 : Phase de cycle favorable ──
        favorable_phases = ("MID_BULL", "LATE_BULL")
        btc_cycle_norm = btc_cycle.replace(" ", "_").upper()
        if any(p in btc_cycle_norm for p in favorable_phases):
            score += 1
            conditions.append({"condition": f"Phase cycle favorable ({btc_cycle})",
                                "value": btc_cycle, "met": True})
        else:
            conditions.append({"condition": f"Phase cycle",
                                "value": btc_cycle, "met": False})

        score = round(score)  # 0-4

        if score >= 4:
            signal       = "FORT"
            signal_color = "#4CAF50"
            message      = "Rotation altseason confirmée — DCA agressif justifié"
            recommended_segments = ["mid_cap", "small_cap"]
        elif score == 3:
            signal       = "MODERE"
            signal_color = "#FF9800"
            message      = "Rotation probable — privilégier large/mid cap"
            recommended_segments = ["large_cap", "mid_cap"]
        elif score == 2:
            signal       = "FAIBLE"
            signal_color = "#2196F3"
            message      = "Signaux mitigés — positions prudentes uniquement"
            recommended_segments = ["large_cap"]
        else:
            signal       = "ABSENT"
            signal_color = "#F44336"
            message      = "Pas de rotation — rester sur BTC ou cash"
            recommended_segments = []

        return {
            "signal":       signal,
            "signal_color": signal_color,
            "score":        score,
            "score_max":    4,
            "message":      message,
            "conditions":   conditions,
            "metrics": {
                "btc_dominance":     round(btc_d, 2),
                "altcoin_dominance": round(alt_dom, 2),
                "total3_change_24h": round(total3_chg, 2),
                "rs_total3_vs_btc":  round(rs_total3_vs_btc, 2),
                "btc_change_7d":     round(btc_chg7d, 2),
                "cycle_phase":       btc_cycle,
            },
            "recommended_segments": recommended_segments,
            "watchlist_priority": self._get_priority_watchlist(score, btc_d),
        }

    def _get_priority_watchlist(self, score: int, btc_d: float) -> list:
        """Retourne la watchlist priorisée selon le score de rotation."""
        if score >= 3:
            # Altseason confirmée → small/micro en tête
            return ["TAO", "HYPE", "VIRTUAL", "INJ", "LINK",
                    "ARB", "OP", "KAS", "ALPH", "AAVE", "SOL", "ETH", "BNB"]
        elif score == 2:
            # Rotation modérée → large/mid
            return ["ETH", "SOL", "BNB", "LINK", "AAVE",
                    "ARB", "OP", "INJ", "TAO", "HYPE", "VIRTUAL", "KAS", "ALPH"]
        else:
            # Pas de rotation → seules les large cap
            return ["ETH", "BNB", "SOL"]

    def screen_all(self, btc_state: dict = None,
                   cycle_phase: str = "MID_BULL",
                   macro_score: float = 55.0,
                   top_n: int = 5) -> dict:
        """
        Lance le screening complet sur toute la watchlist.
        Retourne le top N + signal de rotation.
        """
        bs = btc_state or {}
        total3_data = bs.get("total3_data", {})

        # Signal de rotation global
        rotation = self.compute_altseason_rotation_signal(bs, total3_data)

        # Score chaque alt
        results = []
        for symbol in WATCHLIST:
            try:
                s = self.score_altcoin(symbol, bs, cycle_phase, macro_score)
                if "error" not in s:
                    results.append(s)
            except Exception as exc:
                logger.warning(f"screen_all {symbol}: {exc}")

        # Trier par score composite
        results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        top = results[:top_n]

        # Ajouter plan DCA pour le top 3
        for alt in top[:3]:
            symbol = alt["symbol"]
            price  = alt.get("price", 0)
            conv   = alt.get("conviction", "FAIBLE")
            try:
                alt["dca_plan"] = self.get_dca_plan(
                    symbol=symbol,
                    available_capital=10000,   # capital de référence (sera remplacé par vrai capital)
                    conviction=conv,
                    current_price=price,
                )
            except Exception as exc:
                logger.debug(f"DCA plan {symbol}: {exc}")

        return {
            "rotation_signal":   rotation,
            "top_picks":         top,
            "all_scores":        results,
            "screened_count":    len(results),
            "cycle_phase":       cycle_phase,
            "macro_score":       macro_score,
        }
