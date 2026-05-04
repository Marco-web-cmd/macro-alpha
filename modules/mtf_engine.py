"""
mtf_engine.py — Moteur Multi-Timeframe
========================================
Principe : un signal n'a de valeur que si aligné sur ≥2 TF supérieures.
1H seul = bruit. 1H + 4H + 1D = haute conviction.
"""

import logging
import numpy as np
import pandas as pd
import requests
from typing import Optional

logger = logging.getLogger(__name__)

BINANCE_URL = "https://api.binance.com/api/v3/klines"

TIMEFRAMES = {
    "5m":  {"interval": "5m",  "limit": 288, "weight": 0.10},
    "15m": {"interval": "15m", "limit": 192, "weight": 0.15},
    "1h":  {"interval": "1h",  "limit": 300, "weight": 0.25},
    "4h":  {"interval": "4h",  "limit": 200, "weight": 0.30},
    "1d":  {"interval": "1d",  "limit": 150, "weight": 0.45},
    "1w":  {"interval": "1w",  "limit": 52,  "weight": 0.60},
}

# TF analysés par défaut (exclure 5m/15m trop bruiteux sauf si demandé)
DEFAULT_TFS = ["1h", "4h", "1d", "1w"]


def _fetch_ohlcv(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(BINANCE_URL, params={
            "symbol": symbol, "interval": interval, "limit": limit
        }, timeout=8)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "ct", "qav", "nt", "tbb", "tbq", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.warning(f"_fetch_ohlcv({symbol},{interval}): {exc}")
        return None


def _compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    half = close.rolling(period // 2).mean()
    full = close.rolling(period).mean()
    raw  = 2 * half - full
    return raw.rolling(int(np.sqrt(period))).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _compute_macd_hist(close: pd.Series) -> float:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    return float((macd - sig).iloc[-1]) if len(close) > 0 else 0.0


def _detect_pattern_simple(close: np.ndarray, high: np.ndarray,
                             low: np.ndarray) -> tuple[str, str]:
    """Détection légère de pattern sur les données brutes (pas pandas-ta)."""
    if len(close) < 30:
        return "none", "neutral"
    n = len(close)
    # Slope des 20 dernières bougies
    x  = np.arange(20)
    slope_c = np.polyfit(x, close[-20:], 1)[0]
    slope_h = np.polyfit(x, high[-20:],  1)[0]
    slope_l = np.polyfit(x, low[-20:],   1)[0]

    price_range = (high[-20:].max() - low[-20:].min()) / close[-1]

    # Wedge descendant (bull)
    if slope_h < 0 and slope_l < 0 and slope_h < slope_l and price_range < 0.08:
        return "falling_wedge", "bull"
    # Wedge montant (bear)
    if slope_h > 0 and slope_l > 0 and slope_l > slope_h and price_range < 0.08:
        return "rising_wedge", "bear"
    # Triangle symétrique
    if slope_h < 0 and slope_l > 0 and price_range < 0.05:
        return "symmetrical_triangle", "neutral"
    # Bull flag : fort mouvement puis consolidation légèrement descendante
    move_40 = (close[-1] - close[-40]) / close[-40] * 100 if n >= 40 else 0
    move_10 = (close[-1] - close[-10]) / close[-10] * 100 if n >= 10 else 0
    if move_40 > 12 and -6 < move_10 < 0:
        return "bull_flag", "bull"
    if move_40 < -12 and 0 < move_10 < 6:
        return "bear_flag", "bear"
    # Accumulation (range serré)
    if price_range < 0.03 and abs(slope_c) < close[-1] * 0.001:
        return "accumulation", "neutral"

    return "none", "neutral"


def _analyze_tf(df: pd.DataFrame) -> dict:
    """Analyse un unique timeframe et retourne son biais directionnel."""
    if df is None or len(df) < 30:
        return {"bias": "neutral", "trend": "flat", "rsi": 50.0,
                "macd_hist": 0.0, "pattern": "none", "pattern_dir": "neutral",
                "score": 0.5}

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    price  = float(close.iloc[-1])

    hma    = _compute_hma(close, 55)
    rsi    = _compute_rsi(close, 14)
    rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    macd_h  = _compute_macd_hist(close)

    hma_now  = float(hma.iloc[-1])  if not pd.isna(hma.iloc[-1])  else price
    hma_prev = float(hma.iloc[-4])  if len(hma) >= 4 and not pd.isna(hma.iloc[-4]) else hma_now

    # Trend via HMA slope
    hma_slope_pct = (hma_now - hma_prev) / hma_prev * 100 if hma_prev else 0
    if hma_slope_pct > 0.05 and price > hma_now:
        trend = "up"
    elif hma_slope_pct < -0.05 and price < hma_now:
        trend = "down"
    else:
        trend = "flat"

    # Score directionnel 0-1
    score = 0.5

    # Trend contribution
    if trend == "up":
        score += 0.20
    elif trend == "down":
        score -= 0.20

    # RSI contribution
    if rsi_now > 55:
        score += 0.12
    elif rsi_now < 45:
        score -= 0.12
    elif rsi_now < 30:
        score += 0.05  # oversold rebond
    elif rsi_now > 70:
        score -= 0.05  # overbought

    # MACD contribution
    if macd_h > 0:
        score += 0.10
    elif macd_h < 0:
        score -= 0.10

    # Pattern
    pat_name, pat_dir = _detect_pattern_simple(
        close.values, high.values, low.values)
    if pat_dir == "bull":
        score += 0.08
    elif pat_dir == "bear":
        score -= 0.08

    score = max(0.0, min(1.0, score))

    if score > 0.60:
        bias = "bullish"
    elif score < 0.40:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "bias":        bias,
        "trend":       trend,
        "rsi":         round(rsi_now, 1),
        "macd_hist":   round(macd_h, 4),
        "hma":         round(hma_now, 1),
        "pattern":     pat_name,
        "pattern_dir": pat_dir,
        "score":       round(score, 3),
        "price":       round(price, 2),
    }


class MultiTimeframeEngine:
    """
    Calcule le biais directionnel sur plusieurs timeframes et
    produit un score de confluence pondéré.
    """

    def get_mtf_bias(self, symbol: str = "BTCUSDT",
                      timeframes: list = None) -> dict:
        """Calcule le biais MTF et retourne un score de confluence global."""
        tfs = timeframes or DEFAULT_TFS
        by_tf = {}
        weighted_score = 0.0
        total_weight   = 0.0

        for tf in tfs:
            cfg = TIMEFRAMES.get(tf)
            if not cfg:
                continue
            df = _fetch_ohlcv(symbol, cfg["interval"], cfg["limit"])
            analysis = _analyze_tf(df)
            by_tf[tf] = analysis
            weighted_score += analysis["score"] * cfg["weight"]
            total_weight   += cfg["weight"]

        if total_weight == 0:
            return {"error": "Aucune donnée disponible"}

        confluence = weighted_score / total_weight

        # Conviction
        if confluence > 0.72:
            global_bias = "BULLISH"
            conviction  = "HIGH"
        elif confluence > 0.60:
            global_bias = "BULLISH"
            conviction  = "MEDIUM"
        elif confluence < 0.28:
            global_bias = "BEARISH"
            conviction  = "HIGH"
        elif confluence < 0.40:
            global_bias = "BEARISH"
            conviction  = "MEDIUM"
        else:
            global_bias = "NEUTRAL"
            conviction  = "LOW"

        # TF alignées vs conflictuelles (par rapport au biais majoritaire)
        aligned      = []
        conflicting  = []
        for tf, data in by_tf.items():
            tf_bull = data["score"] > 0.55
            tf_bear = data["score"] < 0.45
            if global_bias == "BULLISH" and tf_bull:
                aligned.append(tf)
            elif global_bias == "BEARISH" and tf_bear:
                aligned.append(tf)
            elif global_bias != "NEUTRAL":
                conflicting.append(tf)

        # Qualité du setup
        n_aligned = len(aligned)
        if n_aligned >= 3 and conviction == "HIGH":
            entry_quality = "A"
        elif n_aligned >= 2 and conviction in ("HIGH", "MEDIUM"):
            entry_quality = "B"
        elif n_aligned >= 1:
            entry_quality = "C"
        else:
            entry_quality = "D"

        # Si 4H ET 1D ET 1W tous alignés → A+
        high_tfs = {"4h", "1d", "1w"}
        if high_tfs.issubset(set(aligned)):
            entry_quality = "A+"

        # Narratif
        pct = round(confluence * 100, 1)
        if global_bias == "BULLISH":
            if "1h" in conflicting:
                narrative = (f"Biais haussier ({pct}%) — 1H en pullback sur support "
                             f"{', '.join(t.upper() for t in aligned)}. "
                             "Setup d'entrée en formation.")
            else:
                narrative = (f"Biais haussier fort ({pct}%) — alignement "
                             f"{', '.join(t.upper() for t in aligned)}. "
                             "Entrée possible immédiatement.")
        elif global_bias == "BEARISH":
            narrative = (f"Biais baissier ({pct}%) — alignement "
                         f"{', '.join(t.upper() for t in aligned)}. "
                         "Réduire l'exposition, pas de long.")
        else:
            narrative = (f"Biais neutre ({pct}%) — TF en conflit. "
                         "Attendre un signal directionnel clair avant d'entrer.")

        return {
            "global_bias":           global_bias,
            "confluence_score":      round(confluence, 3),
            "conviction":            conviction,
            "aligned_timeframes":    aligned,
            "conflicting_timeframes": conflicting,
            "entry_quality":         entry_quality,
            "by_timeframe":          by_tf,
            "narrative":             narrative,
            "symbol":                symbol,
        }

    def find_optimal_entry_zone(self, mtf_bias: dict, key_levels: dict,
                                 current_price: float) -> dict:
        """Identifie la zone d'entrée optimale selon la confluence MTF."""
        bias = mtf_bias.get("global_bias", "NEUTRAL")
        quality = mtf_bias.get("entry_quality", "C")
        by_tf = mtf_bias.get("by_timeframe", {})

        # Support/résistance depuis key_levels
        pp = (key_levels.get("pivot_points") or {})
        vp = (key_levels.get("volume_profile") or {})
        fib = key_levels.get("fibonacci", {}) or {}

        s1  = float(pp.get("s1", current_price * 0.97) or current_price * 0.97)
        r1  = float(pp.get("r1", current_price * 1.03) or current_price * 1.03)
        poc = float(vp.get("poc", current_price) or current_price)
        fib_618 = float((fib.get("retracements") or {}).get("0.618", current_price * 0.96) or current_price * 0.96)

        # 1H en pullback = attendre
        tf_1h = by_tf.get("1h", {})
        h1_bearish = tf_1h.get("bias") == "bearish"

        if bias == "BULLISH":
            if h1_bearish:
                # Attendre retour support 4H ≈ S1 ou Fib 0.618
                zone_low  = min(s1, fib_618) * 0.999
                zone_high = max(s1, fib_618) * 1.001
                entry_type = "LIMIT_ORDER"
            else:
                # Entrée immédiate proche du POC ou prix actuel
                zone_low  = max(poc, current_price * 0.999)
                zone_high = current_price * 1.002
                entry_type = "IMMEDIATE"
            invalidation = s1 * 0.990
        elif bias == "BEARISH":
            zone_low   = current_price * 0.998
            zone_high  = min(r1, current_price * 1.003)
            entry_type = "LIMIT_ORDER"
            invalidation = r1 * 1.010
        else:
            return {
                "entry_zone_low":  current_price,
                "entry_zone_high": current_price,
                "entry_type":      "WAIT",
                "invalidation":    current_price,
                "risk_reward":     0.0,
                "setup_quality":   "D",
            }

        entry_mid = (zone_low + zone_high) / 2
        risk_dist  = abs(entry_mid - invalidation)
        reward_dist = abs(r1 - entry_mid) if bias == "BULLISH" else abs(entry_mid - s1)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0.0

        return {
            "entry_zone_low":  round(zone_low, 2),
            "entry_zone_high": round(zone_high, 2),
            "entry_type":      entry_type,
            "invalidation":    round(invalidation, 2),
            "risk_reward":     round(rr, 2),
            "setup_quality":   quality,
        }

    def detect_divergences_mtf(self, symbol: str = "BTCUSDT") -> list:
        """Détecte les divergences RSI inter-timeframes."""
        divergences = []
        for tf in ["4h", "1d"]:
            cfg = TIMEFRAMES.get(tf)
            if not cfg:
                continue
            df = _fetch_ohlcv(symbol, cfg["interval"], cfg["limit"])
            if df is None or len(df) < 50:
                continue

            close  = df["close"]
            rsi_s  = _compute_rsi(close, 14)
            # Identifier 2 derniers hauts de prix et RSI correspondants
            prices = close.values
            rsids  = rsi_s.values

            # Minima locaux sur 10 dernières barres
            n = len(prices)
            pivot_lows_p  = []
            pivot_lows_r  = []
            for i in range(5, min(60, n - 5)):
                idx = n - 1 - i
                if idx < 5:
                    break
                if prices[idx] < prices[idx-5:idx+5].min() * 1.001:
                    pivot_lows_p.append(prices[idx])
                    if not np.isnan(rsids[idx]):
                        pivot_lows_r.append(rsids[idx])

            if len(pivot_lows_p) >= 2 and len(pivot_lows_r) >= 2:
                # Divergence haussière cachée : prix higher low, RSI lower low
                if pivot_lows_p[-1] > pivot_lows_p[-2] and pivot_lows_r[-1] < pivot_lows_r[-2]:
                    divergences.append({
                        "tf":      tf,
                        "type":    "hidden_bullish",
                        "label":   f"Divergence haussière cachée {tf.upper()}",
                        "signal":  "bull",
                    })
                # Divergence régulière haussière : prix lower low, RSI higher low
                if pivot_lows_p[-1] < pivot_lows_p[-2] and pivot_lows_r[-1] > pivot_lows_r[-2]:
                    divergences.append({
                        "tf":      tf,
                        "type":    "regular_bullish",
                        "label":   f"Divergence haussière régulière {tf.upper()}",
                        "signal":  "bull",
                    })

        return divergences
