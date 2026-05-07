"""
technical.py
Analyse technique complète sur données OHLCV.
Indicateurs : VWAP, RSI, MACD, HMA, Bollinger, ATR, ADX, Stochastic
Patterns    : H&S, H&S inversé, Double top/bottom, Bull/Bear flag,
              Consolidation, Wedge, Breakout de compression
Structure   : pandas-ta candlestick patterns, geometric structures,
              Fibonacci levels, SMC (Order Blocks, FVG, MSS), key levels
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning("pandas-ta non disponible — patterns chandelier désactivés")

try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


# ═══════════════════════════════════════════════════════════
# INDICATEURS DE BASE
# ═══════════════════════════════════════════════════════════

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast  = close.ewm(span=fast,  adjust=False).mean()
    ema_slow  = close.ewm(span=slow,  adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("macd")
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean().rename("signal")
    histogram = (macd_line - sig_line).rename("hist")
    return pd.concat([macd_line, sig_line, histogram], axis=1)


def compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    """Hull Moving Average — réduit le lag vs SMA/EMA."""
    half  = close.rolling(period // 2).mean()
    full  = close.rolling(period).mean()
    raw   = 2 * half - full
    sqrt_n = int(np.sqrt(period))
    return raw.rolling(sqrt_n).mean().rename("hma")


def compute_bollinger(close: pd.Series, period=20, std=2.0) -> pd.DataFrame:
    mid   = close.rolling(period).mean().rename("bb_mid")
    sigma = close.rolling(period).std()
    upper = (mid + std * sigma).rename("bb_up")
    lower = (mid - std * sigma).rename("bb_low")
    width = ((upper - lower) / mid * 100).rename("bb_width")
    return pd.concat([mid, upper, lower, width], axis=1)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean().rename("atr")


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.DataFrame:
    atr = compute_atr(high, low, close, period)
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    di_plus  = (dm_plus.ewm(com=period-1, min_periods=period).mean() / atr * 100).rename("di_plus")
    di_minus = (dm_minus.ewm(com=period-1, min_periods=period).mean() / atr * 100).rename("di_minus")
    dx = (((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100)
    adx = dx.ewm(com=period-1, min_periods=period).mean().rename("adx")
    return pd.concat([adx, di_plus, di_minus], axis=1)


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series) -> pd.Series:
    """VWAP cumulatif (reset quotidien si données intraday, sinon global)."""
    tp      = (high + low + close) / 3
    vwap    = (tp * volume).cumsum() / volume.cumsum()
    return vwap.rename("vwap")


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k_period=14, d_period=3) -> pd.DataFrame:
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = ((close - lowest) / (highest - lowest) * 100).rolling(3).mean().rename("stoch_k")
    d = k.rolling(d_period).mean().rename("stoch_d")
    return pd.concat([k, d], axis=1)


def find_pivots(series: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    Détecte les pivots hauts et bas locaux via scipy.signal.argrelextrema.
    Vectorisé C/NumPy — ~100x plus rapide que les boucles Python.
    """
    try:
        from scipy.signal import argrelextrema
        arr        = series.values
        highs_idx  = argrelextrema(arr, np.greater, order=window)[0]
        lows_idx   = argrelextrema(arr, np.less,    order=window)[0]
        pivot_highs = pd.Series(arr[highs_idx], index=series.index[highs_idx])
        pivot_lows  = pd.Series(arr[lows_idx],  index=series.index[lows_idx])
        return pivot_highs, pivot_lows
    except ImportError:
        # Fallback manuel si scipy absent
        pivot_highs = pd.Series(np.nan, index=series.index)
        pivot_lows  = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series) - window):
            sl = series.iloc[i - window: i + window + 1]
            if series.iloc[i] == sl.max():
                pivot_highs.iloc[i] = series.iloc[i]
            if series.iloc[i] == sl.min():
                pivot_lows.iloc[i] = series.iloc[i]
        return pivot_highs.dropna(), pivot_lows.dropna()


# ═══════════════════════════════════════════════════════════
# DÉTECTION DE DIVERGENCES RSI
# ═══════════════════════════════════════════════════════════

def detect_divergences(close: pd.Series, rsi: pd.Series,
                        lookback: int = 30) -> dict:
    """
    Divergence haussière : prix fait lower low, RSI fait higher low
    Divergence baissière : prix fait higher high, RSI fait lower high
    """
    if len(close) < lookback:
        return {"bull": False, "bear": False}

    recent_close = close.iloc[-lookback:]
    recent_rsi   = rsi.iloc[-lookback:]

    # Trouver les extremes locaux
    ph_c, pl_c = find_pivots(recent_close, window=5)
    ph_r, pl_r = find_pivots(recent_rsi,   window=5)

    bull_div = False
    bear_div = False

    # Divergence haussière : 2 derniers lows de prix + RSI
    if len(pl_c) >= 2 and len(pl_r) >= 2:
        price_ll = pl_c.iloc[-1] < pl_c.iloc[-2]   # prix lower low
        rsi_hl   = pl_r.iloc[-1] > pl_r.iloc[-2]   # RSI higher low
        bull_div = price_ll and rsi_hl

    # Divergence baissière : 2 derniers highs de prix + RSI
    if len(ph_c) >= 2 and len(ph_r) >= 2:
        price_hh = ph_c.iloc[-1] > ph_c.iloc[-2]   # prix higher high
        rsi_lh   = ph_r.iloc[-1] < ph_r.iloc[-2]   # RSI lower high
        bear_div = price_hh and rsi_lh

    return {"bull": bull_div, "bear": bear_div}


# ═══════════════════════════════════════════════════════════
# DÉTECTION DE PATTERNS CHARTISTES
# ═══════════════════════════════════════════════════════════

@dataclass
class Pattern:
    name: str
    direction: str     # "bull", "bear", "neutral"
    confidence: float  # 0-100
    description: str
    target_pct: float  # objectif de mouvement en %


def detect_head_and_shoulders(high: pd.Series, low: pd.Series,
                               close: pd.Series) -> Optional[Pattern]:
    """
    Détecte H&S (baissier) et H&S inversé (haussier).
    Méthode : identification de 3 pivots avec épaule-tête-épaule
    sur les 60 dernières bougies.
    """
    if len(close) < 60:
        return None

    ph, pl = find_pivots(close.iloc[-60:], window=6)

    # H&S inversé (bull) : 3 pivots bas avec le milieu plus bas
    if len(pl) >= 3:
        l1, l2, l3 = float(pl.iloc[-3]), float(pl.iloc[-2]), float(pl.iloc[-1])
        if l2 < l1 * 0.98 and l2 < l3 * 0.98:   # tête plus basse que les épaules
            sym    = 1 - abs(l1 - l3) / ((l1 + l3) / 2)  # symétrie épaules
            conf   = min(100, sym * 80 + 20)
            neckline = (l1 + l3) / 2
            target = (neckline - l2) / close.iloc[-1] * 100
            if conf > 50:
                return Pattern(
                    name="Tête & Épaules Inversé",
                    direction="bull",
                    confidence=round(conf, 1),
                    description=f"H&S inversé détecté — épaules: {l1:.0f}/{l3:.0f}, tête: {l2:.0f}",
                    target_pct=round(target, 1)
                )

    # H&S (bear) : 3 pivots hauts avec le milieu plus haut
    if len(ph) >= 3:
        h1, h2, h3 = float(ph.iloc[-3]), float(ph.iloc[-2]), float(ph.iloc[-1])
        if h2 > h1 * 1.02 and h2 > h3 * 1.02:
            sym  = 1 - abs(h1 - h3) / ((h1 + h3) / 2)
            conf = min(100, sym * 80 + 20)
            neckline = (h1 + h3) / 2
            target   = (h2 - neckline) / close.iloc[-1] * 100
            if conf > 50:
                return Pattern(
                    name="Tête & Épaules",
                    direction="bear",
                    confidence=round(conf, 1),
                    description=f"H&S baissier — épaules: {h1:.0f}/{h3:.0f}, tête: {h2:.0f}",
                    target_pct=round(-target, 1)
                )
    return None


def detect_double_top_bottom(close: pd.Series) -> Optional[Pattern]:
    """Double Top (bear) et Double Bottom (bull)."""
    if len(close) < 40:
        return None

    ph, pl = find_pivots(close.iloc[-40:], window=5)

    # Double Bottom
    if len(pl) >= 2:
        b1, b2 = float(pl.iloc[-2]), float(pl.iloc[-1])
        similarity = 1 - abs(b1 - b2) / ((b1 + b2) / 2)
        if similarity > 0.97:   # les deux bas sont très proches
            conf   = min(100, similarity * 100)
            target = (close.iloc[-1] - b1) / b1 * 100
            return Pattern(
                name="Double Bottom",
                direction="bull",
                confidence=round(conf, 1),
                description=f"Double bottom à ~{(b1+b2)/2:.0f}$",
                target_pct=round(target, 1)
            )

    # Double Top
    if len(ph) >= 2:
        t1, t2 = float(ph.iloc[-2]), float(ph.iloc[-1])
        similarity = 1 - abs(t1 - t2) / ((t1 + t2) / 2)
        if similarity > 0.97:
            conf   = min(100, similarity * 100)
            target = (t1 - close.iloc[-1]) / close.iloc[-1] * 100
            return Pattern(
                name="Double Top",
                direction="bear",
                confidence=round(conf, 1),
                description=f"Double top à ~{(t1+t2)/2:.0f}$",
                target_pct=round(-target, 1)
            )
    return None


def detect_flag(close: pd.Series, volume: pd.Series) -> Optional[Pattern]:
    """
    Bull Flag et Bear Flag.
    Bull flag : forte hausse puis consolidation en canal descendant.
    Bear flag : forte baisse puis rebond en canal montant.
    """
    if len(close) < 30:
        return None

    # Rechercher le mât (mouvement fort)
    move_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
    move_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100

    # Vérifier si le volume a baissé durant la consolidation (caractéristique du flag)
    vol_avg_old = float(volume.iloc[-20:-10].mean())
    vol_avg_new = float(volume.iloc[-10:].mean())
    vol_declining = vol_avg_new < vol_avg_old * 0.85

    # Bull Flag : mât haussier + consolidation + volume décroissant
    if move_20 > 15 and abs(move_10) < 5 and vol_declining:
        conf = min(100, abs(move_20) * 2 + 40)
        return Pattern(
            name="Bull Flag",
            direction="bull",
            confidence=round(conf, 1),
            description=f"Mât: +{move_20:.1f}%, consolidation actuelle: {move_10:.1f}%",
            target_pct=round(move_20 * 0.8, 1)   # target = 80% du mât
        )

    # Bear Flag
    if move_20 < -15 and abs(move_10) < 5 and vol_declining:
        conf = min(100, abs(move_20) * 2 + 40)
        return Pattern(
            name="Bear Flag",
            direction="bear",
            confidence=round(conf, 1),
            description=f"Mât: {move_20:.1f}%, consolidation actuelle: {move_10:.1f}%",
            target_pct=round(move_20 * 0.8, 1)
        )

    return None


def detect_compression(close: pd.Series, atr: pd.Series) -> Optional[Pattern]:
    """
    Détecte une compression de volatilité (coil) précédant un breakout.
    Méthode : ATR actuel bien en dessous de sa moyenne sur 30 bougies.
    """
    if len(atr) < 30:
        return None

    atr_avg = float(atr.iloc[-30:].mean())
    atr_now = float(atr.iloc[-1])
    ratio   = atr_now / atr_avg

    if ratio < 0.65:
        conf = min(100, (1 - ratio) * 200)
        return Pattern(
            name="Compression (Coil)",
            direction="neutral",
            confidence=round(conf, 1),
            description=f"Volatilité à {ratio*100:.0f}% de la moyenne — breakout imminent",
            target_pct=0.0
        )
    return None


def detect_wedge(close: pd.Series) -> Optional[Pattern]:
    """Wedge montant (bear) et descendant (bull)."""
    if len(close) < 30:
        return None

    recent = close.iloc[-30:]
    x      = np.arange(len(recent))

    # Régression linéaire des hauts et bas locaux
    highs_idx = pd.Series(range(len(recent)))[
        pd.Series(recent.values).rolling(3, center=True).max() == pd.Series(recent.values)
    ].values
    lows_idx = pd.Series(range(len(recent)))[
        pd.Series(recent.values).rolling(3, center=True).min() == pd.Series(recent.values)
    ].values

    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return None

    highs_vals = recent.values[highs_idx.astype(int)]
    lows_vals  = recent.values[lows_idx.astype(int)]

    slope_h = np.polyfit(highs_idx, highs_vals, 1)[0]
    slope_l = np.polyfit(lows_idx,  lows_vals,  1)[0]

    # Wedge convergent montant (bear)
    if slope_h > 0 and slope_l > 0 and slope_l > slope_h:
        return Pattern(
            name="Wedge Montant",
            direction="bear",
            confidence=65.0,
            description="Convergence haussière — retournement probable",
            target_pct=-8.0
        )

    # Wedge convergent descendant (bull)
    if slope_h < 0 and slope_l < 0 and slope_h < slope_l:
        return Pattern(
            name="Wedge Descendant",
            direction="bull",
            confidence=65.0,
            description="Convergence baissière — retournement probable",
            target_pct=8.0
        )

    return None


# ═══════════════════════════════════════════════════════════
# NOUVEAUX INDICATEURS AVANCÉS
# ═══════════════════════════════════════════════════════════

def compute_pi_cycle(close_daily: pd.Series) -> dict:
    """
    Pi Cycle Top Indicator.
    Signal de sommet : EMA 111 croise à la hausse 2x EMA 350.
    Signal de bottom : EMA 111 très en dessous de 2x EMA 350 après une longue descente.
    Nécessite des données daily.
    """
    if len(close_daily) < 350:
        return {"available": False, "reason": "Données insuffisantes (< 350 jours)"}

    ema111 = close_daily.ewm(span=111, adjust=False).mean()
    ema350 = close_daily.ewm(span=350, adjust=False).mean()
    ema350_2x = ema350 * 2

    e111_now  = float(ema111.iloc[-1])
    e350_2x_now = float(ema350_2x.iloc[-1])
    ratio = e111_now / e350_2x_now

    # Détection croisement récent (30 derniers jours)
    recent_111   = ema111.iloc[-30:]
    recent_350_2x = ema350_2x.iloc[-30:]
    diff_now  = float((recent_111 - recent_350_2x).iloc[-1])
    diff_prev = float((recent_111 - recent_350_2x).iloc[-2])

    top_cross    = diff_prev < 0 and diff_now >= 0   # EMA111 passe au-dessus de 2xEMA350 → TOP
    bottom_signal = ratio < 0.5                       # EMA111 < 50% de 2xEMA350 → zone BOTTOM

    if top_cross:
        signal = "TOP_SIGNAL"
        signal_fr = "Pi Cycle TOP — signal de sommet majeur"
        score_impact = -15
    elif ratio > 0.95 and ratio < 1.05:
        signal = "APPROACHING_TOP"
        signal_fr = f"Pi Cycle proche du croisement (ratio {ratio:.2f})"
        score_impact = -8
    elif bottom_signal:
        signal = "BOTTOM_ZONE"
        signal_fr = f"Pi Cycle zone BOTTOM (ratio {ratio:.2f})"
        score_impact = +10
    else:
        signal = "NEUTRAL"
        signal_fr = f"Pi Cycle neutre (ratio EMA111/2xEMA350 = {ratio:.2f})"
        score_impact = 0

    return {
        "available":    True,
        "ema111":       round(e111_now, 1),
        "ema350_2x":    round(e350_2x_now, 1),
        "ratio":        round(ratio, 3),
        "signal":       signal,
        "signal_fr":    signal_fr,
        "score_impact": score_impact,
        "top_cross":    top_cross,
        "bottom_zone":  bottom_signal,
    }


def compute_stoch_monthly(close_daily: pd.Series) -> dict:
    """
    Stochastique mensuel agrégé depuis les closes daily.
    Resample mensuel → Stoch(3 mois) sur les closes mensuels.
    Croisement sous 20 = bottom de cycle (signal rare, tous les 3-4 ans).
    """
    if len(close_daily) < 90:
        return {"available": False, "reason": "Données insuffisantes (< 90 jours)"}

    monthly = close_daily.resample("M").last().dropna()
    if len(monthly) < 14:
        return {"available": False, "reason": f"Seulement {len(monthly)} mois disponibles"}

    # High/Low mensuel (approximation via close mensuel)
    monthly_high = close_daily.resample("M").max().reindex(monthly.index)
    monthly_low  = close_daily.resample("M").min().reindex(monthly.index)

    k_period = 3
    lowest_low   = monthly_low.rolling(k_period).min()
    highest_high = monthly_high.rolling(k_period).max()
    raw_k = ((monthly - lowest_low) / (highest_high - lowest_low) * 100)
    stoch_k = raw_k.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()

    k_now = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
    d_now = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0

    k_prev = float(stoch_k.iloc[-2]) if len(stoch_k) >= 2 and not pd.isna(stoch_k.iloc[-2]) else k_now

    bottom_cross  = k_prev < 20 and k_now >= 20   # remontée depuis zone de bottom
    in_bottom     = k_now < 20
    top_cross     = k_prev > 80 and k_now <= 80   # sortie de zone de surachat
    in_top        = k_now > 80

    if bottom_cross:
        signal = "BOTTOM_CROSS"
        signal_fr = f"Stoch monthly croisement haussier depuis zone de bottom ({k_now:.0f})"
        score_impact = +12
    elif in_bottom:
        signal = "BOTTOM_ZONE"
        signal_fr = f"Stoch monthly en zone de bottom ({k_now:.0f}) — opportunité long terme"
        score_impact = +8
    elif top_cross:
        signal = "TOP_CROSS"
        signal_fr = f"Stoch monthly sortie zone de surachat ({k_now:.0f})"
        score_impact = -8
    elif in_top:
        signal = "OVERBOUGHT"
        signal_fr = f"Stoch monthly suracheté ({k_now:.0f})"
        score_impact = -5
    else:
        signal = "NEUTRAL"
        signal_fr = f"Stoch monthly neutre ({k_now:.0f})"
        score_impact = 0

    return {
        "available":    True,
        "stoch_k":      round(k_now, 1),
        "stoch_d":      round(d_now, 1),
        "signal":       signal,
        "signal_fr":    signal_fr,
        "score_impact": score_impact,
        "bottom_zone":  in_bottom or bottom_cross,
        "n_months":     len(monthly),
    }


def compute_bbw_percentile(close: pd.Series, period: int = 20,
                           history_len: int = 504) -> dict:
    """
    BB Width percentile historique sur ~2 ans de données.
    history_len = 504 bougies daily ≈ 2 ans (ou ~504h ≈ 21 jours en 1h).
    Adaptatif : utilise tout ce qui est disponible.
    """
    if len(close) < period + 10:
        return {"available": False, "reason": "Données insuffisantes"}

    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    bb_width = ((mid + 2 * sigma) - (mid - 2 * sigma)) / mid * 100
    bb_width = bb_width.dropna()

    use_len  = min(len(bb_width), history_len)
    hist_bbw = bb_width.iloc[-use_len:]
    bbw_now  = float(hist_bbw.iloc[-1])
    pctile   = float(np.sum(hist_bbw < bbw_now) / len(hist_bbw) * 100)

    if pctile <= 5:
        level = "COMPRESSION_EXTREME"
        level_fr = f"BBW compression extrême ({pctile:.0f}ème percentile) — breakout imminent"
        score_impact = +8
    elif pctile <= 15:
        level = "COMPRESSION_FORTE"
        level_fr = f"BBW compression forte ({pctile:.0f}ème percentile)"
        score_impact = +4
    elif pctile >= 90:
        level = "EXPANSION_EXTREME"
        level_fr = f"BBW expansion extrême ({pctile:.0f}ème percentile) — pic de volatilité"
        score_impact = -3
    else:
        level = "NORMAL"
        level_fr = f"BBW normal ({pctile:.0f}ème percentile)"
        score_impact = 0

    return {
        "available":    True,
        "bbw_current":  round(bbw_now, 3),
        "bbw_percentile": round(pctile, 1),
        "bbw_min_hist": round(float(hist_bbw.min()), 3),
        "bbw_max_hist": round(float(hist_bbw.max()), 3),
        "level":        level,
        "level_fr":     level_fr,
        "score_impact": score_impact,
        "history_bars": use_len,
    }


def compute_production_oscillator(close: pd.Series,
                                   miner_cost: float = 58000.0) -> dict:
    """
    Bitcoin Production Oscillator.
    Mesure la distance entre le prix BTC et le coût de production estimé des mineurs.
    oscillator = (prix - coût_production) / coût_production × 100
    Positif → au-dessus du coût (rendement mineur)
    Négatif → sous le coût (capitulation mineure probable)
    """
    if close.empty:
        return {"available": False}

    price_now = float(close.iloc[-1])
    osc_now   = (price_now - miner_cost) / miner_cost * 100

    if len(close) >= 30:
        osc_series = (close - miner_cost) / miner_cost * 100
        osc_ma30   = float(osc_series.rolling(30).mean().iloc[-1])
    else:
        osc_ma30 = osc_now

    if osc_now < -10:
        signal = "CAPITULATION"
        signal_fr = f"Prix {abs(osc_now):.0f}% sous coût de production — capitulation mineurs"
        score_impact = +12
    elif osc_now < 0:
        signal = "BELOW_COST"
        signal_fr = f"Prix légèrement sous coût de production ({osc_now:.1f}%)"
        score_impact = +6
    elif osc_now < 20:
        signal = "NEAR_COST"
        signal_fr = f"Prix proche du coût de production (+{osc_now:.1f}%)"
        score_impact = +3
    elif osc_now > 100:
        signal = "EUPHORIA"
        signal_fr = f"Prix très au-dessus du coût (+{osc_now:.0f}%) — zone de distribution"
        score_impact = -8
    else:
        signal = "NORMAL"
        signal_fr = f"Prix {osc_now:.0f}% au-dessus du coût de production"
        score_impact = 0

    return {
        "available":     True,
        "oscillator":    round(osc_now, 2),
        "oscillator_ma30": round(osc_ma30, 2),
        "miner_cost":    miner_cost,
        "price_now":     round(price_now, 0),
        "signal":        signal,
        "signal_fr":     signal_fr,
        "score_impact":  score_impact,
    }


# ═══════════════════════════════════════════════════════════
# ANALYSE TECHNIQUE COMPLÈTE
# ═══════════════════════════════════════════════════════════

_tech_cache: dict = {}   # cache incrémental par fingerprint

def full_technical_analysis(df: pd.DataFrame,
                            df_daily: Optional[pd.DataFrame] = None,
                            interval: str = "1h") -> dict:
    """
    Lance tous les indicateurs et la détection de patterns.
    df doit avoir les colonnes : open, high, low, close, volume
    df_daily (optionnel) : données daily pour Pi Cycle, Stoch monthly, BBW historique
    interval : timeframe pour adapter les paramètres des indicateurs
    Cache incrémental par fingerprint des 5 dernières clôtures.
    """
    # ── Cache incrémental (fingerprint des 5 dernières clôtures) ──
    try:
        fingerprint = hash(tuple(df["close"].iloc[-5:].round(2).tolist()))
        cache_key   = f"tech_{interval}_{fingerprint}"
        if cache_key in _tech_cache:
            return _tech_cache[cache_key]
    except Exception:
        cache_key = None

    # ── Paramètres adaptés au timeframe ──
    TF_PARAMS = {
        "5m":  {"rsi": 14, "hma": 20,  "bb": 20, "atr": 14, "adx": 14},
        "15m": {"rsi": 14, "hma": 34,  "bb": 20, "atr": 14, "adx": 14},
        "1h":  {"rsi": 14, "hma": 55,  "bb": 20, "atr": 14, "adx": 14},
        "4h":  {"rsi": 14, "hma": 100, "bb": 20, "atr": 14, "adx": 14},
        "1d":  {"rsi": 14, "hma": 55,  "bb": 20, "atr": 14, "adx": 14},
        "1w":  {"rsi": 14, "hma": 21,  "bb": 20, "atr": 14, "adx": 14},
    }
    p = TF_PARAMS.get(interval, TF_PARAMS["1h"])

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Indicateurs via pandas-ta (vectorisé en une passe) ──
    if _HAS_PANDAS_TA:
        try:
            import pandas_ta as _ta
            _df = df.copy()
            _df.ta.strategy(_ta.Strategy(
                name="macro_alpha",
                ta=[
                    {"kind": "rsi",    "length": p["rsi"]},
                    {"kind": "macd",   "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "bbands", "length": p["bb"], "std": 2.0},
                    {"kind": "atr",    "length": p["atr"]},
                    {"kind": "adx",    "length": p["adx"]},
                    {"kind": "stoch",  "k": 14, "d": 3, "smooth_k": 3},
                    {"kind": "vwap"},
                    {"kind": "hma",    "length": p["hma"]},
                    {"kind": "obv"},
                    {"kind": "cci",    "length": 20},
                    {"kind": "mfi",    "length": 14},
                    {"kind": "willr",  "length": 14},
                ]
            ))
            def _last(col):
                s = _df.get(col)
                if s is None or s.empty: return None
                v = s.iloc[-1]
                return None if pd.isna(v) else float(v)
            def _series(col):
                s = _df.get(col)
                if s is None or s.empty: return None
                return s

            rsi_s    = _series(f"RSI_{p['rsi']}") or compute_rsi(close, p["rsi"])
            macd_col = _series("MACDh_12_26_9")
            macd_l_col = _series("MACD_12_26_9")
            bb_up_col  = _series(f"BBU_{p['bb']}_2.0")
            bb_low_col = _series(f"BBL_{p['bb']}_2.0")
            bb_bw_col  = _series(f"BBB_{p['bb']}_2.0")
            bb_mid_col = _series(f"BBM_{p['bb']}_2.0")
            atr_col    = _series(f"ATRr_{p['atr']}") or _series(f"ATR_{p['atr']}")
            adx_col    = _series(f"ADX_{p['adx']}")
            stoch_k_col = _series("STOCHk_14_3_3")
            vwap_col   = _series("VWAP_D")
            hma_col    = _series(f"HMA_{p['hma']}")
            obv_col    = _series("OBV")
            cci_col    = _series("CCI_20_0.015")
            mfi_col    = _series("MFI_14")
            willr_col  = _series("WILLR_14")
        except Exception as _e:
            logger.warning(f"pandas-ta strategy failed: {_e}, falling back to manual")
            _df = None
    else:
        _df = None

    # ── Fallback manuel si pandas-ta échoue ──
    if _df is None:
        rsi_s    = compute_rsi(close, p["rsi"])
        macd_df  = compute_macd(close)
        macd_col = macd_df["hist"]
        macd_l_col = macd_df["macd"]
        bb_df    = compute_bollinger(close, p["bb"])
        bb_up_col  = bb_df["bb_up"]
        bb_low_col = bb_df["bb_low"]
        bb_bw_col  = bb_df["bb_width"]
        bb_mid_col = bb_df["bb_mid"]
        atr_col    = compute_atr(high, low, close, p["atr"])
        adx_df_fb  = compute_adx(high, low, close, p["adx"])
        adx_col    = adx_df_fb["adx"]
        stoch_df_fb = compute_stochastic(high, low, close)
        stoch_k_col = stoch_df_fb["stoch_k"]
        vwap_col   = compute_vwap(high, low, close, volume)
        hma_col    = compute_hma(close, p["hma"])
        obv_col    = (volume * close.diff().apply(lambda x: 1 if x > 0 else -1)).cumsum()
        cci_col = mfi_col = willr_col = None

    def _last_s(s):
        if s is None: return None
        try:
            v = s.iloc[-1]
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    def _prev_s(s, n=3):
        if s is None or len(s) <= n: return _last_s(s)
        try:
            v = s.iloc[-1 - n]
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    rsi_now    = _last_s(rsi_s) or 50.0
    macd_hist  = _last_s(macd_col) or 0.0
    macd_prev  = _prev_s(macd_col, 1) or 0.0
    hma_now    = _last_s(hma_col) or float(close.iloc[-1])
    hma_prev   = _prev_s(hma_col, 3) or hma_now
    bb_width   = _last_s(bb_bw_col) or 0.0
    atr_now    = _last_s(atr_col) or float(close.iloc[-1]) * 0.015
    adx_now    = _last_s(adx_col) or 20.0
    vwap_now   = _last_s(vwap_col) or float(close.iloc[-1])
    stoch_k    = _last_s(stoch_k_col) or 50.0
    close_now  = float(close.iloc[-1])

    # BB width pour le régime (calcul direct si pandas-ta n'a pas fonctionné)
    if bb_bw_col is not None and not bb_bw_col.dropna().empty:
        bb_bw_arr = bb_bw_col.dropna()
    else:
        bb_df_fb = compute_bollinger(close, p["bb"])
        bb_bw_arr = bb_df_fb["bb_width"].dropna()

    # ── Indicateurs avancés (sur données daily si disponibles) ──
    close_for_adv = df_daily["close"] if df_daily is not None else close
    pi_cycle    = compute_pi_cycle(close_for_adv)
    stoch_month = compute_stoch_monthly(close_for_adv)
    bbw_pctile  = compute_bbw_percentile(close_for_adv, period=20,
                                          history_len=730 if df_daily is not None else len(close))
    prod_osc    = compute_production_oscillator(close_for_adv)

    # ── Détection de patterns ──
    atr_for_patterns = atr_col if atr_col is not None else compute_atr(high, low, close, p["atr"])
    patterns = []
    for detector in [
        lambda: detect_head_and_shoulders(high, low, close),
        lambda: detect_double_top_bottom(close),
        lambda: detect_flag(close, volume),
        lambda: detect_compression(close, atr_for_patterns),
        lambda: detect_wedge(close),
    ]:
        pat = detector()
        if pat:
            patterns.append({
                "name":       pat.name,
                "direction":  pat.direction,
                "confidence": pat.confidence,
                "description":pat.description,
                "target_pct": pat.target_pct
            })

    # ── Divergences ──
    diverg = detect_divergences(close, rsi_s)

    # ── Régime de marché ──
    trend_up   = close_now > hma_now and hma_now > hma_prev
    trend_down = close_now < hma_now and hma_now < hma_prev
    bb_low20q  = float(bb_bw_arr.iloc[-30:].quantile(0.2)) if len(bb_bw_arr) >= 30 else bb_width
    atr_mean10 = float(atr_for_patterns.iloc[-10:].mean()) if len(atr_for_patterns) >= 10 else atr_now
    regime     = "BULL TREND"  if adx_now > 25 and trend_up   else \
                 "BEAR TREND"  if adx_now > 25 and trend_down  else \
                 "BREAKOUT"    if bb_width < bb_low20q and atr_now > atr_mean10 * 1.3 else \
                 "RANGE"       if adx_now < 20 else "TRANSITION"

    # ── Score technique (0-100) ──
    tech_score = 50.0
    tech_sigs  = []

    # Tendance
    if trend_up:
        tech_score += 12
        tech_sigs.append(("TREND", "+", f"HMA haussière ({hma_now:.0f}$)"))
    elif trend_down:
        tech_score -= 12
        tech_sigs.append(("TREND", "-", f"HMA baissière ({hma_now:.0f}$)"))

    # RSI
    if 45 < rsi_now < 70:
        tech_score += 8
        tech_sigs.append(("RSI", "+", f"RSI haussier ({rsi_now:.1f})"))
    elif rsi_now > 75:
        tech_score -= 5
        tech_sigs.append(("RSI", "-", f"RSI suracheté ({rsi_now:.1f})"))
    elif rsi_now < 30:
        tech_score += 5
        tech_sigs.append(("RSI", "+", f"RSI survendu ({rsi_now:.1f}) — rebond potentiel"))
    elif rsi_now < 45:
        tech_score -= 5
        tech_sigs.append(("RSI", "-", f"RSI baissier ({rsi_now:.1f})"))

    # MACD
    if macd_hist > 0 and macd_hist > macd_prev:
        tech_score += 8
        tech_sigs.append(("MACD", "+", "MACD histogramme haussier et croissant"))
    elif macd_hist < 0 and macd_hist < macd_prev:
        tech_score -= 8
        tech_sigs.append(("MACD", "-", "MACD histogramme baissier et décroissant"))

    # VWAP
    if close_now > vwap_now:
        tech_score += 6
        tech_sigs.append(("VWAP", "+", f"Prix au-dessus du VWAP ({vwap_now:.0f}$)"))
    else:
        tech_score -= 6
        tech_sigs.append(("VWAP", "-", f"Prix en dessous du VWAP ({vwap_now:.0f}$)"))

    # Stochastic intraday
    if stoch_k < 25:
        tech_score += 5
        tech_sigs.append(("STOCH", "+", f"Stoch survendu ({stoch_k:.1f}) — rebond potentiel"))
    elif stoch_k > 75:
        tech_score -= 5
        tech_sigs.append(("STOCH", "-", f"Stoch suracheté ({stoch_k:.1f})"))

    # Pi Cycle
    if pi_cycle.get("available"):
        impact = pi_cycle["score_impact"]
        if impact != 0:
            tech_score += impact
            d = "+" if impact > 0 else "-"
            tech_sigs.append(("PI_CYCLE", d, pi_cycle["signal_fr"]))

    # Stochastique monthly
    if stoch_month.get("available"):
        impact = stoch_month["score_impact"]
        if impact != 0:
            tech_score += impact
            d = "+" if impact > 0 else "-"
            tech_sigs.append(("STOCH_MONTHLY", d, stoch_month["signal_fr"]))

    # BBW percentile
    if bbw_pctile.get("available"):
        impact = bbw_pctile["score_impact"]
        if impact != 0:
            tech_score += impact
            d = "+" if impact > 0 else "-"
            tech_sigs.append(("BBW_HIST", d, bbw_pctile["level_fr"]))

    # Production Oscillator
    if prod_osc.get("available"):
        impact = prod_osc["score_impact"]
        if impact != 0:
            tech_score += impact
            d = "+" if impact > 0 else "-"
            tech_sigs.append(("PROD_OSC", d, prod_osc["signal_fr"]))

    # Divergences
    if diverg["bull"]:
        tech_score += 10
        tech_sigs.append(("DIVERGENCE", "+", "Divergence haussière RSI"))
    if diverg["bear"]:
        tech_score -= 10
        tech_sigs.append(("DIVERGENCE", "-", "Divergence baissière RSI"))

    # Patterns
    for pat in patterns:
        if pat["direction"] == "bull":
            tech_score += pat["confidence"] * 0.1
            tech_sigs.append(("PATTERN", "+", f"{pat['name']} ({pat['confidence']:.0f}%)"))
        elif pat["direction"] == "bear":
            tech_score -= pat["confidence"] * 0.1
            tech_sigs.append(("PATTERN", "-", f"{pat['name']} ({pat['confidence']:.0f}%)"))

    tech_score = max(0, min(100, tech_score))

    # ── Market Structure P5 ──
    market_structure = compute_market_structure(df, df_daily)

    # BBW percentile courant
    bb_pct_now = round(float((bb_bw_arr <= bb_width).mean() * 100), 1) if len(bb_bw_arr) > 0 else None

    # Historique complet pour les graphiques
    hist_n = len(df)

    def _to_list(s):
        if s is None: return [None] * hist_n
        return [None if pd.isna(v) else round(float(v), 4)
                for v in s.iloc[-hist_n:]]

    # MACD line pour le graphique (signal de la ligne MACD, pas histogramme)
    if _df is not None:
        macd_line_col = _series("MACD_12_26_9")
        macd_sig_col  = _series("MACDs_12_26_9")
    else:
        macd_df_out  = compute_macd(close)
        macd_line_col = macd_df_out["macd"]
        macd_sig_col  = macd_df_out["signal"]
        macd_col      = macd_df_out["hist"]

    result = {
        "score":            round(tech_score, 1),
        "regime":           regime,
        "interval":         interval,
        "signals":          tech_sigs,
        "patterns":         patterns,
        "divergences":      diverg,
        "market_structure": market_structure,
        "indicators": {
            "rsi":               round(rsi_now, 1),
            "macd_hist":         round(macd_hist, 2),
            "hma":               round(hma_now, 1),
            "hma_period":        p["hma"],
            "bb_width":          round(bb_width, 2),
            "bb_width_percentile": bb_pct_now,
            "atr":               round(atr_now, 1),
            "atr_pct":           round(atr_now / close_now * 100, 2) if close_now > 0 else None,
            "adx":               round(adx_now, 1),
            "vwap":              round(vwap_now, 1),
            "stoch_k":           round(stoch_k, 1),
            "cci":               round(_last_s(cci_col), 1) if _last_s(cci_col) is not None else None,
            "mfi":               round(_last_s(mfi_col), 1) if _last_s(mfi_col) is not None else None,
            "willr":             round(_last_s(willr_col), 1) if _last_s(willr_col) is not None else None,
            "obv_slope":         "hausse" if (_last_s(obv_col) or 0) > 0 else "baisse",
        },
        "advanced": {
            "pi_cycle":          pi_cycle,
            "stoch_monthly":     stoch_month,
            "bbw_percentile":    bbw_pctile,
            "production_osc":    prod_osc,
        },
        "history": {
            "timestamps": [t.isoformat() for t in df.index[-hist_n:]],
            "close":  _to_list(close),
            "rsi":    _to_list(rsi_s),
            "macd":   _to_list(macd_line_col),
            "signal": _to_list(macd_sig_col),
            "hist":   _to_list(macd_col),
            "hma":    _to_list(hma_col),
            "bb_up":  _to_list(bb_up_col),
            "bb_low": _to_list(bb_low_col),
            "vwap":   _to_list(vwap_col),
        }
    }

    # ── Cache incrémental (garde les 8 dernières clés) ──
    if cache_key:
        _tech_cache[cache_key] = result
        if len(_tech_cache) > 8:
            oldest = next(iter(_tech_cache))
            del _tech_cache[oldest]

    return result


# ═══════════════════════════════════════════════════════════
# P5 — DÉTECTION EXHAUSTIVE DE STRUCTURES DE MARCHÉ
# ═══════════════════════════════════════════════════════════

# ── Mapping pandas-ta CDL patterns → noms français + direction ──
_CDL_META = {
    "CDL_DOJI_10_0.1":        ("Doji",                  "neutral"),
    "CDL_HAMMER_10_1.5":      ("Hammer",                "bull"),
    "CDL_SHOOTINGSTAR_10_1.5":("Shooting Star",         "bear"),
    "CDL_ENGULFING":          ("Engloutissante",        "bull"),  # runtime direction
    "CDL_MORNINGSTAR_10_0.3": ("Morning Star",          "bull"),
    "CDL_EVENINGSTAR_10_0.3": ("Evening Star",          "bear"),
    "CDL_DRAGONFLYDOJI_10":   ("Dragonfly Doji",        "bull"),
    "CDL_GRAVESTONEDOJI_10":  ("Gravestone Doji",       "bear"),
    "CDL_MARUBOZU_10_1.0_1.0":("Marubozu",             "neutral"),
    "CDL_INVERTEDHAMMER_10_1.5":("Hammer Inversé",     "bull"),
    "CDL_HANGINGMAN_10_1.5":  ("Hanging Man",           "bear"),
    "CDL_SPINNINGTOP_10_0.35_0.1":("Toupie",           "neutral"),
    "CDL_HARAMI_10_0.6":      ("Harami",               "neutral"),
    "CDL_PIERCING_10_0.2":    ("Percée Haussière",     "bull"),
    "CDL_DARKCLOUDCOVER_10_0.2":("Nuage Sombre",       "bear"),
    "CDL_MORNINGDOJISTAR_10_0.1_0.3":("Morning Doji Star","bull"),
    "CDL_EVENINGDOJISTAR_10_0.1_0.3":("Evening Doji Star","bear"),
    "CDL_3WHITESOLDIERS_10_0.0_0.0":("3 Soldats Blancs","bull"),
    "CDL_3BLACKCROWS_10_0.0_0.0":("3 Corbeaux Noirs",  "bear"),
    "CDL_ABANDONEDBABY_10_0.1":("Abandoned Baby",      "neutral"),
}

# Seuil minimal de confiance pour inclure un pattern (%)
_MIN_CONFIDENCE = 55.0


def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Détecte les patterns chandelier via pandas-ta.
    Renvoie la liste des patterns actifs sur les 3 dernières bougies,
    triés par confiance décroissante, seuil minimum 55 %.
    """
    if not _HAS_PANDAS_TA or len(df) < 30:
        return []

    open_s  = df["open"]
    high_s  = df["high"]
    low_s   = df["low"]
    close_s = df["close"]

    results = []
    try:
        cdl_all = ta.cdl_pattern(open_s, high_s, low_s, close_s, name="all")
    except Exception as exc:
        logger.warning(f"pandas-ta cdl_pattern error: {exc}")
        return []

    if cdl_all is None or cdl_all.empty:
        return []

    # N'examiner que les 3 dernières bougies
    last_rows = cdl_all.tail(3)

    for col in last_rows.columns:
        series = last_rows[col]
        active = series[series != 0]
        if active.empty:
            continue

        raw_val = int(active.iloc[-1])   # 100 = bull, -100 = bear, 0 = neutral
        # Confiance basée sur l'ancienneté de la bougie (bougie actuelle = max confiance)
        bar_age = len(last_rows) - 1 - last_rows.index.get_loc(active.index[-1])
        base_conf = 80.0 - bar_age * 12.0

        meta = _CDL_META.get(col)
        if meta:
            name_fr, direction = meta
            if direction == "neutral":
                direction = "bull" if raw_val > 0 else "bear" if raw_val < 0 else "neutral"
        else:
            # Inférer depuis le nom de la colonne
            name_fr = col.replace("CDL_", "").split("_")[0].title()
            direction = "bull" if raw_val > 0 else "bear" if raw_val < 0 else "neutral"

        conf = round(base_conf, 1)
        if conf < _MIN_CONFIDENCE:
            continue

        price_now = float(close_s.iloc[-1])
        atr_val   = float(compute_atr(high_s, low_s, close_s).iloc[-1])

        results.append({
            "type":             "candlestick",
            "name":             name_fr,
            "direction":        direction,
            "confidence":       conf,
            "description":      f"Pattern {name_fr} détecté ({direction})",
            "target_price":     round(price_now * (1 + 0.02) if direction == "bull"
                                      else price_now * (1 - 0.02), 0),
            "invalidation_price": round(price_now - atr_val if direction == "bull"
                                        else price_now + atr_val, 0),
            "touches_high":     None,
            "touches_low":      None,
            "r2_score":         None,
            "lines":            None,
        })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)


def _linreg_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Régression linéaire → (slope, intercept, r2)."""
    if len(x) < 2:
        return 0.0, float(y.mean()), 0.0
    coeffs  = np.polyfit(x, y, 1)
    y_hat   = np.polyval(coeffs, x)
    ss_res  = np.sum((y - y_hat) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coeffs[0]), float(coeffs[1]), float(r2)


def _fit_trendline(prices: np.ndarray, indices: np.ndarray,
                   min_touches: int = 2, min_r2: float = 0.80
                   ) -> Optional[dict]:
    """
    Ajuste une droite sur les pivots fournis.
    Retourne dict si R²≥min_r2 et touches≥min_touches, sinon None.
    """
    if len(indices) < min_touches:
        return None
    slope, intercept, r2 = _linreg_r2(indices.astype(float), prices)
    if r2 < min_r2:
        return None
    # Valeur projetée au dernier indice
    last_idx   = float(indices[-1])
    price_last = slope * last_idx + intercept
    return {
        "slope":     round(slope, 4),
        "intercept": round(intercept, 2),
        "r2":        round(r2, 3),
        "touches":   len(indices),
        "price_now": round(price_last, 2),
    }


def detect_geometric_structures(df: pd.DataFrame,
                                  lookback: int = 120) -> list[dict]:
    """
    Détecte 17 structures géométriques via analyse de pivots.
    R² ≥ 0.80, minimum 2 touches.
    Structures: trendlines, canaux, triangles, wedges, drapeaux, pennants, rectangles,
                triple top/bottom, cup & handle, double top/bottom (géométrique).
    """
    if len(df) < 40:
        return []

    close = df["close"].values[-lookback:]
    high  = df["high"].values[-lookback:]
    low   = df["low"].values[-lookback:]
    n     = len(close)
    x_all = np.arange(n)
    price_now = float(close[-1])
    atr_val   = float(compute_atr(df["high"], df["low"], df["close"]).iloc[-1])

    # Pivots hauts et bas
    win = 5
    ph_idx, pl_idx = [], []
    for i in range(win, n - win):
        if high[i] == high[i-win:i+win+1].max():
            ph_idx.append(i)
        if low[i]  == low[i-win:i+win+1].min():
            pl_idx.append(i)

    ph_idx  = np.array(ph_idx)
    pl_idx  = np.array(pl_idx)
    ph_vals = high[ph_idx]  if len(ph_idx) else np.array([])
    pl_vals = low[pl_idx]   if len(pl_idx) else np.array([])

    structures = []

    def _add(name, direction, conf, desc, target, invalide, r2=None, lines=None,
             touches_h=None, touches_l=None):
        if conf < _MIN_CONFIDENCE:
            return
        structures.append({
            "type":             "geometric",
            "name":             name,
            "direction":        direction,
            "confidence":       round(conf, 1),
            "description":      desc,
            "target_price":     round(target, 0),
            "invalidation_price": round(invalide, 0),
            "touches_high":     touches_h,
            "touches_low":      touches_l,
            "r2_score":         round(r2, 3) if r2 else None,
            "lines":            lines,
        })

    # ── 1. Trendline haussière (support ascendant) ──
    if len(pl_idx) >= 2:
        tl = _fit_trendline(pl_vals, pl_idx)
        if tl:
            _add("Trendline Haussière", "bull",
                 min(95, 55 + tl["touches"] * 8 + tl["r2"] * 20),
                 f"Support ascendant R²={tl['r2']:.2f} ({tl['touches']} touches)",
                 price_now * 1.05, tl["price_now"] * 0.99,
                 r2=tl["r2"], touches_l=tl["touches"],
                 lines={"support": tl})

    # ── 2. Trendline baissière (résistance descendante) ──
    if len(ph_idx) >= 2:
        tl = _fit_trendline(ph_vals, ph_idx)
        if tl:
            _add("Trendline Baissière", "bear",
                 min(95, 55 + tl["touches"] * 8 + tl["r2"] * 20),
                 f"Résistance descendante R²={tl['r2']:.2f} ({tl['touches']} touches)",
                 tl["price_now"] * 1.01, price_now * 1.05,
                 r2=tl["r2"], touches_h=tl["touches"],
                 lines={"resistance": tl})

    # ── 3. Canal haussier ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_sup and tl_res:
            slope_diff = abs(tl_sup["slope"] - tl_res["slope"]) / (abs(tl_sup["slope"]) + 1e-9)
            if tl_sup["slope"] > 0 and tl_res["slope"] > 0 and slope_diff < 0.3:
                conf = min(92, 60 + (tl_sup["r2"] + tl_res["r2"]) * 15)
                _add("Canal Haussier", "bull",
                     conf,
                     f"Canal ascendant — sup R²={tl_sup['r2']:.2f}, res R²={tl_res['r2']:.2f}",
                     tl_res["price_now"], tl_sup["price_now"] * 0.99,
                     r2=(tl_sup["r2"] + tl_res["r2"]) / 2,
                     touches_h=tl_res["touches"], touches_l=tl_sup["touches"],
                     lines={"support": tl_sup, "resistance": tl_res})

    # ── 4. Canal baissier ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_sup and tl_res:
            slope_diff = abs(tl_sup["slope"] - tl_res["slope"]) / (abs(tl_sup["slope"]) + 1e-9)
            if tl_sup["slope"] < 0 and tl_res["slope"] < 0 and slope_diff < 0.3:
                conf = min(92, 60 + (tl_sup["r2"] + tl_res["r2"]) * 15)
                _add("Canal Baissier", "bear",
                     conf,
                     f"Canal descendant — sup R²={tl_sup['r2']:.2f}, res R²={tl_res['r2']:.2f}",
                     tl_sup["price_now"], tl_res["price_now"] * 1.01,
                     r2=(tl_sup["r2"] + tl_res["r2"]) / 2,
                     touches_h=tl_res["touches"], touches_l=tl_sup["touches"],
                     lines={"support": tl_sup, "resistance": tl_res})

    # ── 5. Triangle symétrique ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_sup and tl_res:
            if tl_sup["slope"] > 0 and tl_res["slope"] < 0:
                conf = min(90, 58 + (tl_sup["r2"] + tl_res["r2"]) * 16)
                _add("Triangle Symétrique", "neutral",
                     conf,
                     "Convergence bi-directionnelle — breakout imminent",
                     price_now * 1.06, price_now * 0.94,
                     r2=(tl_sup["r2"] + tl_res["r2"]) / 2,
                     touches_h=tl_res["touches"], touches_l=tl_sup["touches"],
                     lines={"support": tl_sup, "resistance": tl_res})

    # ── 6. Triangle ascendant ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        if tl_sup and tl_sup["slope"] > 0:
            # Résistance horizontale : hauts proches les uns des autres
            if len(ph_vals) >= 2:
                ph_std = np.std(ph_vals[-4:]) / np.mean(ph_vals[-4:])
                if ph_std < 0.015:
                    res_level = float(np.mean(ph_vals[-4:]))
                    conf = min(88, 60 + tl_sup["r2"] * 20 + (0.015 - ph_std) * 1000)
                    _add("Triangle Ascendant", "bull",
                         conf,
                         f"Résistance horizontale à {res_level:.0f}$, support ascendant",
                         res_level * 1.03, tl_sup["price_now"] * 0.99,
                         r2=tl_sup["r2"], touches_l=tl_sup["touches"],
                         lines={"support": tl_sup,
                                "resistance": {"price_now": res_level, "slope": 0, "r2": 0.9}})

    # ── 7. Triangle descendant ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_res and tl_res["slope"] < 0:
            if len(pl_vals) >= 2:
                pl_std = np.std(pl_vals[-4:]) / np.mean(pl_vals[-4:])
                if pl_std < 0.015:
                    sup_level = float(np.mean(pl_vals[-4:]))
                    conf = min(88, 60 + tl_res["r2"] * 20 + (0.015 - pl_std) * 1000)
                    _add("Triangle Descendant", "bear",
                         conf,
                         f"Support horizontal à {sup_level:.0f}$, résistance descendante",
                         tl_res["price_now"] * 1.01, sup_level * 0.97,
                         r2=tl_res["r2"], touches_h=tl_res["touches"],
                         lines={"support": {"price_now": sup_level, "slope": 0, "r2": 0.9},
                                "resistance": tl_res})

    # ── 8. Wedge montant (bear) ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_sup and tl_res:
            if tl_sup["slope"] > 0 and tl_res["slope"] > 0 and tl_sup["slope"] > tl_res["slope"]:
                conf = min(85, 58 + (tl_sup["r2"] + tl_res["r2"]) * 13)
                _add("Wedge Montant", "bear",
                     conf,
                     "Convergence haussière — retournement baissier probable",
                     price_now * 0.93, tl_res["price_now"] * 1.02,
                     r2=(tl_sup["r2"] + tl_res["r2"]) / 2,
                     touches_h=tl_res["touches"], touches_l=tl_sup["touches"],
                     lines={"support": tl_sup, "resistance": tl_res})

    # ── 9. Wedge descendant (bull) ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals, pl_idx)
        tl_res = _fit_trendline(ph_vals, ph_idx)
        if tl_sup and tl_res:
            if tl_sup["slope"] < 0 and tl_res["slope"] < 0 and tl_res["slope"] < tl_sup["slope"]:
                conf = min(85, 58 + (tl_sup["r2"] + tl_res["r2"]) * 13)
                _add("Wedge Descendant", "bull",
                     conf,
                     "Convergence baissière — retournement haussier probable",
                     price_now * 1.07, tl_sup["price_now"] * 0.98,
                     r2=(tl_sup["r2"] + tl_res["r2"]) / 2,
                     touches_h=tl_res["touches"], touches_l=tl_sup["touches"],
                     lines={"support": tl_sup, "resistance": tl_res})

    # ── 10. Rectangle (zone de consolidation) ──
    if len(ph_vals) >= 3 and len(pl_vals) >= 3:
        res_std = np.std(ph_vals[-6:]) / np.mean(ph_vals[-6:])
        sup_std = np.std(pl_vals[-6:]) / np.mean(pl_vals[-6:])
        if res_std < 0.012 and sup_std < 0.012:
            res_lv = float(np.mean(ph_vals[-6:]))
            sup_lv = float(np.mean(pl_vals[-6:]))
            range_pct = (res_lv - sup_lv) / sup_lv * 100
            if 2 < range_pct < 15:
                conf = min(88, 65 + (0.012 - (res_std + sup_std) / 2) * 2000)
                _add("Rectangle", "neutral",
                     conf,
                     f"Consolidation {sup_lv:.0f}$ — {res_lv:.0f}$ ({range_pct:.1f}%)",
                     res_lv * 1.03, sup_lv * 0.97,
                     lines={"support": {"price_now": sup_lv, "slope": 0, "r2": 0.95},
                             "resistance": {"price_now": res_lv, "slope": 0, "r2": 0.95}})

    # ── 11. Double Top (géométrique) ──
    if len(ph_vals) >= 2:
        t1, t2 = float(ph_vals[-2]), float(ph_vals[-1])
        sim = 1 - abs(t1 - t2) / ((t1 + t2) / 2)
        if sim > 0.97 and len(ph_idx) >= 2:
            neckline = float(low[ph_idx[-2]:ph_idx[-1]+1].min())
            conf = min(90, sim * 90)
            _add("Double Top", "bear",
                 conf,
                 f"Double sommet à ~{(t1+t2)/2:.0f}$, neckline {neckline:.0f}$",
                 neckline - (t1 - neckline), neckline * 1.01,
                 lines={"resistance": {"price_now": (t1+t2)/2, "slope": 0, "r2": sim}})

    # ── 12. Double Bottom (géométrique) ──
    if len(pl_vals) >= 2:
        b1, b2 = float(pl_vals[-2]), float(pl_vals[-1])
        sim = 1 - abs(b1 - b2) / ((b1 + b2) / 2)
        if sim > 0.97 and len(pl_idx) >= 2:
            neckline = float(high[pl_idx[-2]:pl_idx[-1]+1].max())
            conf = min(90, sim * 90)
            _add("Double Bottom", "bull",
                 conf,
                 f"Double creux à ~{(b1+b2)/2:.0f}$, neckline {neckline:.0f}$",
                 neckline + (neckline - b1), neckline * 0.99,
                 lines={"support": {"price_now": (b1+b2)/2, "slope": 0, "r2": sim}})

    # ── 13. Triple Top ──
    if len(ph_vals) >= 3:
        t1, t2, t3 = float(ph_vals[-3]), float(ph_vals[-2]), float(ph_vals[-1])
        avg = (t1 + t2 + t3) / 3
        sim = 1 - np.std([t1, t2, t3]) / avg
        if sim > 0.97:
            conf = min(92, sim * 92)
            _add("Triple Top", "bear",
                 conf,
                 f"Triple sommet à ~{avg:.0f}$ — résistance majeure",
                 price_now * 0.90, avg * 1.01,
                 touches_h=3)

    # ── 14. Triple Bottom ──
    if len(pl_vals) >= 3:
        b1, b2, b3 = float(pl_vals[-3]), float(pl_vals[-2]), float(pl_vals[-1])
        avg = (b1 + b2 + b3) / 3
        sim = 1 - np.std([b1, b2, b3]) / avg
        if sim > 0.97:
            conf = min(92, sim * 92)
            _add("Triple Bottom", "bull",
                 conf,
                 f"Triple creux à ~{avg:.0f}$ — support majeur",
                 price_now * 1.10, avg * 0.99,
                 touches_l=3)

    # ── 15. Bull Flag (géométrique) ──
    if len(df) >= 30:
        move_20 = (close[-1] - close[-20]) / close[-20] * 100
        move_10 = (close[-1] - close[-10]) / close[-10] * 100
        vol = df["volume"].values[-lookback:]
        vol_old = float(vol[-20:-10].mean())
        vol_new = float(vol[-10:].mean())
        if move_20 > 15 and abs(move_10) < 5 and vol_new < vol_old * 0.85:
            tl_flag = _fit_trendline(close[-10:], np.arange(10))
            r2_flag = tl_flag["r2"] if tl_flag else 0.5
            conf = min(88, 55 + abs(move_20) * 1.5 + r2_flag * 15)
            _add("Bull Flag", "bull",
                 conf,
                 f"Mât haussier +{move_20:.1f}%, consolidation {move_10:.1f}%",
                 price_now * (1 + abs(move_20) * 0.008),
                 price_now * 0.97)

    # ── 16. Bear Flag (géométrique) ──
    if len(df) >= 30:
        if move_20 < -15 and abs(move_10) < 5 and vol_new < vol_old * 0.85:
            conf = min(88, 55 + abs(move_20) * 1.5)
            _add("Bear Flag", "bear",
                 conf,
                 f"Mât baissier {move_20:.1f}%, rebond {move_10:.1f}%",
                 price_now * (1 - abs(move_20) * 0.008),
                 price_now * 1.03)

    # ── 17. Pennant ──
    if len(ph_idx) >= 2 and len(pl_idx) >= 2:
        tl_sup = _fit_trendline(pl_vals[-4:], pl_idx[-4:]) if len(pl_idx) >= 2 else None
        tl_res = _fit_trendline(ph_vals[-4:], ph_idx[-4:]) if len(ph_idx) >= 2 else None
        if tl_sup and tl_res:
            if tl_sup["slope"] > 0 and tl_res["slope"] < 0:
                n_bars_converge = int(abs(
                    (tl_sup["intercept"] - tl_res["intercept"]) /
                    (tl_res["slope"] - tl_sup["slope"] + 1e-9)
                ))
                if 0 < n_bars_converge < 20:
                    conf = min(82, 58 + (tl_sup["r2"] + tl_res["r2"]) * 12)
                    _add("Pennant", "neutral",
                         conf,
                         f"Convergence en {n_bars_converge} bougies — breakout imminent",
                         price_now * 1.05, price_now * 0.95,
                         r2=(tl_sup["r2"] + tl_res["r2"]) / 2)

    return sorted(structures, key=lambda x: x["confidence"], reverse=True)


def compute_fibonacci_levels(df: pd.DataFrame,
                              lookback: int = 100) -> dict:
    """
    Calcule les niveaux Fibonacci sur le swing High/Low des `lookback` dernières bougies.
    Retracements : 0.236, 0.382, 0.5, 0.618, 0.786
    Extensions   : 1.272, 1.618, 2.618
    Retourne aussi l'état actuel (prix vs niveaux) et la zone de confluence la plus proche.
    """
    if len(df) < 20:
        return {"available": False}

    window = df.tail(lookback)
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())
    range_sz   = swing_high - swing_low
    price_now  = float(df["close"].iloc[-1])

    if range_sz <= 0:
        return {"available": False}

    retr_levels = {
        "0.236": round(swing_high - 0.236 * range_sz, 2),
        "0.382": round(swing_high - 0.382 * range_sz, 2),
        "0.500": round(swing_high - 0.500 * range_sz, 2),
        "0.618": round(swing_high - 0.618 * range_sz, 2),
        "0.786": round(swing_high - 0.786 * range_sz, 2),
    }
    ext_levels = {
        "1.272": round(swing_low + 1.272 * range_sz, 2),
        "1.618": round(swing_low + 1.618 * range_sz, 2),
        "2.618": round(swing_low + 2.618 * range_sz, 2),
    }

    all_levels = {**retr_levels, **ext_levels}
    # Niveau le plus proche du prix actuel
    closest_key = min(all_levels, key=lambda k: abs(all_levels[k] - price_now))
    closest_val = all_levels[closest_key]
    dist_pct    = (price_now - closest_val) / price_now * 100

    # Zone de confluence Fib (prix dans ±0.5% d'un niveau)
    in_fib_zone = any(abs(v - price_now) / price_now < 0.005 for v in all_levels.values())

    return {
        "available":     True,
        "swing_high":    round(swing_high, 2),
        "swing_low":     round(swing_low, 2),
        "retracements":  retr_levels,
        "extensions":    ext_levels,
        "closest_level": closest_key,
        "closest_price": closest_val,
        "distance_pct":  round(dist_pct, 2),
        "in_fib_zone":   in_fib_zone,
        "lookback":      lookback,
    }


def detect_smart_money(df: pd.DataFrame,
                        lookback: int = 100) -> dict:
    """
    Smart Money Concepts (SMC) :
    - Order Blocks (OB) : dernière bougie haussière/baissière avant un mouvement impulsif
    - Fair Value Gaps (FVG) : gap entre la mèche haute d'une bougie et la mèche basse de la suivante+1
    - Market Structure Shifts (MSS) : cassure d'un pivot clé avec changement de structure
    """
    if len(df) < 20:
        return {"order_blocks": [], "fvgs": [], "mss": [], "available": False}

    window = df.tail(lookback).copy()
    o = window["open"].values
    h = window["high"].values
    l = window["low"].values
    c = window["close"].values
    n = len(c)
    price_now = float(c[-1])

    order_blocks = []
    fvgs         = []
    mss_list     = []

    # ── Order Blocks ──
    # Bull OB : dernière bougie baissière avant une impulsion haussière ≥ 2×ATR
    atr_arr = np.array([float(compute_atr(
        window["high"].iloc[max(0, i-14):i+1],
        window["low"].iloc[max(0, i-14):i+1],
        window["close"].iloc[max(0, i-14):i+1]
    ).iloc[-1]) for i in range(n)])

    for i in range(1, n - 2):
        move_fwd = c[i+2] - c[i+1]
        atr_i    = atr_arr[i] if atr_arr[i] > 0 else 1

        # Bull OB
        if c[i] < o[i] and move_fwd > 2 * atr_i:
            ob_top = max(o[i], c[i])
            ob_bot = min(o[i], c[i])
            # N'inclure que les OB encore actifs (prix au-dessus)
            if price_now >= ob_top:
                order_blocks.append({
                    "type":      "bull",
                    "top":       round(float(ob_top), 2),
                    "bottom":    round(float(ob_bot), 2),
                    "bar_index": i,
                    "active":    price_now >= ob_bot,
                })

        # Bear OB
        if c[i] > o[i] and -move_fwd > 2 * atr_i:
            ob_top = max(o[i], c[i])
            ob_bot = min(o[i], c[i])
            if price_now <= ob_bot:
                order_blocks.append({
                    "type":      "bear",
                    "top":       round(float(ob_top), 2),
                    "bottom":    round(float(ob_bot), 2),
                    "bar_index": i,
                    "active":    price_now <= ob_top,
                })

    # Garder les 3 OB les plus récents de chaque type
    bull_obs = sorted([ob for ob in order_blocks if ob["type"] == "bull"],
                      key=lambda x: x["bar_index"], reverse=True)[:3]
    bear_obs = sorted([ob for ob in order_blocks if ob["type"] == "bear"],
                      key=lambda x: x["bar_index"], reverse=True)[:3]
    order_blocks = bull_obs + bear_obs

    # ── Fair Value Gaps ──
    for i in range(1, n - 1):
        # Bull FVG : low[i+1] > high[i-1]
        if l[i+1] > h[i-1]:
            gap_size = (l[i+1] - h[i-1]) / price_now * 100
            if gap_size > 0.3:
                fvgs.append({
                    "type":      "bull",
                    "top":       round(float(l[i+1]), 2),
                    "bottom":    round(float(h[i-1]), 2),
                    "gap_pct":   round(gap_size, 2),
                    "bar_index": i,
                    "filled":    price_now <= l[i+1],
                })
        # Bear FVG : high[i+1] < low[i-1]
        if h[i+1] < l[i-1]:
            gap_size = (l[i-1] - h[i+1]) / price_now * 100
            if gap_size > 0.3:
                fvgs.append({
                    "type":      "bear",
                    "top":       round(float(l[i-1]), 2),
                    "bottom":    round(float(h[i+1]), 2),
                    "gap_pct":   round(gap_size, 2),
                    "bar_index": i,
                    "filled":    price_now >= h[i+1],
                })

    # Garder les 5 FVG les plus récents non remplis
    fvgs = sorted([f for f in fvgs if not f["filled"]],
                  key=lambda x: x["bar_index"], reverse=True)[:5]

    # ── Market Structure Shifts ──
    # Détecter les pivots et leur cassure
    ph_prices, pl_prices = [], []
    ph_bars,   pl_bars   = [], []
    win = 5
    for i in range(win, n - win):
        if h[i] == h[i-win:i+win+1].max():
            ph_prices.append(h[i]); ph_bars.append(i)
        if l[i] == l[i-win:i+win+1].min():
            pl_prices.append(l[i]); pl_bars.append(i)

    # Bullish MSS : prix casse au-dessus d'un pivot haut précédent (CHoCH)
    if len(ph_prices) >= 2:
        last_ph = ph_prices[-1]
        if c[-1] > last_ph and c[-2] <= last_ph:
            mss_list.append({
                "type":    "bull",
                "level":   round(float(last_ph), 2),
                "bar":     ph_bars[-1],
                "signal":  "CHoCH haussier",
            })

    # Bearish MSS : prix casse en dessous d'un pivot bas précédent
    if len(pl_prices) >= 2:
        last_pl = pl_prices[-1]
        if c[-1] < last_pl and c[-2] >= last_pl:
            mss_list.append({
                "type":    "bear",
                "level":   round(float(last_pl), 2),
                "bar":     pl_bars[-1],
                "signal":  "CHoCH baissier",
            })

    return {
        "available":    True,
        "order_blocks": order_blocks,
        "fvgs":         fvgs,
        "mss":          mss_list,
        "n_bull_obs":   len(bull_obs),
        "n_bear_obs":   len(bear_obs),
        "n_fvgs":       len(fvgs),
    }


def compute_key_levels(df: pd.DataFrame,
                        lookback: int = 200) -> dict:
    """
    Niveaux clés :
    - Pivot Points (classiques + Camarilla)
    - Volume Profile (POC, VAH, VAL)
    - Cluster Analysis (zones de congestion)
    """
    if len(df) < 10:
        return {"available": False}

    window = df.tail(lookback)
    close  = window["close"].values
    high   = window["high"].values
    low    = window["low"].values
    volume = window["volume"].values
    price_now = float(close[-1])

    # ── Pivot Points classiques (sur la dernière bougie complète) ──
    h_prev = float(high[-2])
    l_prev = float(low[-2])
    c_prev = float(close[-2])

    pp  = (h_prev + l_prev + c_prev) / 3
    r1  = 2 * pp - l_prev
    s1  = 2 * pp - h_prev
    r2  = pp + (h_prev - l_prev)
    s2  = pp - (h_prev - l_prev)
    r3  = h_prev + 2 * (pp - l_prev)
    s3  = l_prev - 2 * (h_prev - pp)

    # Camarilla
    cam_r4 = c_prev + (h_prev - l_prev) * 1.1 / 2
    cam_s4 = c_prev - (h_prev - l_prev) * 1.1 / 2
    cam_r3 = c_prev + (h_prev - l_prev) * 1.1 / 4
    cam_s3 = c_prev - (h_prev - l_prev) * 1.1 / 4

    # ── Volume Profile ──
    n_bins = 50
    price_min, price_max = float(low.min()), float(high.max())
    if price_max <= price_min:
        price_max = price_min * 1.01

    bins = np.linspace(price_min, price_max, n_bins + 1)
    vol_profile = np.zeros(n_bins)

    for i in range(len(close)):
        # Distribuer le volume dans les bins couverts par [low, high]
        lo_i, hi_i = float(low[i]), float(high[i])
        bin_lo = int(np.searchsorted(bins, lo_i, side="right") - 1)
        bin_hi = int(np.searchsorted(bins, hi_i, side="left"))
        bin_lo = max(0, min(bin_lo, n_bins - 1))
        bin_hi = max(0, min(bin_hi, n_bins - 1))
        n_covered = bin_hi - bin_lo + 1
        if n_covered > 0 and volume[i] > 0:
            vol_profile[bin_lo:bin_hi+1] += volume[i] / n_covered

    poc_bin    = int(np.argmax(vol_profile))
    poc_price  = round(float((bins[poc_bin] + bins[poc_bin+1]) / 2), 2)

    # Value Area = 70% du volume total
    total_vol  = vol_profile.sum()
    target_vol = total_vol * 0.70
    # Élargir depuis le POC
    va_lo, va_hi = poc_bin, poc_bin
    va_vol = vol_profile[poc_bin]
    while va_vol < target_vol and (va_lo > 0 or va_hi < n_bins - 1):
        add_lo = vol_profile[va_lo - 1] if va_lo > 0 else 0
        add_hi = vol_profile[va_hi + 1] if va_hi < n_bins - 1 else 0
        if add_lo >= add_hi and va_lo > 0:
            va_lo -= 1
            va_vol += add_lo
        elif va_hi < n_bins - 1:
            va_hi += 1
            va_vol += add_hi
        else:
            break

    vah = round(float((bins[va_hi] + bins[va_hi+1]) / 2), 2)
    val = round(float((bins[va_lo] + bins[va_lo+1]) / 2), 2)

    # ── Cluster Analysis (zones de congestion via density) ──
    clusters = []
    touch_threshold = price_now * 0.003   # ±0.3%
    # Compter les fois où le prix a visité chaque niveau
    all_prices = np.concatenate([close, high, low])
    price_counts = {}
    for p in all_prices:
        bucket = round(p / touch_threshold) * touch_threshold
        price_counts[bucket] = price_counts.get(bucket, 0) + 1

    # Top 5 clusters les plus visités
    sorted_clusters = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    for level, touches in sorted_clusters:
        if touches >= 3:
            clusters.append({
                "price":   round(float(level), 2),
                "touches": int(touches),
                "type":    "resistance" if level > price_now else "support",
            })

    # Niveau clé le plus proche
    key_prices = [pp, r1, s1, r2, s2, poc_price]
    closest_kl = min(key_prices, key=lambda x: abs(x - price_now))
    dist_kl    = (price_now - closest_kl) / price_now * 100

    return {
        "available": True,
        "pivot_points": {
            "pp": round(pp, 2), "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
            "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
            "cam_r4": round(cam_r4, 2), "cam_s4": round(cam_s4, 2),
            "cam_r3": round(cam_r3, 2), "cam_s3": round(cam_s3, 2),
        },
        "volume_profile": {
            "poc":  poc_price,
            "vah":  vah,
            "val":  val,
            "histogram": [round(float(v), 0) for v in vol_profile.tolist()],
            "price_bins": [round(float((bins[i]+bins[i+1])/2), 2) for i in range(n_bins)],
        },
        "clusters":        clusters,
        "closest_key_level": round(closest_kl, 2),
        "distance_to_kl_pct": round(dist_kl, 2),
    }


def compute_confluence_score(price: float,
                              fibonacci: dict,
                              key_levels: dict,
                              smc: dict,
                              patterns: list[dict],
                              volume_ratio: float = 1.0) -> dict:
    """
    Score de confluence multi-facteurs pour valider un niveau/signal.
    Retourne un score 0-100 et les contributeurs.
    Pondérations :
      Fib zone      +15
      POC           +12
      Pivot Point   +10
      Order Block   +10
      Candle pattern+8
      Volume surge  +8
      Multi-TF      +15 (si multi-TF activé, sinon ignoré)
    """
    score        = 0
    contributors = []

    # Fib zone (±0.5%)
    if fibonacci.get("available") and fibonacci.get("in_fib_zone"):
        score += 15
        contributors.append(f"Fib {fibonacci['closest_level']} +15")

    # POC (±0.5%)
    if key_levels.get("available"):
        vp = key_levels.get("volume_profile", {})
        poc = vp.get("poc", 0)
        if poc and abs(poc - price) / price < 0.005:
            score += 12
            contributors.append(f"POC {poc:.0f}$ +12")

        # Pivot Point (PP, R1, S1)
        pp_levels = key_levels.get("pivot_points", {})
        for k in ["pp", "r1", "s1", "r2", "s2"]:
            lv = pp_levels.get(k, 0)
            if lv and abs(lv - price) / price < 0.005:
                score += 10
                contributors.append(f"Pivot {k.upper()} {lv:.0f}$ +10")
                break

    # Order Block actif
    if smc.get("available"):
        for ob in smc.get("order_blocks", []):
            if ob.get("bottom", 0) <= price <= ob.get("top", 0):
                score += 10
                contributors.append(f"OB {ob['type']} actif +10")
                break

    # Pattern chandelier récent haute confiance
    high_conf_candles = [p for p in patterns
                         if p.get("type") == "candlestick" and p.get("confidence", 0) >= 70]
    if high_conf_candles:
        score += 8
        contributors.append(f"Pattern {high_conf_candles[0]['name']} +8")

    # Volume surge (ratio > 1.5)
    if volume_ratio > 1.5:
        score += min(8, int((volume_ratio - 1) * 5))
        contributors.append(f"Volume ×{volume_ratio:.1f} +{min(8, int((volume_ratio-1)*5))}")

    return {
        "score":        round(score, 1),
        "contributors": contributors,
        "strong":       score >= 30,
        "very_strong":  score >= 45,
    }


def compute_market_structure(df: pd.DataFrame,
                              df_daily: Optional[pd.DataFrame] = None) -> dict:
    """
    Entrée principale P5 — détection exhaustive de structures.
    Retourne un dict complet avec :
      - candlestick_patterns  : liste triée par confiance
      - geometric_structures  : liste triée par confiance
      - fibonacci             : niveaux Fibonacci
      - smc                   : Order Blocks, FVG, MSS
      - key_levels            : Pivot Points, Volume Profile, Clusters
      - confluence            : score de confluence global
      - dominant_structure    : structure la plus significative
    """
    if len(df) < 20:
        return {"available": False}

    price_now = float(df["close"].iloc[-1])
    vol_21d_avg = float(df["volume"].tail(21).mean()) if len(df) >= 21 else float(df["volume"].mean())
    vol_now     = float(df["volume"].iloc[-1])
    volume_ratio = vol_now / vol_21d_avg if vol_21d_avg > 0 else 1.0

    candles    = detect_candlestick_patterns(df)
    geometrics = detect_geometric_structures(df)
    fibonacci  = compute_fibonacci_levels(df)
    smc        = detect_smart_money(df)
    kl         = compute_key_levels(df)
    confluence = compute_confluence_score(price_now, fibonacci, kl, smc, candles, volume_ratio)

    # Dominant structure : pattern géométrique le plus confiant
    all_structs = geometrics + candles
    dominant = None
    if all_structs:
        best = max(all_structs, key=lambda x: x["confidence"])
        if best["confidence"] >= _MIN_CONFIDENCE:
            dominant = best

    return {
        "available":            True,
        "candlestick_patterns": candles[:5],       # top 5 seulement
        "geometric_structures": geometrics[:5],    # top 5 seulement
        "fibonacci":            fibonacci,
        "smc":                  smc,
        "key_levels":           kl,
        "confluence":           confluence,
        "dominant_structure":   dominant,
        "volume_ratio":         round(volume_ratio, 2),
    }
