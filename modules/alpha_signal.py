"""
alpha_signal.py
Fusionne les 4 couches pour produire le signal Alpha global.

Score final = weighted_avg(macro, technique, forecast, collateral)
Poids calibrés sur l'horizon 4H/Daily :
  - Macro & liquidité  : 30%  (dominant sur 1-6 mois)
  - Technique          : 35%  (dominant sur 1-14 jours)
  - Forecast IA        : 20%  (Chronos + MOIRAI)
  - Collatéraux/risque : 15%  (stress systémique)
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from config import FRED_KEY as _FRED_KEY_CFG

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timezone
from typing import Optional, List, Literal

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True

    class TradeSetup(BaseModel):
        timeframe_analyse:  str
        horizon_de_trade:   str
        strategie: Literal["INTRADAY_MOMENTUM","SWING_TECHNIQUE","STRUCTUREL_MACRO","DCA_ACCUMULATION","GEM_HUNTER","ATTENTE"]
        signal: Literal["LONG FORT","LONG","NEUTRE","SHORT","SHORT FORT"]
        alpha_score:        float
        conviction:         float
        dominant_factor:    str
        invalidation_sl:    float
        sl_distance_pct:    float
        sl_atr_multiple:    float
        objectif_tp1:       float
        objectif_tp2:       float
        objectif_tp3:       Optional[float] = None
        risk_reward:        float
        regime:             str
        patterns_actifs:    List[str]
        score_macro:        float
        score_technique:    float
        score_forecast:     float
        score_collateral:   float
        resume:             str = ""
        reasoning:          List[str]
        warnings:           List[str] = []
        position_advice:    str = ""
        timestamp:          str
        modele_dominant:    str
        regime_detected:    str = ""
        regime_weights:     dict = {}
        regime_reason:      str = ""

except ImportError:
    _HAS_PYDANTIC = False
    TradeSetup = None


# ── Config par Timeframe ──
TF_CONFIG = {
    "5m":  {"strategy":"INTRADAY_MOMENTUM", "horizon":"intraday",
             "weights":{"macro":0.00,"tech":0.55,"forecast":0.35,"collateral":0.10},
             "sl_atr_mult":1.5, "tp1_mult":1.5, "tp2_mult":2.5, "tp3_mult":None},
    "15m": {"strategy":"INTRADAY_MOMENTUM", "horizon":"intraday",
             "weights":{"macro":0.05,"tech":0.55,"forecast":0.30,"collateral":0.10},
             "sl_atr_mult":1.5, "tp1_mult":1.5, "tp2_mult":2.5, "tp3_mult":None},
    "1h":  {"strategy":"INTRADAY_MOMENTUM", "horizon":"intraday",
             "weights":{"macro":0.15,"tech":0.50,"forecast":0.25,"collateral":0.10},
             "sl_atr_mult":2.0, "tp1_mult":2.0, "tp2_mult":3.5, "tp3_mult":5.0},
    "4h":  {"strategy":"SWING_TECHNIQUE",   "horizon":"swing_2_5j",
             "weights":{"macro":0.25,"tech":0.40,"forecast":0.25,"collateral":0.10},
             "sl_atr_mult":2.0, "tp1_mult":2.0, "tp2_mult":4.0, "tp3_mult":6.0},
    "1d":  {"strategy":"STRUCTUREL_MACRO",  "horizon":"positionnel_2_8sem",
             "weights":{"macro":0.40,"tech":0.30,"forecast":0.20,"collateral":0.10},
             "sl_atr_mult":2.5, "tp1_mult":2.5, "tp2_mult":5.0, "tp3_mult":8.0},
    "1w":  {"strategy":"STRUCTUREL_MACRO",  "horizon":"positionnel_2_8sem",
             "weights":{"macro":0.50,"tech":0.25,"forecast":0.15,"collateral":0.10},
             "sl_atr_mult":3.0, "tp1_mult":3.0, "tp2_mult":6.0, "tp3_mult":10.0},
}


# ─── DONNÉES COLLATÉRAUX & STRESS SYSTÉMIQUE ─────────────────────────────────

_collateral_cache = {}
_collateral_ts    = {}
CACHE_TTL = 1800


def get_collateral_stress() -> dict:
    """
    Proxy du marché des collatéraux et dette via :
    - Spread LIBOR-OIS (stress bancaire) → FRED : TEDRATE
    - Corporate credit spreads → HY via FRED
    - VIX → proxy peur actions (corrélé BTC en stress)
    - Gold/BTC ratio → flight to safety
    """
    now = time.time()
    if "stress" in _collateral_cache and (now - _collateral_ts.get("stress", 0)) < CACHE_TTL:
        return _collateral_cache["stress"]

    FRED_KEY  = _FRED_KEY_CFG
    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(series, limit=30):
        try:
            r = requests.get(FRED_BASE, params={
                "series_id": series, "api_key": FRED_KEY,
                "file_type": "json", "limit": limit, "sort_order": "desc"
            }, timeout=8)
            obs = r.json().get("observations", [])
            vals = [float(o["value"]) for o in obs if o["value"] != "."]
            return vals[0] if vals else None, vals
        except:
            return None, []

    ted_now, ted_hist   = fetch("TEDRATE", 30)    # TED spread (stress bancaire)
    sofr_now, _         = fetch("SOFR",    10)    # SOFR (taux repo overnight)
    corp_now, corp_hist = fetch("BAMLH0A0HYM2", 30)  # HY spread

    # VIX via Yahoo Finance
    vix_now = None
    try:
        import yfinance as yf
        vix_data = yf.Ticker("^VIX").history(period="5d")
        if not vix_data.empty:
            vix_now = round(float(vix_data["Close"].iloc[-1]), 1)
    except:
        pass

    # Score stress (0 = aucun stress, 100 = stress extrême)
    stress_score = 0.0
    stress_sigs  = []

    if ted_now is not None:
        # TED spread > 50bp = stress, > 100bp = crise
        if ted_now > 1.0:
            stress_score += 30
            stress_sigs.append(f"TED spread élevé ({ted_now:.2f}%) — stress bancaire")
        elif ted_now > 0.5:
            stress_score += 15
            stress_sigs.append(f"TED spread modéré ({ted_now:.2f}%)")
        else:
            stress_sigs.append(f"TED spread normal ({ted_now:.2f}%)")

    if corp_now is not None:
        if corp_now > 5.0:
            stress_score += 35
            stress_sigs.append(f"HY spread très élevé ({corp_now:.2f}%) — risk-off fort")
        elif corp_now > 3.5:
            stress_score += 15
            stress_sigs.append(f"HY spread modéré ({corp_now:.2f}%)")
        else:
            stress_sigs.append(f"HY spread faible ({corp_now:.2f}%) — risk-on")

    if vix_now is not None:
        if vix_now > 30:
            stress_score += 25
            stress_sigs.append(f"VIX élevé ({vix_now}) — peur actions")
        elif vix_now > 20:
            stress_score += 10
            stress_sigs.append(f"VIX modéré ({vix_now})")
        else:
            stress_sigs.append(f"VIX faible ({vix_now}) — complacency")

    stress_score = min(100, stress_score)

    # Le score stress est inversé pour le score alpha
    # (stress élevé = baissier → score alpha faible)
    collateral_alpha_score = max(0, 100 - stress_score)

    result = {
        "stress_score":         round(stress_score, 1),
        "collateral_score":     round(collateral_alpha_score, 1),
        "ted_spread":           ted_now,
        "hy_spread":            corp_now,
        "vix":                  vix_now,
        "sofr":                 sofr_now,
        "signals":              stress_sigs,
        "market_condition":     "STRESS CRITIQUE" if stress_score > 60 else
                                "STRESS MODÉRÉ"   if stress_score > 30 else
                                "NORMAL"          if stress_score > 10 else "RISK-ON",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    _collateral_cache["stress"] = result
    _collateral_ts["stress"]    = now
    return result


# ─── FILTRES DE CONFLUENCE ──────────────────────────────────────────────────────

def _apply_confluence_filters(forecasts: dict, tech_detail: dict,
                               current_price: float) -> tuple[float, list]:
    """
    Applique 4 garde-fous qui filtrent / pénalisent les signaux IA.
    Retourne (delta_conviction, confluence_signals).
    """
    conviction_delta = 0.0
    conf_signals     = []
    indicators       = tech_detail.get("indicators", {})

    # ── 1. Filtre ATR : volatilité excessive ──
    atr_now  = indicators.get("atr", 0)
    # On compare l'ATR actuel à une valeur raisonnée (pas d'historique complet ici)
    # Heuristique : ATR > 3% du prix = volatilité excessive
    atr_pct = atr_now / current_price * 100 if current_price > 0 else 0
    if atr_pct > 3.0:
        conviction_delta -= 20
        conf_signals.append(("CONFLUENCE", "-",
            f"ATR élevé ({atr_pct:.1f}% du prix) — volatilité excessive, conviction -20pts"))

    # ── 2. Filtre RSI confluence ──
    rsi = indicators.get("rsi", 50)
    # Sera utilisé par l'appelant pour invalider certains signaux directionnels

    # ── 3. Filtre divergence inter-modèles ──
    divergence = forecasts.get("meta", {}).get("divergence", 0)
    if divergence > 3.0:
        conviction_delta -= 15
        conf_signals.append(("CONFLUENCE", "-",
            f"Divergence inter-modèles {divergence:.1f}% > 3% — conviction -15pts"))
    elif divergence > 1.5:
        conviction_delta -= 5
        conf_signals.append(("CONFLUENCE", "=",
            f"Divergence modérée entre modèles ({divergence:.1f}%)"))

    return conviction_delta, conf_signals, rsi


# ─── INTÉGRATION CHRONOS / MOIRAI / LAG-LLAMA ──────────────────────────────────

def get_forecast_signal(forecasts: dict, current_price: float,
                        tech_detail: dict = None) -> dict:
    """
    Transforme les sorties des 3 modèles en signal directionnel.
    Pondération dynamique basée sur confidence ET calibration.

    Weights :
      confidence >= 60 → weight_factor = 1.0
      confidence 40-60 → weight_factor = 0.75
      confidence 20-40 → weight_factor = 0.5 (LOW)
      confidence  < 20 → weight_factor = 0.0 (exclu)

    Coverage (walk-forward Lag-Llama) :
      coverage < 60% → poids Lag-Llama réduit à 0.10
      coverage 60-75% → poids 0.20
      coverage > 75% → poids standard

    Si tous les modèles ont confidence < 20 → NEUTRE forcé.
    """
    if not forecasts or forecasts.get("meta", {}).get("error"):
        return {"score": 50, "label": "INDISPONIBLE", "signals": []}

    models    = ["chronos", "moirai", "lagllama"]
    available = [m for m in models if m in forecasts and "p50" in forecasts[m]]
    if not available:
        return {"score": 50, "label": "INDISPONIBLE", "signals": []}

    signals = []

    # ── Calcul des poids dynamiques ──
    weights = {}
    for m in available:
        conf_info = forecasts[m].get("confidence", {})
        wf = conf_info.get("weight_factor", 1.0)

        # Ajustement spécifique Lag-Llama selon coverage walk-forward
        if m == "lagllama":
            calib = forecasts[m].get("calibration", {})
            cov   = calib.get("coverage", None) if calib.get("available") else None
            if cov is not None:
                if cov < 60:
                    wf = min(wf, 0.10)
                elif cov < 75:
                    wf = min(wf, 0.20)
                else:
                    wf = min(wf, 0.35)

        weights[m] = max(0.0, wf)

    total_w = sum(weights.values())
    if total_w == 0:
        # Tous les modèles non fiables → NEUTRE forcé
        signals.append(("FORECAST", "=", "Tous modèles non fiables — forecast NEUTRE forcé"))
        return {
            "score":           50.0,
            "p50_consensus":   round(current_price, 0),
            "delta_pct":       0.0,
            "uncertainty_pct": 0.0,
            "divergence":      0.0,
            "prob_bull_avg":   50.0,
            "implied_vol_avg": 0.0,
            "signals":         signals,
            "models_used":     available,
            "weights":         weights,
            "forced_neutral":  True,
        }

    # Normalisation des poids
    norm_w = {m: weights[m] / total_w for m in available}

    # ── Consensus pondéré ──
    p50_consensus = sum(norm_w[m] * forecasts[m]["p50"][-1] for m in available)
    p10_consensus = sum(norm_w[m] * forecasts[m]["p10"][-1] for m in available)
    p90_consensus = sum(norm_w[m] * forecasts[m]["p90"][-1] for m in available)
    divergence    = forecasts.get("meta", {}).get("divergence", 1.0)

    delta_pct   = (p50_consensus - current_price) / current_price * 100
    uncertainty = (p90_consensus - p10_consensus) / current_price * 100

    # ── Filtres de confluence ──
    confluence_delta = 0.0
    rsi = 50.0
    if tech_detail:
        confluence_delta, conf_sigs, rsi = _apply_confluence_filters(
            forecasts, tech_detail, current_price)
        signals.extend(conf_sigs)

    # ── Score directionnel de base ──
    forecast_score = 50.0
    if delta_pct > 2:
        # Filtre RSI : LONG validé si RSI > 35
        if rsi > 35:
            forecast_score += min(20, delta_pct * 3)
            signals.append(("FORECAST", "+", f"Consensus IA haussier: +{delta_pct:.1f}%"))
        else:
            signals.append(("FORECAST", "=",
                f"Consensus IA haussier ({delta_pct:.1f}%) invalidé — RSI trop bas ({rsi:.0f})"))
    elif delta_pct < -2:
        # Filtre RSI : SHORT validé si RSI < 65
        if rsi < 65:
            forecast_score -= min(20, abs(delta_pct) * 3)
            signals.append(("FORECAST", "-", f"Consensus IA baissier: {delta_pct:.1f}%"))
        else:
            signals.append(("FORECAST", "=",
                f"Consensus IA baissier ({delta_pct:.1f}%) invalidé — RSI trop haut ({rsi:.0f})"))
    else:
        signals.append(("FORECAST", "=", f"Consensus IA neutre: {delta_pct:.1f}%"))

    # ── Divergence inter-modèles ──
    if divergence > 1.5:
        forecast_score -= 10
        signals.append(("DIVERGENCE", "-",
            f"Forte divergence modèles ({divergence:.2f}%) — signal peu fiable"))
    elif divergence < 0.5:
        forecast_score += 5
        signals.append(("DIVERGENCE", "+",
            f"Consensus fort entre modèles ({divergence:.2f}%)"))

    # ── Probabilité haussière pondérée ──
    prob_bulls = [forecasts[m]["prob_bull"] for m in available if "prob_bull" in forecasts[m]]
    avg_bull   = float(np.average(
        [forecasts[m]["prob_bull"] for m in available if "prob_bull" in forecasts[m]],
        weights=[norm_w[m] for m in available if "prob_bull" in forecasts[m]]
    )) if prob_bulls else 50.0
    if avg_bull > 65:
        forecast_score += 5
        signals.append(("PROB_BULL", "+", f"P(hausse) pondérée: {avg_bull:.0f}%"))
    elif avg_bull < 35:
        forecast_score -= 5
        signals.append(("PROB_BULL", "-", f"P(hausse) pondérée: {avg_bull:.0f}%"))

    # ── Volatilité implicite ──
    impl_vols = [forecasts[m].get("implied_vol", 0) for m in available if forecasts[m].get("implied_vol")]
    avg_vol   = float(np.mean(impl_vols)) if impl_vols else 0
    vol_signal = "FORTE" if avg_vol > 100 else "MODÉRÉE" if avg_vol > 60 else "FAIBLE"
    signals.append(("VOL_IMPL", "=", f"Volatilité implicite IA: {avg_vol:.0f}% — {vol_signal}"))

    forecast_score = max(0, min(100, forecast_score))

    # Log structuré
    log_parts = []
    for m in available:
        conf  = forecasts[m].get("confidence", {}).get("confidence", "?")
        calib = forecasts[m].get("calibration", {})
        cov   = calib.get("coverage", "?") if calib.get("available") else "?"
        w     = round(norm_w.get(m, 0) * 100, 0)
        flag  = " (réduit)" if weights.get(m, 1) < 0.5 else ""
        log_parts.append(f"{m}: conf={conf}% cov={cov}% poids={w:.0f}%{flag}")
    dir_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
    sig_str = "HAUSSIER" if forecast_score > 60 else "BAISSIER" if forecast_score < 40 else "NEUTRE"
    import logging as _logging
    _logging.getLogger("alpha_signal").debug(
        "[FORECAST] %s → %s → %s", " | ".join(log_parts), dir_str, sig_str
    )

    return {
        "score":           round(forecast_score, 1),
        "p50_consensus":   round(p50_consensus, 0),
        "delta_pct":       round(delta_pct, 2),
        "uncertainty_pct": round(uncertainty, 1),
        "divergence":      round(divergence, 2),
        "prob_bull_avg":   round(avg_bull, 1),
        "implied_vol_avg": round(avg_vol, 1),
        "signals":         signals,
        "models_used":     available,
        "weights":         {m: round(norm_w.get(m, 0), 3) for m in available},
        "confluence_delta": round(confluence_delta, 1),
    }


# ─── RÉGIMES DE MARCHÉ ───────────────────────────────────────────────────────

from enum import Enum

class MarketRegime(Enum):
    BULL_TREND      = "bull_trend"
    BEAR_TREND      = "bear_trend"
    ACCUMULATION    = "accumulation"
    DISTRIBUTION    = "distribution"
    SYSTEMIC_STRESS = "systemic_stress"   # VIX > 30
    COMPRESSION     = "compression"        # BBW < 10e percentile
    HIGH_VOLATILITY = "high_volatility"    # ATR > 3%

REGIME_WEIGHTS = {
    MarketRegime.BULL_TREND: {
        "macro": 0.20, "tech": 0.45, "forecast": 0.25, "collateral": 0.10,
        "description": "Tendance haussière — technique domine",
    },
    MarketRegime.BEAR_TREND: {
        "macro": 0.25, "tech": 0.40, "forecast": 0.25, "collateral": 0.10,
        "description": "Tendance baissière — technique + macro",
    },
    MarketRegime.ACCUMULATION: {
        "macro": 0.35, "tech": 0.30, "forecast": 0.25, "collateral": 0.10,
        "description": "Accumulation — macro et forecast priment",
    },
    MarketRegime.SYSTEMIC_STRESS: {
        "macro": 0.50, "tech": 0.15, "forecast": 0.10, "collateral": 0.25,
        "description": "Stress systémique — macro et collatéraux dominent",
    },
    MarketRegime.COMPRESSION: {
        "macro": 0.25, "tech": 0.30, "forecast": 0.35, "collateral": 0.10,
        "description": "Compression — forecast IA priment (breakout imminent)",
    },
    MarketRegime.HIGH_VOLATILITY: {
        "macro": 0.30, "tech": 0.25, "forecast": 0.20, "collateral": 0.25,
        "description": "Haute volatilité — réduction exposition, macro + collat",
    },
    MarketRegime.DISTRIBUTION: {
        "macro": 0.30, "tech": 0.35, "forecast": 0.20, "collateral": 0.15,
        "description": "Distribution — équilibre technique et macro",
    },
}

def detect_market_regime(tech_detail: dict, macro_detail: dict,
                          collat_detail: dict) -> tuple:
    """
    Détecte le régime de marché et retourne (MarketRegime, weights_dict).
    Priorité : stress systémique > haute vol > compression > trend > accum/distrib.
    """
    import logging as _log_mod
    _alog = _log_mod.getLogger("alpha_signal")

    indicators = tech_detail.get("indicators", {})
    vix        = float(collat_detail.get("vix", 20) or 20)
    adx        = float(indicators.get("adx", 20) or 20)
    bb_pct     = float(indicators.get("bb_width_percentile", 50) or 50)
    atr_pct    = float(indicators.get("atr_pct", 2.0) or 2.0)
    regime_str = tech_detail.get("regime", "TRANSITION")

    if vix > 30:
        detected = MarketRegime.SYSTEMIC_STRESS
        _alog.info("[Regime] SYSTEMIC_STRESS (VIX=%.1f)", vix)
    elif atr_pct > 3.0:
        detected = MarketRegime.HIGH_VOLATILITY
        _alog.info("[Regime] HIGH_VOLATILITY (ATR=%.1f%%)", atr_pct)
    elif bb_pct < 10:
        detected = MarketRegime.COMPRESSION
        _alog.info("[Regime] COMPRESSION (BBW_pct=%.1f)", bb_pct)
    elif adx > 25 and "BULL" in regime_str:
        detected = MarketRegime.BULL_TREND
    elif adx > 25 and "BEAR" in regime_str:
        detected = MarketRegime.BEAR_TREND
    elif regime_str == "ACCUMULATION":
        detected = MarketRegime.ACCUMULATION
    else:
        detected = MarketRegime.DISTRIBUTION

    return detected, REGIME_WEIGHTS[detected]


# ─── FUSION FINALE ────────────────────────────────────────────────────────────

def compute_alpha_signal(
    macro_score:   float,
    tech_score:    float,
    forecast_data: dict,
    current_price: float,
    macro_detail:  dict,
    tech_detail:   dict,
    interval:      str = "1h",
) -> dict:
    """
    Signal Alpha final = fusion pondérée des 4 couches, adapté au timeframe.
    """
    cfg = TF_CONFIG.get(interval, TF_CONFIG["1h"])

    collateral    = get_collateral_stress()
    forecast_sig  = get_forecast_signal(forecast_data, current_price, tech_detail=tech_detail)

    # ── Détection du régime — poids dynamiques ──
    regime_enum, regime_w = detect_market_regime(
        tech_detail, macro_detail, collateral
    )

    # Les poids TF_CONFIG restent en fallback ; le régime les remplace
    w = regime_w   # overwrite avec poids régime

    # ── Filtre volume : si volume 24H < 50% moyenne 20J → réduire poids forecast ──
    volume_signals  = []
    w_forecast_adj  = w["forecast"]
    try:
        import requests as _req
        r_vol = _req.get("https://api.binance.com/api/v3/klines",
                         params={"symbol":"BTCUSDT","interval":"1d","limit":21}, timeout=5)
        if r_vol.ok:
            klines = r_vol.json()
            vols   = [float(k[5]) for k in klines]
            vol_24h = vols[-1]
            vol_20d_avg = float(np.mean(vols[:-1]))
            if vol_20d_avg > 0 and vol_24h < vol_20d_avg * 0.5:
                w_forecast_adj *= 0.70
                volume_signals.append(("CONFLUENCE", "-",
                    f"Volume 24H ({vol_24h/1e3:.0f}k) < 50% moy 20J — poids forecast -30%"))
    except Exception:
        pass

    # Recalibrer les poids (somme = 1)
    w_macro    = w["macro"]
    w_tech     = w["tech"]
    w_forecast = w_forecast_adj
    w_collat   = max(0.05, 1.0 - w_macro - w_tech - w_forecast)

    alpha_score = (
        macro_score                    * w_macro    +
        tech_score                     * w_tech     +
        forecast_sig["score"]          * w_forecast +
        collateral["collateral_score"] * w_collat
    )
    alpha_score = round(max(0, min(100, alpha_score)), 1)

    # ── Signal directionnel ──
    if alpha_score >= 72:   signal = "LONG FORT";  signal_color = "green"
    elif alpha_score >= 60: signal = "LONG";        signal_color = "lightgreen"
    elif alpha_score >= 45: signal = "NEUTRE";      signal_color = "gray"
    elif alpha_score >= 35: signal = "SHORT";       signal_color = "orange"
    else:                   signal = "SHORT FORT";  signal_color = "red"

    # ── Conviction ──
    scores = [macro_score, tech_score, forecast_sig["score"], collateral["collateral_score"]]
    conviction = max(0, min(100, 100 - float(np.std(scores)) * 1.5
                             + forecast_sig.get("confluence_delta", 0.0)))

    # ── SL / TP basés sur ATR du TF ──
    indicators = tech_detail.get("indicators", {})
    atr_now    = float(indicators.get("atr", 0) or 0) or current_price * 0.015
    atr_pct    = float(indicators.get("atr_pct", 0) or 0)
    sl_mult    = cfg["sl_atr_mult"]
    tp1_mult   = cfg["tp1_mult"]
    tp2_mult   = cfg["tp2_mult"]
    tp3_mult   = cfg["tp3_mult"]

    if alpha_score >= 60:        # LONG
        sl_price = current_price - atr_now * sl_mult
        tp1      = current_price + atr_now * tp1_mult
        tp2      = current_price + atr_now * tp2_mult
        tp3      = (current_price + atr_now * tp3_mult) if tp3_mult else None
    elif alpha_score <= 40:      # SHORT
        sl_price = current_price + atr_now * sl_mult
        tp1      = current_price - atr_now * tp1_mult
        tp2      = current_price - atr_now * tp2_mult
        tp3      = (current_price - atr_now * tp3_mult) if tp3_mult else None
    else:                        # NEUTRE
        sl_price = tp1 = tp2 = tp3 = current_price

    sl_dist_pct = abs(current_price - sl_price) / current_price * 100 if current_price else 0
    rr = abs(tp2 - current_price) / max(abs(sl_price - current_price), 0.01)

    # ── Warnings ──
    warnings = []
    if atr_pct > 3.0:
        warnings.append(f"Volatilité élevée (ATR {atr_pct:.1f}%) — réduire la taille de position")
    if interval in ["5m", "15m"] and macro_score < 40:
        warnings.append("Macro défavorable — éviter les longs en intraday")
    if forecast_sig.get("divergence", 0) > 2.0:
        warnings.append(f"Forte divergence inter-modèles ({forecast_sig['divergence']:.1f}%) — signal peu fiable")
    if conviction < 40:
        warnings.append("Conviction faible — attendre confirmation")

    # ── Reasoning ──
    reasoning = []
    for s in tech_detail.get("signals", [])[:5]:
        reasoning.append(f"[{s[0]}] {s[2]}")
    for s in macro_detail.get("signals", [])[:3]:
        reasoning.append(f"[{s[0]}] {s[2]}")

    # ── Facteur dominant ──
    layer_scores = {
        "Macro/Liquidité": macro_score,
        "Technique":       tech_score,
        "Forecast IA":     forecast_sig["score"],
        "Collatéraux":     collateral["collateral_score"],
    }
    dominant_factor = max(layer_scores, key=lambda k: abs(layer_scores[k] - 50))

    # ── Position advice ──
    if conviction >= 70 and alpha_score >= 65:
        position_advice = "Taille pleine (100% du sizing)"
    elif conviction >= 50 and alpha_score >= 58:
        position_advice = "Taille réduite (50-60% du sizing)"
    elif conviction >= 40:
        position_advice = "Position minimale ou attente"
    else:
        position_advice = "Rester flat — signal non fiable"

    # ── Patterns actifs ──
    patterns_actifs = [
        pat["name"] for pat in tech_detail.get("patterns", [])
        if pat.get("confidence", 0) > 55
    ]

    # ── Agrégation signaux ──
    all_signals = []
    for s in macro_detail.get("signals", []):
        all_signals.append({"layer": "MACRO",    "dir": s[1], "msg": s[2]})
    for s in tech_detail.get("signals", []):
        all_signals.append({"layer": "TECH",     "dir": s[1], "msg": s[2]})
    for s in forecast_sig.get("signals", []):
        all_signals.append({"layer": "FORECAST", "dir": s[1], "msg": s[2]})
    for s in collateral.get("signals", []):
        all_signals.append({"layer": "COLLAT",   "dir": "+" if any(k in s.lower() for k in ["risk-on","faible","normal"]) else "-", "msg": s})
    for s in volume_signals:
        all_signals.append({"layer": s[0], "dir": s[1], "msg": s[2]})

    import logging as _logging
    _logging.getLogger("alpha_signal").info(
        "[ALPHA %s] strategy=%s signal=%s score=%s "
        "w=macro%.0f%%/tech%.0f%%/fc%.0f%% SL=%.0f(-%.1f%%) TP2=%.0f RR=%.1f",
        interval.upper(), cfg["strategy"], signal, alpha_score,
        w_macro * 100, w_tech * 100, w_forecast * 100,
        sl_price, sl_dist_pct, tp2, rr,
    )

    result = {
        # Compatibilité existante
        "alpha_score":      alpha_score,
        "signal":           signal,
        "signal_color":     signal_color,
        "conviction":       round(conviction, 1),
        "position_advice":  position_advice,
        "dominant_factor":  dominant_factor,
        "layer_scores": {
            "macro":      round(macro_score, 1),
            "tech":       round(tech_score, 1),
            "forecast":   round(forecast_sig["score"], 1),
            "collateral": round(collateral["collateral_score"], 1),
        },
        "forecast_detail":  forecast_sig,
        "collateral_detail":collateral,
        "all_signals":      all_signals,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        # Nouveaux champs TF-aware
        "timeframe_analyse":  interval,
        "horizon_de_trade":   cfg["horizon"],
        "strategie":          cfg["strategy"],
        "invalidation_sl":    round(sl_price, 2),
        "sl_distance_pct":    round(sl_dist_pct, 2),
        "sl_atr_multiple":    sl_mult,
        "objectif_tp1":       round(tp1, 2),
        "objectif_tp2":       round(tp2, 2),
        "objectif_tp3":       round(tp3, 2) if tp3 else None,
        "risk_reward":        round(rr, 2),
        "regime":             tech_detail.get("regime", "INCONNU"),
        "patterns_actifs":    patterns_actifs,
        "score_macro":        round(macro_score, 1),
        "score_technique":    round(tech_score, 1),
        "score_forecast":     round(forecast_sig["score"], 1),
        "score_collateral":   round(collateral["collateral_score"], 1),
        "reasoning":          reasoning,
        "warnings":           warnings,
        "tf_weights":         {k: round(v, 2) for k, v in
                               {"macro": w_macro, "tech": w_tech,
                                "forecast": w_forecast, "collateral": w_collat}.items()},
        # ── Régime dynamique ──
        "regime_detected":    regime_enum.value,
        "regime_weights":     {k: round(v, 3) for k, v in regime_w.items()
                               if isinstance(v, float)},
        "regime_reason":      regime_w.get("description", ""),
    }
    # Champ resume (génération inline pour éviter référence circulaire)
    result["resume"] = generate_resume(
        signal=signal,
        alpha_score=alpha_score,
        dominant_factor=dominant_factor,
        tech_regime=tech_detail.get("regime", "RANGE"),
        cycle_phase=macro_detail.get("cycle", {}).get("phase", ""),
        warnings=warnings,
    )
    return result


# ─── FEAR & GREED INDEX (alternative.me — gratuit) ───────────────────────────

_fg_cache  = {}
_fg_ts     = 0

def get_fear_greed() -> dict:
    """Fetche le Fear & Greed Index depuis alternative.me."""
    global _fg_cache, _fg_ts
    now = time.time()
    if _fg_cache and (now - _fg_ts) < 3600:
        return _fg_cache
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=6)
        data = r.json().get("data", [{}])[0]
        val  = int(data.get("value", 50))
        label = data.get("value_classification", "Neutral")
        # Bonus/malus pour le signal alpha
        if val <= 20:
            bonus = +10  # Extreme Fear → souvent bottom
            interp = "Peur extrême — zone d'accumulation historique"
        elif val >= 80:
            bonus = -10  # Extreme Greed → souvent top
            interp = "Cupidité extrême — zone de distribution historique"
        elif val <= 35:
            bonus = +4
            interp = "Peur — marché capitule, opportunité potentielle"
        elif val >= 65:
            bonus = -4
            interp = "Optimisme — prudence recommandée"
        else:
            bonus = 0
            interp = "Sentiment neutre — pas de signal extrême"
        _fg_cache = {
            "value": val, "label": label,
            "alpha_bonus": bonus, "interpretation": interp,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        _fg_ts = now
    except Exception:
        _fg_cache = {"value": 50, "label": "Neutral", "alpha_bonus": 0,
                     "interpretation": "Données indisponibles", "last_updated": None}
        _fg_ts = now
    return _fg_cache


# ─── AXE 5 : DÉTECTION D'ANOMALIES (IsolationForest) ─────────────────────────

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.04) -> dict:
    """
    Détecte les bougies anormales (prix + volume) avec IsolationForest.
    Retourne un dict avec : anomaly_now (bool), score_now (float 0-1),
    recent_count (int), severity ("none"/"low"/"high"), interpretation (str).
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return {"anomaly_now": False, "score_now": 0.0, "recent_count": 0,
                "severity": "unavailable", "interpretation": "scikit-learn non installé"}

    if df is None or len(df) < 50:
        return {"anomaly_now": False, "score_now": 0.0, "recent_count": 0,
                "severity": "none", "interpretation": "Données insuffisantes"}

    try:
        df2 = df.copy()
        # Features : retour log, range relatif, volume relatif (z-score), body/range
        df2["log_ret"]   = np.log(df2["close"] / df2["close"].shift(1)).fillna(0)
        df2["range_rel"] = ((df2["high"] - df2["low"]) / df2["close"]).fillna(0)
        df2["vol_zscore"]= ((df2["volume"] - df2["volume"].rolling(20).mean()) /
                            (df2["volume"].rolling(20).std() + 1e-9)).fillna(0)
        df2["body_rel"]  = (abs(df2["close"] - df2["open"]) / (df2["high"] - df2["low"] + 1e-9)).fillna(0)

        features = df2[["log_ret", "range_rel", "vol_zscore", "body_rel"]].values
        model = IsolationForest(n_estimators=100, contamination=contamination,
                                random_state=42, n_jobs=-1)
        model.fit(features)

        scores_raw = model.decision_function(features)   # négatif = plus anomal
        # Normalise en [0,1] : 1 = anomalie certaine
        s_min, s_max = scores_raw.min(), scores_raw.max()
        norm_scores = 1 - (scores_raw - s_min) / (s_max - s_min + 1e-9)

        preds = model.predict(features)   # -1 = anomalie, 1 = normal

        last_score  = float(norm_scores[-1])
        anomaly_now = (preds[-1] == -1)
        recent_count = int(np.sum(preds[-20:] == -1))   # anomalies sur 20 dernières bougies

        if anomaly_now and last_score > 0.7:
            severity = "high"
            interp = f"Bougie anormale détectée (score={last_score:.2f}) — volume/volatilité extrêmes"
        elif anomaly_now or recent_count >= 3:
            severity = "low"
            interp = f"{recent_count} anomalie(s) sur les 20 dernières bougies — surveillance accrue"
        else:
            severity = "none"
            interp = "Aucune anomalie de marché détectée"

        return {
            "anomaly_now":   anomaly_now,
            "score_now":     round(last_score, 3),
            "recent_count":  recent_count,
            "severity":      severity,
            "interpretation": interp,
        }
    except Exception as e:
        return {"anomaly_now": False, "score_now": 0.0, "recent_count": 0,
                "severity": "error", "interpretation": str(e)[:120]}


# ─── RÉSUMÉ LISIBLE EN 5 SECONDES ─────────────────────────────────────────────

def generate_resume(signal: str, alpha_score: float, dominant_factor: str,
                    tech_regime: str, cycle_phase: str, warnings: list) -> str:
    """
    Génère un résumé en 1-2 phrases du signal actuel.
    Compréhensible en 5 secondes. Expose dans compute_alpha_signal()
    pour que /api/full retourne alpha.resume directement.
    """
    if signal == "LONG FORT":
        base = f"Signal haussier fort ({alpha_score:.0f}/100). "
        base += f"Dominant : {dominant_factor}. "
        if tech_regime in ("BULL TREND", "ACCUMULATION"):
            base += "Confluence technique-macro."
    elif signal == "LONG":
        base = f"Biais haussier ({alpha_score:.0f}/100). "
        base += "Momentum positif mais conviction modérée."
    elif signal == "SHORT FORT":
        base = f"Signal baissier fort ({alpha_score:.0f}/100). "
        base += f"Dominant : {dominant_factor}."
    elif signal == "SHORT":
        base = f"Biais baissier ({alpha_score:.0f}/100). "
        base += "Réduction d'exposition recommandée."
    else:
        base = f"Signal neutre ({alpha_score:.0f}/100). "
        base += "Attendre confirmation directionnelle."

    if warnings:
        base += " ⚠ " + warnings[0]

    return base[:200]


# ─── SIGNAL STRUCTURÉ : tldr / boussole / timing ──────────────────────────────

_URGENCE_MAP = {
    "LONG FORT":  "MAINTENANT",
    "LONG":       "ATTENDRE_PULLBACK",
    "NEUTRE":     "SURVEILLER",
    "SHORT":      "ATTENDRE_PULLBACK",
    "SHORT FORT": "MAINTENANT",
}

_CONVICTION_LABEL = {
    (0,   40): ("TRÈS FAIBLE", "rgba(255,68,68,.15)",   "#ff4444"),
    (40,  55): ("FAIBLE",      "rgba(255,140,66,.15)",  "#ff8c42"),
    (55,  70): ("MODÉRÉE",     "rgba(255,171,0,.15)",   "#ffab00"),
    (70,  85): ("HAUTE",       "rgba(0,230,118,.12)",   "#00e676"),
    (85, 101): ("TRÈS HAUTE",  "rgba(79,195,247,.15)",  "#4fc3f7"),
}

def _conviction_label(pct: float) -> tuple:
    for (lo, hi), info in _CONVICTION_LABEL.items():
        if lo <= pct < hi:
            return info
    return ("MODÉRÉE", "rgba(255,171,0,.15)", "#ffab00")


def _make_resume(signal: str, alpha: dict, tech_detail: dict,
                 macro_detail: dict, forecast_sig: dict,
                 current_price: float, interval: str) -> str:
    """Génère un résumé de 2 lignes actionnable."""
    sl   = alpha.get("invalidation_sl", 0)
    tp1  = alpha.get("objectif_tp1", 0)
    rr   = alpha.get("risk_reward", 0)
    strat = alpha.get("strategie", "SWING_TECHNIQUE")
    regime = tech_detail.get("regime", "RANGE")

    patterns = [p["name"] for p in tech_detail.get("patterns", []) if p.get("confidence",0) > 60]
    pat_str  = patterns[0] if patterns else regime

    delta = forecast_sig.get("delta_pct", 0)
    fc_str = f"modèles IA prévoient {delta:+.1f}%" if abs(delta) > 0.5 else ""

    if signal in ("LONG FORT", "LONG"):
        dir_txt = "haussier"
        action  = "Entrée validée" if signal == "LONG FORT" else "Entrée prudente"
    elif signal in ("SHORT FORT", "SHORT"):
        dir_txt = "baissier"
        action  = "Short validé" if signal == "SHORT FORT" else "Short prudent"
    else:
        dir_txt = "neutre"
        action  = "Attente confirmationn"

    parts = [f"{action}. {pat_str} {dir_txt} sur {interval.upper()}."]
    if fc_str:
        parts.append(fc_str + ".")
    if sl and sl != current_price:
        sl_pct = abs(current_price - sl) / current_price * 100
        parts.append(f"SL ${sl:,.0f} (-{sl_pct:.1f}%) · TP1 ${tp1:,.0f} · R/R 1:{rr:.1f}.")
    return " ".join(parts)


def build_structured_signal(
    alpha:        dict,
    macro_detail: dict,
    tech_detail:  dict,
    forecasts:    dict,
    current_price: float,
    interval:      str = "1h",
    df:            "pd.DataFrame | None" = None,
) -> dict:
    """
    Construit le signal structuré tldr/boussole/timing depuis les données alpha.
    Compatible avec /api/signal endpoint.
    df : DataFrame OHLCV pour la détection d'anomalies IsolationForest (AXE 5).
    """
    signal     = alpha.get("signal", "NEUTRE")
    score      = float(alpha.get("alpha_score", 50))
    conviction = float(alpha.get("conviction", 50))
    fg         = get_fear_greed()
    anomalies  = detect_anomalies(df) if df is not None else {
        "anomaly_now": False, "score_now": 0.0, "recent_count": 0,
        "severity": "none", "interpretation": "DataFrame non fourni"
    }
    forecast_sig = alpha.get("forecast_detail", {})

    # ── TLDR ──────────────────────────────────────────────────────────────────
    conv_label, conv_bg, conv_color = _conviction_label(conviction)
    urgence = _URGENCE_MAP.get(signal, "SURVEILLER")
    resume  = _make_resume(signal, alpha, tech_detail, macro_detail, forecast_sig,
                           current_price, interval)

    atr_txt  = f"{alpha.get('sl_atr_multiple',2.0):.1f}×ATR · ATR {tech_detail.get('indicators',{}).get('atr_pct',0):.1f}%"
    pos_size = alpha.get("position_advice", "Position minimale")

    tldr = {
        "signal":          signal,
        "signal_dir":      "LONG" if "LONG" in signal else ("SHORT" if "SHORT" in signal else "NEUTRE"),
        "conviction":      conv_label,
        "conviction_pct":  round(conviction, 0),
        "conviction_color": conv_color,
        "alpha_score":     round(score, 1),
        "action":          alpha.get("position_advice", "—"),
        "resume":          resume,
        "urgence":         urgence,
        "strategie":       alpha.get("strategie", "—"),
        "horizon":         alpha.get("horizon_de_trade", "—"),
        "fear_greed":      fg,
        "risk_management": {
            "sl_price":           alpha.get("invalidation_sl"),
            "sl_pct":             -round(alpha.get("sl_distance_pct", 0), 2),
            "tp1":                alpha.get("objectif_tp1"),
            "tp2":                alpha.get("objectif_tp2"),
            "tp3":                alpha.get("objectif_tp3"),
            "risk_reward":        alpha.get("risk_reward"),
            "atr_basis":          atr_txt,
            "position_size_advice": pos_size,
        },
    }

    # ── BOUSSOLE (contexte macro/LT) ───────────────────────────────────────────
    macro_score   = float(macro_detail.get("score", 50))
    macro_label   = "FAVORABLE" if macro_score > 60 else "DÉFAVORABLE" if macro_score < 40 else "NEUTRE"
    liq_data      = macro_detail.get("liquidity", {}) or {}
    nli_chg       = float(liq_data.get("nli_change_4w", 0) or 0)
    cycle         = macro_detail.get("cycle", {}) or {}
    dom_signal    = ""
    for s in macro_detail.get("signals", []):
        if isinstance(s, (list, tuple)) and len(s) >= 3 and s[1] in ("+", "-"):
            dom_signal = s[2]
            break

    # Forecasts LT (7J / 30J depuis modèles)
    def _fc_summary(model_key: str, steps_idx: int = -1) -> dict:
        fc = forecasts.get(model_key, {})
        if not fc or not fc.get("p50"):
            return {}
        p50_last = fc["p50"][steps_idx]
        delta    = (p50_last - current_price) / current_price * 100 if current_price else 0
        conf     = fc.get("confidence", {}).get("confidence", 0)
        return {
            "p50": f"{delta:+.1f}%",
            "conf": round(conf, 0),
            "signal": "LONG" if delta > 1 else ("SHORT" if delta < -1 else "NEUTRE"),
        }

    indicators = tech_detail.get("indicators", {})
    macro_indics = {}
    mkt_data = macro_detail.get("market_data", {}) or {}
    for k, v_raw, interp_fn in [
        ("nli",   f"{nli_chg:+.2f}%/4w", lambda: (
            "Liquidité Fed en expansion — haussier BTC" if nli_chg > 1
            else "Liquidité Fed en contraction — baissier BTC" if nli_chg < -1
            else "Liquidité Fed stable")),
        ("m2",    f"{float(mkt_data.get('m2_yoy',0) or 0):+.1f}% YoY",
                  lambda: "Masse monétaire croissante — favorable actifs risqués" if float(mkt_data.get('m2_yoy',0) or 0) > 1
                  else "Masse monétaire contractée — risque déflationniste"),
        ("dxy",   f"{float(mkt_data.get('dxy_1m_chg',0) or 0):+.1f}%/1m",
                  lambda: "Dollar fort — pression baissière BTC" if float(mkt_data.get('dxy_1m_chg',0) or 0) > 1
                  else "Dollar faible — favorable BTC"),
        ("curve", f"{float(mkt_data.get('curve_10y2y',0) or 0):+.2f}%",
                  lambda: "Courbe inversée — signal récession" if float(mkt_data.get('curve_10y2y',0) or 0) < 0
                  else "Courbe normale — pas de signal récession"),
        ("vix",   str(float(alpha.get("collateral_detail", {}).get("vix", 0) or 0)),
                  lambda: "VIX élevé — risk-off" if float(alpha.get("collateral_detail",{}).get("vix",0) or 0) > 25
                  else "VIX faible — appétit au risque"),
    ]:
        macro_indics[k] = {"valeur": v_raw, "interpretation": interp_fn()}

    boussole = {
        "description":  "Contexte de fond — timeframes lents",
        "macro": {
            "score":            round(macro_score, 1),
            "label":            macro_label,
            "signal_dominant":  dom_signal,
            "nli_change_4w":    round(nli_chg, 2),
            "cycle_btc":        cycle,
            "interpretation_globale": (
                f"Macro {macro_label.lower()} (score {macro_score:.0f}/100). "
                + ("Liquidité en expansion." if nli_chg > 1 else
                   "Liquidité en contraction." if nli_chg < -1 else "")
            ),
            "indicateurs":      macro_indics,
        },
        "forecast_lt": {
            "moirai_7j":   _fc_summary("moirai"),
            "moirai_30j":  _fc_summary("moirai", -1),
            "chronos_7j":  _fc_summary("chronos"),
            "consensus":   "Modèles alignés" if abs(forecast_sig.get("divergence", 1)) < 1
                           else "Divergence inter-modèles",
        },
    }

    # ── TIMING (technique/CT) ──────────────────────────────────────────────────
    patterns    = tech_detail.get("patterns", [])
    dom_pattern = max(patterns, key=lambda p: p.get("confidence", 0)) if patterns else {}
    ms          = tech_detail.get("market_structure", {}) or {}
    kl          = ms.get("key_levels", {}) or {}
    pp          = kl.get("pivot_points", {}) or {}
    fib_data    = ms.get("fibonacci", {}) or {}

    # Supports / résistances depuis les niveaux clés
    supports     = []
    resistances  = []
    price_f      = current_price
    for label_s, key in [("S1","s1"),("S2","s2"),("S3","s3")]:
        val = float(pp.get(key, 0) or 0)
        if val > 0 and val < price_f:
            supports.append({"prix": round(val,0), "type": f"Pivot {label_s}"})
    for label_r, key in [("R1","r1"),("R2","r2"),("R3","r3")]:
        val = float(pp.get(key, 0) or 0)
        if val > 0 and val > price_f:
            resistances.append({"prix": round(val,0), "type": f"Pivot {label_r}"})

    # Zone d'entrée optimale (confluence fib + support)
    fib_levels   = fib_data.get("retracement_levels", {}) or {}
    entry_low    = price_f * 0.99
    entry_high   = price_f * 1.005
    entry_type   = "MARKET"
    entry_qual   = "B"
    if supports:
        s1 = supports[0]["prix"]
        if abs(s1 - price_f) / price_f < 0.03:
            entry_low  = round(s1 * 0.999, 0)
            entry_high = round(s1 * 1.002, 0)
            entry_type = "LIMIT_ORDER"
            entry_qual = "A"

    rsi_val   = float(indicators.get("rsi", 50) or 50)
    adx_val   = float(indicators.get("adx", 20) or 20)
    macd_h    = float(indicators.get("macd_hist", 0) or 0)
    vwap_val  = float(indicators.get("vwap", price_f) or price_f)
    stoch_k   = float(indicators.get("stoch_k", 50) or 50)
    bb_pct    = float((tech_detail.get("advanced",{}) or {}).get("bbw_percentile",{}).get("bbw_percentile",50) or 50)

    tech_indics = {
        "rsi":    {"valeur": round(rsi_val,1),
                   "interpretation": "Survente — rebond probable" if rsi_val < 30 else
                   "Surachat — correction possible" if rsi_val > 70 else "Neutre"},
        "macd":   {"valeur": round(macd_h,2),
                   "interpretation": "Momentum haussier" if macd_h > 0 else "Momentum baissier"},
        "vwap":   {"valeur": "dessus" if price_f > vwap_val else "dessous",
                   "interpretation": "Au-dessus du VWAP — momentum positif" if price_f > vwap_val
                   else "Sous le VWAP — pression vendeuse"},
        "adx":    {"valeur": round(adx_val,1),
                   "interpretation": "Tendance forte" if adx_val > 25 else
                   "Tendance modérée" if adx_val > 20 else "Pas de tendance — range"},
        "stoch":  {"valeur": round(stoch_k,1),
                   "interpretation": "Zone de survente" if stoch_k < 20 else
                   "Zone de surachat" if stoch_k > 80 else "Zone neutre"},
        "bb_pct": {"valeur": f"{bb_pct:.0f}e pctile",
                   "interpretation": "Compression historique — breakout imminent" if bb_pct < 15 else
                   "Volatilité normale" if bb_pct < 70 else "Expansion — tendance forte"},
    }

    timing = {
        "description":  "Timing d'exécution — timeframes rapides",
        "technique": {
            "score":    round(float(tech_detail.get("score", 50)), 1),
            "regime":   tech_detail.get("regime", "RANGE"),
            "pattern_dominant": {
                "nom":           dom_pattern.get("name", "—"),
                "direction":     dom_pattern.get("direction", "neutral"),
                "confidence":    dom_pattern.get("confidence", 0),
                "target":        round(price_f * (1 + dom_pattern.get("target_pct", 0)/100), 0)
                                 if dom_pattern.get("target_pct") else None,
                "description":   dom_pattern.get("description", ""),
            },
            "entree_optimale": {
                "zone_low":  round(entry_low,  0),
                "zone_high": round(entry_high, 0),
                "type":      entry_type,
                "qualite":   entry_qual,
            },
            "indicateurs_cles": tech_indics,
        },
        "forecast_ct": {
            "chronos_24h":  _fc_summary("chronos", 0) if forecasts.get("chronos",{}).get("p50") else {},
            "lagllama_24h": _fc_summary("lagllama", 0) if forecasts.get("lagllama",{}).get("p50") else {},
            "ensemble_delta_pct": round(forecast_sig.get("delta_pct", 0), 2),
            "modele_dominant": forecast_sig.get("models_used", [""])[0] if forecast_sig.get("models_used") else "—",
        },
        "niveaux_cles": {
            "supports":    supports[:3],
            "resistances": resistances[:3],
        },
    }

    # ── META ──────────────────────────────────────────────────────────────────
    models_actifs = [m for m in ["chronos","moirai","lagllama"]
                     if forecasts.get(m,{}).get("status")]

    warnings = list(alpha.get("warnings", []))
    if anomalies.get("severity") == "high":
        warnings.append(f"⚠ ANOMALIE DETECTEE : {anomalies['interpretation']}")
    elif anomalies.get("severity") == "low":
        warnings.append(f"Anomalies mineures : {anomalies['interpretation']}")

    meta = {
        "interval":          interval,
        "timestamp":         alpha.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "modeles_actifs":    models_actifs,
        "warnings":          warnings,
        "confiance_globale": round(conviction, 0),
        "dominant_factor":   alpha.get("dominant_factor", "—"),
        "anomalies":         anomalies,
    }

    return {"tldr": tldr, "boussole": boussole, "timing": timing, "meta": meta}


# ─── PRIORITÉ 6 : FEATURE SCORING DES SETUPS ──────────────────────────────────

def compute_setup_quality(alpha_data: dict,
                           ict_data: dict | None = None,
                           mtf_data: dict | None = None) -> dict:
    """
    Score de qualité du setup de trading (0-100).
    Critères pondérés :
      A) Confluence technique  25 pts
      B) Alignement MTF        20 pts
      C) ICT confirmation      20 pts
      D) Risk/Reward           15 pts
      E) Volume confirmation   10 pts
      F) Macro alignment       10 pts
    """
    score = 0.0
    breakdown: dict = {}

    # A — Confluence technique
    tech_score = float(alpha_data.get("score_technique", 50) or 50)
    a = max(0.0, (tech_score - 50) / 2.0)
    score += a
    breakdown["technique"] = round(a, 1)

    # B — Alignement MTF
    mtf_conf = float((mtf_data or {}).get("confluence_score", 0.5) or 0.5)
    b = mtf_conf * 20.0
    score += b
    breakdown["mtf"] = round(b, 1)

    # C — ICT confirmation
    ict_conf = float(
        ((ict_data or {}).get("mm_model") or {}).get("confidence", 0) or
        ((ict_data or {}).get("ict_lt_score", 50) or 50) - 50
    )
    c = max(0.0, ict_conf * 0.2)
    score += c
    breakdown["ict"] = round(c, 1)

    # D — Risk/Reward (idéal ≥ 2.5)
    rr = float(alpha_data.get("risk_reward", 1.0) or 1.0)
    d = min(15.0, rr * 5.0)
    score += d
    breakdown["risk_reward"] = round(d, 1)

    # E — Volume confirmation
    vol_ratio = float(alpha_data.get("volume_vs_avg", 1.0) or 1.0)
    e = min(10.0, (vol_ratio - 1.0) * 10.0) if vol_ratio > 1.0 else 0.0
    score += e
    breakdown["volume"] = round(e, 1)

    # F — Macro
    macro_score = float(alpha_data.get("score_macro", 50) or 50)
    f = max(0.0, (macro_score - 50.0) / 5.0)
    score += f
    breakdown["macro"] = round(f, 1)

    total = min(100.0, round(score, 1))
    grade = (
        "A+" if total >= 85 else
        "A"  if total >= 75 else
        "B"  if total >= 65 else
        "C"  if total >= 55 else "D"
    )

    return {
        "score":   total,
        "grade":   grade,
        "breakdown": breakdown,
        "interpretation": (
            f"Setup qualité {grade} ({total}/100). "
            + ("Setup institutionnel — haute conviction." if total >= 80 else
               "Setup solide — risque géré." if total >= 65 else
               "Setup moyen — taille réduite recommandée." if total >= 50 else
               "Setup faible — attendre confirmation.")
        ),
    }
