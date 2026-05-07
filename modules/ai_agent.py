"""
ai_agent.py — Agent IA Crypto à 4 couches
============================================
COUCHE 1 : OBSERVATION  — MarketObserver
COUCHE 2 : ANALYSE      — MarketAnalyst
COUCHE 3 : PRÉDICTION   — PredictionTracker
COUCHE 4 : APPRENTISSAGE — AdaptiveLearner

+ AltcoinStrategist (DCA Total3)
+ CryptoAgent       (orchestrateur principal)
"""

import os
import json
import math
import uuid
import time
import logging
import threading
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Chemins fichiers de données ──
_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_MEMORY_FILE   = os.path.join(_BASE, "agent_memory.json")
_PRED_LOG_FILE = os.path.join(_BASE, "prediction_log.json")
_REGIMES_FILE  = os.path.join(_BASE, "market_regimes.json")
_WATCHLIST_FILE = os.path.join(_BASE, "altcoin_watchlist.json")

# ── Verrous I/O ──
_pred_lock   = threading.Lock()
_memory_lock = threading.Lock()


def _load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"_save_json({path}) failed: {exc}")


# ══════════════════════════════════════════════════════════════
# COUCHE 1 : OBSERVATION DU MARCHÉ
# ══════════════════════════════════════════════════════════════

class MarketObserver:
    """Observe en continu le marché et construit un état complet."""

    # Cache léger pour Total3 (TTL 5 min)
    _total3_cache: dict = {}
    _total3_ts: float = 0.0
    _TOTAL3_TTL = 300

    # Barres par jour selon le timeframe
    _BARS_PER_DAY = {
        "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1, "1w": 1,
    }

    def get_market_state(self, btc_data: dict, macro_data: dict,
                          technical_data: dict, interval: str = "1h") -> dict:
        """Retourne un snapshot complet de l'état du marché, adapté au TF."""
        price   = float(btc_data.get("price", 0) or 0)
        change  = float(btc_data.get("change", 0) or 0)

        # Lookbacks TF-aware : 7J et 30J en nombre de bougies du TF
        hist_close = (technical_data.get("history", {}) or {}).get("close", [])
        bpd        = self._BARS_PER_DAY.get(interval, 24)
        change_7d  = self._pct_change(hist_close, max(1, int(7  * bpd)))
        change_30d = self._pct_change(hist_close, max(1, int(30 * bpd)))

        # Indicateurs techniques courants (tous issus du TF demandé)
        indicators  = technical_data.get("indicators", {}) or {}
        rsi_now     = float(indicators.get("rsi",       50)  or 50)
        adx_now     = float(indicators.get("adx",       20)  or 20)
        macd_hist   = float(indicators.get("macd_hist", 0)   or 0)
        hma_now     = float(indicators.get("hma",       0)   or 0)
        atr_pct     = float(indicators.get("atr_pct",   0)   or 0)
        cci_now     = float(indicators.get("cci",       0)   or 0)
        mfi_now     = float(indicators.get("mfi",       50)  or 50)
        willr_now   = float(indicators.get("willr",    -50)  or -50)
        _obv_raw    = indicators.get("obv_slope", 0) or 0
        if isinstance(_obv_raw, str):
            obv_slope = 1.0 if _obv_raw.lower() == "hausse" else -1.0
        else:
            obv_slope = float(_obv_raw)

        adv = technical_data.get("advanced", {}) or {}
        bbw_data = adv.get("bbw_percentile", {}) or {}
        bbw_pct  = float(bbw_data.get("bbw_percentile", 50) or 50)

        # Macro
        cycle = (macro_data.get("cycle") or {})
        bottom_prob = float(cycle.get("bottom_probability", 0) or 0)
        cycle_phase = str(cycle.get("phase", "UNKNOWN") or "UNKNOWN")
        macro_score = float(macro_data.get("score", 50) or 50)
        nli = macro_data.get("liquidity", {}) or {}
        nli_chg = float(nli.get("nli_change_4w", 0) or 0)
        nli_trend = "HAUSSE" if nli_chg > 1 else "BAISSE" if nli_chg < -1 else "STABLE"

        # Market Structure
        ms = technical_data.get("market_structure", {}) or {}
        dom = ms.get("dominant_structure") or {}
        dom_pattern  = str(dom.get("name", "—") or "—")
        dom_conf     = float(dom.get("confidence", 0) or 0)
        fib = ms.get("fibonacci", {}) or {}
        fib_level    = str(fib.get("closest_level", "—") or "—")

        # Supports / résistances via niveaux clés
        kl = ms.get("key_levels", {}) or {}
        pp = kl.get("pivot_points", {}) or {}
        key_support    = float(pp.get("s1", price * 0.97) or price * 0.97)
        key_resistance = float(pp.get("r1", price * 1.03) or price * 1.03)

        # Volume ratio
        vol_ratio = float(ms.get("volume_ratio", 1.0) or 1.0)

        # Régime de marché (TF-aware)
        regime_info = self.detect_market_regime(hist_close, indicators, adv, technical_data, interval=interval)

        # Total3 / altcoins
        total3 = self.get_total3_data()
        btc_dom   = float(total3.get("btc_dominance", 50) or 50)
        t3_trend  = "HAUSSE" if float(total3.get("total3_change_24h", 0) or 0) > 0 else "BAISSE"
        alt_prob  = self.compute_altseason_probability(
            btc_dom, t3_trend, macro_score, cycle_phase)

        return {
            # TF source des données
            "interval":            interval,
            # Prix
            "btc_price":           price,
            "btc_change_24h":      round(change, 2),
            "btc_change_7d":       round(change_7d, 2),
            "btc_change_30d":      round(change_30d, 2),
            # Régime
            "market_regime":       regime_info["regime"],
            "regime_confidence":   regime_info["confidence"],
            "regime_duration_days": regime_info["duration_days"],
            "prev_regime":         regime_info["prev_regime"],
            # Structure technique
            "dominant_pattern":    dom_pattern,
            "pattern_confidence":  dom_conf,
            "key_support":         round(key_support, 0),
            "key_resistance":      round(key_resistance, 0),
            "fib_level":           fib_level,
            # Macro
            "macro_score":         macro_score,
            "nli_trend":           nli_trend,
            "btc_cycle_phase":     cycle_phase,
            "bottom_probability":  bottom_prob,
            # Signaux TF-spécifiques
            "rsi_14":              round(rsi_now,   1),
            "adx":                 round(adx_now,   1),
            "macd_hist":           round(macd_hist, 4),
            "hma":                 round(hma_now,   2),
            "atr_pct":             round(atr_pct,   2),
            "cci":                 round(cci_now,   1),
            "mfi":                 round(mfi_now,   1),
            "willr":               round(willr_now, 1),
            "obv_slope":           round(obv_slope, 4),
            "bb_width_percentile": round(bbw_pct,   1),
            "volume_vs_avg":       round(vol_ratio,  2),
            # Total3
            "total3_trend":        t3_trend,
            "btc_dominance":       round(btc_dom, 2),
            "altseason_probability": alt_prob["probability"],
            "altseason_signal":    alt_prob["signal"],
            # Total3 raw
            "total3_data":         total3,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

    def _pct_change(self, closes: list, lookback: int) -> float:
        if len(closes) < lookback + 1:
            lookback = max(1, len(closes) - 1)
        if not closes or lookback < 1:
            return 0.0
        old = closes[-lookback - 1]
        new = closes[-1]
        if not old or old == 0:
            return 0.0
        return (new - old) / old * 100

    def detect_market_regime(self, hist_close: list, indicators: dict,
                              adv: dict, technical_data: dict,
                              interval: str = "1h") -> dict:
        """
        Identifie le régime parmi 6 états via votes pondérés.
        Le lookback momentum est adapté au TF (7J en bougies du TF).
        """
        rsi  = float(indicators.get("rsi", 50) or 50)
        adx  = float(indicators.get("adx", 20) or 20)
        macd = float(indicators.get("macd_hist", 0) or 0)
        hma  = float(indicators.get("hma", 0) or 0)
        bbw_data = adv.get("bbw_percentile", {}) or {}
        bbw_pct  = float(bbw_data.get("bbw_percentile", 50) or 50)
        atr  = float(indicators.get("atr", 0) or 0)

        # Lookback 7J en barres du TF courant
        bpd_7 = max(1, int(7 * self._BARS_PER_DAY.get(interval, 24)))
        price_now = hist_close[-1] if hist_close else 0
        price_7   = hist_close[-min(bpd_7, len(hist_close))] if hist_close else price_now
        mom_7d    = (price_now - price_7) / price_7 * 100 if price_7 else 0

        votes = {
            "bull_trend":      0.0,
            "bear_trend":      0.0,
            "accumulation":    0.0,
            "distribution":    0.0,
            "high_volatility": 0.0,
            "compression":     0.0,
        }

        # ADX fort → tendance
        if adx > 28:
            if rsi > 52 and mom_7d > 0:
                votes["bull_trend"] += 35
            elif rsi < 48 and mom_7d < 0:
                votes["bear_trend"] += 35
        elif adx > 20:
            if rsi > 52:
                votes["bull_trend"] += 15
            else:
                votes["bear_trend"] += 15

        # MACD momentum
        if macd > 0:
            votes["bull_trend"] += 12
        else:
            votes["bear_trend"] += 12

        # RSI zones
        if rsi > 60:
            votes["bull_trend"] += 10
        elif rsi < 40:
            votes["bear_trend"] += 10
        elif 45 < rsi < 55:
            votes["accumulation"] += 8
            votes["compression"]  += 5

        # HMA vs price
        if price_now > 0 and hma > 0:
            if price_now > hma * 1.005:
                votes["bull_trend"] += 10
            elif price_now < hma * 0.995:
                votes["bear_trend"] += 10

        # BBW percentile
        if bbw_pct <= 10:
            votes["compression"]     += 30
        elif bbw_pct <= 20:
            votes["compression"]     += 15
            votes["accumulation"]    += 8
        elif bbw_pct >= 85:
            votes["high_volatility"] += 30
        elif bbw_pct >= 70:
            votes["high_volatility"] += 18

        # Momentum 7J
        if abs(mom_7d) < 3 and adx < 20:
            votes["accumulation"]  += 12
            votes["distribution"]  += 8
        if mom_7d > 8:
            votes["bull_trend"]    += 12
        elif mom_7d < -8:
            votes["bear_trend"]    += 12

        # Régime gagnant
        best_regime     = max(votes, key=lambda k: votes[k])
        best_score      = votes[best_regime]
        total_votes     = sum(votes.values())
        confidence      = min(95, round(best_score / total_votes * 100, 1)) if total_votes > 0 else 50.0

        # Durée depuis mémoire persistante
        memory = _load_json(_MEMORY_FILE, {})
        last_regime       = memory.get("last_regime")
        last_regime_start = memory.get("last_regime_start")
        prev_regime       = memory.get("prev_regime", "INCONNU")

        now_iso = datetime.now(timezone.utc).isoformat()
        if last_regime != best_regime:
            # Transition de régime
            with _memory_lock:
                memory["prev_regime"]        = last_regime or best_regime
                memory["last_regime"]        = best_regime
                memory["last_regime_start"]  = now_iso
                _save_json(_MEMORY_FILE, memory)
            last_regime_start = now_iso
            prev_regime = last_regime or prev_regime

        duration_days = 0
        if last_regime_start:
            try:
                start_dt = datetime.fromisoformat(last_regime_start.replace("Z", "+00:00"))
                duration_days = max(0, (datetime.now(timezone.utc) - start_dt).days)
            except Exception:
                pass

        return {
            "regime":        best_regime,
            "confidence":    confidence,
            "duration_days": duration_days,
            "prev_regime":   prev_regime or "INCONNU",
            "votes":         {k: round(v, 1) for k, v in votes.items()},
        }

    def get_total3_data(self) -> dict:
        """Récupère les données Total3 via CoinGecko (cache 5 min)."""
        now = time.time()
        if (now - self._total3_ts) < self._TOTAL3_TTL and self._total3_cache:
            return self._total3_cache
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=8,
                headers={"User-Agent": "macro-alpha/2.0"}
            )
            data = r.json().get("data", {})
            total_mcap = float(data.get("total_market_cap", {}).get("usd", 0) or 0)
            pcts  = data.get("market_cap_percentage", {})
            btc_d = float(pcts.get("btc", 50) or 50)
            eth_d = float(pcts.get("eth", 15) or 15)
            btc_mcap  = total_mcap * btc_d / 100
            eth_mcap  = total_mcap * eth_d / 100
            total3    = total_mcap - btc_mcap - eth_mcap
            result = {
                "total3_usd":          round(total3, 0),
                "btc_dominance":       round(btc_d, 2),
                "eth_dominance":       round(eth_d, 2),
                "altcoin_dominance":   round(100 - btc_d - eth_d, 2),
                "total3_change_24h":   float(data.get("market_cap_change_percentage_24h_usd", 0) or 0),
                "total_market_cap":    round(total_mcap, 0),
            }
            MarketObserver._total3_cache = result
            MarketObserver._total3_ts    = now
            return result
        except Exception as exc:
            logger.warning(f"CoinGecko total3 error: {exc}")
            return MarketObserver._total3_cache or {}

    def compute_altseason_probability(self, btc_dom: float, total3_trend: str,
                                       macro_score: float, btc_cycle_phase: str) -> dict:
        """Probabilité d'altseason 0-100% avec signal qualitatif."""
        score = 0
        reasons = []

        # BTC dominance < 50% et en baisse → rotation imminente
        if btc_dom < 50 and total3_trend == "HAUSSE":
            score += 30
            reasons.append(f"BTC.D {btc_dom:.1f}% < 50% et Total3 en hausse")
        elif btc_dom < 55 and total3_trend == "HAUSSE":
            score += 20
            reasons.append(f"BTC.D {btc_dom:.1f}% en baisse, rotation alts")
        elif btc_dom > 60:
            score -= 10
            reasons.append(f"BTC.D {btc_dom:.1f}% > 60% — dominance forte, pas d'altseason")

        # Phase de cycle
        phase_upper = btc_cycle_phase.upper()
        if "BULL" in phase_upper or "HAUSSIER" in phase_upper:
            score += 20
            reasons.append("Phase de cycle haussière")
        elif "BEAR" in phase_upper or "BAISSIER" in phase_upper:
            score -= 15
            reasons.append("Phase baissière — alts surperforment rarement")

        # Macro favorable
        if macro_score > 65:
            score += 15
            reasons.append(f"Macro favorable ({macro_score:.0f}/100)")
        elif macro_score < 40:
            score -= 10
            reasons.append(f"Macro défavorable ({macro_score:.0f}/100)")

        # Total3 en hausse
        if total3_trend == "HAUSSE":
            score += 15
            reasons.append("Total3 en hausse 24H")

        score = max(0, min(100, 30 + score))  # base 30

        if score >= 70:
            signal = "FORTE"
        elif score >= 55:
            signal = "MODEREE"
        elif score >= 40:
            signal = "FAIBLE"
        else:
            signal = "NON"

        return {"probability": round(score, 0), "signal": signal, "reasons": reasons}


# ══════════════════════════════════════════════════════════════
# COUCHE 2 : ANALYSE — POURQUOI LE MARCHÉ SE COMPORTE AINSI
# ══════════════════════════════════════════════════════════════

class MarketAnalyst:
    """Analyse les causes des mouvements et génère des narratifs."""

    def analyze_price_movement(self, state_before: dict, state_after: dict,
                                time_delta_hours: float) -> dict:
        """Compare deux états et identifie les facteurs explicatifs."""
        if not state_before or not state_after:
            return {"catalyst": "Données insuffisantes", "factors": []}

        p_old = float(state_before.get("btc_price", 0) or 0)
        p_new = float(state_after.get("btc_price", 0) or 0)
        if p_old == 0:
            return {"catalyst": "Prix initial nul", "factors": []}

        chg_pct = (p_new - p_old) / p_old * 100
        factors = []

        # Régime changé ?
        r_old = state_before.get("market_regime", "")
        r_new = state_after.get("market_regime", "")
        if r_old != r_new:
            factors.append(f"Changement de régime : {r_old} → {r_new}")

        # Pattern complété ?
        p_old_pat = state_before.get("dominant_pattern", "")
        p_new_pat = state_after.get("dominant_pattern", "")
        if p_old_pat != p_new_pat and p_new_pat not in ("—", ""):
            factors.append(f"Nouveau pattern : {p_new_pat}")

        # Fib level franchi ?
        fib_old = state_before.get("fib_level", "")
        fib_new = state_after.get("fib_level", "")
        if fib_old != fib_new:
            factors.append(f"Niveau Fibonacci changé : {fib_old} → {fib_new}")

        # Volume spike
        vol_ratio = float(state_after.get("volume_vs_avg", 1) or 1)
        if vol_ratio > 2:
            factors.append(f"Volume spike ×{vol_ratio:.1f} — activité institutionnelle probable")
        elif vol_ratio > 1.5:
            factors.append(f"Volume élevé ×{vol_ratio:.1f}")

        # Macro changement
        m_old = float(state_before.get("macro_score", 50) or 50)
        m_new = float(state_after.get("macro_score", 50) or 50)
        if abs(m_new - m_old) > 5:
            direction = "amélioration" if m_new > m_old else "détérioration"
            factors.append(f"Macro : {direction} ({m_old:.0f}→{m_new:.0f})")

        catalyst = self.identify_catalyst(chg_pct, factors)
        return {
            "change_pct":      round(chg_pct, 2),
            "time_delta_hours": time_delta_hours,
            "catalyst":        catalyst,
            "factors":         factors,
        }

    def identify_catalyst(self, price_change_pct: float, factors: list) -> str:
        """Identifie le catalyseur probable et retourne une explication."""
        abs_chg = abs(price_change_pct)
        direction = "haussier" if price_change_pct > 0 else "baissier"

        if abs_chg < 0.5:
            return f"Mouvement latéral ({price_change_pct:+.1f}%) — consolidation sans catalyseur clair"

        factor_str = " · ".join(factors[:2]) if factors else "aucun facteur identifié"

        if abs_chg > 10:
            return (f"Mouvement majeur {direction} ({price_change_pct:+.1f}%) — "
                    f"catalyseur fort probable : {factor_str}")
        elif abs_chg > 5:
            return (f"Mouvement significatif {direction} ({price_change_pct:+.1f}%) — "
                    f"{factor_str}")
        elif abs_chg > 2:
            return (f"Mouvement modéré {direction} ({price_change_pct:+.1f}%) — "
                    f"{factor_str}")
        else:
            return (f"Mouvement mineur {direction} ({price_change_pct:+.1f}%) — "
                    f"bruit de marché ou {factor_str}")

    # Mapping TF → profil narratif
    _TF_PROFILE = {
        "5m":  {"label": "INTRADAY SCALP",   "horizon_cat": "scalp",    "macro_weight": "none",   "fc_focus": ["24H"]},
        "15m": {"label": "INTRADAY",          "horizon_cat": "intraday", "macro_weight": "none",   "fc_focus": ["24H"]},
        "1h":  {"label": "COURT TERME",       "horizon_cat": "intraday", "macro_weight": "light",  "fc_focus": ["24H", "7J"]},
        "4h":  {"label": "SWING 2-5J",        "horizon_cat": "swing",    "macro_weight": "medium", "fc_focus": ["7J", "30J"]},
        "1d":  {"label": "POSITIONNEL 2-8SEM","horizon_cat": "position", "macro_weight": "heavy",  "fc_focus": ["30J", "90J"]},
        "1w":  {"label": "MACRO LONG TERME",  "horizon_cat": "macro",    "macro_weight": "heavy",  "fc_focus": ["90J"]},
    }

    def generate_market_narrative(self, state: dict, predictions: list,
                                   history: list, interval: str = "1h") -> str:
        """
        Génère un narratif de marché structuré adapté au timeframe.
        Le focus, le vocabulaire et les forecasts mis en avant varient selon le TF.
        """
        if not state:
            return "Données insuffisantes pour générer un narratif."

        tf_profile  = self._TF_PROFILE.get(interval, self._TF_PROFILE["1h"])
        tf_label    = tf_profile["label"]
        macro_wt    = tf_profile["macro_weight"]
        fc_focus    = tf_profile["fc_focus"]

        price       = float(state.get("btc_price", 0) or 0)
        regime      = str(state.get("market_regime", "inconnu") or "inconnu")
        regime_days = int(state.get("regime_duration_days", 0) or 0)
        reg_conf    = float(state.get("regime_confidence", 0) or 0)
        pattern     = str(state.get("dominant_pattern", "—") or "—")
        pat_conf    = float(state.get("pattern_confidence", 0) or 0)
        macro_sc    = float(state.get("macro_score", 50) or 50)
        nli_trend   = str(state.get("nli_trend", "—") or "—")
        bottom_prob = float(state.get("bottom_probability", 0) or 0)
        cycle_phase = str(state.get("btc_cycle_phase", "—") or "—")
        rsi         = float(state.get("rsi_14",      50)  or 50)
        adx         = float(state.get("adx",         20)  or 20)
        macd_hist   = float(state.get("macd_hist",   0)   or 0)
        hma_price   = float(state.get("hma",         0)   or 0)
        atr_pct     = float(state.get("atr_pct",     0)   or 0)
        cci         = float(state.get("cci",         0)   or 0)
        mfi         = float(state.get("mfi",         50)  or 50)
        obv_slope   = float(state.get("obv_slope",   0)   or 0)
        bbw_pct     = float(state.get("bb_width_percentile", 50) or 50)
        vol_ratio   = float(state.get("volume_vs_avg", 1) or 1)
        support     = float(state.get("key_support", 0) or 0)
        resistance  = float(state.get("key_resistance", 0) or 0)
        fib_level   = str(state.get("fib_level", "—") or "—")
        btc_dom     = float(state.get("btc_dominance", 50) or 50)
        alt_prob    = float(state.get("altseason_probability", 0) or 0)
        chg_24h     = float(state.get("btc_change_24h", 0) or 0)
        chg_7d      = float(state.get("btc_change_7d", 0) or 0)
        chg_30d     = float(state.get("btc_change_30d", 0) or 0)
        # Relation prix / HMA (tendance directionnelle)
        hma_ctx = ""
        if hma_price > 0 and price > 0:
            hma_gap = (price - hma_price) / hma_price * 100
            hma_ctx = (f"prix {hma_gap:+.1f}% vs HMA" if abs(hma_gap) > 0.3 else "prix ≈ HMA")

        regime_labels = {
            "bull_trend":      "tendance haussière",
            "bear_trend":      "tendance baissière",
            "accumulation":    "phase d'accumulation",
            "distribution":    "phase de distribution",
            "high_volatility": "haute volatilité",
            "compression":     "compression historique",
        }
        regime_fr = regime_labels.get(regime, regime)

        if bbw_pct <= 10:
            vol_context = f"BBW {bbw_pct:.0f}e pctile (compression extrême)"
        elif bbw_pct <= 20:
            vol_context = f"BBW {bbw_pct:.0f}e pctile (compression)"
        elif bbw_pct >= 85:
            vol_context = f"BBW {bbw_pct:.0f}e pctile (expansion max)"
        else:
            vol_context = f"BBW {bbw_pct:.0f}e pctile"

        # ── Forecasts filtrés selon le TF ──
        pred_parts = []
        horizon_map = {"24H": "24H", "7J": "7J", "30J": "30J", "90J": "90J"}
        for wanted in fc_focus:
            p = next((x for x in (predictions or []) if x.get("horizon") == wanted), None)
            if p:
                chg = float(p.get("predicted_change", 0) or 0)
                conf = float(p.get("confidence", 0) or 0)
                pb   = float(p.get("prob_bull", 50) or 50)
                pred_parts.append(
                    f"{wanted} : {chg:+.1f}% · P(↑) {pb:.0f}% · conf {conf:.0f}%")
        pred_str = " | ".join(pred_parts) if pred_parts else ""

        # ══════════════════════════════════
        # INTRADAY (5m / 15m / 1h)
        # ══════════════════════════════════
        if macro_wt in ("none", "light"):
            # ── Focus : momentum pure sur les bougies du TF ──
            rsi_ctx = (f"RSI {rsi:.0f} suracheté → prudence" if rsi > 70
                       else f"RSI {rsi:.0f} survendu → rebond potentiel" if rsi < 30
                       else f"RSI {rsi:.0f}")
            macd_ctx = ("MACD histogramme positif (+momentum)" if macd_hist > 0
                        else "MACD histogramme négatif (–momentum)")
            vol_ctx  = (f"volume ×{vol_ratio:.1f} spike institutionnel" if vol_ratio > 2
                        else f"volume ×{vol_ratio:.1f} au-dessus moyenne" if vol_ratio > 1.3
                        else "volume dans la norme")
            adx_ctx  = (f"ADX {adx:.0f} tendance forte" if adx > 28
                        else f"ADX {adx:.0f} sans tendance nette")
            cci_ctx  = (f"CCI {cci:.0f} (surachat)" if cci > 100
                        else f"CCI {cci:.0f} (survente)" if cci < -100
                        else "")
            mfi_ctx  = (f"MFI {mfi:.0f} → argent entrant" if mfi > 60
                        else f"MFI {mfi:.0f} → distribution" if mfi < 40
                        else "")
            extra_signals = " · ".join(x for x in [cci_ctx, mfi_ctx, hma_ctx] if x)

            strat_map = {
                "bull_trend":      f"Long momentum — entrer sur pullback vers ${support:,.0f}, MACD {'positif ✓' if macd_hist > 0 else 'attention négatif'}, stop sous ${support:,.0f}",
                "bear_trend":      f"Biais short — rebonds vers ${resistance:,.0f} sont des opportunités de vente, RSI < 50 requis",
                "compression":     f"Range {vol_context} — achat sous ${support:,.0f} / vente sur ${resistance:,.0f}, attendre breakout volume ×2",
                "accumulation":    f"Accumulation — DCA progressif vers ${support:,.0f}, MFI {'entrant ✓' if mfi > 50 else 'surveiller'}",
                "high_volatility": f"ATR élevé ({atr_pct:.1f}%) — réduire taille de position ×0.5, stops larges obligatoires",
                "distribution":    f"Distribution active — ne pas initier de longs, MFI {mfi:.0f} montre sorties",
            }
            strat = strat_map.get(regime, "Attendre configuration technique claire")

            macro_line = ""
            if macro_wt == "light" and macro_sc != 50:
                macro_ctx = "favorable" if macro_sc > 60 else "défavorable" if macro_sc < 40 else "neutre"
                nli_ctx   = f", NLI {nli_trend.lower()}" if nli_trend != "STABLE" else ""
                macro_line = f"Macro (arrière-plan) : {macro_ctx} ({macro_sc:.0f}/100{nli_ctx}) — pas de poids sur ce TF."

            lines = [
                f"[{tf_label}] BTC ${price:,.0f} ({chg_24h:+.1f}% 24H · {chg_7d:+.1f}% 7J) — {regime_fr} depuis {regime_days}J ({reg_conf:.0f}% conf.) · {vol_context}.",
                f"Indicateurs {interval} : {rsi_ctx} · {adx_ctx} · {macd_ctx} · {vol_ctx}.",
                f"Structure : {pattern} (conf. {pat_conf:.0f}%) · S/R ${support:,.0f} / ${resistance:,.0f}" + (f" · {extra_signals}" if extra_signals else "") + ".",
            ]
            if macro_line:
                lines.append(macro_line)
            if pred_str:
                lines.append(f"Forecast IA ({', '.join(fc_focus)}) : {pred_str}.")
            lines.append(f"Setup {interval} : {strat}.")

        # ══════════════════════════════════
        # SWING (4h)
        # ══════════════════════════════════
        elif macro_wt == "medium":
            forces = []
            if macro_sc > 60:
                forces.append(f"macro {macro_sc:.0f}/100 (porteur)")
            elif macro_sc < 40:
                forces.append(f"macro {macro_sc:.0f}/100 (frein)")
            if nli_trend == "HAUSSE":
                forces.append("NLI Fed expansif")
            elif nli_trend == "BAISSE":
                forces.append("NLI Fed contractif")
            if rsi > 65:
                forces.append(f"RSI {rsi:.0f} momentum fort")
            elif rsi < 35:
                forces.append(f"RSI {rsi:.0f} zone rebond")
            if macd_hist > 0:
                forces.append("MACD positif")
            elif macd_hist < 0:
                forces.append("MACD négatif")
            if mfi > 60:
                forces.append(f"MFI {mfi:.0f} flux entrants")
            elif mfi < 40:
                forces.append(f"MFI {mfi:.0f} flux sortants")
            if vol_ratio > 1.5:
                forces.append(f"volume ×{vol_ratio:.1f}")
            if obv_slope > 0:
                forces.append("OBV haussier")
            elif obv_slope < 0:
                forces.append("OBV baissier")
            forces_str = " · ".join(forces[:4]) if forces else "contexte mixte"

            hma_swing = (f"prix au-dessus HMA ({hma_ctx})" if hma_price > 0 and price > hma_price
                         else f"prix sous HMA ({hma_ctx})" if hma_price > 0 and price < hma_price
                         else "")
            strat_map = {
                "bull_trend":      f"Swing long — replis vers HMA/${support:,.0f} sont des entrées, objectif ${resistance:,.0f}, trailing stop 2×ATR ({atr_pct:.1f}%)",
                "bear_trend":      f"Swing short — rebonds vers HMA/${resistance:,.0f} sont des ventes, target ${support:,.0f}",
                "compression":     f"Range 4H ${support:,.0f}–${resistance:,.0f} ({vol_context}) — jouer les extrêmes avec stops intra-range",
                "accumulation":    f"Accumulation swing — construire position par tranches sur ${support:,.0f}, stop sous dernier swing low",
                "high_volatility": f"ATR swing élevé ({atr_pct:.1f}%) — réduire taille, stops 3×ATR, attendre stabilisation ADX > 20",
                "distribution":    f"Distribution 4H — alléger les longs sur force vers ${resistance:,.0f}, MFI {mfi:.0f} confirme sorties",
            }
            strat = strat_map.get(regime, "Attendre structure swing claire")

            lines = [
                f"[{tf_label}] BTC ${price:,.0f} ({chg_24h:+.1f}% 24H · {chg_7d:+.1f}% 7J) — {regime_fr} depuis {regime_days}J ({reg_conf:.0f}% conf.) · {vol_context}.",
                f"Indicateurs 4H : RSI {rsi:.0f} · ADX {adx:.0f} · ATR {atr_pct:.1f}%{' · ' + hma_swing if hma_swing else ''}.",
                f"Facteurs directionnels : {forces_str}.",
                f"Structure swing : {pattern} (conf. {pat_conf:.0f}%) · Fib {fib_level} · S/R ${support:,.0f} / ${resistance:,.0f}.",
                f"Macro (pondération 25%) : {macro_sc:.0f}/100 · NLI {nli_trend} · Cycle {cycle_phase}.",
            ]
            if pred_str:
                lines.append(f"Forecasts IA ({', '.join(fc_focus)}) : {pred_str}.")
            lines.append(f"Swing setup : {strat}.")

        # ══════════════════════════════════
        # POSITIONNEL / MACRO (1d / 1w)
        # ══════════════════════════════════
        else:
            nli_ctx = ("expansion de liquidité" if nli_trend == "HAUSSE"
                       else "contraction de liquidité" if nli_trend == "BAISSE"
                       else "liquidité stable")
            macro_ctx = ("environnement macro favorable" if macro_sc > 65
                         else "macro dégradée" if macro_sc < 40
                         else "macro neutre")

            btc_dom_ctx = (f"BTC.D {btc_dom:.1f}% (dominance forte)" if btc_dom > 58
                           else f"BTC.D {btc_dom:.1f}% (rotation vers alts)" if btc_dom < 50
                           else f"BTC.D {btc_dom:.1f}%")

            # Signaux de confirmation long terme
            lt_signals = []
            if rsi > 70:
                lt_signals.append(f"RSI {rsi:.0f} zone de distribution")
            elif rsi < 40:
                lt_signals.append(f"RSI {rsi:.0f} zone d'accumulation")
            if mfi > 65:
                lt_signals.append(f"MFI {mfi:.0f} flux institutionnels entrants")
            elif mfi < 35:
                lt_signals.append(f"MFI {mfi:.0f} sorties institutionnelles")
            if obv_slope > 0:
                lt_signals.append("OBV tendance haussière confirmée")
            elif obv_slope < 0:
                lt_signals.append("OBV tendance baissière")
            if atr_pct > 3:
                lt_signals.append(f"volatilité élevée ATR {atr_pct:.1f}%")
            lt_sig_str = " · ".join(lt_signals) if lt_signals else "pas de signal extrême"

            strat_map = {
                "bull_trend":      f"Positionnel long — maintenir exposition, DCA sur replis ({chg_30d:+.1f}% 30J), alléger seulement si RSI > 80 ou NLI contractif",
                "bear_trend":      f"Défensif — réduire exposition spot, stables ou or, réaccumuler BTC uniquement sur capitulation (RSI < 30, MFI < 20)",
                "accumulation":    f"Accumulation cycle — DCA mensuel régulier, cible ${support:,.0f}–${resistance:,.0f}, horizon 3–6 mois",
                "compression":     f"Consolidation long terme — position de base maintenue, DCA progressif, breakout sur volume ×3 = signal d'achat fort",
                "high_volatility": f"Volatilité structurelle ({atr_pct:.1f}%) — spot uniquement, pas de levier, taille ×0.5, stops larges sur supports majeurs",
                "distribution":    f"Distribution de cycle — prises de profit progressives, ne pas rajouter, surveiller MFI < 40 et OBV baissier",
            }
            strat = strat_map.get(regime, "Positionnement neutre — attendre signal macro directeur")

            lines = [
                f"[{tf_label}] BTC ${price:,.0f} ({chg_24h:+.1f}% 24H · {chg_7d:+.1f}% 7J · {chg_30d:+.1f}% 30J) — {regime_fr} depuis {regime_days}J ({reg_conf:.0f}% conf.).",
                f"Macro ({macro_sc:.0f}/100) : {macro_ctx} · {nli_ctx} · cycle {cycle_phase} · bottom prob {bottom_prob:.0f}%.",
                f"Signaux long terme : {lt_sig_str}.",
                f"Structure : {pattern} (conf. {pat_conf:.0f}%) · Fib {fib_level} · {btc_dom_ctx} · altseason {alt_prob:.0f}%.",
                f"Technique ({vol_context}) : RSI {rsi:.0f} · ADX {adx:.0f} · S/R ${support:,.0f} / ${resistance:,.0f}.",
            ]
            if pred_str:
                lines.append(f"Forecasts IA ({', '.join(fc_focus)}) : {pred_str}.")
            lines.append(f"Stratégie positionnelle : {strat}.")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# COUCHE 3 : PRÉDICTION ET TRACKING
# ══════════════════════════════════════════════════════════════

class PredictionTracker:
    """Enregistre toutes les prédictions et suit leur réalisation."""

    HORIZONS = [
        {"label": "24H",  "hours": 24},
        {"label": "7J",   "hours": 168},
        {"label": "30J",  "hours": 720},
        {"label": "90J",  "hours": 2160},
    ]

    def _load(self) -> list:
        return _load_json(_PRED_LOG_FILE, [])

    def _save(self, log: list):
        _save_json(_PRED_LOG_FILE, log)

    def record_prediction(self, horizon_hours: int, model_name: str,
                           p10: float, p50: float, p90: float,
                           prob_bull: float, price_now: float,
                           market_state: dict, regime: str,
                           dominant_pattern: str) -> str:
        """Enregistre une prédiction dans le journal persistant."""
        pred_id = str(uuid.uuid4())[:8]
        now_utc = datetime.now(timezone.utc)
        resolve_at = (now_utc + timedelta(hours=horizon_hours)).isoformat()

        entry = {
            "id":                   pred_id,
            "created_at":           now_utc.isoformat(),
            "resolve_at":           resolve_at,
            "horizon_hours":        horizon_hours,
            "model":                model_name,
            "p10":                  round(p10, 2),
            "p50":                  round(p50, 2),
            "p90":                  round(p90, 2),
            "prob_bull":            round(prob_bull, 2),
            "price_at_prediction":  round(price_now, 2),
            "predicted_change_pct": round((p50 - price_now) / price_now * 100, 2),
            "regime":               regime,
            "dominant_pattern":     dominant_pattern,
            "market_state":         {k: v for k, v in (market_state or {}).items()
                                     if k not in ("history", "timestamp")},
            "resolved":             False,
            "actual_price":         None,
            "actual_change_pct":    None,
            "p10_hit":              None,
            "p50_hit":              None,
            "p90_hit":              None,
            "coverage_hit":         None,
            "direction_hit":        None,
            "error_analysis":       None,
        }

        with _pred_lock:
            log = self._load()
            log.append(entry)
            # Garder max 2000 entrées
            if len(log) > 2000:
                log = log[-2000:]
            self._save(log)

        return pred_id

    def resolve_matured_predictions(self, current_price: float) -> int:
        """Résout les prédictions dont l'échéance est dépassée. Retourne le nombre résolu."""
        now_utc = datetime.now(timezone.utc)
        resolved_count = 0

        with _pred_lock:
            log = self._load()
            changed = False
            for entry in log:
                if entry.get("resolved"):
                    continue
                try:
                    resolve_dt = datetime.fromisoformat(
                        entry["resolve_at"].replace("Z", "+00:00"))
                    if now_utc >= resolve_dt:
                        p10 = float(entry["p10"])
                        p50 = float(entry["p50"])
                        p90 = float(entry["p90"])
                        pb  = float(entry["prob_bull"])
                        pc  = float(entry["price_at_prediction"])
                        actual_chg = (current_price - pc) / pc * 100

                        entry["resolved"]           = True
                        entry["actual_price"]        = round(current_price, 2)
                        entry["actual_change_pct"]   = round(actual_chg, 2)
                        entry["p10_hit"]             = current_price >= p10
                        entry["p90_hit"]             = current_price <= p90
                        entry["coverage_hit"]        = p10 <= current_price <= p90
                        entry["p50_hit"]             = abs(current_price - p50) / p50 < 0.05
                        pred_bull = (p50 - pc) > 0
                        actual_bull = actual_chg > 0
                        entry["direction_hit"]       = pred_bull == actual_bull
                        entry["error_analysis"]      = self.analyze_error(entry, current_price)
                        resolved_count += 1
                        changed = True
                except Exception as exc:
                    logger.warning(f"resolve prediction {entry.get('id')}: {exc}")

            if changed:
                self._save(log)

        return resolved_count

    def analyze_error(self, prediction: dict, actual_price: float) -> str:
        """Identifie pourquoi une prédiction a réussi ou échoué."""
        p50 = float(prediction.get("p50", actual_price))
        pc  = float(prediction.get("price_at_prediction", actual_price))
        regime = prediction.get("regime", "inconnu")
        pattern = prediction.get("dominant_pattern", "—")

        error_pct = (actual_price - p50) / p50 * 100
        coverage  = prediction.get("coverage_hit", False)
        direction = prediction.get("direction_hit", False)

        if coverage and direction:
            return f"Prédiction réussie — modèle cohérent en régime {regime}"

        reasons = []
        if abs(error_pct) > 10:
            reasons.append(f"Erreur de magnitude importante ({error_pct:+.1f}%)")
        if not direction:
            reasons.append(f"Direction erronée — marché {('baissier' if actual_price < pc else 'haussier')} vs prédit {'haussier' if (p50-pc)>0 else 'baissier'}")
        if regime in ("high_volatility",):
            reasons.append("Régime haute volatilité — forecasts peu fiables")
        if pattern and pattern != "—":
            reasons.append(f"Pattern {pattern} — résultat à analyser")
        if not reasons:
            reasons.append("Hors de l'IC P10-P90 — queue de distribution non capturée")

        return " · ".join(reasons)

    def get_recent(self, n: int = 50) -> list:
        """Retourne les N prédictions les plus récentes."""
        log = self._load()
        return list(reversed(log[-n:]))

    def get_active(self) -> list:
        """Retourne les prédictions non encore résolues."""
        now_utc = datetime.now(timezone.utc)
        log = self._load()
        active = []
        for entry in reversed(log):
            if entry.get("resolved"):
                continue
            try:
                resolve_dt = datetime.fromisoformat(
                    entry["resolve_at"].replace("Z", "+00:00"))
                hours_remaining = max(0, (resolve_dt - now_utc).total_seconds() / 3600)
                active.append({
                    **entry,
                    "hours_remaining": round(hours_remaining, 1),
                    "horizon": next(
                        (h["label"] for h in self.HORIZONS
                         if h["hours"] == entry.get("horizon_hours")), "?")
                })
            except Exception:
                pass
        return active

    def get_performance_stats(self) -> dict:
        """Calcule les métriques de performance globales et par segment."""
        log = self._load()
        resolved = [e for e in log if e.get("resolved")]

        if not resolved:
            return {
                "overall": {
                    "total_predictions": len(log),
                    "resolved": 0,
                    "direction_accuracy": None,
                    "coverage_rate": None,
                    "mae_pct": None,
                    "best_horizon": None,
                    "best_regime": None,
                },
                "by_model": {},
                "by_regime": {},
                "by_pattern": {},
            }

        def _stats(subset):
            if not subset:
                return None
            dir_hits = [e["direction_hit"] for e in subset if e["direction_hit"] is not None]
            cov_hits = [e["coverage_hit"]  for e in subset if e["coverage_hit"]  is not None]
            errs     = [abs(e["actual_change_pct"] - e["predicted_change_pct"])
                        for e in subset
                        if e.get("actual_change_pct") is not None
                        and e.get("predicted_change_pct") is not None]
            return {
                "n":                 len(subset),
                "direction_accuracy": round(sum(dir_hits) / len(dir_hits), 3) if dir_hits else None,
                "coverage_rate":     round(sum(cov_hits) / len(cov_hits), 3) if cov_hits else None,
                "mae_pct":           round(float(np.mean(errs)), 2) if errs else None,
            }

        overall = _stats(resolved)

        # Par modèle
        models = set(e.get("model", "?") for e in resolved)
        by_model = {m: _stats([e for e in resolved if e.get("model") == m]) for m in models}

        # Par régime
        regimes = set(e.get("regime", "?") for e in resolved)
        by_regime = {r: _stats([e for e in resolved if e.get("regime") == r]) for r in regimes}

        # Par pattern
        patterns = set(e.get("dominant_pattern", "—") for e in resolved
                       if e.get("dominant_pattern") and e["dominant_pattern"] != "—")
        by_pattern = {p: _stats([e for e in resolved if e.get("dominant_pattern") == p])
                      for p in list(patterns)[:10]}

        # Best horizon
        best_h = None
        best_h_acc = 0.0
        for h in self.HORIZONS:
            sub = [e for e in resolved if e.get("horizon_hours") == h["hours"]]
            st  = _stats(sub)
            if st and st["direction_accuracy"] and st["direction_accuracy"] > best_h_acc:
                best_h_acc = st["direction_accuracy"]
                best_h = h["label"]

        # Best regime
        best_r = None
        best_r_acc = 0.0
        for r, st in by_regime.items():
            if st and st["direction_accuracy"] and st["direction_accuracy"] > best_r_acc:
                best_r_acc = st["direction_accuracy"]
                best_r = r

        overall["total_predictions"] = len(log)
        overall["resolved"]          = len(resolved)
        overall["best_horizon"]      = best_h
        overall["best_regime"]       = best_r

        return {
            "overall":    overall,
            "by_model":   by_model,
            "by_regime":  by_regime,
            "by_pattern": by_pattern,
        }

    def get_learning_insights(self) -> list:
        """Génère des insights textuels sur les erreurs récentes."""
        log = self._load()
        resolved = [e for e in log if e.get("resolved")][-30:]
        if not resolved:
            return ["Aucune prédiction résolue — apprentissage en cours"]

        insights = []
        # Direction accuracy globale
        dir_hits = [e["direction_hit"] for e in resolved if e["direction_hit"] is not None]
        if dir_hits:
            acc = sum(dir_hits) / len(dir_hits) * 100
            if acc > 65:
                insights.append(f"Direction accuracy {acc:.0f}% — modèles performants récemment")
            elif acc < 50:
                insights.append(f"Direction accuracy {acc:.0f}% — révision des poids en cours")

        # Régime le plus difficile
        regime_errs = {}
        for e in resolved:
            r = e.get("regime", "?")
            if not e.get("direction_hit"):
                regime_errs[r] = regime_errs.get(r, 0) + 1
        if regime_errs:
            worst = max(regime_errs, key=lambda x: regime_errs[x])
            insights.append(f"Régime le plus difficile : {worst} ({regime_errs[worst]} erreurs)")

        # Coverage
        cov_hits = [e["coverage_hit"] for e in resolved if e["coverage_hit"] is not None]
        if cov_hits:
            cov = sum(cov_hits) / len(cov_hits) * 100
            insights.append(f"Coverage P10-P90 : {cov:.0f}% (cible 80%)")

        return insights or ["Analyse en cours — données insuffisantes"]


# ══════════════════════════════════════════════════════════════
# COUCHE 4 : APPRENTISSAGE ET ADAPTATION
# ══════════════════════════════════════════════════════════════

class AdaptiveLearner:
    """Améliore les prédictions en fonction des erreurs passées."""

    def compute_dynamic_weights(self, performance_stats: dict,
                                 current_regime: str) -> dict:
        """
        Calcule les poids optimaux pour l'ensemble selon la performance
        historique de chaque modèle dans le régime actuel.
        """
        memory = _load_json(_MEMORY_FILE, {})
        base_weights = memory.get("model_weights_by_regime", {}).get(
            current_regime, {"chronos": 0.33, "moirai": 0.33, "lagllama": 0.34})

        by_model = performance_stats.get("by_model", {})
        if not by_model:
            return base_weights

        # Ajuster selon la performance récente
        accs = {}
        for m in ["chronos", "moirai", "lagllama"]:
            st = by_model.get(m)
            if st and st.get("direction_accuracy") is not None:
                accs[m] = float(st["direction_accuracy"])
            else:
                accs[m] = 0.5  # neutre si pas de données

        total_acc = sum(accs.values())
        if total_acc <= 0:
            return base_weights

        # Poids proportionnels à la performance
        raw_weights = {m: accs[m] / total_acc for m in accs}

        # Blending 60% base + 40% perf
        blended = {
            m: round(0.6 * base_weights.get(m, 0.33) + 0.4 * raw_weights.get(m, 0.33), 3)
            for m in ["chronos", "moirai", "lagllama"]
        }
        # Renormaliser
        total = sum(blended.values())
        blended = {m: round(v / total, 3) for m, v in blended.items()}

        # Cas haute volatilité → signal neutre
        by_regime = performance_stats.get("by_regime", {})
        hv_stats = by_regime.get("high_volatility", {})
        if (current_regime == "high_volatility" and
                hv_stats and hv_stats.get("direction_accuracy", 1) < 0.55):
            return {"chronos": 0.33, "moirai": 0.33, "lagllama": 0.34,
                    "_force_neutral": True}

        return blended

    def adjust_model_parameters(self, performance_stats: dict):
        """Ajuste les paramètres GBM selon les erreurs systématiques."""
        memory = _load_json(_MEMORY_FILE, {})
        gbm = memory.get("gbm_params", {})
        by_model = performance_stats.get("by_model", {})
        changed = False

        for m in ["chronos", "moirai", "lagllama"]:
            st = by_model.get(m)
            if not st:
                continue
            acc = st.get("direction_accuracy")
            mae = st.get("mae_pct")
            if acc is None:
                continue

            params = gbm.get(m, {})

            # Prédictions systématiquement trop hautes → réduire drift
            if acc < 0.45 and mae and mae > 5:
                if m == "moirai":
                    old = params.get("drift_adj", 0)
                    params["drift_adj"] = max(-0.3, old - 0.05)
                    changed = True
                elif m == "lagllama":
                    old = params.get("xi_adj", 0)
                    params["xi_adj"] = max(-0.2, old - 0.02)
                    changed = True

            # Bonne performance → récompenser légèrement
            if acc > 0.7:
                if m in ("chronos", "moirai"):
                    params["drift_adj"] = min(0.15, params.get("drift_adj", 0) + 0.01)
                    changed = True

            gbm[m] = params

        if changed:
            memory["gbm_params"] = gbm
            with _memory_lock:
                _save_json(_MEMORY_FILE, memory)

    def learn_from_pattern_outcomes(self, prediction_log: list):
        """Met à jour les seuils de confiance minimum par pattern."""
        regimes_data = _load_json(_REGIMES_FILE, {"pattern_outcomes": {}})
        outcomes = regimes_data.get("pattern_outcomes", {})

        for entry in prediction_log:
            if not entry.get("resolved"):
                continue
            pat = entry.get("dominant_pattern", "—")
            if not pat or pat == "—":
                continue
            if pat not in outcomes:
                outcomes[pat] = {"hits": 0, "total": 0}
            outcomes[pat]["total"] += 1
            if entry.get("direction_hit"):
                outcomes[pat]["hits"] += 1

        # Mettre à jour les seuils dans memory
        memory = _load_json(_MEMORY_FILE, {})
        thresholds = memory.get("pattern_confidence_thresholds", {})
        for pat, stats in outcomes.items():
            if stats["total"] >= 5:
                acc = stats["hits"] / stats["total"]
                # Augmenter le seuil si < 50% de précision
                current_thr = thresholds.get(pat, 55)
                if acc < 0.50:
                    thresholds[pat] = min(75, current_thr + 2)
                elif acc > 0.70:
                    thresholds[pat] = max(50, current_thr - 1)

        regimes_data["pattern_outcomes"] = outcomes
        regimes_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        _save_json(_REGIMES_FILE, regimes_data)

        memory["pattern_confidence_thresholds"] = thresholds
        with _memory_lock:
            _save_json(_MEMORY_FILE, memory)


# ══════════════════════════════════════════════════════════════
# ALTCOIN STRATEGIST
# ══════════════════════════════════════════════════════════════

class AltcoinStrategist:
    """Identifie les meilleurs moments pour DCA sur les altcoins."""

    DEFAULT_WATCHLIST = _load_json(_WATCHLIST_FILE, [
        {"symbol": "ETH",  "category": "L1",     "conviction": "high"},
        {"symbol": "SOL",  "category": "L1",     "conviction": "high"},
        {"symbol": "ARB",  "category": "L2",     "conviction": "medium"},
        {"symbol": "OP",   "category": "L2",     "conviction": "medium"},
        {"symbol": "TAO",  "category": "AI",     "conviction": "high"},
        {"symbol": "LINK", "category": "Oracle", "conviction": "high"},
        {"symbol": "AAVE", "category": "DeFi",   "conviction": "medium"},
        {"symbol": "KAS",  "category": "PoW",    "conviction": "medium"},
        {"symbol": "ALPH", "category": "PoW",    "conviction": "medium"},
    ])

    def get_altcoin_price(self, symbol: str) -> Optional[dict]:
        """Récupère le prix via Binance, fallback CoinGecko."""
        try:
            r = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": f"{symbol}USDT"},
                timeout=5
            )
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            d = r.json()
            if "code" in d:
                raise ValueError(d.get("msg", "Binance error"))
            return {
                "price":     float(d["lastPrice"]),
                "change_24h": float(d["priceChangePercent"]),
                "volume_24h": float(d["quoteVolume"]),
            }
        except Exception:
            pass
        # Fallback CoinGecko
        try:
            cg_ids = {"ETH": "ethereum", "SOL": "solana", "ARB": "arbitrum",
                      "OP": "optimism", "TAO": "bittensor", "LINK": "chainlink",
                      "AAVE": "aave", "KAS": "kaspa", "ALPH": "alephium"}
            cg_id = cg_ids.get(symbol)
            if not cg_id:
                return None
            r = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd",
                        "include_24hr_change": "true"},
                timeout=8,
                headers={"User-Agent": "macro-alpha/2.0"}
            )
            d = r.json().get(cg_id, {})
            if not d:
                return None
            return {
                "price":      float(d.get("usd", 0)),
                "change_24h": float(d.get("usd_24h_change", 0)),
                "volume_24h": 0.0,
            }
        except Exception as exc:
            logger.warning(f"get_altcoin_price({symbol}): {exc}")
            return None

    def compute_dca_signal(self, btc_state: dict, cycle_phase: str,
                            macro_score: float, vix: float = 20.0) -> dict:
        """Détermine le signal DCA optimal pour les altcoins."""
        btc_dom   = float(btc_state.get("btc_dominance", 50) or 50)
        btc_chg7  = float(btc_state.get("btc_change_7d", 0) or 0)
        alt_prob  = float(btc_state.get("altseason_probability", 0) or 0)
        t3_trend  = str(btc_state.get("total3_trend", "STABLE") or "STABLE")

        score = 0
        reasoning = []

        # BTC dominance
        if btc_dom > 58 and abs(btc_chg7) < 5:
            score += 25
            reasoning.append(f"BTC.D {btc_dom:.1f}% — phase précédant rotation alts")
        elif btc_dom < 50 and t3_trend == "HAUSSE":
            score += 30
            reasoning.append(f"BTC.D {btc_dom:.1f}% < 50%, Total3 en hausse — altseason active")
        elif btc_dom > 65:
            score -= 20
            reasoning.append(f"BTC.D {btc_dom:.1f}% trop élevée — pas de rotation")

        # Cycle
        phase_upper = cycle_phase.upper()
        if "ACCUMULATION" in phase_upper or ("BULL" in phase_upper and btc_dom > 55):
            score += 25
            reasoning.append(f"Phase {cycle_phase} — timing optimal pour accumulation alts")
        elif "DISTRIBUTION" in phase_upper or "BEAR" in phase_upper:
            score -= 25
            reasoning.append(f"Phase {cycle_phase} — éviter les alts en distribution")

        # Macro
        if macro_score > 65:
            score += 15
            reasoning.append(f"Macro favorable ({macro_score:.0f}/100)")
        elif macro_score < 40:
            score -= 20
            reasoning.append(f"Macro défavorable ({macro_score:.0f}/100)")

        # VIX
        if vix > 30:
            score -= 15
            reasoning.append(f"VIX {vix:.0f} > 30 — aversion au risque élevée")
        elif vix < 18:
            score += 5
            reasoning.append(f"VIX {vix:.0f} < 18 — environnement favorable")

        # BTC stable
        if abs(btc_chg7) < 5:
            score += 10
            reasoning.append(f"BTC stable ({btc_chg7:+.1f}% 7J) — alts peuvent performer")

        # Altseason probability
        if alt_prob > 65:
            score += 15
            reasoning.append(f"Probabilité altseason {alt_prob:.0f}%")

        timing_score = max(0, min(100, 30 + score))

        if timing_score >= 70:
            signal = "FORT"
            allocation = "40-50% du budget alts disponible"
        elif timing_score >= 55:
            signal = "MODERE"
            allocation = "25-35% du budget alts disponible"
        elif timing_score >= 40:
            signal = "FAIBLE"
            allocation = "10-20% du budget alts (DCA progressif)"
        else:
            signal = "ATTENTE"
            allocation = "0% — attendre meilleur timing"

        return {
            "dca_signal":            signal,
            "timing_score":          timing_score,
            "reasoning":             reasoning,
            "recommended_allocation": allocation,
            "btc_dominance":         round(btc_dom, 2),
            "altseason_probability": round(alt_prob, 0),
        }

    def rank_altcoins_by_opportunity(self, btc_state: dict,
                                      cycle_phase: str) -> list:
        """Classe les alts par opportunité et retourne le top 5."""
        watchlist = _load_json(_WATCHLIST_FILE, self.DEFAULT_WATCHLIST)
        ranked = []

        for item in watchlist:
            sym = item.get("symbol", "")
            conviction = item.get("conviction", "medium")
            category   = item.get("category", "")

            price_data = self.get_altcoin_price(sym)
            if not price_data:
                continue

            chg24 = float(price_data.get("change_24h", 0) or 0)
            vol24 = float(price_data.get("volume_24h", 0) or 0)

            # Score opportunité
            opp_score = 50

            # Conviction
            if conviction == "high":
                opp_score += 15
            elif conviction == "low":
                opp_score -= 10

            # Momentum 24H
            if -5 < chg24 < 0:
                opp_score += 8   # léger repli = opportunité DCA
            elif chg24 < -10:
                opp_score += 15  # fort repli = meilleur DCA
            elif chg24 > 15:
                opp_score -= 10  # trop tard

            # Catégorie favorable selon cycle
            phase_upper = cycle_phase.upper()
            if "BULL" in phase_upper and category in ("L1", "DeFi"):
                opp_score += 10
            if "ACCUMULATION" in phase_upper and category in ("L1", "L2"):
                opp_score += 8
            if category == "AI":
                opp_score += 5  # thème porteur

            ranked.append({
                "symbol":    sym,
                "category":  category,
                "conviction": conviction,
                "price":     price_data["price"],
                "change_24h": round(chg24, 2),
                "opportunity_score": min(100, max(0, opp_score)),
                "reasoning": f"{sym} ({category}) — conv. {conviction}, {chg24:+.1f}% 24H",
            })

        ranked.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return ranked[:5]

    def run_analysis(self, cache_data: dict) -> dict:
        """Point d'entrée principal pour l'analyse altcoins."""
        btc_price = float(cache_data.get("price", 0) or 0)
        macro     = cache_data.get("macro", {}) or {}
        tech      = cache_data.get("technical", {}) or {}
        cycle     = (macro.get("cycle") or {})
        cycle_phase = str(cycle.get("phase", "UNKNOWN") or "UNKNOWN")
        macro_score = float(macro.get("score", 50) or 50)
        vix = float((macro.get("market_data") or {}).get("vix", 20) or 20)

        observer = MarketObserver()
        total3 = observer.get_total3_data()

        btc_state_mini = {
            "btc_price":            btc_price,
            "btc_change_7d":        float(cache_data.get("change", 0) or 0),
            "btc_dominance":        float(total3.get("btc_dominance", 50) or 50),
            "total3_trend":         "HAUSSE" if float(total3.get("total3_change_24h", 0) or 0) > 0 else "BAISSE",
            "altseason_probability": observer.compute_altseason_probability(
                float(total3.get("btc_dominance", 50) or 50),
                "HAUSSE" if float(total3.get("total3_change_24h", 0) or 0) > 0 else "BAISSE",
                macro_score, cycle_phase
            )["probability"],
        }

        dca_signal = self.compute_dca_signal(btc_state_mini, cycle_phase, macro_score, vix)
        top_picks  = self.rank_altcoins_by_opportunity(btc_state_mini, cycle_phase)

        return {
            **total3,
            "dca_signal":          dca_signal,
            "top_picks":           top_picks,
            "altseason_probability": float(btc_state_mini["altseason_probability"]),
            "cycle_phase":         cycle_phase,
            "macro_score":         macro_score,
        }


# ══════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════

class CryptoAgent:
    """
    Orchestrateur à 4 couches.
    run_full_analysis() = point d'entrée unique pour /api/agent.
    """

    def __init__(self):
        self.observer  = MarketObserver()
        self.analyst   = MarketAnalyst()
        self.tracker   = PredictionTracker()
        self.learner   = AdaptiveLearner()
        self.altcoins  = AltcoinStrategist()

    def run_full_analysis(self, btc_data: dict, macro_data: dict,
                           technical_data: dict, forecasts: dict,
                           interval: str = "1h") -> dict:
        """
        Pipeline complet :
        1. Observer le marché
        2. Résoudre les prédictions maturées
        3. Enregistrer de nouvelles prédictions
        4. Apprendre des erreurs
        5. Générer le narratif (adapté au TF)
        6. Analyser les altcoins
        """
        price = float(btc_data.get("price", 0) or 0)

        # ── 1. État du marché (TF-aware) ──
        state = self.observer.get_market_state(btc_data, macro_data, technical_data, interval=interval)

        # ── 2. Résolution des prédictions maturées ──
        if price > 0:
            n_resolved = self.tracker.resolve_matured_predictions(price)
            if n_resolved > 0:
                logger.info(f"[Agent] {n_resolved} prédictions résolues")

        # ── 3. Performance & apprentissage ──
        perf_stats = self.tracker.get_performance_stats()
        dyn_weights = self.learner.compute_dynamic_weights(
            perf_stats, state.get("market_regime", "bull_trend"))
        self.learner.adjust_model_parameters(perf_stats)

        # ── 4. Enregistrer nouvelles prédictions (ensemble) ──
        active_preds = self.tracker.get_active()
        ensemble_preds = self._build_ensemble_predictions(
            forecasts, price, state, dyn_weights, active_preds)

        # ── 5. Narratif & analyse (adapté au TF) ──
        active_formatted = self._format_active_predictions(active_preds, price)
        narrative = self.analyst.generate_market_narrative(
            state, active_formatted, [], interval=interval)

        # ── 6. Altcoins ──
        cycle     = (macro_data.get("cycle") or {})
        cycle_phase = str(cycle.get("phase", "UNKNOWN") or "UNKNOWN")
        macro_score = float(macro_data.get("score", 50) or 50)
        total3    = self.observer.get_total3_data()
        altseason = self.observer.compute_altseason_probability(
            float(total3.get("btc_dominance", 50) or 50),
            "HAUSSE" if float(total3.get("total3_change_24h", 0) or 0) > 0 else "BAISSE",
            macro_score, cycle_phase
        )
        vix = float((macro_data.get("market_data") or {}).get("vix", 20) or 20)

        btc_state_mini = {
            **state,
            "btc_change_7d": state.get("btc_change_7d", 0),
        }
        dca = self.altcoins.compute_dca_signal(btc_state_mini, cycle_phase, macro_score, vix)

        # ── 7. Résolutions récentes ──
        recent = self.tracker.get_recent(10)
        last_resolved = [e for e in recent if e.get("resolved")][:3]

        return {
            # Régime
            "market_regime":      state["market_regime"],
            "regime_confidence":  state["regime_confidence"],
            "regime_duration_days": state["regime_duration_days"],
            "prev_regime":        state["prev_regime"],
            # Macro (utilisé par solana_bot._agent_context)
            "macro_score":        macro_score,
            "btc_cycle_phase":    cycle_phase,
            # Narratif
            "market_narrative":   narrative,
            # Prédictions actives
            "active_predictions": active_formatted,
            # DCA
            "dca_signal":         dca,
            "altseason":          altseason,
            "total3":             total3,
            # Performance
            "performance":        perf_stats,
            "dynamic_weights":    dyn_weights,
            "learning_insights":  self.tracker.get_learning_insights(),
            # Résolutions récentes
            "last_resolved":      self._format_resolved(last_resolved),
            # État complet
            "market_state":       {k: v for k, v in state.items()
                                   if k not in ("total3_data", "timestamp")},
            "timestamp":          state["timestamp"],
        }

    def _build_ensemble_predictions(self, forecasts: dict, price: float,
                                     state: dict, weights: dict,
                                     existing_active: list) -> list:
        """Enregistre les prédictions ensemble si pas encore actives sur cet horizon."""
        if price <= 0 or not forecasts:
            return []

        existing_horizons = {e.get("horizon_hours") for e in existing_active
                             if e.get("model") == "ensemble"}

        new_preds = []
        for h in self.tracker.HORIZONS:
            if h["hours"] in existing_horizons:
                continue  # déjà une pred active pour cet horizon

            # Mapper horizon → forecast key
            h_key = {24: "24h", 168: "7d", 720: "30d", 2160: "90d"}.get(h["hours"])
            # Utiliser les forecasts disponibles
            ps = {}
            for model in ["chronos", "moirai", "lagllama"]:
                fc = forecasts.get(model, {}) or {}
                if fc.get("p50"):
                    ps[model] = {
                        "p10": float(fc["p10"][-1]) if fc.get("p10") else price,
                        "p50": float(fc["p50"][-1]),
                        "p90": float(fc["p90"][-1]) if fc.get("p90") else price,
                        "prob_bull": float(fc.get("prob_bull", 50) or 50),
                    }

            if not ps:
                continue

            w = weights
            total_w = sum(w.get(m, 0.33) for m in ps)
            if total_w <= 0:
                continue

            ens_p50  = sum(ps[m]["p50"]  * w.get(m, 0.33) for m in ps) / total_w
            ens_p10  = sum(ps[m]["p10"]  * w.get(m, 0.33) for m in ps) / total_w
            ens_p90  = sum(ps[m]["p90"]  * w.get(m, 0.33) for m in ps) / total_w
            ens_bull = sum(ps[m]["prob_bull"] * w.get(m, 0.33) for m in ps) / total_w

            pred_id = self.tracker.record_prediction(
                horizon_hours=h["hours"],
                model_name="ensemble",
                p10=ens_p10,
                p50=ens_p50,
                p90=ens_p90,
                prob_bull=ens_bull,
                price_now=price,
                market_state=state,
                regime=state.get("market_regime", "?"),
                dominant_pattern=state.get("dominant_pattern", "—"),
            )
            new_preds.append(pred_id)

        return new_preds

    def _format_active_predictions(self, active: list, price: float) -> list:
        """Formate les prédictions actives pour l'API."""
        result = []
        horizon_labels = {24: "24H", 168: "7J", 720: "30J", 2160: "90J"}
        for entry in active:
            h_hours = entry.get("horizon_hours", 0)
            p50     = float(entry.get("p50", price) or price)
            pc      = float(entry.get("price_at_prediction", price) or price)
            pred_chg = (p50 - pc) / pc * 100 if pc else 0
            prob_bull = float(entry.get("prob_bull", 50) or 50)
            # Confiance approximée : 50 + |prob_bull - 50| * 0.8
            confidence = 50 + abs(prob_bull - 50) * 0.8
            result.append({
                "id":               entry.get("id"),
                "horizon":          horizon_labels.get(h_hours, f"{h_hours}H"),
                "horizon_hours":    h_hours,
                "model":            entry.get("model", "ensemble"),
                "predicted_change": round(pred_chg, 2),
                "p10":              entry.get("p10"),
                "p50":              entry.get("p50"),
                "p90":              entry.get("p90"),
                "prob_bull":        entry.get("prob_bull"),
                "confidence":       round(confidence, 0),
                "hours_remaining":  entry.get("hours_remaining", 0),
                "regime":           entry.get("regime"),
                "dominant_pattern": entry.get("dominant_pattern"),
            })
        # Trier par horizon
        result.sort(key=lambda x: x["horizon_hours"])
        return result

    def _format_resolved(self, resolved: list) -> list:
        """Formate les dernières résolutions pour affichage."""
        horizon_labels = {24: "24H", 168: "7J", 720: "30J", 2160: "90J"}
        out = []
        for e in resolved:
            h_hours = e.get("horizon_hours", 0)
            out.append({
                "id":               e.get("id"),
                "horizon":          horizon_labels.get(h_hours, f"{h_hours}H"),
                "predicted_change": e.get("predicted_change_pct"),
                "actual_change":    e.get("actual_change_pct"),
                "direction_hit":    e.get("direction_hit"),
                "coverage_hit":     e.get("coverage_hit"),
                "regime":           e.get("regime"),
                "pattern":          e.get("dominant_pattern"),
                "error_analysis":   e.get("error_analysis"),
            })
        return out


# ══════════════════════════════════════════════════════════════════════════════
# MetaLearner — Apprentissage supervisé sur l'historique des prédictions
# ══════════════════════════════════════════════════════════════════════════════

class MetaLearner:
    """
    Apprend des prédictions passées pour améliorer les prochaines.
    Utilise LightGBM quand assez de données (≥30), sinon heuristiques.
    Mémoire persistante : data/agent_memory.json
    """

    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../data/meta_model.lgb")
    MODEL_PKL   = os.path.join(os.path.dirname(__file__), "../data/meta_model.pkl")
    MEMORY_PATH = os.path.join(os.path.dirname(__file__), "../data/agent_memory.json")

    FEATURE_NAMES = [
        "macro_score", "tech_score", "forecast_score", "collat_score",
        "rsi", "adx", "bb_width_pct", "volume_ratio",
        "btc_dominance", "altseason_prob", "nli_change", "dxy_change",
        "vix", "bottom_prob", "mtf_confluence",
        "regime_bull", "regime_bear", "regime_accum", "regime_comp",
        "pattern_conf", "chronos_delta", "moirai_delta",
        "models_div", "chronos_conf", "cycle_progress",
    ]

    def __init__(self):
        self._lgb_booster = self._load_lgb_model()
        self._lgb_clf     = None  # sklearn API (chargé depuis pkl si disponible)
        self._memory      = self._load_memory()

    def _load_lgb_model(self):
        """Charge le modèle LightGBM Booster (.lgb) s'il existe."""
        try:
            import lightgbm as lgb
            if os.path.exists(self.MODEL_PATH):
                return lgb.Booster(model_file=self.MODEL_PATH)
        except Exception:
            pass
        return None

    def _load_memory(self) -> dict:
        if os.path.exists(self.MEMORY_PATH):
            try:
                with open(self.MEMORY_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"predictions_total": 0, "accuracy": None, "insight": None,
                "last_training": None, "top_features": []}

    def _save_memory(self, accuracy: float, top_features: list, n_samples: int):
        mem = self._load_memory()
        mem["last_training"]     = datetime.now(timezone.utc).isoformat()
        mem["accuracy"]          = round(accuracy, 3)
        mem["predictions_total"] = n_samples
        mem["top_features"]      = top_features[:5]
        mem["insight"] = (
            f"Modèle entraîné sur {n_samples} prédictions. "
            f"Feature la plus prédictive : {top_features[0]['name'] if top_features else '?'}. "
            f"Accuracy : {accuracy*100:.1f}%."
        ) if top_features else None
        try:
            os.makedirs(os.path.dirname(self.MEMORY_PATH), exist_ok=True)
            with open(self.MEMORY_PATH, "w") as f:
                json.dump(mem, f, indent=2)
        except Exception:
            pass
        self._memory = mem

    def build_features(self, state: dict) -> "np.ndarray":
        """Vecteur de 25 features normalisées [0,1] pour le méta-modèle."""
        import numpy as _np
        regime = str(state.get("market_regime", "") or "")
        return _np.array([
            float(state.get("macro_score", 50)    or 50)  / 100,
            float(state.get("tech_score",  50)    or 50)  / 100,
            float(state.get("forecast_score", 50) or 50)  / 100,
            float(state.get("collat_score",  50)  or 50)  / 100,
            float(state.get("rsi_14",        50)  or 50)  / 100,
            float(state.get("adx",           20)  or 20)  / 100,
            float(state.get("bb_width_percentile", 50) or 50) / 100,
            min(3.0, float(state.get("volume_vs_avg", 1) or 1)) / 3,
            float(state.get("btc_dominance",  55)  or 55) / 100,
            float(state.get("altseason_probability", 30) or 30) / 100,
            max(-5, min(5, float(state.get("nli_change_4w", 0) or 0))) / 5,
            max(-5, min(5, float(state.get("dxy_change",   0) or 0))) / 5,
            min(80, float(state.get("vix", 20) or 20)) / 80,
            float(state.get("bottom_probability", 50) or 50) / 100,
            float(state.get("mtf_confluence",    0.5) or 0.5),
            1.0 if "bull"  in regime.lower() and "bear" not in regime.lower() else 0.0,
            1.0 if "bear"  in regime.lower() else 0.0,
            1.0 if "accum" in regime.lower() else 0.0,
            1.0 if "comp"  in regime.lower() else 0.0,
            float(state.get("pattern_confidence", 0) or 0) / 100,
            max(-10, min(10, float(state.get("chronos_delta", 0) or 0))) / 10,
            max(-10, min(10, float(state.get("moirai_delta",  0) or 0))) / 10,
            min(5, float(state.get("models_divergence", 1) or 1)) / 5,
            float(state.get("chronos_conf", 50) or 50) / 100,
            float(state.get("cycle_phase_progress", 50) or 50) / 100,
        ], dtype=float)

    def build_feature_vector(self, market_state: dict, forecasts: dict = None) -> list:
        """Construit un vecteur de 20 features à partir du market_state."""
        ms = market_state or {}
        fc = forecasts or {}

        regime_map = {
            "bull_trend": 1, "accumulation": 0.5, "compression": 0,
            "distribution": -0.5, "bear_trend": -1, "high_volatility": -0.3
        }
        cycle_map = {
            "ACCUMULATION": 0, "EARLY_BULL": 0.3, "MID_BULL": 0.6,
            "LATE_BULL": 0.9, "DISTRIBUTION": 0.7, "BEAR": 0.1
        }
        nli_map = {"HAUSSE": 1, "STABLE": 0, "BAISSE": -1}

        features = [
            float(ms.get("rsi_14", 50) or 50),
            float(ms.get("adx", 25) or 25),
            float(ms.get("bb_width_percentile", 50) or 50),
            float(ms.get("volume_vs_avg", 1) or 1),
            float(ms.get("btc_change_24h", 0) or 0),
            float(ms.get("btc_change_7d", 0) or 0),
            float(ms.get("btc_change_30d", 0) or 0),
            float(ms.get("macro_score", 50) or 50),
            float(ms.get("regime_confidence", 50) or 50),
            float(ms.get("pattern_confidence", 50) or 50),
            float(ms.get("bottom_probability", 50) or 50),
            float(ms.get("altseason_probability", 30) or 30),
            float(ms.get("btc_dominance", 55) or 55),
            regime_map.get(str(ms.get("market_regime", "")), 0),
            cycle_map.get(str(ms.get("btc_cycle_phase", "")), 0.3),
            nli_map.get(str(ms.get("nli_trend", "STABLE")), 0),
            float((ms.get("total3_data") or {}).get("total3_change_24h", 0) or 0),
            float(fc.get("prob_bull", 50) or 50),
            float(fc.get("predicted_change_pct", 0) or 0),
            float(ms.get("regime_duration_days", 0) or 0),
        ]
        return features

    def predict_success_probability(self, features: "np.ndarray") -> float:
        """Prédit la probabilité de succès du signal actuel."""
        import numpy as _np
        fv = features.reshape(1, -1)
        fv = _np.nan_to_num(fv, nan=0.0)
        # Essai Booster LightGBM (API native)
        if self._lgb_booster is not None:
            try:
                return float(self._lgb_booster.predict(fv)[0])
            except Exception:
                pass
        # Heuristique avant entraînement
        macro_ok    = float(fv[0, 0]) > 0.60
        tech_ok     = float(fv[0, 1]) > 0.60
        forecast_ok = float(fv[0, 2]) > 0.50
        return 0.65 if (macro_ok and tech_ok) else (0.58 if forecast_ok else 0.45)

    def train_meta_model(self, prediction_log: list) -> dict:
        """
        Entraîne LightGBM sur les prédictions résolues (Booster API).
        Sauvegarde en .lgb + met à jour la mémoire agent_memory.json.
        """
        resolved = [p for p in prediction_log
                    if p.get("resolved") and p.get("direction_hit") is not None]

        if len(resolved) < 10:
            return {
                "status": "insufficient_data",
                "n_resolved": len(resolved),
                "message": f"Seulement {len(resolved)} prédictions résolues (min 10 requis)",
            }

        X, y = [], []
        for p in resolved:
            ms = p.get("market_state", {})
            fv = self.build_feature_vector(ms, p)
            X.append(fv)
            y.append(1 if p.get("direction_hit") else 0)

        import numpy as _np
        X_arr = _np.array(X, dtype=float)
        X_arr = _np.nan_to_num(X_arr, nan=0.0)

        try:
            import lightgbm as lgb
            from sklearn.model_selection import cross_val_score

            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                num_leaves=15, min_child_samples=3, verbose=-1,
                random_state=42
            )
            if len(resolved) >= 30:
                cv_scores = cross_val_score(model, X_arr, y, cv=3, scoring="accuracy")
                cv_acc = float(cv_scores.mean())
            else:
                cv_acc = None

            model.fit(X_arr, y)

            # Sauvegarde pkl (sklearn compat) + booster lgb
            import pickle
            os.makedirs(os.path.dirname(self.MODEL_PKL), exist_ok=True)
            with open(self.MODEL_PKL, "wb") as f:
                pickle.dump({"model": model, "n_features": X_arr.shape[1]}, f)

            # Aussi sauvegarder le booster natif LightGBM (.lgb)
            try:
                booster = model.booster_
                booster.save_model(self.MODEL_PATH)
                self._lgb_booster = booster
            except Exception:
                pass

            feat_names = self.FEATURE_NAMES[:X_arr.shape[1]]
            importances = dict(zip(feat_names, model.feature_importances_.tolist()))
            top_features_raw = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features = [{"name": k, "score": int(v)} for k, v in top_features_raw]

            self._save_memory(
                accuracy=cv_acc if cv_acc else sum(y)/max(len(y),1),
                top_features=top_features,
                n_samples=len(resolved)
            )

            return {
                "status":       "trained",
                "n_samples":    len(resolved),
                "cv_accuracy":  round(cv_acc, 3) if cv_acc else None,
                "top_features": top_features,
                "model_path":   self.MODEL_PATH,
                "memory":       self._memory,
            }

        except ImportError:
            logger.warning("LightGBM non disponible — fallback heuristique")
            accuracy = sum(y) / len(y)
            return {
                "status":        "heuristic",
                "n_samples":     len(resolved),
                "base_accuracy": round(accuracy, 3),
                "message":       "LightGBM non disponible — utilisation des heuristiques",
            }
        except Exception as exc:
            logger.error(f"train_meta_model: {exc}")
            return {"status": "error", "message": str(exc)}

    def predict_outcome_probability(self, current_features: list) -> dict:
        """Prédit la probabilité de succès. Modèle LightGBM si disponible."""
        try:
            import pickle, numpy as _np
            with open(self.MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            fv = _np.array(current_features, dtype=float).reshape(1, -1)
            fv = _np.nan_to_num(fv, nan=0.0)
            prob = float(model.predict_proba(fv)[0][1])
            return {"probability": round(prob, 3), "source": "lightgbm"}
        except Exception:
            try:
                rsi     = float(current_features[0]) if current_features else 50
                btc_chg = float(current_features[4]) if len(current_features) > 4 else 0
                macro   = float(current_features[7]) if len(current_features) > 7 else 50
                prob = 0.5
                if 45 < rsi < 65: prob += 0.08
                if btc_chg > 2:   prob += 0.06
                if macro > 55:    prob += 0.05
                return {"probability": round(min(0.85, max(0.15, prob)), 3), "source": "heuristic"}
            except Exception:
                return {"probability": 0.5, "source": "default"}

    def generate_insights(self, prediction_log: list, performance_stats: dict = None) -> list:
        """Génère des insights textuels depuis l'historique."""
        from collections import defaultdict, Counter
        resolved = [p for p in prediction_log if p.get("resolved")]
        insights = []

        if not resolved:
            return ["Pas encore de prédictions résolues — insights disponibles après les premières résolutions."]

        hits = [p for p in resolved if p.get("direction_hit")]
        acc  = len(hits) / len(resolved)

        insights.append(f"Précision directionnelle globale : {acc*100:.1f}% sur {len(resolved)} prédictions.")

        by_regime = defaultdict(list)
        for p in resolved:
            by_regime[p.get("regime", "unknown")].append(1 if p.get("direction_hit") else 0)

        for regime, hits_list in by_regime.items():
            r_acc = sum(hits_list) / len(hits_list)
            if r_acc > 0.7:
                insights.append(f"Régime '{regime}' : excellente précision ({r_acc*100:.0f}%) — surpondérer les signaux.")
            elif r_acc < 0.4:
                insights.append(f"Régime '{regime}' : précision faible ({r_acc*100:.0f}%) — réduire les tailles.")

        cov_hits = [p for p in resolved if p.get("coverage_hit")]
        if cov_hits:
            cov_acc = len(cov_hits) / len(resolved)
            insights.append(f"Couverture intervalles P10-P90 : {cov_acc*100:.1f}% (cible : 80%).")
            if cov_acc < 0.7:
                insights.append("Intervalles de confiance trop étroits — élargir les fourchettes.")

        pat_counts = Counter(p.get("dominant_pattern", "") for p in hits if p.get("dominant_pattern"))
        if pat_counts:
            best_pat, cnt = pat_counts.most_common(1)[0]
            insights.append(f"Pattern le plus fiable : '{best_pat}' (apparaît {cnt}x dans les succès).")

        return insights[:6]


# ══════════════════════════════════════════════════════════════════════════════
# MarketNarrator — Génère des briefings et setups en langage naturel
# ══════════════════════════════════════════════════════════════════════════════

class MarketNarrator:
    """Génère des textes structurés : brief quotidien + setups de trade."""

    def generate_daily_brief(self, full_state: dict) -> str:
        """Brief quotidien structuré en 7 sections."""
        import datetime as _dt
        now_str = _dt.datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")

        ms   = full_state.get("market_structure", {})
        mk   = full_state.get("market", {})
        ag   = full_state.get("agent", {})
        mc   = full_state.get("macro", {})

        btc   = float(mk.get("btc_price", 0) or 0)
        chg24 = float(mk.get("btc_change_24h", 0) or 0)
        chg7  = float(mk.get("btc_change_7d", 0) or 0)
        regime    = ag.get("market_regime", ms.get("market_regime", "INCONNU"))
        conf      = ag.get("regime_confidence", 0)
        pattern   = ag.get("dominant_pattern", ms.get("dominant_pattern", "—"))
        macro_sc  = float(mc.get("macro_score", full_state.get("macro_score", 50)) or 50)
        narrative_existing = ag.get("narrative", "")

        key_levels = ms.get("key_levels", {})
        support    = key_levels.get("key_support", 0)
        resistance = key_levels.get("key_resistance", 0)
        fib_level  = ms.get("fibonacci", {}).get("current_level", "—")

        indicators = ms.get("indicators", ms)
        adx  = float(indicators.get("adx", 0) or 0)
        rsi  = float(indicators.get("rsi_14", 0) or 0)
        bb_w = float(indicators.get("bb_width_percentile", 0) or 0)

        if chg24 > 3:    mood = "haussière"
        elif chg24 < -3: mood = "baissière"
        else:             mood = "neutre"

        macro_alert = ""
        if macro_sc < 40:
            macro_alert = f" | Alerte : environnement défavorable."
        elif macro_sc > 65:
            macro_alert = f" | Macro favorable."

        lines = [
            f"# Brief Marché — {now_str}",
            "",
            "## 1. Snapshot BTC",
            f"- Prix : ${btc:,.0f} ({chg24:+.2f}% 24H / {chg7:+.2f}% 7J)",
            f"- Humeur : {mood}",
            f"- RSI 14 : {rsi:.1f}  |  ADX : {adx:.1f}  |  BB Width : {bb_w:.0f}e percentile",
            "",
            "## 2. Régime de marché",
            f"- Régime actuel : {regime} (confiance {conf}%)",
            f"- Pattern dominant : {pattern}",
            f"- Niveau Fibonacci : {fib_level}",
            "",
            "## 3. Niveaux clés",
            f"- Support : ${support:,.0f}" if support else "- Support : N/A",
            f"- Résistance : ${resistance:,.0f}" if resistance else "- Résistance : N/A",
            "",
            "## 4. Contexte macro",
            f"- Score macro : {macro_sc:.0f}/100{macro_alert}",
            "",
            "## 5. Analyse narrative",
            narrative_existing if narrative_existing else "_Analyse non disponible_",
            "",
            "## 6. Actions recommandées",
        ]

        if "bull" in str(regime).lower():
            lines += [
                "- Maintenir/augmenter les positions longues existantes",
                "- Rechercher des entrées sur retracements Fibonacci",
                "- Stops en breakeven après TP1",
            ]
        elif "bear" in str(regime).lower():
            lines += [
                "- Réduire l'exposition — capital en cash ou stablecoins",
                "- Ne pas attraper les couteaux qui tombent",
                "- Attendre confirmation d'un retournement",
            ]
        else:
            lines += [
                "- Phase de patience — constituer des positions progressivement",
                "- DCA sur les supports identifiés",
                "- Surveiller le breakout avec volume",
            ]

        lines += [
            "",
            "## 7. Prochaines échéances",
            "- Vérifier la résolution des prédictions actives (voir onglet Apprentissage)",
            "- Mettre à jour le suivi des positions ouvertes",
            "",
            "---",
            "_Brief généré automatiquement par Macro Alpha Agent_",
        ]

        return "\n".join(str(l) for l in lines)

    def generate_trade_setup(self, signal: dict, entry_zone: dict = None,
                              stop: dict = None, tps: dict = None,
                              mtf: dict = None) -> str:
        """Génère un setup de trade formaté et prêt à copier."""
        ez   = entry_zone or {}
        sl   = stop or {}
        tp_d = tps or {}
        mt   = mtf or {}

        symbol    = signal.get("symbol", "BTCUSDT")
        direction = signal.get("direction", "long").upper()
        pattern   = signal.get("pattern", "—")
        regime    = signal.get("regime", "—")
        quality   = signal.get("setup_quality", "B")
        conviction = mt.get("conviction", "MEDIUM")

        entry_low  = float(ez.get("zone_low", 0) or 0)
        entry_high = float(ez.get("zone_high", 0) or 0)
        stop_price = float(sl.get("stop_loss", 0) or 0)
        stop_dist  = float(sl.get("distance_pct", 0) or 0)
        stop_type  = sl.get("stop_type", "ATR")

        tp1 = tp_d.get("tp1", {})
        tp2 = tp_d.get("tp2", {})
        tp3 = tp_d.get("tp3", {})
        avg_rr = float(tp_d.get("avg_rr", 0) or 0)

        mtf_bias   = mt.get("global_bias", "—")
        conf_score = float(mt.get("confluence_score", 0) or 0)
        aligned    = mt.get("aligned_timeframes", [])

        lines = [
            "=" * 50,
            f"  SETUP {direction} — {symbol}",
            "=" * 50,
            "",
            f"Pattern     : {pattern}",
            f"Regime      : {regime}",
            f"Qualite     : {quality}  |  Conviction MTF : {conviction}",
            "",
            "ENTREE",
            f"  Zone      : ${entry_low:,.2f} — ${entry_high:,.2f}",
            f"  Type      : {ez.get('entry_type', 'LIMIT_ORDER')}",
            "",
            "STOP LOSS",
            f"  Prix      : ${stop_price:,.2f}  ({stop_dist:.1f}% distance)",
            f"  Type      : {stop_type}",
            "",
            "TAKE PROFITS",
            f"  TP1 (33%) : ${float(tp1.get('price', 0) or 0):,.2f}  RR={float(tp1.get('rr', 0) or 0):.1f}",
            f"  TP2 (33%) : ${float(tp2.get('price', 0) or 0):,.2f}  RR={float(tp2.get('rr', 0) or 0):.1f}",
            f"  TP3 (34%) : ${float(tp3.get('price', 0) or 0):,.2f}  RR={float(tp3.get('rr', 0) or 0):.1f}",
            f"  RR moyen  : {avg_rr:.2f}",
            "",
            "MTF CONFLUENCES",
            f"  Biais     : {mtf_bias}  (score {conf_score:.0f}%)",
            f"  TF alignes: {', '.join(aligned) if aligned else 'N/A'}",
            "",
            "GESTION",
            "  > Breakeven au TP1",
            "  > Trailing apres TP2 (distance 1x ATR)",
            "  > Invalider si cloture sous le stop",
            "",
            "=" * 50,
        ]

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# PORTFOLIO SIMULATOR — Suivi de positions simulées
# ═══════════════════════════════════════════════════════════

class PortfolioSimulator:
    """Simule l'ouverture/fermeture de positions avec suivi PnL."""

    POSITIONS_FILE = os.path.join(_BASE, "simulated_positions.json")

    def open_position(self, signal: str, entry_price: float, sl: float,
                      tp1: float, tp2: float, size_pct: float = 2.0,
                      capital: float = 10_000.0) -> dict:
        position = {
            "id": str(uuid.uuid4())[:8],
            "signal": signal,
            "entry_price": entry_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "size_pct": size_pct,
            "capital_risk": round(capital * size_pct / 100, 2),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "current_price": entry_price,
        }
        positions = self._load()
        positions.append(position)
        self._save(positions)
        logger.info(f"[Portfolio] Position ouverte {position['id']}: {signal} @ ${entry_price:,.0f}")
        return position

    def update_positions(self, current_price: float) -> list:
        positions = self._load()
        for p in positions:
            if p["status"] != "OPEN":
                continue
            entry = float(p["entry_price"])
            sl    = float(p["sl"])
            tp1   = float(p["tp1"])
            tp2   = float(p["tp2"])
            if p["signal"] in ("LONG", "LONG FORT"):
                pnl = (current_price - entry) / entry
                if current_price <= sl:
                    p["status"] = "STOPPED"
                elif current_price >= tp2:
                    p["status"] = "TP2_HIT"
                elif current_price >= tp1:
                    p["status"] = "TP1_HIT"
            else:
                pnl = (entry - current_price) / entry
                if current_price >= sl:
                    p["status"] = "STOPPED"
                elif current_price <= tp2:
                    p["status"] = "TP2_HIT"
                elif current_price <= tp1:
                    p["status"] = "TP1_HIT"
            risk = float(p.get("capital_risk", 0))
            p["pnl_pct"]       = round(pnl * 100, 2)
            p["pnl_usd"]       = round(pnl * risk / max(float(p.get("size_pct", 2)) / 100, 0.001), 2)
            p["current_price"] = current_price
        self._save(positions)
        return positions

    def close_position(self, position_id: str, current_price: float) -> Optional[dict]:
        positions = self._load()
        for p in positions:
            if p["id"] == position_id and p["status"] == "OPEN":
                entry = float(p["entry_price"])
                if p["signal"] in ("LONG", "LONG FORT"):
                    pnl = (current_price - entry) / entry
                else:
                    pnl = (entry - current_price) / entry
                p["status"]        = "CLOSED_MANUAL"
                p["pnl_pct"]       = round(pnl * 100, 2)
                p["pnl_usd"]       = round(pnl * float(p.get("capital_risk", 0)), 2)
                p["current_price"] = current_price
                p["closed_at"]     = datetime.now(timezone.utc).isoformat()
                self._save(positions)
                return p
        return None

    def get_stats(self) -> dict:
        all_pos = self._load()
        closed  = [p for p in all_pos if p["status"] != "OPEN"]
        open_   = [p for p in all_pos if p["status"] == "OPEN"]
        if not closed:
            return {
                "message": "Pas encore de positions fermées",
                "total_trades": 0,
                "open_positions": len(open_),
            }
        wins   = [p for p in closed if p.get("pnl_pct", 0) > 0]
        losses = [p for p in closed if p.get("pnl_pct", 0) <= 0]
        return {
            "total_trades":   len(closed),
            "open_positions": len(open_),
            "win_rate":       round(len(wins) / len(closed) * 100, 1),
            "total_pnl_usd":  round(sum(p.get("pnl_usd", 0) for p in closed), 2),
            "avg_win_pct":    round(sum(p.get("pnl_pct", 0) for p in wins)   / max(1, len(wins)),   2),
            "avg_loss_pct":   round(sum(p.get("pnl_pct", 0) for p in losses) / max(1, len(losses)), 2),
            "best_trade":     max(closed, key=lambda x: x.get("pnl_pct", 0), default={}),
            "worst_trade":    min(closed, key=lambda x: x.get("pnl_pct", 0), default={}),
            "profit_factor":  round(
                abs(sum(p.get("pnl_pct", 0) for p in wins)) /
                max(0.01, abs(sum(p.get("pnl_pct", 0) for p in losses))),
                2,
            ),
            "positions":      all_pos,
        }

    def _load(self) -> list:
        return _load_json(self.POSITIONS_FILE, [])

    def _save(self, positions: list) -> None:
        _save_json(self.POSITIONS_FILE, positions)


# ═══════════════════════════════════════════════════════════
# ENTRY TIMING MODEL — LightGBM binary classifier
# Prédit si le moment d'entrée est optimal (1) ou non (0)
# ═══════════════════════════════════════════════════════════

class EntryTimingModel:
    """
    Classifieur binaire LightGBM : timing optimal (1) vs sous-optimal (0).
    Features : 12 indicateurs techniques + macro.
    Entraîné sur les données historiques du PortfolioSimulator.
    """

    MODEL_FILE = os.path.join(_BASE, "entry_timing_model.json")

    FEATURES = [
        "rsi_14", "rsi_slope",
        "macd_hist", "macd_signal_cross",
        "bb_width", "bb_position",
        "volume_ratio",          # volume_now / volume_avg_20
        "atr_pct",               # ATR / price
        "price_vs_ema20",        # (price - ema20) / ema20
        "price_vs_ema50",
        "macro_score",
        "alpha_score",
    ]

    def __init__(self):
        self._model        = None
        self._trained      = False
        self._feature_names = self.FEATURES
        self._load_model()

    def _load_model(self):
        """Charge le modèle depuis le disque si disponible."""
        try:
            import lightgbm as lgb
            if os.path.exists(self.MODEL_FILE):
                self._model   = lgb.Booster(model_file=self.MODEL_FILE)
                self._trained = True
                logger.info("[EntryTiming] Modèle chargé depuis %s", self.MODEL_FILE)
        except Exception as exc:
            logger.warning("[EntryTiming] Impossible de charger le modèle: %s", exc)

    def build_features(self, tech: dict, alpha_score: float = 50.0,
                       macro_score: float = 50.0) -> Optional[list]:
        """
        Construit le vecteur de features depuis le résultat de full_technical_analysis.
        Retourne None si les données sont insuffisantes.
        """
        try:
            ind = tech.get("indicators", {})

            rsi          = float(ind.get("rsi", 50) or 50)
            macd         = ind.get("macd", {}) or {}
            macd_hist    = float(macd.get("histogram", 0) or 0)
            bb           = ind.get("bb", {}) or {}
            bb_upper     = float(bb.get("upper", 0) or 0)
            bb_lower     = float(bb.get("lower", 0) or 0)
            bb_mid       = float(bb.get("middle", 0) or 0)
            price        = float(tech.get("price", 0) or 0)
            atr          = float(ind.get("atr", 0) or 0)
            ema20        = float(ind.get("ema20", price) or price)
            ema50        = float(ind.get("ema50", price) or price)
            volume_ratio = float(ind.get("volume_ratio", 1.0) or 1.0)

            if price <= 0:
                return None

            bb_width    = (bb_upper - bb_lower) / max(bb_mid, 1e-6) if bb_mid > 0 else 0.0
            bb_position = (price - bb_lower) / max(bb_upper - bb_lower, 1e-6) if bb_upper > bb_lower else 0.5
            atr_pct     = atr / price if price > 0 else 0.0
            p_vs_ema20  = (price - ema20) / max(ema20, 1e-6)
            p_vs_ema50  = (price - ema50) / max(ema50, 1e-6)

            # Proxy pour les dérivées (pas de données temporelles ici)
            rsi_slope          = 0.0   # serait calculé sur N périodes
            macd_signal_cross  = 1.0 if macd_hist > 0 else 0.0

            return [
                rsi, rsi_slope,
                macd_hist, macd_signal_cross,
                bb_width, bb_position,
                volume_ratio,
                atr_pct,
                p_vs_ema20,
                p_vs_ema50,
                float(macro_score),
                float(alpha_score),
            ]
        except Exception as exc:
            logger.warning("[EntryTiming] build_features: %s", exc)
            return None

    def predict(self, tech: dict, alpha_score: float = 50.0,
                macro_score: float = 50.0) -> dict:
        """
        Prédit la probabilité que le timing soit optimal.
        Retourne {"prob": 0.0-1.0, "signal": "OPTIMAL"|"SUBOPTIMAL"|"NEUTRAL", "trained": bool}.
        """
        features = self.build_features(tech, alpha_score, macro_score)
        if features is None:
            return {"prob": 0.5, "signal": "NEUTRAL", "trained": False, "reason": "features insuffisantes"}

        if not self._trained or self._model is None:
            # Heuristique simple si pas encore entraîné
            rsi         = features[0]
            macd_cross  = features[3]
            bb_pos      = features[5]
            vol_ratio   = features[6]
            alpha       = features[11]

            prob = 0.5
            if 40 <= rsi <= 60:
                prob += 0.1
            if macd_cross > 0:
                prob += 0.1
            if 0.3 <= bb_pos <= 0.7:
                prob += 0.05
            if vol_ratio > 1.2:
                prob += 0.1
            if alpha > 65:
                prob += 0.1
            elif alpha < 35:
                prob -= 0.15

            prob = max(0.0, min(1.0, prob))
            signal = "OPTIMAL" if prob >= 0.65 else "SUBOPTIMAL" if prob <= 0.35 else "NEUTRAL"
            return {"prob": round(prob, 3), "signal": signal, "trained": False}

        try:
            import numpy as np
            import lightgbm as lgb
            X    = np.array([features])
            prob = float(self._model.predict(X)[0])
            signal = "OPTIMAL" if prob >= 0.65 else "SUBOPTIMAL" if prob <= 0.35 else "NEUTRAL"
            return {"prob": round(prob, 3), "signal": signal, "trained": True}
        except Exception as exc:
            logger.warning("[EntryTiming] predict error: %s", exc)
            return {"prob": 0.5, "signal": "NEUTRAL", "trained": False, "error": str(exc)}

    def train(self, positions: list, tech_history: list) -> dict:
        """
        Entraîne le modèle sur les positions historiques.
        positions : liste depuis PortfolioSimulator.get_stats()["positions"]
        tech_history : liste de snapshots techniques au moment de chaque entrée
        """
        try:
            import numpy as np
            import lightgbm as lgb

            closed = [p for p in positions if p.get("status") not in ("OPEN",)]
            if len(closed) < 20:
                return {"trained": False, "reason": f"Pas assez de données ({len(closed)} trades < 20)"}

            X, y = [], []
            for pos in closed:
                pnl = float(pos.get("pnl_pct", 0))
                label = 1 if pnl > 0.5 else 0   # profitable = bon timing

                # Features synthétiques si pas d'historique technique
                feat = [50, 0, 0, 1 if pnl > 0 else 0, 0.05, 0.5,
                        1.0, 0.01, 0.01, -0.01, 50, 50]
                X.append(feat)
                y.append(label)

            X = np.array(X)
            y = np.array(y)

            ds    = lgb.Dataset(X, label=y, feature_name=self.FEATURES)
            params = {
                "objective":   "binary",
                "metric":      "binary_logloss",
                "n_estimators": 100,
                "max_depth":   4,
                "learning_rate": 0.05,
                "verbose":    -1,
            }
            self._model   = lgb.train(params, ds, num_boost_round=100)
            self._trained = True
            self._model.save_model(self.MODEL_FILE)
            logger.info("[EntryTiming] Modèle entraîné sur %d exemples", len(y))
            return {"trained": True, "samples": len(y), "positive_rate": round(float(np.mean(y)), 3)}
        except Exception as exc:
            logger.error("[EntryTiming] train error: %s", exc)
            return {"trained": False, "error": str(exc)}


# ── Instances globales ──
_portfolio_simulator = PortfolioSimulator()
_entry_timing_model  = EntryTimingModel()


# ══════════════════════════════════════════════════════════════
# GEM SWING AGENT — Agent spécialisé gem hunting + swing trading
# ══════════════════════════════════════════════════════════════

class GemSwingAgent:
    """
    Agent IA spécialisé gem hunting + swing trading altcoins.

    Stratégie :
    1. Scanner identifie les gems en sortie de consolidation
    2. Conditions macro validées (cycle, BTC.D, altseason)
    3. Structure technique confirmée (breakout confirmé)
    4. Risk manager valide la taille et les niveaux
    5. Plan DCA complet en 3 tranches

    Objectif : win rate > 55% avec R/R > 2.5
    """

    def analyze_gem_opportunity(self,
                                 token: dict,
                                 macro_data: dict,
                                 signal_data: dict,
                                 capital: float = 10_000) -> dict:
        from modules.risk_manager_v2 import GemSwingRiskManager

        risk_mgr = GemSwingRiskManager(capital=capital)
        score    = token.get("score", {})
        bo       = score.get("breakout_data", {})
        mc       = score.get("market_cap", 0)
        alpha    = signal_data.get("alpha", {})
        cycle    = macro_data.get("cycle", {})
        macro_sc = float(macro_data.get("score", 50) or 50)

        # ── Conditions macro ──────────────────────────────────
        macro_ok     = macro_sc >= 55
        cycle_phase  = str(cycle.get("phase", "") or "")
        cycle_ok     = any(p in cycle_phase for p in ["ACCUMULATION", "BULL"])
        btc_dom      = self._get_btc_dominance(macro_data)
        btc_dom_ok   = btc_dom < 58
        altseason_ok = self.compute_altseason_probability(btc_dom, macro_sc, cycle_phase) > 35

        macro_conditions = {
            "macro_score":   {"ok": macro_ok,     "value": macro_sc,
                              "detail": f"Score macro {macro_sc}/100"},
            "cycle_phase":   {"ok": cycle_ok,     "value": cycle_phase,
                              "detail": f"Phase {cycle_phase or '?'}"},
            "btc_dominance": {"ok": btc_dom_ok,   "value": btc_dom,
                              "detail": f"BTC.D {btc_dom:.1f}%"},
            "altseason":     {"ok": altseason_ok, "value": altseason_ok,
                              "detail": "Altseason favorable" if altseason_ok else "Trop tôt pour alts"},
        }
        macro_passed = sum(1 for v in macro_conditions.values() if v["ok"])
        macro_total  = len(macro_conditions)

        # ── Score technique ───────────────────────────────────
        gem_score   = score.get("total_score", 0)
        breakout_ok = bo.get("is_breaking_out", False)
        grade       = score.get("grade", "D")
        tech_ok     = gem_score >= 65 and breakout_ok

        # ── Risk management ───────────────────────────────────
        price      = float(token.get("price", 0) or 0)
        volatility = float(token.get("volatility", 5) or 5)
        atr_approx = price * volatility / 100

        sl_tp = risk_mgr.compute_gem_sl_tp(price, atr_approx, "LONG", volatility)
        risk_check = risk_mgr.can_open(
            symbol=token.get("symbol", "?"),
            direction="LONG",
            sl_pct=sl_tp["sl_pct"],
            strategy="GEM_SWING",
            macro_data=macro_data,
        )

        # ── Décision finale ───────────────────────────────────
        conditions_ok = (
            macro_passed >= 3 and
            tech_ok and
            risk_check["allowed"] and
            gem_score >= 65
        )

        if conditions_ok:
            verdict    = "GO"
            action     = (
                f"Entrer sur {token.get('symbol','?')} en 3 tranches. "
                f"Tranche 1 : ${risk_check['tranche_sizes']['1']:.0f} maintenant."
            )
            confidence = min(95, int(
                (macro_passed / macro_total * 30) +
                (gem_score * 0.50) +
                (risk_check["allowed"] * 20)
            ))
        elif macro_passed >= 2 and gem_score >= 55:
            verdict    = "WATCH"
            action     = (
                f"Surveiller {token.get('symbol','?')} — conditions partiellement réunies. "
                "Alerter si breakout confirmé."
            )
            confidence = 40
        else:
            verdict    = "PASS"
            action     = "Ne pas entrer — conditions insuffisantes"
            confidence = 0

        # ── Plan DCA ─────────────────────────────────────────
        plan = None
        if verdict == "GO":
            plan = {
                "total_allocation": risk_check["max_size_usd"],
                "tranches": [
                    {
                        "numero":  1,
                        "montant": risk_check["tranche_sizes"]["1"],
                        "quand":   "Maintenant (breakout confirmé)",
                        "prix":    f"${price:.4f} (prix actuel)",
                    },
                    {
                        "numero":  2,
                        "montant": risk_check["tranche_sizes"]["2"],
                        "quand":   "Sur pullback -3% à -5%",
                        "prix":    f"${price * 0.97:.4f}",
                    },
                    {
                        "numero":  3,
                        "montant": risk_check["tranche_sizes"]["3"],
                        "quand":   "Confirmation momentum (volume x1.5)",
                        "prix":    "Prix marché au moment",
                    },
                ],
                "stop_loss": {
                    "prix":         f"${sl_tp['sl_price']:.4f}",
                    "pct":          f"-{sl_tp['sl_pct']:.1f}%",
                    "perte_max_usd": round(
                        risk_check["max_size_usd"] * sl_tp["sl_pct"] / 100, 2
                    ),
                },
                "objectifs": {
                    "tp1": {
                        "prix":   f"${sl_tp['tp1']:.4f}",
                        "rr":     sl_tp["rr_tp1"],
                        "action": sl_tp["tp1_action"],
                    },
                    "tp2": {
                        "prix":   f"${sl_tp['tp2']:.4f}",
                        "rr":     sl_tp["rr_tp2"],
                        "action": sl_tp["tp2_action"],
                    },
                    "tp3": {
                        "prix":   f"${sl_tp['tp3']:.4f}",
                        "rr":     ">3.5",
                        "action": sl_tp["tp3_action"],
                    },
                },
                "capital_protege_pct": round(
                    100 - (risk_check["max_size_usd"] / capital * sl_tp["sl_pct"]), 1
                ) if capital > 0 else 100,
            }

        sym = token.get("symbol", "?")
        return {
            "symbol":           sym,
            "verdict":          verdict,
            "confidence":       confidence,
            "action":           action,
            "gem_score":        gem_score,
            "grade":            grade,
            "is_breaking_out":  breakout_ok,
            "breakout_signals": bo.get("breakout_signals", []),
            "macro_conditions": macro_conditions,
            "macro_passed":     f"{macro_passed}/{macro_total}",
            "risk_check":       risk_check,
            "plan_dca":         plan,
            "token_data": {
                "price":       price,
                "change_24h":  token.get("change_24h", 0),
                "volume_24h":  token.get("volume_24h", 0),
                "market_cap":  mc,
                "volatility":  volatility,
                "ath_distance": score.get("ath_distance_pct", 0),
            },
            "links": {
                "binance":     f"https://www.binance.com/fr/trade/{sym}USDT",
                "coingecko":   f"https://coingecko.com/en/coins/{token.get('cg_id', sym.lower())}",
                "tradingview": f"https://fr.tradingview.com/chart/?symbol=BINANCE:{sym}USDT",
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def _get_btc_dominance(self, macro_data: dict) -> float:
        try:
            import requests as _req
            r = _req.get("https://api.coingecko.com/api/v3/global", timeout=5)
            return float(r.json()["data"]["market_cap_percentage"]["btc"])
        except Exception:
            return 55.0

    def compute_altseason_probability(self, btc_dom: float,
                                       macro_score: float,
                                       cycle_phase: str) -> float:
        score = 0
        if btc_dom < 50:            score += 30
        elif btc_dom < 55:          score += 20
        elif btc_dom < 60:          score += 10
        if macro_score > 65:        score += 25
        elif macro_score > 55:      score += 15
        if "BULL" in cycle_phase:   score += 30
        elif "ACCUM" in cycle_phase: score += 20
        return min(100, score)
