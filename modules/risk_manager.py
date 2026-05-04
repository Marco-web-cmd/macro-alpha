"""
risk_manager.py — Gestionnaire de risque dynamique
====================================================
Kelly Criterion modifié + stops intelligents + TP en 3 tranches.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DynamicRiskManager:
    """
    La gestion du risque est plus importante que les signaux.
    Un bon signal mal géré = perte. Un signal moyen bien géré = profit.
    """

    # Paramètres par défaut (ajustables via agent_memory)
    DEFAULT_WIN_RATE   = 0.55
    DEFAULT_AVG_WIN    = 0.025   # 2.5%
    DEFAULT_AVG_LOSS   = 0.015   # 1.5%
    MAX_POSITION_PCT   = 0.10    # 10% max du capital
    MAX_RISK_PER_TRADE = 0.02    # 2% max du capital à risquer

    def compute_position_size(self, capital: float, risk_pct: float,
                               entry: float, stop_loss: float,
                               atr: float, regime: str,
                               win_rate: float = None,
                               setup_quality: str = "B",
                               atr_avg: float = None) -> dict:
        """
        Kelly Criterion modifié (quart-Kelly) + ajustements de régime.
        """
        wr   = win_rate or self.DEFAULT_WIN_RATE
        avg_w = self.DEFAULT_AVG_WIN
        avg_l = self.DEFAULT_AVG_LOSS

        # Quart-Kelly
        if avg_w > 0:
            kelly_f = (wr * avg_w - (1 - wr) * avg_l) / avg_w
            kelly_adj = max(0.0, kelly_f * 0.25)
        else:
            kelly_adj = 0.01

        # Taille basée sur le risque par trade (méthode % du capital)
        risk_dist = abs(entry - stop_loss)
        if risk_dist <= 0 or entry <= 0:
            return {"error": "Stop loss invalide", "position_size_pct": 0}

        risk_per_share = risk_dist / entry   # en % du prix
        base_size_pct  = min(self.MAX_RISK_PER_TRADE / risk_per_share,
                             self.MAX_POSITION_PCT)

        # Fusion Kelly + risk-based
        size_pct = (base_size_pct * 0.7 + kelly_adj * 0.3)

        # Ajustements de régime
        regime_multiplier = {
            "bull_trend":      1.00,
            "accumulation":    1.00,
            "compression":     0.85,
            "distribution":    0.70,
            "high_volatility": 0.60,
            "bear_trend":      0.70,
        }.get(regime, 0.80)
        size_pct *= regime_multiplier

        # Ajustement qualité setup
        quality_mult = {"A+": 1.20, "A": 1.10, "B": 1.00, "C": 0.85, "D": 0.60}
        size_pct *= quality_mult.get(setup_quality, 1.0)

        # ATR élevé → réduire
        if atr_avg and atr > 0 and atr > atr_avg * 2:
            size_pct *= 0.70

        size_pct = min(self.MAX_POSITION_PCT, max(0.005, size_pct))

        position_usd = capital * size_pct
        max_loss_usd = position_usd * risk_per_share
        target_pct   = avg_w / avg_l if avg_l > 0 else 2.0
        reward_dist  = risk_dist * target_pct

        reasoning_parts = [
            f"Kelly ajusté {kelly_adj*100:.1f}%",
            f"Régime {regime} ×{regime_multiplier}",
            f"Setup {setup_quality}",
        ]
        if atr_avg and atr > atr_avg * 2:
            reasoning_parts.append(f"ATR élevé ×0.70")

        return {
            "position_size_pct": round(size_pct * 100, 2),
            "position_size_usd": round(position_usd, 2),
            "max_loss_usd":      round(max_loss_usd, 2),
            "risk_per_trade_pct": round(risk_per_share * size_pct * 100, 2),
            "risk_reward_ratio": round(target_pct, 2),
            "kelly_raw":         round(kelly_adj * 100, 2),
            "reasoning":         " | ".join(reasoning_parts),
        }

    def compute_stop_loss(self, entry: float, direction: str, atr: float,
                           key_levels: dict = None, pattern: str = None,
                           regime: str = "neutral") -> dict:
        """
        Stop loss intelligent basé sur la structure marché.
        Prend le plus conservateur parmi : structure, ATR, pattern.
        """
        pp  = (key_levels or {}).get("pivot_points", {}) or {}
        kl_support    = float(pp.get("s1", 0) or 0)
        kl_resistance = float(pp.get("r1", 0) or 0)

        stops = {}

        # ── ATR Stop ──
        atr_mult = 1.5 if regime != "high_volatility" else 2.0
        if direction == "long":
            stops["atr"] = entry - atr * atr_mult
        else:
            stops["atr"] = entry + atr * atr_mult

        # ── Structure Stop (S1 / R1) ──
        if direction == "long" and kl_support > 0:
            # Légèrement en dessous du support pour éviter les faux cassages
            stops["structure"] = kl_support * 0.995
        elif direction == "short" and kl_resistance > 0:
            stops["structure"] = kl_resistance * 1.005

        # ── Pattern Stop (invalidation) ──
        pattern_stops = {
            "falling_wedge":       entry * 0.975,
            "bull_flag":           entry * 0.972,
            "accumulation":        entry * 0.970,
            "double_bottom":       entry * 0.965,
            "inverse_head_shoulders": entry * 0.968,
        }
        if pattern and pattern.lower() in pattern_stops and direction == "long":
            stops["pattern"] = pattern_stops[pattern.lower()]

        # ── Choisir le stop le plus proche (plus conservateur) ──
        if not stops:
            chosen_stop = entry * (0.97 if direction == "long" else 1.03)
            stop_type   = "default"
        else:
            if direction == "long":
                chosen_stop = max(stops.values())  # le plus haut = plus proche
                stop_type   = max(stops, key=stops.get)
            else:
                chosen_stop = min(stops.values())  # le plus bas = plus proche
                stop_type   = min(stops, key=stops.get)

        # Contraintes
        max_stop_dist = entry * 0.05  # max 5%
        min_stop_dist = atr * 1.2

        if direction == "long":
            chosen_stop = max(chosen_stop, entry - max_stop_dist)
            chosen_stop = min(chosen_stop, entry - min_stop_dist)
            # Éviter les niveaux ronds (chassés par les algos)
            chosen_stop = self._avoid_round_level(chosen_stop, -1)
        else:
            chosen_stop = min(chosen_stop, entry + max_stop_dist)
            chosen_stop = max(chosen_stop, entry + min_stop_dist)
            chosen_stop = self._avoid_round_level(chosen_stop, +1)

        dist_pct = abs(entry - chosen_stop) / entry * 100

        return {
            "stop_loss":    round(chosen_stop, 2),
            "stop_type":    stop_type,
            "distance_pct": round(dist_pct, 2),
            "all_stops":    {k: round(v, 2) for k, v in stops.items()},
        }

    def _avoid_round_level(self, price: float, direction: int) -> float:
        """Décale légèrement le stop des niveaux ronds (×1000 ou ×500)."""
        magnitude = 10 ** (len(str(int(price))) - 3)
        remainder = price % magnitude
        if remainder < magnitude * 0.05 or remainder > magnitude * 0.95:
            # Niveau rond détecté → décaler
            price += direction * magnitude * 0.07
        return price

    def compute_take_profits(self, entry: float, stop_loss: float,
                              direction: str, key_levels: dict = None,
                              fibonacci: dict = None,
                              pattern_target: float = None) -> dict:
        """Stratégie de sortie en 3 tranches : 1:1, 2:1, 3.8:1."""
        risk_dist = abs(entry - stop_loss)
        if risk_dist <= 0:
            return {"error": "Stop invalide"}

        pp  = (key_levels or {}).get("pivot_points", {}) or {}
        fib_ext = (fibonacci or {}).get("extensions", {}) or {}

        r1  = float(pp.get("r1", 0) or 0) if direction == "long" else float(pp.get("s1", 0) or 0)
        fib_1618 = float(fib_ext.get("1.618", 0) or 0)

        if direction == "long":
            tp1_base = entry + risk_dist * 1.0
            tp2_base = entry + risk_dist * 2.1
            tp3_base = entry + risk_dist * 3.8

            # Ajuster TP2 sur R1 si disponible et proche
            if r1 > entry and abs(r1 - tp2_base) / tp2_base < 0.03:
                tp2_base = r1

            # Ajuster TP3 sur pattern target ou Fib 1.618
            if pattern_target and pattern_target > entry:
                tp3_base = pattern_target
            elif fib_1618 > entry:
                tp3_base = fib_1618
        else:
            tp1_base = entry - risk_dist * 1.0
            tp2_base = entry - risk_dist * 2.1
            tp3_base = entry - risk_dist * 3.8
            if r1 < entry and r1 > 0 and abs(r1 - tp2_base) / tp2_base < 0.03:
                tp2_base = r1
            if pattern_target and pattern_target < entry:
                tp3_base = pattern_target

        rr1 = abs(tp1_base - entry) / risk_dist
        rr2 = abs(tp2_base - entry) / risk_dist
        rr3 = abs(tp3_base - entry) / risk_dist

        return {
            "tp1": {"price": round(tp1_base, 2), "rr": round(rr1, 2), "allocation": 0.33},
            "tp2": {"price": round(tp2_base, 2), "rr": round(rr2, 2), "allocation": 0.33},
            "tp3": {"price": round(tp3_base, 2), "rr": round(rr3, 2), "allocation": 0.34},
            "breakeven_after":        "TP1",
            "trailing_after":         "TP2",
            "trailing_distance_atr":  1.0,
            "avg_rr":                 round((rr1 * 0.33 + rr2 * 0.33 + rr3 * 0.34), 2),
        }

    def portfolio_heat(self, open_positions: list) -> dict:
        """Calcule le risque total du portefeuille (heat)."""
        total_heat = sum(float(p.get("risk_pct", 0) or 0) for p in open_positions)
        n_pos = len(open_positions)
        if total_heat > 10:
            action = "REDUCE_POSITIONS"
            message = f"Portfolio heat {total_heat:.1f}% > 10% — réduire les positions"
        elif total_heat > 6:
            action = "NO_NEW_TRADES"
            message = f"Portfolio heat {total_heat:.1f}% > 6% — pas de nouveau trade"
        else:
            action = "OK"
            message = f"Portfolio heat {total_heat:.1f}% — acceptable"
        return {
            "total_heat_pct": round(total_heat, 2),
            "n_positions":    n_pos,
            "action":         action,
            "message":        message,
        }

    def assess_market_risk(self, vix: float, btc_change_24h: float,
                            macro_score: float,
                            correlation_spx: float = 0.5) -> dict:
        """Score de risque macro global (0-100)."""
        risk_score = 0
        factors = []

        if vix > 35:
            risk_score += 35
            factors.append(f"VIX {vix:.0f} (marché en panique)")
        elif vix > 25:
            risk_score += 20
            factors.append(f"VIX {vix:.0f} (stress élevé)")
        elif vix > 20:
            risk_score += 10
            factors.append(f"VIX {vix:.0f} (vigilance)")

        if btc_change_24h < -10:
            risk_score += 30
            factors.append(f"BTC {btc_change_24h:+.1f}% 24H (dump)")
        elif btc_change_24h < -5:
            risk_score += 15
            factors.append(f"BTC {btc_change_24h:+.1f}% 24H")

        if macro_score < 35:
            risk_score += 20
            factors.append(f"Macro défavorable ({macro_score:.0f}/100)")
        elif macro_score < 45:
            risk_score += 10

        if correlation_spx > 0.8:
            risk_score += 15
            factors.append(f"Corrélation SPX {correlation_spx:.2f} (risque contagion)")

        risk_score = min(100, risk_score)

        if risk_score >= 70:
            mode = "DEFENSIVE"
            message = "Mode défensif — aucune nouvelle position"
        elif risk_score >= 45:
            mode = "CAUTIOUS"
            message = "Mode prudent — réduire la taille des positions de 50%"
        else:
            mode = "NORMAL"
            message = "Risque acceptable — tailles normales"

        return {
            "risk_score": risk_score,
            "mode":       mode,
            "message":    message,
            "factors":    factors,
        }


# ═══════════════════════════════════════════════════════════
# PRODUCTION RISK MANAGER — Règles de trading fermes
# ═══════════════════════════════════════════════════════════

class ProductionRiskManager:
    """
    Gestionnaire de risque production : règles discrètes et non négociables.
    Règles :
      - Max 1% du capital risqué par trade
      - Max 5% de perte journalière (circuit-breaker)
      - Max 3 positions ouvertes simultanément
    """

    def __init__(self, max_risk_pct: float = 1.0,
                 max_daily_loss_pct: float = 5.0,
                 max_positions: int = 3):
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from config import MAX_RISK, MAX_DAILY_LOSS, MAX_POSITIONS
            self.max_risk_pct       = MAX_RISK
            self.max_daily_loss_pct = MAX_DAILY_LOSS
            self.max_positions      = MAX_POSITIONS
        except Exception:
            self.max_risk_pct       = max_risk_pct
            self.max_daily_loss_pct = max_daily_loss_pct
            self.max_positions      = max_positions

        self._daily_pnl:    float = 0.0   # PnL cumulé du jour (en %)
        self._daily_reset:  str   = ""    # date de dernier reset (YYYY-MM-DD)
        self._open_count:   int   = 0     # positions ouvertes actuelles

    def _check_daily_reset(self):
        """Remet à zéro le PnL journalier si changement de jour."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset:
            self._daily_pnl   = 0.0
            self._daily_reset = today

    def can_open_trade(self, signal: str, alpha_score: float,
                       open_positions: int = 0) -> dict:
        """
        Vérifie si un nouveau trade peut être ouvert.
        Retourne {"allowed": bool, "reason": str, "adjusted_size_pct": float}.
        """
        self._check_daily_reset()
        self._open_count = open_positions

        # Circuit-breaker journalier
        if self._daily_pnl <= -self.max_daily_loss_pct:
            return {
                "allowed":           False,
                "reason":            f"Circuit-breaker : perte journalière {self._daily_pnl:.1f}% ≥ {self.max_daily_loss_pct}%",
                "adjusted_size_pct": 0.0,
            }

        # Limite de positions
        if self._open_count >= self.max_positions:
            return {
                "allowed":           False,
                "reason":            f"Maximum {self.max_positions} positions simultanées atteint",
                "adjusted_size_pct": 0.0,
            }

        # Signal neutre → pas de trade
        if signal in ("NEUTRE",):
            return {
                "allowed":           False,
                "reason":            "Signal NEUTRE — pas d'entrée",
                "adjusted_size_pct": 0.0,
            }

        # Taille de base (max_risk_pct)
        size = self.max_risk_pct

        # Réduction si conviction faible
        if alpha_score < 50:
            size *= 0.5
        elif alpha_score < 65:
            size *= 0.75

        # Réduction si proche du daily loss
        remaining_loss = self.max_daily_loss_pct + self._daily_pnl
        if remaining_loss < self.max_daily_loss_pct * 0.3:
            size *= 0.5

        return {
            "allowed":           True,
            "reason":            "OK",
            "adjusted_size_pct": round(size, 2),
            "max_risk_pct":      self.max_risk_pct,
            "daily_pnl":         round(self._daily_pnl, 2),
            "remaining_budget":  round(remaining_loss, 2),
        }

    def record_trade_result(self, pnl_pct: float):
        """Enregistre le résultat d'un trade dans le PnL journalier."""
        self._check_daily_reset()
        self._daily_pnl += pnl_pct
        logger.info("[ProductionRisk] PnL jour: %.2f%% (limite: %.1f%%)",
                    self._daily_pnl, self.max_daily_loss_pct)

    def get_status(self) -> dict:
        """Retourne l'état courant du gestionnaire."""
        self._check_daily_reset()
        circuit_open = self._daily_pnl <= -self.max_daily_loss_pct
        return {
            "circuit_breaker_open": circuit_open,
            "daily_pnl_pct":        round(self._daily_pnl, 2),
            "max_daily_loss_pct":   self.max_daily_loss_pct,
            "max_risk_per_trade":   self.max_risk_pct,
            "open_positions":       self._open_count,
            "max_positions":        self.max_positions,
            "mode":                 "BLOCKED" if circuit_open else "ACTIVE",
        }
