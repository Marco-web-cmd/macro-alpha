"""
risk_manager_v2.py — Risk Manager strict pour gem hunting + swing trading.
"""
import json
import os
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger("risk_manager_v2")


@dataclass
class Position:
    id:            str
    symbol:        str
    entry_price:   float
    current_price: float
    size_usd:      float
    sl_price:      float
    tp1_price:     float
    tp2_price:     float
    direction:     str      # LONG / SHORT
    opened_at:     str
    status:        str  = "OPEN"
    pnl_usd:       float = 0.0
    pnl_pct:       float = 0.0
    tranche:       int   = 1
    strategy:      str   = "GEM_SWING"


class GemSwingRiskManager:
    """
    Risk manager conçu pour :
    1. Gem hunting (altcoins faible MC en breakout)
    2. Swing trading 4H-Daily

    Règles absolues basées sur les statistiques réelles
    des altcoins à faible market cap.
    """

    POSITIONS_FILE = "data/positions.json"
    STATS_FILE     = "data/daily_stats.json"

    RULES = {
        "max_risk_per_trade_pct":  1.0,
        "max_portfolio_risk_pct":  6.0,
        "max_positions":           6,
        "max_single_asset_pct":    10.0,
        "hard_stop_pct":           12.0,
        "gem_max_stop_pct":        15.0,
        "daily_loss_limit_pct":    4.0,
        "weekly_loss_limit_pct":   8.0,
        "pause_if_btc_drop_24h":  -8.0,
        "reduce_if_vix_above":     25,
        "stop_if_vix_above":       35,
        "gem_max_allocation_pct":  3.0,
        "gem_dca_tranches":        3,
        "gem_tranche_sizes":       [0.30, 0.40, 0.30],
        "move_sl_breakeven_at_tp1": True,
        "trailing_stop_after_tp2":  True,
        "trailing_atr_multiplier":  1.5,
    }

    def __init__(self, capital: float = 10_000.0):
        self.capital = capital
        os.makedirs("data", exist_ok=True)

    def can_open(self, symbol: str, direction: str,
                  sl_pct: float, strategy: str,
                  macro_data: dict = None,
                  tech_data: dict = None) -> dict:
        """
        Vérifie toutes les règles avant d'ouvrir une position.
        Retourne {allowed, reason, max_size_usd, warnings}
        """
        blocked   = []
        warnings  = []
        positions = self._load_positions()
        open_pos  = [p for p in positions if p.get("status") == "OPEN"]
        daily     = self._get_daily_stats()

        # Règle 1 : Nombre de positions
        if len(open_pos) >= self.RULES["max_positions"]:
            blocked.append(
                f"Max positions atteint ({len(open_pos)}/{self.RULES['max_positions']})"
            )

        # Règle 2 : Déjà une position sur ce symbole
        existing = [p for p in open_pos if p.get("symbol") == symbol]
        if existing and strategy != "GEM_DCA":
            blocked.append(f"Position déjà ouverte sur {symbol}")

        # Règle 3 : Perte journalière
        daily_loss = daily.get("loss_pct", 0.0)
        if daily_loss <= -self.RULES["daily_loss_limit_pct"]:
            blocked.append(
                f"Perte journalière atteinte ({daily_loss:.1f}%) — pause obligatoire"
            )
        elif daily_loss <= -self.RULES["daily_loss_limit_pct"] * 0.7:
            warnings.append(
                f"Perte journalière à {daily_loss:.1f}% — proche du seuil de pause"
            )

        # Règle 4 : Circuit breaker BTC
        btc_chg = (tech_data or {}).get("btc_change_24h", 0) or 0
        if btc_chg < self.RULES["pause_if_btc_drop_24h"]:
            blocked.append(
                f"BTC en chute ({btc_chg:.1f}%) — attendre stabilisation"
            )

        # Règle 5 : VIX
        vix = (macro_data or {}).get("collateral", {}).get("vix", 20) or 20
        if vix > self.RULES["stop_if_vix_above"]:
            blocked.append(f"VIX critique ({vix:.0f}) — fermer toutes les positions")
        elif vix > self.RULES["reduce_if_vix_above"]:
            warnings.append(f"VIX élevé ({vix:.0f}) — taille réduite de 50%")

        # Règle 6 : Cycle de marché
        cycle = (macro_data or {}).get("cycle", {}).get("phase", "")
        if "BEAR" in cycle:
            blocked.append(f"Phase bear market ({cycle}) — pas de nouveaux longs")

        # Calcul taille max si non bloqué
        if not blocked:
            sl_decimal = abs(sl_pct) / 100
            max_pct = self.RULES["max_risk_per_trade_pct"] / max(sl_decimal, 0.01)

            if strategy == "GEM_SWING":
                max_pct = min(max_pct, self.RULES["gem_max_allocation_pct"])
            else:
                max_pct = min(max_pct, 5.0)

            if vix > self.RULES["reduce_if_vix_above"]:
                max_pct *= 0.5

            current_exposure = sum(
                p.get("size_usd", 0) / self.capital * 100
                for p in open_pos
            )
            remaining = max(0, self.RULES["max_portfolio_risk_pct"] - current_exposure)
            max_pct = min(max_pct, remaining)

            max_size_usd = round(self.capital * max_pct / 100, 2)
        else:
            max_pct      = 0
            max_size_usd = 0

        logger.info("risk_check symbol=%s allowed=%s max_size=%.2f",
                    symbol, len(blocked) == 0, max_size_usd)

        return {
            "allowed":      len(blocked) == 0,
            "blocked":      blocked,
            "warnings":     warnings,
            "max_pct":      round(max_pct, 2),
            "max_size_usd": max_size_usd,
            "tranche_sizes": {
                "1": round(max_size_usd * 0.30, 2),
                "2": round(max_size_usd * 0.40, 2),
                "3": round(max_size_usd * 0.30, 2),
            },
        }

    def compute_gem_sl_tp(self, entry_price: float,
                           atr: float, direction: str,
                           token_volatility: float = 5.0) -> dict:
        """SL/TP adaptés aux gems altcoin (3 niveaux de TP)."""
        if token_volatility > 15:   atr_mult = 2.5
        elif token_volatility > 10: atr_mult = 2.2
        elif token_volatility > 5:  atr_mult = 2.0
        else:                       atr_mult = 1.8

        if direction == "LONG":
            sl  = max(
                entry_price * (1 - self.RULES["gem_max_stop_pct"] / 100),
                entry_price - atr * atr_mult,
            )
            tp1 = entry_price + atr * 2.0
            tp2 = entry_price + atr * 4.0
            tp3 = entry_price + atr * 7.0
        else:
            sl  = min(
                entry_price * (1 + self.RULES["gem_max_stop_pct"] / 100),
                entry_price + atr * atr_mult,
            )
            tp1 = entry_price - atr * 2.0
            tp2 = entry_price - atr * 4.0
            tp3 = entry_price - atr * 7.0

        sl_pct = abs(entry_price - sl) / entry_price * 100
        rr1    = abs(tp1 - entry_price) / max(abs(sl - entry_price), 0.0001)
        rr2    = abs(tp2 - entry_price) / max(abs(sl - entry_price), 0.0001)

        return {
            "sl_price":   round(sl,  6),
            "sl_pct":     round(sl_pct, 2),
            "atr_mult":   atr_mult,
            "tp1":        round(tp1, 6),
            "tp2":        round(tp2, 6),
            "tp3":        round(tp3, 6),
            "rr_tp1":     round(rr1, 2),
            "rr_tp2":     round(rr2, 2),
            "tp1_action": "Sortir 33% + déplacer SL au breakeven",
            "tp2_action": "Sortir 33% + trailing stop 1.5×ATR",
            "tp3_action": "Laisser courir — trailing stop suit",
        }

    def _load_positions(self) -> list:
        if os.path.exists(self.POSITIONS_FILE):
            try:
                with open(self.POSITIONS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _get_daily_stats(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(self.STATS_FILE):
            try:
                with open(self.STATS_FILE) as f:
                    stats = json.load(f)
                return stats.get(today, {"loss_pct": 0.0, "trades": 0})
            except Exception:
                pass
        return {"loss_pct": 0.0, "trades": 0}
