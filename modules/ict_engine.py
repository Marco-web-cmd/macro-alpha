"""
ict_engine.py
Concepts ICT (Inner Circle Trader) pour l'analyse institutionnelle.

Deux classes :
  - ICTEngine          : concepts intraday (Kill Zones, PO3, FVG, OB)
  - ICTLongTermEngine  : concepts moyen/long terme (Dealing Range,
                         Quarterly Theory, Accumulation institutionnelle)

ICT part du principe que les institutions laissent des empreintes
lisibles dans les données de prix. Ce module les décode.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ICT INTRADAY ENGINE (stub — enrichi par les méthodes existantes)
# ═══════════════════════════════════════════════════════════════════

class ICTEngine:
    """
    Concepts ICT courts termes : Kill Zones, Power of Three,
    Fair Value Gaps, Order Blocks intraday.
    """

    def compute_fvg(self, df: pd.DataFrame) -> list:
        """Fair Value Gaps — imbalances de prix à combler."""
        fvgs = []
        for i in range(2, len(df)):
            prev2_high = float(df["high"].iloc[i - 2])
            curr_low   = float(df["low"].iloc[i])
            prev2_low  = float(df["low"].iloc[i - 2])
            curr_high  = float(df["high"].iloc[i])

            # FVG haussier : gap entre high[i-2] et low[i]
            if curr_low > prev2_high:
                fvgs.append({
                    "type":      "BULLISH",
                    "top":       round(curr_low, 2),
                    "bottom":    round(prev2_high, 2),
                    "midpoint":  round((curr_low + prev2_high) / 2, 2),
                    "bar_index": i,
                    "ts":        str(df.index[i]),
                })
            # FVG baissier : gap entre low[i-2] et high[i]
            elif curr_high < prev2_low:
                fvgs.append({
                    "type":      "BEARISH",
                    "top":       round(prev2_low, 2),
                    "bottom":    round(curr_high, 2),
                    "midpoint":  round((prev2_low + curr_high) / 2, 2),
                    "bar_index": i,
                    "ts":        str(df.index[i]),
                })
        return fvgs[-5:]   # 5 FVG les plus récents

    def compute_order_blocks(self, df: pd.DataFrame, lookback: int = 20) -> list:
        """
        Order Blocks — dernières bougies avant un mouvement fort.
        Une OB est la dernière bougie opposée avant un déplacement impulsif.
        """
        obs = []
        recent = df.iloc[-lookback:]
        for i in range(1, len(recent) - 1):
            curr   = recent.iloc[i]
            nxt    = recent.iloc[i + 1]
            # OB haussier : bougie baissière avant fort mouvement haussier
            if curr["close"] < curr["open"]:
                move = (nxt["close"] - nxt["open"]) / nxt["open"] * 100
                if move > 0.5:
                    obs.append({
                        "type":   "BULLISH_OB",
                        "top":    round(float(curr["open"]), 2),
                        "bottom": round(float(curr["low"]),  2),
                        "ts":     str(recent.index[i]),
                    })
            # OB baissier : bougie haussière avant fort mouvement baissier
            elif curr["close"] > curr["open"]:
                move = (nxt["close"] - nxt["open"]) / nxt["open"] * 100
                if move < -0.5:
                    obs.append({
                        "type":   "BEARISH_OB",
                        "top":    round(float(curr["high"]), 2),
                        "bottom": round(float(curr["open"]), 2),
                        "ts":     str(recent.index[i]),
                    })
        return obs[-3:]

    def compute_kill_zones(self) -> dict:
        """
        Kill Zones ICT — fenêtres horaires où les institutions sont actives.
        Londres : 02:00-05:00 UTC, New York : 07:00-10:00 UTC,
        London Close : 10:00-12:00 UTC, Asian : 20:00-00:00 UTC.
        """
        now_utc  = datetime.now(timezone.utc)
        hour_utc = now_utc.hour + now_utc.minute / 60

        zones = {
            "asian":     {"start": 20, "end": 24, "label": "Asian Session",     "active": False},
            "london":    {"start":  2, "end":  5, "label": "London Kill Zone",  "active": False},
            "ny":        {"start":  7, "end": 10, "label": "New York Kill Zone", "active": False},
            "london_close": {"start": 10, "end": 12, "label": "London Close",   "active": False},
        }
        for name, z in zones.items():
            if z["start"] <= hour_utc < z["end"]:
                z["active"] = True

        active = [v["label"] for v in zones.values() if v["active"]]
        return {
            "zones":  zones,
            "active": active,
            "is_kill_zone": len(active) > 0,
            "interpretation": (
                f"Kill Zone active : {', '.join(active)} — "
                "institutions potentiellement actives, signaux plus fiables."
                if active else
                "Hors Kill Zone — liquidité institutionnelle réduite."
            ),
        }

    def full_ict_analysis(self, df: pd.DataFrame, interval: str = "1h") -> dict:
        """Analyse ICT intraday complète."""
        try:
            fvgs     = self.compute_fvg(df)
            obs      = self.compute_order_blocks(df)
            kz       = self.compute_kill_zones()
            price    = float(df["close"].iloc[-1])

            # Liquidity Pools : highs/lows récents non encore chassés
            recent_highs = df["high"].iloc[-20:].nlargest(3).tolist()
            recent_lows  = df["low"].iloc[-20:].nsmallest(3).tolist()

            return {
                "interval":       interval,
                "fair_value_gaps": fvgs,
                "order_blocks":    obs,
                "kill_zones":      kz,
                "liquidity_pools": {
                    "buy_side":  [round(h, 2) for h in recent_highs],
                    "sell_side": [round(l, 2) for l in recent_lows],
                },
                "interpretation": (
                    f"{len(fvgs)} FVG actifs, {len(obs)} Order Blocks. "
                    + kz["interpretation"]
                ),
            }
        except Exception as exc:
            logger.warning(f"ICTEngine.full_ict_analysis: {exc}")
            return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# ICT LONG TERM ENGINE — Horizons Weekly / Monthly / Quarterly
# ═══════════════════════════════════════════════════════════════════

class ICTLongTermEngine:
    """
    Concepts ICT adaptés aux horizons Weekly, Monthly, Quarterly.
    Ces concepts décrivent comment les institutions construisent
    leurs positions sur des semaines et des mois.
    """

    # ── DEALING RANGE ───────────────────────────────────────────────

    def compute_dealing_range(self, df_weekly: pd.DataFrame,
                               lookback_weeks: int = 12) -> dict | None:
        """
        DEALING RANGE — Concept central ICT long terme.

        Le Dealing Range est la zone entre le dernier High et Low
        significatifs sur une période de 8-12 semaines.
        C'est dans cette zone que les institutions accumulent
        ou distribuent avant le prochain grand mouvement.

        Structure :
        - Top du range (>65%) : Premium Zone (institutions distribuent)
        - 50% du range        : Equilibrium  (zone de rééquilibrage)
        - Bottom (<35%)       : Discount Zone (institutions accumulent)
        """
        if df_weekly is None or len(df_weekly) < lookback_weeks:
            return None

        recent     = df_weekly.iloc[-lookback_weeks:]
        range_high = float(recent["high"].max())
        range_low  = float(recent["low"].min())
        equilibrium = (range_high + range_low) / 2
        current    = float(df_weekly["close"].iloc[-1])
        range_width = (range_high - range_low) / max(range_low, 1) * 100

        if range_high == range_low:
            return None

        position_pct = (current - range_low) / (range_high - range_low) * 100
        position_pct = max(0.0, min(100.0, position_pct))

        target_long  = range_low  + (range_high - range_low) * 0.9
        target_short = range_low  + (range_high - range_low) * 0.1

        if position_pct > 65:
            zone = "PREMIUM"
            bias = "DISTRIBUANT"
            expected = "vers le bas"
            interpretation = (
                f"Prix en PREMIUM ({position_pct:.0f}% du Dealing Range). "
                "Les institutions distribuent en zone haute. "
                f"Cible baissière institutionnelle : ${target_short:,.0f} "
                "(base du range). Éviter les nouveaux longs."
            )
        elif position_pct < 35:
            zone = "DISCOUNT"
            bias = "ACCUMULANT"
            expected = "vers le haut"
            interpretation = (
                f"Prix en DISCOUNT ({position_pct:.0f}% du Dealing Range). "
                "Les institutions accumulent en zone basse. "
                f"Cible haussière institutionnelle : ${target_long:,.0f} "
                "(sommet du range). Favorable aux DCA et longs."
            )
        else:
            zone = "EQUILIBRIUM"
            bias = "NEUTRE"
            expected = "indéfini"
            interpretation = (
                f"Prix à l'Equilibrium ({position_pct:.0f}% du Dealing Range). "
                "Zone de transition — attendre un biais directionnel clair. "
                "Les institutions se repositionnent."
            )

        return {
            "range_high":         round(range_high, 2),
            "range_low":          round(range_low, 2),
            "equilibrium":        round(equilibrium, 2),
            "position_pct":       round(position_pct, 1),
            "current_zone":       zone,
            "institutional_bias": bias,
            "expected_move":      expected,
            "range_width_pct":    round(range_width, 1),
            "weeks_analyzed":     lookback_weeks,
            "target_if_long":     round(target_long, 2),
            "target_if_short":    round(target_short, 2),
            "interpretation":     interpretation,
        }

    # ── QUARTERLY BIAS ──────────────────────────────────────────────

    def compute_quarterly_bias(self, df_daily: pd.DataFrame) -> dict:
        """
        QUARTERLY THEORY — ICT.

        ICT divise l'année en 4 trimestres :
        Q1 Jan-Mar → souvent accumulation
        Q2 Avr-Jun → souvent expansion (le plus haussier sur BTC)
        Q3 Jul-Sep → souvent consolidation/correction
        Q4 Oct-Déc → expansion finale ou capitulation

        Niveaux clés : Annual Open (1er Jan), Monthly Open, Weekly Open.
        """
        try:
            now = pd.Timestamp.now(tz="UTC")
        except Exception:
            now = pd.Timestamp.utcnow()

        current_month   = now.month
        current_quarter = (current_month - 1) // 3 + 1

        quarter_names = {
            1: "Q1 (Jan-Mar)", 2: "Q2 (Avr-Jun)",
            3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Déc)",
        }
        quarter_bias_crypto = {
            1: "Accumulation/Rebond — historiquement haussier en bull market",
            2: "Expansion — Q2 le plus haussier statistiquement sur BTC",
            3: "Correction/Consolidation — Q3 souvent baissier",
            4: "Expansion finale ou Capitulation — très volatil",
        }

        current_price = float(df_daily["close"].iloc[-1])

        # Annual Open (1er janvier de l'année en cours)
        year_data = df_daily[df_daily.index.year == now.year]
        annual_open = float(year_data["open"].iloc[0]) if not year_data.empty else None

        # Monthly Open
        month_data  = df_daily[
            (df_daily.index.month == now.month) &
            (df_daily.index.year  == now.year)
        ]
        monthly_open = float(month_data["open"].iloc[0]) if not month_data.empty else None

        # Weekly Open (7 derniers jours)
        week_data  = df_daily.iloc[-7:]
        weekly_open = float(week_data["open"].iloc[0]) if not week_data.empty else None

        result: dict = {
            "current_quarter": f"Q{current_quarter}",
            "quarter_name":    quarter_names[current_quarter],
            "quarter_bias":    quarter_bias_crypto[current_quarter],
            "monthly_open":    round(monthly_open, 2) if monthly_open else None,
            "weekly_open":     round(weekly_open,  2) if weekly_open  else None,
            "annual_open":     round(annual_open,  2) if annual_open  else None,
        }

        if annual_open:
            above = current_price > annual_open
            dist_pct = (current_price - annual_open) / annual_open * 100
            result["price_vs_annual_open"]      = "AU-DESSUS" if above else "EN DESSOUS"
            result["annual_open_distance_pct"]  = round(dist_pct, 1)
            result["annual_open_interpretation"] = (
                f"Prix {'au-dessus' if above else 'en dessous'} de l'Annual Open "
                f"(${annual_open:,.0f}). "
                + ("Signal haussier long terme — institutions en profit sur l'année."
                   if above else
                   "Signal baissier long terme — institutions en perte sur l'année. "
                   "L'Annual Open devient une résistance majeure.")
            )

        if monthly_open:
            above_m = current_price > monthly_open
            result["monthly_open_interpretation"] = (
                f"Prix {'au-dessus' if above_m else 'en dessous'} de l'ouverture "
                f"mensuelle (${monthly_open:,.0f}). "
                + ("Biais mensuel haussier." if above_m else "Biais mensuel baissier.")
            )

        return result

    # ── ACCUMULATION INSTITUTIONNELLE ───────────────────────────────

    def detect_institutional_accumulation(self, df_daily: pd.DataFrame,
                                           df_weekly: pd.DataFrame,
                                           macro_data: dict) -> dict:
        """
        Détecte si les institutions sont en phase d'accumulation
        sur un horizon de 4-12 semaines.

        Signaux :
        1. Volume asymétrique (plus élevé sur les jours haussiers)
        2. Higher Lows hebdomadaires (Wyckoff Spring)
        3. Compression de range (pré-markup)
        4. NLI Fed en expansion
        5. Prix sous le coût de production estimé
        6. Bottom probability élevée (indicateurs cycle)
        """
        evidence: list[str] = []
        score = 0
        current = float(df_daily["close"].iloc[-1])

        # Signal 1 — Volume asymétrique
        if len(df_daily) >= 30:
            last_30 = df_daily.iloc[-30:]
            up_days = last_30[last_30["close"] > last_30["open"]]
            dn_days = last_30[last_30["close"] < last_30["open"]]
            if not up_days.empty and not dn_days.empty:
                vol_ratio = (float(up_days["volume"].mean()) /
                             max(float(dn_days["volume"].mean()), 1))
                if vol_ratio > 1.3:
                    score += 25
                    evidence.append(
                        f"Volume haussier {vol_ratio:.1f}× supérieur au baissier "
                        "— accumulation institutionnelle silencieuse"
                    )

        # Signal 2 — Higher Lows hebdomadaires
        if df_weekly is not None and len(df_weekly) >= 4:
            recent_lows = df_weekly["low"].iloc[-4:]
            if recent_lows.is_monotonic_increasing:
                score += 20
                evidence.append(
                    "Higher Lows consécutifs sur 4 semaines — "
                    "structure Wyckoff Spring (accumulation)"
                )

        # Signal 3 — Compression de range
        if len(df_daily) >= 28:
            recent_range = (float(df_daily["high"].iloc[-14:].max()) -
                            float(df_daily["low"].iloc[-14:].min()))
            prev_range   = (float(df_daily["high"].iloc[-28:-14].max()) -
                            float(df_daily["low"].iloc[-28:-14].min()))
            if prev_range > 0 and recent_range / prev_range < 0.6:
                ratio_pct = recent_range / prev_range * 100
                score += 15
                evidence.append(
                    f"Range 14J réduit à {ratio_pct:.0f}% du range précédent "
                    "— compression pré-markup"
                )

        # Signal 4 — NLI Fed
        nli_change = (macro_data.get("liquidity", {}) or {}).get("nli_change_4w", 0) or 0
        if nli_change > 1.0:
            score += 15
            evidence.append(f"NLI Fed +{nli_change:.1f}%/4w — liquidité disponible")
        elif nli_change < -1.0:
            score -= 15
            evidence.append(f"NLI Fed {nli_change:.1f}%/4w — contraction liquidité")

        # Signal 5 — Sous coût de production
        MINING_COST = 58_000.0
        if current < MINING_COST:
            score += 20
            evidence.append(
                f"Prix ${current:,.0f} sous coût de production estimé "
                f"(~${MINING_COST:,.0f}) — zone d'accumulation historique"
            )

        # Signal 6 — Bottom probability
        bottom_prob = (
            (macro_data.get("cycle") or {}).get("bottom_probability", 0) or
            (macro_data.get("bottom_indicators") or {}).get("score", 0) or 0
        )
        if bottom_prob > 60:
            score += 15
            evidence.append(
                f"Bottom probability {bottom_prob:.0f}% — "
                "convergence d'indicateurs de fond de cycle"
            )

        score = max(0, min(100, score))

        if score >= 70:
            phase  = "ACCUMULATION"
            target = float(df_daily["high"].iloc[-min(100, len(df_daily)):].max())
        elif score <= 30:
            phase  = "DISTRIBUTION"
            target = float(df_daily["low"].iloc[-min(100, len(df_daily)):].min())
        elif current > float(df_daily["close"].iloc[-30:].mean()):
            phase  = "MARKUP"
            target = current * 1.15
        else:
            phase  = "MARKDOWN"
            target = current * 0.85

        return {
            "accumulation_probability": score,
            "phase":    phase,
            "evidence": evidence,
            "target_zone": round(target, 2),
            "interpretation": (
                f"Phase institutionnelle : {phase} (score {score}/100). "
                + ("Accumulation confirmée — markup probable dans 4-12 semaines."
                   if phase == "ACCUMULATION" else "")
                + ("Distribution en cours — markdown probable."
                   if phase == "DISTRIBUTION" else "")
                + f" Cible institutionnelle estimée : ${target:,.0f}."
            ),
        }

    # ── CIBLES LONG TERME ───────────────────────────────────────────

    def compute_long_term_targets(self, df_weekly: pd.DataFrame,
                                   df_monthly: pd.DataFrame,
                                   current_price: float) -> dict:
        """
        Cibles institutionnelles long terme.
        En ICT, les cibles sont toujours des zones de liquidité :
        Previous Week High/Low (PWH/PWL), Previous Month High/Low (PMH/PML).
        """
        targets: dict = {}

        if df_weekly is not None and len(df_weekly) >= 2:
            targets["pwh"] = {
                "price": round(float(df_weekly["high"].iloc[-2]), 2),
                "label": "Previous Week High",
                "type":  "BSL",
                "interpretation": "Cible court terme — institutions visent cette liquidité",
            }
            targets["pwl"] = {
                "price": round(float(df_weekly["low"].iloc[-2]), 2),
                "label": "Previous Week Low",
                "type":  "SSL",
                "interpretation": "Support majeur — sweep possible avant rebond",
            }

        if df_monthly is not None and len(df_monthly) >= 2:
            targets["pmh"] = {
                "price": round(float(df_monthly["high"].iloc[-2]), 2),
                "label": "Previous Month High",
                "type":  "BSL_major",
                "interpretation": "Résistance mensuelle majeure — zone de distribution",
            }
            targets["pml"] = {
                "price": round(float(df_monthly["low"].iloc[-2]), 2),
                "label": "Previous Month Low",
                "type":  "SSL_major",
                "interpretation": "Support mensuel majeur — zone d'accumulation",
            }

        for key, target in targets.items():
            dist = (target["price"] - current_price) / max(current_price, 1) * 100
            target["distance_pct"] = round(dist, 1)
            target["direction"]    = "au-dessus" if dist > 0 else "en dessous"

        return targets

    # ── ANALYSE COMPLÈTE ────────────────────────────────────────────

    def full_long_term_ict_analysis(self, df_daily: pd.DataFrame,
                                     df_weekly: pd.DataFrame,
                                     df_monthly: pd.DataFrame,
                                     macro_data: dict,
                                     current_price: float) -> dict:
        """
        Analyse ICT complète pour horizons 7J/30J/90J.
        Connectée aux forecasts existants pour validation croisée.
        """
        try:
            dealing_range = self.compute_dealing_range(df_weekly, lookback_weeks=12)
        except Exception as exc:
            logger.warning(f"dealing_range: {exc}")
            dealing_range = None

        try:
            quarterly = self.compute_quarterly_bias(df_daily)
        except Exception as exc:
            logger.warning(f"quarterly: {exc}")
            quarterly = {}

        try:
            accumulation = self.detect_institutional_accumulation(
                df_daily, df_weekly, macro_data
            )
        except Exception as exc:
            logger.warning(f"accumulation: {exc}")
            accumulation = {
                "accumulation_probability": 50, "phase": "NEUTRE",
                "evidence": [], "target_zone": current_price,
                "interpretation": str(exc),
            }

        try:
            lt_targets = self.compute_long_term_targets(
                df_weekly, df_monthly, current_price
            )
        except Exception as exc:
            logger.warning(f"lt_targets: {exc}")
            lt_targets = {}

        # ── Score ICT long terme global ──
        ict_lt_score = 50.0
        if dealing_range:
            if dealing_range["current_zone"] == "DISCOUNT":
                ict_lt_score += 20
            elif dealing_range["current_zone"] == "PREMIUM":
                ict_lt_score -= 20

        ict_lt_score += (accumulation["accumulation_probability"] - 50) * 0.3
        ict_lt_score = round(max(0.0, min(100.0, ict_lt_score)), 1)

        narrative = self._generate_lt_narrative(
            dealing_range, quarterly, accumulation, lt_targets, current_price
        )

        return {
            "ict_lt_score":  ict_lt_score,
            "dealing_range": dealing_range,
            "quarterly_bias": quarterly,
            "accumulation":  accumulation,
            "lt_targets":    lt_targets,
            "narrative":     narrative,
            "horizon_signals": {
                "7j":  self._signal_for_horizon(dealing_range, accumulation, 7),
                "30j": self._signal_for_horizon(dealing_range, accumulation, 30),
                "90j": self._signal_for_horizon(dealing_range, accumulation, 90),
            },
        }

    # ── HELPERS ─────────────────────────────────────────────────────

    def _signal_for_horizon(self, dealing_range, accumulation, days: int) -> dict:
        """Signal directionnel pour un horizon donné en jours."""
        score = 50.0

        if dealing_range:
            if dealing_range["current_zone"] == "DISCOUNT":
                score += 15 if days <= 30 else 25
            elif dealing_range["current_zone"] == "PREMIUM":
                score -= 15 if days <= 30 else 25

        score += (accumulation["accumulation_probability"] - 50) * 0.2

        signal = "LONG" if score > 60 else ("SHORT" if score < 40 else "NEUTRE")
        return {
            "signal":  signal,
            "score":   round(score, 1),
            "horizon": f"{days}J",
            "basis":   "ICT Dealing Range + Institutional Accumulation",
        }

    def _generate_lt_narrative(self, dealing_range, quarterly,
                                accumulation, targets, price: float) -> str:
        """Narratif long terme en langage naturel."""
        parts: list[str] = []

        if dealing_range:
            parts.append(
                f"Dealing Range ({dealing_range['weeks_analyzed']}sem) : "
                f"prix en {dealing_range['current_zone']} "
                f"({dealing_range['position_pct']:.0f}% du range). "
                f"{dealing_range['interpretation']}"
            )

        if quarterly:
            parts.append(
                f"{quarterly.get('quarter_name', '')} — {quarterly.get('quarter_bias', '')}. "
                + quarterly.get("annual_open_interpretation", "")
            )

        parts.append(
            f"Phase institutionnelle : {accumulation['phase']} "
            f"({accumulation['accumulation_probability']:.0f}% probabilité). "
            f"Cible : ${accumulation['target_zone']:,.0f}."
        )

        if targets.get("pmh"):
            parts.append(
                f"Previous Month High ${targets['pmh']['price']:,.0f} "
                f"({targets['pmh']['distance_pct']:+.1f}%) — "
                "cible institutionnelle si accumulation confirmée."
            )

        return " | ".join(parts)
