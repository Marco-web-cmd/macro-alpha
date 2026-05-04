"""
alerting.py — Système d'alertes Telegram pour macro_alpha
Envoie des notifications push sur les signaux forts et anomalies.
"""
import logging
import time
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("alerting")


class AlertManager:
    """
    Gestionnaire d'alertes Telegram.
    Nécessite TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID dans .env
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

    # Throttle : max 1 alerte par type par intervalle (secondes)
    THROTTLE_SECONDS = {
        "signal":    300,   # 5 min entre deux alertes signal
        "anomaly":   600,   # 10 min entre deux alertes anomalie
        "scanner":   1800,  # 30 min entre deux alertes scanner
        "risk":      120,   # 2 min entre deux alertes risque
        "default":   60,
    }

    def __init__(self):
        try:
            import sys, os
            from dotenv import load_dotenv
            _env = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
            load_dotenv(_env, override=True)
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            self._token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            self._chat  = os.getenv("TELEGRAM_CHAT_ID", "")
        except Exception as exc:
            logger.warning("[Alerting] config non disponible: %s", exc)
            self._token = ""
            self._chat  = ""

        self._last_sent: dict = {}   # kind → timestamp dernier envoi
        self._sent_count: int = 0
        self._enabled = bool(self._token and self._chat)

        if self._enabled:
            logger.info("[Alerting] Telegram configuré (chat=%s)", self._chat)
        else:
            logger.info("[Alerting] Telegram non configuré — alertes désactivées")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _throttled(self, kind: str) -> bool:
        """Retourne True si l'alerte doit être throttlée."""
        now     = time.time()
        delay   = self.THROTTLE_SECONDS.get(kind, self.THROTTLE_SECONDS["default"])
        last    = self._last_sent.get(kind, 0)
        return (now - last) < delay

    def _send(self, message: str, kind: str = "default",
              parse_mode: str = "Markdown") -> bool:
        """Envoie un message Telegram. Retourne True si succès."""
        if not self._enabled:
            logger.debug("[Alerting] Telegram non configuré — message ignoré: %s", message[:50])
            return False

        if self._throttled(kind):
            logger.debug("[Alerting] Throttle actif pour '%s'", kind)
            return False

        try:
            url  = self.TELEGRAM_API.format(token=self._token)
            data = {
                "chat_id":    self._chat,
                "text":       message,
                "parse_mode": parse_mode,
            }
            r = requests.post(url, data=data, timeout=10)
            r.raise_for_status()
            self._last_sent[kind] = time.time()
            self._sent_count      += 1
            logger.info("[Alerting] Message envoyé (kind=%s, total=%d)", kind, self._sent_count)
            return True
        except Exception as exc:
            logger.warning("[Alerting] Erreur envoi Telegram: %s", exc)
            return False

    # ── Méthodes publiques ─────────────────────────────────

    def alert_signal(self, signal: str, alpha_score: float, price: float,
                     resume: str = "", interval: str = "1h") -> bool:
        """Alerte sur un signal fort (LONG FORT ou SHORT FORT)."""
        if signal not in ("LONG FORT", "SHORT FORT"):
            return False

        emoji  = "🟢" if "LONG" in signal else "🔴"
        grade  = "A+" if alpha_score >= 80 else "A" if alpha_score >= 70 else "B"
        msg    = (
            f"{emoji} *{signal}* [{grade}] — `{interval.upper()}`\n"
            f"💰 Prix : `${price:,.0f}`\n"
            f"📊 Score : `{alpha_score:.0f}/100`\n"
        )
        if resume:
            msg += f"📝 _{resume}_\n"
        msg += f"\n_macro\\_alpha • {datetime.now(timezone.utc).strftime('%H:%M UTC')}_"

        return self._send(msg, kind="signal")

    def alert_scanner(self, top_tokens: list) -> bool:
        """Alerte sur les meilleurs tokens du scan microcap."""
        if not top_tokens:
            return False

        lines = ["🔍 *Scanner Microcap — Top Opportunités*\n"]
        for i, t in enumerate(top_tokens[:5], 1):
            grade  = t.get("grade", "?")
            score  = t.get("total_score", 0)
            change = t.get("price_change_24h", 0)
            arrow  = "▲" if change > 0 else "▼"
            lines.append(
                f"{i}. `{t['symbol']}` — Grade {grade} ({score:.0f}/100) "
                f"{arrow}{abs(change):.1f}%"
            )
        lines.append(f"\n_macro\\_alpha • {datetime.now(timezone.utc).strftime('%H:%M UTC')}_")
        return self._send("\n".join(lines), kind="scanner")

    def alert_risk(self, event: str, details: str = "") -> bool:
        """Alerte risque / circuit-breaker."""
        msg = (
            f"⚠️ *Alerte Risque*\n"
            f"{event}\n"
        )
        if details:
            msg += f"_{details}_\n"
        msg += f"\n_macro\\_alpha • {datetime.now(timezone.utc).strftime('%H:%M UTC')}_"
        return self._send(msg, kind="risk")

    def test(self) -> dict:
        """Envoie un message de test et retourne le résultat."""
        if not self._enabled:
            return {
                "success": False,
                "reason":  "Telegram non configuré (TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant)",
            }

        msg = (
            "✅ *macro\\_alpha — Test d'alerte*\n"
            f"Connexion Telegram opérationnelle.\n"
            f"_ts: {datetime.now(timezone.utc).isoformat()}_"
        )
        # Bypass throttle pour le test
        old_last = self._last_sent.get("default", 0)
        self._last_sent["default"] = 0

        ok = self._send(msg, kind="default")
        if not ok:
            self._last_sent["default"] = old_last

        return {
            "success":    ok,
            "configured": self._enabled,
            "chat_id":    self._chat,
        }

    def get_stats(self) -> dict:
        return {
            "enabled":     self._enabled,
            "sent_total":  self._sent_count,
            "last_sent":   {k: datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
                            for k, v in self._last_sent.items()},
        }
