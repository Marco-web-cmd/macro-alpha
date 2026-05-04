"""
memory.py — Persistance DuckDB + Feedback Loop nocturne.
Sauvegarde les signaux de forte conviction et évalue leur précision a posteriori.
Ajuste les trust scores des modèles IA en conséquence.
"""
import os
import json
import uuid
import logging
import httpx
import asyncio
import duckdb
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("memory")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "macro_alpha.duckdb")


def get_db():
    """Connexion DuckDB (analytique, zéro config, fichier local)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)


def init_db():
    """Crée les tables si elles n'existent pas."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id                   VARCHAR PRIMARY KEY,
                ts                   TIMESTAMP,
                symbol               VARCHAR,
                interval             VARCHAR,
                signal               VARCHAR,
                alpha_score          FLOAT,
                conviction           FLOAT,
                price_at_signal      FLOAT,
                sl_price             FLOAT,
                tp1_price            FLOAT,
                tp2_price            FLOAT,
                risk_reward          FLOAT,
                regime               VARCHAR,
                macro_score          FLOAT,
                tech_score           FLOAT,
                fc_score             FLOAT,
                collat_score         FLOAT,
                dominant_pattern     VARCHAR,
                llm_synthesis        TEXT,
                market_state         JSON,
                resolved             BOOLEAN DEFAULT FALSE,
                price_n1             FLOAT,
                price_n7             FLOAT,
                direction_correct_n1 BOOLEAN,
                direction_correct_n7 BOOLEAN,
                sl_hit               BOOLEAN,
                tp1_hit              BOOLEAN
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS model_trust (
                model_name        VARCHAR PRIMARY KEY,
                trust_score       FLOAT DEFAULT 1.0,
                last_updated      TIMESTAMP,
                correct_n1        INTEGER DEFAULT 0,
                wrong_n1          INTEGER DEFAULT 0,
                correct_n7        INTEGER DEFAULT 0,
                wrong_n7          INTEGER DEFAULT 0,
                consecutive_wrong INTEGER DEFAULT 0
            )
        """)

        for model in ["chronos", "moirai", "lagllama", "ensemble"]:
            db.execute("""
                INSERT OR IGNORE INTO model_trust (model_name, trust_score, last_updated)
                VALUES (?, 1.0, CURRENT_TIMESTAMP)
            """, [model])

    logger.info("[Memory] DuckDB initialisé: %s", DB_PATH)


def save_signal(signal_data: dict, symbol: str = "BTCUSDT",
                interval: str = "1h") -> str | None:
    """
    Sauvegarde un signal de forte conviction (alpha_score > 65).
    Retourne l'ID si sauvegardé, None sinon.
    """
    alpha = signal_data.get("alpha", {})
    score = float(alpha.get("alpha_score", 0) or 0)

    if score < 65:
        return None

    signal_id = str(uuid.uuid4())[:12]
    try:
        with get_db() as db:
            db.execute("""
                INSERT INTO signals (
                    id, ts, symbol, interval, signal, alpha_score,
                    conviction, price_at_signal, sl_price, tp1_price,
                    tp2_price, risk_reward, regime, macro_score,
                    tech_score, fc_score, collat_score,
                    dominant_pattern, llm_synthesis, market_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                signal_id,
                datetime.now(timezone.utc),
                symbol, interval,
                alpha.get("signal"),
                score,
                float(alpha.get("conviction") or 0),
                float(signal_data.get("price") or 0),
                float(alpha.get("invalidation_sl") or 0),
                float(alpha.get("objectif_tp1") or 0),
                float(alpha.get("objectif_tp2") or 0),
                float(alpha.get("risk_reward") or 0),
                alpha.get("regime_detected") or alpha.get("regime"),
                float(alpha.get("score_macro") or alpha.get("layer_scores", {}).get("macro") or 0),
                float(alpha.get("score_technique") or alpha.get("layer_scores", {}).get("tech") or 0),
                float(alpha.get("score_forecast") or alpha.get("layer_scores", {}).get("forecast") or 0),
                float(alpha.get("score_collateral") or alpha.get("layer_scores", {}).get("collateral") or 0),
                (signal_data.get("technical", {}).get("patterns", [{}]) or [{}])[0].get("name"),
                signal_data.get("llm_synthesis"),
                json.dumps({"regime": alpha.get("regime_detected"),
                            "score": score,
                            "ts": signal_data.get("ts")}),
            ])
        logger.info("[Memory] Signal sauvegardé id=%s score=%.0f signal=%s",
                    signal_id, score, alpha.get("signal"))
        return signal_id
    except Exception as exc:
        logger.warning("[Memory] save_signal error: %s", exc)
        return None


def get_model_trust_scores() -> dict:
    """Retourne les trust scores actuels des modèles."""
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT model_name, trust_score, consecutive_wrong FROM model_trust"
            ).fetchall()
        return {
            row[0]: {
                "trust_score":       round(float(row[1]), 3),
                "consecutive_wrong": int(row[2]),
            }
            for row in rows
        }
    except Exception as exc:
        logger.warning("[Memory] get_model_trust_scores: %s", exc)
        return {}


async def nightly_evaluation() -> dict:
    """
    Évaluation nocturne — compare prédictions vs réalité.
    Ajuste les trust scores des modèles.

    Logique :
    - 3 erreurs consécutives → trust_score *= 0.85
    - Peu d'erreurs → trust_score = min(1.0, score * 1.02)
    """
    logger.info("[Memory] nightly_evaluation démarré")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
            )
            current_price = float(r.json()["price"])
    except Exception as exc:
        logger.error("[Memory] Impossible de récupérer le prix: %s", exc)
        return {"error": str(exc)}

    resolved_count = 0
    regime_stats   = []

    try:
        cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)

        with get_db() as db:
            unresolved = db.execute("""
                SELECT id, signal, price_at_signal, sl_price, tp1_price, ts
                FROM signals
                WHERE resolved = FALSE AND ts < ?
            """, [cutoff_24h]).fetchall()

            for row in unresolved:
                sig_id, signal, entry_price, sl, tp1, ts = row
                entry_price = float(entry_price or 0)
                sl  = float(sl  or 0)
                tp1 = float(tp1 or 0)

                is_long  = "LONG"  in (signal or "")
                is_short = "SHORT" in (signal or "")

                if is_long and entry_price > 0:
                    dir_correct = current_price > entry_price
                    sl_hit  = sl > 0 and current_price < sl
                    tp1_hit = tp1 > 0 and current_price > tp1
                elif is_short and entry_price > 0:
                    dir_correct = current_price < entry_price
                    sl_hit  = sl > 0 and current_price > sl
                    tp1_hit = tp1 > 0 and current_price < tp1
                else:
                    dir_correct = entry_price > 0 and abs(current_price - entry_price) < entry_price * 0.02
                    sl_hit = tp1_hit = False

                db.execute("""
                    UPDATE signals SET
                        resolved             = TRUE,
                        price_n1             = ?,
                        direction_correct_n1 = ?,
                        sl_hit               = ?,
                        tp1_hit              = ?
                    WHERE id = ?
                """, [current_price, dir_correct, sl_hit, tp1_hit, sig_id])
                resolved_count += 1

            # Statistiques par régime (30 derniers jours)
            try:
                stats = db.execute("""
                    SELECT
                        regime,
                        COUNT(*) as total,
                        SUM(CASE WHEN direction_correct_n1 THEN 1 ELSE 0 END) as correct,
                        AVG(alpha_score) as avg_score
                    FROM signals
                    WHERE resolved = TRUE
                    AND ts > CURRENT_TIMESTAMP - INTERVAL '30 days'
                    GROUP BY regime
                """).fetchall()
                regime_stats = [
                    {"regime": r[0], "total": r[1], "correct": r[2], "avg_score": round(float(r[3] or 0), 1)}
                    for r in stats
                ]
            except Exception:
                pass

            # Ajustement des trust scores
            for model in ["chronos", "moirai", "lagllama"]:
                try:
                    recent_wrong = db.execute("""
                        SELECT COUNT(*) FROM signals
                        WHERE resolved = TRUE
                        AND direction_correct_n1 = FALSE
                        AND ts > CURRENT_TIMESTAMP - INTERVAL '7 days'
                    """).fetchone()[0]

                    if recent_wrong >= 3:
                        db.execute("""
                            UPDATE model_trust
                            SET trust_score = GREATEST(0.3, trust_score * 0.85),
                                consecutive_wrong = consecutive_wrong + 1,
                                last_updated = CURRENT_TIMESTAMP
                            WHERE model_name = ?
                        """, [model])
                        logger.warning("[Memory] Modèle pénalisé: %s (%d erreurs)",
                                       model, recent_wrong)
                    else:
                        db.execute("""
                            UPDATE model_trust
                            SET trust_score = LEAST(1.0, trust_score * 1.02),
                                consecutive_wrong = 0,
                                last_updated = CURRENT_TIMESTAMP
                            WHERE model_name = ?
                        """, [model])
                except Exception as _e:
                    logger.debug("[Memory] trust update %s: %s", model, _e)

    except Exception as exc:
        logger.error("[Memory] nightly_evaluation error: %s", exc)
        return {"error": str(exc), "resolved": resolved_count}

    result = {
        "resolved_count": resolved_count,
        "current_price":  current_price,
        "trust_scores":   get_model_trust_scores(),
        "regime_stats":   regime_stats,
        "ts":             datetime.now(timezone.utc).isoformat(),
    }
    logger.info("[Memory] nightly_evaluation terminé: %d signaux résolus", resolved_count)
    return result
