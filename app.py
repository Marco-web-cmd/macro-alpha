"""
app.py — macro_alpha dashboard v3 (FastAPI + async + diskcache)
Dashboard institutionnel : Macro + Technique + Forecast IA + Collatéraux

v3 :
  - FastAPI async (remplace Flask)
  - Requêtes Binance + FRED parallèles via httpx.AsyncClient (~1.5s vs 8s)
  - Modèles IA préchargés au démarrage via lifespan
  - diskcache : cache disque persistant entre redémarrages
  - APScheduler : résolution prédictions + ré-entraînement meta-learner auto
  - /api/signal : signal structuré tldr/boussole/timing

Usage :
  uvicorn app:app --host 0.0.0.0 --port 5001 --reload
  OU python app.py (lance uvicorn directement)
"""

import os, sys, time, json, math, asyncio, logging
from datetime import datetime, timezone, date
from contextlib import asynccontextmanager

# ── Charge le .env en tout premier, avant tous les imports ──
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
except Exception:
    pass
from typing import Optional, List

import httpx
import numpy as np
import pandas as pd
import requests   # gardé pour la compatibilité des fonctions sync

try:
    import diskcache
    _disk_cache = diskcache.Cache("./data/cache", size_limit=200*1024*1024)
    DISKCACHE_OK = True
except ImportError:
    _disk_cache = None
    DISKCACHE_OK = False

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    _scheduler = AsyncIOScheduler(timezone="UTC")
    SCHEDULER_OK = True
except ImportError:
    _scheduler  = None
    SCHEDULER_OK = False

from fastapi import FastAPI, Query, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import websockets as _ws_lib

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("macro_alpha")

# ── Encodeur JSON custom ──
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.dirname(__file__))
from modules.macro_data   import compute_macro_score
from modules.technical    import full_technical_analysis
from modules.alpha_signal import compute_alpha_signal
from modules.ict_engine   import ICTEngine, ICTLongTermEngine

# ── Imports production (scanner, alertes, risk) ──
try:
    from modules.microcap_scanner import MicrocapScanner, SocialMomentumAnalyzer
    _scanner = MicrocapScanner()
    SCANNER_OK = True
except Exception as _e:
    _scanner   = None
    SCANNER_OK = False
    logger.warning("microcap_scanner non disponible: %s", _e)

try:
    from modules.screener import background_screener_loop, run_screener_once
    SCREENER_OK = True
except Exception as _e:
    SCREENER_OK = False
    logger.warning("screener non disponible: %s", _e)

try:
    from modules.llm_narrator import generate_llm_synthesis
    LLM_OK = True
except Exception as _e:
    LLM_OK = False
    logger.warning("llm_narrator non disponible: %s", _e)

try:
    from modules.memory import init_db, save_signal
    MEMORY_OK = True
except Exception as _e:
    MEMORY_OK = False
    logger.warning("memory non disponible: %s", _e)

try:
    from modules.alerting import AlertManager
    _alerts = AlertManager()
    ALERTS_OK = True
except Exception as _e:
    _alerts   = None
    ALERTS_OK = False
    logger.warning("alerting non disponible: %s", _e)

try:
    from modules.risk_manager import ProductionRiskManager
    _prod_risk = ProductionRiskManager()
    PROD_RISK_OK = True
except Exception as _e:
    _prod_risk   = None
    PROD_RISK_OK = False
    logger.warning("ProductionRiskManager non disponible: %s", _e)

# ── Imports modèles IA (optionnels) ──
try:
    import torch
    from chronos import ChronosPipeline
    CHRONOS_OK = True
except Exception:
    CHRONOS_OK = False

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset as MoiraiListDataset
    MOIRAI_OK = True
except Exception:
    MOIRAI_OK = False

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator as _LagLlamaEstimator
    from gluonts.dataset.common import ListDataset as _LagLlamaDataset
    import pandas as _pd_ll
    LAGLLAMA_OK = True
except Exception:
    LAGLLAMA_OK = False

try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

ML_MODELS: dict = {}          # modèles chargés au démarrage

async def _preload_models():
    """Précharge les modèles IA en background thread (ne bloque pas FastAPI)."""
    await asyncio.sleep(2)
    try:
        logger.info("[Warmup] Préchargement Lag-Llama...")
        dummy = np.linspace(50000, 67000, 600) + np.random.randn(600) * 500
        await asyncio.to_thread(run_lagllama_forecast, dummy, 24, 5)
        logger.info("[Warmup] Lag-Llama prêt — calibration walk-forward...")
        await asyncio.to_thread(_calibrate_lagllama, dummy, 10, 24)
    except Exception as exc:
        logger.warning("[Warmup] %s", exc)

def _setup_scheduler():
    """Configure les jobs APScheduler."""
    if not SCHEDULER_OK:
        return

    async def _auto_resolve():
        try:
            from modules.ai_agent import PredictionTracker
            tracker = PredictionTracker()
            price_r = requests.get("https://api.binance.com/api/v3/ticker/price",
                                   params={"symbol": "BTCUSDT"}, timeout=5)
            price   = float(price_r.json()["price"])
            n = tracker.resolve_matured_predictions(price)
            if n:
                logger.info(f"[Scheduler] {n} prédictions résolues")
        except Exception as exc:
            logger.warning(f"[Scheduler auto_resolve] {exc}")

    async def _auto_train():
        try:
            from modules.ai_agent import MetaLearner, PredictionTracker
            learner = MetaLearner()
            log     = PredictionTracker()._load()
            await asyncio.to_thread(learner.train_meta_model, log)
            logger.info("[Scheduler] MetaLearner ré-entraîné")
        except Exception as exc:
            logger.warning(f"[Scheduler auto_train] {exc}")

    _scheduler.add_job(_auto_resolve, "interval", minutes=5,  id="auto_resolve")
    _scheduler.add_job(_auto_train,   "interval", hours=24,   id="auto_train")

    async def _auto_scan():
        """Scan microcap toutes les 30 minutes + alerte si top tokens."""
        try:
            if _scanner is None:
                return
            results = await asyncio.to_thread(_scanner.scan, False)
            log_event("SCANNER", f"Scan terminé: {len(results)} tokens", {"top": results[:3] if results else []})
            if _alerts and results:
                grade_a = [t for t in results if t.get("grade") == "A"]
                if grade_a:
                    await asyncio.to_thread(_alerts.alert_scanner, grade_a[:5])
        except Exception as exc:
            logger.warning("[Scheduler auto_scan] %s", exc)

    _scheduler.add_job(_auto_scan, "interval", minutes=30, id="auto_scan")

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # ── Init DuckDB ──
    if MEMORY_OK:
        try:
            await asyncio.to_thread(init_db)
            logger.info("[Memory] DuckDB initialisé")
        except Exception as _dbe:
            logger.warning("[Memory] init_db: %s", _dbe)

    # ── Modèles IA (background) ──
    asyncio.create_task(_preload_models())

    # ── Screener background (top-50 Binance, 5min) ──
    if SCREENER_OK:
        asyncio.create_task(background_screener_loop())
        logger.info("[Screener] Worker démarré")

    # ── APScheduler ──
    if SCHEDULER_OK:
        _setup_scheduler()
        _scheduler.start()
        logger.info("[Scheduler] APScheduler démarré")

    # ── Solana Bot (auto-démarrage) ──
    try:
        bot = _get_solana_bot()
        await bot.start()
        logger.info("[SolanaBot] Démarré automatiquement au lancement")
    except Exception as _be:
        logger.warning("[SolanaBot] Échec auto-start: %s", _be)

    logger.info("=" * 65)
    logger.info("  macro_alpha v4 — FastAPI + WebSocket + DuckDB")
    logger.info("  diskcache : %s", "OK" if DISKCACHE_OK else "désactivé")
    logger.info("  scheduler : %s", "OK" if SCHEDULER_OK else "désactivé")
    logger.info("  screener  : %s", "OK" if SCREENER_OK  else "désactivé")
    logger.info("  memory    : %s", "OK" if MEMORY_OK    else "désactivé")
    logger.info("  llm       : %s", "OK" if LLM_OK       else "template")
    logger.info("  → http://localhost:5001")
    logger.info("=" * 65)
    yield
    if SCHEDULER_OK and _scheduler.running:
        _scheduler.shutdown(wait=False)
    if DISKCACHE_OK:
        _disk_cache.close()
    try:
        await _get_solana_bot().close()
    except Exception:
        pass

app       = FastAPI(title="macro_alpha v3", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

SYMBOL      = "BTCUSDT"
N_SAMPLES   = 50

# ── WebSocket : clients connectés ────────────────────────────
_ws_clients: set = set()
N_SAMPLES_LL = 100        # Lag-Llama : plus d'échantillons pour meilleurs percentiles
CONTEXT_LEN = 512

# ── Configuration des horizons de forecast ──
HORIZON_CONFIG = {
    "24h": {"steps": 24,  "data_interval": "1h", "limit": 300, "label": "24H",  "cache_ttl": 60,   "interval_h": 1},
    "7d":  {"steps": 7,   "data_interval": "1d", "limit": 200, "label": "7J",   "cache_ttl": 1800, "interval_h": 24},
    "30d": {"steps": 30,  "data_interval": "1d", "limit": 200, "label": "30J",  "cache_ttl": 3600, "interval_h": 24},
    "90d": {"steps": 90,  "data_interval": "1d", "limit": 200, "label": "90J",  "cache_ttl": 7200, "interval_h": 24},
}

_cache          = {}
_calib_history  = []   # [{model, ts_forecast, p10, p90, resolve_ts, resolved, hit}]
_signal_history = []   # les 10 derniers signaux avec confirmation

# ── TTL par type de données (secondes) ──
TTL_CONFIG = {
    "ohlcv_1h":   60,
    "ohlcv_4h":   120,
    "ohlcv_1d":   300,
    "ohlcv_1w":   900,
    "ticker":     15,
    "macro":      3600,
    "signal_1h":  60,
    "signal_4h":  120,
    "signal_1d":  180,
    "forecast":   300,
}

# ── Session HTTP persistante avec retry ──
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_http_retry = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])
HTTP_SESSION = requests.Session()
HTTP_SESSION.mount("https://", HTTPAdapter(max_retries=_http_retry))
HTTP_SESSION.mount("http://",  HTTPAdapter(max_retries=_http_retry))

# ── Event log (timeline en mémoire, 200 entrées max) ──
_event_log: list = []

def log_event(kind: str, message: str, data: dict = None) -> None:
    """Ajoute un événement dans le log timeline (limité à 200 entrées)."""
    global _event_log
    _event_log.append({
        "ts":      datetime.now(timezone.utc).isoformat(),
        "kind":    kind,
        "message": message,
        "data":    data or {},
    })
    if len(_event_log) > 200:
        _event_log = _event_log[-200:]


def _sanitize(obj):
    """Remplace récursivement NaN/Inf par None pour JSON valide."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, np.ndarray)):
        return [_sanitize(v) for v in obj]
    return obj


# ── Binance ──
def fetch_ohlcv(interval="1h", limit=300):
    url = "https://api.binance.com/api/v3/klines"
    r   = HTTP_SESSION.get(url, params={
        "symbol": SYMBOL, "interval": interval, "limit": limit
    }, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "timestamp","open","high","low","close","volume",
        "ct","qav","nt","tbb","tbq","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]


def fetch_ticker():
    r = HTTP_SESSION.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbol": SYMBOL}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Modèles IA (instances globales) ──
_chronos_pipe   = None
_moirai_model   = None
_lagllama_pred  = None

LAG_LLAMA_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--time-series-foundation-models--Lag-Llama"
    "/snapshots/72dcfc29da106acfe38250a60f4ae29d1e56a3d9/lag-llama.ckpt"
)


# ═══════════════════════════════════════════════════════════
# GBM DIFFÉRENCIÉS — 3 comportements distincts et réalistes
# ═══════════════════════════════════════════════════════════

def _gbm_chronos(closes, horizon, n):
    """
    Chronos GBM : processus Ornstein-Uhlenbeck (mean-reverting).
    - Tendance de retour vers la moyenne long terme
    - Faible skew (distribution quasi-symétrique)
    - Calibré sur la volatilité historique réelle
    """
    last      = float(closes[-1])
    hist      = closes[-min(60, len(closes)):]
    rets      = np.diff(np.log(np.clip(hist, 1e-10, None)))
    sigma     = float(np.std(rets)) if len(rets) > 1 else 0.02
    mu        = float(np.mean(rets))
    long_mean = float(np.mean(closes[-min(200, len(closes)):]))

    theta = 0.03   # vitesse de mean reversion modérée
    rng   = np.random.default_rng(seed=int(last * 100) % 99991)
    samples = np.zeros((n, horizon))

    for i in range(n):
        log_p    = np.log(max(last, 1e-10))
        log_mean = np.log(max(long_mean, 1e-10))
        for j in range(horizon):
            eps   = rng.standard_normal()   # quasi-symétrique
            d_log = theta * (log_mean - log_p) + mu + sigma * eps
            log_p = log_p + d_log - 0.5 * sigma**2
            samples[i, j] = np.exp(log_p)

    r = _build_result(samples, last)
    r["status"]     = "simulated"
    r["model_type"] = "chronos_ou"
    return r


def _gbm_moirai(closes, horizon, n):
    """
    MOIRAI GBM : momentum-following, distribution asymétrique positive.
    - Le drift suit le momentum récent (décroit exponentiellement)
    - Queue droite plus grasse (upside bias)
    - Distribution skewée via mélange Gaussienne/Exponentielle
    """
    last   = float(closes[-1])
    hist   = closes[-min(60, len(closes)):]
    rets   = np.diff(np.log(np.clip(hist, 1e-10, None)))
    sigma  = float(np.std(rets)) if len(rets) > 1 else 0.02

    # Momentum sur 20 périodes
    hist20   = closes[-min(21, len(closes)):]
    rets20   = np.diff(np.log(np.clip(hist20, 1e-10, None)))
    momentum = float(np.mean(rets20)) if len(rets20) > 1 else 0.0

    rng     = np.random.default_rng(seed=int(last * 137) % 99991)
    samples = np.zeros((n, horizon))

    for i in range(n):
        p = last
        for j in range(horizon):
            decay = np.exp(-j / max(horizon / 2, 5))   # momentum décroissant
            drift = momentum * decay
            # Distribution asymétrique : biais positif + queue exponentielle droite
            if rng.random() < 0.75:
                eps = rng.standard_normal() + 0.06      # biais positif léger
            else:
                eps = rng.exponential(1.0) - 0.35       # queue grasse droite
            p = p * np.exp(drift - 0.5 * sigma**2 + sigma * eps)
            samples[i, j] = max(p, 1.0)

    r = _build_result(samples, last)
    r["status"]     = "simulated"
    r["model_type"] = "moirai_momentum"
    return r


def _gbm_lagllama(closes, horizon, n):
    """
    Lag-Llama GBM : fat tails + volatilité stochastique (Heston simplifié).
    - Variance suit un processus CIR (mean-reverting)
    - Corrélation prix-vol négative (effet levier BTC ≈ -0.65)
    - Queues grasses via Student-t (15% des pas)
    - Paramètres Heston calibrés sur comportement BTC historique
    """
    last   = float(closes[-1])
    hist   = closes[-min(60, len(closes)):]
    rets   = np.diff(np.log(np.clip(hist, 1e-10, None)))
    sigma0 = float(np.std(rets)) if len(rets) > 1 else 0.02
    mu     = float(np.mean(rets))

    # Paramètres Heston — xi relatif à sigma0 pour satisfaire la condition de Feller
    # (2κθ > ξ² nécessaire pour que la variance reste positive)
    kappa   = 5.0              # mean reversion forte → vol stable
    theta_v = sigma0 ** 2      # variance long terme = vol historique²
    xi      = 0.35 * sigma0    # vol de vol = 35% de la vol spot (relatif, scalé)
    rho     = -0.65            # corrélation prix-vol (effet levier BTC)

    rng     = np.random.default_rng(seed=int(last * 42) % 99991)
    samples = np.zeros((n, horizon))

    for i in range(n):
        p = last
        v = sigma0 ** 2   # variance initiale

        for j in range(horizon):
            z1 = rng.standard_normal()
            z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal()

            # Queue grasse : événement extrême rare (15%)
            if rng.random() < 0.15:
                z1 = float(rng.standard_t(4)) / np.sqrt(2)

            # Processus CIR pour la variance avec plancher (Feller condition : 2κθ > ξ²)
            v_new = v + kappa * (theta_v - v) + xi * np.sqrt(max(v, 1e-10)) * z2
            v     = max(v_new, theta_v * 0.1)   # plancher à 10% de la variance long terme

            sigma_t = np.sqrt(v)
            p = p * np.exp(mu - 0.5 * v + sigma_t * z1)
            samples[i, j] = max(p, 1.0)

    r = _build_result(samples, last)
    r["status"]     = "simulated"
    r["model_type"] = "lagllama_heston"
    return r


def _compute_forecast_confidence(fc: dict, price: float) -> dict:
    """
    Calcule le score de confiance d'un forecast.
    confidence = max(0, 100 - (P90-P10)/Prix * 100)
    """
    p10_last = fc.get("p10", [price])[-1] if fc.get("p10") else price
    p90_last = fc.get("p90", [price])[-1] if fc.get("p90") else price
    spread_pct = (p90_last - p10_last) / price * 100 if price > 0 else 100
    confidence = max(0.0, min(100.0, 100 - spread_pct))
    model_type = fc.get("model_type", "unknown")
    is_real    = fc.get("status") == "real"

    if confidence < 20:
        level = "VERY_LOW"
        level_fr = "NON FIABLE — exclu du signal"
        weight_factor = 0.0
    elif confidence < 40:
        level = "LOW"
        level_fr = "Incertitude élevée — poids réduit (0.5x)"
        weight_factor = 0.5
    elif confidence < 60:
        level = "MEDIUM"
        level_fr = "Confiance modérée"
        weight_factor = 0.75
    else:
        level = "HIGH"
        level_fr = "Confiance élevée"
        weight_factor = 1.0

    return {
        "confidence":          round(confidence, 1),
        "confidence_score":    round(confidence, 1),   # alias pour compatibilité
        "spread_pct":          round(spread_pct, 1),
        "level":               level,
        "level_fr":            level_fr,
        "model_type":          model_type,
        "is_real_model":       is_real,
        "weight_factor":       weight_factor,
        "exclude_from_signal": confidence < 20,
    }


_ll_calib_cache  = {}   # {ts, mae_pct, mape, coverage, n_tests}
_ll_calib_ts     = 0


def _make_lagllama_predictor(block_size: int, n: int):
    """Crée un predictor Lag-Llama (cache pour block_size=24)."""
    global _lagllama_pred
    if _lagllama_pred is not None and _lagllama_pred.prediction_length == block_size:
        return _lagllama_pred
    ckpt   = torch.load(LAG_LLAMA_CKPT, map_location=DEVICE)
    kwargs = ckpt["hyper_parameters"]["model_kwargs"]
    est = _LagLlamaEstimator(
        ckpt_path=LAG_LLAMA_CKPT,
        prediction_length=block_size,
        context_length=CONTEXT_LEN,
        input_size=kwargs["input_size"],
        n_layer=kwargs["n_layer"],
        n_embd_per_head=kwargs["n_embd_per_head"],
        n_head=kwargs["n_head"],
        scaling=kwargs.get("scaling", "mean"),
        time_feat=kwargs.get("time_feat", False),
        rope_scaling=None,
        num_parallel_samples=n,
        device=torch.device(DEVICE),
    )
    pred = est.create_predictor(est.create_transformation(), est.create_lightning_module())
    if block_size == 24:
        _lagllama_pred = pred
    return pred


def _ll_predict_block(predictor, ctx_list: list) -> np.ndarray:
    """Exécute une prédiction Lag-Llama et retourne les samples."""
    ds = _LagLlamaDataset(
        [{"start": _pd_ll.Timestamp("2020-01-01"), "target": ctx_list}],
        freq="1h"
    )
    return list(predictor.predict(ds))[0].samples   # (n_samples, horizon)


def _ll_normalize(closes: np.ndarray):
    """Normalisation robuste median/MAD."""
    hist    = closes[-min(60, len(closes)):]
    med     = float(np.median(hist))
    mad     = float(np.median(np.abs(hist - med)))
    if mad < 1e-8:
        mad = float(np.std(hist)) + 1e-8
    return (closes - med) / mad, med, mad


def _ll_cap_trajectories(samples: np.ndarray, price: float,
                          closes: np.ndarray, horizon: int) -> np.ndarray:
    """
    Anti-explosion : cap chaque trajectoire à ±3σ où σ = vol_30J × √horizon.
    Si p90/p10 > 2.0 : rescale vers ±20% autour du p50 final.
    """
    rets    = np.diff(np.log(np.clip(closes[-min(30, len(closes)):], 1e-10, None)))
    vol_30  = float(np.std(rets)) if len(rets) > 1 else 0.02
    max_log = vol_30 * np.sqrt(horizon) * 3
    lower   = price * np.exp(-max_log)
    upper   = price * np.exp(max_log)
    samples = np.clip(samples, lower, upper)

    p10 = float(np.percentile(samples[:, -1], 10))
    p90 = float(np.percentile(samples[:, -1], 90))
    if p10 > 0 and p90 / p10 > 2.0:
        p50 = float(np.percentile(samples[:, -1], 50))
        scale = p50 * 0.20
        logger.warning("[Lag-Llama] rescale: p90/p10=%.2f → ±20%% autour p50=%.0f", p90/p10, p50)
        samples = np.clip(samples, p50 - scale, p50 + scale)

    return samples


def _calibrate_lagllama(closes: np.ndarray, n_tests: int = 10,
                         horizon: int = 24) -> dict:
    """
    Walk-forward calibration sur les n_tests dernières périodes.
    Calcule MAE%, MAPE, taux de couverture P10-P90.
    Résultat mis en cache 6h.
    """
    global _ll_calib_cache, _ll_calib_ts
    now = time.time()
    if _ll_calib_cache and (now - _ll_calib_ts) < 21600:
        return _ll_calib_cache

    if not LAGLLAMA_OK or not os.path.exists(LAG_LLAMA_CKPT):
        return {"available": False}

    min_ctx = CONTEXT_LEN + horizon + n_tests
    if len(closes) < min_ctx:
        return {"available": False, "reason": "Données insuffisantes"}

    try:
        predictor = _make_lagllama_predictor(horizon, N_SAMPLES_LL)
        maes, mapes, hits = [], [], []

        for i in range(n_tests):
            end_ctx  = len(closes) - n_tests + i - horizon
            if end_ctx < CONTEXT_LEN:
                continue
            ctx  = closes[max(0, end_ctx - CONTEXT_LEN): end_ctx]
            actual = closes[end_ctx: end_ctx + horizon]
            if len(actual) < horizon:
                continue

            ctx_norm, med, mad = _ll_normalize(ctx)
            samples_norm = _ll_predict_block(predictor, ctx_norm.tolist())
            samples = samples_norm * mad + med
            p10  = np.percentile(samples[:, :len(actual)], 10, axis=0)
            p50  = np.percentile(samples[:, :len(actual)], 50, axis=0)
            p90  = np.percentile(samples[:, :len(actual)], 90, axis=0)
            mae  = float(np.mean(np.abs(p50 - actual))) / float(np.mean(actual)) * 100
            mape = float(np.mean(np.abs((p50 - actual) / actual))) * 100
            coverage = float(np.mean((actual >= p10) & (actual <= p90))) * 100
            maes.append(mae)
            mapes.append(mape)
            hits.append(coverage)

        if not maes:
            return {"available": False}

        result = {
            "available":  True,
            "n_tests":    len(maes),
            "mae_pct":    round(float(np.mean(maes)), 2),
            "mape":       round(float(np.mean(mapes)), 2),
            "coverage":   round(float(np.mean(hits)), 1),
            "ts":         datetime.now(timezone.utc).isoformat(),
        }
        _ll_calib_cache = result
        _ll_calib_ts    = now
        logger.info("[Lag-Llama Calib] MAE=%.1f%% MAPE=%.1f%% Coverage=%.0f%%",
                    result["mae_pct"], result["mape"], result["coverage"])
        return result
    except Exception as e:
        logger.warning("[Lag-Llama Calib] Erreur: %s", e)
        return {"available": False, "error": str(e)}


def run_lagllama_forecast(closes, horizon=24, n=N_SAMPLES_LL):
    """
    Lag-Llama réel amélioré :
    - Normalisation robuste median/MAD avant inférence
    - num_samples=100, context_length=512
    - Anti-explosion : cap ±3σ, rescale si p90/p10 > 2
    - Mode récursif (blocs de 24) pour horizons > 24
    - Fallback GBM Heston si erreur
    """
    if not LAGLLAMA_OK or not os.path.exists(LAG_LLAMA_CKPT):
        return _gbm_lagllama(closes, horizon, n)
    try:
        predictor  = _make_lagllama_predictor(24, n)
        block_size = 24
        price      = float(closes[-1])

        # Normalisation robuste
        closes_norm, median_val, mad_val = _ll_normalize(closes)
        ctx_list   = closes_norm[-CONTEXT_LEN:].tolist()
        all_blocks = []
        remaining  = horizon

        while remaining > 0:
            block_norm = _ll_predict_block(predictor, ctx_list)   # (n, 24) en espace normalisé
            # Dénormaliser
            block = block_norm * mad_val + median_val
            steps = min(block_size, remaining)
            all_blocks.append(block[:, :steps])
            # Contexte suivant en espace normalisé
            ctx_list = ctx_list + ((np.median(block_norm, axis=0)).tolist())
            remaining -= steps

        samples = np.concatenate(all_blocks, axis=1)

        # Anti-explosion
        samples = _ll_cap_trajectories(samples, price, closes, horizon)

        r = _build_result(samples, price)
        r["status"]     = "real"
        r["model_type"] = "lagllama_real"

        # Calibration walk-forward (en arrière-plan, ne bloque pas)
        calib = _ll_calib_cache if _ll_calib_cache else {"available": False}
        r["calibration"] = calib

        return r
    except Exception as e:
        logger.warning("[Lag-Llama] %s", e)
        return _gbm_lagllama(closes, horizon, n)


def _build_result(samples, last):
    """Calcule les percentiles et métriques à partir des trajectoires simulées."""
    from scipy.stats import skew as scipy_skew
    finals = samples[:, -1]
    # Vol implicite annualisée
    log_r  = np.log(np.clip(samples[:, 1:] / samples[:, :-1], 1e-10, None))
    return {
        "p5":          np.percentile(samples, 5,  axis=0).tolist(),
        "p10":         np.percentile(samples, 10, axis=0).tolist(),
        "p50":         np.percentile(samples, 50, axis=0).tolist(),
        "p90":         np.percentile(samples, 90, axis=0).tolist(),
        "p95":         np.percentile(samples, 95, axis=0).tolist(),
        "prob_bull":   float(np.mean(finals > last) * 100),
        "implied_vol": float(np.std(log_r) * np.sqrt(365 * 24) * 100),
        "skewness":    float(scipy_skew(finals)),
        "status":      "real"
    }


# ── Chronos avec mode récursif pour horizons longs ──
def run_chronos_forecast(closes, horizon=24, n=N_SAMPLES):
    global _chronos_pipe
    if not CHRONOS_OK:
        return _gbm_chronos(closes, horizon, n)
    try:
        if _chronos_pipe is None:
            _chronos_pipe = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small", device_map=DEVICE, dtype=torch.float32)

        if horizon <= 24:
            # Forecast direct
            ctx = torch.tensor(closes[-CONTEXT_LEN:], dtype=torch.float32).unsqueeze(0)
            fc  = _chronos_pipe.predict(ctx, prediction_length=horizon,
                                        num_samples=n, limit_prediction_length=False)
            samples = fc[0].numpy()
        else:
            # Mode récursif : blocs de 24 pas réinjectés comme contexte
            block_size = 24
            ctx_list   = closes[-CONTEXT_LEN:].tolist()
            all_blocks = []
            remaining  = horizon
            while remaining > 0:
                this_block = min(block_size, remaining)
                ctx_t = torch.tensor(ctx_list[-CONTEXT_LEN:], dtype=torch.float32).unsqueeze(0)
                fc    = _chronos_pipe.predict(ctx_t, prediction_length=this_block,
                                              num_samples=n, limit_prediction_length=False)
                block = fc[0].numpy()
                all_blocks.append(block)
                ctx_list  = ctx_list + np.median(block, axis=0).tolist()
                remaining -= this_block
            samples = np.concatenate(all_blocks, axis=1)

        r = _build_result(samples, closes[-1])
        r["model_type"] = "chronos_real"
        return r
    except Exception as e:
        logger.warning("[Chronos] %s", e)
        return _gbm_chronos(closes, horizon, n)


# ── MOIRAI avec mode récursif pour horizons longs ──
def run_moirai_forecast(closes, horizon=24, n=N_SAMPLES):
    global _moirai_model
    if not MOIRAI_OK:
        return _gbm_moirai(closes, horizon, n)
    try:
        if _moirai_model is None:
            _moirai_model = MoiraiForecast(
                module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
                prediction_length=24, context_length=CONTEXT_LEN,
                patch_size="auto", num_samples=n, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)

        block_size = 24
        ctx_list   = closes[-CONTEXT_LEN:].tolist()
        all_blocks = []
        remaining  = horizon

        while remaining > 0:
            this_block = min(block_size, remaining)
            _moirai_model.hparams.prediction_length = this_block
            _moirai_model.hparams.num_samples       = n
            predictor = _moirai_model.create_predictor(batch_size=1)
            ds = MoiraiListDataset(
                [{"start": pd.Timestamp("2020-01-01"), "target": ctx_list[-CONTEXT_LEN:]}],
                freq="1h")
            fc = list(predictor.predict(ds))
            block = fc[0].samples
            all_blocks.append(block)
            ctx_list  = ctx_list + np.median(block, axis=0).tolist()
            remaining -= this_block

        samples = np.concatenate(all_blocks, axis=1)
        r = _build_result(samples, closes[-1])
        r["model_type"] = "moirai_real"
        return r
    except Exception as e:
        logger.warning("[MOIRAI] %s", e)
        return _gbm_moirai(closes, horizon, n)


# ═══════════════════════════════════════════════════════════
# CALIBRATION DES FORECASTS
# ═══════════════════════════════════════════════════════════

def _record_calib(model, p10_final, p90_final, horizon_steps, interval_h=1):
    """Enregistre une prédiction pour vérification future (prix dans P10-P90 ?)."""
    _calib_history.append({
        "model":       model,
        "ts_forecast": time.time(),
        "p10":         float(p10_final),
        "p90":         float(p90_final),
        "resolve_ts":  time.time() + horizon_steps * interval_h * 3600,
        "resolved":    False,
        "hit":         None,
    })
    while len(_calib_history) > 300:
        _calib_history.pop(0)


def _compute_calibration(current_price):
    """
    Vérifie les prédictions matures et calcule le taux de calibration.
    Un forecast est 'calibré' si le prix réel tombe dans le range P10-P90 prédit.
    Valeur cible théorique : ~80% (c'est un intervalle de confiance 80%).
    """
    now = time.time()
    for e in _calib_history:
        if not e["resolved"] and now >= e["resolve_ts"]:
            e["resolved"] = True
            e["hit"]      = (e["p10"] <= current_price <= e["p90"])

    per_model = {}
    for e in _calib_history:
        if e["resolved"]:
            m = e["model"]
            per_model.setdefault(m, {"n": 0, "hits": 0})
            per_model[m]["n"] += 1
            if e["hit"]:
                per_model[m]["hits"] += 1

    total_n    = sum(v["n"] for v in per_model.values())
    total_hits = sum(v["hits"] for v in per_model.values())

    return {
        "overall":   round(total_hits / total_n * 100, 1) if total_n > 0 else None,
        "per_model": {m: round(v["hits"]/v["n"]*100, 1) for m, v in per_model.items() if v["n"] > 0},
        "n_matured": total_n,
        "n_pending": len(_calib_history) - total_n,
        "target_pct": 80.0,   # cible théorique pour intervalle P10-P90
        "note":      f"Sur {total_n} prédictions matures" if total_n > 0 else "En attente de maturité",
    }


# ═══════════════════════════════════════════════════════════
# HISTORIQUE DES SIGNAUX
# ═══════════════════════════════════════════════════════════

def _update_signal_history(signal, alpha_score, price):
    """
    Maintient les 10 derniers signaux avec leur confirmation a posteriori.
    Un signal LONG est confirmé si le prix monte > 0.8% au prochain update.
    """
    # Renseigner le next_price des entrées en attente
    for entry in _signal_history:
        if entry.get("next_price") is None:
            entry["next_price"] = price
            delta = (price - entry["price"]) / entry["price"] * 100
            if entry["signal"] in ["LONG FORT", "LONG"] and delta > 0.8:
                entry["confirmed"] = True
            elif entry["signal"] in ["SHORT FORT", "SHORT"] and delta < -0.8:
                entry["confirmed"] = True
            elif entry["signal"] == "NEUTRE" and abs(delta) <= 1.5:
                entry["confirmed"] = True
            elif abs(delta) > 0.8:
                entry["confirmed"] = False
            # sinon : mouvement insuffisant, on garde None (en attente)

    _signal_history.append({
        "ts":          datetime.now(timezone.utc).isoformat(),
        "signal":      signal,
        "alpha_score": round(alpha_score, 1),
        "price":       price,
        "next_price":  None,
        "confirmed":   None,
    })

    while len(_signal_history) > 10:
        _signal_history.pop(0)

    return list(reversed(_signal_history))   # plus récent en premier


def safe_jsonify(obj, status_code: int = 200) -> JSONResponse:
    """Sérialise obj en JSONResponse avec SafeEncoder (compatible Flask + FastAPI)."""
    content = json.loads(json.dumps(_sanitize(obj), cls=SafeEncoder))
    return JSONResponse(content=content, status_code=status_code)


# ═══════════════════════════════════════════════════════════
# FETCH ASYNCHRONE — requêtes Binance en parallèle
# ═══════════════════════════════════════════════════════════

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"

def _parse_klines(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=[
        "timestamp","open","high","low","close","volume",
        "ct","qav","nt","tbb","tbq","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]

async def _async_ohlcv(client: httpx.AsyncClient, interval: str, limit: int) -> pd.DataFrame:
    r = await client.get(BINANCE_KLINES, params={"symbol": SYMBOL, "interval": interval, "limit": limit})
    r.raise_for_status()
    return _parse_klines(r.json())

async def _async_ticker(client: httpx.AsyncClient) -> dict:
    r = await client.get(BINANCE_TICKER, params={"symbol": SYMBOL})
    r.raise_for_status()
    return r.json()

async def fetch_all_parallel(interval: str, chart_limit: int = 500,
                              daily_limit: int = 730) -> tuple:
    """
    Lance toutes les requêtes Binance en parallèle.
    Retourne (df, ticker, df_daily, df_weekly, df_monthly).
    Avant (séquentiel) : ~3s. Après (parallèle) : ~0.8s.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [
            _async_ohlcv(client, interval, chart_limit),   # 0: df principal
            _async_ticker(client),                          # 1: ticker
            _async_ohlcv(client, "1w",  104),               # 2: weekly  (2 ans)
            _async_ohlcv(client, "1M",  36),                # 3: monthly (3 ans)
        ]
        if interval != "1d":
            tasks.append(_async_ohlcv(client, "1d", daily_limit))  # 4: daily
        results = await asyncio.gather(*tasks, return_exceptions=True)

    def _safe(r):
        return r if not isinstance(r, Exception) else None

    df        = _safe(results[0])
    ticker    = _safe(results[1]) or {}
    df_weekly = _safe(results[2])
    df_monthly = _safe(results[3])
    df_daily  = None
    if interval == "1d":
        df_daily = df
    elif len(results) > 4 and not isinstance(results[4], Exception):
        df_daily = results[4]
    return df, ticker, df_daily, df_weekly, df_monthly


# ═══════════════════════════════════════════════════════════
# ROUTES API
# ═══════════════════════════════════════════════════════════

@app.get("/api/full")
async def api_full(interval: str = Query("1h"),
                   forecast_horizon: str = Query("24h")):

    h_cfg       = HORIZON_CONFIG.get(forecast_horizon, HORIZON_CONFIG["24h"])
    chart_limit = {"1h": 500, "4h": 500, "1d": 500}.get(interval, 500)

    cache_key = f"full_{interval}_{forecast_horizon}"
    now       = time.time()
    if cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < h_cfg["cache_ttl"]:
        return safe_jsonify(_cache[cache_key])

    try:
        # ── Requêtes Binance parallèles (httpx async) ──
        df, ticker, df_daily, df_weekly, df_monthly = await fetch_all_parallel(interval, chart_limit)
        if df is None:
            return safe_jsonify({"error": "Binance indisponible"}, 503)
        price  = float(ticker["lastPrice"])
        change = float(ticker["priceChangePercent"])

        # ── Analyse technique + macro en parallèle (threads) ──
        tech, macro = await asyncio.gather(
            asyncio.to_thread(full_technical_analysis, df, df_daily, interval),
            asyncio.to_thread(compute_macro_score),
        )

        # ── Données forecast (TF propre à chaque horizon) ──
        fc_interval = h_cfg["data_interval"]
        fc_limit    = h_cfg["limit"]
        horizon     = h_cfg["steps"]
        interval_h  = h_cfg["interval_h"]

        if fc_interval == interval:
            closes_fc = df["close"].values.astype(float)
        else:
            df_fc     = fetch_ohlcv(interval=fc_interval, limit=fc_limit)
            closes_fc = df_fc["close"].values.astype(float)

        # ── Forecasts : 3 modèles (réels ou GBM fallback) ──
        fc_chronos = run_chronos_forecast(closes_fc, horizon) if CHRONOS_OK \
                     else _gbm_chronos(closes_fc, horizon, N_SAMPLES)

        if MOIRAI_OK:
            closes_m  = df["close"].values.astype(float)
            moirai_h  = min(horizon * interval_h, 168)
            fc_moirai = run_moirai_forecast(closes_m, moirai_h)
        else:
            fc_moirai = _gbm_moirai(closes_fc, horizon, N_SAMPLES)

        # Lag-Llama réel sur 24h seulement (trop lent sur CPU pour horizons longs)
        # → GBM Heston pour 7d/30d/90d (résultat quasi-instantané, cache ensuite)
        if horizon <= 24 and interval_h == 1:
            fc_lag = run_lagllama_forecast(closes_fc, horizon, N_SAMPLES)
        else:
            fc_lag = _gbm_lagllama(closes_fc, horizon, N_SAMPLES)

        # ── Scores de confiance par modèle ──
        conf_chronos = _compute_forecast_confidence(fc_chronos, price)
        conf_moirai  = _compute_forecast_confidence(fc_moirai,  price)
        conf_lag     = _compute_forecast_confidence(fc_lag,     price)

        fc_chronos["confidence"] = conf_chronos
        fc_moirai["confidence"]  = conf_moirai
        fc_lag["confidence"]     = conf_lag

        # ── Enregistrer pour calibration (si confiance suffisante) ──
        for m_name, fc in [("chronos", fc_chronos), ("moirai", fc_moirai), ("lagllama", fc_lag)]:
            if fc.get("p10") and fc.get("p90") and not fc["confidence"]["exclude_from_signal"]:
                _record_calib(m_name, fc["p10"][-1], fc["p90"][-1], horizon, interval_h)

        forecasts = {"chronos": fc_chronos, "moirai": fc_moirai, "lagllama": fc_lag}

        # Divergence inter-modèles (normalisée, exclure les modèles peu fiables)
        reliable_models = [m for m in ["chronos","moirai","lagllama"]
                           if not forecasts[m]["confidence"]["exclude_from_signal"]]
        if len(reliable_models) < 2:
            reliable_models = ["chronos","moirai","lagllama"]   # fallback : tous
        p50s_raw = [np.array(forecasts[m]["p50"]) for m in reliable_models]
        min_len  = min(len(p) for p in p50s_raw)
        p50s     = np.array([p[:min_len] for p in p50s_raw])
        forecasts["meta"] = {
            "horizon":          horizon,
            "horizon_label":    h_cfg["label"],
            "divergence":       float(np.mean(np.std(p50s, axis=0) / price * 100)),
            "reliable_models":  reliable_models,
            "ts":               datetime.now(timezone.utc).isoformat(),
        }

        # ── Signal alpha (on exclut les modèles < 15% confiance) ──
        forecasts_for_alpha = dict(forecasts)
        for m in ["chronos","moirai","lagllama"]:
            if forecasts[m]["confidence"]["exclude_from_signal"]:
                # Remplacer par un forecast neutre
                forecasts_for_alpha[m] = {**forecasts[m],
                    "p50": [price] * horizon, "prob_bull": 50.0}

        alpha = compute_alpha_signal(
            macro_score=macro["score"],
            tech_score=tech["score"],
            forecast_data=forecasts_for_alpha,
            current_price=price,
            macro_detail=macro,
            tech_detail=tech,
            interval=interval,
        )

        # ── ICT intraday (TF courts) ──
        ict_data: dict = {}
        if interval in ("5m", "15m", "1h", "4h"):
            try:
                ict_data = await asyncio.to_thread(
                    ICTEngine().full_ict_analysis, df, interval
                )
            except Exception as _e:
                ict_data = {"error": str(_e)}

        # ── ICT long terme (toujours, horizons 7J/30J/90J) ──
        ict_lt_data: dict = {}
        if df_daily is not None:
            try:
                _ict_lt = ICTLongTermEngine()
                ict_lt_data = await asyncio.to_thread(
                    _ict_lt.full_long_term_ict_analysis,
                    df_daily, df_weekly, df_monthly, macro, price
                )
            except Exception as _e:
                logger.warning(f"ICTLongTerm: {_e}")
                ict_lt_data = {"error": str(_e)}

        # ── Validation croisée ICT ↔ forecast IA 7J ──
        ict_forecast_alignment: dict = {}
        try:
            fc_7j  = forecasts.get("moirai", {})
            ict_7j = ict_lt_data.get("horizon_signals", {}).get("7j", {})
            if fc_7j and ict_7j and ict_7j.get("signal") != "NEUTRE":
                fc_dir  = "LONG" if (fc_7j.get("prob_bull", 50) or 50) > 55 else "SHORT"
                aligned = fc_dir == ict_7j["signal"]
                ict_forecast_alignment = {
                    "7j_aligned": aligned,
                    "fc_direction": fc_dir,
                    "ict_signal":  ict_7j["signal"],
                    "interpretation": (
                        f"ICT et MOIRAI {'ALIGNÉS' if aligned else 'DIVERGENTS'} "
                        f"sur 7J — {'signal renforcé' if aligned else 'prudence recommandée'}"
                    ),
                }
        except Exception:
            pass

        # ── Calibration & historique signaux ──
        calib       = _compute_calibration(price)
        signal_hist = _update_signal_history(alpha["signal"], alpha["alpha_score"], price)

        # ── Setup quality (PRIORITÉ 6) ──
        try:
            from modules.alpha_signal import compute_setup_quality
            _sq = compute_setup_quality(alpha, ict_data=ict_data)
            if _sq.get("grade") in ("A+", "A"):
                log_event("SIGNAL",
                          f"Setup {_sq['grade']} — {alpha.get('signal','?')} @ ${price:,.0f}",
                          {"grade": _sq["grade"], "score": _sq["score"]})
        except Exception:
            _sq = {}

        result = {
            "price":                   price,
            "change":                  change,
            "interval":                interval,
            "forecast_horizon":        forecast_horizon,
            "alpha":                   alpha,
            "macro":                   macro,
            "technical":               tech,
            "forecasts":               forecasts,
            "ict":                     ict_data,
            "ict_lt":                  ict_lt_data,
            "ict_forecast_alignment":  ict_forecast_alignment,
            "calibration":             calib,
            "signal_history":          signal_hist,
            "setup_quality":           _sq,
            "ts":                      datetime.now(timezone.utc).isoformat(),
        }

        # ── Synthèse LLM (async, ne bloque pas si indisponible) ──
        if LLM_OK:
            try:
                result["llm_synthesis"] = await generate_llm_synthesis(result)
            except Exception as _le:
                logger.debug("[LLM] synthesis skipped: %s", _le)
                result["llm_synthesis"] = None
        else:
            result["llm_synthesis"] = None

        # ── Sauvegarde DuckDB (signaux conviction > 65) ──
        if MEMORY_OK:
            try:
                await asyncio.to_thread(save_signal, result, SYMBOL, interval)
            except Exception as _me:
                logger.debug("[Memory] save_signal: %s", _me)

        result_safe = json.loads(json.dumps(_sanitize(result), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        # Cache df brut pour réutilisation dans /api/signal (anomaly detection)
        _cache[f"df_{interval}"]  = df
        return safe_jsonify(result_safe)

    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/signal")
async def api_signal(interval: str = Query("1h"),
                     forecast_horizon: str = Query("24h"),
                     force: str = Query("0")):
    """
    Signal structuré : tldr / boussole / timing
    Nouveau endpoint v3 — données actionnables en 3 secondes.
    """
    cache_key = f"signal_{interval}_{forecast_horizon}"
    now       = time.time()
    ttl       = {"1h": 60, "4h": 120, "1d": 180, "1w": 300}.get(interval, 60)
    if force != "1" and cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < ttl:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.alpha_signal import build_structured_signal
        # Récupère les données via l'endpoint full (réutilise son cache si disponible)
        full_data = _cache.get(f"full_{interval}_{forecast_horizon}")
        df_for_anomaly = None
        if not full_data:
            # Si pas en cache : fetch rapide parallèle
            chart_limit = 500
            df, ticker, df_daily, df_weekly, df_monthly = await fetch_all_parallel(interval, chart_limit)
            if df is None:
                return safe_jsonify({"error": "Binance indisponible"}, 503)
            df_for_anomaly = df
            price  = float(ticker["lastPrice"])
            change = float(ticker["priceChangePercent"])
            tech, macro = await asyncio.gather(
                asyncio.to_thread(full_technical_analysis, df, df_daily, interval),
                asyncio.to_thread(compute_macro_score),
            )
            from modules.alpha_signal import compute_alpha_signal
            closes_fc = df["close"].values.astype(float)
            h_cfg = HORIZON_CONFIG.get(forecast_horizon, HORIZON_CONFIG["24h"])
            horizon = h_cfg["steps"]
            fc_chronos = _gbm_chronos(closes_fc, horizon, N_SAMPLES)
            fc_moirai  = _gbm_moirai(closes_fc, horizon, N_SAMPLES)
            fc_lag     = _gbm_lagllama(closes_fc, horizon, N_SAMPLES)
            for fc_m in [fc_chronos, fc_moirai, fc_lag]:
                fc_m["confidence"] = _compute_forecast_confidence(fc_m, price)
            forecasts = {"chronos": fc_chronos, "moirai": fc_moirai, "lagllama": fc_lag}
            forecasts["meta"] = {"horizon": horizon, "divergence": 1.0,
                                  "ts": datetime.now(timezone.utc).isoformat()}
            alpha = compute_alpha_signal(
                macro_score=macro["score"], tech_score=tech["score"],
                forecast_data=forecasts, current_price=price,
                macro_detail=macro, tech_detail=tech, interval=interval,
            )
        else:
            price     = full_data["price"]
            tech      = full_data["technical"]
            macro     = full_data["macro"]
            forecasts = full_data["forecasts"]
            alpha     = full_data["alpha"]
            # Tente de récupérer le df depuis un cache dédié
            df_for_anomaly = _cache.get(f"df_{interval}")

        structured = build_structured_signal(
            alpha=alpha, macro_detail=macro, tech_detail=tech,
            forecasts=forecasts, current_price=price, interval=interval,
            df=df_for_anomaly,
        )
        # Setup quality score (PRIORITÉ 6)
        try:
            from modules.alpha_signal import compute_setup_quality
            mtf_data = _cache.get(f"mtf_{interval}")
            ict_data = _cache.get("ict_data")
            quality  = compute_setup_quality(alpha, ict_data=ict_data, mtf_data=mtf_data)
            structured["setup_quality"] = quality
            if quality.get("grade") in ("A+", "A"):
                log_event("SIGNAL", f"Setup {quality['grade']} détecté — {alpha.get('signal','?')} @ ${price:,.0f}",
                          {"grade": quality["grade"], "score": quality["score"]})
        except Exception:
            pass
        result_safe = json.loads(json.dumps(_sanitize(structured), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.get("/api/health")
async def api_health():
    """Status des modèles et services."""
    return safe_jsonify({
        "status":     "ok",
        "fastapi":    True,
        "diskcache":  DISKCACHE_OK,
        "scheduler":  SCHEDULER_OK and (_scheduler.running if SCHEDULER_OK else False),
        "chronos":    CHRONOS_OK,
        "moirai":     MOIRAI_OK,
        "lagllama":   LAGLLAMA_OK and os.path.exists(LAG_LLAMA_CKPT),
        "cache_keys": len(_cache),
        "ts":         datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/candles")
async def api_candles(interval: str = Query("1h")):
    tf_map   = {"1h": ("1h", 500), "4h": ("4h", 500), "1d": ("1d", 500)}
    iv, limit = tf_map.get(interval, ("1h", 500))
    try:
        df = await asyncio.to_thread(fetch_ohlcv, iv, limit)
        return safe_jsonify({
            "timestamps": [t.isoformat() for t in df.index],
            "open":   df["open"].tolist(),
            "high":   df["high"].tolist(),
            "low":    df["low"].tolist(),
            "close":  df["close"].tolist(),
            "volume": df["volume"].tolist(),
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}, 500)


@app.get("/api/macro")
async def api_macro():
    try:
        return safe_jsonify(await asyncio.to_thread(compute_macro_score))
    except Exception as e:
        return safe_jsonify({"error": str(e)}, 500)


@app.get("/api/status")
async def api_status():
    ll_real = LAGLLAMA_OK and os.path.exists(LAG_LLAMA_CKPT)
    calib   = _ll_calib_cache if _ll_calib_cache else {"available": False}
    ll_label = "réel (checkpoint HF)" if ll_real else "GBM Heston (fat tails)"
    if calib.get("available"):
        ll_label += f" | MAE={calib['mae_pct']:.1f}% Coverage={calib['coverage']:.0f}%"
    return safe_jsonify({
        "chronos":  {"available": CHRONOS_OK,  "device": DEVICE, "gbm_fallback": "OU mean-reverting"},
        "moirai":   {"available": MOIRAI_OK,   "device": DEVICE, "gbm_fallback": "Momentum skewed"},
        "lagllama": {"available": ll_real,      "device": DEVICE,
                     "label": ll_label, "calibration": calib},
    })


def _fetch_tf_snapshot(tf: str) -> dict:
    """
    Retourne un snapshot technique complet pour le TF demandé.
    Cherche d'abord dans le cache (toutes horizons), sinon fetch Binance en direct.
    Garantit que les indicateurs (RSI, ADX, HMA…) sont calculés sur les vraies bougies du TF.
    """
    for fh in ("24h", "7d", "30d", "90d"):
        d = _cache.get(f"full_{tf}_{fh}")
        if d:
            return d
    # Pas en cache → fetch frais (sans forecasts pour rester rapide)
    try:
        limit     = {"5m": 500, "15m": 500, "1h": 500, "4h": 500, "1d": 500, "1w": 200}.get(tf, 500)
        df        = fetch_ohlcv(interval=tf, limit=limit)
        df_daily  = fetch_ohlcv(interval="1d", limit=365) if tf not in ("1d", "1w") else df
        ticker    = fetch_ticker()
        price     = float(ticker["lastPrice"])
        change    = float(ticker["priceChangePercent"])
        tech      = full_technical_analysis(df, df_daily=df_daily, interval=tf)
        macro     = compute_macro_score()
        return {
            "price":    price,
            "change":   change,
            "interval": tf,
            "technical": tech,
            "macro":    macro,
            "forecasts": {},
        }
    except Exception as exc:
        logger.warning(f"_fetch_tf_snapshot({tf}): {exc}")
        return _cache.get("full_1h_24h", {})


@app.get("/api/agent")
async def api_agent(interval: str = Query("1h"), force: str = Query("0")):
    """Agent IA — analyse complète + recommandations. Cache 5 min par TF."""
    tf    = interval
    _force = force == "1"
    cache_key = f"agent_full_{tf}"
    now = time.time()
    if not _force and cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < 300:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.ai_agent import CryptoAgent
        agent = CryptoAgent()
        d = await asyncio.to_thread(_fetch_tf_snapshot, tf)
        if not d:
            return safe_jsonify({"error": "Données non disponibles"}, 503)
        analysis = await asyncio.to_thread(
            agent.run_full_analysis,
            {"price": d.get("price"), "change": d.get("change")},
            d.get("macro", {}),
            d.get("technical", {}),
            d.get("forecasts", {}),
            tf,
        )
        result_safe = json.loads(json.dumps(_sanitize(analysis), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/predictions")
async def api_agent_predictions():
    """Historique des prédictions et leur réalisation."""
    try:
        from modules.ai_agent import PredictionTracker
        tracker = PredictionTracker()
        return safe_jsonify({
            "predictions":      tracker.get_recent(50),
            "performance":      tracker.get_performance_stats(),
            "learning_insights": tracker.get_learning_insights(),
        })
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/altcoins")
async def api_agent_altcoins(interval: str = Query("1h")):
    """Signal DCA et ranking des altcoins."""
    cache_key = "agent_altcoins"
    now = time.time()
    if cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < 300:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.ai_agent import AltcoinStrategist
        strat = AltcoinStrategist()
        state = _cache.get(f"full_{interval}_24h", _cache.get("full_1h_24h", {}))
        result = await asyncio.to_thread(strat.run_analysis, state)
        result_safe = json.loads(json.dumps(_sanitize(result), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/mtf")
async def api_agent_mtf(interval: str = Query("1h"), force: str = Query("0"),
                         symbol: str = Query("BTCUSDT"),
                         tf: List[str] = Query([])):
    """Analyse Multi-Timeframe — biais + confluence + qualité d'entrée. Cache TF-aware."""
    _tf   = interval
    _force = force == "1"
    cache_key = f"agent_mtf_{_tf}"
    ttl   = {"1h": 60, "4h": 120, "1d": 180, "1w": 300}.get(_tf, 60)
    now   = time.time()
    if not _force and cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < ttl:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.mtf_engine import MultiTimeframeEngine
        engine = MultiTimeframeEngine()
        TF_TO_ANALYZE = {
            "1h":  ["5m", "15m", "1h", "4h", "1d"],
            "4h":  ["1h", "4h", "1d", "1w"],
            "1d":  ["4h", "1d", "1w"],
            "1w":  ["1d", "1w"],
        }
        tfs    = tf if tf else TF_TO_ANALYZE.get(_tf, ["1h", "4h", "1d"])
        mtf    = await asyncio.to_thread(engine.get_mtf_bias, symbol, tfs)
        state  = await asyncio.to_thread(_fetch_tf_snapshot, _tf)
        key_levels    = state.get("technical", {}).get("key_levels", {})
        current_price = float(state.get("price", 0) or 0)
        entry_zone  = engine.find_optimal_entry_zone(mtf, key_levels, current_price)
        divergences = engine.detect_divergences_mtf(symbol)
        result = _sanitize({
            "mtf_bias":    mtf,
            "entry_zone":  entry_zone,
            "divergences": divergences,
            "interval":    _tf,
            "tfs_analyzed": tfs,
        })
        result_safe = json.loads(json.dumps(result, cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/setup")
async def api_agent_setup(interval: str = Query("1h"), force: str = Query("0")):
    """Génère un setup de trade complet (MTF + Risk + Narration). Cache TF-aware."""
    _tf   = interval
    _force = force == "1"
    cache_key = f"agent_setup_{_tf}"
    ttl   = {"1h": 60, "4h": 120, "1d": 180, "1w": 300}.get(_tf, 60)
    now   = time.time()
    if not _force and cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < ttl:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.mtf_engine   import MultiTimeframeEngine
        from modules.risk_manager import DynamicRiskManager
        from modules.ai_agent     import MarketNarrator

        state       = await asyncio.to_thread(_fetch_tf_snapshot, _tf)
        technical   = state.get("technical", {})
        key_levels  = technical.get("key_levels", {})
        price       = float(state.get("price", 0) or 0)
        agent_data  = _cache.get(f"agent_full_{_tf}", _cache.get("agent_full_1h", {}))
        regime      = agent_data.get("market_regime", "neutral")
        pattern     = agent_data.get("dominant_pattern", "")
        atr         = float(technical.get("indicators", {}).get("atr", price * 0.015) or price * 0.015)

        engine   = MultiTimeframeEngine()
        risk_mg  = DynamicRiskManager()
        narrator = MarketNarrator()

        TF_TO_ANALYZE = {
            "1h": ["1h", "4h", "1d", "1w"],
            "4h": ["1h", "4h", "1d", "1w"],
            "1d": ["4h", "1d", "1w"],
            "1w": ["1d", "1w"],
        }
        mtf = await asyncio.to_thread(engine.get_mtf_bias, "BTCUSDT",
                                      TF_TO_ANALYZE.get(_tf, ["1h", "4h", "1d", "1w"]))
        entry_zone = engine.find_optimal_entry_zone(mtf, key_levels, price)
        direction  = "long" if mtf.get("global_bias") == "BULLISH" else "short"
        entry_price = float(entry_zone.get("zone_low", price) or price)

        stop = risk_mg.compute_stop_loss(
            entry=entry_price, direction=direction, atr=atr,
            key_levels={"pivot_points": key_levels.get("pivot_points", {})},
            pattern=pattern, regime=regime,
        )
        tps = risk_mg.compute_take_profits(
            entry=entry_price, stop_loss=stop.get("stop_loss", entry_price * 0.97),
            direction=direction,
            key_levels={"pivot_points": key_levels.get("pivot_points", {})},
            fibonacci=technical.get("fibonacci", {}),
        )
        signal_info = {
            "symbol": "BTCUSDT", "direction": direction,
            "pattern": pattern, "regime": regime,
            "setup_quality": mtf.get("entry_quality", "B"),
        }
        setup_text = narrator.generate_trade_setup(signal_info, entry_zone, stop, tps, mtf)

        result = _sanitize({
            "mtf": mtf, "entry_zone": entry_zone,
            "stop": stop, "tps": tps,
            "setup_text": setup_text, "signal": signal_info,
        })
        result_safe = json.loads(json.dumps(result, cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/altcoins/screen")
async def api_agent_altcoins_screen(interval: str = Query("1h"), top: int = Query(5)):
    """Screening complet altcoins — top picks + DCA plans + signal altseason."""
    cache_key = "agent_altcoins_screen"
    now = time.time()
    if cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < 300:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.altcoin_screener import AltcoinScreener
        screener  = AltcoinScreener()
        state     = _cache.get(f"full_{interval}_24h", _cache.get("full_1h_24h", {}))
        agent_data = _cache.get("agent_full", {})
        btc_state = {
            "btc_change_24h":  state.get("change", 0),
            "btc_change_7d":   state.get("technical", {}).get("btc_change_7d", 0),
            "btc_cycle_phase": agent_data.get("cycle_phase", "MID_BULL"),
            "total3_data":     agent_data.get("total3_data", {}),
        }
        cycle_phase = agent_data.get("cycle_phase", "MID_BULL")
        macro_score = float(state.get("macro", {}).get("score", 55) or 55)
        result = await asyncio.to_thread(screener.screen_all, btc_state, cycle_phase, macro_score, top)
        result_safe = json.loads(json.dumps(_sanitize(result), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/backtest/{strategy}")
async def api_agent_backtest(strategy: str,
                              symbol: str = Query("BTCUSDT"),
                              interval: str = Query("4h"),
                              lookback_days: int = Query(365),
                              capital: float = Query(10000)):
    """Walk-forward backtest — stratégies : trend_following, mean_reversion."""
    cache_key = f"agent_bt_{strategy}"
    now = time.time()
    if cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < 600:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.backtester import StrategyBacktester
        bt = StrategyBacktester()
        result = await asyncio.to_thread(
            bt.backtest_signal,
            strategy_name=strategy, symbol=symbol,
            interval=interval, lookback_days=lookback_days,
            initial_capital=capital,
        )
        result_safe = json.loads(json.dumps(_sanitize(result), cls=SafeEncoder))
        _cache[cache_key]         = result_safe
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result_safe)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/brief")
async def api_agent_brief(interval: str = Query("1h"), force: str = Query("0")):
    """Brief quotidien structuré en 7 sections. Cache 10 min par TF."""
    _tf    = interval
    _force = force == "1"
    cache_key = f"agent_brief_{_tf}"
    now = time.time()
    if not _force and cache_key in _cache and (now - _cache.get(f"ts_{cache_key}", 0)) < 600:
        return safe_jsonify(_cache[cache_key])
    try:
        from modules.ai_agent import MarketNarrator
        narrator = MarketNarrator()
        state = _cache.get(f"full_{_tf}_24h", _cache.get("full_1h_24h", {}))
        agent_data = _cache.get(f"agent_full_{_tf}", _cache.get("agent_full_1h", {}))
        full_state = {
            "market":           {"btc_price": state.get("price"), "btc_change_24h": state.get("change"),
                                 "btc_change_7d": state.get("technical", {}).get("btc_change_7d")},
            "market_structure": state.get("technical", {}),
            "macro":            state.get("macro", {}),
            "agent":            agent_data,
        }
        brief_text = await asyncio.to_thread(narrator.generate_daily_brief, full_state)
        result = {"brief": brief_text, "generated_at": datetime.now(timezone.utc).isoformat()}
        _cache[cache_key]         = result
        _cache[f"ts_{cache_key}"] = now
        return safe_jsonify(result)
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/agent/learn")
async def api_agent_learn():
    """Entraîne le MetaLearner + génère des insights depuis l'historique."""
    try:
        from modules.ai_agent import MetaLearner, PredictionTracker
        learner = MetaLearner()
        tracker = PredictionTracker()
        pred_log = tracker._load()
        train_result = await asyncio.to_thread(learner.train_meta_model, pred_log)
        perf_stats   = tracker.get_performance_stats()
        insights     = learner.generate_insights(pred_log, perf_stats)
        return safe_jsonify({
            "training":    train_result,
            "performance": perf_stats,
            "insights":    insights,
        })
    except Exception as e:
        import traceback
        return safe_jsonify({"error": str(e), "trace": traceback.format_exc()}, 500)


@app.get("/api/events")
async def api_events(limit: int = Query(50, ge=1, le=200)):
    """Timeline des événements système (signaux, anomalies, résolutions)."""
    events = _event_log[-limit:][::-1]
    return safe_jsonify({"count": len(events), "events": events})


@app.get("/api/portfolio")
async def api_portfolio(action: str = Query("stats"), position_id: str = Query(None)):
    """
    Gestion des positions simulées.
    ?action=stats          → statistiques globales
    ?action=positions      → toutes les positions
    ?action=close&position_id=<id>&price=<price>  → ferme une position
    """
    try:
        from modules.ai_agent import _portfolio_simulator
        if action == "close" and position_id:
            # Le prix courant est lu depuis le cache ticker
            ticker_cached = _cache.get("ticker_cached")
            current_price = float(ticker_cached["lastPrice"]) if ticker_cached else 0.0
            result = _portfolio_simulator.close_position(position_id, current_price)
            if result:
                log_event("PORTFOLIO", f"Position {position_id} fermée", result)
                return safe_jsonify({"closed": result})
            return safe_jsonify({"error": "Position non trouvée ou déjà fermée"}, 404)
        if action == "positions":
            return safe_jsonify({"positions": _portfolio_simulator._load()})
        # Défaut : stats
        stats = await asyncio.to_thread(_portfolio_simulator.get_stats)
        return safe_jsonify(stats)
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.websocket("/ws/price/{symbol}")
async def ws_price(websocket: WebSocket, symbol: str = "BTCUSDT"):
    """
    Proxy WebSocket Binance → dashboard.
    Envoie le prix en temps réel et notifie à la clôture de bougie.
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("[WS] Client connecté — symbol=%s total=%d", symbol, len(_ws_clients))

    binance_url = (
        f"wss://stream.binance.com:9443/stream?streams="
        f"{symbol.lower()}@kline_1m/"
        f"{symbol.lower()}@kline_4h/"
        f"{symbol.lower()}@ticker"
    )

    try:
        import ssl as _ssl
        _ssl_ctx = _ssl.create_default_context()
        _ssl_ctx.check_hostname = False
        _ssl_ctx.verify_mode    = _ssl.CERT_NONE
        async with _ws_lib.connect(binance_url, ssl=_ssl_ctx) as binance_ws:
            async for raw_msg in binance_ws:
                # Vérifier si le client est encore connecté
                try:
                    msg    = json.loads(raw_msg)
                    stream = msg.get("stream", "")
                    data   = msg.get("data", {})

                    if "ticker" in stream:
                        await websocket.send_json({
                            "type":   "price",
                            "price":  float(data.get("c", 0)),
                            "change": float(data.get("P", 0)),
                            "volume": float(data.get("q", 0)),
                            "symbol": symbol,
                        })

                    elif "kline" in stream:
                        kline = data.get("k", {})
                        if kline.get("x"):   # bougie clôturée
                            tf = kline.get("i", "1h")
                            logger.info("[WS] Bougie %s clôturée @ %s", tf, kline.get("c"))
                            # Invalide le cache pour ce TF
                            for hk in [f"full_{tf}_24h", f"full_{tf}_7d",
                                       f"full_{tf}_30d", f"full_{tf}_90d"]:
                                _cache.pop(hk, None)
                            await websocket.send_json({
                                "type":   "candle_close",
                                "tf":     tf,
                                "symbol": symbol,
                                "close":  float(kline.get("c", 0)),
                                "action": "refresh_analysis",
                            })
                except WebSocketDisconnect:
                    break
                except Exception as _send_err:
                    logger.debug("[WS] send error: %s", _send_err)
                    break

    except Exception as e:
        logger.warning("[WS] Connexion Binance perdue: %s", e)
    finally:
        _ws_clients.discard(websocket)
        logger.info("[WS] Client déconnecté — reste=%d", len(_ws_clients))


@app.get("/api/screener")
async def api_screener():
    """Top 50 Binance + détection d'anomalies (cache 5min)."""
    try:
        if DISKCACHE_OK and "screener_latest" in _disk_cache:
            return safe_jsonify(_disk_cache["screener_latest"])
        if SCREENER_OK:
            result = await run_screener_once()
            return safe_jsonify(result)
        return safe_jsonify({"error": "Screener non disponible"}, 503)
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.get("/api/inference/{symbol}")
async def api_inference_result(symbol: str):
    """Résultat de l'inférence Celery pour un token donné."""
    key = f"inference_{symbol.upper()}"
    if DISKCACHE_OK and key in _disk_cache:
        return safe_jsonify(_disk_cache[key])
    # Déclenche une inférence si le résultat n'est pas en cache
    try:
        from tasks import run_heavy_inference
        run_heavy_inference.delay(symbol=symbol.upper(), reason="api_request")
        return safe_jsonify({"status": "queued", "symbol": symbol.upper()})
    except Exception:
        return safe_jsonify({"status": "pending", "symbol": symbol.upper()})


@app.get("/api/memory/signals")
async def api_signals_history():
    """50 derniers signaux enregistrés en DuckDB."""
    if not MEMORY_OK:
        return safe_jsonify({"error": "DuckDB non disponible"}, 503)
    try:
        from modules.memory import get_db
        with get_db() as db:
            rows = db.execute("""
                SELECT id, ts, signal, alpha_score, price_at_signal,
                       regime, direction_correct_n1, sl_hit, tp1_hit,
                       llm_synthesis
                FROM signals ORDER BY ts DESC LIMIT 50
            """).fetchall()
        return safe_jsonify([
            dict(zip(["id","ts","signal","score","price","regime",
                      "correct","sl_hit","tp1_hit","synthesis"], r))
            for r in rows
        ])
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.get("/api/memory/trust")
async def api_trust_scores():
    """Trust scores actuels des modèles IA."""
    if not MEMORY_OK:
        return safe_jsonify({"error": "DuckDB non disponible"}, 503)
    try:
        from modules.memory import get_model_trust_scores
        return safe_jsonify(await asyncio.to_thread(get_model_trust_scores))
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.post("/api/memory/evaluate")
async def api_manual_evaluate():
    """Déclenche l'évaluation nocturne manuellement."""
    if not MEMORY_OK:
        return safe_jsonify({"error": "DuckDB non disponible"}, 503)
    try:
        from modules.memory import nightly_evaluation
        result = await nightly_evaluation()
        return safe_jsonify(result)
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.get("/api/scanner")
async def api_scanner(force: bool = Query(False)):
    """
    Lance ou retourne le dernier scan microcap.
    ?force=true  → force un nouveau scan (ignore le cache)
    """
    try:
        if _scanner is None:
            return safe_jsonify({"error": "Scanner non disponible (microcap_scanner non chargé)"}, 503)
        results  = await asyncio.to_thread(_scanner.scan, force)
        summary  = _scanner.get_summary()
        return safe_jsonify({"summary": summary, "tokens": results})
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.get("/api/scanner/token/{symbol}")
async def api_scanner_token(symbol: str):
    """Détail + score social d'un token."""
    try:
        if _scanner is None:
            return safe_jsonify({"error": "Scanner non disponible"}, 503)
        detail = await asyncio.to_thread(_scanner.get_token_detail, symbol.upper())
        return safe_jsonify(detail)
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.get("/api/alerts/test")
async def api_alerts_test():
    """Envoie un message Telegram de test (lecture directe du .env)."""
    def _do_test():
        from dotenv import dotenv_values
        env  = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
        token = env.get("TELEGRAM_BOT_TOKEN", "")
        chat  = env.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return {"success": False, "reason": "TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant dans .env"}
        import requests as _req
        r = _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat,
                  "text": "✅ <b>macro_alpha</b> — Telegram opérationnel",
                  "parse_mode": "HTML"},
            timeout=8,
        )
        return {"success": r.ok, "status": r.status_code,
                "chat_id": chat, "bot": token.split(":")[0]}
    try:
        result = await asyncio.to_thread(_do_test)
        return safe_jsonify(result)
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.get("/api/risk")
async def api_risk():
    """Statut du ProductionRiskManager."""
    try:
        if _prod_risk is None:
            return safe_jsonify({"error": "ProductionRiskManager non disponible"}, 503)
        return safe_jsonify(_prod_risk.get_status())
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


@app.get("/")
async def index():
    from fastapi.responses import HTMLResponse
    html = open("templates/dashboard.html", encoding="utf-8").read()
    return HTMLResponse(content=html)


# ══════════════════════════════════════════════════════════════
# ROUTES GEM SCANNER
# ══════════════════════════════════════════════════════════════

@app.get("/api/gems")
async def api_gems():
    """Top gems identifiées par le scanner — avec cache 10 min."""
    try:
        # Vérifier cache disque d'abord
        if _disk_cache is not None:
            cached = _disk_cache.get("scanner_latest")
            if cached and (time.time() - cached.get("ts_epoch", 0)) < 600:
                return safe_jsonify(cached)

        from modules.microcap_scanner import MicrocapScanner
        scanner = MicrocapScanner()
        result  = await asyncio.to_thread(scanner.run_scan, True)
        return safe_jsonify(result)
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.get("/api/gems/analyze/{symbol}")
async def api_gem_analyze(symbol: str, capital: float = Query(10000.0)):
    """Analyse complète d'une gem avec plan de trade DCA."""
    try:
        cache_key = f"full_1h_24h"
        full_data = _cache.get(cache_key, {})
        macro     = full_data.get("macro", {})
        signal    = full_data.get("alpha", {})

        from modules.microcap_scanner import MicrocapScanner
        scanner = MicrocapScanner()
        scan    = await asyncio.to_thread(scanner.run_scan, False)

        token = next(
            (t for t in scan.get("tokens", [])
             if t.get("symbol", "").upper() == symbol.upper()),
            None,
        )
        if not token:
            return safe_jsonify({"error": f"{symbol} non trouvé dans le scan"}, 404)

        from modules.ai_agent import GemSwingAgent
        agent  = GemSwingAgent()
        result = await asyncio.to_thread(
            agent.analyze_gem_opportunity,
            token,
            macro,
            {"alpha": signal},
            capital,
        )
        return safe_jsonify(result)
    except Exception as exc:
        import traceback
        return safe_jsonify({"error": str(exc), "trace": traceback.format_exc()}, 500)


@app.get("/api/scanner/test")
async def api_scanner_test():
    """Diagnostic connexions API (Binance, CoinGecko, DexScreener, Fear&Greed)."""
    import requests as _req

    tests = [
        ("binance",     "https://api.binance.com/api/v3/ticker/price", {"symbol": "BTCUSDT"}),
        ("coingecko",   "https://api.coingecko.com/api/v3/ping",       None),
        ("dexscreener", "https://api.dexscreener.com/latest/dex/pairs/solana/So11111111111111111111111111111111111111112", None),
        ("fear_greed",  "https://api.alternative.me/fng/?limit=1",     None),
        ("rugcheck",    "https://api.rugcheck.xyz/v1/tokens/So11111111111111111111111111111111111111112/report", None),
    ]

    results = {}
    for name, url, params in tests:
        try:
            r = _req.get(url, params=params, timeout=8,
                         headers={"User-Agent": "macro_alpha/5.0"})
            results[name] = {"ok": r.status_code == 200, "status": r.status_code}
            if name == "binance" and r.ok:
                results[name]["btc"] = r.json().get("price")
            if name == "fear_greed" and r.ok:
                fg = r.json()["data"][0]
                results[name]["value"] = fg["value"]
                results[name]["label"] = fg["value_classification"]
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)[:60]}

    return safe_jsonify({
        "all_ok":      all(v.get("ok") for v in results.values()),
        "connections": results,
        "ts":          datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════
# TWITTER SCANNER
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/twitter/scan")
async def api_twitter_scan():
    """Scan les tendances Twitter crypto."""
    def _do():
        from modules.twitter_scanner import TwitterScanner
        scanner = TwitterScanner()
        if scanner.mode == "unavailable":
            return {
                "error":    "Twitter scraping non disponible",
                "solution": "pip install ntscraper",
            }
        return scanner.scan_crypto_trends()
    result = await asyncio.to_thread(_do)
    return safe_jsonify(result)


@app.get("/api/twitter/account/{username}")
async def api_twitter_account(username: str):
    """Analyse les tokens mentionnés par un compte Twitter."""
    def _do():
        from modules.twitter_scanner import TwitterScanner
        scanner = TwitterScanner()
        return scanner.scan_target_account(username, limit=10)
    result = await asyncio.to_thread(_do)
    return safe_jsonify(result)


# ═══════════════════════════════════════════════════════════════════
# TRADING BOT (paper par défaut)
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/bot/account")
async def api_bot_account():
    """Informations du compte Binance (paper ou live)."""
    def _do():
        from modules.trading_bot import TradingBot
        return TradingBot().get_account_info()
    result = await asyncio.to_thread(_do)
    return safe_jsonify(result)


@app.get("/api/bot/orders")
async def api_bot_orders():
    """Positions ouvertes + PnL mis à jour."""
    def _do():
        from modules.trading_bot import TradingBot
        bot     = TradingBot()
        updated = bot.update_positions()
        return {
            "orders":      updated,
            "performance": bot.get_performance(),
            "mode":        "PAPER" if bot.dry_run else "LIVE",
        }
    result = await asyncio.to_thread(_do)
    return safe_jsonify(result)


@app.post("/api/bot/buy")
async def api_bot_buy(request: Request):
    """Place un ordre d'achat (paper ou live selon DRY_RUN)."""
    data    = await request.json()
    symbol  = data.get("symbol", "").upper()
    amount  = float(data.get("amount_usdt", 100))
    sl_pct  = float(data.get("sl_pct", 10))
    tp1_pct = float(data.get("tp1_pct", 15))

    if not symbol:
        return safe_jsonify({"error": "symbol requis"}, 400)

    def _do():
        from modules.trading_bot import TradingBot
        bot   = TradingBot()
        order = bot.place_order(
            symbol=symbol, side="buy",
            amount_usdt=amount,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            strategy=data.get("strategy", "GEM_SWING"),
        )
        return {
            "success": True,
            "order":   order.__dict__,
            "mode":    "PAPER" if bot.dry_run else "LIVE",
        }
    try:
        result = await asyncio.to_thread(_do)
        return safe_jsonify(result)
    except Exception as exc:
        return safe_jsonify({"error": str(exc)}, 500)


# ═══════════════════════════════════════════════════════════════════
# WHALE TRACKER
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/whales/{symbol}")
async def api_whales(symbol: str):
    """Analyse whale order book + trades pour un token."""
    from modules.whale_tracker import WhaleTracker
    tracker = WhaleTracker()
    ob, tr  = await asyncio.gather(
        tracker.detect_whale_orders_binance(symbol.upper()),
        tracker.analyze_recent_trades(symbol.upper(), 500),
    )
    await tracker.close()
    return safe_jsonify({
        "symbol":         symbol.upper(),
        "order_book":     ob,
        "trade_analysis": tr,
    })


@app.get("/api/whales/scan/top")
async def api_whales_scan():
    """Scan whale sur les top tokens du scanner."""
    cache_key = "whale_scan"
    cached    = _disk_cache.get(cache_key) if DISKCACHE_OK else None
    if cached and (time.time() - cached.get("ts_epoch", 0)) < 120:
        return safe_jsonify(cached)

    scanner_data = _disk_cache.get("scanner_latest") if DISKCACHE_OK else {}
    tokens = [
        t["symbol"] for t in (scanner_data or {}).get("tokens", [])[:20]
    ]
    if not tokens:
        tokens = ["BTC", "ETH", "SOL", "BNB", "ADA", "DOT",
                  "AVAX", "MATIC", "LINK", "UNI"]

    from modules.whale_tracker import WhaleTracker
    tracker = WhaleTracker()
    results = await tracker.scan_multiple_tokens(tokens)
    await tracker.close()

    result = {
        "whales":     results,
        "top_signal": results[0] if results else None,
        "ts":         datetime.now(timezone.utc).isoformat(),
        "ts_epoch":   time.time(),
    }
    if DISKCACHE_OK:
        _disk_cache.set(cache_key, result, expire=120)
    return safe_jsonify(result)


# ═══════════════════════════════════════════════════════════════════
# SIGNAL COMBINÉ : gem + whale + twitter + claude
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/combined/{symbol}")
async def api_combined_signal(symbol: str):
    """Signal composite : scanner gem + whale + twitter + Claude narrative."""
    sym = symbol.upper()

    # Données scanner en cache
    scan_data = _disk_cache.get("scanner_latest") if DISKCACHE_OK else {}
    token     = next(
        (t for t in (scan_data or {}).get("tokens", [])
         if t["symbol"] == sym),
        {"symbol": sym, "score": {}},
    )

    # Whale + twitter en parallèle (async)
    from modules.whale_tracker import WhaleTracker
    tracker = WhaleTracker()
    ob, tr  = await asyncio.gather(
        tracker.detect_whale_orders_binance(sym),
        tracker.analyze_recent_trades(sym, 200),
    )
    await tracker.close()

    # Twitter (sync dans thread)
    def _tw():
        from modules.twitter_scanner import TwitterScanner
        sc = TwitterScanner()
        if sc.mode == "unavailable":
            return {}
        return sc.scan_target_account(sym, limit=5)

    tw_data = await asyncio.to_thread(_tw)

    # Claude narrative
    from modules.claude_narrator import analyze_with_claude
    narrative = await analyze_with_claude(token, tw_data)

    # Score composite
    gem_sc   = token.get("score", {}).get("total_score", 0)
    whale_ob = ob.get("buy_pressure", 50)
    whale_tr = 80 if tr.get("pattern") == "WHALE_ACCUMULATION" else 40
    narr_sc  = narrative.get("score_narratif", 50)
    tw_sc    = min(100, tw_data.get("tokens_mentioned", {}).get(sym, 0) * 2)

    final_score = int(
        gem_sc   * 0.35 +
        whale_ob * 0.25 +
        whale_tr * 0.20 +
        narr_sc  * 0.15 +
        tw_sc    * 0.05
    )

    verdict = (
        "ROCKET GO"      if final_score >= 75 else
        "SURVEILLER"     if final_score >= 55 else
        "PASSER"
    )

    return safe_jsonify({
        "symbol":        sym,
        "final_score":   final_score,
        "verdict":       verdict,
        "gem_score":     gem_sc,
        "whale_signal":  ob.get("signal", "?"),
        "whale_pattern": tr.get("pattern", "?"),
        "buy_pressure":  whale_ob,
        "narrative":     narrative,
        "twitter":       tw_data,
        "timing":        narrative.get("timing", "?"),
        "conviction":    narrative.get("conviction", "?"),
        "ts":            datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════
# JUPITER SWAP — DEX Solana
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/jupiter/price/{symbol}")
async def api_jupiter_price(symbol: str):
    """Prix d'un token Solana via Jupiter."""
    from modules.jupiter_swap import JupiterSwap
    j = JupiterSwap()
    return safe_jsonify(await j.get_token_price(symbol))


@app.get("/api/jupiter/quote")
async def api_jupiter_quote(
    symbol: str = Query(..., description="Token à acheter (ex: SOL, BONK)"),
    amount: float = Query(100.0, description="Montant USDC à dépenser"),
    slippage: int = Query(100, description="Slippage en basis points (100 = 1%)"),
):
    """Obtient un quote Jupiter sans exécuter le swap."""
    from modules.jupiter_swap import JupiterSwap, USDC_MINT
    j    = JupiterSwap()
    mint = await j.resolve_token_mint(symbol.upper())
    if not mint:
        return safe_jsonify({"ok": False, "error": f"{symbol} non trouvé"}, 404)
    quote = await j.get_quote(
        input_mint=USDC_MINT,
        output_mint=mint,
        amount_usdc=amount,
        slippage_bps=slippage,
    )
    return safe_jsonify({**quote, "symbol": symbol.upper(), "amount_usdc": amount})


@app.post("/api/jupiter/buy")
async def api_jupiter_buy(request: Request):
    """
    Swap USDC → token via Jupiter.
    Paper trading si DRY_RUN=true (défaut).
    Body JSON : {"symbol": "BONK", "amount_usdc": 50, "slippage_bps": 100}
    """
    data        = await request.json()
    symbol      = data.get("symbol", "").upper()
    amount_usdc = float(data.get("amount_usdc", 50))
    slippage    = int(data.get("slippage_bps", 100))

    if not symbol:
        return safe_jsonify({"error": "symbol requis"}, 400)
    if amount_usdc < 1:
        return safe_jsonify({"error": "amount_usdc minimum 1"}, 400)

    from modules.jupiter_swap import JupiterSwap
    j      = JupiterSwap()
    result = await j.execute_swap(symbol, amount_usdc, slippage)
    return safe_jsonify(result)


@app.post("/api/jupiter/sell")
async def api_jupiter_sell(request: Request):
    """
    Vend des tokens → USDC via Jupiter.
    Body JSON : {"symbol": "BONK", "amount_tokens": 1000000, "slippage_bps": 150}
    """
    data          = await request.json()
    symbol        = data.get("symbol", "").upper()
    amount_tokens = float(data.get("amount_tokens", 0))
    slippage      = int(data.get("slippage_bps", 150))

    if not symbol or amount_tokens <= 0:
        return safe_jsonify({"error": "symbol et amount_tokens requis"}, 400)

    from modules.jupiter_swap import JupiterSwap
    j      = JupiterSwap()
    result = await j.execute_sell(symbol, amount_tokens, slippage)
    return safe_jsonify(result)


@app.get("/api/jupiter/portfolio")
async def api_jupiter_portfolio():
    """Solde du wallet Solana (USDC + tokens)."""
    from modules.jupiter_swap import JupiterSwap
    j = JupiterSwap()
    return safe_jsonify(await j.get_portfolio())


@app.get("/api/jupiter/tokens")
async def api_jupiter_tokens():
    """Liste des tokens Solana connus + résolution disponible."""
    from modules.jupiter_swap import KNOWN_TOKENS
    return safe_jsonify({
        "known_tokens": KNOWN_TOKENS,
        "count":        len(KNOWN_TOKENS),
        "note":         "Tout token Solana listé sur Jupiter est supporté",
    })


# ══════════════════════════════════════════════════════════════
# ROUTES SOLANA BOT AUTONOME
# ══════════════════════════════════════════════════════════════

_solana_bot = None

def _get_solana_bot():
    global _solana_bot
    if _solana_bot is None:
        from modules.solana_bot import SolanaBot
        _solana_bot = SolanaBot()
    return _solana_bot


@app.post("/api/solana-bot/start")
async def api_bot_start():
    """Démarrer le bot autonome."""
    bot = _get_solana_bot()
    result = await bot.start()
    return safe_jsonify(result)


@app.post("/api/solana-bot/stop")
async def api_bot_stop():
    """Arrêter le bot autonome."""
    bot = _get_solana_bot()
    result = await bot.stop()
    return safe_jsonify(result)


@app.get("/api/solana-bot/status")
async def api_bot_status():
    """Statut complet du bot : positions, perf, log."""
    bot = _get_solana_bot()
    return safe_jsonify(bot.get_status())


@app.post("/api/solana-bot/buy")
async def api_bot_manual_buy(request: Request):
    """Achat manuel via le bot."""
    data       = await request.json()
    symbol     = data.get("symbol", "").upper()
    amount     = float(data.get("amount_usdc", 10))
    bot        = _get_solana_bot()
    result     = await bot.manual_buy(symbol, amount)
    return safe_jsonify(result)


@app.post("/api/solana-bot/close/{position_id}")
async def api_bot_close(position_id: str):
    """Fermer une position manuellement."""
    bot    = _get_solana_bot()
    result = await bot.manual_close(position_id)
    return safe_jsonify(result)


@app.get("/api/solana-bot/live")
async def api_bot_live():
    """
    SSE — pousse le statut du bot toutes les 3 secondes.
    Le dashboard s'y connecte une fois et reçoit les updates en continu.
    """
    bot = _get_solana_bot()

    async def event_stream():
        import json as _json
        while True:
            try:
                status = bot.get_status()
                data   = _json.dumps(_sanitize(status), cls=SafeEncoder)
                yield f"data: {data}\n\n"
            except Exception as e:
                yield f"data: {{}}\n\n"
                logger.warning("sse_bot_error: %s", e)
            await asyncio.sleep(3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=False, log_level="info")
