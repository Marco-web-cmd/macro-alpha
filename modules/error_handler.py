"""
error_handler.py — Gestion d'erreurs classifiée avec retry (tenacity).
"""
import logging
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)


class TradingAppError(Exception):
    """Erreur de base de l'application."""
    pass

class NetworkError(TradingAppError):
    """Problème réseau / API externe."""
    pass

class DataError(TradingAppError):
    """Données manquantes ou invalides."""
    pass

class ModelError(TradingAppError):
    """Erreur des modèles IA."""
    pass

class CacheError(TradingAppError):
    """Problème de cache."""
    pass


def safe_fetch(
    fn: Callable,
    fallback: Any = None,
    error_class: Type[TradingAppError] = NetworkError,
    max_attempts: int = 3,
) -> Any:
    """
    Wrapper avec retry exponentiel (via tenacity) et logging.
    Retourne fallback si toutes les tentatives échouent.
    """
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.5, min=1, max=5),
            reraise=False,
        )
        def _inner():
            return fn()

        try:
            return _inner()
        except RetryError as e:
            logger.warning(f"safe_fetch: all {max_attempts} attempts failed — {e}")
            return fallback
    except ImportError:
        # Fallback sans tenacity
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:
                logger.warning(f"safe_fetch attempt {attempt+1}/{max_attempts}: {exc}")
        return fallback
    except Exception as exc:
        logger.error(f"safe_fetch[{error_class.__name__}]: {exc}")
        return fallback
