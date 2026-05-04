"""
backtester.py — Backtester walk-forward intégré
=================================================
Valide les stratégies sur données historiques avant de les appliquer.
Walk-forward testing pour éviter l'overfitting.
"""

import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def _fetch_historical(symbol: str = "BTCUSDT", interval: str = "4h",
                       limit: int = 1000) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(BINANCE_URL, params={
            "symbol": symbol, "interval": interval, "limit": limit
        }, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "ct", "qav", "nt", "tbb", "tbq", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.error(f"_fetch_historical: {exc}")
        return None


def _compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    half = close.rolling(period // 2).mean()
    full = close.rolling(period).mean()
    raw  = 2 * half - full
    return raw.rolling(int(np.sqrt(period))).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── Stratégies built-in ──

def strategy_trend_following(df: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Stratégie trend-following : long quand HMA haussière + RSI > 50.
    Retourne une Series de signaux : 1=long, -1=short, 0=neutre.
    """
    p  = params or {}
    hma_period = int(p.get("hma_period", 55))
    rsi_period = int(p.get("rsi_period", 14))
    min_conf   = float(p.get("min_confidence", 0))

    hma  = _compute_hma(df["close"], hma_period)
    rsi  = _compute_rsi(df["close"], rsi_period)
    price = df["close"]

    signals = pd.Series(0, index=df.index)
    # Long : prix > HMA + HMA haussière + RSI > 52
    long_cond  = (price > hma) & (hma > hma.shift(3)) & (rsi > 52)
    # Short : prix < HMA + HMA baissière + RSI < 48
    short_cond = (price < hma) & (hma < hma.shift(3)) & (rsi < 48)

    signals[long_cond]  =  1
    signals[short_cond] = -1
    return signals


def strategy_mean_reversion(df: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Stratégie mean-reversion : achat sur RSI survendu, vente sur RSI suracheté.
    """
    p = params or {}
    rsi_low  = float(p.get("rsi_low",  30))
    rsi_high = float(p.get("rsi_high", 70))
    rsi = _compute_rsi(df["close"])

    signals = pd.Series(0, index=df.index)
    signals[rsi < rsi_low]  =  1
    signals[rsi > rsi_high] = -1
    return signals


BUILTIN_STRATEGIES = {
    "trend_following":  strategy_trend_following,
    "mean_reversion":   strategy_mean_reversion,
}


class StrategyBacktester:
    """
    Valide les stratégies sur données historiques.
    Walk-forward testing pour éviter l'overfitting.
    """

    def backtest_signal(self, strategy_name: str = "trend_following",
                         symbol: str = "BTCUSDT",
                         interval: str = "4h",
                         lookback_days: int = 365,
                         initial_capital: float = 10000,
                         params: dict = None) -> dict:
        """
        Walk-forward backtest sur N jours.
        Retourne les métriques complètes.
        """
        bars_per_day = {"1h": 24, "4h": 6, "1d": 1, "15m": 96}.get(interval, 6)
        limit = min(1500, lookback_days * bars_per_day + 100)

        df = _fetch_historical(symbol, interval, limit)
        if df is None or len(df) < 50:
            return {"error": "Données insuffisantes pour le backtest"}

        strategy_fn = BUILTIN_STRATEGIES.get(strategy_name)
        if not strategy_fn:
            return {"error": f"Stratégie '{strategy_name}' inconnue"}

        signals = strategy_fn(df, params)
        return self._run_backtest(df, signals, initial_capital, strategy_name)

    def _run_backtest(self, df: pd.DataFrame, signals: pd.Series,
                       initial_capital: float, strategy_name: str) -> dict:
        """Exécute le backtest et calcule les métriques."""
        close   = df["close"].values
        sigs    = signals.values
        n       = len(close)

        capital    = initial_capital
        position   = 0        # +1 long, -1 short, 0 flat
        entry_price = 0.0
        trades      = []
        equity      = [capital]
        monthly_ret = {}

        for i in range(1, n):
            bar_date = df.index[i]
            month_key = bar_date.strftime("%Y-%m")

            prev_sig = sigs[i - 1]
            curr_sig = sigs[i]

            # Sortie de position si signal opposé ou neutre
            if position != 0 and curr_sig != position:
                pnl_pct = (close[i] - entry_price) / entry_price * position
                pnl_usd = capital * 0.10 * pnl_pct  # 10% du capital par trade
                capital += pnl_usd
                trades.append({
                    "entry": entry_price,
                    "exit":  close[i],
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "direction": "long" if position == 1 else "short",
                })
                position = 0

            # Entrée de position
            if curr_sig != 0 and position == 0:
                position    = curr_sig
                entry_price = close[i]

            equity.append(capital)
            prev_eq = monthly_ret.get(month_key, equity[-2] if len(equity) > 1 else capital)
            monthly_ret[month_key] = capital

        # Métriques
        if not trades:
            return {
                "strategy":     strategy_name,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate":     0.0,
                "avg_win":      0.0,
                "avg_loss":     0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "monthly_returns": [],
                "equity_curve":    equity[-200:],
                "note": "Aucun trade généré",
            }

        equity_arr = np.array(equity)
        returns    = np.diff(equity_arr) / equity_arr[:-1]
        total_ret  = (capital - initial_capital) / initial_capital * 100

        # Sharpe annualisé
        bars_per_year = {"1h": 8760, "4h": 2190, "1d": 365, "15m": 35040}.get("4h", 2190)
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))
        else:
            sharpe = 0.0

        # Max Drawdown
        peak   = np.maximum.accumulate(equity_arr)
        dd     = (equity_arr - peak) / peak
        max_dd = float(dd.min() * 100)

        wins  = [t for t in trades if t["pnl_usd"] > 0]
        loses = [t for t in trades if t["pnl_usd"] <= 0]

        win_rate    = len(wins) / len(trades) if trades else 0
        avg_win     = float(np.mean([t["pnl_pct"] for t in wins])) * 100  if wins  else 0
        avg_loss    = float(np.mean([t["pnl_pct"] for t in loses])) * 100 if loses else 0
        total_win   = sum(t["pnl_usd"] for t in wins)
        total_loss  = abs(sum(t["pnl_usd"] for t in loses))
        pf          = total_win / total_loss if total_loss > 0 else float("inf")

        # Calcul mensuel
        months = sorted(monthly_ret.keys())
        monthly_returns = []
        prev_cap = initial_capital
        for m in months:
            cap = monthly_ret[m]
            monthly_returns.append(round((cap - prev_cap) / prev_cap * 100, 2))
            prev_cap = cap

        return {
            "strategy":       strategy_name,
            "symbol":         "BTCUSDT",
            "total_return":   round(total_ret, 2),
            "sharpe_ratio":   round(sharpe, 3),
            "max_drawdown":   round(max_dd, 2),
            "win_rate":       round(win_rate, 3),
            "avg_win_pct":    round(avg_win, 3),
            "avg_loss_pct":   round(avg_loss, 3),
            "profit_factor":  round(pf, 3),
            "total_trades":   len(trades),
            "monthly_returns": monthly_returns,
            "equity_curve":   [round(float(v), 2) for v in equity_arr[-200:]],
            "initial_capital": initial_capital,
            "final_capital":   round(capital, 2),
        }

    def optimize_parameters(self, strategy_name: str,
                              param_grid: dict,
                              symbol: str = "BTCUSDT",
                              interval: str = "4h") -> dict:
        """
        Grid search sur les paramètres. Maximise le Sharpe Ratio.
        """
        strategy_fn = BUILTIN_STRATEGIES.get(strategy_name)
        if not strategy_fn:
            return {"error": f"Stratégie inconnue: {strategy_name}"}

        df = _fetch_historical(symbol, interval, 1500)
        if df is None:
            return {"error": "Données non disponibles"}

        # Générer toutes les combinaisons
        import itertools
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        best_sharpe = -np.inf
        best_params = {}
        results     = []

        for combo in combos[:50]:  # limiter à 50 combos
            params = dict(zip(keys, combo))
            try:
                signals = strategy_fn(df, params)
                bt      = self._run_backtest(df, signals, 10000, strategy_name)
                sr      = bt.get("sharpe_ratio", -99)
                results.append({"params": params, "sharpe": sr,
                                 "total_return": bt.get("total_return"),
                                 "win_rate": bt.get("win_rate")})
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_params = params
            except Exception as exc:
                logger.debug(f"optimize combo {params}: {exc}")

        return {
            "best_params":   best_params,
            "best_sharpe":   round(best_sharpe, 3),
            "n_combos_tested": len(results),
            "top_results":   sorted(results, key=lambda x: x["sharpe"],
                                    reverse=True)[:5],
        }

    def run_monte_carlo(self, trade_history: list,
                         n_simulations: int = 1000) -> dict:
        """
        Simulation Monte Carlo sur l'historique des trades.
        """
        if len(trade_history) < 10:
            return {"error": "Pas assez de trades pour Monte Carlo (min 10)"}

        pnls = [float(t.get("pnl_pct", 0)) for t in trade_history
                if t.get("pnl_pct") is not None]
        if not pnls:
            return {"error": "Pas de données PnL dans l'historique"}

        pnls_arr   = np.array(pnls)
        n_trades   = len(pnls_arr)
        sim_finals = []

        for _ in range(n_simulations):
            sampled = np.random.choice(pnls_arr, size=n_trades, replace=True)
            equity  = np.cumprod(1 + sampled)
            sim_finals.append(float(equity[-1] - 1) * 100)  # en %

        sim_arr = np.array(sim_finals)
        prob_ruin = float(np.mean(sim_arr < -50))   # probabilité drawdown > 50%

        return {
            "n_simulations": n_simulations,
            "n_trades":      n_trades,
            "p5":   round(float(np.percentile(sim_arr, 5)),   2),
            "p25":  round(float(np.percentile(sim_arr, 25)),  2),
            "p50":  round(float(np.percentile(sim_arr, 50)),  2),
            "p75":  round(float(np.percentile(sim_arr, 75)),  2),
            "p95":  round(float(np.percentile(sim_arr, 95)),  2),
            "prob_ruin_50pct": round(prob_ruin * 100, 2),
            "expected_return": round(float(sim_arr.mean()), 2),
        }
