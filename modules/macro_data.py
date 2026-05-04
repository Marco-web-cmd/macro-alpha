"""
macro_data.py
Récupère toutes les données macro depuis FRED (gratuit) et Yahoo Finance.
Calcule le Global Liquidity Index (proxy M2 amélioré).
v2 : Intégration du cycle 4 ans Bitcoin (halving avril 2024).

FRED séries utilisées :
  M2SL         — M2 Money Supply USA
  WALCL        — Fed Balance Sheet (total assets)
  WTREGEN      — TGA (Treasury General Account)
  RRPONTSYD    — Overnight Reverse Repo
  DFF          — Fed Funds Rate effectif
  DGS2         — US Treasury 2Y yield
  DGS10        — US Treasury 10Y yield
  T10Y2Y       — Spread 10Y-2Y (courbe des taux)
  DTWEXBGS     — Dollar Index (broad)
  BAMLH0A0HYM2 — High Yield spread (risk appetite)
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

_log = logging.getLogger("macro_data")

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from config import FRED_KEY

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

_macro_cache = {}
_cache_ts    = {}
CACHE_TTL    = 3600  # 1 heure


def _fetch_fred(series_id: str, limit: int = 200) -> pd.Series:
    """Fetch une série FRED, retourne une pd.Series indexée par date."""
    cache_key = f"{series_id}_{limit}"
    now = time.time()
    if cache_key in _macro_cache and (now - _cache_ts.get(cache_key, 0)) < CACHE_TTL:
        return _macro_cache[cache_key]

    params = {
        "series_id":         series_id,
        "api_key":           FRED_KEY,
        "file_type":         "json",
        "limit":             limit,
        "sort_order":        "desc",
        "observation_start": (datetime.now() - timedelta(days=limit * 7)).strftime("%Y-%m-%d")
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("observations", [])
        s = pd.Series(
            {d["date"]: float(d["value"]) for d in data if d["value"] != "."},
            name=series_id
        )
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        _macro_cache[cache_key] = s
        _cache_ts[cache_key] = now
        return s
    except Exception as e:
        _log.warning("[FRED] Erreur %s: %s", series_id, e)
        return pd.Series(name=series_id, dtype=float)


def _fetch_yahoo(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch une série Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        data = yf.Ticker(ticker).history(period=period)
        s = data["Close"].rename(ticker)
        s.index = s.index.tz_localize(None)
        return s
    except Exception as e:
        _log.warning("[Yahoo] Erreur %s: %s", ticker, e)
        return pd.Series(name=ticker, dtype=float)


# ─── CYCLE 4 ANS BITCOIN ─────────────────────────────────────────────────────

# Dates des halvings historiques
HALVING_DATES = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),
    datetime(2016, 7,   9, tzinfo=timezone.utc),
    datetime(2020, 5,  11, tzinfo=timezone.utc),
    datetime(2024, 4,  19, tzinfo=timezone.utc),   # ← référence actuelle
]
LAST_HALVING  = HALVING_DATES[-1]
NEXT_HALVING  = datetime(2028, 4, 19, tzinfo=timezone.utc)   # approximatif
PREV_ATH_DATE = datetime(2021, 11, 10, tzinfo=timezone.utc)
PREV_ATH_PRICE = 68789.0
CYCLE_DURATION = 1461   # ~4 ans en jours

def _fetch_btc_daily(limit: int = 730) -> pd.DataFrame:
    """Fetch données daily BTC depuis Binance (avec cache 1h)."""
    cache_key = f"btc_daily_{limit}"
    now = time.time()
    if cache_key in _macro_cache and (now - _cache_ts.get(cache_key, 0)) < CACHE_TTL:
        return _macro_cache[cache_key]
    try:
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(url, params={
            "symbol": "BTCUSDT", "interval": "1d", "limit": limit
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
        df = df[["open","high","low","close","volume"]]
        _macro_cache[cache_key] = df
        _cache_ts[cache_key] = now
        return df
    except Exception as e:
        _log.warning("[Binance Daily] Erreur: %s", e)
        return pd.DataFrame()


def _compute_bottom_indicators(df_daily: pd.DataFrame, current_price: float,
                                nli_chg: float, dxy_chg: float) -> dict:
    """
    Score multi-indicateurs de bottom de cycle (0-100%).
    Combinaison de 5 indicateurs indépendants pondérés par contexte macro.
    """
    if df_daily.empty or len(df_daily) < 50:
        return {"score": None, "indicators": {}, "available": False}

    close = df_daily["close"]
    high  = df_daily["high"]
    low   = df_daily["low"]
    volume = df_daily["volume"]
    signals = {}

    # ── 1. Prix vs coût de production des mineurs ──
    # Proxy : prix < 1.2x ~55k$ = zone sous coût de production
    MINER_COST_PROXY = 55000.0
    below_cost = current_price < MINER_COST_PROXY * 1.2
    price_ratio = current_price / MINER_COST_PROXY
    if below_cost:
        miner_score = max(0, min(100, (1.2 - price_ratio) / 0.4 * 100))
    else:
        miner_score = 0.0
    signals["miner_cost"] = {
        "score": round(miner_score, 1),
        "below_cost": below_cost,
        "price_ratio_to_cost": round(price_ratio, 2),
        "label": f"Prix {'sous' if below_cost else 'au-dessus'} du coût mineur (~{MINER_COST_PROXY/1000:.0f}k$)",
    }

    # ── 2. Compression Bollinger Band Width ──
    period = 20
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    bb_width = ((mid + 2*sigma - (mid - 2*sigma)) / mid * 100).dropna()
    if len(bb_width) >= 100:
        bbw_now = float(bb_width.iloc[-1])
        pctile  = float(np.sum(bb_width < bbw_now) / len(bb_width) * 100)
        bb_compressed = pctile <= 15
        bb_score = max(0, min(100, (15 - pctile) / 15 * 100)) if bb_compressed else 0.0
    else:
        bb_compressed = False
        bb_score = 0.0
        pctile = None
    signals["bb_compression"] = {
        "score":        round(bb_score, 1),
        "compressed":   bb_compressed,
        "bbw_percentile": round(pctile, 1) if pctile is not None else None,
        "label": f"BBW {f'{pctile:.0f}ème percentile' if pctile is not None else 'N/A'} — {'compression extrême' if bb_compressed else 'normal'}",
    }

    # ── 3. Stochastique mensuel ──
    monthly = close.resample("ME").last().dropna()
    monthly_high = high.resample("ME").max().reindex(monthly.index)
    monthly_low  = low.resample("ME").min().reindex(monthly.index)
    stoch_m_score = 0.0
    stoch_m_value = None
    if len(monthly) >= 6:
        lowest  = monthly_low.rolling(3).min()
        highest = monthly_high.rolling(3).max()
        raw_k   = ((monthly - lowest) / (highest - lowest) * 100)
        stoch_m = raw_k.rolling(3).mean()
        if not stoch_m.empty and not pd.isna(stoch_m.iloc[-1]):
            stoch_m_value = float(stoch_m.iloc[-1])
            if stoch_m_value < 20:
                stoch_m_score = (20 - stoch_m_value) / 20 * 100
    signals["stoch_monthly"] = {
        "score":     round(stoch_m_score, 1),
        "value":     round(stoch_m_value, 1) if stoch_m_value is not None else None,
        "in_bottom": stoch_m_value is not None and stoch_m_value < 20,
        "label": f"Stoch mensuel {f'{stoch_m_value:.0f}' if stoch_m_value is not None else 'N/A'} — {'zone BOTTOM' if stoch_m_value and stoch_m_value < 20 else 'neutre'}",
    }

    # ── 4. Pi Cycle Indicator ──
    pi_score = 0.0
    pi_ratio = None
    if len(close) >= 350:
        ema111   = close.ewm(span=111, adjust=False).mean()
        ema350   = close.ewm(span=350, adjust=False).mean()
        pi_ratio = float(ema111.iloc[-1]) / float(ema350.iloc[-1] * 2)
        if pi_ratio < 0.5:
            pi_score = (0.5 - pi_ratio) / 0.3 * 100
            pi_score = min(100, pi_score)
    signals["pi_cycle"] = {
        "score":     round(pi_score, 1),
        "ratio":     round(pi_ratio, 3) if pi_ratio is not None else None,
        "bottom_zone": pi_ratio is not None and pi_ratio < 0.5,
        "label": f"Pi Cycle ratio {f'{pi_ratio:.2f}' if pi_ratio is not None else 'N/A'} — {'zone BOTTOM' if pi_ratio and pi_ratio < 0.5 else 'neutre'}",
    }

    # ── 5. Score de capitulation : volume bas + volatilité compressée ──
    cap_score = 0.0
    if len(close) >= 60 and len(volume) >= 60:
        vol_now  = float(volume.iloc[-20:].mean())
        vol_hist = float(volume.iloc[-60:-20].mean())
        vol_ratio = vol_now / vol_hist if vol_hist > 0 else 1.0

        atr_now_c  = float((high - low).iloc[-20:].mean())
        atr_hist_c = float((high - low).iloc[-60:-20].mean())
        atr_ratio  = atr_now_c / atr_hist_c if atr_hist_c > 0 else 1.0

        if vol_ratio < 0.7 and atr_ratio < 0.7:
            cap_score = (1 - (vol_ratio + atr_ratio) / 2) / 0.3 * 50
            cap_score = min(100, cap_score)
        elif vol_ratio < 0.7 or atr_ratio < 0.7:
            cap_score = 30.0
    signals["capitulation"] = {
        "score":     round(cap_score, 1),
        "low_volume": cap_score > 0,
        "label": f"Capitulation {'détectée' if cap_score > 50 else 'partielle' if cap_score > 0 else 'absente'} (vol+vol compressés)",
    }

    # ── Agrégation pondérée ──
    weights = {
        "miner_cost":   0.25,
        "bb_compression": 0.20,
        "stoch_monthly": 0.25,
        "pi_cycle":     0.20,
        "capitulation": 0.10,
    }
    raw_score = sum(signals[k]["score"] * w for k, w in weights.items())

    # Pondération par contexte macro institutionnel
    # NLI en expansion + DXY en baisse = cycle potentiellement raccourci
    macro_boost = 0.0
    if nli_chg > 2.0:    macro_boost += 10
    elif nli_chg > 0.5:  macro_boost += 5
    if dxy_chg < -1.5:   macro_boost += 5
    elif dxy_chg > 1.5:  macro_boost -= 5

    final_score = min(100, max(0, raw_score + macro_boost))

    if final_score >= 70:
        interpretation = "BOTTOM_PROBABLE"
        interpretation_fr = "Bottom de cycle probable — convergence forte des indicateurs"
    elif final_score >= 45:
        interpretation = "BOTTOM_POSSIBLE"
        interpretation_fr = "Signaux de bottom partiels — surveiller"
    elif final_score >= 20:
        interpretation = "NEUTRAL"
        interpretation_fr = "Pas de signal de bottom clair"
    else:
        interpretation = "NO_BOTTOM"
        interpretation_fr = "Pas en zone de bottom"

    return {
        "available":       True,
        "score":           round(final_score, 1),
        "raw_score":       round(raw_score, 1),
        "macro_boost":     round(macro_boost, 1),
        "interpretation":  interpretation,
        "interpretation_fr": interpretation_fr,
        "indicators":      signals,
    }


def _fetch_market_data_for_cycle() -> dict:
    """
    Fetch données de marché via yfinance pour le cycle institutionnalisé.
    BTC/SPX correlation, VIX, Copper/Gold ratio.
    Cache 1h.
    """
    cache_key = "market_cycle_data"
    now = time.time()
    if cache_key in _macro_cache and (now - _cache_ts.get(cache_key, 0)) < CACHE_TTL:
        return _macro_cache[cache_key]

    result = {}
    try:
        import yfinance as yf
        data = yf.download(
            ['BTC-USD', '^GSPC', '^VIX', 'HG=F', 'GC=F'],
            period='3mo', progress=False, auto_adjust=True
        )
        close = data['Close']

        # BTC/SPX correlation 30J
        btc  = close['BTC-USD'].dropna()
        spx  = close['^GSPC'].dropna()
        idx  = btc.index.intersection(spx.index)
        if len(idx) >= 20:
            corr = float(btc.loc[idx].pct_change().dropna().rolling(30).corr(
                spx.loc[idx].pct_change().dropna()).iloc[-1])
            result["btc_spx_corr_30d"] = round(corr, 3)
        else:
            result["btc_spx_corr_30d"] = None

        # VIX niveau + tendance 30J
        vix = close['^VIX'].dropna()
        if not vix.empty:
            vix_now   = float(vix.iloc[-1])
            vix_30ago = float(vix.iloc[-min(30, len(vix))])
            result["vix_now"]   = round(vix_now, 1)
            result["vix_trend"] = round(vix_now - vix_30ago, 2)
        else:
            result["vix_now"]   = None
            result["vix_trend"] = None

        # Copper/Gold ratio (proxy risk appetite)
        copper = close['HG=F'].dropna()
        gold   = close['GC=F'].dropna()
        if not copper.empty and not gold.empty:
            idx2 = copper.index.intersection(gold.index)
            if len(idx2) >= 5:
                ratio_now  = float(copper.loc[idx2[-1]] / gold.loc[idx2[-1]])
                ratio_prev = float(copper.loc[idx2[-min(30, len(idx2))]] /
                                   gold.loc[idx2[-min(30, len(idx2))]])
                result["copper_gold_ratio"] = round(ratio_now * 1000, 4)   # scaled
                result["copper_gold_trend"] = round((ratio_now / ratio_prev - 1) * 100, 2)
            else:
                result["copper_gold_ratio"] = None
                result["copper_gold_trend"] = None
        else:
            result["copper_gold_ratio"] = None
            result["copper_gold_trend"] = None

    except Exception as e:
        _log.warning("[Market Cycle Data] yfinance error: %s", e)
        result.setdefault("btc_spx_corr_30d", None)
        result.setdefault("vix_now",          None)
        result.setdefault("vix_trend",         None)
        result.setdefault("copper_gold_ratio", None)
        result.setdefault("copper_gold_trend", None)

    # INDPRO (Industrial Production — proxy PMI expansion)
    try:
        indpro = _fetch_fred("INDPRO", 6)
        if len(indpro) >= 4:
            # 3 derniers mois en expansion ?
            last3   = indpro.iloc[-3:].values
            expanding = all(last3[i] > last3[i-1] for i in range(1, len(last3)))
            result["indpro_expanding_3m"] = bool(expanding)
            result["indpro_yoy"]          = round(float(
                (indpro.iloc[-1] / indpro.iloc[-13] - 1) * 100), 2) if len(indpro) >= 13 else None
        else:
            result["indpro_expanding_3m"] = None
            result["indpro_yoy"]          = None
    except Exception:
        result["indpro_expanding_3m"] = None
        result["indpro_yoy"]          = None

    _macro_cache[cache_key] = result
    _cache_ts[cache_key]    = now
    return result


def _dim_a_technical_onchain(df_daily: pd.DataFrame, current_price: float) -> dict:
    """
    Dimension A : Cycle Technique On-Chain (40% du score total).
    Pi Cycle, Stoch Monthly, BBW, proxy coût de production.
    """
    if df_daily.empty or len(df_daily) < 50:
        return {"score": 50, "signals": [], "available": False}

    close  = df_daily["close"]
    high   = df_daily["high"]
    low    = df_daily["low"]
    volume = df_daily["volume"]
    score  = 50.0
    sigs   = []

    # ── Pi Cycle ──
    if len(close) >= 350:
        ema111   = close.ewm(span=111, adjust=False).mean()
        ema350_2 = close.ewm(span=350, adjust=False).mean() * 2
        ratio = float(ema111.iloc[-1] / ema350_2.iloc[-1])
        diff_now  = float((ema111 - ema350_2).iloc[-1])
        diff_prev = float((ema111 - ema350_2).iloc[-2])
        top_cross = diff_prev < 0 and diff_now >= 0
        if top_cross:
            score -= 20
            sigs.append(f"Pi Cycle TOP croisement — signal de sommet majeur")
        elif ratio < 0.5:
            bonus = min(15, (0.5 - ratio) / 0.2 * 15)
            score += bonus
            sigs.append(f"Pi Cycle zone BOTTOM (ratio {ratio:.2f}) +{bonus:.0f}pts")
        elif ratio > 0.95:
            score -= 10
            sigs.append(f"Pi Cycle proche du TOP (ratio {ratio:.2f})")

    # ── Stochastique Monthly ──
    monthly      = close.resample("ME").last().dropna()
    monthly_high = high.resample("ME").max().reindex(monthly.index)
    monthly_low  = low.resample("ME").min().reindex(monthly.index)
    if len(monthly) >= 6:
        lowest  = monthly_low.rolling(3).min()
        highest = monthly_high.rolling(3).max()
        raw_k   = ((monthly - lowest) / (highest - lowest) * 100)
        stoch_m = raw_k.rolling(3).mean()
        k_now   = float(stoch_m.iloc[-1]) if not pd.isna(stoch_m.iloc[-1]) else 50.0
        k_prev  = float(stoch_m.iloc[-2]) if len(stoch_m) >= 2 and not pd.isna(stoch_m.iloc[-2]) else k_now
        if k_now < 20:
            bonus = (20 - k_now) / 20 * 15
            score += bonus
            sigs.append(f"Stoch mensuel bottom ({k_now:.0f}) +{bonus:.0f}pts")
        elif k_prev > 80 and k_now <= 80:
            score -= 12
            sigs.append(f"Stoch mensuel sortie zone surachat ({k_now:.0f})")
        elif k_now > 80:
            score -= 8
            sigs.append(f"Stoch mensuel suracheté ({k_now:.0f})")

    # ── BBW Percentile (500 jours) ──
    if len(close) >= 100:
        period = 20
        mid    = close.rolling(period).mean()
        sigma  = close.rolling(period).std()
        bbw    = ((mid + 2*sigma - (mid - 2*sigma)) / mid * 100).dropna()
        use_n  = min(len(bbw), 500)
        hist   = bbw.iloc[-use_n:]
        pctile = float(np.sum(hist < float(hist.iloc[-1])) / len(hist) * 100)
        if pctile <= 5:
            score += 8
            sigs.append(f"BBW {pctile:.0f}ème percentile sur {use_n}J — compression extrême")
        elif pctile >= 90:
            score -= 5
            sigs.append(f"BBW {pctile:.0f}ème percentile — expansion extrême")

    # ── Proxy coût de production mineurs ──
    MINER_COST = 58000.0
    below_cost = current_price < MINER_COST
    near_cost  = current_price < MINER_COST * 1.15
    if below_cost:
        bonus = min(15, (1 - current_price / MINER_COST) * 100)
        score += bonus
        sigs.append(f"BTC sous coût de production estimé ({MINER_COST/1000:.0f}k$) +{bonus:.0f}pts")
    elif near_cost:
        score += 5
        sigs.append(f"BTC proche du coût de production ({current_price/1000:.0f}k$)")

    score = max(0, min(100, score))
    return {"score": round(score, 1), "signals": sigs, "available": True}


def _dim_b_institutional(df_daily: pd.DataFrame, mkt: dict) -> dict:
    """
    Dimension B : Facteur Institutionnel ETF (25% du score).
    BTC/SPX correlation, volume pattern.
    """
    score = 50.0
    sigs  = []

    corr = mkt.get("btc_spx_corr_30d")
    institutional_factor = 1.0   # 1.0 = pas d'atténuation, < 1 = cycle atténué

    if corr is not None:
        if corr > 0.7:
            sigs.append(f"Corrélation BTC/SPX élevée ({corr:.2f}) — cycle institutionnalisé, tops/bottoms atténués")
            institutional_factor = 0.70   # tops/bottoms 30% moins extrêmes
            score -= 5   # marché moins autonome
        elif corr > 0.5:
            sigs.append(f"Corrélation BTC/SPX modérée ({corr:.2f}) — influence institutionnelle partielle")
            institutional_factor = 0.85
        elif corr is not None and corr < 0.3:
            sigs.append(f"Corrélation BTC/SPX faible ({corr:.2f}) — cycle crypto-natif dominant")
            institutional_factor = 1.0
            score += 5   # BTC plus autonome = cycles plus forts

    # Volume pattern : accumulation silencieuse
    if not df_daily.empty and len(df_daily) >= 20:
        close  = df_daily["close"]
        volume = df_daily["volume"]
        price_chg_20 = float(close.pct_change(20).iloc[-1]) * 100
        vol_now_avg  = float(volume.iloc[-5:].mean())
        vol_hist_avg = float(volume.iloc[-20:-5].mean())
        vol_ratio    = vol_now_avg / vol_hist_avg if vol_hist_avg > 0 else 1.0

        # Accumulation silencieuse : hausse modérée + volume faible
        if price_chg_20 > 0 and vol_ratio < 0.8:
            score += 8
            sigs.append(f"Accumulation institutionnelle silencieuse (+{price_chg_20:.1f}% sur 20J, vol -{(1-vol_ratio)*100:.0f}%)")

    score = max(0, min(100, score))
    return {"score": round(score, 1), "signals": sigs,
            "institutional_factor": institutional_factor,
            "btc_spx_corr": corr}


def _dim_c_macro_geo(bonds: dict, m2: dict, mkt: dict,
                      fed_rate_series: pd.Series) -> dict:
    """
    Dimension C : Sensibilité Macro-Géopolitique (20% du score).
    DXY, courbe taux, Fed pivot, PMI, VIX, Copper/Gold.
    """
    score = 50.0
    sigs  = []

    # ── DXY momentum ──
    dxy_chg = m2.get("dxy_1m_chg", 0) or 0
    if dxy_chg < -2.0:
        score += 10
        sigs.append(f"DXY -2%+/mois ({dxy_chg:.1f}%) — fuite vers actifs alternatifs")
    elif dxy_chg < -1.0:
        score += 5
        sigs.append(f"DXY en baisse ({dxy_chg:.1f}%) — favorable BTC")
    elif dxy_chg > 2.0:
        score -= 10
        sigs.append(f"DXY +2%+/mois ({dxy_chg:.1f}%) — dollar fort, défavorable BTC")

    # ── Courbe des taux ──
    curve = bonds.get("yield_curve", 0) or 0
    spread_chg = bonds.get("spread_1m_change", 0) or 0
    if curve > 0.5 and spread_chg > 0:
        score += 8
        sigs.append(f"Courbe normale en hausse ({curve:.2f}%, Δ+{spread_chg:.2f}%) — normalisation favorable")
    elif curve < 0:
        score -= 5
        sigs.append(f"Courbe inversée ({curve:.2f}%) — signal récession")

    # ── Fed pivot signal : taux stable ou en baisse depuis ≥2 meetings ──
    fed_pivot = False
    if not fed_rate_series.empty and len(fed_rate_series) >= 10:
        recent = fed_rate_series.iloc[-10:]
        if float(recent.iloc[-1]) <= float(recent.iloc[-5]):
            fed_pivot = True
            score += 8
            sigs.append(f"Fed pivot : taux stable/baissier — contexte favorable")

    # ── PMI proxy (INDPRO expansion) ──
    if mkt.get("indpro_expanding_3m"):
        score += 6
        sigs.append(f"Production industrielle en expansion 3 mois consécutifs")
    elif mkt.get("indpro_yoy") is not None and mkt["indpro_yoy"] < 0:
        score -= 4
        sigs.append(f"Production industrielle en contraction (YoY {mkt['indpro_yoy']:.1f}%)")

    # ── VIX ──
    vix = mkt.get("vix_now")
    if vix is not None:
        if vix < 20:
            score += 5
            sigs.append(f"VIX faible ({vix}) — marché serein, favorable BTC")
        elif vix > 30:
            score -= 10
            sigs.append(f"VIX élevé ({vix}) — stress systémique, défavorable BTC")
        elif vix > 25:
            score -= 4
            sigs.append(f"VIX modéré-élevé ({vix})")

    # ── Copper/Gold ratio ──
    cg_trend = mkt.get("copper_gold_trend")
    if cg_trend is not None:
        if cg_trend > 2:
            score += 5
            sigs.append(f"Copper/Gold ratio en hausse (+{cg_trend:.1f}%) — risk appetite élevé")
        elif cg_trend < -2:
            score -= 5
            sigs.append(f"Copper/Gold ratio en baisse ({cg_trend:.1f}%) — risk-off")

    score = max(0, min(100, score))
    return {"score": round(score, 1), "signals": sigs, "fed_pivot": fed_pivot,
            "vix": vix, "copper_gold_trend": cg_trend}


def _dim_d_adaptive_phase(days_since: int, inst_factor: float,
                           dim_c_score: float) -> dict:
    """
    Dimension D : Phase de Cycle Adaptative (15% du score).
    Garde le modèle post-halving mais pondère par les facteurs B et C.
    """
    if days_since < 180:
        phase = "ACCUMULATION"; phase_fr = "Accumulation post-halving"
        base_bonus = 8; color = "amber"
        pct_done = days_since / 180 * 100
    elif days_since < 540:
        phase = "BULL MARKET"; phase_fr = "Bull Market primaire"
        base_bonus = 18; color = "green"
        pct_done = (days_since - 180) / 360 * 100
    elif days_since < 900:
        phase = "DISTRIBUTION"; phase_fr = "Distribution / Sommet de cycle"
        base_bonus = -5; color = "amber"
        pct_done = (days_since - 540) / 360 * 100
    elif days_since < 1260:
        phase = "BEAR MARKET"; phase_fr = "Bear Market — préserver le capital"
        base_bonus = -18; color = "red"
        pct_done = (days_since - 900) / 360 * 100
    else:
        phase = "BOTTOM"; phase_fr = "Fond de cycle — opportunité long terme"
        base_bonus = 5; color = "amber"
        pct_done = (days_since - 1260) / 201 * 100

    # Atténuation institutionnelle
    adj_bonus = base_bonus * inst_factor

    # Bonus macro indépendant (dimension C très favorable)
    macro_bonus = 0.0
    if dim_c_score > 70:
        macro_bonus = min(15, (dim_c_score - 70) / 30 * 15)
    elif dim_c_score < 30:
        macro_bonus = max(-10, (dim_c_score - 30) / 30 * 10)

    total_bonus = adj_bonus + macro_bonus
    score = max(0, min(100, 50 + total_bonus))

    return {
        "score":         round(score, 1),
        "phase":         phase,
        "phase_fr":      phase_fr,
        "color":         color,
        "base_bonus":    round(base_bonus, 1),
        "adj_bonus":     round(adj_bonus, 1),
        "macro_bonus":   round(macro_bonus, 1),
        "phase_pct_done": round(min(100, pct_done), 1),
    }


def compute_btc_cycle(nli_chg: float = 0.0, dxy_chg: float = 0.0) -> dict:
    """
    Cycle BTC institutionnalisé — 4 dimensions (post-ETFs spot Jan 2024).

    A) Cycle Technique On-Chain  40% — Pi Cycle, Stoch Monthly, BBW, coût mineurs
    B) Facteur Institutionnel     25% — BTC/SPX corr, accumulation silencieuse
    C) Macro-Géopolitique         20% — DXY, courbe taux, Fed pivot, VIX, PMI, Cu/Au
    D) Phase Adaptative           15% — post-halving pondéré par B+C

    Score final → bottom_probability (0-100%)
    """
    now        = datetime.now(timezone.utc)
    days_since = (now - LAST_HALVING).days
    days_to_next = max(0, (NEXT_HALVING - now).days)
    progress_pct = min(100.0, round(days_since / CYCLE_DURATION * 100, 1))

    # Données
    df_daily      = _fetch_btc_daily(limit=730)
    current_price = float(df_daily["close"].iloc[-1]) if not df_daily.empty else 0.0
    mkt           = _fetch_market_data_for_cycle()
    bonds         = {}
    m2_data       = {}
    fed_rate_s    = pd.Series(dtype=float)
    try:
        bonds      = get_rates_and_bonds()
        m2_data    = get_m2_global()
        fed_rate_s = _fetch_fred("DFF", 15)
    except Exception:
        pass

    # ── Dimensions ──
    dim_a = _dim_a_technical_onchain(df_daily, current_price)
    dim_b = _dim_b_institutional(df_daily, mkt)
    dim_c = _dim_c_macro_geo(bonds, m2_data, mkt, fed_rate_s)
    dim_d = _dim_d_adaptive_phase(days_since, dim_b["institutional_factor"], dim_c["score"])

    # ── Score final pondéré ──
    bottom_probability = (
        dim_a["score"] * 0.40 +
        dim_b["score"] * 0.25 +
        dim_c["score"] * 0.20 +
        dim_d["score"] * 0.15
    )
    bottom_probability = round(min(100, max(0, bottom_probability)), 1)

    if bottom_probability >= 70:
        interpretation = "BOTTOM_PROBABLE"
        interpretation_fr = "Bottom de cycle probable — forte convergence des 4 dimensions"
    elif bottom_probability >= 50:
        interpretation = "BOTTOM_POSSIBLE"
        interpretation_fr = "Signaux de bottom partiels — surveillance accrue"
    elif bottom_probability >= 35:
        interpretation = "NEUTRAL"
        interpretation_fr = "Pas de signal de bottom clair — phase neutre"
    else:
        interpretation = "NO_BOTTOM"
        interpretation_fr = "Pas en zone de bottom"

    # Top 3 signaux les plus forts
    all_sigs = (
        [("A-ONCHAIN",     s) for s in dim_a["signals"]] +
        [("B-INSTITUTIONAL", s) for s in dim_b["signals"]] +
        [("C-MACRO",       s) for s in dim_c["signals"]]
    )
    top_signals = [s[1] for s in all_sigs[:3]]

    # score_bonus pour le macro score global (±18 pts max, atténué par facteur institutionnel)
    score_bonus = round(dim_d["adj_bonus"], 1)

    # Compatibilité avec l'ancien format (bottom_score)
    bottom_score = {
        "available":        True,
        "score":            bottom_probability,
        "interpretation":   interpretation,
        "interpretation_fr": interpretation_fr,
        "indicators": {
            "dim_a_technical":     {"score": dim_a["score"], "label": "Cycle Technique On-Chain"},
            "dim_b_institutional": {"score": dim_b["score"], "label": "Facteur Institutionnel ETF"},
            "dim_c_macro":         {"score": dim_c["score"], "label": "Macro-Géopolitique"},
            "dim_d_phase":         {"score": dim_d["score"], "label": "Phase Adaptative"},
        }
    }

    return {
        "days_since_halving":   days_since,
        "days_to_next_halving": days_to_next,
        "cycle_progress_pct":   progress_pct,
        "phase_progress_pct":   dim_d["phase_pct_done"],
        "phase":                dim_d["phase"],
        "phase_fr":             dim_d["phase_fr"],
        "score_bonus":          score_bonus,
        "color":                dim_d["color"],
        "halving_date":         LAST_HALVING.strftime("%Y-%m-%d"),
        "next_halving_approx":  NEXT_HALVING.strftime("%Y-%m-%d"),
        "prev_ath_price":       PREV_ATH_PRICE,
        "bottom_probability":   bottom_probability,
        "interpretation_fr":    interpretation_fr,
        "btc_spx_corr":         dim_b.get("btc_spx_corr"),
        "vix":                  dim_c.get("vix"),
        "fed_pivot":            dim_c.get("fed_pivot"),
        "copper_gold_trend":    dim_c.get("copper_gold_trend"),
        "top_signals":          top_signals,
        "dimensions": {
            "A_technical":     dim_a,
            "B_institutional": dim_b,
            "C_macro":         dim_c,
            "D_phase":         dim_d,
        },
        "bottom_score":         bottom_score,   # rétrocompat
    }


# ─── PROXY DE LIQUIDITÉ GLOBALE ───────────────────────────────────────────────

def compute_net_liquidity() -> dict:
    """
    Net Liquidity Index (NLI) de la Fed.
    NLI = Fed Balance Sheet - TGA - Overnight Reverse Repo
    Corrélation historique très élevée avec BTC (~0.85 sur 2020-2024)
    """
    fed_bs = _fetch_fred("WALCL",     52)
    tga    = _fetch_fred("WTREGEN",   52)
    rrp    = _fetch_fred("RRPONTSYD", 52)

    if fed_bs.empty:
        return {"error": "FRED non disponible", "nli": None, "nli_change_4w": None}

    df = pd.DataFrame({"fed_bs": fed_bs, "tga": tga, "rrp": rrp})
    df = df.resample("W").last().ffill().dropna()

    df["nli"]         = df["fed_bs"] - df["tga"] - df["rrp"]
    df["nli_pct_4w"]  = df["nli"].pct_change(4) * 100
    df["nli_pct_13w"] = df["nli"].pct_change(13) * 100

    latest = df.iloc[-1]
    prev   = df.iloc[-4] if len(df) >= 4 else df.iloc[0]

    trend = "EXPANSION"   if latest["nli_pct_4w"] >  1.0 else \
            "CONTRACTION" if latest["nli_pct_4w"] < -1.0 else "NEUTRE"

    return {
        "nli_current":    round(float(latest["nli"]), 1),
        "nli_4w_ago":     round(float(prev["nli"]), 1),
        "nli_change_4w":  round(float(latest["nli_pct_4w"]), 2),
        "nli_change_13w": round(float(latest["nli_pct_13w"]), 2),
        "fed_bs":         round(float(latest["fed_bs"]), 1),
        "tga":            round(float(latest["tga"]), 1),
        "rrp":            round(float(latest["rrp"]), 1),
        "trend":          trend,
        "history":        {str(k): v for k, v in df["nli"].tail(52).items()},
        "last_updated":   datetime.now(timezone.utc).isoformat()
    }


def get_rates_and_bonds() -> dict:
    """Taux d'intérêt et obligations US. Spread 10Y-2Y = signal récession."""
    fed_rate  = _fetch_fred("DFF",          30)
    y2        = _fetch_fred("DGS2",         60)
    y10       = _fetch_fred("DGS10",        60)
    spread    = _fetch_fred("T10Y2Y",       60)
    hy_spread = _fetch_fred("BAMLH0A0HYM2", 60)

    result = {}

    if not fed_rate.empty:
        result["fed_rate"]      = round(float(fed_rate.iloc[-1]), 2)
        result["fed_rate_prev"] = round(float(fed_rate.iloc[-5]), 2) if len(fed_rate) >= 5 else None

    if not y2.empty and not y10.empty:
        result["yield_2y"]       = round(float(y2.iloc[-1]), 2)
        result["yield_10y"]      = round(float(y10.iloc[-1]), 2)
        result["yield_curve"]    = round(float(y10.iloc[-1] - y2.iloc[-1]), 2)
        result["curve_inverted"] = result["yield_curve"] < 0
        result["y2_history"]     = {str(k): v for k, v in y2.tail(30).items()}
        result["y10_history"]    = {str(k): v for k, v in y10.tail(30).items()}

    if not spread.empty:
        result["spread_10y2y"]     = round(float(spread.iloc[-1]), 2)
        result["spread_1m_change"] = round(float(spread.iloc[-1] - spread.iloc[-22]), 2) if len(spread) >= 22 else None

    if not hy_spread.empty:
        result["hy_spread"]       = round(float(hy_spread.iloc[-1]), 2)
        result["hy_spread_trend"] = "HAUSSE" if hy_spread.iloc[-1] > hy_spread.iloc[-10] else "BAISSE"
        result["risk_appetite"]   = "RISK-OFF" if hy_spread.iloc[-1] > 4.5 else \
                                    "RISK-ON"  if hy_spread.iloc[-1] < 3.0 else "NEUTRE"

    result["last_updated"] = datetime.now(timezone.utc).isoformat()
    return result


def get_m2_global() -> dict:
    """
    M2 USA + proxy global via DXY inverse.
    M2 global (BCE + Fed + BoJ + PBoC) = moteur principal des bull markets crypto.
    """
    m2_usa = _fetch_fred("M2SL",     36)
    dxy    = _fetch_fred("DTWEXBGS", 60)

    result = {}

    if not m2_usa.empty:
        result["m2_usa_current"] = round(float(m2_usa.iloc[-1]), 1)
        result["m2_usa_yoy"]     = round(float((m2_usa.iloc[-1] / m2_usa.iloc[-13] - 1) * 100), 2) if len(m2_usa) >= 13 else None
        yoy = result.get("m2_usa_yoy") or 0
        result["m2_usa_trend"]   = "EXPANSION" if yoy > 2 else \
                                   "CONTRACTION" if yoy < -1 else "STABLE"
        result["m2_history"]     = {str(k): v for k, v in m2_usa.tail(24).items()}
        # Momentum M2 : accélération récente
        if len(m2_usa) >= 7:
            m2_3m_ago = float(m2_usa.iloc[-4]) if len(m2_usa) >= 4 else m2_usa.iloc[-1]
            result["m2_momentum"] = round(float((m2_usa.iloc[-1] / m2_3m_ago - 1) * 100), 2)
        else:
            result["m2_momentum"] = None

    if not dxy.empty:
        dxy_now    = float(dxy.iloc[-1])
        dxy_1m_ago = float(dxy.iloc[-22]) if len(dxy) >= 22 else dxy_now
        dxy_3m_ago = float(dxy.iloc[-66]) if len(dxy) >= 66 else dxy_now
        result["dxy_current"] = round(dxy_now, 2)
        result["dxy_1m_chg"]  = round((dxy_now / dxy_1m_ago - 1) * 100, 2)
        result["dxy_3m_chg"]  = round((dxy_now / dxy_3m_ago - 1) * 100, 2)
        result["dxy_signal"]  = "HAUSSIER BTC" if result["dxy_1m_chg"] < -1.5 else \
                                "BAISSIER BTC" if result["dxy_1m_chg"] > 1.5 else "NEUTRE"
        result["dxy_history"] = {str(k): v for k, v in dxy.tail(60).items()}

    result["last_updated"] = datetime.now(timezone.utc).isoformat()
    return result


def compute_macro_score() -> dict:
    """
    Score macro global de 0 à 100.
    100 = conditions macro parfaites pour BTC
    0   = conditions macro maximalement défavorables

    Composantes :
      NLI Fed       ±20 pts   (liquidité injectée dans le système)
      Cycle BTC     ±18 pts   (phase du cycle 4 ans)
      Courbe taux   ±10 pts   (normale vs inversée)
      DXY           ±10 pts   (dollar faible = haussier BTC)
      HY Spread     ±10 pts   (risk-on vs risk-off)
      M2 YoY        ±10 pts   (expansion monétaire)
    """
    liquidity = compute_net_liquidity()
    bonds     = get_rates_and_bonds()
    m2        = get_m2_global()
    nli_chg   = liquidity.get("nli_change_4w", 0) or 0
    dxy_chg   = m2.get("dxy_1m_chg", 0) or 0
    cycle     = compute_btc_cycle(nli_chg=nli_chg, dxy_chg=dxy_chg)

    score   = 50.0
    signals = []

    # ── Liquidité nette Fed (±20 pts) ──
    nli_chg = liquidity.get("nli_change_4w", 0) or 0
    if nli_chg > 2:
        score += 20; signals.append(("LIQUIDITY", "+", "NLI expansion forte (+{:.1f}%)".format(nli_chg)))
    elif nli_chg > 0.5:
        score += 10; signals.append(("LIQUIDITY", "+", "NLI expansion modérée"))
    elif nli_chg < -2:
        score -= 20; signals.append(("LIQUIDITY", "-", "NLI contraction forte ({:.1f}%)".format(nli_chg)))
    elif nli_chg < -0.5:
        score -= 10; signals.append(("LIQUIDITY", "-", "NLI contraction modérée"))
    else:
        signals.append(("LIQUIDITY", "=", "NLI stable"))

    # ── Cycle 4 ans Bitcoin (±18 pts) ──
    cycle_bonus = cycle.get("score_bonus", 0)
    score += cycle_bonus
    phase = cycle.get("phase", "INCONNU")
    days_h = cycle.get("days_since_halving", 0)
    if cycle_bonus > 0:
        signals.append(("CYCLE", "+", f"Cycle BTC: {phase} — J+{days_h} post-halving"))
    elif cycle_bonus < 0:
        signals.append(("CYCLE", "-", f"Cycle BTC: {phase} — J+{days_h} post-halving"))
    else:
        signals.append(("CYCLE", "=", f"Cycle BTC: {phase} — J+{days_h} post-halving"))

    # Bonus M2 + cycle convergents
    m2_mom = m2.get("m2_momentum", 0) or 0
    if cycle["phase"] == "BULL MARKET" and m2_mom > 1.5:
        score += 5
        signals.append(("CYCLE_M2", "+", f"M2 momentum ({m2_mom:+.1f}%) renforce phase BULL"))
    elif cycle["phase"] == "BEAR MARKET" and m2_mom < -1.0:
        score -= 5
        signals.append(("CYCLE_M2", "-", f"M2 momentum ({m2_mom:+.1f}%) renforce phase BEAR"))

    # ── Courbe des taux (±10 pts) ──
    curve = bonds.get("yield_curve", 0) or 0
    if curve > 0.5:
        score += 10; signals.append(("BONDS", "+", "Courbe normale ({:.2f}%)".format(curve)))
    elif curve < 0:
        score -= 10; signals.append(("BONDS", "-", "Courbe inversée ({:.2f}%) — signal récession".format(curve)))
    else:
        signals.append(("BONDS", "=", "Courbe plate ({:.2f}%)".format(curve)))

    # ── DXY (±10 pts) ──
    dxy_chg = m2.get("dxy_1m_chg", 0) or 0
    if dxy_chg < -1.5:
        score += 10; signals.append(("DXY", "+", "Dollar en baisse ({:.2f}%)".format(dxy_chg)))
    elif dxy_chg > 1.5:
        score -= 10; signals.append(("DXY", "-", "Dollar en hausse ({:.2f}%)".format(dxy_chg)))
    else:
        signals.append(("DXY", "=", "Dollar stable"))

    # ── High Yield spread (±10 pts) ──
    risk = bonds.get("risk_appetite", "NEUTRE")
    if risk == "RISK-ON":
        score += 10; signals.append(("CREDIT", "+", "HY spread faible → risk-on"))
    elif risk == "RISK-OFF":
        score -= 10; signals.append(("CREDIT", "-", "HY spread élevé → risk-off"))
    else:
        signals.append(("CREDIT", "=", "Appétit au risque neutre"))

    # ── M2 YoY (±10 pts) ──
    m2_yoy = m2.get("m2_usa_yoy", 0) or 0
    if m2_yoy > 5:
        score += 10; signals.append(("M2", "+", "M2 USA expansion forte ({:.1f}% YoY)".format(m2_yoy)))
    elif m2_yoy > 2:
        score += 5;  signals.append(("M2", "+", "M2 USA expansion modérée"))
    elif m2_yoy < -1:
        score -= 10; signals.append(("M2", "-", "M2 USA contraction ({:.1f}% YoY)".format(m2_yoy)))
    else:
        signals.append(("M2", "=", "M2 USA stable"))

    score = max(0, min(100, score))

    label = "TRÈS HAUSSIER" if score >= 80 else \
            "HAUSSIER"      if score >= 65 else \
            "NEUTRE"        if score >= 45 else \
            "BAISSIER"      if score >= 30 else "TRÈS BAISSIER"

    return {
        "score":        round(score, 1),
        "label":        label,
        "signals":      signals,
        "liquidity":    liquidity,
        "bonds":        bonds,
        "m2":           m2,
        "cycle":        cycle,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
