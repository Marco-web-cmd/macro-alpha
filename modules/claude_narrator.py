"""
claude_narrator.py — Analyse narrative des tokens via Claude API (haiku, ~$0.001/analyse).
Fallback sans LLM si ANTHROPIC_API_KEY absent.
"""
import os
import json
import asyncio
import httpx
import structlog

log = structlog.get_logger()

CLAUDE_URL = "https://api.anthropic.com/v1/messages"


async def analyze_with_claude(token_data: dict, twitter_data: dict = None) -> dict:
    """
    Analyse narrative d'un token via claude-haiku-4-5.
    Retourne verdict JSON : narrative, bull_case, bear_case, timing, score_narratif, conviction.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _template_fallback(token_data)

    symbol    = token_data.get("symbol", "?")
    price     = token_data.get("price", 0)
    change_24h = token_data.get("change_24h", 0)
    volume    = token_data.get("volume_24h", 0)
    mc        = token_data.get("market_cap", 0)
    score_obj = token_data.get("score", {})
    gem_score = score_obj.get("total_score", 0)
    breakout  = score_obj.get("breakout_data", {})

    twitter_context = ""
    if twitter_data:
        mentions = twitter_data.get("tokens_mentioned", {}).get(symbol, 0)
        if mentions > 0:
            twitter_context = (
                f"\nContexte Twitter : Score de mention {mentions}/100. "
                f"Sentiment : {twitter_data.get('sentiment', '?')}."
            )

    prompt = f"""Tu es un analyste crypto spécialisé en altcoins à faible market cap.
Analyse ce token en 3 lignes maximum. Sois direct et factuel.

Token : ${symbol}
Prix : ${price:.4f}
Variation 24H : {change_24h:+.1f}%
Volume 24H : ${volume/1e6:.1f}M
Market Cap : ${mc/1e6:.1f}M (0 si non listé CoinGecko)
Score technique : {gem_score}/100
Breakout détecté : {breakout.get('is_breaking_out', False)}
Signaux : {', '.join(breakout.get('breakout_signals', [])[:3])}{twitter_context}

Réponds UNIQUEMENT en JSON valide :
{{
  "narrative": "description courte du token et de sa thèse (1 phrase)",
  "bull_case": "raison principale d'être haussier (1 phrase)",
  "bear_case": "risque principal (1 phrase)",
  "timing": "ENTRER_MAINTENANT | ATTENDRE_PULLBACK | TROP_TARD | EVITER",
  "score_narratif": <int 0-100>,
  "conviction": "FORTE | MODEREE | FAIBLE"
}}"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                CLAUDE_URL,
                headers={
                    "x-api-key":          api_key,
                    "anthropic-version":  "2023-06-01",
                    "content-type":       "application/json",
                },
                json={
                    "model":      "claude-haiku-4-5",
                    "max_tokens": 300,
                    "messages":   [{"role": "user", "content": prompt}],
                }
            )
        content = r.json()["content"][0]["text"]
        start   = content.find("{")
        end     = content.rfind("}") + 1
        data    = json.loads(content[start:end])
        log.info("claude_analysis_ok", symbol=symbol, score=data.get("score_narratif"))
        return data
    except Exception as e:
        log.warning("claude_analysis_failed", symbol=symbol, error=str(e))
        return _template_fallback(token_data)


def _template_fallback(token_data: dict) -> dict:
    chg   = token_data.get("change_24h", 0)
    score = token_data.get("score", {}).get("total_score", 0)
    return {
        "narrative":      f"Token avec momentum {'+' if chg > 0 else ''}{chg:.1f}% 24H",
        "bull_case":      "Volume et momentum positifs" if chg > 0 else "Zone de support potentielle",
        "bear_case":      "Volatilité élevée — position sizing réduit",
        "timing":         "ENTRER_MAINTENANT" if score >= 75 else "ATTENDRE_PULLBACK",
        "score_narratif": min(100, int(score * 0.8)),
        "conviction":     "FORTE" if score >= 80 else "MODEREE" if score >= 65 else "FAIBLE",
        "source":         "fallback",
    }
