"""
llm_narrator.py — Synthèse stratégique en langage naturel via LLM.

Priorité :
1. Ollama local (Llama 3.1 8B) — gratuit, privé, rapide
2. Claude API (Anthropic)      — si ANTHROPIC_API_KEY configuré
3. Fallback template-based    — toujours disponible sans LLM
"""
import os
import httpx
import logging

logger = logging.getLogger("llm_narrator")


class LLMNarrator:
    """Génère une synthèse en 3 lignes du signal alpha courant."""

    OLLAMA_URL   = "http://localhost:11434/api/generate"
    CLAUDE_URL   = "https://api.anthropic.com/v1/messages"
    OLLAMA_MODEL = "llama3.1:8b"

    def __init__(self):
        self.mode = self._detect_mode()
        logger.info("[LLM] mode=%s", self.mode)

    def _detect_mode(self) -> str:
        """Détecte quel backend LLM est disponible."""
        try:
            import requests as _r
            r = _r.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                if any("llama" in m.lower() for m in models):
                    return "ollama"
        except Exception:
            pass
        if os.getenv("ANTHROPIC_API_KEY"):
            return "claude"
        return "template"

    async def generate_synthesis(self, analysis: dict) -> str:
        """
        Génère une synthèse de 3 lignes.
        Retourne TOUJOURS quelque chose même si le LLM est down.
        """
        prompt = self._build_prompt(analysis)

        if self.mode == "ollama":
            result = await self._call_ollama(prompt)
        elif self.mode == "claude":
            result = await self._call_claude(prompt)
        else:
            result = self._template_fallback(analysis)

        return result[:300] if result else self._template_fallback(analysis)

    def _build_prompt(self, analysis: dict) -> str:
        alpha   = analysis.get("alpha", {})
        macro   = analysis.get("macro", {})
        tech    = analysis.get("technical", {})
        signal  = alpha.get("signal", "NEUTRE")
        score   = alpha.get("alpha_score", 50)
        price   = analysis.get("price", 0)
        regime  = alpha.get("regime_detected", "inconnu")
        cycle   = macro.get("cycle", {}).get("phase", "?")
        nli     = macro.get("liquidity", {}).get("nli_change_4w")
        patterns = [p["name"] for p in tech.get("patterns", [])[:2]]
        rw       = alpha.get("regime_weights", {})

        nli_str = f"{nli:+.1f}%/4w" if nli is not None else "N/A"
        return (
            "Tu es un analyste crypto institutionnel senior. "
            "En 3 lignes maximum, explique pourquoi ce signal est généré. "
            "Sois direct, précis, professionnel. Pas de jargon inutile.\n\n"
            f"Données :\n"
            f"- Signal : {signal} (score {score}/100)\n"
            f"- Prix BTC : ${price:,.0f}\n"
            f"- Régime : {regime}\n"
            f"- Phase cycle : {cycle}\n"
            f"- NLI Fed : {nli_str}\n"
            f"- Patterns : {', '.join(patterns) if patterns else 'Aucun'}\n"
            f"- Poids macro : {rw.get('macro', 0)*100:.0f}% | "
            f"tech : {rw.get('tech', 0)*100:.0f}%\n\n"
            "Synthèse (3 lignes max) :"
        )

    async def _call_ollama(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    self.OLLAMA_URL,
                    json={
                        "model":  self.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 150},
                    },
                )
            result = r.json().get("response", "").strip()
            logger.info("[LLM] Ollama OK (%d chars)", len(result))
            return result
        except Exception as e:
            logger.warning("[LLM] Ollama failed: %s", e)
            self.mode = "claude" if os.getenv("ANTHROPIC_API_KEY") else "template"
            return self._template_fallback({})

    async def _call_claude(self, prompt: str) -> str:
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    self.CLAUDE_URL,
                    headers={
                        "x-api-key":        api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      "claude-haiku-4-5-20251001",
                        "max_tokens": 200,
                        "messages":   [{"role": "user", "content": prompt}],
                    },
                )
            result = r.json()["content"][0]["text"].strip()
            logger.info("[LLM] Claude OK (%d chars)", len(result))
            return result
        except Exception as e:
            logger.warning("[LLM] Claude failed: %s", e)
            return self._template_fallback({})

    def _template_fallback(self, analysis: dict) -> str:
        """Fallback template — toujours disponible, sans LLM."""
        alpha   = analysis.get("alpha", {})
        signal  = alpha.get("signal", "NEUTRE")
        score   = alpha.get("alpha_score", 50)
        regime  = alpha.get("regime_detected", "transition")
        rw      = alpha.get("regime_weights", {})

        dominant = max(
            {"Macro": rw.get("macro", 0.33),
             "Technique": rw.get("tech", 0.33),
             "Forecast IA": rw.get("forecast", 0.34)},
            key=lambda k: rw.get(k.lower().split()[0], 0.33),
        )

        templates = {
            "LONG FORT": (
                f"Signal haussier fort ({score}/100) en régime {regime}. "
                f"La couche {dominant} domine l'analyse ({rw.get(dominant.lower().split()[0], 0)*100:.0f}% de pondération). "
                f"Conditions favorables à une position longue."
            ),
            "LONG": (
                f"Biais haussier modéré ({score}/100). "
                f"Régime {regime} — {dominant} confirme la direction. "
                f"Entrée possible avec taille réduite."
            ),
            "NEUTRE": (
                f"Signal neutre ({score}/100) — régime {regime}. "
                f"Divergence entre les couches d'analyse. "
                f"Attendre confirmation avant de s'exposer."
            ),
            "SHORT": (
                f"Biais baissier modéré ({score}/100). "
                f"Régime {regime} — {dominant} indique une pression vendeuse. "
                f"Réduction d'exposition recommandée."
            ),
            "SHORT FORT": (
                f"Signal baissier fort ({score}/100) en régime {regime}. "
                f"Convergence défavorable sur {dominant}. "
                f"Préserver le capital — sortir ou shorter."
            ),
        }
        return templates.get(signal, templates["NEUTRE"])


# Instance globale
_narrator = LLMNarrator()


async def generate_llm_synthesis(analysis: dict) -> str:
    """Point d'entrée public — génère la synthèse LLM."""
    return await _narrator.generate_synthesis(analysis)
