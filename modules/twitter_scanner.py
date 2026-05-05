"""
twitter_scanner.py — Scraping Twitter/X sans API payante.
Mode A : ntscraper (aucun compte requis)
Mode B : twikit (compte Twitter gratuit requis)
"""
import asyncio
import json
import re
import time
import os
from datetime import datetime, timezone
from diskcache import Cache
import structlog

log   = structlog.get_logger()
cache = Cache("./data/cache")


class TwitterScanner:
    """
    Scraper Twitter/X sans API officielle.
    Mode A : ntscraper (nitter, sans auth)
    Mode B : twikit (avec compte Twitter)
    """

    def __init__(self):
        self.mode = self._detect_mode()
        log.info("twitter_scanner_init", mode=self.mode)

    def _detect_mode(self) -> str:
        try:
            import ntscraper  # noqa
            return "ntscraper"
        except ImportError:
            pass
        try:
            import twikit  # noqa
            return "twikit"
        except ImportError:
            pass
        return "unavailable"

    # ── MODE A : ntscraper ───────────────────────────────────────

    def scrape_tweets_ntscraper(self, query: str, limit: int = 20) -> list:
        try:
            from ntscraper import Nitter
            scraper = Nitter(log_level=0, skip_instance_check=False)
            tweets  = scraper.get_tweets(query, mode="term", number=limit)
            result  = []
            for t in (tweets.get("tweets", []) or []):
                result.append({
                    "text":      t.get("text", ""),
                    "user":      t.get("user", {}).get("username", "?"),
                    "followers": t.get("user", {}).get("followers", 0),
                    "likes":     t.get("stats", {}).get("likes", 0),
                    "retweets":  t.get("stats", {}).get("retweets", 0),
                    "date":      t.get("date", ""),
                    "link":      t.get("link", ""),
                    "source":    "ntscraper",
                })
            return result
        except Exception as e:
            log.warning("ntscraper_failed", query=query, error=str(e))
            return []

    # ── MODE B : twikit ──────────────────────────────────────────

    async def scrape_tweets_twikit(self, query: str, limit: int = 20) -> list:
        try:
            import twikit
            username = os.getenv("TWITTER_USERNAME", "")
            email    = os.getenv("TWITTER_EMAIL", "")
            password = os.getenv("TWITTER_PASSWORD", "")

            if not all([username, email, password]):
                log.warning("twikit_no_credentials")
                return []

            client       = twikit.Client("en-US")
            cookies_file = "data/twitter_cookies.json"

            if os.path.exists(cookies_file):
                client.load_cookies(cookies_file)
            else:
                await client.login(
                    auth_info_1=username,
                    auth_info_2=email,
                    password=password,
                )
                client.save_cookies(cookies_file)

            tweets = await client.search_tweet(query, "Latest", count=limit)
            result = []
            for t in tweets:
                result.append({
                    "text":      t.full_text or t.text or "",
                    "user":      t.user.screen_name,
                    "followers": t.user.followers_count,
                    "likes":     t.favorite_count,
                    "retweets":  t.retweet_count,
                    "date":      str(t.created_at),
                    "link":      f"https://twitter.com/{t.user.screen_name}/status/{t.id}",
                    "source":    "twikit",
                })
            return result
        except Exception as e:
            log.error("twikit_failed", error=str(e))
            return []

    # ── Interface unifiée ────────────────────────────────────────

    def get_tweets(self, query: str, limit: int = 20) -> list:
        cache_key = f"tweets_{query}_{limit}"
        cached    = cache.get(cache_key)
        if cached:
            return cached

        result = []
        if self.mode == "ntscraper":
            result = self.scrape_tweets_ntscraper(query, limit)
        elif self.mode == "twikit":
            result = asyncio.run(self.scrape_tweets_twikit(query, limit))

        if result:
            cache.set(cache_key, result, expire=300)
        return result

    # ── Analyse des mentions de tokens ──────────────────────────

    def extract_token_mentions(self, tweets: list) -> dict:
        mentions  = {}
        excluded  = {"BTC", "ETH", "USD", "USDT", "SOL", "BNB", "USDC",
                     "DAI", "WBTC", "WETH", "NFT", "DeFi", "AI", "CEO"}

        for tweet in tweets:
            text      = tweet.get("text", "")
            followers = tweet.get("followers", 0)
            likes     = tweet.get("likes", 0)
            retweets  = tweet.get("retweets", 0)

            weight = 1.0
            if followers > 100_000: weight = 5.0
            elif followers > 10_000: weight = 3.0
            elif followers > 1_000:  weight = 1.5

            engagement = likes + retweets * 2
            if engagement > 100:
                weight *= 1.5

            symbols = re.findall(r'\$([A-Z]{2,10})', text.upper())
            symbols = [s for s in symbols if s not in excluded]

            for sym in set(symbols):
                mentions[sym] = mentions.get(sym, 0) + weight

        if mentions:
            max_score = max(mentions.values())
            mentions  = {k: round(v / max_score * 100, 1) for k, v in mentions.items()}

        return dict(sorted(mentions.items(), key=lambda x: x[1], reverse=True))

    def scan_target_account(self, username: str, limit: int = 10) -> dict:
        tweets = self.get_tweets(f"from:{username}", limit)
        if not tweets:
            tweets = self.get_tweets(username, limit)

        mentions  = self.extract_token_mentions(tweets)
        sentiment = self._analyze_sentiment_simple(tweets)

        return {
            "account":          username,
            "tweets":           len(tweets),
            "tokens_mentioned": mentions,
            "top_token":        max(mentions, key=mentions.get) if mentions else None,
            "sentiment":        sentiment,
            "raw_tweets":       tweets[:5],
            "ts":               datetime.now(timezone.utc).isoformat(),
        }

    def _analyze_sentiment_simple(self, tweets: list) -> str:
        bull_words = ["bullish", "pump", "moon", "breakout", "buy",
                      "long", "accumulate", "gem", "potential", "strong"]
        bear_words = ["bearish", "dump", "short", "sell", "crash",
                      "weak", "avoid", "rug", "scam", "exit"]
        bull = bear = 0
        for t in tweets:
            text = t.get("text", "").lower()
            bull += sum(w in text for w in bull_words)
            bear += sum(w in text for w in bear_words)
        if bull > bear * 1.5:   return "BULLISH"
        elif bear > bull * 1.5: return "BEARISH"
        else:                   return "NEUTRE"

    def scan_crypto_trends(self) -> dict:
        queries = [
            "crypto altcoin gem",
            "altcoin breakout 2026",
        ]
        all_mentions: dict = {}
        for q in queries:
            tweets   = self.get_tweets(q, limit=15)
            mentions = self.extract_token_mentions(tweets)
            for sym, score in mentions.items():
                all_mentions[sym] = all_mentions.get(sym, 0) + score
            time.sleep(2)

        if all_mentions:
            mx = max(all_mentions.values())
            all_mentions = {k: round(v / mx * 100, 1) for k, v in all_mentions.items()}

        return {
            "trending_tokens": dict(
                sorted(all_mentions.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "mode": self.mode,
            "ts":   datetime.now(timezone.utc).isoformat(),
        }
