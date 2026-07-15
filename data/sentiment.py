"""
Phase 7 — Autonomous Reddit Sentiment Ingestion.

Scrapes r/Forex and r/algotrading using Reddit's public JSON API
(no auth required — public posts only). Extracts posts mentioning
each currency pair, then uses Haiku to produce a per-pair sentiment
score (-10 to +10 pts) that feeds into the confluence scorer.

Refresh cadence: every 2 hours (cheap — Haiku + 1 API call).
Cache: in-memory, refreshed by the scheduler.

Score mapping:
  -10 to -5  = strongly bearish sentiment   (blocks weak longs)
   -4 to  0  = neutral/slightly bearish
    1 to  4  = slightly bullish
    5 to 10  = strongly bullish             (boosts high-score longs)

Cost: ~$0.002/refresh × 12/day = ~$0.024/day = ~$0.72/month
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger


# ── Reddit scraping ───────────────────────────────────────────────────────────

_SUBREDDITS = ["Forex", "algotrading", "Forex_technical"]
_USER_AGENT  = "ARIA:sentiment-reader:1.0 (research bot)"
_FETCH_LIMIT = 30   # posts per subreddit per fetch


def _fetch_reddit_posts(subreddit: str) -> list[dict]:
    """Fetch newest posts from a subreddit using Reddit's public JSON API."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={_FETCH_LIMIT}"
        r = httpx.get(url, headers={"User-Agent": _USER_AGENT}, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        return [p["data"] for p in posts if p.get("data")]
    except Exception as e:
        logger.debug(f"[Sentiment] Reddit fetch error ({subreddit}): {e}")
        return []


def _posts_mentioning_pair(posts: list[dict], pair: str) -> list[str]:
    """Filter posts that mention the given currency pair or its currencies."""
    pair_base = pair.upper().rstrip("M")
    if len(pair_base) < 6:
        return []

    base  = pair_base[:3]   # e.g. EUR
    quote = pair_base[3:6]  # e.g. USD
    keywords = {pair_base, base, quote, f"{base}/{quote}", f"{base}{quote}"}

    results = []
    for post in posts:
        title = post.get("title", "")
        selftext = post.get("selftext", "")[:500]
        combined = f"{title} {selftext}".upper()
        if any(kw in combined for kw in keywords):
            results.append(f"[{post.get('score', 0)}↑] {title[:120]}")
    return results


# ── AI sentiment analysis ─────────────────────────────────────────────────────

_SENTIMENT_PROMPT = """You are a forex sentiment analyst. Rate the following Reddit posts about {pair} on a scale from -10 to +10:
  -10 = extremely bearish (everyone expects {pair} to fall)
   0  = neutral/mixed
  +10 = extremely bullish (everyone expects {pair} to rise)

Posts:
{posts}

Reply with ONLY a JSON object: {{"score": <number -10 to 10>, "summary": "<15 words max>"}}"""


def _ai_sentiment(pair: str, posts: list[str]) -> tuple[float, str]:
    """Use Haiku to score sentiment of Reddit posts about a pair."""
    if not posts:
        return 0.0, "No relevant posts"

    try:
        import anthropic
        from config.settings import settings

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        posts_text = "\n".join(f"- {p}" for p in posts[:10])

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": _SENTIMENT_PROMPT.format(pair=pair, posts=posts_text),
            }],
        )
        import json
        raw = msg.content[0].text.strip()
        data = json.loads(raw)
        score   = float(data.get("score", 0.0))
        summary = data.get("summary", "")
        return max(-10.0, min(10.0, score)), summary
    except Exception as e:
        logger.debug(f"[Sentiment] AI scoring error: {e}")
        return 0.0, "AI error"


# ── Cache ─────────────────────────────────────────────────────────────────────

class _SentimentEntry:
    __slots__ = ("score", "summary", "fetched_at")

    def __init__(self, score: float, summary: str) -> None:
        self.score      = score
        self.summary    = summary
        self.fetched_at = datetime.now(timezone.utc)


class SentimentCache:
    """Thread-safe per-pair sentiment cache with 2h TTL."""

    REFRESH_INTERVAL_S = 7_200   # 2 hours

    def __init__(self) -> None:
        self._lock: threading.Lock   = threading.Lock()
        self._cache: dict[str, _SentimentEntry] = {}
        self._last_full_refresh: float = 0.0
        self._posts_cache: list[dict] = []  # raw posts reused across pairs

    # ── Public API ────────────────────────────────────────────────────────────

    def get_pts(self, pair: str) -> float:
        """Return sentiment score in points (-10 to +10). 0 if not cached yet."""
        with self._lock:
            entry = self._cache.get(pair)
        if entry is None:
            return 0.0
        return entry.score

    def get_summary(self, pair: str) -> str:
        with self._lock:
            entry = self._cache.get(pair)
        return entry.summary if entry else ""

    def all_scores(self) -> dict[str, float]:
        with self._lock:
            return {p: e.score for p, e in self._cache.items()}

    def refresh(self, pairs: list[str]) -> None:
        """Fetch Reddit posts for all pairs and score sentiment. Called by scheduler."""
        logger.info(f"[Sentiment] Refreshing for {len(pairs)} pairs")
        now = time.time()

        # Fetch raw posts once, reuse across pairs
        all_posts: list[dict] = []
        for sub in _SUBREDDITS:
            all_posts.extend(_fetch_reddit_posts(sub))
        self._last_full_refresh = now

        if not all_posts:
            logger.warning("[Sentiment] No posts fetched — Reddit may be throttling")
            return

        logger.info(f"[Sentiment] Fetched {len(all_posts)} posts from {len(_SUBREDDITS)} subs")

        for pair in pairs:
            relevant = _posts_mentioning_pair(all_posts, pair)
            score, summary = _ai_sentiment(pair, relevant)
            with self._lock:
                self._cache[pair] = _SentimentEntry(score, summary)
            logger.debug(f"[Sentiment] {pair}: {score:+.1f} — {summary} ({len(relevant)} posts)")

    def is_stale(self) -> bool:
        return time.time() - self._last_full_refresh > self.REFRESH_INTERVAL_S


# Singleton
sentiment_cache = SentimentCache()
