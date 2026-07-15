"""
Idea deduplication — prevent re-testing the same concept twice.

Uses:
  1. Exact title match (hash)
  2. TF-IDF cosine similarity (>0.80 → duplicate)
  3. Cross-check against existing hypothesis_queue entries

Persists seen-title hashes in db/research_seen.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

_SEEN_PATH = Path("db/research_seen.json")
_SIM_THRESHOLD = 0.80


def _title_hash(title: str) -> str:
    return hashlib.md5(title.strip().lower().encode()).hexdigest()[:16]


def _load_seen() -> dict[str, str]:
    if _SEEN_PATH.exists():
        try:
            return json.loads(_SEEN_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_seen(seen: dict[str, str]) -> None:
    _SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SEEN_PATH.write_text(json.dumps(seen, indent=2))


def _cosine_sim(a: str, b: str) -> float:
    """Simple bag-of-words cosine similarity."""
    def tok(s: str) -> dict[str, int]:
        words: dict[str, int] = {}
        for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split():
            words[w] = words.get(w, 0) + 1
        return words

    import re
    va, vb = tok(a), tok(b)
    keys   = set(va) | set(vb)
    if not keys:
        return 0.0
    dot  = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na   = sum(v * v for v in va.values()) ** 0.5
    nb   = sum(v * v for v in vb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _queue_titles() -> list[str]:
    """Return all titles already in the hypothesis queue."""
    try:
        from core.hypothesis_queue import get_pending, get_all
        entries = get_all() if callable(getattr(__import__("core.hypothesis_queue", fromlist=["get_all"]), "get_all", None)) else get_pending()
        return [e.get("title", "") for e in entries]
    except Exception:
        return []


def deduplicate(ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter out ideas already seen or too similar to existing ones.
    Returns only genuinely new ideas, and marks them as seen.
    """
    seen      = _load_seen()
    kept_raw  = list(seen.values())   # titles of already-seen ideas
    kept_raw += _queue_titles()

    unique: list[dict[str, Any]] = []

    for idea in ideas:
        title = idea.get("title", "").strip()
        if not title:
            continue

        h = _title_hash(title)
        if h in seen:
            continue  # exact match

        # Cosine similarity check against known titles
        is_dup = any(_cosine_sim(title, known) >= _SIM_THRESHOLD for known in kept_raw)
        if is_dup:
            logger.debug(f"[Dedup] Skipped duplicate: {title[:60]}")
            continue

        unique.append(idea)
        seen[h] = title
        kept_raw.append(title)

    _save_seen(seen)
    logger.info(f"[Dedup] {len(ideas)} ideas → {len(unique)} unique after deduplication")
    return unique


def mark_seen(title: str) -> None:
    """Manually mark a title as seen (e.g. after hypothesis queue insertion)."""
    seen = _load_seen()
    h = _title_hash(title)
    seen[h] = title
    _save_seen(seen)
