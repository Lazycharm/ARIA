"""
Dynamic Pair Configuration — hot-reloadable pair list.

Allows adding and removing pairs from the dashboard at runtime
without restarting ARIA. Changes persist across restarts via JSON.

Priority order:
  1. pairs.json (dynamic, written by dashboard)
  2. settings.pairs (from .env, used as initial default)

Thread-safe. All pair names stored with Exness 'm' suffix.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

_STORE_PATH = Path("./db/pairs.json")
_lock = threading.Lock()
_pairs: list[str] = []


def _load_from_settings() -> list[str]:
    from config.settings import settings
    return list(settings.pairs)


def _load() -> None:
    global _pairs
    try:
        if _STORE_PATH.exists():
            data = json.loads(_STORE_PATH.read_text())
            loaded = data.get("pairs", [])
            if loaded:
                _pairs = loaded
                logger.info(f"[Pairs] Loaded {len(_pairs)} pairs from config: {_pairs}")
                return
    except Exception as e:
        logger.warning(f"[Pairs] Load failed: {e}")

    _pairs = _load_from_settings()
    logger.info(f"[Pairs] Using settings default: {_pairs}")


def _save() -> None:
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STORE_PATH.write_text(json.dumps({"pairs": _pairs}, indent=2))
    except Exception as e:
        logger.warning(f"[Pairs] Save failed: {e}")


# Load on module import
_load()


def get_pairs() -> list[str]:
    """Return current active pair list (thread-safe)."""
    with _lock:
        return list(_pairs)


def add_pair(symbol: str) -> tuple[bool, str]:
    """
    Add a pair to the active list.
    Returns (success, message).
    Symbol should be in Exness format (e.g. 'EURJPYm').
    """
    with _lock:
        sym = symbol.strip().upper()
        # Normalise — add 'm' suffix if missing and not a special symbol
        if not sym.endswith("M") and len(sym) <= 8:
            sym = sym + "m"
        # Lowercase the broker suffix
        if sym.endswith("M"):
            sym = sym[:-1] + "m"

        if sym in _pairs:
            return False, f"{sym} already in watchlist"

        # Validate against MT5
        try:
            from data.mt5_feed import feed
            tick = feed.get_tick(sym)
            if not tick:
                # Try without suffix
                bare = sym.rstrip("m")
                tick = feed.get_tick(bare)
                if tick:
                    sym = bare
                else:
                    return False, f"{sym} — no data from MT5 (symbol may not exist on this broker)"
        except Exception:
            pass  # Allow add even if MT5 check fails

        _pairs.append(sym)
        _save()
        logger.info(f"[Pairs] Added: {sym} — watchlist now {len(_pairs)} pairs")
        return True, f"✅ {sym} added to watchlist"


def remove_pair(symbol: str) -> tuple[bool, str]:
    """Remove a pair from the active list."""
    with _lock:
        sym = symbol.strip()
        if sym not in _pairs:
            # Try case-insensitive
            matches = [p for p in _pairs if p.upper() == sym.upper()]
            if matches:
                sym = matches[0]
            else:
                return False, f"{sym} not in watchlist"

        if len(_pairs) <= 1:
            return False, "Cannot remove last pair — at least one required"

        _pairs.remove(sym)
        _save()
        logger.info(f"[Pairs] Removed: {sym}")
        return True, f"✅ {sym} removed from watchlist"


def reset_to_defaults() -> None:
    """Reset to settings.pairs defaults."""
    global _pairs
    with _lock:
        _pairs = _load_from_settings()
        _save()
        logger.info(f"[Pairs] Reset to defaults: {_pairs}")
