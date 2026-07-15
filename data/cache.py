"""
Historical data cache — avoids re-fetching same bars from MT5.

Primary: Redis (when REDIS_URL is set) — sub-millisecond, survives process restarts
Fallback: Parquet files in db/candle_cache/ — always available, no extra dependency

TTL per timeframe: M15/H1 = 5 min, H4 = 15 min, D1 = 60 min.
Cache is keyed by (pair, timeframe, count).
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

_CACHE_DIR = Path("./db/candle_cache")

_TTL: dict[str, int] = {
    "M1":  60,
    "M5":  120,
    "M15": 300,
    "M30": 300,
    "H1":  300,
    "H4":  900,
    "D1":  3600,
    "W1":  3600,
}

# ── Redis client (lazy, optional) ─────────────────────────────────────────────

_redis_client = None
_redis_tried  = False


def _redis():
    global _redis_client, _redis_tried
    if _redis_tried:
        return _redis_client
    _redis_tried = True
    try:
        from config.settings import settings
        if not settings.redis_url:
            return None
        import redis as redis_lib
        c = redis_lib.from_url(settings.redis_url, decode_responses=False, socket_timeout=1.0)
        c.ping()
        _redis_client = c
        logger.info("[Cache] Redis connected — using Redis fast-path")
    except Exception as e:
        logger.debug(f"[Cache] Redis unavailable, using parquet fallback: {e}")
        _redis_client = None
    return _redis_client


def _key(pair: str, timeframe: str, count: int) -> str:
    raw = f"{pair}:{timeframe}:{count}"
    return hashlib.md5(raw.encode()).hexdigest()


def _redis_key(pair: str, timeframe: str, count: int) -> str:
    return f"aria:candle:{pair}:{timeframe}:{count}"


def _path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.parquet"


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    return buf.getvalue()


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


# ── Public API ────────────────────────────────────────────────────────────────

def get(pair: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if fresh, else None. Tries Redis then parquet."""
    ttl = _TTL.get(timeframe.upper(), 300)

    # ── Redis path ──────────────────────────────────────────────────────
    rc = _redis()
    if rc is not None:
        try:
            rk   = _redis_key(pair, timeframe, count)
            data = rc.get(rk)
            if data:
                df = _bytes_to_df(data)
                logger.debug(f"[Cache] REDIS HIT {pair} {timeframe} count={count}")
                return df
        except Exception as e:
            logger.debug(f"[Cache] Redis read error: {e}")

    # ── Parquet fallback ────────────────────────────────────────────────
    try:
        key  = _key(pair, timeframe, count)
        path = _path(key)
        if not path.exists():
            return None

        age = time.time() - path.stat().st_mtime
        if age > ttl:
            path.unlink(missing_ok=True)
            return None

        df = pd.read_parquet(path)
        logger.debug(f"[Cache] PARQUET HIT {pair} {timeframe} count={count} age={age:.0f}s")
        return df

    except Exception as e:
        logger.debug(f"[Cache] Read error: {e}")
        return None


def put(pair: str, timeframe: str, count: int, df: pd.DataFrame) -> None:
    """Store DataFrame. Writes to Redis (with TTL) and parquet simultaneously."""
    if df.empty:
        return
    ttl = _TTL.get(timeframe.upper(), 300)

    # ── Redis write ─────────────────────────────────────────────────────
    rc = _redis()
    if rc is not None:
        try:
            rk = _redis_key(pair, timeframe, count)
            rc.setex(rk, ttl, _df_to_bytes(df))
            logger.debug(f"[Cache] REDIS STORE {pair} {timeframe} count={count}")
        except Exception as e:
            logger.debug(f"[Cache] Redis write error: {e}")

    # ── Parquet write (always — fallback persistence) ───────────────────
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key  = _key(pair, timeframe, count)
        path = _path(key)
        df.to_parquet(path)
        logger.debug(f"[Cache] PARQUET STORE {pair} {timeframe} count={count} rows={len(df)}")
    except Exception as e:
        logger.debug(f"[Cache] Parquet write error: {e}")


def invalidate(pair: str = "", timeframe: str = "") -> int:
    """Remove cache entries (both Redis and parquet). Returns count removed."""
    removed = 0

    # ── Redis flush ─────────────────────────────────────────────────────
    rc = _redis()
    if rc is not None:
        try:
            pattern = f"aria:candle:{pair or '*'}:{timeframe or '*'}:*"
            keys = rc.keys(pattern)
            if keys:
                rc.delete(*keys)
                removed += len(keys)
        except Exception as e:
            logger.debug(f"[Cache] Redis invalidate error: {e}")

    # ── Parquet flush ───────────────────────────────────────────────────
    if not _CACHE_DIR.exists():
        return removed
    for f in _CACHE_DIR.glob("*.parquet"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass

    if removed:
        logger.debug(f"[Cache] Invalidated {removed} cache entries")
    return removed
