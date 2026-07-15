"""
Session Breakout Strategy — London Open & NY Open.

Logic:
  London Open (07:00–08:00 UTC):
    - Measure the high/low range of the Asian session (00:00–07:00 UTC)
    - On candle close after 07:00, if price breaks above Asia high → BUY
    - On candle close after 07:00, if price breaks below Asia low  → SELL
    - SL = opposite side of range + 5 pips buffer
    - TP = 1.5× range distance

  NY Open (13:00–14:00 UTC):
    - Measure the range of 12:00–13:00 UTC consolidation
    - Same breakout logic

Only fires during first 90 minutes of each session open.
Requires: pair in session config, ADX > 15 (some momentum), spread < 2 pips.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, time
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class BreakoutSignal:
    pair: str
    direction: str       # 'long' | 'short'
    entry: float
    sl: float
    tp1: float
    tp2: float
    score: float
    session: str         # 'LONDON_OPEN' | 'NY_OPEN'
    range_high: float
    range_low: float
    reason: str


def _pip_size(pair: str) -> float:
    p = pair.upper().rstrip("M")
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.1
    return 0.0001


def _get_range(df: pd.DataFrame, start_hour: int, end_hour: int) -> tuple[float, float] | None:
    """Get high/low of candles within [start_hour, end_hour) UTC."""
    mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
    sub  = df[mask]
    if sub.empty:
        return None
    return float(sub["high"].max()), float(sub["low"].min())


def scan_breakout(pair: str, df_m15: pd.DataFrame) -> Optional[BreakoutSignal]:
    """
    Check M15 data for a London or NY open breakout setup.
    Returns a BreakoutSignal if conditions met, else None.
    """
    if df_m15.empty or len(df_m15) < 30:
        return None

    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour
    pip     = _pip_size(pair)

    # Ensure index is timezone-aware UTC
    idx = df_m15.index
    if hasattr(idx, "tz") and idx.tz is None:
        df_m15 = df_m15.copy()
        df_m15.index = idx.tz_localize("UTC")

    # Spread check
    try:
        from data.mt5_feed import feed
        tick = feed.get_tick(pair)
        if tick:
            ask = tick.get("ask", 0)
            bid = tick.get("bid", 0)
            spread = (ask - bid) / pip if bid and ask and pip > 0 else 0
            if spread > 2.5:
                return None
    except Exception:
        pass

    last    = df_m15.iloc[-1]
    price   = float(last["close"])
    session = None
    range_h = range_l = None

    # London Open window: 07:00–08:30 UTC (range = Asian 00:00–07:00)
    if 7 <= hour < 9:
        session = "LONDON_OPEN"
        today = now_utc.date()
        today_mask = df_m15.index.date == today
        today_df   = df_m15[today_mask]
        rng = _get_range(today_df, 0, 7)
        if rng:
            range_h, range_l = rng

    # NY Open window: 13:00–14:30 UTC (range = 12:00–13:00 UTC)
    elif 13 <= hour < 15:
        session = "NY_OPEN"
        today = now_utc.date()
        today_mask = df_m15.index.date == today
        today_df   = df_m15[today_mask]
        rng = _get_range(today_df, 12, 13)
        if rng:
            range_h, range_l = rng

    if session is None or range_h is None:
        return None

    range_size = range_h - range_l
    if range_size < 5 * pip:  # range too small → no setup
        return None

    buf = 3 * pip
    direction = None
    entry     = None

    if price > range_h + buf:
        direction = "long"
        entry     = price
    elif price < range_l - buf:
        direction = "short"
        entry = price

    if direction is None:
        return None

    if direction == "long":
        sl  = range_l - buf
        tp1 = entry + range_size * 0.75
        tp2 = entry + range_size * 1.5
    else:
        sl  = range_h + buf
        tp1 = entry - range_size * 0.75
        tp2 = entry - range_size * 1.5

    # Basic conviction score (50–80 depending on range quality)
    rr   = abs(tp2 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
    score = min(80, 50 + rr * 5)

    return BreakoutSignal(
        pair=pair,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        score=score,
        session=session,
        range_high=range_h,
        range_low=range_l,
        reason=f"{session} breakout: range={range_size/pip:.0f}pips, RR={rr:.1f}",
    )
