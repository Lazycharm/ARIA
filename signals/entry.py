"""
Entry calculator — converts a confluence signal into a full trade plan.

Given: pair, direction, current_price, df_m15 (with indicators), signal
Returns: entry, SL, TP1, TP2, TP3 with RR ratios

SL placement (in priority order):
  1. Beyond the Order Block (structural level) — most accurate
  2. Beyond recent swing high/low — structure-based
  3. ATR-based fallback — when no structure available

TP cascade:
  TP1 → 1.5× SL distance  (close 50%, move SL to entry)
  TP2 → 3.0× SL distance  (close 30%, trail remaining)
  TP3 → 5.0× SL distance  (let runner go)

Using 1.5:1 for TP1 (vs. old 2:1) makes TPs more reachable — higher win rate
on partials at the cost of slightly lower per-trade reward. The re-entry
cycling compensates by taking more setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from analysis.confluence import ConfluenceScore
from analysis.indicators import atr_value


@dataclass
class TradeSetup:
    pair: str
    direction: str          # "long" | "short"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    sl_pips: float
    rr1: float
    rr2: float
    rr3: float
    reason: str
    sl_type: str = "atr"    # "ob" | "swing" | "atr"

    @property
    def sl_distance(self) -> float:
        return abs(self.entry - self.sl)


def build_setup(
    signal: ConfluenceScore,
    current_price: float,
    df_m15: pd.DataFrame,
    atr_multiplier_sl: float = 1.2,
) -> Optional[TradeSetup]:
    """Convert a confluence signal into a full entry/exit plan."""
    if signal.direction == "wait":
        return None

    # Alternative strategies pre-compute strategy-appropriate SL/TP; use them directly.
    if getattr(signal, "_preset_setup", None) is not None:
        return signal._preset_setup

    atr = atr_value(df_m15)
    if atr <= 0:
        return None

    pip_size  = _pip_size(signal.pair)
    direction = signal.direction

    sl, sl_type = _find_sl(signal, current_price, df_m15, atr, atr_multiplier_sl)

    sl_distance = abs(current_price - sl)

    # Minimum SL distance: 0.5× ATR to prevent getting stopped on spread noise
    min_sl_dist = atr * 0.5
    if sl_distance < min_sl_dist:
        if direction == "long":
            sl = current_price - min_sl_dist
        else:
            sl = current_price + min_sl_dist
        sl_distance = min_sl_dist
        sl_type = "atr_min"

    if sl_distance <= 0:
        return None

    # TPs — 1.5:1 / 3:1 / 5:1  (more reachable than old 2:4:6)
    if direction == "long":
        tp1 = current_price + sl_distance * 1.5
        tp2 = current_price + sl_distance * 3.0
        tp3 = current_price + sl_distance * 5.0
    else:
        tp1 = current_price - sl_distance * 1.5
        tp2 = current_price - sl_distance * 3.0
        tp3 = current_price - sl_distance * 5.0

    sl_pips = sl_distance / pip_size

    return TradeSetup(
        pair=signal.pair,
        direction=direction,
        entry=round(current_price, 5),
        sl=round(sl, 5),
        tp1=round(tp1, 5),
        tp2=round(tp2, 5),
        tp3=round(tp3, 5),
        sl_pips=round(sl_pips, 1),
        rr1=1.5,
        rr2=3.0,
        rr3=5.0,
        reason=signal.entry_reason,
        sl_type=sl_type,
    )


def _find_sl(
    signal: ConfluenceScore,
    price: float,
    df_m15: pd.DataFrame,
    atr: float,
    atr_mult: float,
) -> tuple[float, str]:
    """
    SL placement priority:
    1. OB structure level (most reliable — institutional level)
    2. Recent swing high/low from M15 (swing structure)
    3. ATR-based fallback
    """
    direction = signal.direction
    buffer    = atr * 0.3   # small buffer beyond the structural level

    # ── Priority 1: Order Block structural SL ─────────────────────
    if signal.nearest_ob_level:
        if direction == "long":
            # SL below the bullish OB bottom (OB mid minus half the OB range)
            sl = signal.nearest_ob_level - buffer
        else:
            sl = signal.nearest_ob_level + buffer

        sl_dist = abs(price - sl)
        # OB SL must be within reasonable range: 0.5–3× ATR
        if atr * 0.5 <= sl_dist <= atr * 4:
            return sl, "ob"

    # ── Priority 2: Swing high/low ────────────────────────────────
    if not df_m15.empty and len(df_m15) >= 10:
        swing_sl = _swing_sl(direction, price, df_m15, buffer)
        if swing_sl is not None:
            sl_dist = abs(price - swing_sl)
            if atr * 0.5 <= sl_dist <= atr * 4:
                return swing_sl, "swing"

    # ── Priority 3: ATR fallback ──────────────────────────────────
    if direction == "long":
        return price - atr * atr_mult, "atr"
    else:
        return price + atr * atr_mult, "atr"


def _swing_sl(
    direction: str,
    price: float,
    df_m15: pd.DataFrame,
    buffer: float,
) -> Optional[float]:
    """Find nearest swing high/low within the last 20 candles as SL anchor."""
    lookback = df_m15.tail(20)
    if direction == "long":
        # SL below recent swing low
        swing_lows = []
        lows = lookback["low"].values
        for i in range(2, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append(lows[i])
        if swing_lows:
            # Use the most recent swing low below current price
            valid = [l for l in reversed(swing_lows) if l < price]
            if valid:
                return valid[0] - buffer
    else:
        # SL above recent swing high
        swing_highs = []
        highs = lookback["high"].values
        for i in range(2, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append(highs[i])
        if swing_highs:
            valid = [h for h in reversed(swing_highs) if h > price]
            if valid:
                return valid[0] + buffer
    return None


def _pip_size(pair: str) -> float:
    p = pair.upper()
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.1
    if any(x in p for x in ("NAS", "SPX", "US30", "US500", "USTEC", "NDX")):
        return 1.0
    return 0.0001
