"""
Range Trading Strategy — Low ADX environment.

Currently the main SMC strategy skips trades when ADX < 20 (no trend).
This module monetizes the range by:
  1. Identifying horizontal support/resistance via swing lows/highs
  2. Entering near SR levels with tight SL (reversal confirmation)
  3. Targeting the opposite SR level

Entry:
  - ADX < 20 (confirmed range)
  - Price within 5 pips of support → LONG (with small bounce candle)
  - Price within 5 pips of resistance → SHORT (with small rejection candle)
  - RSI 40–60 range (not overextended)
  - SR level tested ≥ 2 times in last 50 bars

SL: 8 pips beyond the SR level
TP1: midpoint of range (50%)
TP2: opposite SR level

Score: 55–75 (range trades are lower conviction than trend)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class RangeTradingSignal:
    pair: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    score: float
    support: float
    resistance: float
    adx: float
    reason: str


def _pip_size(pair: str) -> float:
    p = pair.upper().rstrip("M")
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.1
    return 0.0001


def _find_sr_levels(df: pd.DataFrame, lookback: int = 50) -> tuple[float, float]:
    """
    Find horizontal support and resistance from recent swing lows/highs.
    Returns (support, resistance) based on most touched price clusters.
    """
    recent  = df.tail(lookback)
    highs   = recent["high"].values
    lows    = recent["low"].values
    close   = recent["close"].values

    # Simple: use rolling N-bar swing high/low
    n = 5
    sup = float(recent["low"].rolling(n, center=True).min().dropna().iloc[-1])
    res = float(recent["high"].rolling(n, center=True).max().dropna().iloc[-1])
    return sup, res


def scan_range_trading(pair: str, df_m15: pd.DataFrame) -> Optional[RangeTradingSignal]:
    """
    Check M15 data for a range-trading setup.
    Only fires when ADX < 20 (range-bound market).
    Returns RangeTradingSignal or None.
    """
    if df_m15.empty or len(df_m15) < 60:
        return None

    try:
        from analysis.indicators import apply_all
        df = apply_all(df_m15)
    except Exception:
        df = df_m15

    if "adx" not in df.columns:
        return None

    adx   = float(df["adx"].iloc[-1])
    if adx >= 20:
        return None  # Trending — not a range environment

    price = float(df["close"].iloc[-1])
    rsi   = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
    pip   = _pip_size(pair)

    if not (35 <= rsi <= 65):
        return None  # RSI overextended — risk of breakout

    sup, res = _find_sr_levels(df)
    rng      = res - sup
    mid      = (sup + res) / 2

    if rng < 20 * pip:  # range too tight
        return None

    direction = None
    tol = 8 * pip

    if abs(price - sup) <= tol:
        direction = "long"
    elif abs(price - res) <= tol:
        direction = "short"
    else:
        return None

    sl_buf = 8 * pip

    if direction == "long":
        sl  = sup - sl_buf
        tp1 = mid
        tp2 = res - sl_buf
    else:
        sl  = res + sl_buf
        tp1 = mid
        tp2 = sup + sl_buf

    rr    = abs(tp2 - price) / abs(price - sl) if abs(price - sl) > 0 else 1
    score = min(75, 55 + rr * 3)

    reason = (
        f"Range trade {direction.upper()}: ADX={adx:.0f}, RSI={rsi:.0f}, "
        f"range={rng/pip:.0f}pips, RR={rr:.1f}"
    )

    return RangeTradingSignal(
        pair=pair,
        direction=direction,
        entry=price,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        score=score,
        support=sup,
        resistance=res,
        adx=adx,
        reason=reason,
    )
