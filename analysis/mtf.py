"""
Multi-timeframe (MTF) analysis — top-down bias engine.

Hierarchy:
  D1  → macro bias (trend direction)
  H4  → intermediate trend
  H1  → key levels and context
  M15 → entry trigger zone
  M5  → execution

Returns a bias score and aligned direction across timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from analysis import indicators as ind


@dataclass
class TimeframeBias:
    timeframe: str
    direction: str          # "bullish" | "bearish" | "ranging"
    strength: float         # 0.0 – 1.0
    rsi: float
    rsi_zone: str           # "overbought" | "oversold" | "neutral"
    price: float
    ema21: Optional[float]
    ema50: Optional[float]
    ema200: Optional[float]


@dataclass
class MTFBias:
    d1: TimeframeBias
    h4: TimeframeBias
    h1: TimeframeBias
    m15: TimeframeBias
    aligned: bool           # True if D1+H4+H1 all agree
    direction: str          # dominant direction
    confidence: float       # 0.0 – 1.0
    signal_direction: str   # "long" | "short" | "wait"

    def summary(self) -> str:
        arrow = "↑" if self.direction == "bullish" else "↓" if self.direction == "bearish" else "→"
        aligned_str = "ALIGNED" if self.aligned else "MIXED"
        return (
            f"{arrow} {self.direction.upper()} ({aligned_str}) "
            f"D1:{self.d1.direction[0].upper()} "
            f"H4:{self.h4.direction[0].upper()} "
            f"H1:{self.h1.direction[0].upper()} "
            f"M15:{self.m15.direction[0].upper()} "
            f"Conf:{self.confidence:.0%}"
        )


def _bias_from_df(df: pd.DataFrame, tf: str) -> TimeframeBias:
    if df.empty:
        return TimeframeBias(tf, "ranging", 0.0, 50.0, "neutral", 0.0, None, None, None)

    df = ind.apply_all(df)
    last = df.iloc[-1]
    direction = ind.trend_direction(df)
    rsi_val = float(last.get("rsi", 50.0) or 50.0)
    rsi_z = ind.rsi_zone(df)

    # Strength: how far price is from EMA21 (normalized by ATR)
    atr = ind.atr_value(df)
    ema21 = float(last.get("ema21") or last["close"])
    strength = min(abs(last["close"] - ema21) / (atr or 1), 1.0)

    return TimeframeBias(
        timeframe=tf,
        direction=direction,
        strength=strength,
        rsi=rsi_val,
        rsi_zone=rsi_z,
        price=float(last["close"]),
        ema21=float(last["ema21"]) if "ema21" in last and not pd.isna(last["ema21"]) else None,
        ema50=float(last["ema50"]) if "ema50" in last and not pd.isna(last["ema50"]) else None,
        ema200=float(last["ema200"]) if "ema200" in last and not pd.isna(last["ema200"]) else None,
    )


def analyze(
    df_d1: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m15: pd.DataFrame,
) -> MTFBias:
    """Build top-down MTF bias from four timeframe DataFrames."""
    d1  = _bias_from_df(df_d1,  "D1")
    h4  = _bias_from_df(df_h4,  "H4")
    h1  = _bias_from_df(df_h1,  "H1")
    m15 = _bias_from_df(df_m15, "M15")

    directions = [d1.direction, h4.direction, h1.direction]

    # Count votes (excluding "ranging")
    bull_votes = directions.count("bullish")
    bear_votes = directions.count("bearish")

    if bull_votes >= 2:
        direction = "bullish"
        confidence = bull_votes / 3 * 0.8 + m15.strength * 0.2
    elif bear_votes >= 2:
        direction = "bearish"
        confidence = bear_votes / 3 * 0.8 + m15.strength * 0.2
    else:
        direction = "ranging"
        confidence = 0.3

    aligned = bull_votes == 3 or bear_votes == 3

    # Signal direction — only go long in bullish, short in bearish
    if direction == "bullish" and confidence >= 0.5:
        signal = "long"
    elif direction == "bearish" and confidence >= 0.5:
        signal = "short"
    else:
        signal = "wait"

    return MTFBias(
        d1=d1, h4=h4, h1=h1, m15=m15,
        aligned=aligned,
        direction=direction,
        confidence=round(confidence, 3),
        signal_direction=signal,
    )
