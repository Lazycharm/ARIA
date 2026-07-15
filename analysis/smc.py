"""
Smart Money Concepts (SMC) analysis.

Detects:
  - Order Blocks (OBs): institutional footprints
  - Fair Value Gaps (FVGs): imbalance zones price often revisits
  - Break of Structure (BOS): trend continuation confirmation
  - Change of Character (ChoCH): early reversal signal
  - Liquidity Levels: equal highs/lows (sweep targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    index: int
    direction: str          # "bullish" | "bearish"
    top: float
    bottom: float
    candle_time: object     # timestamp
    strength: float = 0.0   # impulse strength after the OB
    tested: bool = False
    valid: bool = True

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def price_inside(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class FVG:
    index: int
    direction: str          # "bullish" | "bearish"
    top: float
    bottom: float
    candle_time: object
    filled: bool = False

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def price_inside(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class SMCResult:
    bullish_obs: list[OrderBlock] = field(default_factory=list)
    bearish_obs: list[OrderBlock] = field(default_factory=list)
    bullish_fvgs: list[FVG] = field(default_factory=list)
    bearish_fvgs: list[FVG] = field(default_factory=list)
    bos: list[dict] = field(default_factory=list)        # list of BOS events
    choch: list[dict] = field(default_factory=list)      # ChoCH events
    liquidity_highs: list[float] = field(default_factory=list)
    liquidity_lows: list[float] = field(default_factory=list)
    current_bias: str = "neutral"                        # "bullish" | "bearish" | "neutral"

    def nearest_ob(self, price: float, direction: str) -> Optional[OrderBlock]:
        """Find the nearest valid OB in the given direction."""
        obs = self.bullish_obs if direction == "bullish" else self.bearish_obs
        candidates = [ob for ob in obs if ob.valid and not ob.tested]
        if not candidates:
            return None
        return min(candidates, key=lambda ob: abs(ob.mid - price))

    def nearest_fvg(self, price: float, direction: str) -> Optional[FVG]:
        fvgs = self.bullish_fvgs if direction == "bullish" else self.bearish_fvgs
        candidates = [f for f in fvgs if not f.filled]
        if not candidates:
            return None
        return min(candidates, key=lambda f: abs((f.top + f.bottom) / 2 - price))


def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> tuple[list[OrderBlock], list[OrderBlock]]:
    """
    Identify Order Blocks.

    Bullish OB: last bearish candle before a strong bullish impulse move up
    Bearish OB: last bullish candle before a strong bearish impulse move down

    Strength = (impulse body) / ATR
    """
    if len(df) < 10:
        return [], []

    df_slice = df.tail(lookback).copy()
    atr = df_slice["atr"].iloc[-1] if "atr" in df_slice.columns else (df_slice["range"].mean())
    if pd.isna(atr) or atr == 0:
        atr = df_slice["range"].mean()

    bullish_obs: list[OrderBlock] = []
    bearish_obs: list[OrderBlock] = []

    for i in range(2, len(df_slice) - 1):
        c = df_slice.iloc[i]
        prev = df_slice.iloc[i - 1]
        nxt = df_slice.iloc[i + 1]

        # Bullish OB: bearish candle followed by a bullish impulse candle that closes above it
        if (
            prev["is_bearish"]                          # previous candle bearish
            and c["is_bullish"]                          # current candle bullish
            and c["close"] > prev["high"]                # closes above previous high
            and c["body"] > 1.5 * atr                   # impulse body
        ):
            ob = OrderBlock(
                index=i - 1,
                direction="bullish",
                top=prev["high"],
                bottom=prev["low"],
                candle_time=df_slice.index[i - 1],
                strength=c["body"] / atr,
            )
            bullish_obs.append(ob)

        # Bearish OB: bullish candle followed by a bearish impulse candle that closes below it
        if (
            prev["is_bullish"]                          # previous candle bullish
            and c["is_bearish"]                          # current candle bearish
            and c["close"] < prev["low"]                 # closes below previous low
            and c["body"] > 1.5 * atr                   # impulse body
        ):
            ob = OrderBlock(
                index=i - 1,
                direction="bearish",
                top=prev["high"],
                bottom=prev["low"],
                candle_time=df_slice.index[i - 1],
                strength=c["body"] / atr,
            )
            bearish_obs.append(ob)

    # Mark tested OBs (price has traded back into the zone)
    current_price = df_slice["close"].iloc[-1]
    for ob in bullish_obs:
        if current_price <= ob.top:  # price has come back into bullish OB
            ob.tested = True
    for ob in bearish_obs:
        if current_price >= ob.bottom:
            ob.tested = True

    # Keep only the most recent 5
    return bullish_obs[-5:], bearish_obs[-5:]


def find_fvgs(df: pd.DataFrame, lookback: int = 50) -> tuple[list[FVG], list[FVG]]:
    """
    Find Fair Value Gaps (3-candle imbalance).

    Bullish FVG: candle[i-1].high < candle[i+1].low (gap between)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap between)
    """
    if len(df) < 5:
        return [], []

    df_slice = df.tail(lookback)
    bullish_fvgs: list[FVG] = []
    bearish_fvgs: list[FVG] = []

    for i in range(1, len(df_slice) - 1):
        prev = df_slice.iloc[i - 1]
        curr = df_slice.iloc[i]
        nxt  = df_slice.iloc[i + 1]

        # Bullish FVG: gap between prev.high and next.low (price moved strongly up)
        if prev["high"] < nxt["low"] and curr["is_bullish"]:
            fvg = FVG(
                index=i,
                direction="bullish",
                top=nxt["low"],
                bottom=prev["high"],
                candle_time=df_slice.index[i],
            )
            bullish_fvgs.append(fvg)

        # Bearish FVG: gap between prev.low and next.high (price moved strongly down)
        if prev["low"] > nxt["high"] and curr["is_bearish"]:
            fvg = FVG(
                index=i,
                direction="bearish",
                top=prev["low"],
                bottom=nxt["high"],
                candle_time=df_slice.index[i],
            )
            bearish_fvgs.append(fvg)

    # Mark filled FVGs
    current_price = df_slice["close"].iloc[-1]
    for fvg in bullish_fvgs:
        if current_price <= fvg.bottom:  # price has filled the gap
            fvg.filled = True
    for fvg in bearish_fvgs:
        if current_price >= fvg.top:
            fvg.filled = True

    return bullish_fvgs[-3:], bearish_fvgs[-3:]


def detect_bos_choch(df: pd.DataFrame, lookback: int = 50) -> tuple[list[dict], list[dict]]:
    """
    Detect Break of Structure (BOS) and Change of Character (ChoCH).

    Uses swing highs/lows over lookback window:
    - BOS bullish: price breaks above a significant swing high → trend continues up
    - BOS bearish: price breaks below a significant swing low → trend continues down
    - ChoCH: opposing structure break after a series of same-direction breaks
    """
    if len(df) < 10:
        return [], []

    df_slice = df.tail(lookback).reset_index(drop=False)
    bos_events: list[dict] = []
    choch_events: list[dict] = []

    highs = []
    lows  = []

    for i in range(2, len(df_slice) - 2):
        h = df_slice.iloc[i]["high"]
        l = df_slice.iloc[i]["low"]

        # Swing high: higher than 2 candles on each side
        if (h > df_slice.iloc[i-1]["high"] and h > df_slice.iloc[i-2]["high"] and
                h > df_slice.iloc[i+1]["high"] and h > df_slice.iloc[i+2]["high"]):
            highs.append({"idx": i, "price": h, "time": df_slice.iloc[i]["time"] if "time" in df_slice.columns else i})

        # Swing low
        if (l < df_slice.iloc[i-1]["low"] and l < df_slice.iloc[i-2]["low"] and
                l < df_slice.iloc[i+1]["low"] and l < df_slice.iloc[i+2]["low"]):
            lows.append({"idx": i, "price": l, "time": df_slice.iloc[i]["time"] if "time" in df_slice.columns else i})

    current_close = df_slice.iloc[-1]["close"]
    current_idx   = len(df_slice) - 1

    # BOS bullish: break above latest swing high
    if highs:
        last_high = highs[-1]
        if current_close > last_high["price"]:
            bos_events.append({
                "type": "BOS",
                "direction": "bullish",
                "level": last_high["price"],
                "broken_at": current_close,
                "idx": current_idx,
            })

    # BOS bearish: break below latest swing low
    if lows:
        last_low = lows[-1]
        if current_close < last_low["price"]:
            bos_events.append({
                "type": "BOS",
                "direction": "bearish",
                "level": last_low["price"],
                "broken_at": current_close,
                "idx": current_idx,
            })

    return bos_events, choch_events


def find_liquidity_levels(df: pd.DataFrame, lookback: int = 100, tolerance_atr: float = 0.5) -> tuple[list[float], list[float]]:
    """
    Identify liquidity pools — equal highs/lows where stops cluster.
    Tolerance: within 0.5 × ATR of each other = considered 'equal'.
    """
    if len(df) < 20:
        return [], []

    df_slice = df.tail(lookback)
    atr = df_slice["atr"].iloc[-1] if "atr" in df_slice.columns else df_slice["range"].mean()
    if pd.isna(atr):
        atr = df_slice["range"].mean()
    tolerance = atr * tolerance_atr

    highs = df_slice["high"].values
    lows  = df_slice["low"].values

    equal_highs: list[float] = []
    equal_lows:  list[float] = []

    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) <= tolerance:
                level = (highs[i] + highs[j]) / 2
                if not any(abs(level - h) <= tolerance for h in equal_highs):
                    equal_highs.append(level)

    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) <= tolerance:
                level = (lows[i] + lows[j]) / 2
                if not any(abs(level - l) <= tolerance for l in equal_lows):
                    equal_lows.append(level)

    return sorted(equal_highs, reverse=True)[:5], sorted(equal_lows)[:5]


def analyze(df: pd.DataFrame) -> SMCResult:
    """Run full SMC analysis on enriched OHLCV DataFrame."""
    bullish_obs, bearish_obs = find_order_blocks(df)
    bullish_fvgs, bearish_fvgs = find_fvgs(df)
    bos, choch = detect_bos_choch(df)
    liq_highs, liq_lows = find_liquidity_levels(df)

    # Determine current bias from BOS signals
    bias = "neutral"
    if bos:
        last_bos = bos[-1]
        bias = last_bos["direction"]

    return SMCResult(
        bullish_obs=bullish_obs,
        bearish_obs=bearish_obs,
        bullish_fvgs=bullish_fvgs,
        bearish_fvgs=bearish_fvgs,
        bos=bos,
        choch=choch,
        liquidity_highs=liq_highs,
        liquidity_lows=liq_lows,
        current_bias=bias,
    )
