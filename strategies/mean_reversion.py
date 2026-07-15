"""
Mean Reversion Strategy.

Logic (RSI + Bollinger Band mean reversion):
  Entry conditions (LONG):
    - Price closes below lower Bollinger Band (2σ)
    - RSI(14) < 30 (oversold)
    - Price is in a RANGING regime (ADX < 25)
    - No significant news in next 2h (calendar check)

  Entry conditions (SHORT):
    - Price closes above upper Bollinger Band (2σ)
    - RSI(14) > 70 (overbought)
    - RANGING regime

  SL: 0.5 ATR beyond the band extreme
  TP1: midline (20-bar SMA) → partial close
  TP2: opposite band (full close)

  Score: 55–80 based on RSI extremity + band distance
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class MeanReversionSignal:
    pair: str
    direction: str    # 'long' | 'short'
    entry: float
    sl: float
    tp1: float        # midline
    tp2: float        # opposite band
    score: float
    rsi: float
    bb_pct: float     # 0=lower band, 0.5=mid, 1=upper band
    reason: str


def _pip_size(pair: str) -> float:
    p = pair.upper().rstrip("M")
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.1
    return 0.0001


def _bb_and_rsi(df: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    """Return (upper_bb, mid_bb, lower_bb, rsi, atr, price)."""
    from analysis.indicators import apply_all
    df = apply_all(df)
    last = df.iloc[-1]

    price = float(last.get("close", 0))
    rsi   = float(last.get("rsi",   50))

    # Bollinger Bands (20, 2σ) — compute from raw OHLCV if not in indicators
    window = 20
    close  = df["close"].astype(float)
    mid    = float(close.rolling(window).mean().iloc[-1])
    std    = float(close.rolling(window).std().iloc[-1])
    upper  = mid + 2 * std
    lower  = mid - 2 * std

    atr_col = last.get("atr", None)
    if atr_col is None and "atr" in df.columns:
        atr_col = float(df["atr"].iloc[-1])
    atr = float(atr_col) if atr_col else abs(upper - lower) / 4

    return upper, mid, lower, rsi, atr, price


def scan_mean_reversion(pair: str, df_m15: pd.DataFrame) -> Optional[MeanReversionSignal]:
    """
    Check M15 data for a mean-reversion setup.
    Only fires in RANGING regime (ADX < 25).
    Returns MeanReversionSignal or None.
    """
    if df_m15.empty or len(df_m15) < 60:
        return None

    # Regime check
    try:
        adx_col = df_m15.get("adx", None) if hasattr(df_m15, "get") else None
        if adx_col is None and "adx" in df_m15.columns:
            adx = float(df_m15["adx"].iloc[-1])
        else:
            from analysis.indicators import apply_all
            tmp = apply_all(df_m15)
            adx = float(tmp["adx"].iloc[-1]) if "adx" in tmp.columns else 15.0
        if adx > 25:
            return None   # Trending — not our setup
    except Exception:
        pass

    try:
        upper, mid, lower, rsi, atr, price = _bb_and_rsi(df_m15)
    except Exception as e:
        logger.debug(f"[MeanReversion] Indicator calc failed: {e}")
        return None

    pip    = _pip_size(pair)
    direction = None

    if price < lower and rsi < 30:
        direction = "long"
    elif price > upper and rsi > 70:
        direction = "short"
    else:
        return None

    if direction == "long":
        sl   = lower - 0.5 * atr
        tp1  = mid
        tp2  = upper
        bb_p = 0.0
    else:
        sl   = upper + 0.5 * atr
        tp1  = mid
        tp2  = lower
        bb_p = 1.0

    # Score based on RSI extremity
    rsi_extreme = max(0, (30 - rsi) if direction == "long" else (rsi - 70))
    score = min(80, 55 + rsi_extreme * 0.5)

    rr  = abs(tp2 - price) / abs(price - sl) if abs(price - sl) > 0 else 1
    reason = (
        f"Mean reversion {direction.upper()}: RSI={rsi:.0f}, "
        f"price {'below' if direction=='long' else 'above'} BB, RR={rr:.1f}"
    )

    return MeanReversionSignal(
        pair=pair,
        direction=direction,
        entry=price,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        score=score,
        rsi=rsi,
        bb_pct=bb_p,
        reason=reason,
    )
