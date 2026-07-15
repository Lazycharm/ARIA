"""
Regime classification per closed trade.

Tags each trade with the market regime at time of entry:
  - TRENDING   : ADX > 25 + directional bias
  - RANGING    : ADX < 20, price oscillating between support/resistance
  - VOLATILE   : ATR > 1.5× 20-bar avg ATR

Exposed as classify_trade_regime(pair, entry_time) → str.
Called in execution/order_manager.py on trade open, stored in Trade.regime.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger


def classify_trade_regime(pair: str, entry_time: Optional[datetime] = None) -> str:
    """
    Classify the current (or historical) market regime for a pair.
    Returns: 'TRENDING' | 'RANGING' | 'VOLATILE'
    """
    try:
        from data.mt5_feed import feed
        from analysis.indicators import apply_all
        import pandas as pd

        df = feed.get_candles(pair, "H1", count=60)
        if df.empty or len(df) < 30:
            return "UNKNOWN"

        df = apply_all(df)

        # Slice to entry time if provided
        if entry_time is not None:
            entry_ts = pd.Timestamp(entry_time)
            df = df[df.index <= entry_ts]
            if df.empty:
                return "UNKNOWN"

        last = df.iloc[-1]

        adx = last.get("adx", 0) if hasattr(last, "get") else getattr(last, "adx", 0)
        atr = last.get("atr", None) if hasattr(last, "get") else getattr(last, "atr", None)

        # ADX-based regime
        if adx is None:
            adx = 0.0
        adx = float(adx) if adx else 0.0

        # Volatile check: current ATR vs 20-bar average
        if atr is not None and "atr" in df.columns:
            atr_series = df["atr"].dropna()
            if len(atr_series) >= 20:
                avg_atr = float(atr_series.iloc[-20:].mean())
                curr_atr = float(atr_series.iloc[-1])
                if avg_atr > 0 and curr_atr > 1.5 * avg_atr:
                    return "VOLATILE"

        if adx > 25:
            return "TRENDING"
        if adx < 20:
            return "RANGING"
        return "TRENDING"  # transition zone → default to trending

    except Exception as e:
        logger.debug(f"[Regime] classify failed for {pair}: {e}")
        return "UNKNOWN"


def tag_open_trade_regime(pair: str) -> str:
    """Convenience wrapper called at trade entry."""
    regime = classify_trade_regime(pair)
    logger.debug(f"[Regime] {pair} → {regime}")
    return regime
