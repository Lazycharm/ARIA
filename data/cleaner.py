"""
Data cleaning pipeline — detects and drops bad candles, fills micro-gaps.

Call clean(df) after any MT5 candle fetch. Idempotent — safe to call multiple times.
Also exposes quality_report(df, pair, timeframe) for diagnostics.
"""

from __future__ import annotations

from datetime import timezone
from typing import Optional

import pandas as pd
from loguru import logger

# Max single-bar move before flagging as outlier (percent of price)
_MAX_BAR_PCT = 5.0
# Max acceptable spread between open/close and high/low (sanity check)
_MAX_WICK_MULT = 10.0


def clean(df: pd.DataFrame, pair: str = "", timeframe: str = "") -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame from MT5.

    Drops:
      - Rows with zero/null prices
      - Rows where high < low (inverted candles)
      - Rows where close is outside [low, high]
      - Outlier candles with > _MAX_BAR_PCT single-bar move

    Fills:
      - Duplicate timestamps (keep last)
      - Sorts by index ascending
    """
    if df.empty:
        return df

    original_len = len(df)

    # Deduplicate timestamps
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Drop zero/null prices
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]

    # Drop inverted candles (high < low)
    if "high" in df.columns and "low" in df.columns:
        df = df[df["high"] >= df["low"]]

    # Drop candles where close is outside [low, high]
    if all(c in df.columns for c in ["close", "high", "low"]):
        df = df[(df["close"] >= df["low"]) & (df["close"] <= df["high"])]

    # Drop extreme outliers — single bar move > _MAX_BAR_PCT of close price
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        bar_range_pct = (df["high"] - df["low"]) / df["close"].shift(1).fillna(df["close"]) * 100
        outlier_mask = bar_range_pct > _MAX_BAR_PCT
        n_outliers = outlier_mask.sum()
        if n_outliers:
            df = df[~outlier_mask]
            logger.debug(f"[Cleaner] {pair} {timeframe}: dropped {n_outliers} outlier candle(s)")

    dropped = original_len - len(df)
    if dropped:
        logger.debug(f"[Cleaner] {pair} {timeframe}: dropped {dropped}/{original_len} bad candles")

    return df


def quality_report(df: pd.DataFrame, pair: str = "", timeframe: str = "") -> dict:
    """
    Generate a data quality report for a candle DataFrame.
    Returns a dict with metrics — useful for Obsidian logging and dashboard.
    """
    if df.empty:
        return {"pair": pair, "timeframe": timeframe, "status": "empty", "total_candles": 0}

    total = len(df)
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    # Null/zero count
    null_count = df[price_cols].isnull().any(axis=1).sum()
    zero_count = (df[price_cols] == 0).any(axis=1).sum()

    # Inverted candles
    inverted = int(((df["high"] < df["low"]) if "high" in df.columns and "low" in df.columns else pd.Series(dtype=bool)).sum())

    # Gap detection (missing bars)
    gaps = 0
    if len(df) > 1 and hasattr(df.index, "freq"):
        expected_freq = pd.infer_freq(df.index)
        if expected_freq:
            full_range = pd.date_range(df.index[0], df.index[-1], freq=expected_freq, tz=timezone.utc)
            gaps = len(full_range) - len(df)

    # Outlier bars
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        bar_range_pct = (df["high"] - df["low"]) / df["close"].shift(1).fillna(df["close"]) * 100
        outliers = int((bar_range_pct > _MAX_BAR_PCT).sum())
    else:
        outliers = 0

    status = "ok" if (null_count + zero_count + inverted + outliers) == 0 else "issues"

    return {
        "pair":           pair,
        "timeframe":      timeframe,
        "status":         status,
        "total_candles":  total,
        "null_prices":    int(null_count),
        "zero_prices":    int(zero_count),
        "inverted_bars":  inverted,
        "gap_bars":       gaps,
        "outlier_bars":   outliers,
        "date_from":      str(df.index[0]) if len(df) else "—",
        "date_to":        str(df.index[-1]) if len(df) else "—",
    }
