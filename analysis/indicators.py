"""
Technical indicators — uses the `ta` library (Python 3.14 compatible, no numba).

All functions take a raw OHLCV DataFrame and return it enriched
with indicator columns. Keeps analysis pure (no MT5 coupling).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

import ta
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def apply_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full indicator suite to OHLCV DataFrame."""
    if len(df) < 50:
        return df

    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Trend (EMA/SMA) ───────────────────────────────────────────
    df["ema9"]   = EMAIndicator(close=close, window=9,   fillna=False).ema_indicator()
    df["ema21"]  = EMAIndicator(close=close, window=21,  fillna=False).ema_indicator()
    df["ema50"]  = EMAIndicator(close=close, window=50,  fillna=False).ema_indicator()
    df["ema200"] = EMAIndicator(close=close, window=200, fillna=False).ema_indicator()
    df["sma50"]  = SMAIndicator(close=close, window=50,  fillna=False).sma_indicator()
    df["sma200"] = SMAIndicator(close=close, window=200, fillna=False).sma_indicator()

    # ── Momentum ──────────────────────────────────────────────────
    df["rsi"] = RSIIndicator(close=close, window=14, fillna=False).rsi()

    macd_obj = MACD(close=close, window_fast=12, window_slow=26, window_sign=9, fillna=False)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3, fillna=False)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── Volatility ────────────────────────────────────────────────
    bb = BollingerBands(close=close, window=20, window_dev=2, fillna=False)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14, fillna=False).average_true_range()

    # ── Trend Strength ────────────────────────────────────────────
    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14, fillna=False)
    df["adx"]  = adx_obj.adx()
    df["di_p"] = adx_obj.adx_pos()
    df["di_m"] = adx_obj.adx_neg()

    # ── Volume ────────────────────────────────────────────────────
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume, fillna=False).on_balance_volume()

    # ── Price Action Helpers ──────────────────────────────────────
    df["body"]       = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["is_bullish"] = df["close"] > df["open"]
    df["is_bearish"] = df["close"] < df["open"]
    df["range"]      = df["high"] - df["low"]

    return df


def trend_direction(df: pd.DataFrame) -> str:
    """Fast EMA trend: 'bullish' | 'bearish' | 'ranging'."""
    if df.empty or "ema21" not in df.columns or "ema50" not in df.columns:
        return "ranging"
    last = df.iloc[-1]
    price  = last["close"]
    ema21  = last.get("ema21")
    ema50  = last.get("ema50")
    if pd.isna(ema21) or pd.isna(ema50):
        return "ranging"
    if price > ema21 > ema50:
        return "bullish"
    if price < ema21 < ema50:
        return "bearish"
    return "ranging"


def rsi_zone(df: pd.DataFrame) -> str:
    """'overbought' | 'oversold' | 'neutral'."""
    if df.empty or "rsi" not in df.columns:
        return "neutral"
    rsi = df["rsi"].iloc[-1]
    if pd.isna(rsi):
        return "neutral"
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    return "neutral"


def atr_value(df: pd.DataFrame) -> float:
    """Return latest ATR value (0 if unavailable)."""
    if df.empty or "atr" not in df.columns:
        return 0.0
    val = df["atr"].iloc[-1]
    return float(val) if not pd.isna(val) else 0.0
