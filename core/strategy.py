"""
Phase 5 — Multi-Strategy Portfolio Management.

Regime-aware strategy routing:
  TREND     ADX > 25   — full SMC confluence, standard params
  BREAKOUT  ADX 15-25  — momentum entry, tighter SL, lower min score
  WAIT      ADX < 15   — ranging market, skip entirely

Portfolio-level guards:
  - Max 2 positions in same base currency (avoid over-concentration)
  - Correlation guard: no opposing USD positions (EUR long + USD long = hedge)
  - Max 1 TREND position and 1 BREAKOUT position open at any time

Strategy configs tune how the confluence score is evaluated and how
orders are sized — without changing the core scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
from loguru import logger


class StrategyMode(Enum):
    TREND     = "trend"      # Strong directional market
    BREAKOUT  = "breakout"   # Building momentum
    WAIT      = "wait"       # Ranging — sit out


@dataclass
class StrategyConfig:
    mode: StrategyMode
    min_score_delta: float   # Added to adaptive threshold (+5 for breakout = pickier)
    lot_multiplier: float    # Applied on top of adaptive multiplier
    tp_ratio: float          # TP1 ratio (distance multiplier)
    atr_sl_mult: float       # ATR SL multiplier (tighter for breakout)
    label: str


STRATEGY_CONFIGS: dict[StrategyMode, StrategyConfig] = {
    StrategyMode.TREND: StrategyConfig(
        mode=StrategyMode.TREND,
        min_score_delta=0.0,   # use adaptive threshold as-is
        lot_multiplier=1.0,
        tp_ratio=1.5,          # 1.5:3:5 — standard cascade
        atr_sl_mult=1.2,
        label="TREND",
    ),
    StrategyMode.BREAKOUT: StrategyConfig(
        mode=StrategyMode.BREAKOUT,
        min_score_delta=3.0,   # slightly pickier — breakouts fail more often
        lot_multiplier=0.8,    # smaller size — less certainty
        tp_ratio=1.2,          # tighter TP1 — take profits faster on breakouts
        atr_sl_mult=1.0,       # tighter SL — less room to prove itself
        label="BKT",
    ),
    StrategyMode.WAIT: StrategyConfig(
        mode=StrategyMode.WAIT,
        min_score_delta=999.0, # effectively blocks all trades
        lot_multiplier=0.0,
        tp_ratio=0.0,
        atr_sl_mult=0.0,
        label="WAIT",
    ),
}


def detect_regime(df_m15: pd.DataFrame) -> StrategyMode:
    """Determine strategy mode from ADX on the M15 dataframe."""
    if df_m15.empty or "adx" not in df_m15.columns:
        return StrategyMode.WAIT

    adx = df_m15["adx"].iloc[-1]
    if pd.isna(adx):
        return StrategyMode.WAIT

    if adx >= 25:
        return StrategyMode.TREND
    elif adx >= 15:
        return StrategyMode.BREAKOUT
    else:
        return StrategyMode.WAIT


def get_strategy(df_m15: pd.DataFrame) -> StrategyConfig:
    """Return the full StrategyConfig for the current market regime."""
    mode = detect_regime(df_m15)
    return STRATEGY_CONFIGS[mode]
