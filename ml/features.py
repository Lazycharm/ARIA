"""
Phase 6 — ML Feature Extraction.

Converts a confluence score breakdown + price data into a fixed-length
feature vector suitable for LightGBM / sklearn classifiers.

Features (22 total):
  score components (10) — the raw pts from each confluence component
  price features   (8)  — ATR ratio, distance to OB, RSI, ADX, etc.
  context          (4)  — session pts, spread, direction encoded, lot size

Training samples are saved to db/ml_samples.jsonl after every closed trade.
When >= 60 samples exist, auto-training kicks in via the trainer.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

_SAMPLES_PATH = Path("./db/ml_samples.jsonl")
_lock = threading.Lock()

FEATURE_NAMES = [
    # Confluence breakdown (raw pts)
    "f_mtf", "f_smc", "f_rsi", "f_ema", "f_adx",
    "f_bos", "f_macd", "f_stoch", "f_session", "f_news",
    # Price-level features
    "f_adx_raw", "f_rsi_raw", "f_spread_pips", "f_atr_ratio",
    "f_ob_distance", "f_macd_hist", "f_stoch_k", "f_di_spread",
    # Context
    "f_direction",    # 1=long, -1=short
    "f_session_pts",  # 0/5/10
    "f_total_score",  # final confluence score
    "f_spread_rank",  # spread / typical_spread (normalised)
]

MIN_TRAINING_SAMPLES = 60


def extract(
    breakdown: dict[str, float],
    df_m15: Optional[pd.DataFrame],
    direction: str,
    session_pts: float,
    spread_pips: float,
    total_score: float,
) -> dict[str, float]:
    """Build a feature dict from confluence components + raw indicator data."""
    feats: dict[str, float] = {}

    # Confluence breakdown components
    feats["f_mtf"]     = breakdown.get("mtf", 0.0)
    feats["f_smc"]     = breakdown.get("smc", 0.0)
    feats["f_rsi"]     = breakdown.get("rsi", 0.0)
    feats["f_ema"]     = breakdown.get("ema", 0.0)
    feats["f_adx"]     = breakdown.get("adx", 0.0)
    feats["f_bos"]     = breakdown.get("bos", 0.0)
    feats["f_macd"]    = breakdown.get("macd", 0.0)
    feats["f_stoch"]   = breakdown.get("stoch", 0.0)
    feats["f_session"] = breakdown.get("session", 0.0)
    feats["f_news"]    = breakdown.get("news", 0.0)

    # Raw indicator values from M15
    feats["f_adx_raw"]    = 0.0
    feats["f_rsi_raw"]    = 50.0
    feats["f_atr_ratio"]  = 0.0
    feats["f_ob_distance"]= 0.0
    feats["f_macd_hist"]  = 0.0
    feats["f_stoch_k"]    = 50.0
    feats["f_di_spread"]  = 0.0

    if df_m15 is not None and not df_m15.empty:
        last = df_m15.iloc[-1]
        price = last.get("close", 1.0)

        if "adx" in df_m15.columns and not pd.isna(last.get("adx")):
            feats["f_adx_raw"] = float(last["adx"])
        if "rsi" in df_m15.columns and not pd.isna(last.get("rsi")):
            feats["f_rsi_raw"] = float(last["rsi"])
        if "atr" in df_m15.columns and not pd.isna(last.get("atr")) and price > 0:
            feats["f_atr_ratio"] = float(last["atr"]) / price * 1000  # normalised
        if "macd_hist" in df_m15.columns and not pd.isna(last.get("macd_hist")):
            feats["f_macd_hist"] = float(last["macd_hist"])
        if "stoch_k" in df_m15.columns and not pd.isna(last.get("stoch_k")):
            feats["f_stoch_k"] = float(last["stoch_k"])
        if "di_p" in df_m15.columns and "di_m" in df_m15.columns:
            di_p = float(last.get("di_p", 0) or 0)
            di_m = float(last.get("di_m", 0) or 0)
            feats["f_di_spread"] = di_p - di_m  # positive = bullish DI dominance

    feats["f_direction"]   = 1.0 if direction == "long" else -1.0
    feats["f_session_pts"] = session_pts
    feats["f_total_score"] = total_score
    feats["f_spread_pips"] = spread_pips
    feats["f_spread_rank"] = spread_pips / 2.0  # normalised vs 2 pip reference

    return feats


def to_vector(feats: dict[str, float]) -> list[float]:
    """Convert feature dict to ordered list matching FEATURE_NAMES."""
    return [feats.get(name, 0.0) for name in FEATURE_NAMES]


def save_sample(
    feats: dict[str, float],
    won: bool,
    pnl: float,
    pair: str,
) -> None:
    """Append one training sample to the JSONL file."""
    sample = {
        "features": {k: round(v, 6) for k, v in feats.items()},
        "label": 1 if won else 0,
        "pnl": round(pnl, 4),
        "pair": pair,
    }
    with _lock:
        _SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SAMPLES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")


def load_samples() -> tuple[list[list[float]], list[int]]:
    """Load all saved samples. Returns (X, y)."""
    if not _SAMPLES_PATH.exists():
        return [], []

    X, y = [], []
    with _lock:
        lines = _SAMPLES_PATH.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        try:
            s = json.loads(line)
            X.append(to_vector(s["features"]))
            y.append(int(s["label"]))
        except Exception:
            continue

    return X, y


def sample_count() -> int:
    """Quick count without loading all data."""
    if not _SAMPLES_PATH.exists():
        return 0
    with _lock:
        return sum(1 for _ in _SAMPLES_PATH.open(encoding="utf-8"))
