"""
Phase 6 — ML Signal Scorer.

Loads the trained model and converts its win-probability output into
a score adjustment (+/- pts) that is blended into the confluence total.

Boost formula:
  prob > 0.72  → +12 pts  (model is confident this is a winner)
  prob > 0.60  → +6  pts  (moderate confidence)
  prob > 0.50  → +2  pts  (slight edge)
  prob < 0.40  → -8  pts  (model thinks this will lose)
  prob < 0.30  → -15 pts  (strong negative signal)
  no model     →  0  pts  (no adjustment)

The model is loaded once at startup and reloaded when it's refreshed.
Thread-safe. Non-blocking — if model fails to load, passes through silently.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

from ml.features import to_vector, FEATURE_NAMES

_MODEL_PATH = Path("./db/ml_model.pkl")


class MLPredictor:
    """Thread-safe ML signal scorer."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._model = None
        self._ready = False
        self._load()

    def _load(self) -> None:
        if not _MODEL_PATH.exists():
            return
        try:
            with _MODEL_PATH.open("rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._ready = True
            logger.info(f"[ML] Model loaded ({data['backend']})")
        except Exception as e:
            logger.warning(f"[ML] Model load failed: {e}")
            self._ready = False

    def reload(self) -> None:
        """Hot-reload model (called after training completes)."""
        with self._lock:
            self._load()

    def is_ready(self) -> bool:
        return self._ready

    def predict_proba(self, feature_dict: dict[str, float]) -> float:
        """Return P(win) as float 0-1. Returns 0.5 if model not ready."""
        if not self._ready or self._model is None:
            return 0.5
        try:
            vec = [feature_dict.get(n, 0.0) for n in FEATURE_NAMES]
            with self._lock:
                proba = self._model.predict_proba([vec])[0]
            return float(proba[1])   # prob of class 1 = win
        except Exception as e:
            logger.debug(f"[ML] Predict error: {e}")
            return 0.5

    def get_boost(self, feature_dict: dict[str, float]) -> float:
        """Convert win probability to a score adjustment in points."""
        if not self._ready:
            return 0.0

        prob = self.predict_proba(feature_dict)

        if prob > 0.72:
            return 12.0
        elif prob > 0.60:
            return 6.0
        elif prob > 0.50:
            return 2.0
        elif prob < 0.30:
            return -15.0
        elif prob < 0.40:
            return -8.0
        return 0.0


# Singleton
predictor = MLPredictor()
