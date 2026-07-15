"""
ML Performance Tracker — measures whether the model's boost actually helps.

Buckets every closed trade by its ML boost at entry:
  boosted   (boost > 0): model was bullish on the trade
  penalized (boost < 0): model flagged it as likely loser
  neutral   (boost == 0): model not ready / no signal

Persists to db/ml_performance.json. Thread-safe singleton.

Call record() on every trade close.
Call get_stats() for dashboard and MLflow reporting.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from loguru import logger

_PATH = Path("./db/ml_performance.json")
_MIN_TRADES_FOR_VERDICT = 10   # need at least this many boosted trades before judging
_EDGE_THRESHOLD_PP      = 5.0  # model must beat neutral by ≥5 percentage points


def _empty_bucket() -> dict:
    return {"trades": 0, "wins": 0}


def _empty_state() -> dict:
    return {
        "boosted":   _empty_bucket(),
        "penalized": _empty_bucket(),
        "neutral":   _empty_bucket(),
        "daily": [],       # [{date, boosted_wr, penalized_wr, neutral_wr, trades}]
        "_today": {},      # transient day accumulator — not persisted as final
    }


class MLPerformanceTracker:
    """Thread-safe ML performance tracker. Use the module-level `tracker` singleton."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._state = _empty_state()
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, ml_boost: float, won: bool, pair: str = "") -> None:
        """Record a closed trade outcome. Call once per close."""
        with self._lock:
            bucket = self._bucket_key(ml_boost)
            self._state[bucket]["trades"] += 1
            if won:
                self._state[bucket]["wins"] += 1

            today = date.today().isoformat()
            td = self._state.setdefault("_today", {})
            if td.get("date") != today:
                # New day — flush yesterday's snapshot
                if td.get("date"):
                    self._flush_day(td)
                self._state["_today"] = {"date": today, "boosted": _empty_bucket(),
                                         "penalized": _empty_bucket(), "neutral": _empty_bucket()}
                td = self._state["_today"]

            td[bucket]["trades"] += 1
            if won:
                td[bucket]["wins"] += 1

            self._save()
            logger.debug(f"[ML Perf] {pair} boost={ml_boost:+.0f} won={won} bucket={bucket}")

    def get_stats(self) -> dict:
        """Return summary dict suitable for dashboard / MLflow logging."""
        with self._lock:
            b = self._state["boosted"]
            p = self._state["penalized"]
            n = self._state["neutral"]

            boosted_wr    = _wr(b)
            penalized_wr  = _wr(p)
            neutral_wr    = _wr(n)

            adding_value = (
                b["trades"] >= _MIN_TRADES_FOR_VERDICT
                and boosted_wr - neutral_wr >= _EDGE_THRESHOLD_PP
            )
            hurting_value = (
                p["trades"] >= _MIN_TRADES_FOR_VERDICT
                and neutral_wr - penalized_wr >= _EDGE_THRESHOLD_PP
            )

            return {
                "boosted_trades":   b["trades"],
                "boosted_wins":     b["wins"],
                "boosted_wr":       round(boosted_wr, 1),
                "penalized_trades": p["trades"],
                "penalized_wins":   p["wins"],
                "penalized_wr":     round(penalized_wr, 1),
                "neutral_trades":   n["trades"],
                "neutral_wins":     n["wins"],
                "neutral_wr":       round(neutral_wr, 1),
                "model_adding_value":  adding_value,
                "model_hurting_value": hurting_value,
                "verdict": self._verdict(boosted_wr, penalized_wr, neutral_wr,
                                         b["trades"], p["trades"]),
                "daily": list(self._state.get("daily", []))[-30:],  # last 30 days
            }

    def log_to_mlflow(self) -> None:
        """Log current performance stats to the active MLflow run (call after train)."""
        try:
            import mlflow
            stats = self.get_stats()
            mlflow.log_metric("perf_boosted_wr",   stats["boosted_wr"])
            mlflow.log_metric("perf_penalized_wr", stats["penalized_wr"])
            mlflow.log_metric("perf_neutral_wr",   stats["neutral_wr"])
            mlflow.log_metric("perf_boosted_n",    stats["boosted_trades"])
            mlflow.log_metric("perf_model_value",  1.0 if stats["model_adding_value"] else 0.0)
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bucket_key(boost: float) -> str:
        if boost > 0:
            return "boosted"
        if boost < 0:
            return "penalized"
        return "neutral"

    @staticmethod
    def _verdict(b_wr: float, p_wr: float, n_wr: float,
                 b_n: int, p_n: int) -> str:
        if b_n < _MIN_TRADES_FOR_VERDICT:
            return f"insufficient data ({b_n}/{_MIN_TRADES_FOR_VERDICT} boosted trades)"
        delta = b_wr - n_wr
        if delta >= _EDGE_THRESHOLD_PP:
            return f"[+] boost +{delta:.1f}pp vs neutral — model is adding value"
        if delta <= -_EDGE_THRESHOLD_PP:
            return f"[-] boost {delta:.1f}pp vs neutral — model may be hurting"
        return f"[=] boost {delta:+.1f}pp vs neutral — neutral (need >={_EDGE_THRESHOLD_PP}pp)"

    def _flush_day(self, td: dict) -> None:
        """Append yesterday's bucket to the daily history list."""
        snap = {
            "date":          td["date"],
            "boosted_wr":   round(_wr(td["boosted"]), 1),
            "penalized_wr": round(_wr(td["penalized"]), 1),
            "neutral_wr":   round(_wr(td["neutral"]), 1),
            "trades":       (td["boosted"]["trades"]
                             + td["penalized"]["trades"]
                             + td["neutral"]["trades"]),
        }
        self._state.setdefault("daily", []).append(snap)
        # Keep last 90 days
        self._state["daily"] = self._state["daily"][-90:]

    def _save(self) -> None:
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v for k, v in self._state.items() if k != "_today"}
            _PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"[ML Perf] Save failed: {e}")

    def _load(self) -> None:
        try:
            if _PATH.exists():
                data = json.loads(_PATH.read_text())
                self._state.update(data)
        except Exception as e:
            logger.debug(f"[ML Perf] Load failed (starting fresh): {e}")


def _wr(bucket: dict) -> float:
    if bucket["trades"] == 0:
        return 0.0
    return bucket["wins"] / bucket["trades"] * 100


# Module-level singleton
tracker = MLPerformanceTracker()
