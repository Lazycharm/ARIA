"""
Trade Analysis Pipeline — pattern library built from live trade closes.

Appends a structured record to db/pattern_library.jsonl on every close.
Aggregates into patterns on read — no separate DB required.

Patterns tracked:
  - By regime   (TREND / BREAKOUT / WAIT)
  - By session  (london / overlap / asian / dead)
  - By score band (65-75 / 75-85 / 85+)
  - By direction (long / short)
  - By pair     (all pairs ranked)
  - Recency     (last 30 vs all-time — detects drift)

Call record() on every trade close.
Call get_patterns() for weekly report, dashboard, and MLflow.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

_PATH   = Path("./db/pattern_library.jsonl")
_lock   = threading.Lock()


# ── Record ────────────────────────────────────────────────────────────────────

def record(
    pair:          str,
    direction:     str,
    score:         float,
    regime:        str,
    session:       str,
    ml_boost:      float,
    pnl:           float,
    hold_minutes:  float,
    entry:         float = 0.0,
    sl:            float = 0.0,
    tp1:           float = 0.0,
) -> None:
    """Append one closed trade to the pattern library. Call from lifecycle."""
    won = pnl > 0
    row = {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "pair":         pair,
        "direction":    direction,
        "score":        round(score, 1),
        "score_band":   _score_band(score),
        "regime":       regime.lower(),
        "session":      session.lower(),
        "ml_boost":     round(ml_boost, 1),
        "pnl":          round(pnl, 4),
        "won":          won,
        "hold_minutes": round(hold_minutes, 1),
        "exit_type":    "tp" if won else "sl",
    }
    with _lock:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    logger.debug(
        f"[Pattern] {pair} {direction} score={score:.0f} regime={regime} "
        f"session={session} pnl=${pnl:+.2f} hold={hold_minutes:.0f}m"
    )


# ── Aggregate ─────────────────────────────────────────────────────────────────

def _load_all() -> list[dict]:
    if not _PATH.exists():
        return []
    with _lock:
        lines = _PATH.read_text(encoding="utf-8").strip().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _bucket_stats(rows: list[dict]) -> dict:
    """Win rate + avg PnL + trade count for a subset of rows."""
    if not rows:
        return {"trades": 0, "wins": 0, "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
    wins     = sum(1 for r in rows if r["won"])
    total_pnl = sum(r["pnl"] for r in rows)
    return {
        "trades":    len(rows),
        "wins":      wins,
        "wr":        round(wins / len(rows) * 100, 1),
        "avg_pnl":   round(total_pnl / len(rows), 4),
        "total_pnl": round(total_pnl, 4),
    }


def _top_n(by_key: dict[str, dict], key: str = "wr", n: int = 3) -> list[tuple]:
    """Return top-N entries sorted by key (descending), minimum 5 trades."""
    eligible = {k: v for k, v in by_key.items() if v["trades"] >= 5}
    return sorted(eligible.items(), key=lambda x: x[1][key], reverse=True)[:n]


def get_patterns(recent_n: int = 30) -> dict:
    """
    Aggregate all pattern library records into structured insights.
    recent_n: number of most recent trades to compare against all-time.
    """
    rows = _load_all()
    if not rows:
        return {"total_trades": 0, "message": "No trades recorded yet"}

    recent = rows[-recent_n:] if len(rows) >= recent_n else rows

    # ── Group by dimension ────────────────────────────────────────────────────
    def group(field: str, data: list[dict]) -> dict[str, dict]:
        buckets: dict[str, list[dict]] = {}
        for r in data:
            k = str(r.get(field, "unknown"))
            buckets.setdefault(k, []).append(r)
        return {k: _bucket_stats(v) for k, v in buckets.items()}

    by_regime    = group("regime",     rows)
    by_session   = group("session",    rows)
    by_score_band = group("score_band", rows)
    by_direction = group("direction",  rows)
    by_pair      = group("pair",       rows)

    by_regime_r    = group("regime",    recent)
    by_session_r   = group("session",   recent)
    by_direction_r = group("direction", recent)

    overall      = _bucket_stats(rows)
    overall_r    = _bucket_stats(recent)

    # ── Hold time analysis ────────────────────────────────────────────────────
    wins_hold  = [r["hold_minutes"] for r in rows if r["won"]]
    loss_hold  = [r["hold_minutes"] for r in rows if not r["won"]]
    avg_win_hold  = round(sum(wins_hold)  / len(wins_hold),  1) if wins_hold  else 0.0
    avg_loss_hold = round(sum(loss_hold) / len(loss_hold), 1) if loss_hold else 0.0

    # ── Drift detection (recent vs all-time) ──────────────────────────────────
    wr_delta    = round(overall_r["wr"] - overall["wr"], 1)
    drift_label = (
        "improving" if wr_delta >= 5 else
        "degrading" if wr_delta <= -5 else
        "stable"
    )

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = _generate_insights(
        by_regime, by_session, by_score_band, by_direction,
        overall, overall_r, avg_win_hold, avg_loss_hold, drift_label
    )

    return {
        "total_trades":    overall["trades"],
        "overall":         overall,
        "recent":          overall_r,
        "drift":           {"wr_delta": wr_delta, "label": drift_label},
        "by_regime":       by_regime,
        "by_session":      by_session,
        "by_score_band":   by_score_band,
        "by_direction":    by_direction,
        "by_pair":         by_pair,
        "by_regime_recent":   by_regime_r,
        "by_session_recent":  by_session_r,
        "hold_time": {
            "avg_winner_minutes": avg_win_hold,
            "avg_loser_minutes":  avg_loss_hold,
        },
        "top_pairs":  _top_n(by_pair, "wr"),
        "worst_pairs": sorted(
            {k: v for k, v in by_pair.items() if v["trades"] >= 5}.items(),
            key=lambda x: x[1]["wr"]
        )[:3],
        "insights": insights,
    }


def _generate_insights(
    by_regime:  dict, by_session: dict, by_score_band: dict,
    by_direction: dict, overall: dict, recent: dict,
    avg_win_hold: float, avg_loss_hold: float, drift: str
) -> list[str]:
    """Auto-generate human-readable pattern observations."""
    out = []
    total = overall["trades"]
    if total < 10:
        return [f"Insufficient data — {total} trades recorded (need 10+)"]

    # Regime insight
    best_regime = max(by_regime.items(), key=lambda x: x[1]["wr"], default=(None, {}))
    worst_regime = min(by_regime.items(), key=lambda x: x[1]["wr"], default=(None, {}))
    if best_regime[0] and best_regime[1]["trades"] >= 5:
        out.append(
            f"Best regime: {best_regime[0].upper()} "
            f"({best_regime[1]['wr']:.0f}% WR over {best_regime[1]['trades']} trades)"
        )
    if worst_regime[0] and worst_regime[1]["trades"] >= 5 and worst_regime[0] != best_regime[0]:
        out.append(
            f"Avoid: {worst_regime[0].upper()} regime "
            f"({worst_regime[1]['wr']:.0f}% WR — lowest performer)"
        )

    # Session insight
    best_session = max(by_session.items(), key=lambda x: x[1]["wr"], default=(None, {}))
    if best_session[0] and best_session[1]["trades"] >= 5:
        out.append(
            f"Best session: {best_session[0].capitalize()} "
            f"({best_session[1]['wr']:.0f}% WR)"
        )

    # Score band
    best_band = max(by_score_band.items(), key=lambda x: x[1]["wr"], default=(None, {}))
    if best_band[0] and best_band[1]["trades"] >= 5:
        out.append(f"Best score band: {best_band[0]} ({best_band[1]['wr']:.0f}% WR)")

    # Direction bias
    long_wr  = by_direction.get("long",  {}).get("wr", 0)
    short_wr = by_direction.get("short", {}).get("wr", 0)
    if abs(long_wr - short_wr) >= 10:
        better    = "LONG" if long_wr > short_wr else "SHORT"
        delta     = abs(long_wr - short_wr)
        out.append(f"Direction bias: {better} trades outperform by {delta:.0f}pp")

    # Hold time
    if avg_win_hold and avg_loss_hold:
        if avg_win_hold < avg_loss_hold:
            out.append(
                f"Winners close faster ({avg_win_hold:.0f}m) than losers ({avg_loss_hold:.0f}m) "
                "— consider tighter time exits"
            )
        else:
            out.append(
                f"Winners held longer ({avg_win_hold:.0f}m vs {avg_loss_hold:.0f}m) "
                "— good trend-following behaviour"
            )

    # Drift
    if drift == "improving":
        out.append(f"Trend: recent win rate improving vs all-time (+{recent['wr'] - overall['wr']:.0f}pp)")
    elif drift == "degrading":
        out.append(f"Trend: recent win rate degrading vs all-time ({recent['wr'] - overall['wr']:.0f}pp) — review")

    return out if out else ["Patterns still forming — keep collecting data"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_band(score: float) -> str:
    if score >= 85:
        return "85+"
    if score >= 75:
        return "75-85"
    if score >= 65:
        return "65-75"
    return "<65"


def trade_count() -> int:
    if not _PATH.exists():
        return 0
    with _lock:
        return sum(1 for _ in _PATH.open(encoding="utf-8"))
