"""
Mistake Detector — analyzes the pattern library to surface systematic errors.

Examples:
  - "Traded against opposing BOS 8× — all losses"
  - "Short trades in London session: 20% WR — avoid"
  - "Score 65-75 trades: losing money overall — threshold too low"

Called from weekly_report.py and can be queried from dashboard.
Returns a list of human-readable mistake strings, each with severity.
"""

from __future__ import annotations

from typing import Optional


def detect_mistakes(recent_n: int = 50) -> list[dict]:
    """
    Scan the pattern library for systematic mistakes.
    Returns list of {"severity": "high"|"medium", "message": str, "stat": str}.
    """
    try:
        from core.pattern_library import get_patterns, _load_all
    except Exception:
        return []

    rows    = _load_all()
    if len(rows) < 10:
        return []

    patterns = get_patterns(recent_n=recent_n)
    mistakes: list[dict] = []

    # ── Mistake 1: Low score band losing money ────────────────────────────────
    by_score = patterns.get("by_score_band", {})
    if "<65" in by_score and by_score["<65"]["trades"] >= 5:
        band = by_score["<65"]
        if band["wr"] < 40:
            mistakes.append({
                "severity": "high",
                "message": f"Score <65 trades: {band['wr']:.0f}% WR over {band['trades']} trades — min_score threshold too low",
                "stat": f"WR={band['wr']:.0f}% PnL=${band['total_pnl']:+.2f}",
            })

    # ── Mistake 2: Direction bias — one direction consistently losing ─────────
    by_dir = patterns.get("by_direction", {})
    long_wr  = by_dir.get("long",  {}).get("wr", 50)
    short_wr = by_dir.get("short", {}).get("wr", 50)
    long_n   = by_dir.get("long",  {}).get("trades", 0)
    short_n  = by_dir.get("short", {}).get("trades", 0)
    if long_n >= 10 and long_wr < 35:
        mistakes.append({
            "severity": "high",
            "message": f"LONG trades: {long_wr:.0f}% WR over {long_n} trades — avoid long bias in current regime",
            "stat": f"WR={long_wr:.0f}%",
        })
    if short_n >= 10 and short_wr < 35:
        mistakes.append({
            "severity": "high",
            "message": f"SHORT trades: {short_wr:.0f}% WR over {short_n} trades — avoid short bias in current regime",
            "stat": f"WR={short_wr:.0f}%",
        })

    # ── Mistake 3: Dead session trades ────────────────────────────────────────
    by_session = patterns.get("by_session", {})
    for session, stats in by_session.items():
        if stats["trades"] >= 8 and stats["wr"] < 35:
            mistakes.append({
                "severity": "medium",
                "message": f"{session.capitalize()} session: {stats['wr']:.0f}% WR over {stats['trades']} trades — avoid trading this session",
                "stat": f"WR={stats['wr']:.0f}% PnL=${stats['total_pnl']:+.2f}",
            })

    # ── Mistake 4: WAIT regime trades still being taken ───────────────────────
    by_regime = patterns.get("by_regime", {})
    if "wait" in by_regime and by_regime["wait"]["trades"] >= 5:
        w = by_regime["wait"]
        if w["wr"] < 40:
            mistakes.append({
                "severity": "medium",
                "message": f"WAIT regime: {w['trades']} trades taken with {w['wr']:.0f}% WR — scanner leaking through WAIT filter",
                "stat": f"WR={w['wr']:.0f}%",
            })

    # ── Mistake 5: Losers held longer than winners ────────────────────────────
    hold = patterns.get("hold_time", {})
    avg_win  = hold.get("avg_winner_minutes", 0)
    avg_loss = hold.get("avg_loser_minutes", 0)
    if avg_win > 0 and avg_loss > 0 and avg_loss > avg_win * 1.5:
        mistakes.append({
            "severity": "medium",
            "message": f"Losers held {avg_loss:.0f}m avg vs winners {avg_win:.0f}m — cutting winners early, letting losers run",
            "stat": f"Win hold={avg_win:.0f}m Loss hold={avg_loss:.0f}m",
        })

    # ── Mistake 6: Significant recent degradation ─────────────────────────────
    drift = patterns.get("drift", {})
    if drift.get("label") == "degrading" and abs(drift.get("wr_delta", 0)) >= 10:
        mistakes.append({
            "severity": "high",
            "message": f"Performance degrading: recent WR {drift['wr_delta']:+.0f}pp vs all-time — market regime may have shifted",
            "stat": f"Delta={drift['wr_delta']:+.0f}pp",
        })

    # ── Mistake 7: Pair consistently underperforming ──────────────────────────
    worst = patterns.get("worst_pairs", [])
    for pair_name, stats in worst:
        if stats["trades"] >= 10 and stats["wr"] < 30:
            mistakes.append({
                "severity": "medium",
                "message": f"{pair_name}: {stats['wr']:.0f}% WR over {stats['trades']} trades — consider removing from watchlist",
                "stat": f"WR={stats['wr']:.0f}% PnL=${stats['total_pnl']:+.2f}",
            })

    return mistakes
