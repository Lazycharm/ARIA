"""
Weekly Learning Report — auto-generated every Sunday 00:01 UTC.

Rules-based (no AI call, $0 cost). Pulls from:
  - ml/performance.py   — model verdict + bucket win rates
  - db/ml_meta.json     — feature importances
  - core/adaptive_learning.py — per-pair thresholds + lot multipliers
  - db/aria.db          — trade history for the last 7 days

Writes to: Obsidian/03 Projects/ARIA/Lessons Learned/YYYY-WXX.md
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger


def _meta_importances() -> list[tuple[str, float]]:
    """Top 5 features from last training run."""
    path = Path("./db/ml_meta.json")
    if not path.exists():
        return []
    try:
        meta = json.loads(path.read_text())
        imps = meta.get("feature_importances", {})
        return sorted(imps.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        return []


def _ml_meta_summary() -> dict:
    """Summary of last training run metadata."""
    path = Path("./db/ml_meta.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _week_trades() -> list[dict]:
    """Closed trades from the last 7 days via SQLAlchemy."""
    try:
        from db.session import get_session
        from db.models import Trade
        cutoff = datetime.utcnow() - timedelta(days=7)
        with get_session() as db:
            trades = (db.query(Trade)
                      .filter(Trade.status == "closed", Trade.opened_at >= cutoff)
                      .all())
            return [
                {
                    "pair":      t.pair,
                    "direction": t.direction,
                    "pnl":       t.pnl or 0.0,
                    "score":     t.score or 0.0,
                    "ml_score":  t.ml_score or 0.0,
                    "opened_at": t.opened_at,
                    "closed_at": t.closed_at,
                }
                for t in trades
            ]
    except Exception as e:
        logger.debug(f"[WeeklyReport] DB fetch failed: {e}")
        return []


def _per_pair_summary(trades: list[dict]) -> dict[str, dict]:
    """Group trade outcomes by pair."""
    by_pair: dict[str, dict] = {}
    for t in trades:
        pair = t["pair"]
        if pair not in by_pair:
            by_pair[pair] = {"wins": 0, "losses": 0, "pnl": 0.0}
        by_pair[pair]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_pair[pair]["wins"] += 1
        else:
            by_pair[pair]["losses"] += 1
    return by_pair


def build_report(week_label: str, dt: datetime) -> str:
    """Build the full weekly learning report as a Markdown string."""
    from ml.performance import tracker as ml_perf
    from core.adaptive_learning import adaptive

    week_start = (dt - timedelta(days=dt.weekday() + 1)).strftime("%Y-%m-%d")
    week_end   = dt.strftime("%Y-%m-%d")

    trades     = _week_trades()
    pair_stats = _per_pair_summary(trades)
    ml_stats   = ml_perf.get_stats()
    meta       = _ml_meta_summary()
    importances = _meta_importances()
    al_stats   = adaptive.all_stats()
    global_conservative = adaptive.is_global_conservative()

    total_trades = len(trades)
    total_wins   = sum(1 for t in trades if t["pnl"] > 0)
    total_losses = total_trades - total_wins
    week_pnl     = sum(t["pnl"] for t in trades)
    week_wr      = (total_wins / total_trades * 100) if total_trades else 0.0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss   = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf           = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    avg_score    = (sum(t["score"] for t in trades) / total_trades) if total_trades else 0.0

    # Best and worst pair by PnL
    best_pair = max(pair_stats.items(), key=lambda x: x[1]["pnl"], default=(None, {}))
    worst_pair = min(pair_stats.items(), key=lambda x: x[1]["pnl"], default=(None, {}))

    # ── Section builders ──────────────────────────────────────────────────────

    trade_section = ""
    if not trades:
        trade_section = "*No closed trades this week.*\n"
    else:
        trade_section = f"""| Metric | Value |
|--------|-------|
| Total trades | `{total_trades}` |
| Wins / Losses | `{total_wins}W / {total_losses}L` |
| Win rate | `{week_wr:.1f}%` |
| Net P&L | `${week_pnl:+.2f}` |
| Profit factor | `{pf:.2f}` |
| Avg confluence score | `{avg_score:.1f}` |
"""

    pair_rows = ""
    for pair, ps in sorted(pair_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        n = ps["wins"] + ps["losses"]
        wr = ps["wins"] / n * 100 if n else 0
        pair_rows += f"| {pair} | {ps['wins']}W / {ps['losses']}L | {wr:.0f}% | ${ps['pnl']:+.2f} |\n"

    pair_section = ""
    if pair_rows:
        pair_section = f"""| Pair | W/L | WR | P&L |
|------|-----|----|-----|
{pair_rows}"""
    else:
        pair_section = "*No trades recorded.*\n"

    # ML section
    ml_model_ready = bool(Path("./db/ml_model.pkl").exists())
    ml_backend     = meta.get("backend", "—")
    ml_n_samples   = meta.get("n_samples", "—")
    ml_cv_acc      = meta.get("cv_accuracy", meta.get("train_accuracy", "—"))
    ml_cv_str      = f"{ml_cv_acc:.1%}" if isinstance(ml_cv_acc, float) else str(ml_cv_acc)

    imp_rows = "\n".join(
        f"| `{feat}` | {val:.4f} |" for feat, val in importances
    ) or "| — | — |"

    ml_section = f"""**Model status:** {"ready" if ml_model_ready else "not yet trained"} | Backend: `{ml_backend}` | Samples: `{ml_n_samples}` | CV accuracy: `{ml_cv_str}`

### Prediction value (live performance)

| Bucket | Trades | Win Rate |
|--------|--------|----------|
| Boosted (ML > 0) | {ml_stats['boosted_trades']} | {ml_stats['boosted_wr']:.1f}% |
| Penalized (ML < 0) | {ml_stats['penalized_trades']} | {ml_stats['penalized_wr']:.1f}% |
| Neutral (ML = 0) | {ml_stats['neutral_trades']} | {ml_stats['neutral_wr']:.1f}% |

**Verdict:** {ml_stats['verdict']}

### Top 5 features (last training run)

| Feature | Importance |
|---------|-----------|
{imp_rows}
"""

    # Adaptive learning section
    al_rows = ""
    for pair, ps in sorted(al_stats.items(), key=lambda x: x[1].win_rate, reverse=True):
        trend = "rising" if ps.min_score > 70 else ("falling" if ps.min_score < 70 else "normal")
        al_rows += (
            f"| {pair} | {ps.wins}W/{ps.losses}L | "
            f"{ps.win_rate:.0f}% | `{ps.min_score:.0f}` ({trend}) | "
            f"`{ps.lot_multiplier:.2f}x` | `${ps.avg_pnl:+.2f}` |\n"
        )

    al_section = ""
    if al_rows:
        al_section = f"""| Pair | W/L | WR | Min Score | Lot Mult | Avg P&L |
|------|-----|----|-----------|----------|---------|
{al_rows}"""
    else:
        al_section = "*No adaptive data yet — needs live trades.*\n"

    global_mode_str = (
        "**GLOBAL CONSERVATIVE MODE ACTIVE** — all thresholds +10, lots x0.5"
        if global_conservative
        else "Normal — no global conservative mode active"
    )

    # Pattern library
    try:
        from core.pattern_library import get_patterns
        patterns = get_patterns()
    except Exception:
        patterns = {}

    pattern_section = ""
    if patterns.get("total_trades", 0) >= 10:
        drift = patterns.get("drift", {})
        hold  = patterns.get("hold_time", {})

        def _tbl(by_key: dict, label: str) -> str:
            if not by_key:
                return ""
            rows = ""
            for k, v in sorted(by_key.items(), key=lambda x: x[1]["wr"], reverse=True):
                rows += f"| {k} | {v['trades']} | {v['wr']:.0f}% | ${v['avg_pnl']:+.4f} |\n"
            return f"\n**{label}**\n\n| | Trades | WR | Avg P&L |\n|---|---|---|---|\n{rows}"

        pattern_section = (
            f"**All-time:** {patterns['overall']['trades']} trades | "
            f"{patterns['overall']['wr']:.1f}% WR | ${patterns['overall']['total_pnl']:+.2f}\n\n"
            f"**Drift (last {min(30, patterns['overall']['trades'])} vs all-time):** "
            f"{drift.get('wr_delta', 0):+.1f}pp — {drift.get('label', '—')}\n\n"
            f"**Hold time:** Winners avg {hold.get('avg_winner_minutes', 0):.0f}m | "
            f"Losers avg {hold.get('avg_loser_minutes', 0):.0f}m\n"
            + _tbl(patterns.get("by_regime", {}),    "By Regime")
            + _tbl(patterns.get("by_session", {}),   "By Session")
            + _tbl(patterns.get("by_score_band", {}), "By Score Band")
            + _tbl(patterns.get("by_direction", {}), "By Direction")
        )

        pattern_insights = patterns.get("insights", [])
        observations = list(pattern_insights)
    else:
        pattern_section = f"*{patterns.get('total_trades', 0)} trades recorded — need 10+ for pattern analysis.*\n"
        # Key observations
        observations = []
    if week_wr >= 60 and total_trades >= 5:
        observations.append(f"Strong week: {week_wr:.0f}% win rate across {total_trades} trades")
    elif week_wr < 40 and total_trades >= 5:
        observations.append(f"Difficult week: {week_wr:.0f}% win rate — review confluence filters")
    if pf > 1.5:
        observations.append(f"Good profit factor: {pf:.2f} — exits working well")
    elif 0 < pf < 1.0 and total_trades >= 3:
        observations.append(f"Profit factor {pf:.2f} < 1 — losses outweigh wins in dollar terms")
    if best_pair[0] and best_pair[1].get("pnl", 0) > 0:
        observations.append(f"Best pair: {best_pair[0]} +${best_pair[1]['pnl']:.2f}")
    if worst_pair[0] and worst_pair[1].get("pnl", 0) < 0:
        observations.append(f"Worst pair: {worst_pair[0]} ${worst_pair[1]['pnl']:.2f}")
    if ml_stats["model_adding_value"]:
        delta = ml_stats["boosted_wr"] - ml_stats["neutral_wr"]
        observations.append(f"ML model confirmed positive: boosted trades +{delta:.1f}pp vs neutral")
    if ml_stats.get("model_hurting_value"):
        observations.append("ML model hurting: penalized trades winning more than expected — review thresholds")
    if not observations:
        observations.append("Insufficient data this week to draw conclusions")

    obs_str = "\n".join(f"- {o}" for o in observations)

    # ── Assemble full report ──────────────────────────────────────────────────
    return f"""---
tags: [ARIA, learning, weekly-review]
date: {week_end}
week: {week_label}
trades: {total_trades}
pnl: {week_pnl:.2f}
win_rate: {week_wr:.1f}
---

# ARIA Weekly Learning Report — {week_label}
*Period: {week_start} to {week_end} | Generated: {dt.strftime("%Y-%m-%d %H:%M UTC")}*

---

## 1. Week Summary

{trade_section}

## 2. Performance by Pair

{pair_section}

## 3. ML Model

{ml_section}

## 4. Adaptive Learning State

{al_section}

**Global mode:** {global_mode_str}

## 5. Pattern Analysis

{pattern_section}

## 6. Key Observations

{obs_str}

---

## 7. Action Items for Next Week

- [ ] Review worst pair setups — are we entering in wrong session?
- [ ] Check if ML feature importances align with confluence scoring weights
- [ ] If global conservative mode triggered — investigate root cause before releasing

---
*[[ARIA Overview]] | Auto-generated by ARIA weekly report engine*
"""
