"""
Performance Attribution — identifies which confluence components drive win rate.

Compares feature values between winning and losing trades from the ML sample store.
Returns ranked list of predictive features vs noise.

Uses stored ML samples (db/ml_samples.jsonl) to correlate features with outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

_SAMPLES_PATH = Path("./db/ml_samples.jsonl")


def _load_samples() -> list[dict]:
    if not _SAMPLES_PATH.exists():
        return []
    rows = []
    for line in _SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def compute_attribution(min_samples: int = 30) -> list[dict]:
    """
    For each feature, compute win rate when feature is HIGH vs LOW.
    Returns list sorted by predictive power (descending).

    Each entry:
      {"feature": str, "high_wr": float, "low_wr": float, "delta_pp": float,
       "verdict": "predictive"|"noise"|"inverse", "n_high": int, "n_low": int}
    """
    rows = _load_samples()
    if len(rows) < min_samples:
        return []

    wins  = [r for r in rows if r.get("won", False)]
    loses = [r for r in rows if not r.get("won", False)]

    if not wins or not loses:
        return []

    # Identify numeric feature keys (exclude metadata fields)
    skip = {"won", "pnl", "pair", "ts", "ml_boost"}
    feature_keys = [k for k in rows[0].keys() if k not in skip and isinstance(rows[0][k], (int, float))]

    results = []
    for feat in feature_keys:
        vals  = [(r.get(feat, 0.0), r.get("won", False)) for r in rows if feat in r]
        if not vals:
            continue

        numeric = [v for v, _ in vals if isinstance(v, (int, float))]
        if not numeric:
            continue

        median = sorted(numeric)[len(numeric) // 2]
        high_trades = [(v, w) for v, w in vals if v >= median]
        low_trades  = [(v, w) for v, w in vals if v < median]

        if len(high_trades) < 5 or len(low_trades) < 5:
            continue

        high_wr = sum(1 for _, w in high_trades if w) / len(high_trades) * 100
        low_wr  = sum(1 for _, w in low_trades  if w) / len(low_trades)  * 100
        delta   = high_wr - low_wr

        if abs(delta) >= 10:
            verdict = "predictive" if delta > 0 else "inverse"
        else:
            verdict = "noise"

        results.append({
            "feature":  feat,
            "high_wr":  round(high_wr, 1),
            "low_wr":   round(low_wr,  1),
            "delta_pp": round(delta,   1),
            "verdict":  verdict,
            "n_high":   len(high_trades),
            "n_low":    len(low_trades),
        })

    return sorted(results, key=lambda x: abs(x["delta_pp"]), reverse=True)


def attribution_summary(top_n: int = 5) -> str:
    """Human-readable summary of top predictive and inverse features."""
    attrs = compute_attribution()
    if not attrs:
        return "Insufficient samples for performance attribution (need 30+)."

    predictive = [a for a in attrs if a["verdict"] == "predictive"][:top_n]
    inverse    = [a for a in attrs if a["verdict"] == "inverse"][:3]
    noise      = [a for a in attrs if a["verdict"] == "noise"]

    lines = [f"Performance Attribution ({len(attrs)} features analyzed):"]
    if predictive:
        lines.append("  Predictive (high value → better WR):")
        for a in predictive:
            lines.append(f"    {a['feature']}: HIGH={a['high_wr']:.0f}% vs LOW={a['low_wr']:.0f}% (+{a['delta_pp']:.0f}pp)")
    if inverse:
        lines.append("  Inverse (high value → worse WR):")
        for a in inverse:
            lines.append(f"    {a['feature']}: HIGH={a['high_wr']:.0f}% vs LOW={a['low_wr']:.0f}% ({a['delta_pp']:.0f}pp)")
    if noise:
        lines.append(f"  Noise features: {', '.join(a['feature'] for a in noise)}")

    return "\n".join(lines)
