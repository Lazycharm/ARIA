"""
Strategy Scorer — ranks active strategies (TREND/BREAKOUT/WAIT) by rolling Sharpe.

Reads from db/pattern_library.jsonl where each trade record includes regime.
Computes per-strategy rolling Sharpe and flags underperformers for retirement.

Retirement gate: rolling 30-trade Sharpe < 0 → strategy disabled.
"""

from __future__ import annotations

import math
from typing import Optional

from loguru import logger


_RETIRE_SHARPE = 0.0   # Sharpe below this triggers retirement flag
_MIN_TRADES    = 10    # minimum trades before scoring is meaningful


def _rolling_sharpe(pnls: list[float], window: int = 30) -> float:
    """Rolling Sharpe ratio on a list of PnL values (last `window` trades)."""
    if len(pnls) < 5:
        return 0.0
    subset = pnls[-window:]
    mean   = sum(subset) / len(subset)
    if len(subset) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    return round(mean / std, 3) if std > 0 else 0.0


def score_strategies() -> list[dict]:
    """
    Compute rolling Sharpe for each strategy label found in the pattern library.
    Returns list sorted by Sharpe descending.

    Each entry:
      {"strategy": str, "trades": int, "sharpe": float,
       "win_rate": float, "avg_pnl": float, "retire": bool}
    """
    try:
        from core.pattern_library import _load_all
        rows = _load_all()
    except Exception:
        return []

    if not rows:
        return []

    # Group by regime (TREND/BREAKOUT/WAIT → strategy label)
    by_strategy: dict[str, list[float]] = {}
    for r in rows:
        strategy = r.get("regime", "unknown").upper()
        by_strategy.setdefault(strategy, []).append(r.get("pnl", 0.0))

    results = []
    for strategy, pnls in by_strategy.items():
        n        = len(pnls)
        wins     = sum(1 for p in pnls if p > 0)
        win_rate = wins / n * 100 if n else 0.0
        avg_pnl  = sum(pnls) / n if n else 0.0
        sharpe   = _rolling_sharpe(pnls)
        retire   = sharpe < _RETIRE_SHARPE and n >= _MIN_TRADES

        if retire:
            logger.warning(f"[StrategyScorer] {strategy} rolling Sharpe={sharpe:.3f} < {_RETIRE_SHARPE} — flagged for retirement")

        results.append({
            "strategy": strategy,
            "trades":   n,
            "sharpe":   sharpe,
            "win_rate": round(win_rate, 1),
            "avg_pnl":  round(avg_pnl,  4),
            "retire":   retire,
        })

    return sorted(results, key=lambda x: x["sharpe"], reverse=True)


def get_disabled_strategies() -> set[str]:
    """Return set of strategy labels that have been flagged for retirement."""
    return {s["strategy"].lower() for s in score_strategies() if s["retire"]}
