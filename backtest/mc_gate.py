"""
Monte Carlo Gate — enforces MC validation before a strategy can go live.

A strategy is blocked from promotion to live if its Monte Carlo result
shows a ruin probability >= 5%. This gate is checked by the research pipeline
before auto-promoting any strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class MCGateResult:
    allowed: bool
    ruin_pct: float
    reason: str


def check_mc_gate(pair: str, days: int = 90) -> MCGateResult:
    """
    Run Monte Carlo (or load cached result) and check against the 5% ruin gate.
    Returns MCGateResult(allowed=True) only if ruin_pct < 5%.

    Runs 500 simulations (faster than the full 1000 for gate checks).
    """
    try:
        from backtest.engine import BacktestEngine
        from backtest.montecarlo import MonteCarlo
        from config.settings import settings

        # Run a quick backtest to get trade list
        engine = BacktestEngine(pair=pair, days=days, min_score=settings.min_confluence_score)
        result = engine.run()

        if not result or not result.trades:
            return MCGateResult(
                allowed=False,
                ruin_pct=100.0,
                reason=f"No backtest trades for {pair} over {days}d — cannot validate",
            )

        if len(result.trades) < 20:
            return MCGateResult(
                allowed=False,
                ruin_pct=100.0,
                reason=f"Insufficient trades ({len(result.trades)}) — need ≥ 20 for MC gate",
            )

        # Run MC
        mc = MonteCarlo(
            trades=result.trades,
            initial_balance=result.starting_balance,
            n_simulations=500,
            ruin_threshold=0.20,
        )
        mc_result = mc.run()
        ruin_pct = mc_result.ruin_probability * 100

        if ruin_pct >= 5.0:
            reason = (
                f"MC gate BLOCKED: ruin probability {ruin_pct:.1f}% ≥ 5.0% "
                f"({mc_result.n_simulations} sims, {len(result.trades)} trades)"
            )
            logger.warning(f"[MCGate] {pair}: {reason}")
            return MCGateResult(allowed=False, ruin_pct=ruin_pct, reason=reason)

        reason = (
            f"MC gate PASSED: ruin probability {ruin_pct:.1f}% < 5.0% "
            f"({mc_result.n_simulations} sims)"
        )
        logger.info(f"[MCGate] {pair}: {reason}")
        return MCGateResult(allowed=True, ruin_pct=ruin_pct, reason=reason)

    except Exception as e:
        logger.error(f"[MCGate] Error for {pair}: {e}")
        return MCGateResult(
            allowed=False,
            ruin_pct=100.0,
            reason=f"MC gate error: {e}",
        )
