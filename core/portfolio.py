"""
Phase 5 — Portfolio-level position management.

Guards against:
  1. Currency concentration — max 2 open positions sharing a base/quote currency
  2. Correlated pairs going same direction (EUR/USD long + GBP/USD long is OK,
     EUR/USD long + USD/CHF long is not — that's a double USD short)
  3. Strategy-type limits — max 2 TREND + 1 BREAKOUT open simultaneously

This runs BEFORE order execution (checked by order_manager) and BEFORE
re-entry (checked by lifecycle).
"""

from __future__ import annotations

from typing import Optional

from loguru import logger


# Pairs that move in the same direction (positive correlation)
_POSITIVE_CORR: list[frozenset[str]] = [
    frozenset({"EURUSD", "GBPUSD"}),
    frozenset({"EURUSD", "AUDUSD"}),
    frozenset({"GBPUSD", "AUDUSD"}),
    frozenset({"USDJPY", "USDCHF"}),
    frozenset({"XAUUSD", "AUDUSD"}),
]

# Pairs that move in OPPOSITE directions (negative correlation):
# holding both in the same direction is a partial hedge — wasteful but not blocked
_NEGATIVE_CORR: list[frozenset[str]] = [
    frozenset({"EURUSD", "USDCHF"}),
    frozenset({"GBPUSD", "USDCHF"}),
]

_MAX_SAME_CURRENCY_POSITIONS = 2
_MAX_TREND_POSITIONS         = 2
_MAX_BREAKOUT_POSITIONS      = 1


def _base_quote(pair: str) -> tuple[str, str]:
    """Extract base and quote currency from a pair like EURUSDm."""
    p = pair.upper().rstrip("M").rstrip("_")
    if "XAU" in p:
        return "XAU", "USD"
    if "USTEC" in p or "NAS" in p or "US30" in p:
        return "IDX", "USD"
    if len(p) == 6:
        return p[:3], p[3:]
    return p[:3], p[3:6]  # best effort


def _pair_base(pair: str) -> str:
    """e.g. EURUSDm → EURUSD"""
    return pair.upper().rstrip("M").rstrip("_")


class PortfolioManager:
    """
    Checks whether adding a new position is safe at the portfolio level.
    Reads from capital.open_positions (passed in) — no internal state.
    """

    def can_open(
        self,
        pair: str,
        direction: str,
        open_positions: dict,          # capital.open_positions
        strategy_label: str = "TREND", # "TREND" | "BKT"
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        open_positions: {pair: {"direction": "long"|"short", "strategy": "TREND"|"BKT", ...}}
        """
        base, quote = _base_quote(pair)
        pair_base   = _pair_base(pair)

        # ── 1. Currency concentration ─────────────────────────────
        same_currency_count = 0
        for op, meta in open_positions.items():
            ob, oq = _base_quote(op)
            if ob == base or oq == base or ob == quote or oq == quote:
                same_currency_count += 1

        if same_currency_count >= _MAX_SAME_CURRENCY_POSITIONS:
            msg = f"Currency concentration: {base}/{quote} already has {same_currency_count} open"
            logger.debug(f"[Portfolio] Block — {msg}")
            return False, msg

        # ── 2. Negative correlation double-up ────────────────────
        for op, meta in open_positions.items():
            op_base = _pair_base(op)
            op_dir  = meta.get("direction", "")
            pair_set = frozenset({pair_base, op_base})

            if pair_set in _NEGATIVE_CORR and op_dir == direction:
                # e.g. EUR/USD long + USD/CHF long = both short USD = redundant
                msg = f"Negative-corr overlap: {pair_base} {direction} + {op_base} {op_dir}"
                logger.debug(f"[Portfolio] Block — {msg}")
                return False, msg

        # ── 3. Strategy exposure limits ───────────────────────────
        trend_count    = sum(1 for m in open_positions.values() if m.get("strategy") == "TREND")
        breakout_count = sum(1 for m in open_positions.values() if m.get("strategy") == "BKT")

        if strategy_label == "TREND" and trend_count >= _MAX_TREND_POSITIONS:
            msg = f"Max TREND positions reached ({trend_count})"
            logger.debug(f"[Portfolio] Block — {msg}")
            return False, msg

        if strategy_label == "BKT" and breakout_count >= _MAX_BREAKOUT_POSITIONS:
            msg = f"Max BREAKOUT positions reached ({breakout_count})"
            logger.debug(f"[Portfolio] Block — {msg}")
            return False, msg

        return True, "ok"

    def portfolio_summary(self, open_positions: dict) -> str:
        """One-line summary for logging/dashboard."""
        if not open_positions:
            return "No positions"
        trend_n    = sum(1 for m in open_positions.values() if m.get("strategy") == "TREND")
        breakout_n = sum(1 for m in open_positions.values() if m.get("strategy") == "BKT")
        pairs = ", ".join(open_positions.keys())
        return f"{len(open_positions)} open ({trend_n}T/{breakout_n}B): {pairs}"


# Singleton
portfolio = PortfolioManager()
