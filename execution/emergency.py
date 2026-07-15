"""
Emergency close-all: closes every open MT5 position instantly.

Call execute_emergency_close(capital, reason) from:
  - Dashboard "EMERGENCY CLOSE ALL" button
  - Telegram command handler

Workflow:
  1. Snapshot all open positions from MT5 feed
  2. Close each position via mt5_client.close_all()
  3. Register every close in CapitalManager so day P&L is accurate
  4. Set trading halted flag
  5. Send Telegram alert
"""

from __future__ import annotations

from loguru import logger


def execute_emergency_close(capital, reason: str = "Manual emergency close") -> dict:
    """
    Close all open MT5 positions and halt trading.

    Returns:
        {"closed": int, "n_total": int, "total_pnl": float, "pairs": list[str]}
    """
    from data.mt5_feed import feed
    from execution.mt5_client import mt5_client

    positions = feed.get_positions()
    n_total = len(positions)

    if n_total == 0:
        logger.info("[Emergency] No open positions to close")
        return {"closed": 0, "n_total": 0, "total_pnl": 0.0, "pairs": []}

    logger.warning(f"[Emergency] Closing {n_total} position(s). Reason: {reason}")

    n_closed = mt5_client.close_all(f"ARIA emergency: {reason[:40]}")

    total_pnl = 0.0
    pairs_closed: list[str] = []
    for pos in positions:
        pair = pos["pair"]
        pnl  = pos.get("pnl", 0.0)
        total_pnl += pnl
        pairs_closed.append(pair)
        try:
            capital.register_close(pair, pos.get("current", 0.0), pnl)
        except Exception as e:
            logger.debug(f"[Emergency] register_close({pair}): {e}")

    # Halt without triggering the generic halt Telegram message
    capital._trading_halted = True
    capital._halt_reason = reason

    try:
        from notifications.telegram import alert_emergency_close
        alert_emergency_close(
            n_closed=n_closed,
            n_total=n_total,
            total_pnl=round(total_pnl, 2),
            reason=reason,
        )
    except Exception as e:
        logger.debug(f"[Emergency] Telegram alert failed: {e}")

    logger.warning(
        f"[Emergency] Done — {n_closed}/{n_total} closed | PnL=${total_pnl:.2f} | Trading halted"
    )

    try:
        from core.risk_log import append_event, EMERGENCY_CLOSE
        append_event(
            EMERGENCY_CLOSE,
            f"Closed {n_closed}/{n_total} positions | P&L: ${total_pnl:+.2f}",
            balance=capital.balance,
            extra={"Reason": reason, "Pairs": ", ".join(pairs_closed) or "none"},
        )
    except Exception:
        pass

    return {
        "closed":    n_closed,
        "n_total":   n_total,
        "total_pnl": round(total_pnl, 2),
        "pairs":     pairs_closed,
    }
