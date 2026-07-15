"""
MT5 execution client — places, modifies, and closes orders.

Wraps MetaTrader5 order_send() with proper request structure,
deviation handling, and error reporting.
"""

from __future__ import annotations

import time
from typing import Optional

import MetaTrader5 as mt5
from loguru import logger

# Retcodes that warrant a retry with a fresh price
_RETRYABLE = {
    mt5.TRADE_RETCODE_REQUOTE,      # 10004 — price moved, retry with new price
    mt5.TRADE_RETCODE_PRICE_OFF,    # 10020 — price too far from market
    mt5.TRADE_RETCODE_REJECT,       # 10006 — broker rejected, often transient
    mt5.TRADE_RETCODE_TIMEOUT,      # 10010 — server timeout
}

from config.settings import settings
from data.mt5_feed import feed


class MT5OrderResult:
    def __init__(self, success: bool, ticket: int = 0, error: str = "") -> None:
        self.success = success
        self.ticket  = ticket
        self.error   = error

    def __repr__(self) -> str:
        if self.success:
            return f"OrderResult(ticket={self.ticket})"
        return f"OrderResult(FAILED: {self.error})"


class MT5Client:
    """Execution-only MT5 wrapper. All data queries go through mt5_feed."""

    def _connected(self) -> bool:
        return feed.ensure_connected()

    def place_market_order(
        self,
        pair: str,
        direction: str,         # "long" | "short"
        lots: float,
        sl: float,
        tp: float,
        comment: str = "ARIA",
        deviation_points: int = 20,
    ) -> MT5OrderResult:
        """Send a market order to MT5."""
        if settings.dry_run:
            logger.info(f"DRY RUN: {direction.upper()} {lots:.2f} {pair} SL={sl} TP={tp}")
            return MT5OrderResult(True, ticket=999999)

        if not self._connected():
            return MT5OrderResult(False, error="MT5 not connected")

        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            return MT5OrderResult(False, error=f"No tick for {pair}")

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "long" else tick.bid

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    pair,
            "volume":    lots,
            "type":      order_type,
            "price":     price,
            "sl":        sl,
            "tp":        tp,
            "deviation": deviation_points,
            "magic":     20260707,          # ARIA magic number
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            result = mt5.order_send(request)
            latency_ms = (time.perf_counter() - t0) * 1000

            if result is None:
                err = str(mt5.last_error())
                logger.error(f"order_send returned None (attempt {attempt}): {err}")
                if attempt < max_attempts:
                    time.sleep(0.3)
                    continue
                return MT5OrderResult(False, error=err)

            logger.debug(f"[Latency] {pair} order_send attempt {attempt}: {latency_ms:.0f}ms retcode={result.retcode}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                filled = getattr(result, "volume", lots)
                if filled < lots - 0.001:
                    # Partial fill — log prominently, proceed with filled volume
                    logger.warning(
                        f"[PartialFill] {pair} requested={lots:.2f} filled={filled:.2f} "
                        f"ticket={result.order} — proceeding with partial fill"
                    )
                else:
                    logger.info(
                        f"Order placed: {direction.upper()} {lots} {pair} @ {result.price} "
                        f"ticket={result.order} latency={latency_ms:.0f}ms attempt={attempt}"
                    )
                return MT5OrderResult(True, ticket=result.order)

            if result.retcode in _RETRYABLE and attempt < max_attempts:
                # Refresh price for next attempt
                tick = mt5.symbol_info_tick(pair)
                if tick:
                    request["price"] = tick.ask if direction == "long" else tick.bid
                logger.warning(
                    f"Retryable rejection: {pair} retcode={result.retcode} "
                    f"({result.comment}) — retrying ({attempt}/{max_attempts})"
                )
                time.sleep(0.2 * attempt)
                continue

            logger.error(f"Order rejected: {result.retcode} — {result.comment}")
            return MT5OrderResult(False, error=f"retcode={result.retcode} {result.comment}")

        return MT5OrderResult(False, error="Max retry attempts exceeded")

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Modify SL/TP on an open position."""
        if settings.dry_run:
            logger.info(f"DRY RUN: Modify ticket={ticket} SL={sl} TP={tp}")
            return True

        if not self._connected():
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found for modify")
            return False

        pos = positions[0]
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "sl":       sl if sl is not None else pos.sl,
            "tp":       tp if tp is not None else pos.tp,
            "position": ticket,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.debug(f"Modified ticket={ticket} SL={sl} TP={tp}")
            return True
        logger.warning(f"Modify failed: ticket={ticket} retcode={result.retcode if result else 'None'}")
        return False

    def close_position(self, ticket: int, lots: Optional[float] = None) -> MT5OrderResult:
        """Close a position (full or partial)."""
        if settings.dry_run:
            logger.info(f"DRY RUN: Close ticket={ticket} lots={lots}")
            return MT5OrderResult(True, ticket=ticket)

        if not self._connected():
            return MT5OrderResult(False, error="Not connected")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return MT5OrderResult(False, error=f"Position {ticket} not found")

        pos = positions[0]
        close_lots = lots if lots else pos.volume
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return MT5OrderResult(False, error=f"No tick for {pos.symbol}")

        # Close = opposite direction
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    pos.symbol,
            "volume":    close_lots,
            "type":      order_type,
            "position":  ticket,
            "price":     price,
            "deviation": 20,
            "magic":     20260707,
            "comment":   "ARIA close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Closed ticket={ticket} lots={close_lots} pnl≈{pos.profit:.2f}")
            return MT5OrderResult(True, ticket=result.order)
        err = result.comment if result else str(mt5.last_error())
        return MT5OrderResult(False, error=err)

    def close_all(self, comment: str = "ARIA emergency") -> int:
        """Close all open positions. Returns count closed."""
        if not self._connected():
            return 0
        positions = mt5.positions_get()
        if not positions:
            return 0
        closed = 0
        for pos in positions:
            r = self.close_position(pos.ticket)
            if r.success:
                closed += 1
        logger.warning(f"Emergency close: {closed}/{len(positions)} positions closed")
        return closed


# Singleton
mt5_client = MT5Client()
