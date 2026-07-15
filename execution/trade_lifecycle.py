"""
Trade lifecycle manager — runs every 60 seconds on open positions.

Cascade logic:
  Stage 1 (entry → TP1):  monitor, update P&L
  Stage 2 (at TP1):       partial close 50%, move SL to entry (breakeven)
  Stage 3 (at TP2):       close another 30%, activate trailing stop
  Stage 4 (trailing):     trail SL by 0.5 × ATR behind price
  Emergency:              close all if capital halt triggered

Re-entry loop:
  When MT5 auto-closes a position (TP or SL hit), lifecycle detects it,
  syncs capital, and immediately scans for re-entry if signal still valid.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from core.capital import CapitalManager
from data.mt5_feed import feed
from execution.mt5_client import mt5_client
from analysis.indicators import atr_value, apply_all


class TradeLifecycle:
    """Manages all open positions — partial exits, trailing, breakeven, re-entry."""

    def __init__(self, capital: CapitalManager) -> None:
        self.capital = capital

    def tick(self) -> None:
        """Called every 60 seconds. Checks and manages all open positions."""
        positions = feed.get_positions()
        mt5_open_pairs = {p["pair"] for p in positions}

        # ── Sync account balance from MT5 ─────────────────────────
        account = feed.get_account_info()
        if account:
            self.capital.sync_balance(account["balance"], account["equity"])

        # ── Detect MT5 auto-closes (TP or SL hit) ─────────────────
        # Any pair in capital.open_positions that disappeared from MT5 was closed
        capital_pairs = list(self.capital.open_positions.keys())
        closed_pairs: list[str] = []

        for pair in capital_pairs:
            if pair not in mt5_open_pairs:
                pos_data = self.capital.open_positions.get(pair, {})
                pnl = self._get_closed_pnl(pair)
                self.capital.register_close(pair, 0, pnl)
                logger.info(f"MT5 auto-close detected: {pair} pnl=${pnl:.2f}")
                closed_pairs.append(pair)

                # Persist close to SQLite
                try:
                    from db.session import get_session
                    from db.models import Trade
                    from sqlalchemy import update as sa_update
                    ticket = pos_data.get("ticket")
                    if ticket:
                        # Pull actual close price from deal history
                        _close_price = 0.0
                        try:
                            deals = feed.get_history_deals(days=1)
                            for d in reversed(deals):
                                if d.get("position_id") == ticket:
                                    _close_price = d.get("entry", 0.0)
                                    break
                        except Exception:
                            pass
                        with get_session() as db:
                            db.execute(
                                sa_update(Trade)
                                .where(Trade.ticket == ticket)
                                .values(
                                    pnl=pnl,
                                    close_price=_close_price,
                                    closed_at=datetime.utcnow(),
                                    status="closed",
                                )
                            )
                except Exception as _db_err:
                    logger.debug(f"DB persist (close) failed for {pair}: {_db_err}")

                # Telegram alert (non-blocking)
                try:
                    from notifications.telegram import alert_trade_closed
                    alert_trade_closed(
                        pair=pair,
                        direction=pos_data.get("direction", "unknown"),
                        pnl=pnl,
                        day_pnl=self.capital.day.realized_pnl + pnl,
                    )
                except Exception:
                    pass

                # Feed result into adaptive learning + ML feature store
                try:
                    from core.adaptive_learning import adaptive
                    from core.session import SessionManager
                    # pos_data captured before register_close() removed the entry
                    session  = SessionManager().current_session().value
                    adaptive.record_trade(
                        pair=pair,
                        won=pnl > 0,
                        pnl=pnl,
                        score=pos_data.get("score", 0),
                        direction=pos_data.get("direction", "unknown"),
                        session=session,
                    )
                    # Save ML training sample if features were captured at entry
                    ml_feats = pos_data.get("ml_features")
                    ml_boost = pos_data.get("ml_boost", 0.0)
                    if ml_feats:
                        from ml.features import save_sample, sample_count, MIN_TRAINING_SAMPLES
                        save_sample(ml_feats, won=pnl > 0, pnl=pnl, pair=pair)
                        # Auto-train when enough samples accumulated
                        if sample_count() >= MIN_TRAINING_SAMPLES:
                            from ml.trainer import maybe_train
                            import threading
                            threading.Thread(target=maybe_train, daemon=True).start()
                    # Track whether the ML boost is actually predictive
                    try:
                        from ml.performance import tracker as ml_perf
                        ml_perf.record(ml_boost=ml_boost, won=pnl > 0, pair=pair)
                    except Exception:
                        pass

                    # Trade analysis pipeline — build pattern library
                    try:
                        from core.pattern_library import record as pattern_record
                        opened_at = pos_data.get("opened_at", "")
                        if opened_at:
                            open_dt = datetime.fromisoformat(opened_at)
                            hold_minutes = (datetime.utcnow() - open_dt).total_seconds() / 60
                        else:
                            hold_minutes = 0.0
                        pattern_record(
                            pair=pair,
                            direction=pos_data.get("direction", "unknown"),
                            score=pos_data.get("score", 0.0),
                            regime=pos_data.get("strategy", "trend"),
                            session=session,
                            ml_boost=ml_boost,
                            pnl=pnl,
                            hold_minutes=hold_minutes,
                            entry=pos_data.get("entry", 0.0),
                            sl=pos_data.get("sl", 0.0),
                            tp1=pos_data.get("tp1", 0.0),
                        )
                    except Exception:
                        pass
                    # Strategy equity tracking
                    try:
                        from core.strategy_equity import record_trade as seq_record
                        strategy_name = pos_data.get("strategy", "SMC_TREND")
                        seq_record(strategy=strategy_name, pnl=pnl, won=pnl > 0)
                    except Exception:
                        pass

                    # A/B test result recording
                    try:
                        from core.ab_testing import record_by_strategy
                        record_by_strategy(pos_data.get("strategy", ""), pnl, pnl > 0)
                    except Exception:
                        pass

                except Exception as ex:
                    logger.debug(f"Adaptive learning update failed for {pair}: {ex}")

        # ── Re-entry after close ───────────────────────────────────
        if closed_pairs:
            self._try_reenter(closed_pairs)

        # ── Emergency halt — close everything ─────────────────────
        if self.capital._trading_halted:
            logger.warning("Capital halt active — closing all positions")
            mt5_client.close_all("ARIA halt")
            for pos in positions:
                self.capital.register_close(pos["pair"], pos["current"], pos["pnl"])
            return

        if not positions:
            return

        for pos in positions:
            self._manage_position(pos)

    def _get_closed_pnl(self, pair: str) -> float:
        """Look up P&L for a position that was auto-closed by MT5 (TP/SL)."""
        try:
            pos_data = self.capital.open_positions.get(pair, {})
            our_ticket = pos_data.get("ticket")
            if not our_ticket:
                return 0.0

            deals = feed.get_history_deals(days=1)
            # Sum all deal profits for this position (entry + close may both appear)
            total = sum(
                d["profit"] + d.get("commission", 0) + d.get("swap", 0)
                for d in deals
                if d.get("position_id") == our_ticket
            )
            return round(total, 2)
        except Exception as e:
            logger.debug(f"Could not fetch closed P&L for {pair}: {e}")
            return 0.0

    def _try_reenter(self, closed_pairs: list[str]) -> None:
        """Scan closed pairs and re-enter immediately if signal is still valid."""
        from signals.scanner import scan_pair
        from signals.entry import build_setup
        from execution.order_manager import OrderManager

        order_mgr = OrderManager(self.capital)

        for pair in closed_pairs:
            try:
                permission = self.capital.can_trade(pair)
                if not permission.allowed:
                    logger.info(f"Re-entry blocked: {pair} — {permission.reason}")
                    continue

                result = scan_pair(pair)
                if not result or not result.auto_executable:
                    logger.debug(f"Re-entry skip: {pair} — score={result.score if result else 0:.0f} (need ≥70)")
                    continue

                tick = feed.get_tick(pair)
                price = tick.get("mid", 0) if tick else 0
                if not price:
                    logger.warning(f"Re-entry: no tick for {pair}")
                    continue

                df_m15 = feed.get_candles(pair, "M15", count=50)
                df_m15 = apply_all(df_m15)
                setup = build_setup(result, price, df_m15)

                if setup:
                    logger.info(f"Re-entry: {result.direction.upper()} {pair} score={result.score:.0f}")
                    order_mgr.execute(result, setup)
                else:
                    logger.debug(f"Re-entry skip: {pair} — no valid setup (ATR/SMC issue)")

            except Exception as e:
                logger.error(f"Re-entry error for {pair}: {e}")

    def _manage_position(self, pos: dict) -> None:
        pair      = pos["pair"]
        ticket    = pos["ticket"]
        direction = pos["direction"]
        entry     = pos["entry"]
        current   = pos["current"]
        sl        = pos["sl"]
        tp        = pos["tp"]
        lots      = pos["lots"]

        # Update unrealized P&L in capital manager
        self.capital.update_pnl(pair, current)

        # Get ATR for trailing calculations
        df = feed.get_candles(pair, "M15", count=50)
        if df.empty:
            return
        df = apply_all(df)
        atr = atr_value(df)
        if atr <= 0:
            return

        if direction == "long":
            sl_dist = entry - sl
            if sl_dist <= 0:
                return
            dist_from_entry = current - entry

            if dist_from_entry >= sl_dist * 2 and sl < entry:
                # TP1 zone — partial close 50%, move SL to breakeven
                partial_lots = round(lots * 0.5, 2)
                if partial_lots >= 0.01:
                    r = mt5_client.close_position(ticket, partial_lots)
                    if r.success:
                        logger.info(f"TP1 partial close: {pair} {partial_lots} lots @ {current:.5f}")
                mt5_client.modify_position(ticket, sl=entry)
                logger.info(f"Breakeven SL: {pair} → {entry:.5f}")

            elif dist_from_entry >= sl_dist * 4:
                # TP2 zone — close another 30%, start trailing
                partial_lots = round(lots * 0.3, 2)
                if partial_lots >= 0.01:
                    mt5_client.close_position(ticket, partial_lots)
                trail_sl = current - atr * 0.5
                if trail_sl > sl:
                    mt5_client.modify_position(ticket, sl=trail_sl)
                    logger.debug(f"Trail SL: {pair} → {trail_sl:.5f}")

        elif direction == "short":
            sl_dist = sl - entry
            if sl_dist <= 0:
                return
            dist_from_entry = entry - current

            if dist_from_entry >= sl_dist * 2 and sl > entry:
                partial_lots = round(lots * 0.5, 2)
                if partial_lots >= 0.01:
                    r = mt5_client.close_position(ticket, partial_lots)
                    if r.success:
                        logger.info(f"TP1 partial close (short): {pair} {partial_lots} lots @ {current:.5f}")
                mt5_client.modify_position(ticket, sl=entry)
                logger.info(f"Breakeven SL (short): {pair} → {entry:.5f}")

            elif dist_from_entry >= sl_dist * 4:
                partial_lots = round(lots * 0.3, 2)
                if partial_lots >= 0.01:
                    mt5_client.close_position(ticket, partial_lots)
                trail_sl = current + atr * 0.5
                if trail_sl < sl:
                    mt5_client.modify_position(ticket, sl=trail_sl)
                    logger.debug(f"Trail SL (short): {pair} → {trail_sl:.5f}")


def _pip_size(pair: str) -> float:
    p = pair.upper()
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.1
    if any(x in p for x in ("NAS", "SPX", "US30", "USTEC", "NDX")):
        return 1.0
    return 0.0001
