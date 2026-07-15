"""
Order manager — coordinates CapitalManager → SignalFilter → MT5Client.

Single entry point for all execution: scan_pair result → live trade.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from analysis.confluence import ConfluenceScore
from core.capital import CapitalManager, TradePermission
from execution.mt5_client import mt5_client, MT5OrderResult
from signals.entry import TradeSetup, build_setup
from data.mt5_feed import feed
import pandas as pd


class OrderManager:
    def __init__(self, capital: CapitalManager) -> None:
        self.capital = capital

    def execute(
        self,
        signal: ConfluenceScore,
        setup: TradeSetup,
    ) -> Optional[int]:
        """
        Full execution pipeline for a qualified signal.
        Returns MT5 ticket number or None if blocked.
        """
        pair = signal.pair

        # ── Capital gate ──────────────────────────────────────────
        permission: TradePermission = self.capital.can_trade(pair)
        if not permission.allowed:
            logger.info(f"Trade blocked: {pair} — {permission.reason}")
            return None

        # ── Phase 5: Portfolio correlation guard ──────────────────
        from core.portfolio import portfolio

        # Derive strategy label from live market regime so equity curves split properly
        _REGIME_TO_STRATEGY = {
            "TRENDING": "SMC_TREND",
            "RANGING":  "RANGE_TRADING",
            "VOLATILE": "MEAN_REVERSION",
        }
        try:
            from core.regime_classifier import classify_trade_regime
            regime_label = classify_trade_regime(pair)
            strategy_label = _REGIME_TO_STRATEGY.get(regime_label, "SESSION_BREAKOUT")
        except Exception:
            strategy_label = getattr(getattr(signal, "_strategy", None), "label", "SMC_TREND")

        port_ok, port_reason = portfolio.can_open(
            pair=pair,
            direction=setup.direction,
            open_positions=self.capital.open_positions,
            strategy_label=strategy_label,
        )
        if not port_ok:
            logger.info(f"Portfolio block: {pair} — {port_reason}")
            return None

        # ── Position sizing (with adaptive + strategy lot multipliers) ─
        from core.adaptive_learning import adaptive
        base_lots = self.capital.calculate_lots(pair, setup.entry, setup.sl)
        if base_lots <= 0:
            logger.warning(f"Zero lots calculated: {pair}")
            return None
        adaptive_mult = adaptive.get_lot_multiplier(pair)

        # Sharpe-based sizing: strategies with proven positive Sharpe get sized up,
        # underperformers get sized down. Requires ≥10 closed trades to activate.
        try:
            from core.strategy_equity import get_curve
            seq_curve = get_curve(strategy_label)
            if seq_curve and seq_curve.trades >= 10:
                sh = seq_curve.sharpe
                sharpe_mult = 1.2 if sh > 1.0 else (1.0 if sh > 0.5 else (0.9 if sh > 0.0 else 0.75))
            else:
                sharpe_mult = 1.0
        except Exception:
            sharpe_mult = 1.0

        lots = max(0.01, round(base_lots * adaptive_mult * sharpe_mult, 2))

        # ── Leverage cap check ────────────────────────────────────
        lev_ok = self.capital.check_leverage(pair, lots, setup.entry)
        if not lev_ok.allowed:
            logger.warning(f"Leverage cap block: {pair} — {lev_ok.reason}")
            return None

        # ── Execute on MT5 ────────────────────────────────────────
        result: MT5OrderResult = mt5_client.place_market_order(
            pair=pair,
            direction=setup.direction,
            lots=lots,
            sl=setup.sl,
            tp=setup.tp1,  # initial TP = TP1; lifecycle manager upgrades to TP2/TP3
            comment=f"ARIA {signal.score:.0f}",
        )

        if not result.success:
            if "10027" in result.error or "AutoTrading" in result.error:
                logger.warning(
                    "⚠️  MT5 AutoTrading is DISABLED — enable it in MT5 toolbar "
                    "(Tools → AutoTrading or click the robot button). "
                    f"Blocking further execution attempts for {pair}."
                )
            else:
                logger.error(f"Execution failed: {pair} — {result.error}")
            return None

        ticket = result.ticket

        # ── Register with capital ─────────────────────────────────
        self.capital.register_open(
            ticket=ticket,
            pair=pair,
            direction=setup.direction,
            lots=lots,
            entry=setup.entry,
            sl=setup.sl,
            tp1=setup.tp1,
            tp2=setup.tp2,
            tp3=setup.tp3,
            score=signal.score,
            strategy=strategy_label,
            ml_features=getattr(signal, "_ml_features", None),
            ml_boost=getattr(signal, "_ml_boost", 0.0),
        )

        # Tag regime at trade entry
        try:
            from core.regime_classifier import tag_open_trade_regime
            regime = tag_open_trade_regime(pair)
            self.capital.set_position_regime(pair, regime)
        except Exception:
            pass

        # Strategy equity tracking
        try:
            from core.strategy_equity import record_trade as seq_record
            # Will be updated on close; initialize with 0
        except Exception:
            pass

        logger.success(
            f"Executed: {setup.direction.upper()} {lots} {pair} "
            f"entry={setup.entry} SL={setup.sl} TP1={setup.tp1} "
            f"ticket={ticket} score={signal.score:.0f}"
        )

        # Persist to SQLite
        try:
            from db.session import get_session
            from db.models import Trade
            from core.session import SessionManager
            session_name = SessionManager().current_session().value
            with get_session() as db:
                db.add(Trade(
                    ticket=ticket,
                    pair=pair,
                    direction=setup.direction,
                    lots=lots,
                    entry=setup.entry,
                    sl=setup.sl,
                    tp1=setup.tp1,
                    tp2=setup.tp2,
                    tp3=setup.tp3,
                    score=signal.score,
                    reason=signal.entry_reason[:200],
                    session=session_name,
                    status="open",
                    strategy_version=strategy_label,
                    regime=getattr(signal, "_regime_label", None),
                    spread_pips=getattr(signal, "_spread_pips", None),
                    risk_pct=self.capital.balance and (
                        abs(setup.entry - setup.sl) /
                        setup.entry * 100
                    ) or None,
                ))
        except Exception as _db_err:
            logger.debug(f"DB persist (open) failed for {pair}: {_db_err}")

        # Telegram alert (non-blocking)
        try:
            from notifications.telegram import alert_trade_opened
            from core.adaptive_learning import adaptive
            alert_trade_opened(
                pair=pair, direction=setup.direction,
                entry=setup.entry, sl=setup.sl, tp1=setup.tp1,
                lots=lots, score=signal.score,
                multiplier=adaptive.get_lot_multiplier(pair),
            )
        except Exception:
            pass

        return ticket

    def close_pair(self, pair: str, pnl: float) -> None:
        """Close a pair and register with capital manager."""
        positions = feed.get_positions()
        ticket = next((p["ticket"] for p in positions if p["pair"] == pair), None)
        if ticket:
            mt5_client.close_position(ticket)
        self.capital.register_close(pair, 0, pnl)
