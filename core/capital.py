"""
Capital Manager — the financial brain of ARIA.

Handles: balance tracking, position sizing, daily P&L limits,
kill switches, and trade allowance logic.

This is the MOST CRITICAL module. All execution goes through here.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# Risk limits (can be moved to settings later)
WEEKLY_DD_LIMIT_PCT  = 6.0    # halt if week drawdown exceeds 6%
MONTHLY_DD_LIMIT_PCT = 10.0   # halt if month drawdown exceeds 10%
MAX_LEVERAGE_X       = 500.0  # hard cap: total exposure / balance

# Abnormal behavior detection
_STREAK_THRESHOLD      = 3     # consecutive losses that trigger cooldown
_STREAK_WINDOW_MINUTES = 60    # losses must occur within this window
_COOLDOWN_HOURS        = 2     # trading pause duration after streak detected

from loguru import logger

from config.settings import settings


@dataclass
class DayStats:
    date: str = field(default_factory=lambda: date.today().isoformat())
    realized_pnl: float = 0.0          # Closed trades P&L today
    unrealized_pnl: float = 0.0         # Open positions P&L
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown_today: float = 0.0
    starting_balance: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.trades_won / self.trades_taken * 100

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return 0.0
        return self.gross_profit / abs(self.gross_loss)


@dataclass
class TradePermission:
    allowed: bool
    reason: str
    score: float = 0.0  # 0-100 confidence in allowance


class CapitalManager:
    """
    Single source of truth for all capital decisions.
    Thread-safe. All execution modules consult this before acting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.balance = settings.account_balance
        self.equity = settings.account_balance
        self.day = DayStats(starting_balance=settings.account_balance)
        self.open_positions: dict[str, dict] = {}    # pair → position data
        self._trading_halted = False
        self._halt_reason = ""

        # Abnormal behavior detection — loss streak cooldown
        self._consecutive_losses: int = 0
        self._loss_streak_start: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

        # Override authority — tracks can_trade() grants consumed by register_open()
        # pair → monotonic time when permission was granted (30s TTL)
        self._auth_grants: dict[str, float] = {}
        self._bypass_count: int = 0   # lifetime count of unauthorized open attempts

        # Weekly / monthly drawdown tracking
        self._week_start_balance:  float = settings.account_balance
        self._month_start_balance: float = settings.account_balance
        self._week_realized_pnl:   float = 0.0
        self._month_realized_pnl:  float = 0.0
        self._week_start_date:  str = self._week_key()
        self._month_start_date: str = self._month_key()

        logger.info(
            "CapitalManager initialized",
            balance=self.balance,
            risk_per_trade=f"{settings.risk_per_trade_pct}%",
            daily_target=f"+${settings.daily_target_amount:.2f}",
            max_loss=f"-${settings.max_loss_amount:.2f}",
            weekly_dd_limit=f"{WEEKLY_DD_LIMIT_PCT}%",
            monthly_dd_limit=f"{MONTHLY_DD_LIMIT_PCT}%",
        )

    # ── Period helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _week_key() -> str:
        """ISO year-week string: '2026-W28'."""
        d = date.today()
        return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"

    @staticmethod
    def _month_key() -> str:
        return date.today().strftime("%Y-%m")

    def _auto_roll_periods(self) -> None:
        """Called inside lock — roll week/month if calendar rolled."""
        wk = self._week_key()
        mo = self._month_key()
        if wk != self._week_start_date:
            self._week_start_balance  = self.balance
            self._week_realized_pnl   = 0.0
            self._week_start_date     = wk
            logger.info("Weekly P&L tracker reset", week=wk, balance=self.balance)
        if mo != self._month_start_date:
            self._month_start_balance = self.balance
            self._month_realized_pnl  = 0.0
            self._month_start_date    = mo
            logger.info("Monthly P&L tracker reset", month=mo, balance=self.balance)

    # ── Permission Gate ───────────────────────────────────────────────────────

    def can_trade(self, pair: str | None = None) -> TradePermission:
        """Central permission check — ALL trades must pass this."""
        with self._lock:
            self._auto_roll_periods()

            # Hard halt
            if self._trading_halted:
                return TradePermission(False, f"Trading halted: {self._halt_reason}")

            # Loss-streak cooldown (auto-expires, no manual intervention needed)
            if self._cooldown_until is not None:
                now = datetime.now(timezone.utc)
                if now < self._cooldown_until:
                    remaining = int((self._cooldown_until - now).total_seconds() / 60)
                    until_str = self._cooldown_until.strftime("%H:%M")
                    return TradePermission(
                        False,
                        f"Cooldown active until {until_str} UTC ({remaining} min) — "
                        f"{_STREAK_THRESHOLD} losses in {_STREAK_WINDOW_MINUTES} min"
                    )
                else:
                    self._cooldown_until = None
                    self._consecutive_losses = 0
                    self._loss_streak_start = None
                    logger.info("Loss-streak cooldown expired — trading resumed automatically")

            # Max trades per day
            if self.day.trades_taken >= settings.max_trades_per_day:
                return TradePermission(False, f"Max trades/day reached ({settings.max_trades_per_day})")

            # Max concurrent
            if len(self.open_positions) >= settings.max_concurrent_trades:
                return TradePermission(False, f"Max concurrent trades ({settings.max_concurrent_trades})")

            # Daily loss limit
            if self.day.realized_pnl <= -settings.max_loss_amount:
                self._halt_trading(f"Daily loss limit hit: ${self.day.realized_pnl:.2f}")
                return TradePermission(False, f"Daily loss limit: -${settings.max_loss_amount:.2f}")

            # Daily profit target (lock in gains)
            if self.day.realized_pnl >= settings.daily_target_amount:
                self._halt_trading(f"Daily target reached: +${self.day.realized_pnl:.2f}")
                return TradePermission(False, f"Daily target reached: +${settings.daily_target_amount:.2f} ✅")

            # Correlation check — no same-direction correlated pairs
            if pair and self._is_correlated_conflict(pair):
                return TradePermission(False, f"Correlated pair already open: {pair}")

            # Weekly drawdown limit
            weekly_dd_pct = (-self._week_realized_pnl / self._week_start_balance * 100
                             if self._week_start_balance > 0 else 0.0)
            if weekly_dd_pct >= WEEKLY_DD_LIMIT_PCT:
                self._halt_trading(
                    f"Weekly drawdown limit hit: -{weekly_dd_pct:.1f}% (limit {WEEKLY_DD_LIMIT_PCT}%)"
                )
                return TradePermission(False,
                    f"Weekly drawdown limit: -{weekly_dd_pct:.1f}% ≥ {WEEKLY_DD_LIMIT_PCT}%")

            # Monthly drawdown limit
            monthly_dd_pct = (-self._month_realized_pnl / self._month_start_balance * 100
                              if self._month_start_balance > 0 else 0.0)
            if monthly_dd_pct >= MONTHLY_DD_LIMIT_PCT:
                self._halt_trading(
                    f"Monthly drawdown limit hit: -{monthly_dd_pct:.1f}% (limit {MONTHLY_DD_LIMIT_PCT}%)"
                )
                return TradePermission(False,
                    f"Monthly drawdown limit: -{monthly_dd_pct:.1f}% ≥ {MONTHLY_DD_LIMIT_PCT}%")

            # Emergency drawdown
            if self.equity <= self.balance * (1 - settings.emergency_drawdown_pct / 100):
                self._halt_trading("EMERGENCY: Equity drawdown exceeded")
                return TradePermission(False, "Emergency drawdown triggered — all trading stopped")

            # Issue authorization token — consumed by register_open() within 30s
            if pair:
                self._auth_grants[pair] = time.monotonic()

            return TradePermission(True, "All checks passed", 100.0)

    # ── Position Sizing ───────────────────────────────────────────────────────

    def calculate_lots(self, pair: str, entry: float, sl: float) -> float:
        """
        Calculate lot size so that if SL is hit, we lose exactly risk_amount.
        Uses proper pip value calculation per instrument type.
        """
        with self._lock:
            risk_amount = self.balance * settings.risk_per_trade_pct / 100

            pip_size = self._get_pip_size(pair)
            pip_value_per_lot = self._get_pip_value_per_lot(pair, entry)

            sl_pips = abs(entry - sl) / pip_size
            if sl_pips <= 0:
                logger.warning("SL pips is zero or negative", pair=pair, entry=entry, sl=sl)
                return 0.01  # min lot

            lots = risk_amount / (sl_pips * pip_value_per_lot)

            # Cap at reasonable limits
            min_lot = 0.01
            max_lot = min(
                risk_amount / (sl_pips * pip_value_per_lot * 0.5),  # safety cap
                10.0  # absolute max
            )
            lots = round(max(min_lot, min(lots, max_lot)), 2)

            logger.debug(
                "Position size calculated",
                pair=pair, risk_amount=risk_amount,
                sl_pips=round(sl_pips, 1), lots=lots,
            )
            return lots

    def _get_pip_size(self, pair: str) -> float:
        """Pip size for different instrument types. Handles broker suffixes (e.g. Exness 'm')."""
        p = pair.upper()
        if "JPY" in p:
            return 0.01
        if "XAU" in p or "GOLD" in p:
            return 0.1
        if any(x in p for x in ("NAS", "SPX", "US30", "US500", "USTEC", "NDX")):
            return 1.0
        if "BTC" in p:
            return 1.0
        return 0.0001

    def _get_pip_value_per_lot(self, pair: str, price: float) -> float:
        """Approximate USD pip value per 1.0 standard lot. Handles broker suffixes."""
        p = pair.upper()
        if "JPY" in p:
            return 1000 / price
        if "XAU" in p or "GOLD" in p:
            return 100.0
        if any(x in p for x in ("NAS", "USTEC", "NDX", "SPX", "US30")):
            return 1.0
        if p.startswith("EUR"):
            return 10.0
        if p.startswith("GBP"):
            return 10.0 * (price / 1.25)
        return 10.0

    def _notional_usd(self, pair: str, lots: float, price: float) -> float:
        """Approximate USD notional value of a position. Used for leverage check."""
        p = pair.upper().rstrip("M")
        if "XAU" in p or "GOLD" in p:
            return lots * 100 * price        # 100 oz per lot
        if any(x in p for x in ("NAS", "USTEC", "NDX", "SPX", "US30", "US500")):
            return lots * price              # index: 1 contract × price
        if p.startswith("USD"):
            return lots * 100_000           # base is USD, no conversion needed
        if "JPY" in p:
            return lots * 100_000 * price / 150.0   # approx JPY → USD
        return lots * 100_000 * price       # standard FX: base × 100k × rate

    def check_leverage(self, pair: str, lots: float, entry: float) -> TradePermission:
        """
        Reject trades that would push total account leverage over MAX_LEVERAGE_X.
        Call from order_manager after lot sizing, before MT5 order placement.
        """
        with self._lock:
            if self.balance <= 0:
                return TradePermission(True, "No balance reference — leverage check skipped")

            new_notional = self._notional_usd(pair, lots, entry)
            existing_notional = sum(
                self._notional_usd(
                    p["pair"], p["lots"], p.get("entry", entry)
                )
                for p in self.open_positions.values()
            )
            total_notional = existing_notional + new_notional
            leverage = total_notional / self.balance

            if leverage > MAX_LEVERAGE_X:
                msg = (
                    f"Leverage {leverage:.0f}× exceeds cap {MAX_LEVERAGE_X:.0f}× "
                    f"(notional ${total_notional:,.0f} on ${self.balance:.0f} balance)"
                )
                logger.warning(f"[LeverageCap] {msg}")
                try:
                    from core.risk_log import append_event, LEVERAGE_BLOCK
                    append_event(
                        LEVERAGE_BLOCK, msg, balance=self.balance,
                        extra={"Pair": pair, "Lots": str(lots), "Entry": f"{entry:.5f}"},
                    )
                except Exception:
                    pass
                return TradePermission(False, msg)

            logger.debug(f"[LeverageCap] {pair} {lots}L — leverage {leverage:.1f}× / {MAX_LEVERAGE_X:.0f}× OK")
            return TradePermission(True, f"Leverage {leverage:.1f}×", 100.0)

    def _is_correlated_conflict(self, pair: str) -> bool:
        """Prevent taking correlated trades in same direction. Handles broker suffixes."""
        # Strip broker suffix for correlation lookup
        base = pair.upper().rstrip("M")
        CORRELATIONS: dict[str, list[str]] = {
            "EURUSD": ["GBPUSD", "AUDUSD"],
            "GBPUSD": ["EURUSD", "GBPJPY"],
            "USDJPY": ["GBPJPY", "EURJPY"],
            "XAUUSD": [],
        }
        correlated_bases = CORRELATIONS.get(base, [])
        open_pairs = list(self.open_positions.keys())
        # Match by stripping suffix from open positions too
        open_bases = [p.upper().rstrip("M") for p in open_pairs]
        return any(c in open_bases for c in correlated_bases)

    # ── Position Tracking ─────────────────────────────────────────────────────

    def register_open(self, ticket: int, pair: str, direction: str,
                      lots: float, entry: float, sl: float, tp1: float,
                      tp2: float, tp3: float, score: float = 0.0,
                      strategy: str = "TREND", ml_features: Optional[dict] = None,
                      ml_boost: float = 0.0) -> None:
        with self._lock:
            # Verify this open was authorized by can_trade() within the last 30 seconds
            grant_time = self._auth_grants.pop(pair, None)
            if grant_time is None or (time.monotonic() - grant_time) > 30.0:
                self._bypass_count += 1
                logger.critical(
                    f"[RiskEngine] UNAUTHORIZED register_open({pair}) — "
                    "no prior can_trade() approval within 30s. "
                    "Possible bypass attempt or code path not going through OrderManager."
                )
                try:
                    from core.risk_log import append_event
                    append_event(
                        "BYPASS ATTEMPT",
                        f"register_open({pair}) called without valid authorization token",
                        balance=self.balance,
                        extra={"Pair": pair, "Direction": direction, "Lots": str(lots),
                               "Lifetime bypass count": str(self._bypass_count)},
                    )
                except Exception:
                    pass

            self.open_positions[pair] = {
                "ticket": ticket,
                "pair": pair,
                "direction": direction,
                "lots": lots,
                "entry": entry,
                "sl": sl,
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "opened_at": datetime.utcnow().isoformat(),
                "pnl": 0.0,
                "stage": "entry",
                "score": score,
                "strategy": strategy,
                "ml_features": ml_features,
                "ml_boost": ml_boost,
            }
            self.day.trades_taken += 1
            logger.info("Position registered", pair=pair, direction=direction, lots=lots, entry=entry)

    def update_pnl(self, pair: str, current_price: float) -> float:
        with self._lock:
            pos = self.open_positions.get(pair)
            if not pos:
                return 0.0
            direction = pos["direction"]
            entry = pos["entry"]
            lots = pos["lots"]
            pip_size = self._get_pip_size(pair)
            pip_val = self._get_pip_value_per_lot(pair, current_price)
            pips = (current_price - entry) / pip_size if direction == "long" else (entry - current_price) / pip_size
            pos["pnl"] = pips * pip_val * lots
            self.day.unrealized_pnl = sum(p["pnl"] for p in self.open_positions.values())
            return pos["pnl"]

    def set_position_regime(self, pair: str, regime: str) -> None:
        """Tag an open position with its entry regime (TRENDING/RANGING/VOLATILE)."""
        with self._lock:
            if pair in self.open_positions:
                self.open_positions[pair]["regime"] = regime

    def register_close(self, pair: str, close_price: float, pnl: float) -> None:
        with self._lock:
            if pair in self.open_positions:
                del self.open_positions[pair]
            self.day.realized_pnl   += pnl
            self._week_realized_pnl  += pnl
            self._month_realized_pnl += pnl
            if pnl > 0:
                self.day.trades_won += 1
                self.day.gross_profit += pnl
                # Win resets the streak
                self._consecutive_losses = 0
                self._loss_streak_start  = None
            else:
                self.day.trades_lost += 1
                self.day.gross_loss  += pnl
                self._update_loss_streak()
            logger.info("Position closed", pair=pair, pnl=f"${pnl:.2f}",
                        daily_pnl=f"${self.day.realized_pnl:.2f}",
                        weekly_pnl=f"${self._week_realized_pnl:.2f}",
                        monthly_pnl=f"${self._month_realized_pnl:.2f}")

    # ── State ─────────────────────────────────────────────────────────────────

    def sync_balance(self, new_balance: float, new_equity: float) -> None:
        with self._lock:
            self.balance = new_balance
            self.equity = new_equity

    def reset_day(self) -> None:
        with self._lock:
            self.day = DayStats(starting_balance=self.balance)
            self._trading_halted = False
            self._halt_reason = ""
            logger.info("Day stats reset", new_day=date.today().isoformat())

    def _update_loss_streak(self) -> None:
        """Called inside lock when a loss is registered. Activates cooldown if streak threshold hit."""
        now = datetime.now(timezone.utc)

        if (self._loss_streak_start is None or
                (now - self._loss_streak_start).total_seconds() > _STREAK_WINDOW_MINUTES * 60):
            # Start a fresh streak window
            self._consecutive_losses = 1
            self._loss_streak_start  = now
        else:
            self._consecutive_losses += 1

        logger.debug(
            f"Loss streak: {self._consecutive_losses}/{_STREAK_THRESHOLD} "
            f"in last {_STREAK_WINDOW_MINUTES} min"
        )

        if self._consecutive_losses >= _STREAK_THRESHOLD and self._cooldown_until is None:
            self._cooldown_until = now + timedelta(hours=_COOLDOWN_HOURS)
            until_str = self._cooldown_until.strftime("%H:%M")
            logger.warning(
                f"Abnormal behavior: {self._consecutive_losses} consecutive losses — "
                f"cooldown until {until_str} UTC"
            )
            try:
                from notifications.telegram import alert_cooldown
                alert_cooldown(self._consecutive_losses, until_str)
            except Exception:
                pass
            try:
                from core.risk_log import append_event, COOLDOWN
                append_event(
                    COOLDOWN,
                    f"{self._consecutive_losses} consecutive losses in {_STREAK_WINDOW_MINUTES} min",
                    balance=self.balance,
                    extra={"Cooldown until": f"{until_str} UTC", "Auto-resumes": "Yes"},
                )
            except Exception:
                pass

    def _halt_trading(self, reason: str) -> None:
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            logger.warning("⛔ Trading HALTED", reason=reason)
            try:
                from notifications.telegram import alert_halt, alert_risk_halt
                alert_halt(reason)
                day_dd = abs(self.day.realized_pnl / self.balance * 100) if self.balance else 0.0
                alert_risk_halt(reason, day_dd)
            except Exception:
                pass
            try:
                from core.risk_log import append_event, HALT
                append_event(HALT, reason, balance=self.balance)
            except Exception:
                pass

    def resume_trading(self) -> None:
        with self._lock:
            self._trading_halted = False
            self._halt_reason = ""
            logger.info("Trading resumed manually")

    @property
    def status_dict(self) -> dict:
        with self._lock:
            return {
                "balance": self.balance,
                "equity": self.equity,
                "realized_pnl": self.day.realized_pnl,
                "unrealized_pnl": self.day.unrealized_pnl,
                "total_pnl": self.day.total_pnl,
                "trades_taken": self.day.trades_taken,
                "trades_open": len(self.open_positions),
                "win_rate": self.day.win_rate,
                "profit_factor": self.day.profit_factor,
                "target_amount": settings.daily_target_amount,
                "max_loss_amount": settings.max_loss_amount,
                "target_progress_pct": (self.day.realized_pnl / settings.daily_target_amount * 100)
                    if settings.daily_target_amount > 0 else 0,
                "halted": self._trading_halted,
                "halt_reason": self._halt_reason,
                "can_trade": not self._trading_halted,
                "open_positions": list(self.open_positions.values()),
                "weekly_pnl": self._week_realized_pnl,
                "monthly_pnl": self._month_realized_pnl,
                "weekly_dd_pct": (-self._week_realized_pnl / self._week_start_balance * 100
                                  if self._week_start_balance > 0 else 0.0),
                "monthly_dd_pct": (-self._month_realized_pnl / self._month_start_balance * 100
                                   if self._month_start_balance > 0 else 0.0),
                "weekly_dd_limit": WEEKLY_DD_LIMIT_PCT,
                "monthly_dd_limit": MONTHLY_DD_LIMIT_PCT,
                "consecutive_losses": self._consecutive_losses,
                "cooldown_active": self._cooldown_until is not None and
                                   datetime.now(timezone.utc) < self._cooldown_until,
                "cooldown_until": (self._cooldown_until.strftime("%H:%M UTC")
                                   if self._cooldown_until else None),
                "bypass_count": self._bypass_count,
                "current_leverage": (
                    sum(
                        self._notional_usd(p["pair"], p["lots"], p.get("entry", 1.0))
                        for p in self.open_positions.values()
                    ) / self.balance
                    if self.balance > 0 else 0.0
                ),
                "max_leverage": MAX_LEVERAGE_X,
            }
