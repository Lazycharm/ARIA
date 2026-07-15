"""
Backtest performance metrics.

Computes institutional-grade statistics from a list of closed trades
and an equity curve. Every number is accurate — no estimations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trade:
    pair: str
    direction: str          # "long" | "short"
    entry: float
    exit: float
    sl: float
    tp1: float
    lots: float
    pnl: float              # USD P&L
    pnl_pct: float          # % of balance at entry
    score: float            # confluence score
    exit_reason: str        # "tp1" | "sl" | "tp2" | "time"
    entry_bar: int          # candle index at entry
    exit_bar: int
    bars_held: int
    session: str = ""
    sl_type: str = "atr"


@dataclass
class BacktestResults:
    pair: str
    days: int
    initial_balance: float
    final_balance: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # ── Computed on demand ────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.wins) / len(self.trades) * 100

    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def net_pnl_pct(self) -> float:
        return self.net_pnl / self.initial_balance * 100

    @property
    def gross_profit(self) -> float:
        return sum(t.pnl for t in self.wins)

    @property
    def gross_loss(self) -> float:
        return sum(t.pnl for t in self.losses)

    @property
    def profit_factor(self) -> float:
        if not self.losses or self.gross_loss == 0:
            return float("inf") if self.wins else 0.0
        return self.gross_profit / abs(self.gross_loss)

    @property
    def avg_win(self) -> float:
        return self.gross_profit / len(self.wins) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return self.gross_loss / len(self.losses) if self.losses else 0.0

    @property
    def expectancy(self) -> float:
        """Expected $ per trade."""
        if not self.trades:
            return 0.0
        wr = self.win_rate / 100
        return wr * self.avg_win + (1 - wr) * self.avg_loss

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown in % of peak equity."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_drawdown_dollars(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def recovery_factor(self) -> float:
        if self.max_drawdown_dollars == 0:
            return float("inf") if self.net_pnl > 0 else 0.0
        return self.net_pnl / self.max_drawdown_dollars

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe on daily returns (assumes 252 trading days)."""
        if len(self.equity_curve) < 2:
            return 0.0
        daily_returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        if daily_returns.std() == 0:
            return 0.0
        return (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)

    @property
    def max_consecutive_losses(self) -> int:
        streak = cur = 0
        for t in self.trades:
            if t.pnl <= 0:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 0
        return streak

    @property
    def sortino_ratio(self) -> float:
        """Annualised Sortino — penalises only downside volatility (252 days)."""
        if len(self.equity_curve) < 2:
            return 0.0
        daily_returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        downside = daily_returns[daily_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if daily_returns.mean() > 0 else 0.0
        return (daily_returns.mean() / downside.std()) * math.sqrt(252)

    @property
    def calmar_ratio(self) -> float:
        """Annual return / Max drawdown. Requires at least 1y data for full meaning."""
        if self.max_drawdown == 0:
            return float("inf") if self.net_pnl > 0 else 0.0
        annual_return = (self.net_pnl_pct / self.days) * 252
        return annual_return / self.max_drawdown

    def summary(self) -> str:
        sep = "─" * 44
        lines = [
            f"\n{'═'*44}",
            f"  ARIA BACKTEST — {self.pair}  ({self.days}d)",
            sep,
            f"  Trades      : {self.total_trades}",
            f"  Win Rate    : {self.win_rate:.1f}%",
            f"  Profit Factor:{self.profit_factor:.2f}",
            f"  Expectancy  : ${self.expectancy:+.2f}/trade",
            sep,
            f"  Net P&L     : ${self.net_pnl:+.2f}  ({self.net_pnl_pct:+.1f}%)",
            f"  Gross Profit: ${self.gross_profit:+.2f}",
            f"  Gross Loss  : ${self.gross_loss:+.2f}",
            f"  Avg Win     : ${self.avg_win:+.2f}",
            f"  Avg Loss    : ${self.avg_loss:+.2f}",
            sep,
            f"  Max Drawdown: {self.max_drawdown:.1f}%  (${self.max_drawdown_dollars:.2f})",
            f"  Recovery F. : {self.recovery_factor:.2f}",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Sortino     : {self.sortino_ratio:.2f}",
            f"  Calmar      : {self.calmar_ratio:.2f}",
            f"  Max Con.Loss: {self.max_consecutive_losses}",
            sep,
            f"  Start Bal   : ${self.initial_balance:.2f}",
            f"  End Bal     : ${self.final_balance:.2f}",
            f"{'═'*44}\n",
        ]
        return "\n".join(lines)

    def verdict(self) -> str:
        """Go/no-go verdict using master-prompt validation thresholds."""
        issues = []
        if self.total_trades < 300:
            issues.append(f"Too few trades ({self.total_trades} < 300 minimum)")
        if self.profit_factor < 1.5:
            issues.append(f"Profit factor too low ({self.profit_factor:.2f} < 1.5)")
        if self.max_drawdown > 10:
            issues.append(f"Drawdown too high ({self.max_drawdown:.1f}% > 10%)")
        if self.win_rate < 40:
            issues.append(f"Win rate too low ({self.win_rate:.1f}% < 40%)")
        if self.expectancy <= 0:
            issues.append("Negative expectancy")
        if self.recovery_factor < 2.0:
            issues.append(f"Recovery factor too low ({self.recovery_factor:.2f} < 2.0)")
        if self.sortino_ratio <= 0:
            issues.append(f"Sortino ratio non-positive ({self.sortino_ratio:.2f})")

        if not issues:
            return "✅ DEPLOY — strategy passes all master-prompt validation criteria"
        return "❌ DO NOT DEPLOY:\n" + "\n".join(f"  • {i}" for i in issues)
