"""
Unit tests for backtest/metrics.py.

All tests use synthetic trade lists and equity curves — no MT5, no filesystem.
"""
import math
import pytest

from backtest.metrics import Trade, BacktestResults


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trade(pnl: float, exit_reason: str = "tp1", score: float = 75.0) -> Trade:
    return Trade(
        pair="EURUSDm", direction="long",
        entry=1.1000, exit=1.1100, sl=1.0950, tp1=1.1050,
        lots=0.01, pnl=pnl, pnl_pct=pnl / 1000 * 100,
        score=score, exit_reason=exit_reason,
        entry_bar=0, exit_bar=5, bars_held=5,
    )


def make_results(
    trades: list[Trade],
    equity_curve: list[float],
    initial: float = 1000.0,
    days: int = 90,
) -> BacktestResults:
    final = equity_curve[-1] if equity_curve else initial
    return BacktestResults(
        pair="EURUSDm", days=days,
        initial_balance=initial, final_balance=final,
        trades=trades, equity_curve=equity_curve,
    )


# ── Win rate ──────────────────────────────────────────────────────────────────

def test_win_rate_no_trades():
    bt = make_results([], [1000.0])
    assert bt.win_rate == 0.0


def test_win_rate_all_wins():
    trades = [make_trade(10), make_trade(20)]
    bt = make_results(trades, [1000, 1010, 1030])
    assert bt.win_rate == 100.0


def test_win_rate_mixed():
    trades = [make_trade(10), make_trade(10), make_trade(10), make_trade(-5), make_trade(-5)]
    bt = make_results(trades, [1000, 1010, 1020, 1030, 1025, 1020])
    assert bt.win_rate == pytest.approx(60.0)


# ── Profit factor ─────────────────────────────────────────────────────────────

def test_profit_factor_balanced():
    # $30 profit, $20 loss → PF = 1.5
    trades = [make_trade(30), make_trade(-20)]
    bt = make_results(trades, [1000, 1030, 1010])
    assert bt.profit_factor == pytest.approx(1.5)


def test_profit_factor_no_losses():
    trades = [make_trade(10), make_trade(20)]
    bt = make_results(trades, [1000, 1010, 1030])
    assert bt.profit_factor == float("inf")


def test_profit_factor_no_wins():
    trades = [make_trade(-10), make_trade(-20)]
    bt = make_results(trades, [1000, 990, 970])
    assert bt.profit_factor == 0.0


def test_profit_factor_no_trades():
    bt = make_results([], [1000.0])
    assert bt.profit_factor == 0.0


# ── Max drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_flat():
    bt = make_results([], [1000, 1000, 1000])
    assert bt.max_drawdown == 0.0


def test_max_drawdown_only_up():
    bt = make_results([], [1000, 1010, 1020, 1030])
    assert bt.max_drawdown == 0.0


def test_max_drawdown_known():
    # Peak at 1200, trough at 1080 → DD = (1200-1080)/1200 * 100 = 10%
    bt = make_results([], [1000, 1100, 1200, 1080, 1150])
    assert bt.max_drawdown == pytest.approx(10.0)


def test_max_drawdown_dollars():
    # Peak 1200, trough 1080 → $120 dollars DD
    bt = make_results([], [1000, 1200, 1080])
    assert bt.max_drawdown_dollars == pytest.approx(120.0)


def test_max_drawdown_empty_curve():
    bt = make_results([], [])
    assert bt.max_drawdown == 0.0


# ── Recovery factor ───────────────────────────────────────────────────────────

def test_recovery_factor_basic():
    # Net PnL $200, max DD $100 → recovery = 2.0
    trades = [make_trade(200)]
    bt = make_results(trades, [1000, 1200], initial=1000)
    bt.final_balance = 1200
    # max_drawdown_dollars = 0 (pure up), so inf
    assert bt.recovery_factor == float("inf")


def test_recovery_factor_with_dd():
    # Up to 1200, down to 1080, back to 1150
    # Net PnL = 150, max DD dollars = 120 → RF = 150/120 = 1.25
    bt = BacktestResults(
        pair="EURUSDm", days=90,
        initial_balance=1000.0, final_balance=1150.0,
        trades=[], equity_curve=[1000, 1200, 1080, 1150],
    )
    assert bt.recovery_factor == pytest.approx(150.0 / 120.0)


# ── Expectancy ────────────────────────────────────────────────────────────────

def test_expectancy_positive():
    # 60% WR, avg win $10, avg loss -$6.67 → E = 0.6*10 + 0.4*(-6.67) = 6 - 2.67 = 3.33
    trades = [make_trade(10), make_trade(10), make_trade(10), make_trade(-10), make_trade(-10)]
    bt = make_results(trades, [1000] + [1000] * 5)
    # avg_win = 10, avg_loss = -10, wr = 0.6
    assert bt.expectancy == pytest.approx(0.6 * 10 + 0.4 * (-10))


def test_expectancy_no_trades():
    bt = make_results([], [1000.0])
    assert bt.expectancy == 0.0


# ── Sortino ratio ─────────────────────────────────────────────────────────────

def test_sortino_all_positive_returns():
    # Only upward equity → no downside returns → sortino = inf
    bt = make_results([], [1000, 1010, 1020, 1030, 1040])
    assert bt.sortino_ratio == float("inf")


def test_sortino_mixed_returns():
    # Some up, some down — should be finite and positive (net upward)
    bt = make_results([], [1000, 1020, 1010, 1030, 1025, 1040])
    s = bt.sortino_ratio
    assert s > 0
    assert not math.isinf(s)


def test_sortino_all_negative():
    # Net losing curve → should be <= 0
    bt = make_results([], [1000, 990, 980, 970])
    assert bt.sortino_ratio <= 0


def test_sortino_short_curve():
    bt = make_results([], [1000.0])
    assert bt.sortino_ratio == 0.0


# ── Calmar ratio ──────────────────────────────────────────────────────────────

def test_calmar_no_drawdown():
    trades = [make_trade(100)]
    bt = BacktestResults(
        pair="EURUSDm", days=252,
        initial_balance=1000.0, final_balance=1100.0,
        trades=trades, equity_curve=[1000, 1100],
    )
    assert bt.calmar_ratio == float("inf")


def test_calmar_basic():
    # net_pnl_pct = 10%, days=252 → annual_return = 10%
    # max_dd = 5%  → calmar = 10/5 = 2.0
    bt = BacktestResults(
        pair="EURUSDm", days=252,
        initial_balance=1000.0, final_balance=1100.0,
        trades=[], equity_curve=[1000, 1100, 1050, 1100],
    )
    # max_dd = (1100-1050)/1100 * 100 ≈ 4.545%
    # annual_return = (100/1000)*100 / 252 * 252 = 10%
    # calmar ≈ 10 / 4.545 ≈ 2.2
    assert bt.calmar_ratio > 0


# ── Consecutive losses ────────────────────────────────────────────────────────

def test_max_consecutive_losses_none():
    trades = [make_trade(10), make_trade(10)]
    bt = make_results(trades, [1000] * 3)
    assert bt.max_consecutive_losses == 0


def test_max_consecutive_losses_streak():
    trades = [
        make_trade(10), make_trade(-5), make_trade(-5), make_trade(-5), make_trade(10)
    ]
    bt = make_results(trades, [1000] * 6)
    assert bt.max_consecutive_losses == 3


def test_max_consecutive_losses_all_loss():
    trades = [make_trade(-5)] * 4
    bt = make_results(trades, [1000] * 5)
    assert bt.max_consecutive_losses == 4


# ── Verdict ───────────────────────────────────────────────────────────────────

def test_verdict_too_few_trades():
    trades = [make_trade(10)] * 5
    bt = make_results(trades, [1000 + i * 10 for i in range(6)])
    v = bt.verdict()
    assert "Too few trades" in v
    assert "DO NOT DEPLOY" in v


def test_verdict_low_profit_factor():
    # 300 trades but PF < 1.5 (equal wins and losses, 50/50 at $1 each → PF=1.0)
    wins   = [make_trade(1.0)] * 150
    losses = [make_trade(-1.0)] * 150
    curve  = [1000.0]
    for t in wins + losses:
        curve.append(curve[-1] + t.pnl)
    bt = make_results(wins + losses, curve)
    v = bt.verdict()
    assert "Profit factor too low" in v


def test_verdict_high_drawdown():
    # Force max_drawdown > 10% by crafting equity curve
    bt = BacktestResults(
        pair="EURUSDm", days=90,
        initial_balance=1000.0, final_balance=1100.0,
        trades=[make_trade(10)] * 300,  # enough trades
        equity_curve=[1000, 1200, 1080, 1100],  # DD = (1200-1080)/1200 = 10%
    )
    # DD is exactly 10%, which is NOT > 10, so no drawdown issue at 10.0
    # Let's use a curve with 11% DD
    bt.equity_curve = [1000, 1200, 1068, 1100]  # DD = 132/1200 = 11%
    v = bt.verdict()
    assert "Drawdown too high" in v


def test_verdict_pass():
    # Build a result that satisfies all thresholds:
    # 300+ trades, WR>40%, PF>1.5, DD<10%, expectancy>0, RF>2, Sortino>0
    n_wins   = 210
    n_losses = 90
    wins     = [make_trade(2.0)] * n_wins     # avg win $2
    losses   = [make_trade(-1.0)] * n_losses  # avg loss $1

    # Equity curve: pure ascending (so DD ≈ 0, sortino = inf, RF = inf)
    all_trades = wins + losses
    curve = [1000.0]
    for t in all_trades:
        curve.append(curve[-1] + t.pnl)

    bt = BacktestResults(
        pair="EURUSDm", days=90,
        initial_balance=1000.0, final_balance=curve[-1],
        trades=all_trades, equity_curve=curve,
    )
    # PF = (210*2) / (90*1) = 420/90 = 4.67 ✅
    # WR = 210/300 = 70% ✅
    # Expectancy = 0.7*2 + 0.3*(-1) = 1.4-0.3 = 1.1 ✅
    v = bt.verdict()
    assert "DEPLOY" in v
    assert "DO NOT DEPLOY" not in v
