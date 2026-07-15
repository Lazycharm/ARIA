"""
Unit tests for core/capital.py.

Tests the CapitalManager permission gate, position sizing, weekly/monthly
drawdown limits, and trade lifecycle tracking. No MT5 dependency.
"""
import pytest

from core.capital import CapitalManager, WEEKLY_DD_LIMIT_PCT, MONTHLY_DD_LIMIT_PCT


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cap() -> CapitalManager:
    """Fresh CapitalManager using default settings."""
    return CapitalManager()


# ── can_trade: happy path ─────────────────────────────────────────────────────

def test_can_trade_fresh(cap):
    perm = cap.can_trade()
    assert perm.allowed is True


def test_can_trade_with_pair(cap):
    perm = cap.can_trade("EURUSDm")
    assert perm.allowed is True


# ── can_trade: daily gates ────────────────────────────────────────────────────

def test_max_trades_per_day_blocks(cap):
    cap.day.trades_taken = cap._lock.__class__  # trick: set to large number
    from config.settings import settings
    cap.day.trades_taken = settings.max_trades_per_day
    perm = cap.can_trade()
    assert perm.allowed is False
    assert "Max trades" in perm.reason


def test_max_concurrent_blocks(cap):
    from config.settings import settings
    # Fill open_positions up to the limit
    for i in range(settings.max_concurrent_trades):
        cap.open_positions[f"PAIR{i}"] = {"pair": f"PAIR{i}", "pnl": 0.0}
    perm = cap.can_trade()
    assert perm.allowed is False
    assert "concurrent" in perm.reason


def test_daily_loss_limit_halts(cap):
    from config.settings import settings
    # Exceed the daily loss limit
    cap.day.realized_pnl = -(settings.max_loss_amount + 0.01)
    perm = cap.can_trade()
    assert perm.allowed is False
    assert cap._trading_halted is True
    assert "loss limit" in perm.reason.lower() or "Daily loss" in perm.reason


def test_daily_target_halts(cap):
    from config.settings import settings
    cap.day.realized_pnl = settings.daily_target_amount + 0.01
    perm = cap.can_trade()
    assert perm.allowed is False
    assert cap._trading_halted is True


def test_hard_halt_blocks(cap):
    cap._trading_halted = True
    cap._halt_reason = "Manual halt"
    perm = cap.can_trade()
    assert perm.allowed is False
    assert "halted" in perm.reason.lower() or "Manual" in perm.reason


# ── can_trade: weekly / monthly drawdown ─────────────────────────────────────

def test_weekly_dd_limit_blocks(cap):
    # Set week PnL to -7% of week start balance → exceeds 6% limit
    cap._week_start_balance = 1000.0
    cap._week_realized_pnl = -70.0   # −7%
    perm = cap.can_trade()
    assert perm.allowed is False
    assert "Weekly" in perm.reason or "weekly" in perm.reason


def test_monthly_dd_limit_blocks(cap):
    cap._month_start_balance = 1000.0
    cap._month_realized_pnl = -110.0   # −11%
    perm = cap.can_trade()
    assert perm.allowed is False
    assert "Monthly" in perm.reason or "monthly" in perm.reason


def test_weekly_dd_exactly_at_limit_blocks(cap):
    cap._week_start_balance = 1000.0
    cap._week_realized_pnl = -(WEEKLY_DD_LIMIT_PCT / 100 * 1000.0)  # exactly at limit
    perm = cap.can_trade()
    assert perm.allowed is False


def test_weekly_dd_just_under_limit_ok(cap):
    cap._week_start_balance = 1000.0
    cap._week_realized_pnl = -(WEEKLY_DD_LIMIT_PCT / 100 * 1000.0 - 0.01)  # just under
    perm = cap.can_trade()
    assert perm.allowed is True


# ── can_trade: emergency drawdown ────────────────────────────────────────────

def test_emergency_drawdown_halts(cap):
    from config.settings import settings
    # Drop equity below the emergency threshold
    cap.equity = cap.balance * (1 - settings.emergency_drawdown_pct / 100) - 0.01
    perm = cap.can_trade()
    assert perm.allowed is False
    assert cap._trading_halted is True


# ── calculate_lots ────────────────────────────────────────────────────────────

def test_calculate_lots_eurusd(cap):
    # 1% risk on $1000 = $10 risk
    # SL = 50 pips, pip value = $10/lot → lots = 10/(50*10) = 0.02
    lots = cap.calculate_lots("EURUSDm", entry=1.10000, sl=1.09500)
    assert lots == pytest.approx(0.02, abs=0.005)


def test_calculate_lots_usdjpy(cap):
    # pip_size = 0.01, pip_value ≈ 1000/150 ≈ 6.67
    # SL = 50 pips → lots = 10 / (50 * 6.67) ≈ 0.03
    lots = cap.calculate_lots("USDJPYm", entry=150.00, sl=149.50)
    assert lots >= 0.01
    assert lots <= 0.10


def test_calculate_lots_xauusd(cap):
    # pip_size = 0.1, pip_value = 100/lot
    # SL = 2 pips → lots = 10 / (20 * 100) = 0.005 → clamped to min 0.01
    lots = cap.calculate_lots("XAUUSDm", entry=2000.00, sl=1999.80)
    assert lots >= 0.01


def test_calculate_lots_zero_sl_returns_min(cap):
    # entry == sl → sl_pips = 0 → return min lot
    lots = cap.calculate_lots("EURUSDm", entry=1.10000, sl=1.10000)
    assert lots == 0.01


def test_calculate_lots_respect_min(cap):
    # Very wide SL → computed lots tiny → clamped to 0.01
    lots = cap.calculate_lots("EURUSDm", entry=1.10000, sl=1.00000)  # 1000 pip SL
    assert lots >= 0.01


# ── register_close: accumulation ─────────────────────────────────────────────

def test_register_close_win(cap):
    cap.open_positions["EURUSDm"] = {"pair": "EURUSDm", "pnl": 0.0}
    cap.register_close("EURUSDm", close_price=1.1050, pnl=10.0)
    assert cap.day.realized_pnl == pytest.approx(10.0)
    assert cap.day.trades_won == 1
    assert cap.day.gross_profit == pytest.approx(10.0)
    assert cap._week_realized_pnl == pytest.approx(10.0)
    assert cap._month_realized_pnl == pytest.approx(10.0)


def test_register_close_loss(cap):
    cap.open_positions["EURUSDm"] = {"pair": "EURUSDm", "pnl": 0.0}
    cap.register_close("EURUSDm", close_price=1.0950, pnl=-10.0)
    assert cap.day.realized_pnl == pytest.approx(-10.0)
    assert cap.day.trades_lost == 1
    assert cap.day.gross_loss == pytest.approx(-10.0)
    assert cap._week_realized_pnl == pytest.approx(-10.0)
    assert cap._month_realized_pnl == pytest.approx(-10.0)


def test_register_close_removes_position(cap):
    cap.open_positions["EURUSDm"] = {"pair": "EURUSDm", "pnl": 0.0}
    cap.register_close("EURUSDm", close_price=1.1050, pnl=5.0)
    assert "EURUSDm" not in cap.open_positions


def test_register_close_multiple_accumulates(cap):
    for pnl in [10.0, -5.0, 8.0]:
        cap.open_positions["EURUSDm"] = {"pair": "EURUSDm", "pnl": 0.0}
        cap.register_close("EURUSDm", close_price=1.1050, pnl=pnl)
    assert cap.day.realized_pnl == pytest.approx(13.0)
    assert cap._week_realized_pnl == pytest.approx(13.0)
    assert cap.day.trades_won == 2
    assert cap.day.trades_lost == 1


# ── status_dict ───────────────────────────────────────────────────────────────

def test_status_dict_has_weekly_monthly_keys(cap):
    s = cap.status_dict
    assert "weekly_pnl" in s
    assert "monthly_pnl" in s
    assert "weekly_dd_pct" in s
    assert "monthly_dd_pct" in s
    assert "weekly_dd_limit" in s
    assert "monthly_dd_limit" in s


def test_status_dict_limits_match_constants(cap):
    s = cap.status_dict
    assert s["weekly_dd_limit"] == WEEKLY_DD_LIMIT_PCT
    assert s["monthly_dd_limit"] == MONTHLY_DD_LIMIT_PCT


def test_status_dict_dd_pct_calculation(cap):
    cap._week_start_balance = 1000.0
    cap._week_realized_pnl = -30.0
    s = cap.status_dict
    assert s["weekly_dd_pct"] == pytest.approx(3.0)   # 30/1000 * 100


# ── resume trading ────────────────────────────────────────────────────────────

def test_resume_trading(cap):
    cap._trading_halted = True
    cap._halt_reason = "test"
    cap.resume_trading()
    assert cap._trading_halted is False
    assert cap._halt_reason == ""
    assert cap.can_trade().allowed is True
