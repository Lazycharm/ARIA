"""
Unit tests for core/adaptive_learning.py.

Tests adaptive threshold adjustments, lot multiplier scaling,
conservative mode, and per-pair statistics.
"""
import threading
import pytest

from core.adaptive_learning import AdaptiveLearning, PairStats


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def al() -> AdaptiveLearning:
    """Fresh AdaptiveLearning instance (no persisted state, no file I/O)."""
    instance = AdaptiveLearning.__new__(AdaptiveLearning)
    instance._lock = threading.Lock()
    instance._pairs: dict = {}
    instance._global_conservative = False
    instance._global_consecutive_losses = 0
    return instance


def _make_pair(wins: int, losses: int, min_score: float = 70.0,
               lot_mult: float = 1.0) -> PairStats:
    return PairStats(
        pair="EURUSDm",
        wins=wins,
        losses=losses,
        min_score=min_score,
        lot_multiplier=lot_mult,
    )


# ── PairStats properties ──────────────────────────────────────────────────────

def test_pairstat_win_rate_basic():
    ps = _make_pair(6, 4)
    assert abs(ps.win_rate - 60.0) < 0.01


def test_pairstat_win_rate_no_trades():
    ps = _make_pair(0, 0)
    assert ps.win_rate == 0.0


def test_pairstat_total_trades():
    ps = _make_pair(3, 7)
    assert ps.total_trades == 10


def test_pairstat_avg_pnl_default():
    ps = _make_pair(5, 5)
    assert ps.avg_pnl == 0.0


# ── get_stats / all_stats ─────────────────────────────────────────────────────

def test_get_stats_no_data(al):
    stats = al.get_stats("EURUSDm")
    assert stats is None


def test_get_stats_after_trade(al):
    al._pairs["EURUSDm"] = _make_pair(2, 1)
    stats = al.get_stats("EURUSDm")
    assert stats is not None
    assert stats.wins == 2


def test_all_stats_empty(al):
    assert al.all_stats() == {}


def test_all_stats_populated(al):
    al._pairs["EURUSDm"] = _make_pair(5, 5)
    al._pairs["GBPUSDm"] = _make_pair(3, 2)
    result = al.all_stats()
    assert "EURUSDm" in result
    assert "GBPUSDm" in result


# ── get_min_score ─────────────────────────────────────────────────────────────

def test_get_min_score_no_data_uses_default(al):
    score = al.get_min_score("EURUSDm")
    assert 60.0 <= score <= 88.0  # within adaptive range


def test_get_min_score_uses_stored(al):
    al._pairs["EURUSDm"] = _make_pair(5, 5, min_score=77.5)
    score = al.get_min_score("EURUSDm")
    assert abs(score - 77.5) < 0.01


# ── get_lot_multiplier ────────────────────────────────────────────────────────

def test_lot_multiplier_no_data(al):
    mult = al.get_lot_multiplier("EURUSDm")
    assert mult == 1.0


def test_lot_multiplier_uses_stored(al):
    al._pairs["EURUSDm"] = _make_pair(10, 2, lot_mult=1.3)
    mult = al.get_lot_multiplier("EURUSDm")
    assert abs(mult - 1.3) < 0.001


# ── conservative mode ─────────────────────────────────────────────────────────

def test_conservative_mode_off_by_default(al):
    assert al.is_global_conservative() is False


def test_conservative_mode_can_be_toggled(al):
    al._global_conservative = True
    assert al.is_global_conservative() is True


# ── record_outcome bounds ─────────────────────────────────────────────────────

def test_record_win_increments_wins(al):
    al.record_trade("EURUSDm", won=True, pnl=10.0, score=75.0, direction="long", session="london")
    s = al.get_stats("EURUSDm")
    assert s is not None
    assert s.wins == 1
    assert s.losses == 0


def test_record_loss_increments_losses(al):
    al.record_trade("EURUSDm", won=False, pnl=-5.0, score=72.0, direction="long", session="london")
    s = al.get_stats("EURUSDm")
    assert s.losses == 1
    assert s.wins == 0


def test_min_score_floor_never_below_60(al):
    # Simulate 20 consecutive losses
    for _ in range(20):
        al.record_trade("EURUSDm", won=False, pnl=-5.0, score=72.0, direction="long", session="london")
    score = al.get_min_score("EURUSDm")
    assert score >= 60.0


def test_min_score_ceiling_never_above_88(al):
    # Simulate 30 consecutive wins
    for _ in range(30):
        al.record_trade("EURUSDm", won=True, pnl=10.0, score=75.0, direction="long", session="london")
    score = al.get_min_score("EURUSDm")
    assert score <= 88.0


def test_lot_multiplier_floor_never_below_quarter(al):
    # After 20 losses: stored floor = 0.5, but conservative mode (5+ global losses)
    # halves it further → minimum effective = 0.25
    for _ in range(20):
        al.record_trade("EURUSDm", won=False, pnl=-10.0, score=70.0, direction="long", session="london")
    mult = al.get_lot_multiplier("EURUSDm")
    # Stored floor is 0.5; conservative mode applies 0.5× → effective min 0.25
    assert mult >= 0.25
    # Also verify the stored multiplier (before conservative halving) is at its floor
    stored = al.get_stats("EURUSDm").lot_multiplier
    assert stored >= 0.5


def test_lot_multiplier_ceiling_never_above_1_5(al):
    for _ in range(30):
        al.record_trade("EURUSDm", won=True, pnl=20.0, score=80.0, direction="long", session="london")
    mult = al.get_lot_multiplier("EURUSDm")
    assert mult <= 1.5


# ── pair isolation ────────────────────────────────────────────────────────────

def test_pair_stats_are_independent(al):
    al.record_trade("EURUSDm", won=True, pnl=10.0, score=75.0, direction="long", session="london")
    al.record_trade("GBPUSDm", won=False, pnl=-5.0, score=68.0, direction="short", session="new_york")
    eu = al.get_stats("EURUSDm")
    gb = al.get_stats("GBPUSDm")
    assert eu.wins   == 1 and eu.losses == 0
    assert gb.losses == 1 and gb.wins   == 0


# ── win_rate convergence ──────────────────────────────────────────────────────

def test_win_rate_converges_to_true_rate(al):
    import random
    random.seed(42)
    true_wr = 0.60
    for _ in range(50):
        won = random.random() < true_wr
        al.record_trade("EURUSDm", won=won, pnl=5.0 if won else -5.0, score=72.0, direction="long", session="london")
    s = al.get_stats("EURUSDm")
    assert abs(s.win_rate - true_wr * 100) < 20  # within 20pp is fine for 50 trades
