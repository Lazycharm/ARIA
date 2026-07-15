"""
Unit tests for analysis/confluence.py.

Builds synthetic MTFBias and SMCResult objects to exercise the scorer
without any MT5 connection or live data fetch.
"""
import pytest

from analysis.confluence import score, ConfluenceScore
from analysis.mtf import MTFBias, TimeframeBias
from analysis.smc import SMCResult, OrderBlock, FVG


# ── MTFBias / SMCResult builders ──────────────────────────────────────────────

def make_tf(direction: str = "bullish", rsi: float = 45.0, strength: float = 0.7) -> TimeframeBias:
    return TimeframeBias(
        timeframe="M15", direction=direction,
        strength=strength, rsi=rsi,
        rsi_zone="neutral", price=1.10000,
        ema21=1.0990, ema50=1.0970, ema200=1.0900,
    )


def make_mtf(
    direction: str = "bullish",
    signal: str = "long",
    aligned: bool = True,
    confidence: float = 0.8,
    m15_direction: str = "bullish",
    m15_rsi: float = 45.0,
) -> MTFBias:
    tf = make_tf(direction=m15_direction, rsi=m15_rsi)
    return MTFBias(
        d1=make_tf(direction),
        h4=make_tf(direction),
        h1=make_tf(direction),
        m15=tf,
        aligned=aligned,
        direction=direction,
        confidence=confidence,
        signal_direction=signal,
    )


def make_smc(
    bullish_ob: bool = False,
    bullish_fvg: bool = False,
    bearish_ob: bool = False,
    bearish_fvg: bool = False,
    bos_direction: str | None = None,
    price: float = 1.10000,
) -> SMCResult:
    smc = SMCResult()
    if bullish_ob:
        smc.bullish_obs.append(OrderBlock(
            index=0, direction="bullish",
            top=price + 0.0010, bottom=price - 0.0010,
            candle_time=None, strength=1.0,
        ))
    if bullish_fvg:
        smc.bullish_fvgs.append(FVG(
            index=0, direction="bullish",
            top=price + 0.0008, bottom=price - 0.0008,
            candle_time=None,
        ))
    if bearish_ob:
        smc.bearish_obs.append(OrderBlock(
            index=0, direction="bearish",
            top=price + 0.0010, bottom=price - 0.0010,
            candle_time=None, strength=1.0,
        ))
    if bearish_fvg:
        smc.bearish_fvgs.append(FVG(
            index=0, direction="bearish",
            top=price + 0.0008, bottom=price - 0.0008,
            candle_time=None,
        ))
    if bos_direction:
        smc.bos.append({"direction": bos_direction, "level": price, "index": 5})
    return smc


# ── score() result type ───────────────────────────────────────────────────────

def test_returns_confluence_score_object():
    result = score(
        pair="EURUSDm",
        mtf=make_mtf(),
        smc=make_smc(),
        spread_pips=1.0,
        news_blocked=False,
        current_price=1.10000,
        session_pts=10.0,
    )
    assert isinstance(result, ConfluenceScore)
    assert result.pair == "EURUSDm"
    assert 0.0 <= result.score <= 100.0


# ── MTF component ─────────────────────────────────────────────────────────────

def test_full_mtf_alignment_gives_35pts():
    mtf = make_mtf(aligned=True, confidence=1.0)
    smc = make_smc()
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("mtf", 0) == 35.0


def test_strong_mtf_partial_gives_22pts():
    mtf = make_mtf(aligned=False, confidence=0.65)
    smc = make_smc()
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("mtf", 0) == 22.0


def test_ranging_mtf_gives_zero_pts():
    mtf = make_mtf(direction="ranging", signal="wait", aligned=False, confidence=0.2)
    smc = make_smc()
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("mtf", 0) == 0.0
    assert result.direction == "wait"


# ── SMC component ─────────────────────────────────────────────────────────────

def test_inside_bullish_ob_gives_12pts():
    price = 1.10000
    mtf = make_mtf(direction="bullish", signal="long")
    smc = make_smc(bullish_ob=True, price=price)  # OB straddles price
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=price, session_pts=0.0)
    assert result.breakdown.get("smc", 0) >= 12.0


def test_inside_bullish_fvg_gives_8pts():
    price = 1.10000
    mtf = make_mtf(direction="bullish", signal="long")
    smc = make_smc(bullish_fvg=True, price=price)
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=price, session_pts=0.0)
    assert result.breakdown.get("smc", 0) >= 8.0


def test_no_smc_gives_zero_pts():
    mtf = make_mtf(direction="bullish", signal="long")
    smc = SMCResult()  # empty — no OBs or FVGs
    result = score("EURUSDm", mtf, smc, spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("smc", 0) == 0.0


# ── RSI component ─────────────────────────────────────────────────────────────

def test_rsi_optimal_long_zone_gives_15pts():
    # RSI 30-55 for long = full 15pts
    mtf = make_mtf(signal="long", m15_rsi=42.0)
    result = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("rsi", 0) == 15.0


def test_rsi_optimal_short_zone_gives_15pts():
    # RSI 45-70 for short = full 15pts
    mtf = make_mtf(direction="bearish", signal="short", m15_rsi=58.0, m15_direction="bearish")
    result = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("rsi", 0) == 15.0


def test_rsi_oversold_long_gives_partial():
    # RSI < 30 for long = 8pts (partial credit)
    mtf = make_mtf(signal="long", m15_rsi=25.0)
    result = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("rsi", 0) == 8.0


# ── EMA filter ────────────────────────────────────────────────────────────────

def test_ema_aligned_long_gives_15pts():
    mtf = make_mtf(signal="long", m15_direction="bullish")
    result = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("ema", 0) == 15.0


def test_ema_ranging_gives_5pts():
    mtf = make_mtf(signal="long", m15_direction="ranging")
    result = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert result.breakdown.get("ema", 0) == 5.0


# ── Session, news, spread ─────────────────────────────────────────────────────

def test_session_10pts_added():
    # session_active=False disables the fallback so session_pts=0 actually gives 0
    mtf = make_mtf()
    r_with    = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000,
                      session_pts=10.0, session_active=False)
    r_without = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000,
                      session_pts=0.0, session_active=False)
    assert r_with.score > r_without.score
    assert r_with.breakdown["session"] == 10.0
    assert r_without.breakdown["session"] == 0.0


def test_news_blocked_subtracts_20pts():
    mtf = make_mtf()
    r_ok = score("EURUSDm", mtf, make_smc(), 1.0, news_blocked=False,
                 current_price=1.10000, session_pts=0.0)
    r_news = score("EURUSDm", mtf, make_smc(), 1.0, news_blocked=True,
                   current_price=1.10000, session_pts=0.0)
    diff = r_ok.breakdown.get("news", 0) - r_news.breakdown.get("news", 0)
    assert diff == pytest.approx(25.0)   # +5 vs -20 = 25 difference


def test_high_spread_penalises_score():
    mtf = make_mtf()
    r_ok   = score("EURUSDm", mtf, make_smc(), spread_pips=1.0,
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    r_wide = score("EURUSDm", mtf, make_smc(), spread_pips=10.0,  # exceeds max_spread_pips
                   news_blocked=False, current_price=1.10000, session_pts=0.0)
    assert r_ok.score > r_wide.score


# ── ML boost & sentiment ──────────────────────────────────────────────────────

def test_ml_boost_increases_score():
    mtf = make_mtf()
    r_base  = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, ml_boost=0.0)
    r_boost = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, ml_boost=10.0)
    assert r_boost.score > r_base.score


def test_ml_boost_negative_decreases_score():
    mtf = make_mtf()
    r_base = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, ml_boost=0.0)
    r_bad  = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, ml_boost=-10.0)
    assert r_bad.score < r_base.score


def test_positive_sentiment_increases_score():
    mtf = make_mtf()
    r_base = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, sentiment_pts=0.0)
    r_pos  = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, sentiment_pts=8.0)
    assert r_pos.score > r_base.score


# ── Score clamping ────────────────────────────────────────────────────────────

def test_score_never_below_zero():
    # Pile on all penalties: news blocked + high spread + ranging MTF
    mtf = make_mtf(direction="ranging", signal="wait", aligned=False, confidence=0.1)
    result = score("EURUSDm", mtf, make_smc(), spread_pips=10.0,
                   news_blocked=True, current_price=1.10000,
                   ml_boost=-15.0, session_pts=0.0)
    assert result.score >= 0.0


def test_score_never_above_100():
    # Everything perfect + ML boost
    mtf = make_mtf(aligned=True, confidence=1.0)
    smc = make_smc(bullish_ob=True, bullish_fvg=True, bos_direction="bullish")
    result = score("EURUSDm", mtf, smc, spread_pips=0.5,
                   news_blocked=False, current_price=1.10000,
                   ml_boost=15.0, session_pts=10.0, sentiment_pts=10.0)
    assert result.score <= 100.0


# ── Tradeable / auto-executable thresholds ────────────────────────────────────

def test_tradeable_above_65():
    cs = ConfluenceScore(pair="EURUSDm", direction="long", score=65.0)
    assert cs.tradeable is True


def test_not_tradeable_below_65():
    cs = ConfluenceScore(pair="EURUSDm", direction="long", score=64.9)
    assert cs.tradeable is False


def test_wait_direction_not_tradeable():
    cs = ConfluenceScore(pair="EURUSDm", direction="wait", score=80.0)
    assert cs.tradeable is False


def test_auto_executable_above_70():
    cs = ConfluenceScore(pair="EURUSDm", direction="short", score=70.0)
    assert cs.auto_executable is True


def test_not_auto_executable_below_70():
    cs = ConfluenceScore(pair="EURUSDm", direction="short", score=69.9)
    assert cs.auto_executable is False


# ── BOS component ─────────────────────────────────────────────────────────────

def test_aligned_bos_adds_10pts():
    mtf = make_mtf(signal="long")
    r_no_bos = score("EURUSDm", mtf, make_smc(), 1.0, False, 1.10000, session_pts=0.0)
    r_bos    = score("EURUSDm", mtf, make_smc(bos_direction="bullish"), 1.0, False,
                     1.10000, session_pts=0.0)
    diff = r_bos.breakdown.get("bos", 0) - r_no_bos.breakdown.get("bos", 0)
    assert diff == pytest.approx(10.0)


def test_opposing_bos_penalises():
    mtf = make_mtf(signal="long")
    result = score("EURUSDm", mtf, make_smc(bos_direction="bearish"),
                   1.0, False, 1.10000, session_pts=0.0)
    assert result.breakdown.get("bos", 0) < 0


# ── signal_pct ────────────────────────────────────────────────────────────────

def test_signal_pct_long_positive():
    cs = ConfluenceScore(pair="EURUSDm", direction="long", score=75.0)
    assert cs.signal_pct == 75.0


def test_signal_pct_short_negative():
    cs = ConfluenceScore(pair="EURUSDm", direction="short", score=75.0)
    assert cs.signal_pct == -75.0


def test_signal_pct_wait_zero():
    cs = ConfluenceScore(pair="EURUSDm", direction="wait", score=40.0)
    assert cs.signal_pct == 0.0
