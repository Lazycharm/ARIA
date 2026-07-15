"""
Signal scanner — the engine that runs every 5 minutes.

Scans ALL pairs in the dynamic watchlist, not just session-filtered ones.
Session quality is captured as a score component (10 / 5 / 0 pts), so
off-session pairs naturally score lower without being silently excluded.

For each pair:
  1. Fetch multi-timeframe candles from MT5
  2. Run technical indicators
  3. Run SMC analysis on M15
  4. Build MTF bias
  5. Calculate confluence score (session-weighted per pair)
  6. Emit signals above threshold to shared state store
  7. Append to signal history (all pairs, all scores)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from analysis import confluence as conf_mod
from analysis import indicators as ind
from analysis import smc as smc_mod
from analysis.mtf import analyze as mtf_analyze
from config.pairs_config import get_pairs
from config.settings import settings
from core.adaptive_learning import adaptive
from core.session import SessionManager, Session, BEST_PAIRS_BY_SESSION_BASES
from core.strategy import get_strategy, StrategyMode, StrategyConfig
from data.mt5_feed import feed
from data.sentiment import sentiment_cache
from ml.features import extract as extract_features
from ml.predictor import predictor as ml_predictor
from signals.filter import SignalFilter

# ── Alternative strategy configs ──────────────────────────────────────────────
# These override min_score_delta so WAIT-regime pairs aren't blocked by the
# SMC 999-delta that would otherwise silence all non-trending markets.
_CFG_SESSION_BREAKOUT = StrategyConfig(
    mode=StrategyMode.BREAKOUT, min_score_delta=0.0, lot_multiplier=0.85,
    tp_ratio=1.2, atr_sl_mult=1.0, label="SESSION_BREAKOUT",
)
_CFG_MEAN_REVERSION = StrategyConfig(
    mode=StrategyMode.WAIT, min_score_delta=0.0, lot_multiplier=0.75,
    tp_ratio=1.0, atr_sl_mult=0.8, label="MEAN_REVERSION",
)
_CFG_RANGE_TRADING = StrategyConfig(
    mode=StrategyMode.WAIT, min_score_delta=0.0, lot_multiplier=0.75,
    tp_ratio=1.0, atr_sl_mult=0.8, label="RANGE_TRADING",
)

# Global signal store — dashboard reads from here
_signals: dict[str, conf_mod.ConfluenceScore] = {}
_signals_lock = threading.Lock()
_last_scan: datetime | None = None

# Signal history — every pair scored on every scan (last 500 entries)
_history: list[dict] = []
_history_lock = threading.Lock()
_HISTORY_MAX = 500

session_mgr = SessionManager()
sig_filter  = SignalFilter()


def get_signals() -> dict[str, conf_mod.ConfluenceScore]:
    with _signals_lock:
        return dict(_signals)


def get_last_scan_time() -> Optional[datetime]:
    return _last_scan


def get_signal_history() -> list[dict]:
    with _history_lock:
        return list(reversed(_history))


def _session_score_for_pair(pair: str) -> float:
    """
    Session weight for this specific pair (0, 5, or 10 pts).

    10 → pair is in this session's optimal list (best liquidity/momentum)
     5 → session is active but pair isn't in the priority list
     0 → dead/inactive session
    """
    session = session_mgr.current_session()
    session_active = session_mgr.is_trading_allowed()
    if not session_active:
        return 0.0

    pair_base = pair.upper().rstrip("M")
    session_bases = BEST_PAIRS_BY_SESSION_BASES.get(session, [])
    if any(pair_base == b.upper() for b in session_bases):
        return 10.0
    return 5.0


def _wrap_confluence(
    pair: str,
    direction: str,
    score: float,
    reason: str,
    strategy_cfg: StrategyConfig,
    entry: float = 0.0,
    sl: float = 0.0,
    tp1: float = 0.0,
    tp2: float = 0.0,
) -> conf_mod.ConfluenceScore:
    """Convert an alternative-strategy signal into a ConfluenceScore so the rest
    of the pipeline (order_manager, lifecycle, equity tracking) handles it uniformly.

    When entry/sl/tp1/tp2 are provided the strategy's own SL/TP targets are stored
    as a preset TradeSetup so build_setup() returns them directly instead of
    computing generic ATR-based exits.
    """
    result = conf_mod.ConfluenceScore(
        pair=pair,
        direction=direction,
        score=score,
        breakdown={"alt_strategy": score},
        entry_reason=reason,
    )
    result._strategy    = strategy_cfg
    result._ml_features = {}
    result._ml_boost    = 0.0
    result._preset_setup = None

    if sl and tp1 and entry:
        from signals.entry import TradeSetup, _pip_size
        pip      = _pip_size(pair)
        sl_dist  = abs(entry - sl)
        sl_pips  = round(sl_dist / pip, 1) if pip > 0 else 0.0
        # TP3 = extend by the same distance past TP2 (let runner run)
        tp2_val  = tp2 if tp2 else tp1
        tp3_val  = tp2_val + (tp2_val - entry) if direction == "long" else tp2_val - (entry - tp2_val)
        rr1 = round(abs(tp1 - entry) / sl_dist, 2) if sl_dist > 0 else 1.5
        rr2 = round(abs(tp2_val - entry) / sl_dist, 2) if sl_dist > 0 else 3.0
        rr3 = round(abs(tp3_val - entry) / sl_dist, 2) if sl_dist > 0 else 5.0
        result._preset_setup = TradeSetup(
            pair=pair,
            direction=direction,
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2_val, 5),
            tp3=round(tp3_val, 5),
            sl_pips=sl_pips,
            rr1=rr1,
            rr2=rr2,
            rr3=rr3,
            reason=reason,
            sl_type="strategy",
        )

    return result


def scan_pair(pair: str) -> Optional[conf_mod.ConfluenceScore]:
    """Full analysis pipeline for one pair. Returns ConfluenceScore or None."""
    try:
        # ── Fetch data ────────────────────────────────────────────
        tick = feed.get_tick(pair)
        if not tick:
            return None

        spread_pips   = tick["spread_pips"]
        current_price = tick["mid"]

        df_d1  = feed.get_candles(pair, "D1",  count=200)
        df_h4  = feed.get_candles(pair, "H4",  count=200)
        df_h1  = feed.get_candles(pair, "H1",  count=100)
        df_m15 = feed.get_candles(pair, "M15", count=100)

        if df_m15.empty or df_h4.empty:
            return None

        # ── Indicators (apply before SMC so ATR is available) ────
        df_d1  = ind.apply_all(df_d1)
        df_h4  = ind.apply_all(df_h4)
        df_h1  = ind.apply_all(df_h1)
        df_m15 = ind.apply_all(df_m15)

        # ── MTF bias ──────────────────────────────────────────────
        mtf = mtf_analyze(df_d1, df_h4, df_h1, df_m15)

        # ── SMC on M15 ────────────────────────────────────────────
        smc = smc_mod.analyze(df_m15)

        # ── Session & news filters ────────────────────────────────
        session_pts = _session_score_for_pair(pair)
        news_blocked, news_reason = sig_filter.news_blocked(pair)

        if news_blocked:
            logger.debug(f"News blocked: {pair} — {news_reason}")

        # ── Phase 5: Strategy regime ──────────────────────────────
        strategy = get_strategy(df_m15)

        # ── Alternative strategies for non-TREND regimes ──────────
        # WAIT (ADX < 15): SMC is silenced — try mean reversion and range trading.
        # BREAKOUT (ADX 15-25): SMC runs but also check session breakout.
        if strategy.mode == StrategyMode.WAIT:
            try:
                from strategies.mean_reversion import scan_mean_reversion
                mr = scan_mean_reversion(pair, df_m15)
                if mr:
                    return _wrap_confluence(
                        pair, mr.direction, mr.score, mr.reason, _CFG_MEAN_REVERSION,
                        entry=mr.entry, sl=mr.sl, tp1=mr.tp1, tp2=mr.tp2,
                    )
            except Exception:
                pass
            try:
                from strategies.range_trading import scan_range_trading
                rt = scan_range_trading(pair, df_m15)
                if rt:
                    return _wrap_confluence(
                        pair, rt.direction, rt.score, rt.reason, _CFG_RANGE_TRADING,
                        entry=rt.entry, sl=rt.sl, tp1=rt.tp1, tp2=rt.tp2,
                    )
            except Exception:
                pass
            return None   # no alternative signal — skip pair

        if strategy.mode == StrategyMode.BREAKOUT:
            try:
                from strategies.session_breakout import scan_breakout
                bk = scan_breakout(pair, df_m15)
                if bk:
                    # Return the better of SMC and breakout signals after SMC is computed;
                    # store for comparison below.
                    _pending_breakout = bk
                else:
                    _pending_breakout = None
            except Exception:
                _pending_breakout = None
        else:
            _pending_breakout = None

        # ── Phase 6: ML boost ─────────────────────────────────────
        ml_boost = 0.0
        _ml_features: dict = {}
        if ml_predictor.is_ready():
            try:
                _ml_features = extract_features(
                    breakdown={},  # filled after score — placeholder
                    df_m15=df_m15,
                    direction=mtf.signal_direction,
                    session_pts=session_pts,
                    spread_pips=spread_pips,
                    total_score=0.0,  # placeholder
                )
                ml_boost = ml_predictor.get_boost(_ml_features)
            except Exception:
                ml_boost = 0.0

        # ── Phase 7: Reddit sentiment ─────────────────────────────
        sentiment_pts = sentiment_cache.get_pts(pair)

        # ── Confluence score (with df_m15 for ADX/MACD/Stoch) ───────
        result = conf_mod.score(
            pair=pair,
            mtf=mtf,
            smc=smc,
            spread_pips=spread_pips,
            session_pts=session_pts,
            news_blocked=news_blocked,
            current_price=current_price,
            df_m15=df_m15,
            ml_boost=ml_boost,
            sentiment_pts=sentiment_pts,
        )

        # Store feature vector on the result so lifecycle can save it for ML training
        result._ml_features = extract_features(
            breakdown=result.breakdown,
            df_m15=df_m15,
            direction=result.direction,
            session_pts=session_pts,
            spread_pips=spread_pips,
            total_score=result.score,
        )
        result._strategy  = strategy
        result._ml_boost  = ml_boost

        # ── Merge breakout signal if it beats SMC score ───────────
        if _pending_breakout and _pending_breakout.score > result.score:
            result = _wrap_confluence(
                pair, _pending_breakout.direction,
                _pending_breakout.score, _pending_breakout.reason,
                _CFG_SESSION_BREAKOUT,
                entry=_pending_breakout.entry,
                sl=_pending_breakout.sl,
                tp1=_pending_breakout.tp1,
                tp2=_pending_breakout.tp2,
            )

        logger.debug(
            f"Scanned {pair} dir={result.direction} score={result.score:.0f} "
            f"regime={strategy.mode.value} ml={ml_boost:+.0f} senti={sentiment_pts:+.0f}"
        )

        # ── Get pair-specific min score from adaptive learning ────
        pair_min_score = adaptive.get_min_score(pair)

        # ── Append to history (all pairs, all scores) ─────────────
        entry = {
            "time": datetime.now(timezone.utc),
            "pair": pair,
            "direction": result.direction,
            "score": result.score,
            "reason": result.entry_reason,
            "above_threshold": result.score >= pair_min_score and result.direction != "wait",
            "session_pts": session_pts,
            "regime": strategy.mode.value,
            "ml_boost": ml_boost,
            "sentiment": sentiment_pts,
        }
        with _history_lock:
            _history.append(entry)
            if len(_history) > _HISTORY_MAX:
                _history.pop(0)

        return result

    except Exception as e:
        logger.error(f"Scan error for {pair}: {e}")
        return None


def scan_all() -> dict[str, conf_mod.ConfluenceScore]:
    """
    Scan ALL pairs in the dynamic watchlist.
    Updates global signal store with results above each pair's adaptive threshold.
    """
    global _last_scan

    # Get all pairs from dynamic config (not session-filtered)
    all_pairs = get_pairs()

    # Skip entirely during dead session to avoid wasted API calls
    if not session_mgr.is_trading_allowed():
        logger.debug("Dead session — scan skipped")
        return {}

    logger.info(f"Scanning {len(all_pairs)} pairs — {session_mgr.current_session().value}")

    new_signals: dict[str, conf_mod.ConfluenceScore] = {}
    for pair in all_pairs:
        result = scan_pair(pair)
        if result and result.direction != "wait":
            pair_min = adaptive.get_min_score(pair)
            # Apply strategy regime delta (e.g. BREAKOUT requires +3 pts above adaptive threshold)
            strategy = getattr(result, "_strategy", None)
            regime_delta = strategy.min_score_delta if strategy else 0.0
            effective_min = pair_min + regime_delta

            if result.score >= effective_min:
                new_signals[pair] = result

    with _signals_lock:
        _signals.clear()
        _signals.update(new_signals)

    _last_scan = datetime.now(timezone.utc)

    if new_signals:
        top = sorted(new_signals.values(), key=lambda s: s.score, reverse=True)
        for s in top[:5]:
            threshold = adaptive.get_min_score(s.pair)
            logger.info(
                f"Signal: {s.pair} {s.label()} "
                f"(min={threshold:.0f}) — {s.entry_reason[:55]}"
            )

    return new_signals


class ScannerLoop:
    """Background scanner that calls scan_all() every N seconds."""

    def __init__(self, interval_seconds: int = 300) -> None:
        self.interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="scanner")
        self._thread.start()
        logger.info(f"Scanner started — interval: {self.interval}s, pairs: {len(get_pairs())}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scanner stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                scan_all()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
            self._stop_event.wait(self.interval)
