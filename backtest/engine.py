"""
ARIA Backtesting Engine.

Uses the SAME analysis pipeline as the live system (indicators, SMC, MTF,
confluence) on historical MT5 data slices — point-in-time accurate.

Walk-forward approach:
  Every 5 M15 candles (matching the live 5-min scan interval):
    1. Slice historical data up to current bar
    2. Resample M15 → H1, H4, D1 (no lookahead)
    3. Run full indicator + SMC + MTF + confluence pipeline
    4. If signal ≥ threshold and no position → enter
    5. Each candle: check SL/TP hit on OHLC
    6. Partial close at TP1 (50%), move SL to entry
    7. Full close at TP2 or SL

No future data leaks. Commission/spread included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from analysis import confluence as conf_mod
from analysis import indicators as ind
from analysis import smc as smc_mod
from analysis.mtf import analyze as mtf_analyze
from backtest.metrics import BacktestResults, Trade


# ── Resampling ────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M15 OHLCV to a higher timeframe without lookahead."""
    r = df.resample(rule, closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    return r[r["open"].notna()]


def _make_htf(df_m15: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build H1, H4, D1 from a M15 slice. Pure resample — no future data."""
    df_h1 = _resample(df_m15, "1h")
    df_h4 = _resample(df_m15, "4h")
    df_d1 = _resample(df_m15, "1D")
    return df_h1, df_h4, df_d1


# ── Position simulation ───────────────────────────────────────────────────────

@dataclass
class _SimPosition:
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    lots: float
    lots_remaining: float
    entry_bar: int
    score: float
    sl_type: str
    at_breakeven: bool = False
    partial_taken: bool = False
    pnl: float = 0.0


def _candle_hits_sl(pos: _SimPosition, candle: pd.Series) -> bool:
    if pos.direction == "long":
        return candle["low"] <= pos.sl
    return candle["high"] >= pos.sl


def _candle_hits_tp1(pos: _SimPosition, candle: pd.Series) -> bool:
    if pos.direction == "long":
        return candle["high"] >= pos.tp1
    return candle["low"] <= pos.tp1


def _candle_hits_tp2(pos: _SimPosition, candle: pd.Series) -> bool:
    if pos.direction == "long":
        return candle["high"] >= pos.tp2
    return candle["low"] <= pos.tp2


def _pnl_per_pip(pair: str, lots: float, price: float) -> float:
    """USD P&L per pip moved."""
    p = pair.upper().rstrip("M")
    if "JPY" in p:
        pv = 1000 / price
    elif "XAU" in p or "GOLD" in p:
        pv = 100.0
    elif any(x in p for x in ("NAS", "USTEC", "US30", "NDX")):
        pv = 1.0
    elif p.startswith("EUR") or p.startswith("GBP"):
        pv = 10.0
    else:
        pv = 10.0
    return pv * lots


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward backtester for a single pair.
    Fetches historical M15 data from MT5 and simulates the live strategy.
    """

    def __init__(
        self,
        pair: str,
        days: int = 90,
        initial_balance: float = 100.0,
        risk_pct: float = 1.0,
        min_score: float = 70.0,
        spread_pips: float = 0.8,
        slippage_pips: float = 0.3,    # random slippage range ±pips on entry/exit
        commission_per_lot: float = 3.5, # USD commission per standard lot (one way)
        scan_step: int = 5,
        max_bars_in_trade: int = 192,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> None:
        self.pair              = pair
        self.days              = days
        self.initial_balance   = initial_balance
        self.risk_pct          = risk_pct
        self.min_score         = min_score
        self.spread_pips       = spread_pips
        self.slippage_pips     = slippage_pips
        self.commission_per_lot = commission_per_lot
        self.scan_step         = scan_step
        self.max_bars          = max_bars_in_trade
        self.start_date        = start_date
        self.end_date          = end_date

    def run(self, df_m15: Optional[pd.DataFrame] = None) -> BacktestResults:
        """Fetch data (or accept pre-fetched) and run the full backtest."""
        logger.info(f"[Backtest] {self.pair} — {self.days}d @ {self.risk_pct}% risk")

        if df_m15 is None:
            df_m15 = self._fetch_m15()

        # Date-range slice (only when caller didn't pass a pre-sliced df)
        if self.start_date:
            df_m15 = df_m15[df_m15.index >= pd.Timestamp(self.start_date)]
        if self.end_date:
            df_m15 = df_m15[df_m15.index <= pd.Timestamp(self.end_date)]
        # Recompute days from actual slice when dates given
        if self.start_date or self.end_date:
            if not df_m15.empty:
                delta = df_m15.index[-1] - df_m15.index[0]
                self.days = max(1, delta.days)

        if df_m15.empty or len(df_m15) < 150:
            logger.error(f"[Backtest] Not enough M15 data for {self.pair}")
            return BacktestResults(self.pair, self.days, self.initial_balance, self.initial_balance)

        logger.info(f"[Backtest] {len(df_m15)} M15 candles loaded ({self.days}d)")
        t0 = time.time()

        balance = self.initial_balance
        trades: list[Trade] = []
        equity: list[float] = [balance]

        position: Optional[_SimPosition] = None
        warmup = 200   # candles needed for indicators to stabilise

        for i in range(warmup, len(df_m15)):
            candle = df_m15.iloc[i]

            # ── Manage open position ──────────────────────────────
            if position is not None:
                bars_held = i - position.entry_bar
                pip_size  = 0.01 if "JPY" in self.pair.upper() else (0.1 if "XAU" in self.pair.upper() else 0.0001)

                closed = False

                # TP1 partial (50% close, move SL to entry)
                if not position.partial_taken and _candle_hits_tp1(position, candle):
                    tp1_pips = abs(position.tp1 - position.entry) / pip_size
                    partial_pnl = tp1_pips * _pnl_per_pip(self.pair, position.lots * 0.5, candle["close"])
                    position.pnl += partial_pnl
                    balance += partial_pnl
                    position.lots_remaining = position.lots * 0.5
                    position.sl = position.entry   # move SL to breakeven
                    position.at_breakeven = True
                    position.partial_taken = True

                # TP2 full close
                elif position.partial_taken and _candle_hits_tp2(position, candle):
                    tp2_pips = abs(position.tp2 - position.entry) / pip_size
                    final_pnl = tp2_pips * _pnl_per_pip(self.pair, position.lots_remaining, candle["close"])
                    position.pnl += final_pnl
                    balance += final_pnl
                    trades.append(self._close_trade(position, candle, i, balance, "tp2"))
                    position = None
                    closed = True

                # SL hit
                elif _candle_hits_sl(position, candle):
                    if position.at_breakeven:
                        sl_pnl = 0.0
                    else:
                        sl_pips = abs(position.sl - position.entry) / pip_size
                        sl_pnl  = -(sl_pips * _pnl_per_pip(self.pair, position.lots_remaining, candle["close"]))
                    position.pnl += sl_pnl
                    balance += sl_pnl
                    trades.append(self._close_trade(position, candle, i, balance, "sl"))
                    position = None
                    closed = True

                # Time-based exit: too long in trade
                elif bars_held >= self.max_bars:
                    current_pips = (candle["close"] - position.entry) / pip_size
                    if position.direction == "short":
                        current_pips = -current_pips
                    time_pnl = current_pips * _pnl_per_pip(self.pair, position.lots_remaining, candle["close"])
                    position.pnl += time_pnl
                    balance += time_pnl
                    trades.append(self._close_trade(position, candle, i, balance, "time"))
                    position = None
                    closed = True

            # ── Scan for new entry ────────────────────────────────
            if position is None and (i - warmup) % self.scan_step == 0 and i > warmup + 10:
                window = df_m15.iloc[max(0, i - 300):i + 1]  # 300-bar rolling window
                sig = self._score_window(window, candle["close"])

                if sig is not None and sig.score >= self.min_score and sig.direction != "wait":
                    sl, sl_type = self._calc_sl(sig, candle["close"], window)
                    tp1, tp2    = self._calc_tps(sig.direction, candle["close"], sl)
                    lots        = self._calc_lots(balance, candle["close"], sl)

                    if lots > 0 and sl > 0 and abs(candle["close"] - sl) > 0:
                        import random
                        pip_size = 0.01 if "JPY" in self.pair.upper() else (0.1 if "XAU" in self.pair.upper() else 0.0001)
                        slip = random.uniform(0, self.slippage_pips) * pip_size
                        actual_entry = candle["close"] + slip if sig.direction == "long" else candle["close"] - slip
                        # Commission: open side (one way per lot)
                        commission = lots * self.commission_per_lot
                        balance -= commission

                        position = _SimPosition(
                            direction=sig.direction,
                            entry=actual_entry,
                            sl=sl,
                            tp1=tp1,
                            tp2=tp2,
                            lots=lots,
                            lots_remaining=lots,
                            entry_bar=i,
                            score=sig.score,
                            sl_type=sl_type,
                        )
                        position.pnl -= commission  # entry commission

            equity.append(balance)

        # Close any still-open position at end of data
        if position is not None:
            last = df_m15.iloc[-1]
            pip_size = 0.01 if "JPY" in self.pair.upper() else (0.1 if "XAU" in self.pair.upper() else 0.0001)
            current_pips = (last["close"] - position.entry) / pip_size
            if position.direction == "short":
                current_pips = -current_pips
            eod_pnl = current_pips * _pnl_per_pip(self.pair, position.lots_remaining, last["close"])
            position.pnl += eod_pnl
            balance += eod_pnl
            trades.append(self._close_trade(position, last, len(df_m15) - 1, balance, "eod"))

        elapsed = time.time() - t0
        logger.info(f"[Backtest] Done in {elapsed:.1f}s — {len(trades)} trades")

        return BacktestResults(
            pair=self.pair,
            days=self.days,
            initial_balance=self.initial_balance,
            final_balance=balance,
            trades=trades,
            equity_curve=equity,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_m15(self) -> pd.DataFrame:
        from data.mt5_feed import feed
        # M15 candles: days × 96 candles/day + 50 buffer
        count = self.days * 96 + 50
        df = feed.get_candles(self.pair, "M15", count=count)
        return df

    def _score_window(self, window: pd.DataFrame, price: float) -> Optional[conf_mod.ConfluenceScore]:
        """Run full analysis pipeline on a historical data window."""
        try:
            if len(window) < 100:
                return None

            df_m15 = ind.apply_all(window.copy())
            df_h1, df_h4, df_d1 = _make_htf(window)

            if len(df_h4) < 20 or len(df_d1) < 5:
                return None

            df_h1 = ind.apply_all(df_h1)
            df_h4 = ind.apply_all(df_h4)
            df_d1 = ind.apply_all(df_d1)

            mtf = mtf_analyze(df_d1, df_h4, df_h1, df_m15)
            smc = smc_mod.analyze(df_m15)

            return conf_mod.score(
                pair=self.pair,
                mtf=mtf,
                smc=smc,
                spread_pips=self.spread_pips,
                session_pts=10.0,   # simplified: assume session always active
                news_blocked=False,
                current_price=price,
                df_m15=df_m15,
            )
        except Exception as e:
            logger.debug(f"[Backtest] Score error at bar: {e}")
            return None

    def _calc_sl(self, sig: conf_mod.ConfluenceScore, price: float, window: pd.DataFrame) -> tuple[float, str]:
        from signals.entry import _find_sl
        from analysis.indicators import atr_value
        atr_val = atr_value(window) if "atr" in window.columns else 0.0
        if atr_val == 0:
            enriched = ind.apply_all(window.copy())
            atr_val = atr_value(enriched)
        if atr_val == 0:
            atr_val = float((window["high"] - window["low"]).mean())
        return _find_sl(sig, price, window, atr_val, 1.2)

    def _calc_tps(self, direction: str, entry: float, sl: float) -> tuple[float, float]:
        dist = abs(entry - sl)
        if direction == "long":
            return entry + dist * 1.5, entry + dist * 3.0
        return entry - dist * 1.5, entry - dist * 3.0

    def _calc_lots(self, balance: float, entry: float, sl: float) -> float:
        risk_amount = balance * self.risk_pct / 100
        pip_size    = 0.01 if "JPY" in self.pair.upper() else (0.1 if "XAU" in self.pair.upper() else 0.0001)
        sl_pips     = abs(entry - sl) / pip_size
        if sl_pips <= 0:
            return 0.01
        pip_val = _pnl_per_pip(self.pair, 1.0, entry)
        lots = risk_amount / (sl_pips * pip_val)
        return max(0.01, round(min(lots, 10.0), 2))

    def _close_trade(
        self,
        pos: _SimPosition,
        candle: pd.Series,
        bar: int,
        balance: float,
        reason: str,
    ) -> Trade:
        # Exit commission (close side)
        exit_commission = pos.lots_remaining * self.commission_per_lot
        pos.pnl -= exit_commission
        return Trade(
            pair=self.pair,
            direction=pos.direction,
            entry=pos.entry,
            exit=candle["close"],
            sl=pos.sl,
            tp1=pos.tp1,
            lots=pos.lots,
            pnl=round(pos.pnl, 2),
            pnl_pct=round(pos.pnl / self.initial_balance * 100, 3),
            score=pos.score,
            exit_reason=reason,
            entry_bar=pos.entry_bar,
            exit_bar=bar,
            bars_held=bar - pos.entry_bar,
            sl_type=pos.sl_type,
        )
