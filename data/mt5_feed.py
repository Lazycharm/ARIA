"""
MT5 data feed — wraps MetaTrader5 Python API.

All data fetching goes through here. Handles connection, reconnection,
and data normalization. Runs synchronously (MT5 API is not async-safe).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from config.settings import settings

# MT5 timeframe constants
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}


class MT5Feed:
    """Singleton wrapper around MetaTrader5 Python API."""

    _connected: bool = False

    def connect(self) -> bool:
        if self._connected:
            return True
        if not settings.mt5_enabled:
            logger.warning("MT5 disabled in settings — running in feed-only mode")
            return False
        ok = mt5.initialize(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
        )
        if ok:
            info = mt5.account_info()
            self._connected = True
            logger.info(
                "MT5 connected",
                account=info.login,
                server=info.server,
                balance=info.balance,
                leverage=info.leverage,
            )
        else:
            err = mt5.last_error()
            logger.error("MT5 connection failed", error=err)
        return ok

    def disconnect(self) -> None:
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        if not self._connected:
            return self.connect()
        return True

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_candles(self, pair: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        """
        Fetch OHLCV candles. Checks cache first, cleans result, falls back to
        Yahoo Finance if MT5 is unavailable (indices/FX only, limited history).
        """
        from data.cache import get as cache_get, put as cache_put
        from data.cleaner import clean

        cached = cache_get(pair, timeframe, count)
        if cached is not None:
            return cached

        df = self._fetch_mt5_candles(pair, timeframe, count)

        if df.empty:
            df = self._fetch_yf_fallback(pair, timeframe, count)

        if not df.empty:
            df = clean(df, pair, timeframe)
            cache_put(pair, timeframe, count, df)

        return df

    def _fetch_mt5_candles(self, pair: str, timeframe: str, count: int) -> pd.DataFrame:
        """Raw MT5 candle fetch — no cleaning or caching."""
        if not self.ensure_connected():
            return pd.DataFrame()

        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(pair, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No candles returned", pair=pair, tf=timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def _fetch_yf_fallback(self, pair: str, timeframe: str, count: int) -> pd.DataFrame:
        """Yahoo Finance fallback when MT5 is unavailable. FX/indices only."""
        try:
            import yfinance as yf

            # Map ARIA pair names → Yahoo Finance tickers
            _YF_MAP = {
                "EURUSD": "EURUSD=X", "EURUSDm": "EURUSD=X",
                "GBPUSD": "GBPUSD=X", "GBPUSDm": "GBPUSD=X",
                "USDJPY": "USDJPY=X", "USDJPYm": "USDJPY=X",
                "AUDUSD": "AUDUSD=X", "AUDUSDm": "AUDUSD=X",
                "XAUUSD": "GC=F",     "XAUUSDm": "GC=F",
                "USTEC":  "NQ=F",     "USTECm":  "NQ=F",
                "GBPJPY": "GBPJPY=X", "GBPJPYm": "GBPJPY=X",
            }
            ticker = _YF_MAP.get(pair)
            if not ticker:
                return pd.DataFrame()

            _TF_MAP = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "1h", "D1": "1d"}
            yf_interval = _TF_MAP.get(timeframe.upper(), "15m")
            period = "5d" if timeframe in ("M5", "M15", "M30") else "60d"

            df = yf.download(ticker, period=period, interval=yf_interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close", "volume"]].tail(count)
            logger.info(f"[YF Fallback] {pair} {timeframe}: {len(df)} bars from Yahoo Finance")
            return df

        except Exception as e:
            logger.debug(f"[YF Fallback] Failed for {pair}: {e}")
            return pd.DataFrame()

    def get_tick(self, pair: str) -> dict:
        """Get current bid/ask/spread for a pair."""
        if not self.ensure_connected():
            return {}

        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            return {}

        symbol_info = mt5.symbol_info(pair)
        digits = symbol_info.digits if symbol_info else 5
        pip_size = 0.01 if "JPY" in pair else 0.0001
        spread_pips = (tick.ask - tick.bid) / pip_size

        return {
            "pair": pair,
            "bid": round(tick.bid, digits),
            "ask": round(tick.ask, digits),
            "mid": round((tick.bid + tick.ask) / 2, digits),
            "spread_pips": round(spread_pips, 1),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    def get_account_info(self) -> dict:
        """Get current account balance, equity, margin."""
        if not self.ensure_connected():
            return {}

        info = mt5.account_info()
        if info is None:
            return {}

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self.ensure_connected():
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "pair": p.symbol,
                "direction": "long" if p.type == mt5.ORDER_TYPE_BUY else "short",
                "lots": p.volume,
                "entry": p.price_open,
                "current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "pnl": p.profit,
                "swap": p.swap,
                "opened_at": datetime.fromtimestamp(p.time, tz=timezone.utc),
                "comment": p.comment,
            })
        return result

    def get_pending_orders(self) -> list[dict]:
        """Get pending limit/stop orders."""
        if not self.ensure_connected():
            return []

        orders = mt5.orders_get()
        if orders is None:
            return []

        return [
            {
                "ticket": o.ticket,
                "pair": o.symbol,
                "type": o.type,
                "volume": o.volume_current,
                "price": o.price_open,
                "sl": o.sl,
                "tp": o.tp,
            }
            for o in orders
        ]

    def get_history_deals(self, days: int = 1) -> list[dict]:
        """Get closed deals from history."""
        if not self.ensure_connected():
            return []

        from datetime import timedelta
        date_from = datetime.now(timezone.utc) - timedelta(days=days)
        deals = mt5.history_deals_get(date_from, datetime.now(timezone.utc))
        if deals is None:
            return []

        return [
            {
                "ticket": d.ticket,
                "order": d.order,
                "position_id": d.position_id,   # matches the original position ticket
                "pair": d.symbol,
                "direction": "long" if d.type == mt5.DEAL_TYPE_BUY else "short",
                "lots": d.volume,
                "entry": d.price,
                "profit": d.profit,
                "commission": d.commission,
                "swap": d.swap,
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc),
                "comment": d.comment,
            }
            for d in deals
            if d.symbol  # skip non-trade deals (deposits etc.)
        ]


# Singleton
feed = MT5Feed()
