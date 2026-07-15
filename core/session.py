"""
Session Manager — knows what market session we're in and what mode to run.

Sessions drive behavior:
  ASIAN      → setup mode, mark zones, low activity
  PRE_LONDON → AI deep analysis (Sonnet), level marking
  LONDON     → active trading, scan every 5 min
  LONDON_MID → position management only
  OVERLAP    → active trading, highest priority
  NY         → active trading, USD focus
  NY_CLOSE   → close day positions, generate report
  DEAD       → idle, update Obsidian vault
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from loguru import logger


class SessionMode(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    PRE_ANALYSIS = "pre_analysis"
    ACTIVE = "active"
    MANAGE = "manage"
    CLOSING = "closing"
    HALTED = "halted"


class Session(str, Enum):
    ASIAN = "asian"
    PRE_LONDON = "pre_london"
    LONDON = "london"
    LONDON_MID = "london_mid"
    OVERLAP = "london_ny_overlap"
    NY = "ny"
    NY_CLOSE = "ny_close"
    DEAD = "dead"


SESSION_CONFIG = {
    Session.ASIAN:      {"hours": (0, 7),   "mode": SessionMode.SETUP,        "active": True,  "label": "🌏 Asian"},
    Session.PRE_LONDON: {"hours": (6, 7),   "mode": SessionMode.PRE_ANALYSIS, "active": False, "label": "🔍 Pre-London Analysis"},
    Session.LONDON:     {"hours": (7, 10),  "mode": SessionMode.ACTIVE,       "active": True,  "label": "🇬🇧 London Open"},
    Session.LONDON_MID: {"hours": (10, 12), "mode": SessionMode.MANAGE,       "active": False, "label": "⏸ London Mid"},
    Session.OVERLAP:    {"hours": (12, 16), "mode": SessionMode.ACTIVE,       "active": True,  "label": "🔥 London-NY Overlap"},
    Session.NY:         {"hours": (16, 17), "mode": SessionMode.CLOSING,      "active": False, "label": "🇺🇸 NY Close"},
    Session.NY_CLOSE:   {"hours": (17, 18), "mode": SessionMode.CLOSING,      "active": False, "label": "📊 End of Day"},
    Session.DEAD:       {"hours": (18, 24), "mode": SessionMode.IDLE,         "active": False, "label": "💤 Dead Zone"},
}

# Base symbols without broker suffix — get_active_pairs() matches by substring
BEST_PAIRS_BY_SESSION_BASES: dict[Session, list[str]] = {
    Session.ASIAN:      ["USDJPY", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY"],
    Session.LONDON:     ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY"],
    Session.OVERLAP:    ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USTEC", "GBPJPY"],
    Session.NY:         ["XAUUSD", "USDJPY", "USTEC", "NAS100", "US30"],
    Session.DEAD:       [],
}
# Keep for backward compat
BEST_PAIRS_BY_SESSION = BEST_PAIRS_BY_SESSION_BASES


class SessionManager:
    """Determine current trading session and appropriate behavior mode."""

    def __init__(self) -> None:
        self._last_session: Session | None = None
        self._session_start_time: datetime | None = None

    def current_session(self, utc_now: datetime | None = None) -> Session:
        now = utc_now or datetime.now(timezone.utc)
        hour = now.hour

        # Pre-London override (30 min window)
        if hour == 6 and now.minute >= 30:
            return Session.PRE_LONDON

        for session, config in SESSION_CONFIG.items():
            if session == Session.PRE_LONDON:
                continue
            start, end = config["hours"]
            if start <= hour < end:
                # More specific sub-sessions
                if session == Session.LONDON and 12 <= hour < 16:
                    return Session.OVERLAP
                if session == Session.LONDON and 10 <= hour < 12:
                    return Session.LONDON_MID
                if session == Session.NY and hour >= 17:
                    return Session.NY_CLOSE
                return session

        return Session.DEAD

    def get_mode(self, utc_now: datetime | None = None) -> SessionMode:
        session = self.current_session(utc_now)
        return SESSION_CONFIG[session]["mode"]

    def is_trading_allowed(self, utc_now: datetime | None = None) -> bool:
        session = self.current_session(utc_now)
        return SESSION_CONFIG[session]["active"]

    def get_active_pairs(self, all_pairs: list[str], utc_now: datetime | None = None) -> list[str]:
        """Return pairs most suited for current session. Handles broker suffixes (e.g. 'm')."""
        session = self.current_session(utc_now)
        session_bases = BEST_PAIRS_BY_SESSION_BASES.get(session, [])
        if not session_bases:
            return []
        # Match watchlist pairs by checking if base symbol name is contained
        result = []
        for pair in all_pairs:
            pair_base = pair.upper().rstrip("M")  # strip trailing broker suffix
            if any(pair_base == base or pair_base == base.upper() for base in session_bases):
                result.append(pair)
        return result

    def session_info(self, utc_now: datetime | None = None) -> dict:
        now = utc_now or datetime.now(timezone.utc)
        session = self.current_session(now)
        config = SESSION_CONFIG[session]
        mode = config["mode"]
        active = config["active"]
        label = config["label"]

        start_h, end_h = config["hours"]
        minutes_remaining = max(0, (end_h - now.hour) * 60 - now.minute)
        next_session = self._next_active_session(session)

        return {
            "session": session.value,
            "label": label,
            "mode": mode.value,
            "active": active,
            "utc_time": now.strftime("%H:%M UTC"),
            "minutes_remaining": minutes_remaining,
            "next_active": next_session,
            "active_pairs": self.get_active_pairs([], now),
        }

    def _next_active_session(self, current: Session) -> str:
        sessions_order = [
            Session.ASIAN, Session.PRE_LONDON, Session.LONDON,
            Session.LONDON_MID, Session.OVERLAP, Session.NY,
            Session.NY_CLOSE, Session.DEAD,
        ]
        idx = sessions_order.index(current) if current in sessions_order else 0
        for i in range(1, len(sessions_order)):
            next_s = sessions_order[(idx + i) % len(sessions_order)]
            if SESSION_CONFIG[next_s]["active"]:
                return SESSION_CONFIG[next_s]["label"]
        return "Unknown"

    def should_run_presession_analysis(self, utc_now: datetime | None = None) -> bool:
        now = utc_now or datetime.now(timezone.utc)
        return self.current_session(now) == Session.PRE_LONDON

    def should_run_daily_report(self, utc_now: datetime | None = None) -> bool:
        now = utc_now or datetime.now(timezone.utc)
        return self.current_session(now) == Session.NY_CLOSE and now.minute < 10
