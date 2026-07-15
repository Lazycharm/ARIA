"""
Signal filters — pre-trade checks that gate execution.

All filters are rules-based (no AI cost).
"""

from __future__ import annotations

from data.calendar import calendar
from config.settings import settings
from core.session import SessionManager

session_mgr = SessionManager()


class SignalFilter:
    """Collection of pre-trade filter checks."""

    def news_blocked(self, pair: str) -> tuple[bool, str]:
        """Check ForexFactory economic calendar news blackout."""
        if not settings.news_filter_enabled:
            return False, ""
        return calendar.is_blocked(pair)

    def spread_ok(self, spread_pips: float) -> tuple[bool, str]:
        if spread_pips > settings.max_spread_pips:
            return False, f"Spread {spread_pips:.1f} > max {settings.max_spread_pips}"
        return True, ""

    def session_ok(self, pair: str) -> tuple[bool, str]:
        if not session_mgr.is_trading_allowed():
            info = session_mgr.session_info()
            return False, f"Session not active: {info['label']}"
        active = session_mgr.get_active_pairs(settings.pairs)
        if pair not in active:
            return False, f"{pair} not in session pair list"
        return True, ""

    def all_pass(self, pair: str, spread_pips: float) -> tuple[bool, list[str]]:
        """Run all filters. Returns (all_clear, list_of_failures)."""
        failures: list[str] = []

        ok, reason = self.session_ok(pair)
        if not ok:
            failures.append(f"SESSION: {reason}")

        ok, reason = self.spread_ok(spread_pips)
        if not ok:
            failures.append(f"SPREAD: {reason}")

        blocked, reason = self.news_blocked(pair)
        if blocked:
            failures.append(f"NEWS: {reason}")

        return len(failures) == 0, failures
