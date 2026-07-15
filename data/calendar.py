"""
Economic calendar — scrapes ForexFactory for high-impact events.
No API key needed.

Used to avoid trading during high-impact news windows.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

# Currency to pairs mapping — which pairs to block for each currency
def _pairs_for_currency(currency: str, all_pairs: list[str]) -> list[str]:
    """Dynamically match pairs from watchlist that involve a given currency."""
    return [p for p in all_pairs if currency in p.upper()]


# Fallback static map if watchlist isn't loaded yet — uses substring matching in blocks_pair()
_CURRENCY_BASE_MAP: dict[str, list[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "USTEC", "NAS100"],
    "EUR": ["EURUSD"],
    "GBP": ["GBPUSD", "GBPJPY"],
    "JPY": ["USDJPY", "GBPJPY", "EURJPY"],
    "AUD": ["AUDUSD"],
    "NZD": ["NZDUSD"],
    "CAD": ["USDCAD"],
    "CHF": ["USDCHF"],
    "XAU": ["XAUUSD"],
}


class NewsEvent:
    def __init__(self, currency: str, title: str, impact: str, dt: datetime) -> None:
        self.currency = currency
        self.title = title
        self.impact = impact  # "High" | "Medium" | "Low"
        self.dt = dt

    def blocks_pair(self, pair: str, buffer_minutes: int = 20) -> bool:
        # Check if this event's currency appears in the pair name (handles 'm' suffix etc.)
        if self.currency not in pair.upper():
            return False
        now = datetime.now(timezone.utc)
        window_start = self.dt - timedelta(minutes=buffer_minutes)
        window_end = self.dt + timedelta(minutes=buffer_minutes)
        return window_start <= now <= window_end

    def __repr__(self) -> str:
        return f"NewsEvent({self.currency} {self.impact} '{self.title}' @ {self.dt:%H:%M})"


class EconomicCalendar:
    """Scrape and cache ForexFactory economic calendar."""

    def __init__(self) -> None:
        self._events: list[NewsEvent] = []
        self._last_fetch: datetime | None = None
        self._fetch_interval = timedelta(hours=12)
        self._failed_until: datetime | None = None  # backoff after failure

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _scrape(self) -> list[NewsEvent]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        url = "https://www.forexfactory.com/calendar"
        with httpx.Client(headers=headers, timeout=15, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        events: list[NewsEvent] = []
        today = datetime.now(timezone.utc).date()
        current_date = today

        rows = soup.select("tr.calendar__row")
        for row in rows:
            # Date cell can update current_date
            date_cell = row.select_one("td.calendar__date")
            if date_cell and date_cell.get_text(strip=True):
                text = date_cell.get_text(strip=True)
                try:
                    parsed = datetime.strptime(text, "%a%b %d")
                    current_date = parsed.replace(year=today.year).date()
                except ValueError:
                    pass

            # Time
            time_cell = row.select_one("td.calendar__time")
            if not time_cell:
                continue
            time_text = time_cell.get_text(strip=True)
            if not time_text or time_text in ("All Day", "Tentative"):
                continue
            try:
                t = datetime.strptime(time_text, "%I:%M%p")
                event_dt = datetime(
                    current_date.year, current_date.month, current_date.day,
                    t.hour, t.minute, tzinfo=timezone.utc,
                )
            except ValueError:
                continue

            # Currency
            currency_cell = row.select_one("td.calendar__currency")
            if not currency_cell:
                continue
            currency = currency_cell.get_text(strip=True)

            # Impact
            impact_cell = row.select_one("td.calendar__impact")
            impact = "Low"
            if impact_cell:
                icon = impact_cell.select_one("span")
                if icon:
                    classes = " ".join(icon.get("class", []))
                    if "red" in classes or "high" in classes:
                        impact = "High"
                    elif "orange" in classes or "medium" in classes:
                        impact = "Medium"

            # Title
            title_cell = row.select_one("td.calendar__event")
            title = title_cell.get_text(strip=True) if title_cell else ""

            events.append(NewsEvent(currency, title, impact, event_dt))

        logger.info(f"Calendar: fetched {len(events)} events")
        return events

    def fetch(self, force: bool = False) -> None:
        """Refresh calendar (cached 12 hours; 2-hour backoff on failure)."""
        now = datetime.now(timezone.utc)
        # Skip if cache still fresh
        if not force and self._last_fetch and now - self._last_fetch < self._fetch_interval:
            return
        # Skip if within failure backoff window
        if not force and self._failed_until and now < self._failed_until:
            return
        try:
            self._events = self._scrape()
            self._last_fetch = now
            self._failed_until = None
        except Exception as e:
            logger.warning(f"Calendar unavailable — news filter disabled for 2h: {e}")
            self._failed_until = now + timedelta(hours=2)

    def get_upcoming(self, hours: int = 8, impact: str = "High") -> list[NewsEvent]:
        """Get high-impact events in the next N hours."""
        self.fetch()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        levels = {"High": 3, "Medium": 2, "Low": 1}
        min_level = levels.get(impact, 3)
        return [
            e for e in self._events
            if now <= e.dt <= cutoff and levels.get(e.impact, 1) >= min_level
        ]

    def is_blocked(self, pair: str) -> tuple[bool, str]:
        """Check if pair is within a news blackout window."""
        if not settings.news_filter_enabled:
            return False, ""
        self.fetch()
        buffer = settings.news_buffer_minutes
        for event in self._events:
            if event.blocks_pair(pair, buffer):
                return True, f"{event.currency} {event.impact}: {event.title} @ {event.dt:%H:%M} UTC"
        return False, ""


# Singleton
calendar = EconomicCalendar()
