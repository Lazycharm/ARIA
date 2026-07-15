"""
Session analyst — runs once per day at pre-London (06:30 UTC).

Uses Haiku. Writes output to shared Obsidian vault.
Cost: ~$0.002/call × 30 days = $0.06/month
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

import core.brain as brain
from ai.prompts import PRE_SESSION_ANALYSIS, DAILY_REPORT
from config.settings import settings
from core.capital import CapitalManager
from core.session import SessionManager
from data.calendar import calendar
from data.mt5_feed import feed
from knowledge.obsidian import ObsidianWriter


session_mgr = SessionManager()
vault       = ObsidianWriter()


class SessionAnalyst:
    def __init__(self, capital: CapitalManager) -> None:
        self.capital = capital

    def run_presession(self) -> str:
        """Pre-session analysis (Haiku, 06:30 UTC). Returns analysis text."""
        now = datetime.now(timezone.utc)
        session = session_mgr.session_info()
        day_stats = self.capital.day

        # Get upcoming news events
        events = calendar.get_upcoming(hours=12, impact="High")
        news_text = "\n".join(
            f"  {e.dt:%H:%M} {e.currency} — {e.title}"
            for e in events
        ) or "No high-impact events scheduled"

        # Active pairs for upcoming session
        active_pairs = ", ".join(settings.pairs)

        prompt = PRE_SESSION_ANALYSIS.format(
            date=now.strftime("%Y-%m-%d"),
            session=session["label"],
            pairs=active_pairs,
            news_events=news_text,
            trades_taken=day_stats.trades_taken,
            pnl=f"{day_stats.realized_pnl:+.2f}",
            win_rate=f"{day_stats.win_rate:.0f}",
        )

        analysis = brain.session_analysis(prompt)
        logger.info(f"Pre-session analysis done ({len(analysis)} chars)")

        # Write to Obsidian
        vault.write_session_analysis(analysis, now)

        return analysis

    def run_daily_report(self, trade_log: list[dict]) -> str:
        """End-of-day report (Sonnet, once/day at NY close)."""
        now = datetime.now(timezone.utc)
        day = self.capital.day
        account = feed.get_account_info()
        ending_balance = account.get("balance", settings.account_balance)

        trade_lines = "\n".join(
            f"  {t.get('pair','?')} {t.get('direction','?')} "
            f"lots={t.get('lots',0)} pnl=${t.get('pnl',0):+.2f} "
            f"reason='{t.get('reason','')[:40]}'"
            for t in trade_log
        ) or "  No trades taken today"

        prompt = DAILY_REPORT.format(
            date=now.strftime("%Y-%m-%d"),
            session=session_mgr.session_info()["label"],
            starting_balance=day.starting_balance,
            ending_balance=ending_balance,
            net_pnl=day.realized_pnl,
            pnl_pct=day.realized_pnl / day.starting_balance * 100 if day.starting_balance else 0,
            trades_taken=day.trades_taken,
            max_trades=settings.max_trades_per_day,
            winners=day.trades_won,
            losers=day.trades_lost,
            win_rate=f"{day.win_rate:.0f}",
            profit_factor=f"{day.profit_factor:.2f}",
            trade_list=trade_lines,
            missed_signals="See dashboard signal history",
        )

        report = brain.daily_report(prompt)
        logger.info("Daily report generated")

        vault.write_daily_report(report, day, now)
        return report
