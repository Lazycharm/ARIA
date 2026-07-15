"""
APScheduler task definitions for ARIA.

Schedule (UTC):
  Every 5 min  — signal scanner (active sessions only, rules-based, free)
  Every 1 min  — trade lifecycle manager (SL moves, partials)
  06:30 daily  — pre-session AI analysis (Haiku)
  17:30 daily  — end-of-day report (Sonnet)
  00:01 daily  — reset day stats + vault index refresh

AI cost total: ~$1.60/month
"""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from config.settings import settings
from core.capital import CapitalManager
from core.session import SessionManager

_session_mgr = SessionManager()


def setup_scheduler(
    capital: CapitalManager,
    auto_execute: bool = False,
) -> BackgroundScheduler:
    """
    Build and return the configured APScheduler.

    auto_execute: if True, qualifying signals trigger MT5 execution.
                  False = analysis + dashboard only (safe default).
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    # ── Signal scanner — every 5 minutes ─────────────────────────
    scheduler.add_job(
        _scan_job,
        trigger=IntervalTrigger(seconds=300),
        id="scanner",
        kwargs={"capital": capital, "auto_execute": auto_execute},
        max_instances=1,
        misfire_grace_time=60,
    )

    # ── Trade lifecycle — every 60 seconds ────────────────────────
    scheduler.add_job(
        _lifecycle_job,
        trigger=IntervalTrigger(seconds=60),
        id="lifecycle",
        kwargs={"capital": capital},
        max_instances=1,
        misfire_grace_time=30,
    )

    # ── Pre-session analysis — 06:30 UTC (Haiku) ─────────────────
    scheduler.add_job(
        _presession_job,
        trigger=CronTrigger(hour=6, minute=30),
        id="presession",
        kwargs={"capital": capital},
        max_instances=1,
    )

    # ── Daily report — 17:30 UTC (Sonnet) ────────────────────────
    scheduler.add_job(
        _daily_report_job,
        trigger=CronTrigger(hour=17, minute=30),
        id="daily_report",
        kwargs={"capital": capital},
        max_instances=1,
    )

    # ── Day reset — 00:01 UTC ─────────────────────────────────────
    scheduler.add_job(
        _reset_job,
        trigger=CronTrigger(hour=0, minute=1),
        id="day_reset",
        kwargs={"capital": capital},
        max_instances=1,
    )

    # ── Weekly learning report — Sunday 00:01 UTC ─────────────────
    scheduler.add_job(
        _weekly_report_job,
        trigger=CronTrigger(day_of_week="sun", hour=0, minute=1),
        id="weekly_report",
        max_instances=1,
    )

    # ── Monthly learning report — 1st of month 00:05 UTC ─────────
    scheduler.add_job(
        _monthly_report_job,
        trigger=CronTrigger(day=1, hour=0, minute=5),
        id="monthly_report",
        max_instances=1,
    )

    # ── Calendar refresh — every 4 hours ─────────────────────────
    scheduler.add_job(
        _calendar_refresh_job,
        trigger=IntervalTrigger(hours=4),
        id="calendar",
        max_instances=1,
    )

    # ── Reddit sentiment — every 2 hours ─────────────────────────
    scheduler.add_job(
        _sentiment_refresh_job,
        trigger=IntervalTrigger(hours=2),
        id="sentiment",
        max_instances=1,
    )

    # ── Experiment generator — Sunday 01:00 UTC (after weekly report) ──
    scheduler.add_job(
        _experiment_generator_job,
        trigger=CronTrigger(day_of_week="sun", hour=1, minute=0),
        id="experiment_generator",
        max_instances=1,
    )

    # ── DB backup — 03:00 UTC daily ───────────────────────────────
    scheduler.add_job(
        _db_backup_job,
        trigger=CronTrigger(hour=3, minute=0),
        id="db_backup",
        max_instances=1,
    )

    # ── Chief Research Agent — Sunday 02:00 UTC (after experiment gen) ──
    scheduler.add_job(
        _research_cadence_job,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="research_cadence",
        max_instances=1,
    )

    # ── Autonomous pipeline step — every 6 hours ──────────────────
    scheduler.add_job(
        _pipeline_step_job,
        trigger=IntervalTrigger(hours=6),
        id="pipeline_step",
        max_instances=1,
        misfire_grace_time=300,
    )

    # ── Paper trade update — every 5 minutes ─────────────────────
    scheduler.add_job(
        _paper_trade_update_job,
        trigger=IntervalTrigger(seconds=300),
        id="paper_trade_update",
        max_instances=1,
    )

    # ── Strategy monitoring — every 12 hours ──────────────────────
    scheduler.add_job(
        _strategy_monitor_job,
        trigger=IntervalTrigger(hours=12),
        id="strategy_monitor",
        max_instances=1,
    )

    return scheduler


# ── Task implementations ──────────────────────────────────────────────────────

def _scan_job(capital: CapitalManager, auto_execute: bool = False) -> None:
    try:
        from signals.scanner import scan_all
        from signals.entry import build_setup
        from execution.order_manager import OrderManager
        from data.mt5_feed import feed
        from core.adaptive_learning import adaptive

        signals = scan_all()

        if not auto_execute or not signals:
            return

        order_mgr = OrderManager(capital)
        for pair, sig in signals.items():
            # Use pair-specific adaptive threshold (not a global hard limit)
            pair_min = adaptive.get_min_score(pair)
            if sig.score < pair_min or sig.direction == "wait":
                logger.debug(f"Auto-exec skip: {pair} score={sig.score:.0f} < threshold={pair_min:.0f}")
                continue

            # Build trade setup — apply indicators so ATR is populated
            from analysis.indicators import apply_all
            df_m15 = feed.get_candles(pair, "M15", count=50)
            df_m15 = apply_all(df_m15)
            tick   = feed.get_tick(pair)
            price  = tick.get("mid", 0) if tick else 0
            if not price:
                logger.warning(f"Auto-exec: no tick for {pair}")
                continue
            setup = build_setup(sig, price, df_m15)
            if setup:
                logger.info(f"Auto-executing: {sig.direction.upper()} {pair} score={sig.score:.0f} (threshold={pair_min:.0f})")
                order_mgr.execute(sig, setup)
            else:
                logger.warning(f"Auto-exec: build_setup returned None for {pair}")

    except Exception as e:
        logger.error(f"Scanner job error: {e}")


def _lifecycle_job(capital: CapitalManager) -> None:
    try:
        from execution.trade_lifecycle import TradeLifecycle
        lc = TradeLifecycle(capital)
        lc.tick()
    except Exception as e:
        logger.error(f"Lifecycle job error: {e}")


def _presession_job(capital: CapitalManager) -> None:
    try:
        from ai.session_analyst import SessionAnalyst
        analyst = SessionAnalyst(capital)
        analyst.run_presession()
    except Exception as e:
        logger.error(f"Pre-session job error: {e}")


def _daily_report_job(capital: CapitalManager) -> None:
    try:
        from ai.session_analyst import SessionAnalyst
        from db.session import get_session
        from db.models import Trade

        analyst = SessionAnalyst(capital)

        # Fetch today's trades from DB
        with get_session() as db:
            from datetime import date
            today = date.today().isoformat()
            trades = db.query(Trade).filter(Trade.opened_at >= today).all()
            trade_log = [
                {
                    "pair": t.pair, "direction": t.direction,
                    "lots": t.lots, "pnl": t.pnl or 0,
                    "reason": t.reason[:60],
                }
                for t in trades
            ]

        analyst.run_daily_report(trade_log)

        # Telegram daily summary
        try:
            from notifications.telegram import alert_daily_summary
            from data.mt5_feed import feed
            account = feed.get_account_info()
            cap = capital.status_dict
            alert_daily_summary(
                day_pnl=cap.get("realized_pnl", 0),
                trades=cap.get("trades_taken", 0),
                win_rate=cap.get("win_rate", 0),
                balance=account.get("balance", cap.get("balance", 0)) if account else 0,
            )
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Daily report job error: {e}")


def _reset_job(capital: CapitalManager) -> None:
    try:
        from datetime import date, datetime, timezone
        yesterday = (date.today().isoformat())   # reset runs at 00:01, so "today" is the new day
        # Snapshot yesterday's stats to DayLog before resetting
        try:
            from db.session import get_session
            from db.models import DayLog
            from sqlalchemy import select
            day_pnl    = capital.day.realized_pnl
            wins       = capital.day.wins
            losses     = capital.day.losses
            taken      = capital.day.trades_taken
            pf         = capital.day.profit_factor
            bal        = capital.balance
            with get_session() as db:
                existing = db.execute(select(DayLog).where(DayLog.date == yesterday)).scalar_one_or_none()
                if existing:
                    existing.ending_balance = bal
                    existing.realized_pnl   = day_pnl
                    existing.trades_taken   = taken
                    existing.trades_won     = wins
                    existing.trades_lost    = losses
                    existing.profit_factor  = pf
                else:
                    db.add(DayLog(
                        date=yesterday,
                        starting_balance=bal - day_pnl,
                        ending_balance=bal,
                        realized_pnl=day_pnl,
                        trades_taken=taken,
                        trades_won=wins,
                        trades_lost=losses,
                        profit_factor=pf,
                    ))
            logger.info(f"DayLog saved: {yesterday} pnl={day_pnl:+.2f} trades={taken}")
        except Exception as _dl_err:
            logger.debug(f"DayLog write failed: {_dl_err}")

        capital.reset_day()
        from knowledge.obsidian import ObsidianWriter
        ObsidianWriter().ensure_overview()
        logger.info("Day reset complete")
    except Exception as e:
        logger.error(f"Day reset error: {e}")


def _weekly_report_job() -> None:
    try:
        from datetime import datetime, timezone
        from knowledge.weekly_report import build_report
        from knowledge.obsidian import ObsidianWriter

        now = datetime.now(timezone.utc)
        iso = now.isocalendar()
        week_label = f"{iso[0]}-W{iso[1]:02d}"
        content = build_report(week_label, now)
        ObsidianWriter().write_weekly_report(content, now)
        logger.info(f"Weekly learning report written: {week_label}")
    except Exception as e:
        logger.error(f"Weekly report job error: {e}")


def _monthly_report_job() -> None:
    try:
        from datetime import datetime, timezone
        from knowledge.monthly_report import write_monthly_report
        now = datetime.now(timezone.utc)
        # Report covers the previous month
        first_of_month = now.replace(day=1)
        from datetime import timedelta
        prev_month_end = first_of_month - timedelta(days=1)
        write_monthly_report(prev_month_end.year, prev_month_end.month)
    except Exception as e:
        logger.error(f"Monthly report job error: {e}")


def _calendar_refresh_job() -> None:
    try:
        from data.calendar import calendar
        calendar.fetch(force=True)
        logger.debug("Calendar refreshed")
    except Exception as e:
        logger.error(f"Calendar refresh error: {e}")


def _sentiment_refresh_job() -> None:
    try:
        from data.sentiment import sentiment_cache
        from config.pairs_config import get_pairs
        pairs = get_pairs()
        sentiment_cache.refresh(pairs)
        logger.info(f"[Sentiment] Refreshed for {len(pairs)} pairs")
    except Exception as e:
        logger.error(f"Sentiment refresh error: {e}")


def _experiment_generator_job() -> None:
    try:
        from core.experiment_generator import generate_next_experiments
        ids = generate_next_experiments(max_hypotheses=3)
        if ids:
            logger.info(f"Experiment generator: {len(ids)} new hypotheses queued")
            try:
                from notifications.telegram import alert_hypothesis_generated
                from core.hypothesis_queue import _load_all
                rows = _load_all()
                latest = next((r for r in reversed(rows) if r["id"] == ids[0]), None)
                if latest:
                    alert_hypothesis_generated(
                        pair=latest["pair"],
                        hypothesis_id=latest["id"],
                        title=latest["title"],
                        source=latest["source"],
                    )
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Experiment generator job error: {e}")


def _research_cadence_job() -> None:
    """Weekly research cycle: scrape → dedup → AI enrich → queue hypotheses."""
    try:
        from research.chief_agent import run_research_cycle
        ids = run_research_cycle(max_per_source=5, ai_enrich=True)
        logger.info(f"[Research] Weekly cycle complete: {len(ids)} new hypotheses")
    except Exception as e:
        logger.error(f"Research cadence job error: {e}")


def _pipeline_step_job() -> None:
    """Run one step of the autonomous pipeline (backtest → WFO → MC → paper)."""
    try:
        from core.autonomous_pipeline import run_pipeline_step
        from core.pipeline_log import record as log_pipeline
        result = run_pipeline_step()
        log_pipeline(result)
        hyp = result.get("hypothesis_processed")
        if hyp:
            logger.info(f"[Pipeline] Step: {hyp} approved={result.get('approved')} reason={result.get('reason','')[:60]}")
    except Exception as e:
        logger.error(f"Pipeline step job error: {e}")


def _paper_trade_update_job() -> None:
    """Update open paper trades against live prices."""
    try:
        from core.paper_trader import update_paper_trades, get_open_trades
        open_trades = get_open_trades()
        if not open_trades:
            return
        from data.mt5_feed import feed
        from config.pairs_config import get_pairs
        prices = {}
        for pair in get_pairs():
            tick = feed.get_tick(pair)
            if tick:
                bid = tick.get("bid", 0)
                ask = tick.get("ask", 0)
                if bid and ask:
                    prices[pair] = (bid + ask) / 2
        if prices:
            update_paper_trades(prices)
    except Exception as e:
        logger.debug(f"Paper trade update error: {e}")


def _strategy_monitor_job() -> None:
    """Monitor live strategies + auto-retire underperformers."""
    try:
        from core.autonomous_pipeline import monitor_live_strategies, auto_retire_strategies
        monitor_live_strategies()
        auto_retire_strategies()
    except Exception as e:
        logger.debug(f"Strategy monitor job error: {e}")


def _db_backup_job() -> None:
    """Daily DB backup — copies SQLite db to db/backups/aria-YYYY-MM-DD.db.gz"""
    try:
        import gzip
        import shutil
        from datetime import datetime, timezone
        from pathlib import Path
        from config.settings import settings

        db_path = settings.db_path
        if not db_path.exists():
            return

        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = backup_dir / f"aria-{date_str}.db.gz"

        with open(db_path, "rb") as f_in:
            with gzip.open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Keep 30 days of backups
        backups = sorted(backup_dir.glob("aria-*.db.gz"))
        for old in backups[:-30]:
            old.unlink(missing_ok=True)

        logger.info(f"DB backup: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
    except Exception as e:
        logger.error(f"DB backup job error: {e}")
