"""
ARIA — Autonomous Research & Intelligent Allocation

Entry point. Starts the scanner, trade lifecycle, and dashboard.

Usage:
  python main.py                    # full bot: scanner + dashboard (DRY RUN)
  python main.py --live             # enable live execution on MT5
  python main.py --dash-only        # dashboard only, no scanner
  python main.py --scan-now         # single scan cycle and exit
  python main.py --presession       # run pre-session analysis now and exit

Cost model (auto mode, NO --live):
  Signals:        $0   (rules-based)
  Pre-session:    ~$0.002/day (Haiku)
  Daily report:   ~$0.05/day  (Sonnet)
  Total:          ~$1.60/month
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from loguru import logger

from config.settings import settings
from core.capital import CapitalManager
from data.mt5_feed import feed
from db.session import init_db
from knowledge.obsidian import ObsidianWriter
from scheduler.tasks import setup_scheduler
from signals.scanner import ScannerLoop


def _reconcile_positions(capital: CapitalManager) -> None:
    """
    On startup, sync any MT5 positions that were opened outside ARIA
    (e.g., manually or from a previous session) into capital.open_positions.
    Prevents lifecycle manager from treating them as auto-closes immediately.
    """
    import MetaTrader5 as mt5
    positions = mt5.positions_get()
    if not positions:
        return
    aria_magic = 20260707
    reconciled = 0
    for pos in positions:
        pair = pos.symbol
        if pair in capital.open_positions:
            continue  # already tracked
        direction = "long" if pos.type == mt5.ORDER_TYPE_BUY else "short"
        # Grant a synthetic auth token so register_open() doesn't flag a bypass
        import time as _time
        capital._auth_grants[pair] = _time.monotonic()
        capital.register_open(
            ticket=pos.ticket,
            pair=pair,
            direction=direction,
            lots=pos.volume,
            entry=pos.price_open,
            sl=pos.sl,
            tp1=pos.tp,
            tp2=pos.tp,
            tp3=pos.tp,
            score=0.0,
            strategy="RECONCILED",
        )
        reconciled += 1
        source = "ARIA" if pos.magic == aria_magic else "EXTERNAL"
        logger.info(f"Reconciled {source} position: {direction.upper()} {pair} lots={pos.volume} entry={pos.price_open}")
    if reconciled:
        logger.info(f"Position reconciliation: {reconciled} position(s) synced to capital manager")


def _validate_env() -> None:
    """Fail fast on startup if required environment variables are missing."""
    required = {
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        for k in missing:
            logger.critical(f"Missing required environment variable: {k}")
        sys.exit(1)

    # Warn on optional but recommended keys
    optional = {
        "TELEGRAM_BOT_TOKEN": settings.telegram_bot_token,
        "TELEGRAM_CHAT_ID": settings.telegram_chat_id,
        "MT5_LOGIN": str(settings.mt5_login) if settings.mt5_login else None,
    }
    for k, v in optional.items():
        if not v:
            logger.warning(f"Optional env var not set: {k} — related features disabled")


def _startup_checklist(capital: CapitalManager) -> None:
    """Log a startup readiness summary — all checks non-blocking."""
    checks: list[tuple[str, bool, str]] = []

    # MT5
    try:
        mt5_ok = settings.mt5_enabled and feed.ensure_connected()
        account = feed.get_account_info() if mt5_ok else None
        bal_str = f"${account['balance']:,.2f}" if account else "N/A"
        checks.append(("MT5", mt5_ok, bal_str))
    except Exception as e:
        checks.append(("MT5", False, str(e)[:40]))

    # DB
    try:
        from db.session import get_session
        from db.models import Trade
        with get_session() as db:
            db.query(Trade).limit(1).all()
        checks.append(("Database", True, "SQLite OK"))
    except Exception as e:
        checks.append(("Database", False, str(e)[:40]))

    # Anthropic API key
    api_ok = bool(settings.anthropic_api_key)
    checks.append(("Anthropic API", api_ok, "key present" if api_ok else "MISSING"))

    # Telegram
    tg_ok = bool(settings.telegram_bot_token and settings.telegram_chat_id)
    checks.append(("Telegram", tg_ok, "configured" if tg_ok else "not configured"))

    # Obsidian vault
    vault_ok = settings.obsidian_vault_path.exists()
    checks.append(("Obsidian vault", vault_ok, str(settings.obsidian_vault_path) if vault_ok else "NOT FOUND"))

    # Capital state
    bal = capital.balance
    checks.append(("Capital", bal > 0, f"${bal:,.2f}"))

    # Print checklist
    logger.info("── ARIA Startup Checklist ──────────────────────────")
    for name, ok, detail in checks:
        icon = "✓" if ok else "✗"
        level = "info" if ok else "warning"
        getattr(logger, level)(f"  [{icon}] {name:<18} {detail}")
    logger.info("────────────────────────────────────────────────────")


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/aria_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        compression="gz",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA Trading System")
    parser.add_argument("--live", action="store_true",
                        help="Enable live MT5 execution (default: DRY RUN)")
    parser.add_argument("--dash-only", action="store_true",
                        help="Start dashboard only, no scanner")
    parser.add_argument("--scan-now", action="store_true",
                        help="Run one scan cycle and exit")
    parser.add_argument("--presession", action="store_true",
                        help="Run pre-session analysis now and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug mode")
    args = parser.parse_args()

    _configure_logging()
    _validate_env()

    if not args.live:
        # Override setting — always default to dry run for safety
        settings.dry_run = True
        logger.warning("DRY RUN mode — no real trades. Use --live to execute on MT5.")

    logger.info("ARIA starting up")
    logger.info(f"Vault: {settings.obsidian_vault_path}")
    logger.info(f"DB: {settings.db_path}")
    logger.info(f"Pairs: {', '.join(settings.pairs)}")

    # ── Init ──────────────────────────────────────────────────────
    settings.ensure_dirs()
    init_db()

    capital = CapitalManager()

    # Sync balance from MT5 if connected
    if settings.mt5_enabled:
        connected = feed.connect()
        if connected:
            account = feed.get_account_info()
            if account:
                capital.sync_balance(account["balance"], account["equity"])
                capital.day.starting_balance = account["balance"]
                logger.info(f"MT5 balance synced: ${account['balance']:,.2f}")
        else:
            logger.warning("MT5 not connected — using configured balance")

    # ── Position reconciliation — sync any MT5 positions opened outside ARIA ──
    if settings.mt5_enabled and feed.ensure_connected():
        _reconcile_positions(capital)

    # Startup checklist — log readiness of all subsystems
    _startup_checklist(capital)

    # Ensure Obsidian overview note exists
    ObsidianWriter().ensure_overview()

    # Seed A/B tests if none exist yet (one test per strategy pair, cross-pair)
    try:
        from core.ab_testing import get_all_tests, create_test
        if not get_all_tests():
            ref_pair = settings.pairs[0] if settings.pairs else "EURUSDm"
            _ab_pairs = [
                ("SMC_TREND",       "SESSION_BREAKOUT", ref_pair),
                ("MEAN_REVERSION",  "RANGE_TRADING",    ref_pair),
            ]
            for a, b, p in _ab_pairs:
                tid = create_test(a, b, p)
                logger.info(f"A/B test created: {a} vs {b} on {p} → {tid}")
    except Exception as _ab_err:
        logger.debug(f"A/B test seeding failed: {_ab_err}")

    # ── One-shot modes ────────────────────────────────────────────
    if args.scan_now:
        from signals.scanner import scan_all
        signals = scan_all()
        logger.info(f"Scan complete: {len(signals)} signals above threshold")
        for pair, sig in signals.items():
            logger.info(f"  {sig.label()} {pair} — {sig.entry_reason[:60]}")
        feed.disconnect()
        return

    if args.presession:
        from ai.session_analyst import SessionAnalyst
        analyst = SessionAnalyst(capital)
        analysis = analyst.run_presession()
        logger.info("Pre-session analysis written to vault")
        print(analysis)
        feed.disconnect()
        return

    # ── Dashboard-only mode ───────────────────────────────────────
    if args.dash_only:
        from dashboard.app import run_dashboard, set_capital, set_order_manager
        from execution.order_manager import OrderManager
        set_capital(capital)
        set_order_manager(OrderManager(capital), live=args.live)
        run_dashboard(debug=args.debug)
        return

    # ── Full mode: scanner + dashboard ───────────────────────────
    logger.info("Starting full ARIA system")

    # APScheduler (runs scanner every 5 min + lifecycle every 60s + AI tasks)
    scheduler = setup_scheduler(capital, auto_execute=args.live)
    scheduler.start()
    logger.info("Scheduler started")

    # Dashboard in background thread
    from dashboard.app import run_dashboard, set_capital, set_order_manager
    from execution.order_manager import OrderManager
    order_mgr = OrderManager(capital)
    set_capital(capital)
    set_order_manager(order_mgr, live=args.live)
    dash_thread = threading.Thread(
        target=run_dashboard,
        kwargs={"debug": args.debug},
        daemon=True,
        name="dashboard",
    )
    dash_thread.start()
    logger.info(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")

    # Run initial scan immediately
    from signals.scanner import scan_all
    scan_all()

    # Telegram startup notification (live mode only — dry run is silent)
    if args.live:
        try:
            from notifications.telegram import send
            from core.session import SessionManager
            session = SessionManager().current_session().value
            send(
                f"<b>🟢 ARIA — LIVE</b>\n"
                f"Balance: <code>${capital.balance:,.2f}</code>\n"
                f"Session: {session} | Pairs: {len(settings.pairs)}\n"
                f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}"
            )
        except Exception:
            pass

    logger.info("ARIA running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Shutting down ARIA — KeyboardInterrupt")
    except Exception as e:
        logger.critical(f"ARIA crashed: {e}", exc_info=True)
        try:
            from notifications.telegram import send
            send(
                f"<b>💥 ARIA — CRASHED</b>\n"
                f"Error: {type(e).__name__}: {str(e)[:200]}\n"
                f"<i>Bot is DOWN — manual restart required.</i>"
            )
        except Exception:
            pass
        raise
    finally:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        feed.disconnect()
        logger.info("Goodbye")


if __name__ == "__main__":
    main()
