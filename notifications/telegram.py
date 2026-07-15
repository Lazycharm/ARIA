"""
Telegram notification channel for ARIA.

Sends real-time alerts for:
  - Trade opened (pair, direction, entry, SL, TP, score)
  - Trade closed (pair, P&L, win/loss, running day P&L)
  - Daily summary (end of day)
  - Capital halt triggered
  - Global conservative mode activated

Non-blocking: uses a background thread queue so notifications
never delay the trading pipeline.
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings

_q: queue.Queue[str] = queue.Queue(maxsize=50)
_started = False
_lock = threading.Lock()


def _worker() -> None:
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    while True:
        text = _q.get()
        if text is None:
            break
        try:
            httpx.post(
                url,
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=8,
            )
        except Exception as e:
            logger.debug(f"[Telegram] Send failed: {e}")
        finally:
            _q.task_done()


def _ensure_started() -> None:
    global _started
    with _lock:
        if not _started:
            t = threading.Thread(target=_worker, daemon=True, name="telegram")
            t.start()
            _started = True


def send(text: str) -> None:
    """Queue a message for delivery. Never blocks the caller."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    _ensure_started()
    try:
        _q.put_nowait(text)
    except queue.Full:
        logger.debug("[Telegram] Queue full — message dropped")


# ── Formatted alert helpers ────────────────────────────────────────────────────

def alert_trade_opened(
    pair: str,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    lots: float,
    score: float,
    multiplier: float = 1.0,
) -> None:
    arrow  = "🟢 BUY" if direction == "long" else "🔴 SELL"
    digits = 3 if ("JPY" in pair.upper() or "XAU" in pair.upper()) else 5
    send(
        f"<b>▲ ARIA — Trade Opened</b>\n"
        f"{arrow} <b>{pair.rstrip('m')}</b> @ {entry:.{digits}f}\n"
        f"SL: {sl:.{digits}f}   TP1: {tp1:.{digits}f}\n"
        f"Lots: {lots:.2f} (×{multiplier:.2f})   Score: {score:.0f}%\n"
        f"<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )


def alert_trade_closed(
    pair: str,
    direction: str,
    pnl: float,
    day_pnl: float,
    reason: str = "",
) -> None:
    icon  = "✅" if pnl >= 0 else "❌"
    arrow = "▲" if direction == "long" else "▼"
    send(
        f"<b>{icon} ARIA — Trade Closed</b>\n"
        f"{arrow} <b>{pair.rstrip('m')}</b>   P&L: <b>${pnl:+.2f}</b>\n"
        f"Day total: ${day_pnl:+.2f}\n"
        f"{f'Reason: {reason}' if reason else ''}"
    )


def alert_halt(reason: str) -> None:
    send(
        f"<b>⛔ ARIA — Trading HALTED</b>\n"
        f"{reason}\n"
        f"<i>Manual intervention required.</i>"
    )


def alert_conservative_mode(consecutive_losses: int) -> None:
    send(
        f"<b>⚠️ ARIA — Conservative Mode ON</b>\n"
        f"{consecutive_losses} consecutive losses detected.\n"
        f"All thresholds raised. Lot sizes reduced 50%.\n"
        f"Will auto-recover on next win."
    )


def alert_cooldown(consecutive_losses: int, until_utc: str) -> None:
    send(
        f"<b>⏸ ARIA — Cooldown Activated</b>\n"
        f"{consecutive_losses} consecutive losses within 60 min.\n"
        f"Trading paused until <b>{until_utc} UTC</b>.\n"
        f"<i>Auto-resumes — no action needed.</i>"
    )


def alert_emergency_close(
    n_closed: int,
    n_total: int,
    total_pnl: float,
    reason: str = "Manual emergency close",
) -> None:
    send(
        f"<b>🚨 ARIA — EMERGENCY CLOSE ALL</b>\n"
        f"Closed: {n_closed}/{n_total} positions\n"
        f"Total P&amp;L: <b>${total_pnl:+.2f}</b>\n"
        f"Reason: {reason}\n"
        f"<b>Trading HALTED — manual restart required.</b>\n"
        f"<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )


def alert_backtest_complete(
    pair: str,
    days: int,
    trades: int,
    win_rate: float,
    profit_factor: float,
    max_dd: float,
    verdict: str,
) -> None:
    icon = "✅" if verdict == "PASS" else "❌"
    send(
        f"<b>{icon} ARIA — Backtest Complete</b>\n"
        f"<b>{pair.rstrip('m')}</b> ({days}d)   Verdict: <b>{verdict}</b>\n"
        f"Trades: {trades}   WR: {win_rate:.1f}%   PF: {profit_factor:.2f}\n"
        f"Max DD: {max_dd:.1f}%"
    )


def alert_risk_halt(reason: str, day_dd: float) -> None:
    send(
        f"<b>🛑 ARIA — Risk Halt Triggered</b>\n"
        f"Day drawdown: <b>{day_dd:.2f}%</b>\n"
        f"Reason: {reason}\n"
        f"<i>Trading halted — resets at midnight UTC.</i>"
    )


def alert_ml_retrained(
    pair: str,
    accuracy: float,
    top_feature: str,
    n_samples: int,
    verdict: str,
) -> None:
    send(
        f"<b>🧠 ARIA — ML Model Retrained</b>\n"
        f"Pair: <b>{pair.rstrip('m')}</b>   Samples: {n_samples}\n"
        f"Accuracy: {accuracy:.1f}%   Top feature: {top_feature}\n"
        f"Verdict: {verdict}"
    )


def alert_hypothesis_generated(
    pair: str,
    hypothesis_id: str,
    title: str,
    source: str,
) -> None:
    send(
        f"<b>💡 ARIA — New Hypothesis</b>\n"
        f"ID: <code>{hypothesis_id}</code>   Pair: {pair.rstrip('m')}\n"
        f"<b>{title[:80]}</b>\n"
        f"Source: {source}"
    )


def alert_daily_summary(
    day_pnl: float,
    trades: int,
    win_rate: float,
    balance: float,
) -> None:
    icon = "📈" if day_pnl >= 0 else "📉"
    send(
        f"<b>{icon} ARIA — Daily Summary</b>\n"
        f"P&L: <b>${day_pnl:+.2f}</b>   Trades: {trades}   Win: {win_rate:.0f}%\n"
        f"Balance: ${balance:,.2f}\n"
        f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</i>"
    )
