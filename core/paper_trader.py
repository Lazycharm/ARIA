"""
Paper Trading Layer — shadow execution with live prices, no real MT5 orders.

After the MC gate passes, strategies run in paper mode for 2 weeks.
Paper trades are tracked in db/paper_trades.jsonl:
  {id, pair, direction, entry, sl, tp1, tp2, lots, open_time, status,
   close_time, close_price, pnl, strategy, score}

Promotion to live: after 2 weeks with positive PnL + Sharpe > 0.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

_PAPER_PATH = Path("db/paper_trades.jsonl")
_LOCK       = threading.Lock()
_PROMOTE_DAYS = 14    # weeks of paper trading before live promotion
_PROMOTE_SHARPE = 0.0  # minimum Sharpe to promote


@dataclass
class PaperTrade:
    id:          str
    pair:        str
    direction:   str
    entry:       float
    sl:          float
    tp1:         float
    tp2:         float
    lots:        float
    open_time:   str
    status:      str    # open | closed_tp1 | closed_tp2 | closed_sl | closed_manual
    close_time:  Optional[str] = None
    close_price: Optional[float] = None
    pnl:         float = 0.0
    strategy:    str   = "SMC_TREND"
    score:       float = 0.0
    partial_taken: bool = False


def _load_all() -> list[dict]:
    if not _PAPER_PATH.exists():
        return []
    out = []
    for line in _PAPER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _append(trade: dict) -> None:
    _PAPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PAPER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trade) + "\n")


def _rewrite(trades: list[dict]) -> None:
    _PAPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PAPER_PATH.open("w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def open_paper_trade(
    pair: str,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    lots: float,
    strategy: str = "SMC_TREND",
    score: float  = 0.0,
) -> str:
    """Open a new paper trade. Returns trade ID."""
    tid = uuid.uuid4().hex[:8].upper()
    trade = PaperTrade(
        id=tid,
        pair=pair,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        lots=lots,
        open_time=datetime.now(timezone.utc).isoformat(),
        status="open",
        strategy=strategy,
        score=score,
    )
    with _LOCK:
        _append(asdict(trade))
    logger.info(f"[Paper] Opened {tid}: {pair} {direction} @ {entry:.5f}")
    return tid


def get_open_trades() -> list[dict]:
    with _LOCK:
        return [t for t in _load_all() if t.get("status") == "open"]


def get_all_trades() -> list[dict]:
    with _LOCK:
        return _load_all()


def update_paper_trades(live_prices: dict[str, float]) -> None:
    """
    Check open paper trades against live prices.
    Apply SL/TP hits, partial closes, breakeven moves.
    live_prices: {pair: mid_price}
    """
    with _LOCK:
        trades = _load_all()

    updated = False
    for t in trades:
        if t.get("status") != "open":
            continue

        pair  = t["pair"]
        price = live_prices.get(pair)
        if price is None:
            continue

        d     = t["direction"]
        sl    = t["sl"]
        tp1   = t["tp1"]
        tp2   = t["tp2"]
        entry = t["entry"]

        def pnl_usd(close_px: float) -> float:
            pip = 0.01 if "JPY" in pair.upper() else 0.0001
            pip_val = 10.0 * t.get("lots", 0.01)
            delta_pips = (close_px - entry) / pip if d == "long" else (entry - close_px) / pip
            return delta_pips * pip_val

        # TP1 partial close
        if not t.get("partial_taken"):
            hit_tp1 = (price >= tp1) if d == "long" else (price <= tp1)
            if hit_tp1:
                t["partial_taken"] = True
                t["pnl"] += pnl_usd(tp1) * 0.5
                t["sl"]   = entry   # move to breakeven
                updated   = True
                logger.info(f"[Paper] {t['id']} TP1 hit @ {tp1:.5f}, SL → breakeven")

        # TP2
        hit_tp2 = (price >= tp2) if d == "long" else (price <= tp2)
        if hit_tp2:
            remaining = 0.5 if t.get("partial_taken") else 1.0
            t["pnl"]         += pnl_usd(tp2) * remaining
            t["status"]       = "closed_tp2"
            t["close_price"]  = tp2
            t["close_time"]   = datetime.now(timezone.utc).isoformat()
            updated = True
            logger.info(f"[Paper] {t['id']} TP2 @ {tp2:.5f} PnL=${t['pnl']:.2f}")
            continue

        # SL
        hit_sl = (price <= sl) if d == "long" else (price >= sl)
        if hit_sl:
            remaining = 0.5 if t.get("partial_taken") else 1.0
            t["pnl"]         += pnl_usd(sl) * remaining
            t["status"]       = "closed_sl"
            t["close_price"]  = sl
            t["close_time"]   = datetime.now(timezone.utc).isoformat()
            updated = True
            logger.info(f"[Paper] {t['id']} SL @ {sl:.5f} PnL=${t['pnl']:.2f}")

    if updated:
        with _LOCK:
            _rewrite(trades)


# ── Performance metrics ───────────────────────────────────────────────────────

def paper_performance(strategy: Optional[str] = None, since_days: int = 14) -> dict:
    """Calculate paper trading performance for promotion check."""
    trades = get_all_trades()
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

    closed = [
        t for t in trades
        if t.get("status", "").startswith("closed")
        and (strategy is None or t.get("strategy") == strategy)
    ]

    # Filter by date
    recent = []
    for t in closed:
        try:
            ct = datetime.fromisoformat(t.get("close_time") or t.get("open_time"))
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            if ct >= cutoff:
                recent.append(t)
        except Exception:
            pass

    if not recent:
        return {"trades": 0, "pnl": 0.0, "win_rate": 0.0, "sharpe": 0.0, "days": since_days}

    pnls    = [t["pnl"] for t in recent]
    wins    = sum(1 for p in pnls if p > 0)
    net     = sum(pnls)
    wr      = wins / len(pnls) * 100

    import statistics
    mu  = statistics.mean(pnls)
    std = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
    sharpe = mu / std if std > 0 else 0.0

    return {
        "trades":   len(recent),
        "pnl":      round(net, 2),
        "win_rate": round(wr, 1),
        "sharpe":   round(sharpe, 3),
        "days":     since_days,
    }


def should_promote(strategy: str = "SMC_TREND") -> tuple[bool, str]:
    """
    Check if paper trading results warrant promotion to live.
    Returns (should_promote, reason).
    """
    perf = paper_performance(strategy, since_days=_PROMOTE_DAYS)

    if perf["trades"] < 10:
        return False, f"Not enough paper trades ({perf['trades']} < 10)"
    if perf["pnl"] <= 0:
        return False, f"Paper PnL negative (${perf['pnl']:.2f})"
    if perf["sharpe"] < _PROMOTE_SHARPE:
        return False, f"Paper Sharpe too low ({perf['sharpe']:.2f} < {_PROMOTE_SHARPE})"

    return True, (
        f"Paper trading passed: {perf['trades']} trades, "
        f"PnL=${perf['pnl']:.2f}, Sharpe={perf['sharpe']:.2f}"
    )
