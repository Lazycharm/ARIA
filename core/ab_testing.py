"""
Strategy A/B Testing — run old vs new version of a strategy in parallel on demo.

When a hypothesis is promoted to paper trading, the new version runs alongside
the existing live strategy. After MIN_TRADES, statistical significance test
determines the winner. Loser is retired.

Tracks variants in db/ab_tests.json:
  {test_id, strategy_a, strategy_b, start_time, trades_a, trades_b,
   pnl_a, pnl_b, wins_a, wins_b, status, winner}
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PATH = Path("db/ab_tests.json")
_LOCK = threading.Lock()
MIN_TRADES = 20   # minimum trades per arm before significance test


@dataclass
class ABTest:
    test_id:     str
    strategy_a:  str          # control (current live)
    strategy_b:  str          # challenger (new hypothesis)
    pair:        str
    start_time:  str
    status:      str = "running"   # running | complete
    winner:      Optional[str] = None
    trades_a:    int = 0
    trades_b:    int = 0
    pnl_a:       float = 0.0
    pnl_b:       float = 0.0
    wins_a:      int = 0
    wins_b:      int = 0


def _load() -> dict[str, dict]:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, indent=2))


def create_test(strategy_a: str, strategy_b: str, pair: str) -> str:
    """Start a new A/B test. Returns test_id."""
    test_id = uuid.uuid4().hex[:8].upper()
    test = ABTest(
        test_id=test_id,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        pair=pair,
        start_time=datetime.now(timezone.utc).isoformat(),
    )
    with _LOCK:
        data = _load()
        data[test_id] = asdict(test)
        _save(data)
    return test_id


def record_result(test_id: str, variant: str, pnl: float, won: bool) -> None:
    """Record a trade outcome for variant 'a' or 'b'."""
    with _LOCK:
        data = _load()
        if test_id not in data:
            return
        t = data[test_id]
        if variant == "a":
            t["trades_a"] += 1; t["pnl_a"] += pnl; t["wins_a"] += 1 if won else 0
        else:
            t["trades_b"] += 1; t["pnl_b"] += pnl; t["wins_b"] += 1 if won else 0
        data[test_id] = t
        _save(data)
    _maybe_conclude(test_id)


def _maybe_conclude(test_id: str) -> None:
    """Run significance test if both arms have MIN_TRADES."""
    with _LOCK:
        data = _load()
        t = data.get(test_id, {})

    if t.get("status") != "running":
        return
    if t["trades_a"] < MIN_TRADES or t["trades_b"] < MIN_TRADES:
        return

    # Simple: compare win rates with a z-test (approx)
    import math
    n_a, n_b = t["trades_a"], t["trades_b"]
    p_a = t["wins_a"] / n_a if n_a else 0
    p_b = t["wins_b"] / n_b if n_b else 0
    p_pool = (t["wins_a"] + t["wins_b"]) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if p_pool * (1 - p_pool) > 0 else 0.001
    z = abs(p_a - p_b) / se

    winner = None
    if z >= 1.96:  # 95% confidence
        winner = t["strategy_a"] if p_a > p_b else t["strategy_b"]

    with _LOCK:
        data = _load()
        if test_id in data:
            data[test_id]["status"] = "complete"
            data[test_id]["winner"] = winner
            _save(data)

    from loguru import logger
    if winner:
        logger.info(f"[AB] Test {test_id}: winner={winner} z={z:.2f} p_a={p_a:.2%} p_b={p_b:.2%}")
        try:
            from notifications.telegram import send
            send(f"📊 A/B Test Complete\nTest: {test_id}\nPair: {t['pair']}\n"
                 f"Winner: {winner}\n"
                 f"A ({t['strategy_a']}): {p_a:.1%} WR, PnL=${t['pnl_a']:.2f}\n"
                 f"B ({t['strategy_b']}): {p_b:.1%} WR, PnL=${t['pnl_b']:.2f}")
        except Exception:
            pass
    else:
        logger.info(f"[AB] Test {test_id}: no winner (z={z:.2f}) — need more trades")


def get_active_tests() -> list[dict]:
    with _LOCK:
        return [v for v in _load().values() if v.get("status") == "running"]


def get_all_tests() -> list[dict]:
    with _LOCK:
        return list(_load().values())


def record_by_strategy(strategy: str, pnl: float, won: bool) -> None:
    """Record a trade result into any running test where strategy is arm A or B."""
    for test in get_active_tests():
        if test["strategy_a"] == strategy:
            record_result(test["test_id"], "a", pnl, won)
        elif test["strategy_b"] == strategy:
            record_result(test["test_id"], "b", pnl, won)
