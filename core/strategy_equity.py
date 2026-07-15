"""
Strategy-level equity curve tracking.

Separate from pair-level tracking — this tracks cumulative PnL per strategy
(SMC_TREND, SESSION_BREAKOUT, MEAN_REVERSION, RANGE_TRADING).

Persists to db/strategy_equity.json.
Used by the A/B testing panel and strategy monitoring.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PATH = Path("db/strategy_equity.json")
_LOCK = threading.Lock()


@dataclass
class StrategyEquityCurve:
    strategy: str
    trades:   int = 0
    net_pnl:  float = 0.0
    wins:     int = 0
    losses:   int = 0
    equity:   list[float] = field(default_factory=lambda: [0.0])
    timestamps: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total * 100 if total > 0 else 0.0

    @property
    def sharpe(self) -> float:
        import statistics
        if len(self.equity) < 2:
            return 0.0
        deltas = [self.equity[i] - self.equity[i-1] for i in range(1, len(self.equity))]
        mu = statistics.mean(deltas)
        std = statistics.stdev(deltas) if len(deltas) > 1 else 1.0
        return mu / std if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        peak = self.equity[0]
        dd   = 0.0
        for v in self.equity:
            if v > peak:
                peak = v
            if peak > 0 and (peak - v) / abs(peak) > dd:
                dd = (peak - v) / abs(peak)
        return round(dd * 100, 2)


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


def _to_curve(data: dict) -> StrategyEquityCurve:
    return StrategyEquityCurve(
        strategy=data.get("strategy", "UNKNOWN"),
        trades=data.get("trades", 0),
        net_pnl=data.get("net_pnl", 0.0),
        wins=data.get("wins", 0),
        losses=data.get("losses", 0),
        equity=data.get("equity", [0.0]),
        timestamps=data.get("timestamps", []),
    )


def _from_curve(c: StrategyEquityCurve) -> dict:
    return {
        "strategy":   c.strategy,
        "trades":     c.trades,
        "net_pnl":    c.net_pnl,
        "wins":       c.wins,
        "losses":     c.losses,
        "equity":     c.equity[-500:],  # keep last 500 points
        "timestamps": c.timestamps[-500:],
    }


def record_trade(strategy: str, pnl: float, won: bool) -> None:
    """Record a closed trade for a strategy. Updates equity curve."""
    ts = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        data = _load()
        if strategy not in data:
            data[strategy] = {"strategy": strategy, "trades": 0, "net_pnl": 0.0,
                               "wins": 0, "losses": 0, "equity": [0.0], "timestamps": []}
        d = data[strategy]
        d["trades"]  += 1
        d["net_pnl"] += pnl
        d["wins"]    += 1 if won else 0
        d["losses"]  += 0 if won else 1
        d["equity"].append(d["net_pnl"])
        d["timestamps"].append(ts)
        data[strategy] = d
        _save(data)


def get_curve(strategy: str) -> Optional[StrategyEquityCurve]:
    with _LOCK:
        data = _load()
    d = data.get(strategy)
    return _to_curve(d) if d else None


def get_all_curves() -> dict[str, StrategyEquityCurve]:
    with _LOCK:
        data = _load()
    return {k: _to_curve(v) for k, v in data.items()}


def get_summary() -> list[dict]:
    """Return sorted summary for dashboard display."""
    curves = get_all_curves()
    out = []
    for name, c in curves.items():
        out.append({
            "strategy":    name,
            "trades":      c.trades,
            "net_pnl":     round(c.net_pnl, 2),
            "win_rate":    round(c.win_rate, 1),
            "sharpe":      round(c.sharpe, 3),
            "max_drawdown": c.max_drawdown,
        })
    return sorted(out, key=lambda x: x["sharpe"], reverse=True)
