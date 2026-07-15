"""
Adaptive Learning Engine — ARIA's self-improvement module.

After every closed trade, the engine:
  1. Records result with full context (pair, score, direction, session, P&L)
  2. Updates per-pair rolling stats (win rate, avg P&L, consecutive losses)
  3. Adjusts per-pair min score threshold dynamically
  4. Adjusts position-size multiplier based on recent performance

Per-pair threshold rules:
  3 consecutive losses → +5 threshold (max 85), lot multiplier × 0.6
  5 wins in last 10    → -3 threshold (min 62), lot multiplier × 1.1
  First win after loss streak → reset lot multiplier to 1.0

Global guards:
  5+ consecutive losses total → raise all thresholds by 10, halve lots
  Recovery after global streak → gradual reset

Stats persist across restarts via JSON. Thread-safe.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


_STORE_PATH = Path("./db/adaptive_learning.json")
_BASE_THRESHOLD = 70.0
_MIN_THRESHOLD  = 60.0
_MAX_THRESHOLD  = 88.0


@dataclass
class PairStats:
    pair: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_10_results: list[bool] = field(default_factory=list)   # True=win
    min_score: float = _BASE_THRESHOLD
    lot_multiplier: float = 1.0
    last_updated: str = ""

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades * 100

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def record(self, won: bool, pnl: float) -> None:
        if won:
            self.wins += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        else:
            self.losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.total_pnl += pnl
        self.last_10_results.append(won)
        if len(self.last_10_results) > 10:
            self.last_10_results.pop(0)

        self._adjust_parameters()
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def _adjust_parameters(self) -> None:
        recent_wins = sum(1 for r in self.last_10_results if r)
        recent_n    = len(self.last_10_results)

        # Raise threshold on losing streak — be more selective
        if self.consecutive_losses >= 3:
            self.min_score = min(_MAX_THRESHOLD, self.min_score + 5.0)
            self.lot_multiplier = max(0.5, self.lot_multiplier * 0.75)
            logger.info(f"[Learn] {self.pair}: {self.consecutive_losses} losses → threshold={self.min_score:.0f} lots×{self.lot_multiplier:.2f}")

        # Lower threshold when performing well — take more trades
        elif recent_n >= 5 and recent_wins / recent_n >= 0.6 and self.consecutive_losses == 0:
            self.min_score = max(_MIN_THRESHOLD, self.min_score - 2.0)
            self.lot_multiplier = min(1.5, self.lot_multiplier * 1.05)

        # Recovery: win after losing streak — reset lots to 1.0 gradually
        elif self.consecutive_losses == 0 and self.lot_multiplier < 1.0:
            self.lot_multiplier = min(1.0, self.lot_multiplier + 0.1)
            if self.lot_multiplier >= 1.0:
                logger.info(f"[Learn] {self.pair}: lot multiplier recovered → 1.0")


class AdaptiveLearning:
    """Thread-safe adaptive learning registry for all pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pairs: dict[str, PairStats] = {}
        self._global_consecutive_losses = 0
        self._global_conservative = False
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(
        self,
        pair: str,
        won: bool,
        pnl: float,
        score: float,
        direction: str,
        session: str,
    ) -> None:
        """Call after every trade close. Updates stats and adjusts thresholds."""
        with self._lock:
            if pair not in self._pairs:
                self._pairs[pair] = PairStats(pair=pair)

            stats = self._pairs[pair]
            stats.record(won, pnl)

            # Global streak tracking
            if won:
                self._global_consecutive_losses = 0
                if self._global_conservative:
                    self._global_conservative = False
                    logger.info("[Learn] Global conservative mode lifted — win recorded")
            else:
                self._global_consecutive_losses += 1
                if self._global_consecutive_losses >= 5 and not self._global_conservative:
                    self._global_conservative = True
                    logger.warning(f"[Learn] Global conservative mode ON — {self._global_consecutive_losses} consecutive losses")
                    try:
                        from notifications.telegram import alert_conservative_mode
                        alert_conservative_mode(self._global_consecutive_losses)
                    except Exception:
                        pass

            logger.info(
                f"[Learn] {pair} {'WIN' if won else 'LOSS'} ${pnl:+.2f} "
                f"score={score:.0f} session={session} | "
                f"pair threshold={stats.min_score:.0f} lot×{stats.lot_multiplier:.2f} "
                f"({stats.wins}W/{stats.losses}L)"
            )
            self._save()

    def get_min_score(self, pair: str) -> float:
        """Dynamic min score for this pair. Replaces the hard-coded threshold."""
        with self._lock:
            base = self._pairs.get(pair, PairStats(pair=pair)).min_score
            # Global conservative mode raises all thresholds
            if self._global_conservative:
                return min(_MAX_THRESHOLD, base + 10.0)
            return base

    def get_lot_multiplier(self, pair: str) -> float:
        """Lot size multiplier for this pair (0.5–1.5)."""
        with self._lock:
            base = self._pairs.get(pair, PairStats(pair=pair)).lot_multiplier
            if self._global_conservative:
                return base * 0.5
            return base

    def get_stats(self, pair: str) -> Optional[PairStats]:
        with self._lock:
            return self._pairs.get(pair)

    def all_stats(self) -> dict[str, PairStats]:
        with self._lock:
            return dict(self._pairs)

    def is_global_conservative(self) -> bool:
        with self._lock:
            return self._global_conservative

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "global_consecutive_losses": self._global_consecutive_losses,
                "global_conservative": self._global_conservative,
                "pairs": {k: asdict(v) for k, v in self._pairs.items()},
            }
            _STORE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"[Learn] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not _STORE_PATH.exists():
                return
            data = json.loads(_STORE_PATH.read_text())
            self._global_consecutive_losses = data.get("global_consecutive_losses", 0)
            self._global_conservative = data.get("global_conservative", False)
            for pair, raw in data.get("pairs", {}).items():
                self._pairs[pair] = PairStats(**raw)
            logger.info(f"[Learn] Loaded stats for {len(self._pairs)} pairs")
        except Exception as e:
            logger.warning(f"[Learn] Load failed (starting fresh): {e}")


# Singleton
adaptive = AdaptiveLearning()
