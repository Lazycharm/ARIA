"""
Phase 7 — Monte Carlo Validation.

Validates that a strategy's edge is real and not a statistical artifact of
trade sequencing. Takes a completed backtest's trade list and runs N simulations
with shuffled/resampled trade order.

What it does:
  1. Shuffle trade order 1000× (order randomisation)
  2. Bootstrap resample 1000× (sample with replacement)
  3. Compute equity curve for each path
  4. Report 5th / 50th / 95th percentile final equity
  5. Calculate probability of ruin (path that hits −RUIN_PCT% drawdown)
  6. Verdict: PASS if ruin probability < 5%

CLI:
  python montecarlo.py --pair EURUSDm --sims 1000 --ruin 20
  python montecarlo.py --pair EURUSDm --days 180 --sims 2000 --ruin 20
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from backtest.metrics import BacktestResults, Trade


# ── Config ─────────────────────────────────────────────────────────────────────

RUIN_PCT_DEFAULT = 20.0      # % drawdown that counts as "ruin"
MIN_TRADES       = 30        # skip MC if too few trades


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    pair: str
    sims: int
    ruin_pct: float
    initial_balance: float

    # Distribution of final equity across all paths
    final_equities: list[float] = field(default_factory=list)

    # Equity curves at percentile bands
    p05_curve: list[float] = field(default_factory=list)  # worst 5%
    p50_curve: list[float] = field(default_factory=list)  # median
    p95_curve: list[float] = field(default_factory=list)  # best 5%

    ruin_count: int = 0

    @property
    def ruin_probability(self) -> float:
        return self.ruin_count / self.sims * 100 if self.sims > 0 else 0.0

    @property
    def p05_final(self) -> float:
        return float(np.percentile(self.final_equities, 5)) if self.final_equities else 0.0

    @property
    def p50_final(self) -> float:
        return float(np.percentile(self.final_equities, 50)) if self.final_equities else 0.0

    @property
    def p95_final(self) -> float:
        return float(np.percentile(self.final_equities, 95)) if self.final_equities else 0.0

    @property
    def expected_max_dd(self) -> float:
        """Mean max drawdown across all paths."""
        if not self.final_equities:
            return 0.0
        dds = getattr(self, "_max_dds", [])
        return float(np.mean(dds)) if dds else 0.0

    @property
    def worst_max_dd(self) -> float:
        dds = getattr(self, "_max_dds", [])
        return float(np.max(dds)) if dds else 0.0

    def verdict(self) -> str:
        ruin = self.ruin_probability
        p50  = self.p50_final - self.initial_balance
        p50_pct = p50 / self.initial_balance * 100 if self.initial_balance else 0

        issues = []
        if ruin >= 5.0:
            issues.append(f"Ruin probability too high ({ruin:.1f}% ≥ 5%)")
        if p50 <= 0:
            issues.append(f"Median path unprofitable (P50 net: ${p50:+.2f})")

        if not issues:
            return (
                f"✅ MC PASS — Ruin {ruin:.1f}%  |  "
                f"P05 ${self.p05_final:+.2f}  P50 ${self.p50_final:+.2f}  P95 ${self.p95_final:+.2f}"
            )
        return "❌ MC FAIL:\n" + "\n".join(f"  • {i}" for i in issues)

    def print_report(self) -> None:
        sep = "─" * 58
        print(f"\n{'═'*58}")
        print(f"  MONTE CARLO VALIDATION — {self.pair}")
        print(f"  Simulations : {self.sims:,}")
        print(f"  Ruin level  : −{self.ruin_pct:.0f}% drawdown")
        print(sep)
        print(f"  Final equity distribution ({self.sims:,} paths):")
        print(f"    Worst 5%  (P05) : ${self.p05_final:>8.2f}  "
              f"({(self.p05_final - self.initial_balance) / self.initial_balance * 100:+.1f}%)")
        print(f"    Median    (P50) : ${self.p50_final:>8.2f}  "
              f"({(self.p50_final - self.initial_balance) / self.initial_balance * 100:+.1f}%)")
        print(f"    Best 5%%   (P95) : ${self.p95_final:>8.2f}  "
              f"({(self.p95_final - self.initial_balance) / self.initial_balance * 100:+.1f}%)")
        print(sep)
        print(f"  Ruin probability : {self.ruin_probability:.2f}%  "
              f"({self.ruin_count:,} / {self.sims:,} paths ruined)")
        print(f"  Expected max DD  : {self.expected_max_dd:.1f}%")
        print(f"  Worst max DD     : {self.worst_max_dd:.1f}%")
        print(sep)
        print(f"  {self.verdict()}")
        print(f"{'═'*58}\n")

    def save_to_obsidian(self) -> None:
        try:
            from config.settings import settings
            vault  = settings.obsidian_vault_path
            folder = settings.obsidian_aria_folder
            if not vault or not folder:
                return

            from pathlib import Path
            from datetime import datetime

            out_dir = Path(vault) / folder / "MonteCarlo"
            out_dir.mkdir(parents=True, exist_ok=True)

            ts    = datetime.now().strftime("%Y-%m-%d_%H%M")
            fname = out_dir / f"{self.pair}_MC_{self.sims}sims_{ts}.md"

            content = f"""# ARIA Monte Carlo — {self.pair}
*Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## Parameters
- Simulations: {self.sims:,}
- Ruin level: −{self.ruin_pct:.0f}%
- Initial balance: ${self.initial_balance:.2f}

## Equity Distribution
| Percentile | Final Equity | Return |
|-----------|-------------|--------|
| P05 (worst 5%) | ${self.p05_final:.2f} | {(self.p05_final - self.initial_balance)/self.initial_balance*100:+.1f}% |
| P50 (median)   | ${self.p50_final:.2f} | {(self.p50_final - self.initial_balance)/self.initial_balance*100:+.1f}% |
| P95 (best 5%)  | ${self.p95_final:.2f} | {(self.p95_final - self.initial_balance)/self.initial_balance*100:+.1f}% |

## Risk
- Ruin probability: {self.ruin_probability:.2f}%  ({self.ruin_count:,} / {self.sims:,} paths)
- Expected max DD: {self.expected_max_dd:.1f}%
- Worst max DD: {self.worst_max_dd:.1f}%

## Verdict
{self.verdict()}
"""
            fname.write_text(content, encoding="utf-8")
            print(f"MC saved to Obsidian: {fname.name}")
        except Exception as e:
            logger.debug(f"[MC] Obsidian save failed: {e}")


# ── Core simulation ────────────────────────────────────────────────────────────

def _equity_curve(pnl_seq: list[float], initial: float) -> tuple[list[float], float]:
    """Build equity curve from a PnL sequence. Returns (curve, max_drawdown_pct)."""
    equity = initial
    curve  = [initial]
    peak   = initial
    max_dd = 0.0
    for pnl in pnl_seq:
        equity += pnl
        curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return curve, max_dd


def run(
    backtest: BacktestResults,
    n_sims: int   = 1000,
    ruin_pct: float = RUIN_PCT_DEFAULT,
    bootstrap: bool = True,
) -> MCResult:
    """
    Run Monte Carlo simulation on a completed backtest.

    Args:
        backtest:  Completed BacktestResults with at least MIN_TRADES trades.
        n_sims:    Number of Monte Carlo simulations (default 1000).
        ruin_pct:  Drawdown threshold that counts as ruin (default 20%).
        bootstrap: If True, run half the sims with bootstrap resampling
                   (sample with replacement) in addition to shuffle sims.
    """
    trades = backtest.trades
    if len(trades) < MIN_TRADES:
        logger.warning(f"[MC] Only {len(trades)} trades — results unreliable (min {MIN_TRADES})")

    pnl_list = [t.pnl for t in trades]
    n        = len(pnl_list)
    initial  = backtest.initial_balance

    result = MCResult(
        pair=backtest.pair,
        sims=n_sims,
        ruin_pct=ruin_pct,
        initial_balance=initial,
    )

    all_curves: list[list[float]] = []
    all_final:  list[float]       = []
    all_maxdds: list[float]       = []

    n_shuffle   = n_sims // 2 if bootstrap else n_sims
    n_bootstrap = n_sims - n_shuffle if bootstrap else 0

    # ── Shuffle simulations ────────────────────────────────────────────────
    for _ in range(n_shuffle):
        shuffled = pnl_list[:]
        random.shuffle(shuffled)
        curve, max_dd = _equity_curve(shuffled, initial)
        all_curves.append(curve)
        all_final.append(curve[-1])
        all_maxdds.append(max_dd)
        if max_dd >= ruin_pct:
            result.ruin_count += 1

    # ── Bootstrap simulations (sample with replacement) ───────────────────
    for _ in range(n_bootstrap):
        resampled = random.choices(pnl_list, k=n)
        curve, max_dd = _equity_curve(resampled, initial)
        all_curves.append(curve)
        all_final.append(curve[-1])
        all_maxdds.append(max_dd)
        if max_dd >= ruin_pct:
            result.ruin_count += 1

    result.final_equities = all_final
    result._max_dds       = all_maxdds     # type: ignore[attr-defined]

    # ── Build percentile equity curves (align lengths by truncating to min) ─
    if all_curves:
        min_len = min(len(c) for c in all_curves)
        matrix  = np.array([c[:min_len] for c in all_curves])
        result.p05_curve = list(np.percentile(matrix, 5,  axis=0))
        result.p50_curve = list(np.percentile(matrix, 50, axis=0))
        result.p95_curve = list(np.percentile(matrix, 95, axis=0))

    return result
