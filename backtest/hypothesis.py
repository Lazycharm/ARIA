"""
Phase 3 — Strategy Hypothesis Generator.

Analyses a BacktestResults object and uses Haiku to:
  1. Identify patterns in winning vs losing trades
  2. Pinpoint the weakest score component / exit type
  3. Suggest one concrete parameter adjustment

Cost: ~$0.001 per call (Haiku, ~600 tokens in / ~300 out)
Saves findings to Obsidian Backtests folder if vault is configured.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from backtest.metrics import BacktestResults, Trade


# ── Trade breakdowns ─────────────────────────────────────────────────────────

@dataclass
class _Bucket:
    n: int = 0
    wins: int = 0
    pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n * 100 if self.n else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.pnl / self.n if self.n else 0.0


def _compute_stats(results: BacktestResults) -> dict:
    """Build structured breakdown from trade list."""
    trades = results.trades
    if not trades:
        return {}

    # Direction split
    by_dir: dict[str, _Bucket] = defaultdict(_Bucket)
    for t in trades:
        b = by_dir[t.direction]
        b.n += 1
        b.pnl += t.pnl
        if t.pnl > 0:
            b.wins += 1

    # Exit reason split
    by_exit: dict[str, _Bucket] = defaultdict(_Bucket)
    for t in trades:
        b = by_exit[t.exit_reason]
        b.n += 1
        b.pnl += t.pnl
        if t.pnl > 0:
            b.wins += 1

    # Score buckets: <68, 68-73, 73-78, 78+
    score_ranges = [("<68", 0, 68), ("68-73", 68, 73), ("73-78", 73, 78), ("78+", 78, 999)]
    by_score: dict[str, _Bucket] = {label: _Bucket() for label, _, _ in score_ranges}
    for t in trades:
        for label, lo, hi in score_ranges:
            if lo <= t.score < hi:
                b = by_score[label]
                b.n += 1
                b.pnl += t.pnl
                if t.pnl > 0:
                    b.wins += 1
                break

    # SL type split
    by_sl: dict[str, _Bucket] = defaultdict(_Bucket)
    for t in trades:
        b = by_sl[t.sl_type]
        b.n += 1
        b.pnl += t.pnl
        if t.pnl > 0:
            b.wins += 1

    # Hold time
    win_bars  = [t.bars_held for t in trades if t.pnl > 0]
    loss_bars = [t.bars_held for t in trades if t.pnl <= 0]
    avg_win_bars  = sum(win_bars) / len(win_bars) if win_bars else 0
    avg_loss_bars = sum(loss_bars) / len(loss_bars) if loss_bars else 0

    return {
        "by_direction": {k: {"n": v.n, "wr": v.win_rate, "avg_pnl": v.avg_pnl} for k, v in by_dir.items()},
        "by_exit":      {k: {"n": v.n, "wr": v.win_rate, "pct": v.n / len(trades) * 100} for k, v in by_exit.items()},
        "by_score":     {k: {"n": v.n, "wr": v.win_rate, "avg_pnl": v.avg_pnl} for k, v in by_score.items() if v.n > 0},
        "by_sl":        {k: {"n": v.n, "wr": v.win_rate} for k, v in by_sl.items()},
        "avg_win_bars":  round(avg_win_bars, 1),
        "avg_loss_bars": round(avg_loss_bars, 1),
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _format_prompt(results: BacktestResults, stats: dict) -> str:
    r = results
    lines = [
        f"ARIA Backtest Analysis — {r.pair} ({r.days}d)\n",
        f"SUMMARY: {r.total_trades} trades | WR={r.win_rate:.1f}% | PF={r.profit_factor:.2f} | "
        f"E=${r.expectancy:+.2f}/trade | MaxDD={r.max_drawdown:.1f}% | Sharpe={r.sharpe_ratio:.2f}",
        f"Net P&L: ${r.net_pnl:+.2f} ({r.net_pnl_pct:+.1f}%) | MaxConsecLoss={r.max_consecutive_losses}\n",
    ]

    if stats.get("by_direction"):
        lines.append("DIRECTION SPLIT:")
        for d, v in stats["by_direction"].items():
            lines.append(f"  {d.upper()}: {v['n']} trades, {v['wr']:.0f}% WR, avg ${v['avg_pnl']:+.2f}")

    if stats.get("by_score"):
        lines.append("\nSCORE BUCKETS:")
        for bucket, v in stats["by_score"].items():
            lines.append(f"  Score {bucket}: {v['n']} trades, {v['wr']:.0f}% WR, avg ${v['avg_pnl']:+.2f}")

    if stats.get("by_exit"):
        lines.append("\nEXIT REASONS:")
        for reason, v in stats["by_exit"].items():
            lines.append(f"  {reason.upper()}: {v['n']} ({v['pct']:.0f}% of trades), {v['wr']:.0f}% WR")

    if stats.get("by_sl"):
        lines.append("\nSL TYPE:")
        for sl_type, v in stats["by_sl"].items():
            lines.append(f"  {sl_type}: {v['n']} trades, {v['wr']:.0f}% WR")

    lines.append(f"\nHOLD TIME: avg win={stats.get('avg_win_bars', 0):.0f} bars, avg loss={stats.get('avg_loss_bars', 0):.0f} bars (1 bar = 15 min)")

    lines.append("""
You are a quantitative FX strategy analyst reviewing a backtest. Based ONLY on the data above:
1. Identify the single weakest element (score bucket, exit pattern, direction bias, or hold time issue)
2. Give one specific, actionable parameter suggestion (e.g., "raise min_score to 73 — score <68 trades are drag", "reduce TP2 ratio, time exits cluster at 48h")
3. Flag any structural concern (e.g., "SL too tight — 70% of losses exit via SL within 10 bars")

Reply in exactly 3 bullet points. Be direct, no preamble.""")

    return "\n".join(lines)


# ── AI call ───────────────────────────────────────────────────────────────────

def _call_haiku(prompt: str) -> str:
    try:
        import anthropic
        from config.settings import settings

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning(f"[Hypothesis] AI call failed: {e}")
        return "(AI analysis unavailable)"


# ── Stats printer ─────────────────────────────────────────────────────────────

def _print_stats(results: BacktestResults, stats: dict) -> None:
    sep = "─" * 44
    print(f"\n{sep}")
    print("  TRADE ANALYSIS BREAKDOWN")
    print(sep)

    if stats.get("by_direction"):
        for d, v in stats["by_direction"].items():
            print(f"  {d.upper():<8} {v['n']:>3} trades  WR={v['wr']:.0f}%  avg ${v['avg_pnl']:+.2f}")

    print(sep)
    if stats.get("by_score"):
        for bucket, v in stats["by_score"].items():
            print(f"  Score {bucket:<6}  {v['n']:>3} trades  WR={v['wr']:.0f}%  avg ${v['avg_pnl']:+.2f}")

    print(sep)
    if stats.get("by_exit"):
        for reason, v in stats["by_exit"].items():
            print(f"  {reason.upper():<8}  {v['n']:>3} ({v['pct']:.0f}%)  WR={v['wr']:.0f}%")

    wbars = stats.get("avg_win_bars", 0)
    lbars = stats.get("avg_loss_bars", 0)
    print(sep)
    print(f"  Avg win hold : {wbars:.0f} bars ({wbars * 15 / 60:.1f}h)")
    print(f"  Avg loss hold: {lbars:.0f} bars ({lbars * 15 / 60:.1f}h)")
    print(sep)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(results: BacktestResults) -> str:
    """
    Run statistical breakdown + AI hypothesis on BacktestResults.
    Prints formatted stats and returns AI insights string.
    """
    if not results.trades:
        print("No trades to analyse.")
        return ""

    stats = _compute_stats(results)
    _print_stats(results, stats)

    prompt = _format_prompt(results, stats)
    print("\n  Calling Haiku for strategy insights...")
    insights = _call_haiku(prompt)

    print(f"\n{'═'*44}")
    print("  STRATEGY HYPOTHESIS")
    print(f"{'═'*44}")
    print(insights)
    print(f"{'═'*44}\n")

    return insights


def save_to_obsidian(results: BacktestResults, insights: str, stats: dict) -> None:
    """Append hypothesis section to the backtest Obsidian note."""
    try:
        from config.settings import settings
        vault  = settings.obsidian_vault_path
        folder = settings.obsidian_aria_folder
        if not vault or not folder:
            return

        from pathlib import Path
        from datetime import datetime

        out_dir = Path(vault) / folder / "Backtests"
        if not out_dir.exists():
            return

        # Find latest backtest note for this pair
        notes = sorted(out_dir.glob(f"{results.pair}_*.md"), reverse=True)
        if not notes:
            return

        latest = notes[0]
        existing = latest.read_text(encoding="utf-8")

        section = f"""
## Strategy Hypothesis (Phase 3 AI Analysis)

### Trade Breakdown
| Category | N | Win Rate | Avg PnL |
|----------|---|----------|---------|
"""
        for d, v in stats.get("by_direction", {}).items():
            section += f"| {d.upper()} | {v['n']} | {v['wr']:.0f}% | ${v['avg_pnl']:+.2f} |\n"
        for bucket, v in stats.get("by_score", {}).items():
            section += f"| Score {bucket} | {v['n']} | {v['wr']:.0f}% | ${v['avg_pnl']:+.2f} |\n"

        section += f"\n### AI Insights\n{insights}\n"

        latest.write_text(existing + section, encoding="utf-8")
        print(f"Hypothesis appended to: {latest.name}")

    except Exception as e:
        logger.debug(f"[Hypothesis] Obsidian save failed: {e}")
