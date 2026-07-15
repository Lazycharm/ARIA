"""
ARIA Sensitivity Analysis CLI.

Sweeps key strategy parameters ±20% around baseline values and measures
how much performance degrades. High sensitivity = likely overfitted to
the specific parameter values found during optimization.

Overfitting signal: if PF drops >30% at ±10% parameter change → overfitted.

Parameters swept:
  1. min_score   — confluence threshold (e.g. 70 → 56, 63, 70, 77, 84)
  2. risk_pct    — position size risk (e.g. 1.0 → 0.6, 0.8, 1.0, 1.2, 1.4)
  3. spread_pips — execution cost (e.g. 0.8 → 0.48, 0.64, 0.8, 0.96, 1.12)

Usage:
  python sensitivity.py --pair EURUSDm
  python sensitivity.py --pair EURUSDm --score 70 --risk 1.0 --spread 0.8 --days 180
  python sensitivity.py --pair XAUUSDm --params score spread
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Sweep config ───────────────────────────────────────────────────────────────

SWEEP_STEPS = [-0.20, -0.10, 0.0, +0.10, +0.20]   # ±20% in 10% increments
STEP_LABELS = ["-20%", "-10%", "BASE", "+10%", "+20%"]

# Overfitting threshold: if PF drops more than this vs base at ±10% → flag
OVERFIT_PF_DROP = 0.30   # 30%


@dataclass
class SweepPoint:
    param: str
    label: str           # "-20%", "-10%", "BASE", "+10%", "+20%"
    value: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    net_pnl_pct: float
    error: bool = False


def _run_point(
    pair: str, days: int, balance: float,
    score: float, risk: float, spread: float,
    start: Optional[str], end: Optional[str],
) -> SweepPoint | None:
    """Run one backtest and return metrics. Returns None on exception."""
    from backtest.engine import BacktestEngine
    try:
        engine = BacktestEngine(
            pair=pair,
            days=days,
            initial_balance=balance,
            risk_pct=risk,
            min_score=score,
            spread_pips=spread,
            start_date=start,
            end_date=end,
        )
        bt = engine.run()
        return bt
    except Exception as e:
        return None


def _sweep_param(
    param_name: str, base_value: float,
    pair: str, days: int, balance: float,
    base_score: float, base_risk: float, base_spread: float,
    start: Optional[str], end: Optional[str],
) -> list[SweepPoint]:
    points = []
    for step, label in zip(SWEEP_STEPS, STEP_LABELS):
        value = base_value * (1 + step)
        # Clamp to reasonable ranges
        if param_name == "score":
            value = max(40.0, min(95.0, value))
        elif param_name == "risk":
            value = max(0.1, min(5.0, value))
        elif param_name == "spread":
            value = max(0.0, min(10.0, value))

        score  = value if param_name == "score"  else base_score
        risk   = value if param_name == "risk"   else base_risk
        spread = value if param_name == "spread" else base_spread

        bt = _run_point(pair, days, balance, score, risk, spread, start, end)
        if bt is None or bt.total_trades == 0:
            points.append(SweepPoint(
                param=param_name, label=label, value=value,
                total_trades=0, win_rate=0, profit_factor=0,
                max_drawdown=0, net_pnl_pct=0, error=True,
            ))
        else:
            points.append(SweepPoint(
                param=param_name, label=label, value=value,
                total_trades=bt.total_trades,
                win_rate=bt.win_rate,
                profit_factor=bt.profit_factor,
                max_drawdown=bt.max_drawdown,
                net_pnl_pct=bt.net_pnl_pct,
            ))
    return points


def _print_sweep(points: list[SweepPoint], param_name: str, base_pf: float) -> None:
    print(f"\n  Parameter: {param_name.upper()}")
    print(f"  {'Label':<8} {'Value':>8} {'Trades':>7} {'WR%':>6} {'PF':>6} {'DD%':>7} {'PnL%':>7}  Δ PF    Flag")
    print(f"  {'─'*80}")
    for p in points:
        if p.error:
            print(f"  {p.label:<8} {p.value:>8.2f} {'—':>7} {'—':>6} {'—':>6} {'—':>7} {'—':>7}  {'—':>6}  NO DATA")
            continue
        delta_pf = (p.profit_factor - base_pf) / base_pf * 100 if base_pf > 0 else 0.0
        is_base  = p.label == "BASE"
        # Flag overfitting if ±10% step causes >30% PF drop
        flag = ""
        if not is_base and abs(delta_pf) / 100 > OVERFIT_PF_DROP and p.label in ("-10%", "+10%"):
            flag = "⚠️  SENSITIVE"
        base_marker = " ◀" if is_base else ""
        print(
            f"  {p.label:<8} {p.value:>8.2f} {p.total_trades:>7} {p.win_rate:>5.0f}% "
            f"{p.profit_factor:>6.2f} {p.max_drawdown:>6.1f}% {p.net_pnl_pct:>+6.1f}%  "
            f"{delta_pf:>+5.0f}%  {flag}{base_marker}"
        )


def _overfitting_verdict(all_sweeps: dict[str, list[SweepPoint]], base_pf: float) -> str:
    flags = []
    for param, points in all_sweeps.items():
        base_pt = next((p for p in points if p.label == "BASE"), None)
        if not base_pt or base_pt.error:
            continue
        for p in points:
            if p.error or p.label not in ("-10%", "+10%"):
                continue
            drop = (base_pf - p.profit_factor) / base_pf if base_pf > 0 else 0
            if drop > OVERFIT_PF_DROP:
                flags.append(f"{param} at {p.label}: PF dropped {drop*100:.0f}%")

    if not flags:
        return "✅ ROBUST — no parameter shows >30% PF degradation at ±10% perturbation"
    return "⚠️  SENSITIVE:\n" + "\n".join(f"  • {f}" for f in flags)


def _save_obsidian(
    all_sweeps: dict[str, list[SweepPoint]],
    pair: str, days: int, base_score: float, base_risk: float, base_spread: float,
    overfitting: str,
) -> None:
    try:
        from config.settings import settings
        from pathlib import Path

        out_dir = Path(settings.obsidian_vault_path) / settings.obsidian_aria_folder / "Optimization"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts    = datetime.now().strftime("%Y-%m-%d_%H%M")
        fname = out_dir / f"{pair}_Sensitivity_{ts}.md"

        sections = ""
        for param, points in all_sweeps.items():
            base_pt = next((p for p in points if p.label == "BASE"), None)
            base_pf = base_pt.profit_factor if base_pt and not base_pt.error else 0
            rows = ""
            for p in points:
                if p.error:
                    rows += f"| {p.label} | {p.value:.2f} | — | — | — | — | — | — |\n"
                else:
                    delta = (p.profit_factor - base_pf) / base_pf * 100 if base_pf > 0 else 0
                    marker = " ◀" if p.label == "BASE" else ""
                    rows += (
                        f"| {p.label}{marker} | {p.value:.2f} | {p.total_trades} | "
                        f"{p.win_rate:.0f}% | {p.profit_factor:.2f} | "
                        f"{p.max_drawdown:.1f}% | {p.net_pnl_pct:+.1f}% | {delta:+.0f}% |\n"
                    )
            sections += f"""
### {param.upper()}

| Step | Value | Trades | WR | PF | DD | PnL | ΔPF |
|------|-------|--------|----|----|-----|-----|-----|
{rows}
"""

        content = f"""# ARIA Sensitivity Analysis — {pair}
*Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## Baseline Parameters
- Min score: {base_score}
- Risk per trade: {base_risk}%
- Spread: {base_spread}p
- Lookback: {days} days

## Sensitivity Sweeps
{sections}
## Overfitting Verdict
{overfitting}

## Interpretation Guide
- **ROBUST**: PF stays within ±30% across all ±10% param perturbations
- **SENSITIVE**: PF drops >30% at ±10% → strategy is curve-fitted to these exact params
- Action for SENSITIVE params: widen search range in WFO, or add regularization
"""
        fname.write_text(content, encoding="utf-8")
        print(f"Saved to Obsidian: {fname.name}")
    except Exception as e:
        print(f"[sensitivity] Obsidian save failed: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA Sensitivity Analysis — sweep params ±20%")
    parser.add_argument("--pair",    required=True,              help="Symbol, e.g. EURUSDm")
    parser.add_argument("--score",   type=float, default=70.0,   help="Baseline min confluence score")
    parser.add_argument("--risk",    type=float, default=1.0,    help="Baseline risk %% per trade")
    parser.add_argument("--spread",  type=float, default=0.8,    help="Baseline spread pips")
    parser.add_argument("--balance", type=float, default=100.0,  help="Starting balance USD")
    parser.add_argument("--days",    type=int,   default=180,    help="Lookback days")
    parser.add_argument("--start",   type=str,   default=None,   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     type=str,   default=None,   help="End date YYYY-MM-DD")
    parser.add_argument("--params",  nargs="*",
                        choices=["score", "risk", "spread"],
                        default=["score", "risk", "spread"],
                        help="Which parameters to sweep (default: all three)")
    args = parser.parse_args()

    if args.start and not args.end:
        args.end = date.today().isoformat()
    if args.start:
        from datetime import datetime as dt
        d0 = dt.strptime(args.start, "%Y-%m-%d")
        d1 = dt.strptime(args.end,   "%Y-%m-%d")
        args.days = max(90, (d1 - d0).days + 30)

    param_map = {
        "score":  args.score,
        "risk":   args.risk,
        "spread": args.spread,
    }

    total_runs = len(args.params) * len(SWEEP_STEPS)
    print(f"\nARIA Sensitivity Analysis — {args.pair}")
    print(f"Baseline: score={args.score}  risk={args.risk}%  spread={args.spread}p  days={args.days}")
    print(f"Params: {args.params}  |  Steps: {STEP_LABELS}  |  Total runs: {total_runs}")
    print()

    all_sweeps: dict[str, list[SweepPoint]] = {}
    run_num = 0

    for param in args.params:
        base_value = param_map[param]
        print(f"  Sweeping {param.upper()} (base={base_value})…")
        points = []
        for step, label in zip(SWEEP_STEPS, STEP_LABELS):
            run_num += 1
            value = base_value * (1 + step)
            print(f"    [{run_num:02d}/{total_runs:02d}] {label} ({param}={value:.2f})…", end=" ", flush=True)

            score  = value if param == "score"  else args.score
            risk   = value if param == "risk"   else args.risk
            spread = value if param == "spread" else args.spread

            bt = _run_point(args.pair, args.days, args.balance, score, risk, spread, args.start, args.end)
            if bt is None or bt.total_trades == 0:
                print("no data")
                points.append(SweepPoint(
                    param=param, label=label, value=value,
                    total_trades=0, win_rate=0, profit_factor=0,
                    max_drawdown=0, net_pnl_pct=0, error=True,
                ))
            else:
                print(f"trades={bt.total_trades}  PF={bt.profit_factor:.2f}  DD={bt.max_drawdown:.1f}%")
                points.append(SweepPoint(
                    param=param, label=label, value=value,
                    total_trades=bt.total_trades,
                    win_rate=bt.win_rate,
                    profit_factor=bt.profit_factor,
                    max_drawdown=bt.max_drawdown,
                    net_pnl_pct=bt.net_pnl_pct,
                ))
        all_sweeps[param] = points

    # Get baseline PF from score sweep (or first available)
    base_pf = 0.0
    for param, points in all_sweeps.items():
        bp = next((p for p in points if p.label == "BASE" and not p.error), None)
        if bp:
            base_pf = bp.profit_factor
            break

    print(f"\n{'═'*90}")
    print(f"  SENSITIVITY RESULTS — {args.pair}")
    print(f"{'═'*90}")
    for param, points in all_sweeps.items():
        _print_sweep(points, param, base_pf)

    overfitting = _overfitting_verdict(all_sweeps, base_pf)
    print(f"\n  Overfitting Verdict: {overfitting}")
    print(f"{'═'*90}\n")

    _save_obsidian(all_sweeps, args.pair, args.days, args.score, args.risk, args.spread, overfitting)


if __name__ == "__main__":
    from datetime import date
    main()
