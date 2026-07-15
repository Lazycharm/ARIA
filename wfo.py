"""
ARIA Walk-Forward Optimizer CLI.

Phase 4 — validates strategy params are not overfitted to one historical period.

Usage:
  python wfo.py --pair EURUSDm --total 365 --is 90 --oos 30 --step 30
  python wfo.py --pair XAUUSDm --total 180 --is 60 --oos 20 --step 20

Options:
  --pair     Symbol (e.g. EURUSDm)
  --total    Total historical days to use (default: 365)
  --is       In-sample window in days (default: 90)
  --oos      Out-of-sample window in days (default: 30)
  --step     Slide step in days (default: 30)
  --balance  Starting balance for each window sim (default: 100)
  --spread   Spread in pips (default: 0.8)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA walk-forward optimizer")
    parser.add_argument("--pair",    required=True,             help="Symbol, e.g. EURUSDm")
    parser.add_argument("--total",   type=int,   default=365,   help="Total historical days")
    parser.add_argument("--is",      type=int,   default=90,    dest="is_days", help="In-sample window days")
    parser.add_argument("--oos",     type=int,   default=30,    dest="oos_days", help="OOS window days")
    parser.add_argument("--step",    type=int,   default=30,    help="Slide step days")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance per window")
    parser.add_argument("--spread",  type=float, default=0.8,   help="Spread in pips")
    args = parser.parse_args()

    if args.is_days <= args.oos_days:
        print(f"Error: IS ({args.is_days}d) must be longer than OOS ({args.oos_days}d)")
        sys.exit(1)

    if args.total < args.is_days + args.oos_days:
        print(f"Error: total ({args.total}d) must be > IS+OOS ({args.is_days + args.oos_days}d)")
        sys.exit(1)

    n_windows_approx = (args.total - args.is_days - args.oos_days) // args.step + 1
    print(f"\nARIA Walk-Forward Optimizer")
    print(f"  Pair    : {args.pair}")
    print(f"  Total   : {args.total}d  |  IS={args.is_days}d  OOS={args.oos_days}d  Step={args.step}d")
    print(f"  Windows : ~{n_windows_approx}")
    print(f"  Grid    : min_score=[65,70,75,80] × risk_pct=[0.5,1.0,1.5]  ({4*3} combos per IS)")
    print(f"  Estimated time: {n_windows_approx * 4 * 3 * 8:.0f}–{n_windows_approx * 4 * 3 * 25:.0f}s\n")

    from backtest.wfo import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(
        pair=args.pair,
        total_days=args.total,
        is_days=args.is_days,
        oos_days=args.oos_days,
        step_days=args.step,
        initial_balance=args.balance,
        spread_pips=args.spread,
    )

    summary = optimizer.run()
    summary.print_report()
    summary.save_to_obsidian()


if __name__ == "__main__":
    main()
