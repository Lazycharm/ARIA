"""
ARIA Monte Carlo Validator CLI.

Phase 7 — runs Monte Carlo simulation on a strategy to validate that its
edge is statistically robust and not an artefact of lucky trade sequencing.

Usage:
  python montecarlo.py --pair EURUSDm --sims 1000
  python montecarlo.py --pair EURUSDm --days 180 --sims 2000 --ruin 20 --score 70
  python montecarlo.py --pair XAUUSDm --start 2024-01-01 --end 2024-12-31 --sims 1000

Options:
  --pair    Symbol (e.g. EURUSDm)
  --days    Historical days to backtest (default: 180)
  --start   Start date YYYY-MM-DD (overrides --days)
  --end     End date YYYY-MM-DD
  --score   Min confluence score for backtest (default: 70)
  --risk    Risk % per trade (default: 1.0)
  --balance Starting balance (default: 100)
  --spread  Spread in pips (default: 0.8)
  --sims    Monte Carlo simulations (default: 1000)
  --ruin    Drawdown % that counts as ruin (default: 20)
  --no-bootstrap  Shuffle-only mode (no resampling)
"""

import argparse
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA Monte Carlo validator")
    parser.add_argument("--pair",         required=True,              help="Symbol, e.g. EURUSDm")
    parser.add_argument("--days",         type=int,   default=180,    help="Historical days to backtest")
    parser.add_argument("--start",        type=str,   default=None,   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",          type=str,   default=None,   help="End date YYYY-MM-DD")
    parser.add_argument("--score",        type=float, default=70.0,   help="Min confluence score")
    parser.add_argument("--risk",         type=float, default=1.0,    help="Risk %% per trade")
    parser.add_argument("--balance",      type=float, default=100.0,  help="Starting balance USD")
    parser.add_argument("--spread",       type=float, default=0.8,    help="Spread in pips")
    parser.add_argument("--sims",         type=int,   default=1000,   help="Monte Carlo simulations")
    parser.add_argument("--ruin",         type=float, default=20.0,   help="Ruin drawdown threshold %%")
    parser.add_argument("--no-bootstrap", action="store_true",        help="Shuffle-only (no resampling)")
    args = parser.parse_args()

    if args.start and not args.end:
        args.end = date.today().isoformat()

    if args.start:
        from datetime import datetime
        d0 = datetime.strptime(args.start, "%Y-%m-%d")
        d1 = datetime.strptime(args.end,   "%Y-%m-%d")
        args.days = max(90, (d1 - d0).days + 30)

    print(f"\nARIA Monte Carlo Validator")
    print(f"  Pair    : {args.pair}")
    if args.start:
        print(f"  Period  : {args.start} → {args.end}")
    else:
        print(f"  Period  : last {args.days}d")
    print(f"  BT Params: score≥{args.score}  risk={args.risk}%  spread={args.spread}p")
    print(f"  MC Params: {args.sims:,} sims  ruin=−{args.ruin:.0f}%  bootstrap={'no' if args.no_bootstrap else 'yes'}")
    print(f"\nStep 1/2 — Running backtest to collect trades…")

    from backtest.engine import BacktestEngine
    from backtest.montecarlo import run as mc_run

    engine = BacktestEngine(
        pair=args.pair,
        days=args.days,
        initial_balance=args.balance,
        risk_pct=args.risk,
        min_score=args.score,
        spread_pips=args.spread,
        start_date=args.start,
        end_date=args.end,
    )
    bt = engine.run()

    if not bt.trades:
        print(f"\nNo trades generated — cannot run Monte Carlo.")
        sys.exit(1)

    print(f"  → {bt.total_trades} trades collected  "
          f"(WR {bt.win_rate:.0f}%  PF {bt.profit_factor:.2f}  DD {bt.max_drawdown:.1f}%)")
    print(f"\nStep 2/2 — Running {args.sims:,} Monte Carlo simulations…")

    mc = mc_run(
        backtest=bt,
        n_sims=args.sims,
        ruin_pct=args.ruin,
        bootstrap=not args.no_bootstrap,
    )
    mc.print_report()
    mc.save_to_obsidian()


if __name__ == "__main__":
    main()
