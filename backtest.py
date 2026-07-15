"""
ARIA Backtest CLI.

Usage:
  python backtest.py --pair USDJPYm --days 90
  python backtest.py --pair EURUSDm --days 60 --risk 1.0 --score 70
  python backtest.py --pair EURUSDm --start 2024-01-01 --end 2024-06-30
  python backtest.py --pair EURUSDm --days 90 --analyze

Options:
  --pair    Pair symbol as listed in MT5 (e.g. USDJPYm, EURUSDm)
  --days    Historical days to test (default: 90; ignored when --start/--end used)
  --start   Start date YYYY-MM-DD (enables date-range mode)
  --end     End date YYYY-MM-DD (defaults to today when --start given)
  --risk    Risk % per trade (default: 1.0)
  --score   Minimum confluence score to enter (default: 70.0)
  --balance Starting balance for simulation (default: 100.0)
  --spread  Spread in pips to deduct per trade (default: 0.8)
  --analyze Run Phase 3 AI hypothesis on results
"""

import argparse
import sys
import os
from datetime import date

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA backtesting engine")
    parser.add_argument("--pair",    required=True, help="Symbol, e.g. USDJPYm")
    parser.add_argument("--days",    type=int,   default=90,   help="Historical days")
    parser.add_argument("--start",   type=str,   default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     type=str,   default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--risk",    type=float, default=1.0,  help="Risk %% per trade")
    parser.add_argument("--score",   type=float, default=70.0, help="Min confluence score")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance USD")
    parser.add_argument("--spread",  type=float, default=0.8,  help="Spread in pips")
    parser.add_argument("--analyze", action="store_true",     help="Run Phase 3 AI hypothesis on results")
    args = parser.parse_args()

    # When --start is given, auto-set --end to today if omitted
    if args.start and not args.end:
        args.end = date.today().isoformat()

    # With date-range mode, compute days for data fetching
    if args.start:
        from datetime import datetime
        d0 = datetime.strptime(args.start, "%Y-%m-%d")
        d1 = datetime.strptime(args.end,   "%Y-%m-%d")
        args.days = max(90, (d1 - d0).days + 30)   # +30 buffer for warmup

    from backtest.engine import BacktestEngine

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

    results = engine.run()

    print(results.summary())
    print(results.verdict())

    # Per-trade breakdown
    if results.trades:
        print("\nTrade breakdown (last 20):")
        print(f"{'#':<4} {'Dir':<6} {'Score':<7} {'PnL':>8} {'Exit':<8} {'Bars':>5} {'SL type'}")
        print("─" * 54)
        for i, t in enumerate(results.trades[-20:], 1):
            print(
                f"{i:<4} {t.direction:<6} {t.score:<7.1f}"
                f" {t.pnl:>+8.2f} {t.exit_reason:<8} {t.bars_held:>5}  {t.sl_type}"
            )

    # Phase 3: AI hypothesis (optional)
    if args.analyze:
        from backtest.hypothesis import analyze as run_hypothesis, save_to_obsidian as save_hyp
        from backtest.hypothesis import _compute_stats
        insights = run_hypothesis(results)
        if insights:
            stats = _compute_stats(results)
            save_hyp(results, insights, stats)

    # Save to Obsidian vault if configured
    _save_to_obsidian(results, args)


def _save_to_obsidian(results, args) -> None:
    """Save backtest results to Obsidian vault if path is configured."""
    try:
        from config.settings import settings
        vault = settings.obsidian_vault_path
        folder = settings.obsidian_aria_folder
        if not vault or not folder:
            return

        from pathlib import Path
        from datetime import datetime

        out_dir = Path(vault) / folder / "Backtests"
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        fname = out_dir / f"{results.pair}_{args.days}d_{timestamp}.md"

        content = f"""# ARIA Backtest — {results.pair} ({args.days}d)
*Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## Parameters
- Pair: `{results.pair}`
- Days: {args.days}
- Risk per trade: {args.risk}%
- Min score: {args.score}
- Starting balance: ${args.balance:.2f}
- Spread: {args.spread} pips

## Results

```
{results.summary()}
```

## Verdict
{results.verdict()}

## Trade Log
| # | Dir | Score | PnL | Exit | Bars | SL type |
|---|-----|-------|-----|------|------|---------|
"""
        for i, t in enumerate(results.trades, 1):
            content += f"| {i} | {t.direction} | {t.score:.0f} | ${t.pnl:+.2f} | {t.exit_reason} | {t.bars_held} | {t.sl_type} |\n"

        fname.write_text(content, encoding="utf-8")
        print(f"\nSaved to Obsidian: {fname}")
    except Exception as e:
        pass  # Obsidian save is best-effort


if __name__ == "__main__":
    main()
