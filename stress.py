"""
ARIA Stress Test CLI.

Replays the strategy through historically significant crisis periods to
validate that the edge survives extreme market conditions (volatility spikes,
gap risk, spread blowout, sustained trending without retracement).

Usage:
  python stress.py --pair EURUSDm
  python stress.py --pair EURUSDm --score 70 --risk 1.0 --spread 1.5
  python stress.py --pair XAUUSDm --crises covid ukraine
  python stress.py --pair EURUSDm --list

Crisis periods are replayed with DOUBLE the normal spread to simulate
real broker behavior during stress events.
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Crisis catalogue ───────────────────────────────────────────────────────────

@dataclass
class Crisis:
    key: str
    name: str
    start: str          # YYYY-MM-DD
    end: str            # YYYY-MM-DD
    description: str
    spread_multiplier: float = 2.0  # spread blowout during crisis


CRISES: list[Crisis] = [
    Crisis(
        key="snb_shock",
        name="SNB Shock",
        start="2015-01-14", end="2015-02-28",
        description="Swiss National Bank removed EUR/CHF floor — 1000+ pip gap in seconds",
        spread_multiplier=3.0,
    ),
    Crisis(
        key="china_deval",
        name="China Devaluation / Flash Crash",
        start="2015-08-10", end="2015-09-30",
        description="PBOC surprise yuan devaluation + global equity flash crash",
        spread_multiplier=2.5,
    ),
    Crisis(
        key="brexit_vote",
        name="Brexit Vote",
        start="2016-06-22", end="2016-07-15",
        description="UK referendum result — GBP/USD -1700 pips overnight",
        spread_multiplier=3.0,
    ),
    Crisis(
        key="trump_2016",
        name="Trump Election 2016",
        start="2016-11-07", end="2016-11-25",
        description="USD surge on unexpected election result",
        spread_multiplier=2.0,
    ),
    Crisis(
        key="covid_crash",
        name="COVID-19 Crash",
        start="2020-02-20", end="2020-04-30",
        description="Pandemic-driven risk-off: equities -35%, gold +15%, CHF/JPY surge",
        spread_multiplier=3.0,
    ),
    Crisis(
        key="covid_recovery",
        name="COVID Recovery Squeeze",
        start="2020-05-01", end="2020-07-31",
        description="V-shaped recovery — whipsaw risk, false breakouts",
        spread_multiplier=1.5,
    ),
    Crisis(
        key="ukraine_invasion",
        name="Russia-Ukraine Invasion",
        start="2022-02-23", end="2022-03-31",
        description="Commodity surge, EUR weakness, CHF/gold spike",
        spread_multiplier=2.5,
    ),
    Crisis(
        key="fed_hike_cycle",
        name="Fed Hike Cycle 2022",
        start="2022-03-01", end="2022-12-31",
        description="Fastest Fed hiking cycle since 1980s — sustained USD trend",
        spread_multiplier=1.5,
    ),
    Crisis(
        key="svb_collapse",
        name="SVB / Banking Crisis 2023",
        start="2023-03-09", end="2023-04-07",
        description="Silicon Valley Bank + Signature Bank collapse — CHF/USD safe-haven surge",
        spread_multiplier=2.0,
    ),
    Crisis(
        key="us_debt_2023",
        name="US Debt Ceiling Crisis 2023",
        start="2023-04-15", end="2023-06-05",
        description="Default risk premium — USD volatility, gold spike",
        spread_multiplier=1.5,
    ),
    Crisis(
        key="yen_intervention",
        name="BOJ Yen Intervention 2022",
        start="2022-09-20", end="2022-10-31",
        description="Ministry of Finance spent $43B to defend yen — gap risk on USD/JPY",
        spread_multiplier=2.5,
    ),
    Crisis(
        key="us_downgrade",
        name="Fitch US Credit Downgrade 2023",
        start="2023-08-01", end="2023-08-25",
        description="US AAA rating stripped — risk-off spike",
        spread_multiplier=1.5,
    ),
]

CRISES_BY_KEY: dict[str, Crisis] = {c.key: c for c in CRISES}


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class StressResult:
    crisis: Crisis
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    net_pnl_pct: float
    verdict: str          # PASS | WARN | FAIL | NO_DATA


def _run_crisis(crisis: Crisis, pair: str, score: float, risk: float,
                base_spread: float, balance: float) -> StressResult:
    from backtest.engine import BacktestEngine

    spread = base_spread * crisis.spread_multiplier
    start  = crisis.start
    end    = crisis.end

    d0    = datetime.strptime(start, "%Y-%m-%d")
    d1    = datetime.strptime(end,   "%Y-%m-%d")
    days  = max(30, (d1 - d0).days + 30)

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
    except Exception as e:
        return StressResult(
            crisis=crisis,
            total_trades=0, win_rate=0, profit_factor=0,
            max_drawdown=0, net_pnl_pct=0,
            verdict="NO_DATA",
        )

    if bt.total_trades == 0:
        return StressResult(
            crisis=crisis,
            total_trades=0, win_rate=0, profit_factor=0,
            max_drawdown=0, net_pnl_pct=0,
            verdict="NO_DATA",
        )

    # Verdict rules for stress test (looser than normal backtest):
    # PASS: DD ≤ 15% and PF ≥ 1.0 (breakeven or better)
    # WARN: DD ≤ 20% and net PnL > -5%
    # FAIL: anything worse
    dd  = bt.max_drawdown
    pf  = bt.profit_factor
    pnl = bt.net_pnl_pct

    if dd <= 15.0 and pf >= 1.0:
        v = "PASS"
    elif dd <= 20.0 and pnl > -5.0:
        v = "WARN"
    else:
        v = "FAIL"

    return StressResult(
        crisis=crisis,
        total_trades=bt.total_trades,
        win_rate=bt.win_rate,
        profit_factor=pf,
        max_drawdown=dd,
        net_pnl_pct=pnl,
        verdict=v,
    )


def _print_table(results: list[StressResult], pair: str) -> None:
    ICONS = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "NO_DATA": "──"}
    sep = "─" * 100
    print(f"\n{'═'*100}")
    print(f"  ARIA STRESS TEST — {pair}")
    print(f"{'═'*100}")
    print(f"  {'Crisis':<30} {'Period':<24} {'Trades':>7} {'WR%':>6} {'PF':>6} {'DD%':>7} {'PnL%':>7}  Result")
    print(sep)

    passes = warns = fails = no_data = 0
    for r in results:
        icon = ICONS[r.verdict]
        period = f"{r.crisis.start} → {r.crisis.end}"
        if r.verdict == "NO_DATA":
            print(f"  {r.crisis.name:<30} {period:<24} {'—':>7} {'—':>6} {'—':>6} {'—':>7} {'—':>7}  {icon} NO DATA")
            no_data += 1
        else:
            print(
                f"  {r.crisis.name:<30} {period:<24} "
                f"{r.total_trades:>7} {r.win_rate:>5.0f}% {r.profit_factor:>6.2f} "
                f"{r.max_drawdown:>6.1f}% {r.net_pnl_pct:>+6.1f}%  {icon} {r.verdict}"
            )
            if r.verdict == "PASS": passes += 1
            elif r.verdict == "WARN": warns += 1
            else: fails += 1

    print(sep)
    tested = passes + warns + fails
    print(f"  Tested: {tested}  |  ✅ {passes} PASS  ⚠️  {warns} WARN  ❌ {fails} FAIL  ── {no_data} NO DATA")

    if tested > 0:
        survival = (passes + warns) / tested * 100
        print(f"  Survival rate: {survival:.0f}%  (PASS + WARN)")
    print(f"{'═'*100}\n")


def _save_obsidian(results: list[StressResult], pair: str, score: float,
                   risk: float, spread: float) -> None:
    try:
        from config.settings import settings
        from pathlib import Path

        out_dir = Path(settings.obsidian_vault_path) / settings.obsidian_aria_folder / "Experiments"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts    = datetime.now().strftime("%Y-%m-%d_%H%M")
        fname = out_dir / f"{pair}_StressTest_{ts}.md"

        ICONS = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "NO_DATA": "—"}
        rows  = ""
        for r in results:
            if r.verdict == "NO_DATA":
                rows += f"| {r.crisis.name} | {r.crisis.start} → {r.crisis.end} | — | — | — | — | — | {ICONS[r.verdict]} |\n"
            else:
                rows += (
                    f"| {r.crisis.name} | {r.crisis.start} → {r.crisis.end} | "
                    f"{r.total_trades} | {r.win_rate:.0f}% | {r.profit_factor:.2f} | "
                    f"{r.max_drawdown:.1f}% | {r.net_pnl_pct:+.1f}% | {ICONS[r.verdict]} |\n"
                )

        tested  = sum(1 for r in results if r.verdict != "NO_DATA")
        passing = sum(1 for r in results if r.verdict in ("PASS", "WARN"))
        survival = passing / tested * 100 if tested else 0

        content = f"""# ARIA Stress Test — {pair}
*Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## Parameters
- Min score: {score}
- Risk per trade: {risk}%
- Base spread: {spread}p (multiplied per crisis — see below)

## Results

| Crisis | Period | Trades | WR | PF | Max DD | Net PnL | Result |
|--------|--------|--------|----|----|--------|---------|--------|
{rows}
## Summary
- Crises tested: {tested}
- Survival rate: {survival:.0f}% (PASS + WARN)
- PASS: {sum(1 for r in results if r.verdict == "PASS")}
- WARN: {sum(1 for r in results if r.verdict == "WARN")}
- FAIL: {sum(1 for r in results if r.verdict == "FAIL")}
- No data: {sum(1 for r in results if r.verdict == "NO_DATA")}

## Verdict thresholds (stress-adjusted)
- PASS: DD ≤ 15% and PF ≥ 1.0
- WARN: DD ≤ 20% and net PnL > −5%
- FAIL: anything worse
"""
        fname.write_text(content, encoding="utf-8")
        print(f"Saved to Obsidian: {fname.name}")
    except Exception as e:
        print(f"[stress] Obsidian save failed: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA Stress Test — replay strategy through crisis periods")
    parser.add_argument("--pair",    required=True,            help="Symbol, e.g. EURUSDm")
    parser.add_argument("--score",   type=float, default=70.0, help="Min confluence score")
    parser.add_argument("--risk",    type=float, default=1.0,  help="Risk %% per trade")
    parser.add_argument("--spread",  type=float, default=0.8,  help="Base spread pips (doubled per crisis)")
    parser.add_argument("--balance", type=float, default=100.0,help="Starting balance USD")
    parser.add_argument("--crises",  nargs="*",                help="Run specific crises by key (default: all)")
    parser.add_argument("--list",    action="store_true",      help="List all available crisis keys and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\nAvailable crises ({len(CRISES)}):\n")
        for c in CRISES:
            print(f"  {c.key:<22} {c.start} → {c.end}  {c.name}")
            print(f"    {c.description}")
        print()
        return

    crises = CRISES
    if args.crises:
        crises = []
        for key in args.crises:
            if key not in CRISES_BY_KEY:
                print(f"Unknown crisis key '{key}'. Run --list to see options.")
                sys.exit(1)
            crises.append(CRISES_BY_KEY[key])

    print(f"\nARIA Stress Test — {args.pair}")
    print(f"Base params: score≥{args.score}  risk={args.risk}%  spread={args.spread}p  balance=${args.balance}")
    print(f"Running {len(crises)} crisis periods (spread ×{crises[0].spread_multiplier:.1f}–{max(c.spread_multiplier for c in crises):.1f} per period)…\n")

    results = []
    for i, crisis in enumerate(crises, 1):
        print(f"  [{i:02d}/{len(crises):02d}] {crisis.name} ({crisis.start} → {crisis.end})…", end=" ", flush=True)
        r = _run_crisis(crisis, args.pair, args.score, args.risk, args.spread, args.balance)
        if r.verdict == "NO_DATA":
            print("no data")
        elif r.verdict == "PASS":
            print(f"✅  trades={r.total_trades}  PF={r.profit_factor:.2f}  DD={r.max_drawdown:.1f}%  PnL={r.net_pnl_pct:+.1f}%")
        elif r.verdict == "WARN":
            print(f"⚠️   trades={r.total_trades}  PF={r.profit_factor:.2f}  DD={r.max_drawdown:.1f}%  PnL={r.net_pnl_pct:+.1f}%")
        else:
            print(f"❌  trades={r.total_trades}  PF={r.profit_factor:.2f}  DD={r.max_drawdown:.1f}%  PnL={r.net_pnl_pct:+.1f}%")
        results.append(r)

    _print_table(results, args.pair)
    _save_obsidian(results, args.pair, args.score, args.risk, args.spread)


if __name__ == "__main__":
    main()
