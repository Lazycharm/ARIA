"""
Out-of-sample profitability check — separate IS/OOS split before running WFO.

Splits historical data 70% IS / 30% OOS.
Optimizes min_score on IS period, validates on OOS.
Fails if OOS profit factor < 1.0 or WR < 40%.

CLI:
  python -m backtest.oos_check EURUSDm --days 180
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


@dataclass
class OOSResult:
    pair:          str
    days:          int
    is_pf:         float
    oos_pf:        float
    is_wr:         float
    oos_wr:        float
    is_trades:     int
    oos_trades:    int
    best_score:    float
    passed:        bool
    verdict:       str


def run_oos_check(
    pair:      str,
    days:      int = 180,
    is_frac:   float = 0.70,
    risk_pct:  float = 1.0,
    score_range: Optional[list[float]] = None,
) -> OOSResult:
    """
    Split data IS/OOS, grid-search min_score on IS, validate on OOS.
    Returns OOSResult with pass/fail verdict.
    """
    score_range = score_range or [65, 70, 75, 80]

    from data.mt5_feed import feed
    import pandas as pd

    df_m15 = feed.get_candles(pair, "M15", count=days * 96 + 50)
    if df_m15.empty or len(df_m15) < 300:
        logger.error(f"[OOS] Not enough data for {pair}")
        return OOSResult(pair, days, 0, 0, 0, 0, 0, 0, 70, False, "INSUFFICIENT DATA")

    split_idx    = int(len(df_m15) * is_frac)
    df_is        = df_m15.iloc[:split_idx]
    df_oos       = df_m15.iloc[split_idx:]
    is_days      = max(1, int(days * is_frac))
    oos_days     = max(1, days - is_days)

    # Grid search min_score on IS
    best_pf    = 0.0
    best_score = 70.0

    for score in score_range:
        engine = BacktestEngine(pair=pair, days=is_days, risk_pct=risk_pct, min_score=score)
        r = engine.run(df_m15=df_is.copy())
        if not r.trades:
            continue
        m = compute_metrics(r.trades, r.equity_curve, r.initial_balance, is_days)
        pf = m.get("profit_factor", 0.0)
        if pf > best_pf:
            best_pf    = pf
            best_score = score

    # Validate on OOS
    engine_oos = BacktestEngine(pair=pair, days=oos_days, risk_pct=risk_pct, min_score=best_score)
    r_is  = BacktestEngine(pair=pair, days=is_days,  risk_pct=risk_pct, min_score=best_score).run(df_m15=df_is.copy())
    r_oos = engine_oos.run(df_m15=df_oos.copy())

    m_is  = compute_metrics(r_is.trades,  r_is.equity_curve,  r_is.initial_balance,  is_days)
    m_oos = compute_metrics(r_oos.trades, r_oos.equity_curve, r_oos.initial_balance, oos_days)

    is_pf  = m_is.get("profit_factor",  0.0)
    oos_pf = m_oos.get("profit_factor", 0.0)
    is_wr  = r_is.win_rate
    oos_wr = r_oos.win_rate

    passed  = oos_pf >= 1.0 and oos_wr >= 40.0 and len(r_oos.trades) >= 10
    verdict = "PASS" if passed else "FAIL"

    logger.info(
        f"[OOS] {pair} | IS PF={is_pf:.2f} WR={is_wr:.0f}% ({len(r_is.trades)}t) | "
        f"OOS PF={oos_pf:.2f} WR={oos_wr:.0f}% ({len(r_oos.trades)}t) | "
        f"Score={best_score} | {verdict}"
    )

    return OOSResult(
        pair=pair, days=days,
        is_pf=round(is_pf, 2),   oos_pf=round(oos_pf, 2),
        is_wr=round(is_wr, 1),   oos_wr=round(oos_wr, 1),
        is_trades=len(r_is.trades), oos_trades=len(r_oos.trades),
        best_score=best_score,
        passed=passed, verdict=verdict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA OOS Profitability Check")
    parser.add_argument("pair",     type=str)
    parser.add_argument("--days",   type=int,   default=180)
    parser.add_argument("--risk",   type=float, default=1.0)
    args = parser.parse_args()

    from data.mt5_feed import feed
    feed.connect()

    result = run_oos_check(args.pair, days=args.days, risk_pct=args.risk)
    print(f"\n=== OOS CHECK: {result.pair} ===")
    print(f"IS:  PF={result.is_pf:.2f}  WR={result.is_wr:.0f}%  Trades={result.is_trades}")
    print(f"OOS: PF={result.oos_pf:.2f}  WR={result.oos_wr:.0f}%  Trades={result.oos_trades}")
    print(f"Best min_score: {result.best_score}")
    print(f"VERDICT: {result.verdict}")
