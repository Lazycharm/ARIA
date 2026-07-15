"""
Phase 4 — Walk-Forward Optimization (WFO).

Prevents strategy overfitting by splitting historical data into repeated
In-Sample (IS) + Out-of-Sample (OOS) windows, optimizing on IS, then
validating on OOS. The OOS results are the honest expected performance.

How it works:
  1. Fetch all M15 data for the full period (once)
  2. Slide a window through the data:
       IS window  → grid-search best {min_score, risk_pct}
       OOS window → run with IS-best params, record results
  3. Aggregate OOS: mean PF, WR, DD
  4. Stability score = 1 - std(OOS_PF) / mean(OOS_PF)
     (1.0 = perfectly consistent, 0.0 = all over the place)

Grid searched:
  min_score: 65, 70, 75, 80
  risk_pct:  0.5, 1.0, 1.5

CLI:
  python wfo.py --pair EURUSDm --total 365 --is-days 90 --oos-days 30 --step 30
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestResults

# ── Grid ─────────────────────────────────────────────────────────────────────

# Legacy grid kept as fallback when Optuna is unavailable
_MIN_SCORE_GRID = [65.0, 70.0, 75.0, 80.0]
_RISK_PCT_GRID  = [0.5, 1.0, 1.5]

_OPTUNA_N_TRIALS = 40   # Bayesian search budget per WFO window


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WFOWindow:
    window_idx: int
    is_start_bar: int
    is_end_bar: int
    oos_start_bar: int
    oos_end_bar: int
    best_score: float
    best_risk: float
    is_pf: float              # profit factor on IS with best params
    oos_result: BacktestResults


@dataclass
class WFOSummary:
    pair: str
    total_days: int
    is_days: int
    oos_days: int
    step_days: int
    windows: list[WFOWindow] = field(default_factory=list)

    @property
    def oos_pf_list(self) -> list[float]:
        return [w.oos_result.profit_factor for w in self.windows if w.oos_result.total_trades > 0]

    @property
    def oos_wr_list(self) -> list[float]:
        return [w.oos_result.win_rate for w in self.windows if w.oos_result.total_trades > 0]

    @property
    def oos_dd_list(self) -> list[float]:
        return [w.oos_result.max_drawdown for w in self.windows if w.oos_result.total_trades > 0]

    @property
    def mean_oos_pf(self) -> float:
        pf = self.oos_pf_list
        return sum(pf) / len(pf) if pf else 0.0

    @property
    def mean_oos_wr(self) -> float:
        wr = self.oos_wr_list
        return sum(wr) / len(wr) if wr else 0.0

    @property
    def mean_oos_dd(self) -> float:
        dd = self.oos_dd_list
        return sum(dd) / len(dd) if dd else 0.0

    @property
    def stability_score(self) -> float:
        """1 - CV of OOS profit factors. 1=rock solid, <0=chaotic."""
        pf = self.oos_pf_list
        if len(pf) < 2 or sum(pf) == 0:
            return 0.0
        mean = sum(pf) / len(pf)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in pf) / len(pf)
        std = variance ** 0.5
        return max(0.0, round(1.0 - std / mean, 3))

    @property
    def recommended_params(self) -> tuple[float, float]:
        """Most-frequently selected IS params across all windows."""
        from collections import Counter
        counts: Counter = Counter()
        for w in self.windows:
            counts[(w.best_score, w.best_risk)] += 1
        if not counts:
            return 70.0, 1.0
        return counts.most_common(1)[0][0]

    def stability_chart(self) -> "go.Figure":
        """OOS profit factor per window as an equity-style line chart."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise RuntimeError("plotly required for stability chart")

        xs  = [w.window_idx for w in self.windows if w.oos_result.total_trades > 0]
        pfs = [w.oos_result.profit_factor for w in self.windows if w.oos_result.total_trades > 0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=pfs,
            mode="lines+markers",
            name="OOS Profit Factor",
            line=dict(color="#7C3AED", width=2),
            marker=dict(size=6),
        ))
        fig.add_hline(y=1.0,  line=dict(color="#EF4444", dash="dot", width=1),
                      annotation_text="Break-even")
        fig.add_hline(y=1.2,  line=dict(color="#F59E0B", dash="dot", width=1),
                      annotation_text="Min threshold")
        fig.update_layout(
            title=f"WFO Stability — {self.pair}",
            xaxis_title="Window #", yaxis_title="OOS Profit Factor",
            template="plotly_dark",
            paper_bgcolor="#0A0B0F", plot_bgcolor="#111827",
            margin=dict(l=50, r=10, t=40, b=30),
        )
        return fig

    def print_report(self) -> None:
        sep = "─" * 62
        rec_score, rec_risk = self.recommended_params

        print(f"\n{'═'*62}")
        print(f"  WALK-FORWARD OPTIMIZATION — {self.pair}")
        print(f"  IS={self.is_days}d  OOS={self.oos_days}d  Step={self.step_days}d  "
              f"Windows={len(self.windows)}")
        print(sep)
        print(f"  {'Win':<4} {'IS best':<14} {'IS PF':>6}  →  {'OOS PF':>6} {'OOS WR':>7} {'OOS DD':>7} {'OOS N':>5}")
        print(sep)

        for w in self.windows:
            oos = w.oos_result
            n   = oos.total_trades
            pf  = f"{oos.profit_factor:.2f}" if n > 0 else "  —  "
            wr  = f"{oos.win_rate:.0f}%"     if n > 0 else "  —  "
            dd  = f"{oos.max_drawdown:.1f}%"  if n > 0 else "  —  "
            params = f"score={w.best_score:.0f}/risk={w.best_risk}"
            print(f"  {w.window_idx:<4} {params:<14} {w.is_pf:>6.2f}  →  {pf:>6} {wr:>7} {dd:>7} {n:>5}")

        print(sep)
        valid = len(self.oos_pf_list)
        if valid == 0:
            print("  No OOS windows produced enough trades for analysis.")
        else:
            verdict = self._oos_verdict()
            print(f"  AGGREGATE OOS  ({valid} windows with trades)")
            print(f"  Mean PF     : {self.mean_oos_pf:.2f}")
            print(f"  Mean WR     : {self.mean_oos_wr:.1f}%")
            print(f"  Mean DD     : {self.mean_oos_dd:.1f}%")
            print(f"  Stability   : {self.stability_score:.2f}  (0=unstable, 1=perfect)")
            print(f"  Best params : min_score={rec_score:.0f}  risk={rec_risk}%  (most IS-optimal)")
            print(sep)
            print(f"  {verdict}")

        print(f"{'═'*62}\n")

    def _oos_verdict(self) -> str:
        pf   = self.mean_oos_pf
        dd   = self.mean_oos_dd
        stab = self.stability_score

        issues = []
        if pf < 1.2:
            issues.append(f"OOS profit factor too low ({pf:.2f} < 1.2)")
        if dd > 15:
            issues.append(f"OOS drawdown too high ({dd:.1f}% > 15%)")
        if stab < 0.4:
            issues.append(f"Strategy unstable across windows (stability={stab:.2f} < 0.4)")

        if not issues:
            return f"✅ STABLE — OOS results consistent. Deploy with min_score={self.recommended_params[0]:.0f}, risk={self.recommended_params[1]}%"
        return "❌ UNSTABLE:\n" + "\n".join(f"  • {i}" for i in issues)

    def save_to_obsidian(self) -> None:
        try:
            from config.settings import settings
            vault  = settings.obsidian_vault_path
            folder = settings.obsidian_aria_folder
            if not vault or not folder:
                return

            from pathlib import Path
            from datetime import datetime

            out_dir = Path(vault) / folder / "Backtests"
            out_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
            fname = out_dir / f"{self.pair}_WFO_{self.total_days}d_{ts}.md"

            rec_score, rec_risk = self.recommended_params
            content = f"""# ARIA Walk-Forward Optimization — {self.pair}
*Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## Parameters
- Total days: {self.total_days}
- IS window: {self.is_days}d
- OOS window: {self.oos_days}d
- Step: {self.step_days}d
- Grid: min_score={_MIN_SCORE_GRID}, risk_pct={_RISK_PCT_GRID}

## OOS Aggregate
- Mean PF: {self.mean_oos_pf:.2f}
- Mean WR: {self.mean_oos_wr:.1f}%
- Mean DD: {self.mean_oos_dd:.1f}%
- Stability: {self.stability_score:.2f}
- Recommended: min_score={rec_score:.0f}, risk={rec_risk}%

## Window Results
| # | IS Best | IS PF | OOS PF | OOS WR | OOS DD | OOS N |
|---|---------|-------|--------|--------|--------|-------|
"""
            for w in self.windows:
                oos = w.oos_result
                content += (
                    f"| {w.window_idx} | score={w.best_score:.0f}/risk={w.best_risk} | "
                    f"{w.is_pf:.2f} | {oos.profit_factor:.2f} | {oos.win_rate:.0f}% | "
                    f"{oos.max_drawdown:.1f}% | {oos.total_trades} |\n"
                )

            fname.write_text(content, encoding="utf-8")
            print(f"WFO saved to Obsidian: {fname.name}")
        except Exception as e:
            logger.debug(f"[WFO] Obsidian save failed: {e}")


# ── Optimizer ─────────────────────────────────────────────────────────────────

class WalkForwardOptimizer:
    """
    Runs IS/OOS walk-forward optimization for a single pair.
    Fetches M15 data once, then slices for each window.
    Uses Optuna Bayesian optimization when available, falls back to grid search.

    anchored=True: IS window always starts from bar 0 (expanding IS, fixed start).
    """

    def __init__(
        self,
        pair: str,
        total_days: int = 365,
        is_days: int    = 90,
        oos_days: int   = 30,
        step_days: int  = 30,
        initial_balance: float = 100.0,
        spread_pips: float     = 0.8,
        anchored: bool         = False,
        n_trials: int          = _OPTUNA_N_TRIALS,
    ) -> None:
        self.pair     = pair
        self.total    = total_days
        self.is_d     = is_days
        self.oos_d    = oos_days
        self.step_d   = step_days
        self.balance  = initial_balance
        self.spread   = spread_pips
        self.anchored = anchored
        self.n_trials = n_trials

        candles_per_day = 96  # M15 = 96 candles/day
        self._bars_is   = is_days  * candles_per_day
        self._bars_oos  = oos_days * candles_per_day
        self._bars_step = step_days * candles_per_day

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _optuna_objective(self, trial, df_is: pd.DataFrame) -> float:
        """Maximize IS profit factor via Bayesian parameter search."""
        min_score = trial.suggest_float("min_score", 60.0, 85.0)
        risk_pct  = trial.suggest_float("risk_pct",  0.5,  2.0)
        scan_step = trial.suggest_int("scan_step",   3,    10)

        engine = BacktestEngine(
            pair=self.pair,
            days=self.is_d,
            initial_balance=self.balance,
            risk_pct=risk_pct,
            min_score=min_score,
            spread_pips=self.spread,
            scan_step=scan_step,
        )
        try:
            res = engine.run(df_m15=df_is.copy())
            if res.total_trades < 5:
                return 0.0
            return float(res.profit_factor)
        except Exception:
            return 0.0

    def _optimize_window(self, df_is: pd.DataFrame) -> tuple[float, float, int, float]:
        """
        Return (best_min_score, best_risk_pct, best_scan_step, best_is_pf).
        Tries Optuna first; falls back to grid search.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda t: self._optuna_objective(t, df_is),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )
            p = study.best_params
            return (
                p["min_score"],
                p["risk_pct"],
                p["scan_step"],
                study.best_value,
            )
        except Exception:
            # Fallback: exhaustive grid
            best_pf = -1.0
            best_score, best_risk, best_step = 70.0, 1.0, 5
            for min_score, risk_pct in itertools.product(_MIN_SCORE_GRID, _RISK_PCT_GRID):
                engine = BacktestEngine(
                    pair=self.pair,
                    days=self.is_d,
                    initial_balance=self.balance,
                    risk_pct=risk_pct,
                    min_score=min_score,
                    spread_pips=self.spread,
                )
                try:
                    res = engine.run(df_m15=df_is.copy())
                    if res.total_trades >= 5 and res.profit_factor > best_pf:
                        best_pf    = res.profit_factor
                        best_score = min_score
                        best_risk  = risk_pct
                except Exception:
                    pass
            return best_score, best_risk, best_step, max(best_pf, 0.0)

    def run(self) -> WFOSummary:
        mode = "anchored" if self.anchored else "rolling"
        logger.info(f"[WFO] {self.pair} — total={self.total}d IS={self.is_d}d OOS={self.oos_d}d "
                    f"step={self.step_d}d mode={mode} trials={self.n_trials}")

        df = self._fetch_all()
        if df.empty or len(df) < self._bars_is + self._bars_oos + 200:
            logger.error(f"[WFO] Not enough data for {self.pair}")
            return WFOSummary(self.pair, self.total, self.is_d, self.oos_d, self.step_d)

        summary = WFOSummary(self.pair, self.total, self.is_d, self.oos_d, self.step_d)

        window_idx = 1
        oos_start  = self._bars_is  # OOS always begins after first IS window

        while True:
            oos_end = oos_start + self._bars_oos
            if oos_end > len(df):
                break

            if self.anchored:
                # IS always from bar 0, expanding
                is_start = 0
                is_end   = oos_start
            else:
                # Rolling IS window
                is_start = oos_start - self._bars_is
                is_end   = oos_start

            df_is  = df.iloc[is_start:is_end]
            df_oos = df.iloc[oos_start:oos_end]

            if len(df_is) < 250 or len(df_oos) < 50:
                oos_start += self._bars_step
                continue

            print(f"  Window {window_idx} ({mode}): IS [{is_start}:{is_end}] OOS [{oos_start}:{oos_end}] … ",
                  end="", flush=True)

            best_score, best_risk, best_step, is_pf = self._optimize_window(df_is)

            oos_engine = BacktestEngine(
                pair=self.pair,
                days=self.oos_d,
                initial_balance=self.balance,
                risk_pct=best_risk,
                min_score=best_score,
                spread_pips=self.spread,
                scan_step=best_step,
            )
            try:
                oos_result = oos_engine.run(df_m15=df_oos.copy())
            except Exception as e:
                logger.debug(f"[WFO] OOS run error: {e}")
                oos_result = BacktestResults(self.pair, self.oos_d, self.balance, self.balance)

            print(f"OOS PF={oos_result.profit_factor:.2f} ({oos_result.total_trades} trades)")

            summary.windows.append(WFOWindow(
                window_idx=window_idx,
                is_start_bar=is_start,
                is_end_bar=is_end,
                oos_start_bar=oos_start,
                oos_end_bar=oos_end,
                best_score=best_score,
                best_risk=best_risk,
                is_pf=is_pf,
                oos_result=oos_result,
            ))

            oos_start  += self._bars_step
            window_idx += 1

        return summary

    def _fetch_all(self) -> pd.DataFrame:
        from data.mt5_feed import feed
        count = self.total * 96 + 250
        return feed.get_candles(self.pair, "M15", count=count)
