"""
Phase 13 — Continuous Autonomous Research Pipeline.

Full loop:
  1. Research cadence  → ChiefAgent generates new hypotheses (weekly)
  2. Hypothesis queue  → pick next pending hypothesis
  3. Backtest          → run full backtest on hypothesis params
  4. WFO               → walk-forward validation (if backtest passes)
  5. MC gate           → Monte Carlo ruin check (blocks if ruin ≥ 5%)
  6. Auto-approve gate → if all 3 pass → enter paper trading
  7. Paper trading     → 2-week shadow execution with live prices
  8. Promotion check   → positive PnL + Sharpe > 0 → promote to live
  9. Strategy monitoring → live alert if rolling Sharpe drops below 0.5
  10. Auto-retirement   → disable + notify + write to Rejected Strategies/

Entry point: run_pipeline_step() — call from scheduler or manually.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger


# ── Stage 3: Backtest ─────────────────────────────────────────────────────────

def _run_backtest(pair: str, hyp_id: str, params: dict) -> Optional[dict]:
    """Run backtest for a hypothesis. Returns summary dict or None on failure."""
    try:
        from backtest.engine import BacktestEngine
        min_score = params.get("min_score", 70.0)
        risk_pct  = params.get("risk_pct",  1.0)
        days      = params.get("days",       90)

        engine = BacktestEngine(
            pair=pair,
            days=days,
            min_score=min_score,
            risk_pct=risk_pct,
        )
        res = engine.run()

        summary = {
            "trades":        res.total_trades,
            "win_rate":      round(res.win_rate, 1),
            "profit_factor": round(res.profit_factor, 2),
            "max_drawdown":  round(res.max_drawdown, 1),
            "net_pnl":       round(res.net_pnl, 2),
            "verdict":       res.verdict(),
            "passed":        ("✅" in res.verdict()),
        }
        logger.info(f"[Pipeline] Backtest {hyp_id}: PF={summary['profit_factor']} WR={summary['win_rate']}% "
                    f"{'PASS' if summary['passed'] else 'FAIL'}")

        # Telegram
        try:
            from notifications.telegram import alert_backtest_complete
            alert_backtest_complete(
                pair=pair,
                days=days,
                trades=res.total_trades,
                win_rate=res.win_rate,
                profit_factor=res.profit_factor,
                max_dd=res.max_drawdown,
                verdict=res.verdict()[:100],
            )
        except Exception:
            pass

        return summary
    except Exception as e:
        logger.error(f"[Pipeline] Backtest failed for {hyp_id}: {e}")
        return None


# ── Stage 4: WFO ─────────────────────────────────────────────────────────────

def _run_wfo(pair: str, hyp_id: str) -> Optional[dict]:
    """Run WFO on the pair. Returns summary dict or None."""
    try:
        from backtest.wfo import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(pair=pair, total_days=365, is_days=90, oos_days=30)
        summary = wfo.run()
        passed  = summary.mean_oos_pf >= 1.2 and summary.stability_score >= 0.4

        logger.info(f"[Pipeline] WFO {hyp_id}: mean_pf={summary.mean_oos_pf:.2f} "
                    f"stability={summary.stability_score:.2f} {'PASS' if passed else 'FAIL'}")
        return {"mean_oos_pf": summary.mean_oos_pf, "stability": summary.stability_score, "passed": passed}
    except Exception as e:
        logger.error(f"[Pipeline] WFO failed for {hyp_id}: {e}")
        return None


# ── Stage 5: Monte Carlo gate ─────────────────────────────────────────────────

def _run_mc_gate(pair: str, hyp_id: str) -> Optional[dict]:
    """Run Monte Carlo gate check. Returns {allowed, ruin_pct, reason}."""
    try:
        from backtest.mc_gate import check_mc_gate
        result = check_mc_gate(pair, days=90)
        logger.info(f"[Pipeline] MC gate {hyp_id}: ruin={result.ruin_pct:.1f}% "
                    f"{'PASS' if result.allowed else 'BLOCK'}")
        return {"allowed": result.allowed, "ruin_pct": result.ruin_pct, "reason": result.reason}
    except Exception as e:
        logger.error(f"[Pipeline] MC gate failed for {hyp_id}: {e}")
        return None


# ── Stage 6: Auto-approve gate ────────────────────────────────────────────────

def _auto_approve(
    hyp_id: str,
    pair: str,
    bt: dict,
    wfo: dict,
    mc: dict,
) -> bool:
    """All 3 validation stages must pass for auto-approve."""
    if not bt.get("passed"):
        _reject(hyp_id, pair, f"Backtest failed: {bt.get('verdict','')[:80]}")
        return False
    if not wfo.get("passed"):
        _reject(hyp_id, pair, f"WFO failed: OOS PF={wfo.get('mean_oos_pf',0):.2f} stability={wfo.get('stability',0):.2f}")
        return False
    if not mc.get("allowed"):
        _reject(hyp_id, pair, f"MC gate blocked: ruin={mc.get('ruin_pct',100):.1f}% ≥ 5%")
        return False

    logger.info(f"[Pipeline] {hyp_id} AUTO-APPROVED → entering paper trading")
    return True


def _reject(hyp_id: str, pair: str, reason: str) -> None:
    try:
        from core.hypothesis_queue import mark_rejected
        mark_rejected(hyp_id, result={"reason": reason, "auto": True})
    except Exception:
        pass
    logger.info(f"[Pipeline] {hyp_id} REJECTED: {reason}")


# ── Stage 7: Paper trading ────────────────────────────────────────────────────

def _enter_paper_trading(pair: str, hyp_id: str, strategy: str) -> None:
    """Mark hypothesis as accepted and begin paper monitoring."""
    try:
        from core.hypothesis_queue import mark_accepted
        mark_accepted(hyp_id, result={"status": "paper_trading", "strategy": strategy})
    except Exception:
        pass
    logger.info(f"[Pipeline] {hyp_id} → paper trading started for {pair} ({strategy})")


# ── Stage 9: Strategy monitoring ─────────────────────────────────────────────

def monitor_live_strategies(sharpe_floor: float = 0.5) -> list[str]:
    """
    Check all active live strategies. Alert if rolling Sharpe < sharpe_floor.
    Returns list of strategies that triggered an alert.
    """
    alerted: list[str] = []
    try:
        from core.strategy_scorer import score_strategies
        strategies = score_strategies()
        for s in strategies:
            if s["sharpe"] < sharpe_floor and not s.get("retire"):
                msg = (f"[Monitor] Strategy {s['strategy']} rolling Sharpe dropped to "
                       f"{s['sharpe']:.2f} (< {sharpe_floor}) — review required")
                logger.warning(msg)
                try:
                    from notifications.telegram import send
                    send(f"⚠️ Strategy Alert\n{msg}")
                except Exception:
                    pass
                alerted.append(s["strategy"])
    except Exception as e:
        logger.debug(f"[Monitor] Strategy monitoring error: {e}")
    return alerted


# ── Stage 10: Auto-retirement ─────────────────────────────────────────────────

def auto_retire_strategies() -> list[str]:
    """
    Auto-retire strategies with rolling Sharpe < 0 (below floor).
    Writes to Rejected Strategies/ in Obsidian.
    Returns list of retired strategy names.
    """
    retired: list[str] = []
    try:
        from core.strategy_scorer import get_disabled_strategies, score_strategies
        disabled = get_disabled_strategies()
        for name in disabled:
            # Write retirement note to Obsidian
            try:
                from config.settings import settings
                from pathlib import Path
                if settings.obsidian_vault_path and settings.obsidian_aria_folder:
                    out = (Path(settings.obsidian_vault_path) / settings.obsidian_aria_folder
                           / "Rejected Strategies" / f"RETIRED_{name}_{datetime.now().strftime('%Y%m%d')}.md")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        f"# Retired Strategy: {name}\n"
                        f"*Auto-retired: {datetime.now().isoformat()}*\n\n"
                        "Rolling 30-trade Sharpe dropped below 0.\n"
                        "Strategy has been disabled from live trading.\n",
                        encoding="utf-8",
                    )
            except Exception:
                pass

            # Telegram notification
            try:
                from notifications.telegram import send
                send(f"🔴 Strategy Retired\n{name} has been auto-retired (Sharpe < 0 over 30 trades)")
            except Exception:
                pass

            retired.append(name)
            logger.info(f"[AutoRetire] Retired strategy: {name}")

    except Exception as e:
        logger.debug(f"[AutoRetire] Error: {e}")
    return retired


# ── Stage 8: Paper → Live promotion ──────────────────────────────────────────

def check_paper_promotions() -> list[str]:
    """
    Check all strategies in paper trading mode.
    Promote to live if conditions met.
    Returns list of promoted strategies.
    """
    promoted: list[str] = []
    try:
        from core.paper_trader import should_promote, paper_performance
        for strategy in ["SMC_TREND", "SESSION_BREAKOUT", "MEAN_REVERSION", "RANGE_TRADING"]:
            ok, reason = should_promote(strategy)
            if ok:
                logger.info(f"[Pipeline] PROMOTE {strategy} to live: {reason}")
                try:
                    from notifications.telegram import send
                    send(f"✅ Strategy Promoted to Live\n{strategy}\n{reason}")
                except Exception:
                    pass
                promoted.append(strategy)
    except Exception as e:
        logger.debug(f"[Pipeline] Promotion check error: {e}")
    return promoted


# ── Main pipeline step ────────────────────────────────────────────────────────

def run_pipeline_step() -> dict:
    """
    Run one step of the autonomous pipeline:
      1. Pick next pending hypothesis
      2. Run backtest → WFO → MC gate
      3. Auto-approve → paper trading
      4. Also check paper promotions and monitor live strategies

    Returns summary of what happened.
    """
    result = {"hypothesis_processed": None, "approved": False, "reason": ""}

    # Pick next hypothesis
    try:
        from core.hypothesis_queue import get_next
        hyp = get_next()
        if not hyp:
            result["reason"] = "No pending hypotheses"
            return result

        hyp_id = hyp["id"]
        pair   = hyp["pair"]
        params = hyp.get("params", {})
        strategy = params.get("strategy", "SMC_TREND")
        result["hypothesis_processed"] = hyp_id

        logger.info(f"[Pipeline] Processing hypothesis {hyp_id}: {hyp['title'][:60]}")

        # Mark as running
        from core.hypothesis_queue import mark_running
        mark_running(hyp_id)

    except Exception as e:
        result["reason"] = f"Queue error: {e}"
        return result

    # Stage 3: Backtest
    bt = _run_backtest(pair, hyp_id, params)
    if not bt:
        _reject(hyp_id, pair, "Backtest engine error")
        result["reason"] = "Backtest failed"
        return result

    if not bt["passed"]:
        _reject(hyp_id, pair, bt["verdict"][:100])
        result["reason"] = f"Backtest FAIL: {bt['verdict'][:60]}"
        return result

    # Stage 4: WFO
    wfo = _run_wfo(pair, hyp_id)
    if not wfo:
        _reject(hyp_id, pair, "WFO engine error")
        result["reason"] = "WFO failed"
        return result

    # Stage 5: MC gate
    mc = _run_mc_gate(pair, hyp_id)
    if not mc:
        _reject(hyp_id, pair, "MC gate error")
        result["reason"] = "MC gate error"
        return result

    # Stage 6: Auto-approve
    approved = _auto_approve(hyp_id, pair, bt, wfo, mc)
    result["approved"] = approved

    if approved:
        _enter_paper_trading(pair, hyp_id, strategy)
        result["reason"] = "All gates passed → paper trading started"
    else:
        result["reason"] = "One or more validation gates failed"

    # Stage 8: Check promotions
    check_paper_promotions()

    # Stage 9: Monitor live strategies
    monitor_live_strategies()

    # Stage 10: Auto-retire
    auto_retire_strategies()

    return result


# ── Parameter suggestion executor (Phase 4) ──────────────────────────────────

def apply_parameter_suggestion(pair: str, suggestion: dict) -> Optional[str]:
    """
    Apply an AI parameter suggestion by creating a hypothesis and queuing it.
    suggestion: {min_score, risk_pct, reason, source}
    Returns hypothesis ID or None.
    """
    try:
        from core.hypothesis_queue import add as hq_add
        title = f"AI Suggestion: {suggestion.get('reason', 'Parameter adjustment')[:80]}"
        hyp   = (f"Adjust {pair} parameters based on backtest AI analysis. "
                 f"min_score={suggestion.get('min_score', 70)}, "
                 f"risk_pct={suggestion.get('risk_pct', 1.0)}")
        params = {
            "min_score":  suggestion.get("min_score", 70),
            "risk_pct":   suggestion.get("risk_pct",  1.0),
            "strategy":   suggestion.get("strategy",  "SMC_TREND"),
            "auto_triggered": True,
        }
        hid = hq_add(
            pair=pair,
            title=title,
            hypothesis=hyp,
            source=suggestion.get("source", "ai_analysis"),
            params=params,
            priority=3,
        )
        logger.info(f"[Pipeline] Parameter suggestion queued as hypothesis {hid}")
        return hid
    except Exception as e:
        logger.error(f"[Pipeline] Parameter suggestion failed: {e}")
        return None
