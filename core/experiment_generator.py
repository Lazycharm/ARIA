"""
Future Experiment Generator — learning engine suggests next hypothesis to test.

Reads pattern library, mistake detector, and attribution results to synthesize
actionable hypotheses and push them into the hypothesis queue.

Cost: $0 — fully rules-based.
"""

from __future__ import annotations

from loguru import logger


def generate_next_experiments(max_hypotheses: int = 3) -> list[str]:
    """
    Analyse current trading data and generate the next batch of hypotheses.
    Returns list of hypothesis IDs added to the queue.

    Strategy:
    1. Best-session hypothesis — test trading only in best-performing session
    2. Score-floor raise — raise min_score for pairs with recent WR < 45%
    3. Feature pruning — disable lowest-delta ML features to reduce noise
    4. Pair exclusion — drop persistently unprofitable pairs
    5. High-volatility filter — add ATR filter before entry
    """
    try:
        from core.pattern_library import get_patterns
        from core.mistake_detector import detect_mistakes
        from core.performance_attribution import compute_attribution
        from core.hypothesis_queue import add, get_pending
        from core.adaptive_learning import adaptive
    except ImportError as e:
        logger.warning(f"[ExperimentGenerator] Import failed: {e}")
        return []

    generated: list[str] = []
    patterns    = get_patterns()
    mistakes    = detect_mistakes(recent_n=100)
    attrs       = compute_attribution(min_samples=30)
    al_stats    = adaptive.all_stats()
    by_pair     = patterns.get("by_pair", {})
    by_session  = patterns.get("by_session", {})

    # ── 1. Best-session filter hypothesis ───────────────────────────────────
    if by_session:
        best_sess = max(by_session.items(), key=lambda x: x[1].get("wr", 0))
        worst_sess = min(by_session.items(), key=lambda x: x[1].get("wr", 0))
        if best_sess[1].get("wr", 0) - worst_sess[1].get("wr", 0) >= 15:
            hyp_id = add(
                pair="ALL",
                title=f"Restrict trading to {best_sess[0].capitalize()} session only",
                hypothesis=(
                    f"IF we only trade during the {best_sess[0].capitalize()} session "
                    f"(WR={best_sess[1]['wr']:.0f}%), THEN win rate improves by avoiding "
                    f"the {worst_sess[0].capitalize()} session (WR={worst_sess[1]['wr']:.0f}%). "
                    f"Edge: +{best_sess[1]['wr']-worst_sess[1]['wr']:.0f}pp"
                ),
                source="learning_engine",
                params={"allowed_sessions": [best_sess[0]], "blocked_sessions": [worst_sess[0]]},
                priority=2,
            )
            generated.append(hyp_id)
            if len(generated) >= max_hypotheses:
                return generated

    # ── 2. Score floor raise for underperforming pairs ───────────────────────
    for pair, stats in al_stats.items():
        if stats.wins + stats.losses >= 20 and stats.win_rate < 45.0:
            new_score = min(stats.min_score + 5, 88)
            hyp_id = add(
                pair=pair,
                title=f"Raise min_score from {stats.min_score:.0f} to {new_score:.0f} for {pair}",
                hypothesis=(
                    f"IF we raise the entry threshold for {pair} from {stats.min_score:.0f} "
                    f"to {new_score:.0f}, THEN the WR (currently {stats.win_rate:.0f}%) should "
                    f"improve by filtering low-quality setups. "
                    f"Sample: {stats.wins}W/{stats.losses}L."
                ),
                source="learning_engine",
                params={"pair": pair, "min_score": new_score, "risk_pct": 0.01},
                priority=2,
            )
            generated.append(hyp_id)
            if len(generated) >= max_hypotheses:
                return generated

    # ── 3. Feature pruning hypothesis ───────────────────────────────────────
    noise_features = [a["feature"] for a in attrs if a["verdict"] == "noise"][:3]
    inverse_features = [a["feature"] for a in attrs if a["verdict"] == "inverse"][:2]
    if noise_features or inverse_features:
        hyp_id = add(
            pair="ALL",
            title=f"Prune {len(noise_features)} noise features from confluence scorer",
            hypothesis=(
                f"IF we remove noise features ({', '.join(noise_features[:3])}) and "
                f"downweight inverse features ({', '.join(inverse_features[:2])}), "
                f"THEN ML accuracy and WR should improve. "
                f"Attribution: {len(attrs)} features analysed."
            ),
            source="learning_engine",
            params={
                "remove_features": noise_features,
                "downweight_features": inverse_features,
            },
            priority=3,
        )
        generated.append(hyp_id)
        if len(generated) >= max_hypotheses:
            return generated

    # ── 4. Pair exclusion hypothesis ────────────────────────────────────────
    for pair, ps in by_pair.items():
        if ps.get("trades", 0) >= 30 and ps.get("wr", 50) < 38:
            hyp_id = add(
                pair=pair,
                title=f"Remove {pair} from watchlist — persistent underperformer",
                hypothesis=(
                    f"IF we remove {pair} from the active watchlist "
                    f"(all-time WR={ps['wr']:.0f}%, {ps['trades']} trades), "
                    f"THEN portfolio WR improves by eliminating a drag pair. "
                    f"Total P&L contribution: ${ps.get('total_pnl', 0):+.2f}."
                ),
                source="learning_engine",
                params={"remove_pair": pair},
                priority=1,
            )
            generated.append(hyp_id)
            if len(generated) >= max_hypotheses:
                return generated

    # ── 5. High-stakes mistake hypothesis ───────────────────────────────────
    high_mistakes = [m for m in mistakes if m["severity"] == "high"]
    if high_mistakes:
        m = high_mistakes[0]
        hyp_id = add(
            pair="ALL",
            title=f"Fix: {m['type'].replace('_', ' ').title()}",
            hypothesis=(
                f"Systematic mistake detected: {m['message']}. "
                f"IF we add a guard for this pattern, "
                f"THEN we eliminate this error category. "
                f"Stat: {m['stat']}"
            ),
            source="mistake_detector",
            params={"mistake_type": m["type"]},
            priority=1,
        )
        generated.append(hyp_id)

    if generated:
        logger.info(f"[ExperimentGenerator] Generated {len(generated)} new hypothesis/hypotheses")
    else:
        logger.debug("[ExperimentGenerator] No new hypotheses — conditions not met")

    return generated
