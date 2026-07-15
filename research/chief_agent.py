"""
Phase 3 & 13 — Chief Research Agent.

Orchestrates all research sources, deduplication, AI enrichment,
and insertion into the hypothesis queue.

Weekly flow:
  1. Scrape all sources (ForexFactory, MQL5, GitHub, arXiv, quant blogs)
  2. Deduplicate against seen + existing queue
  3. AI enrichment (Haiku): extract pair + signal logic + expected edge
  4. Write Obsidian research notes
  5. Insert unique ideas into hypothesis queue (priority 2)
  6. Telegram alert: N new hypotheses generated

Run manually:  python -m research.chief_agent
Scheduled:     Sunday 02:00 UTC (added to scheduler/tasks.py)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger


def _ai_enrich(idea: dict[str, Any], client) -> dict[str, Any]:
    """Use Haiku to extract structured fields from a raw idea."""
    prompt = f"""You are a quantitative trading research analyst.

Given this research idea:
Title: {idea['title']}
Summary: {idea['summary']}
Source: {idea['source']}

Extract and return a JSON object with these fields:
- "pair": best matching forex pair (e.g. "EURUSDm") or null
- "signal_logic": one-sentence description of the entry signal
- "expected_edge": why this should work (market microstructure reasoning)
- "hypothesis": full testable hypothesis in one paragraph
- "priority": integer 1-5 (5 = highest conviction)
- "tags": list of relevant tags

Return ONLY the JSON, no other text."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = msg.content[0].text.strip()
        data = json.loads(raw)
        idea.update(data)
    except Exception as e:
        logger.debug(f"[ChiefAgent] AI enrichment failed for '{idea['title'][:40]}': {e}")
    return idea


def run_research_cycle(max_per_source: int = 6, ai_enrich: bool = True) -> list[str]:
    """
    Full research cycle: scrape → dedup → enrich → note → queue.
    Returns list of hypothesis IDs added to the queue.
    """
    from research.scrapers import scrape_all
    from research.dedup   import deduplicate
    from research.note_writer import write_all_notes

    logger.info("[ChiefAgent] Starting research cycle …")

    # 1. Scrape
    raw = scrape_all(max_per_source=max_per_source)
    if not raw:
        logger.info("[ChiefAgent] No raw ideas scraped")
        return []

    # 2. Deduplicate
    unique = deduplicate(raw)
    if not unique:
        logger.info("[ChiefAgent] All ideas already seen — nothing new")
        return []

    # 3. AI enrichment (optional — needs ANTHROPIC_API_KEY)
    if ai_enrich:
        try:
            import anthropic
            client = anthropic.Anthropic()
            for i, idea in enumerate(unique):
                unique[i] = _ai_enrich(idea, client)
        except Exception as e:
            logger.warning(f"[ChiefAgent] AI enrichment skipped: {e}")

    # 4. Write Obsidian research notes
    write_all_notes(unique)

    # 5. Insert into hypothesis queue
    hyp_ids: list[str] = []
    try:
        from core.hypothesis_queue import add as hq_add
        pairs_cycle = [
            "EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm",
            "AUDUSDm", "USDCADm",
        ]

        for i, idea in enumerate(unique):
            pair = idea.get("pair") or pairs_cycle[i % len(pairs_cycle)]
            title    = idea.get("title", "Research idea")[:120]
            hyp_text = idea.get("hypothesis") or idea.get("summary", "")[:400]
            priority = int(idea.get("priority") or 2)
            source   = idea.get("source", "research")
            params   = {
                "signal_logic":   idea.get("signal_logic", ""),
                "expected_edge":  idea.get("expected_edge", ""),
                "source_url":     idea.get("url", ""),
            }

            hid = hq_add(
                pair=pair,
                title=title,
                hypothesis=hyp_text,
                source=source,
                params=params,
                priority=priority,
            )
            hyp_ids.append(hid)
            logger.info(f"[ChiefAgent] Added hypothesis {hid}: {title[:60]}")
    except Exception as e:
        logger.error(f"[ChiefAgent] Hypothesis queue insert failed: {e}")

    # 6. Telegram alert
    if hyp_ids:
        try:
            from notifications.telegram import alert_hypothesis_generated
            top = unique[0]
            alert_hypothesis_generated(
                pair=top.get("pair") or "ALL",
                hypothesis_id=hyp_ids[0] if hyp_ids else "??",
                title=top.get("title", "")[:80],
                source=top.get("source", "research"),
            )
        except Exception:
            pass

    logger.info(f"[ChiefAgent] Cycle complete: {len(hyp_ids)} new hypotheses queued")
    return hyp_ids


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    ids = run_research_cycle(max_per_source=3)
    print(f"Generated {len(ids)} hypotheses: {ids}")
