"""
ARIA AI Brain — cost-aware LLM routing.

Routing (cheapest-first):
  rules   → no AI call ($0)
  haiku   → signals, briefs, pre-session (~$0.001/call)
  sonnet  → daily report, once/day (~$0.05/call)
  fable5  → manual deep analysis only, NEVER auto-scheduled
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from loguru import logger

from config.settings import settings

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _call(model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def brief(prompt: str, context: str = "") -> str:
    """Haiku call — hourly commentary, cheap."""
    msg = f"{context}\n\n{prompt}" if context else prompt
    logger.debug("Brain.brief (haiku)")
    return _call(settings.model_brief, "You are ARIA, a concise FX market analyst.", msg, 512)


def session_analysis(prompt: str) -> str:
    """Haiku call — pre-session level marking (once/day at 06:30 UTC)."""
    logger.info("Brain.session_analysis (haiku)")
    return _call(
        settings.model_session_analysis,
        (
            "You are ARIA, a professional FX analyst. Identify key support/resistance levels, "
            "order blocks, and session bias. Be precise and concise."
        ),
        prompt,
        1024,
    )


def daily_report(prompt: str) -> str:
    """Sonnet call — end-of-day report (once/day at NY close)."""
    logger.info("Brain.daily_report (sonnet)")
    return _call(
        settings.model_daily_report,
        (
            "You are ARIA, an institutional FX trading analyst. Generate a professional "
            "end-of-day trading report with P&L analysis, trade review, and next-day prep."
        ),
        prompt,
        2048,
    )


def deep_analysis(prompt: str) -> str:
    """Fable 5 call — manual trigger only. NEVER call from scheduled tasks."""
    logger.warning("Brain.deep_analysis (fable-5) — manual trigger")
    return _call(
        settings.model_deep_analysis,
        (
            "You are ARIA, operating as a Goldman Sachs-level quantitative analyst. "
            "Perform deep institutional-grade market analysis with macro, technical, "
            "and sentiment synthesis."
        ),
        prompt,
        4096,
    )


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model response with fallback."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
