"""
Hypothesis Queue — structured backlog of strategy ideas awaiting backtesting.

Stores hypotheses as JSONL in db/hypothesis_queue.jsonl.
Each hypothesis has:
  id          — UUID4 short (8 chars)
  pair        — e.g. EURUSDm
  title       — one-line description
  hypothesis  — full text: IF <condition> THEN <edge>
  source      — reddit | research | ai | manual
  params      — dict of suggested backtest params
  param_hash  — MD5 of sorted(params) for exact reproduction
  status      — pending | running | accepted | rejected
  created_at  — ISO timestamp
  tested_at   — ISO timestamp (set on test)
  result      — brief outcome string (set on test)
  priority    — 1 (high) to 5 (low), default 3
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from loguru import logger

_QUEUE_PATH = Path("./db/hypothesis_queue.jsonl")
_lock = Lock()


def _load_all() -> list[dict]:
    if not _QUEUE_PATH.exists():
        return []
    rows = []
    for line in _QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _save_all(rows: list[dict]) -> None:
    _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _QUEUE_PATH.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _param_hash(params: dict) -> str:
    canonical = json.dumps(dict(sorted(params.items())), sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


def add(
    pair: str,
    title: str,
    hypothesis: str,
    source: str,
    params: dict | None = None,
    priority: int = 3,
) -> str:
    """
    Add a new hypothesis to the queue.
    Returns the assigned hypothesis ID.
    """
    hyp_id = uuid.uuid4().hex[:8].upper()
    params  = params or {}
    entry = {
        "id":         hyp_id,
        "pair":       pair,
        "title":      title,
        "hypothesis": hypothesis,
        "source":     source,
        "params":     params,
        "param_hash": _param_hash(params),
        "status":     "pending",
        "priority":   priority,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tested_at":  None,
        "result":     None,
    }
    with _lock:
        rows = _load_all()
        rows.append(entry)
        _save_all(rows)
    logger.info(f"[HypothesisQueue] Added {hyp_id}: {title[:60]} ({pair}, source={source})")

    # Write to Obsidian
    _write_to_obsidian(entry)

    return hyp_id


def get_pending(pair: str | None = None) -> list[dict]:
    """Return pending hypotheses sorted by priority (1=highest first)."""
    rows = _load_all()
    pending = [r for r in rows if r["status"] == "pending"]
    if pair:
        pending = [r for r in pending if r["pair"] == pair]
    return sorted(pending, key=lambda x: x.get("priority", 3))


def get_next() -> dict | None:
    """Return the highest-priority pending hypothesis."""
    pending = get_pending()
    return pending[0] if pending else None


def mark_running(hyp_id: str) -> None:
    with _lock:
        rows = _load_all()
        for r in rows:
            if r["id"] == hyp_id:
                r["status"] = "running"
        _save_all(rows)


def mark_accepted(hyp_id: str, result: str) -> None:
    with _lock:
        rows = _load_all()
        for r in rows:
            if r["id"] == hyp_id:
                r["status"] = "accepted"
                r["tested_at"] = datetime.now(timezone.utc).isoformat()
                r["result"]    = result
        _save_all(rows)
    logger.info(f"[HypothesisQueue] Accepted {hyp_id}: {result[:60]}")
    _write_accepted_to_obsidian(hyp_id, result)


def mark_rejected(hyp_id: str, result: str) -> None:
    with _lock:
        rows = _load_all()
        for r in rows:
            if r["id"] == hyp_id:
                r["status"]    = "rejected"
                r["tested_at"] = datetime.now(timezone.utc).isoformat()
                r["result"]    = result
        _save_all(rows)
    logger.info(f"[HypothesisQueue] Rejected {hyp_id}: {result[:60]}")

    # Save rejected hypotheses to Obsidian
    all_rows = _load_all()
    entry = next((r for r in all_rows if r["id"] == hyp_id), None)
    if entry:
        _write_rejected_to_obsidian(entry, result)


def stats() -> dict:
    rows = _load_all()
    total    = len(rows)
    pending  = sum(1 for r in rows if r["status"] == "pending")
    accepted = sum(1 for r in rows if r["status"] == "accepted")
    rejected = sum(1 for r in rows if r["status"] == "rejected")
    running  = sum(1 for r in rows if r["status"] == "running")
    return {
        "total": total, "pending": pending,
        "running": running, "accepted": accepted, "rejected": rejected,
    }


def _write_to_obsidian(entry: dict) -> None:
    try:
        from config.settings import settings
        folder = settings.obsidian_vault_path / settings.obsidian_aria_folder / "Hypotheses"
        path   = folder / f"{entry['id']}-{entry['pair'].rstrip('m')}.md"
        folder.mkdir(parents=True, exist_ok=True)
        content = f"""---
tags: [ARIA, hypothesis, {entry['pair']}, {entry['status']}]
id: {entry['id']}
pair: {entry['pair']}
source: {entry['source']}
status: {entry['status']}
priority: {entry['priority']}
created: {entry['created_at'][:10]}
param_hash: {entry['param_hash']}
---

# Hypothesis {entry['id']} — {entry['title']}

**Pair:** {entry['pair'].rstrip('m')}
**Source:** {entry['source']}
**Priority:** {entry['priority']}/5

## Hypothesis

{entry['hypothesis']}

## Parameters

```json
{json.dumps(entry['params'], indent=2)}
```

## Status

`{entry['status'].upper()}`

---
*[[ARIA Overview]] | Auto-generated by ARIA hypothesis queue*
"""
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.debug(f"Hypothesis Obsidian write failed: {e}")


def _write_accepted_to_obsidian(hyp_id: str, result: str) -> None:
    try:
        from config.settings import settings
        folder = settings.obsidian_vault_path / settings.obsidian_aria_folder / "Accepted Strategies"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{hyp_id}.md"
        path.write_text(
            f"# Accepted Strategy {hyp_id}\n\n**Result:** {result}\n\n"
            f"*[[ARIA Overview]]*\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _write_rejected_to_obsidian(entry: dict, result: str) -> None:
    try:
        from config.settings import settings
        folder = settings.obsidian_vault_path / settings.obsidian_aria_folder / "Rejected Strategies"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{entry['id']}-{entry['pair'].rstrip('m')}.md"
        content = f"""---
tags: [ARIA, rejected, {entry['pair']}]
id: {entry['id']}
pair: {entry['pair']}
tested: {entry.get('tested_at', '')[:10]}
---

# Rejected Hypothesis — {entry['title']}

**Reason:** {result}

## Original Hypothesis

{entry['hypothesis']}

## Parameters Tested

```json
{json.dumps(entry['params'], indent=2)}
```

---
*[[ARIA Overview]] | Filed by ARIA rejection gate*
"""
        path.write_text(content, encoding="utf-8")
    except Exception:
        pass
