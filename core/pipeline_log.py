"""
Pipeline run log — persists the result of each autonomous pipeline step.

Appends to db/pipeline_log.jsonl (one JSON object per line).
Readable by the dashboard panel.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PATH = Path("db/pipeline_log.jsonl")
_LOCK = threading.Lock()
_MAX_ENTRIES = 100


def record(result: dict) -> None:
    """Append one pipeline step result to the log."""
    entry = {
        "ts":                  datetime.now(timezone.utc).isoformat(),
        "hypothesis_processed": result.get("hypothesis_processed"),
        "approved":            result.get("approved", False),
        "reason":              result.get("reason", ""),
    }
    with _LOCK:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        _trim()


def _trim() -> None:
    """Keep only the last _MAX_ENTRIES lines (called inside lock)."""
    if not _PATH.exists():
        return
    lines = _PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) > _MAX_ENTRIES:
        _PATH.write_text("\n".join(lines[-_MAX_ENTRIES:]) + "\n", encoding="utf-8")


def get_recent(limit: int = 20) -> list[dict]:
    """Return last `limit` pipeline step results, newest first."""
    with _LOCK:
        if not _PATH.exists():
            return []
        lines = _PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
        if len(out) >= limit:
            break
    return out
