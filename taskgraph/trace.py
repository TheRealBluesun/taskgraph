"""Reading omp's JSON trace files (SPEC §6, §9).

omp writes one JSON object per line to ``<worktree>/logs/<id>-<HHMMSS>.log``.
Everything that inspects a trace — the watchdogs, ``status``'s current tool,
``stats``' wall-time split — shares these helpers instead of each growing its
own line scanner: a tolerant parser (a partial last line while an agent is
still writing is junk, not an error) and the timestamp normalisation (message
``timestamp``/``completedAt`` are epoch milliseconds, ``session.timestamp`` is
ISO; both end up as epoch seconds here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

#: Values at or above this are epoch milliseconds, not seconds (~1973 in ms).
_MS_THRESHOLD = 1e11


@dataclass(frozen=True)
class Call:
    """One tool call: its tool name and, for shell tools, the command string."""

    name: str
    command: str | None = None


@dataclass(frozen=True)
class Message:
    """One ``message_end`` entry, with times normalised to epoch seconds."""

    role: str
    start: float
    end: float
    call: Call | None = None  # first tool call of an assistant message


def event(line: str) -> dict[str, Any] | None:
    """Parse one trace line into an event mapping; junk reports ``None``."""
    try:
        parsed = json.loads(line)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read(path: Path | str) -> str:
    """Read a trace file as text; a missing or unreadable file reads as ``""``."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def epoch(value: Any) -> float | None:
    """Normalise an omp timestamp to epoch seconds, or ``None`` when unusable.

    Accepts epoch milliseconds (omp's ``timestamp``/``completedAt``), epoch
    seconds, and ISO-8601 strings; booleans and junk report ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if abs(number) >= _MS_THRESHOLD else number
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):  # fromisoformat wants +00:00, not Z
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            try:
                return epoch(float(text))
            except ValueError:
                return None
    return None


def tool_command(args: Any) -> str | None:
    """The shell command of a tool call, when it has one (``bash``'s ``command``).

    Non-shell tools (``read``, ``edit``, omp's ``wait``) have no command, so
    their tool name stands in when bucketing trace time (SPEC §9).
    """
    if isinstance(args, dict):
        command = args.get("command")
        if isinstance(command, str) and command.strip():
            return command
    return None
