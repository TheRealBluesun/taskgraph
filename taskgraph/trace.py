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
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Values at or above this are epoch milliseconds, not seconds (~1973 in ms).
_MS_THRESHOLD = 1e11

#: ``type`` of the line omp writes when a tool call starts (SPEC §6/§9).
TOOL_START = "tool_execution_start"


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


def call_signature(parsed: Mapping[str, Any]) -> str:
    """A stable identity for one ``tool_execution_start`` event (SPEC §6).

    The tool name plus its canonicalised arguments.  ``toolCallId`` is excluded
    on purpose: it is fresh for every call, while a degenerate loop repeats the
    same call (the same ``ls``) verbatim over and over.
    """
    name = str(parsed.get("toolName") or "?")
    args = parsed.get("args")
    try:
        rendered = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str covers everything
        rendered = repr(args)
    return f"{name} {rendered}"


def repeated_tail(lines: Iterable[str], window: int) -> bool:
    """Whether the last ``window`` tool calls in ``lines`` are all identical (SPEC §6).

    Only real ``tool_execution_start`` events count, so prose echoing the phrase
    is ignored; a junk line is skipped by the parser.  ``False`` when fewer than
    ``window`` calls happened, and ``window <= 0`` disables the check.  Streams
    the lines, so a huge trace costs no more memory than any other view of it.
    """
    if window <= 0:
        return False
    recent: deque[str] = deque(maxlen=window)
    for line in lines:
        if TOOL_START not in line:
            continue
        parsed = event(line)
        if parsed is None or parsed.get("type") != TOOL_START:
            continue
        recent.append(call_signature(parsed))
    return len(recent) == window and len(set(recent)) == 1


def last_tool(path: Path | str) -> str | None:
    """The tool of the most recent ``tool_execution_start``, or ``None`` (SPEC §9).

    The idle-watch diagnosis names the call an agent is stuck on; a tool event
    is the only record of it (the enclosing message may be a whole tool call
    behind).  Only lines carrying the marker are parsed, so an agent echoing the
    phrase in its own output cannot fake a tool name.
    """
    name: str | None = None
    try:
        handle = Path(path).open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with handle:
        for line in handle:
            if TOOL_START not in line:
                continue
            parsed = event(line)
            if parsed is None or parsed.get("type") != TOOL_START:
                continue
            name = str(parsed.get("toolName") or "?")
    return name


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
