"""The per-project event log (SPEC §9).

Every scheduler state change appends exactly one line, shaped for
``tail -F | grep``::

    HH:MM:SS <kind> <id> <detail>

Kinds are ``start``, ``adopt``, ``retry``, ``merged``, ``blocked``, ``stall``,
``idle`` and ``anomaly`` (SPEC §9); ``pool`` marks a model-pool reload (SPEC §4).
The log lives next to ``state.json`` under ``$TASKGRAPH_STATE``, so it is shared
by every scheduler process for the project and survives a restart.
"""

from __future__ import annotations

import time
from pathlib import Path

#: File name of the per-project event log (SPEC §9 ``state/events.log``).
EVENTS_LOG = "events.log"


def event_line(kind: str, tid: str, detail: str = "", now: float | None = None) -> str:
    """Format one events-log line; whitespace in ``detail`` is collapsed."""
    stamp = time.strftime("%H:%M:%S", time.localtime(time.time() if now is None else now))
    detail = " ".join(detail.split())
    return f"{stamp} {kind} {tid}" + (f" {detail}" if detail else "")


def append(path: Path | str, kind: str, tid: str, detail: str = "", now: float | None = None) -> str:
    """Append one event line to ``path`` (creating parents) and return it.

    A log that cannot be written must never break the scheduler, so an
    :class:`OSError` is swallowed; the caller can still echo the line.
    """
    line = event_line(kind, tid, detail, now)
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    return line
