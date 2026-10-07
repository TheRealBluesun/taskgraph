"""Parse and render the durations taskgraph writes in config and on the CLI.

``stats --since`` (SPEC §10) and ``[scheduler] upgrade_window`` (SPEC §7) accept
the same notation — ``90s``, ``30m``, ``6h``, ``2d``, or a bare number of
seconds — so the parser lives here and neither module imports the other.
"""

from __future__ import annotations

import re

#: Duration units: ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days.
UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

_RE = re.compile(r"(\d+(?:\.\d+)?)([smhd]?)")


def parse_seconds(text: str) -> float:
    """Parse ``90s``/``30m``/``6h``/``2d`` into seconds (a bare number = seconds).

    Raises :class:`ValueError` on anything else.  The message says "window"
    because ``taskgraph stats --since`` is the text users see first; callers
    with their own vocabulary (the config loader) replace it.
    """
    match = _RE.fullmatch(text.strip().lower())
    if match is None:
        raise ValueError(f"invalid window {text!r} (use e.g. 90s, 30m, 6h, 2d)")
    return float(match.group(1)) * UNITS[match.group(2) or "s"]


def format_seconds(seconds: float | None) -> str:
    """Render a duration: ``all time`` for ``None``, else ``6h``/``90s``."""
    if seconds is None:
        return "all time"
    for unit in ("d", "h", "m"):
        size = UNITS[unit]
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"
