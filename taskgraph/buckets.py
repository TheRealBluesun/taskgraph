"""Tool-time buckets: the regex → bucket policy for ``taskgraph stats`` (SPEC §9).

Which bucket a stretch of tool time belongs in is decided by the *first* tool
call's command, matched against project-configurable regexes (``[stats.buckets]``
in ``taskgraph.toml``); :data:`DEFAULT_BUCKETS` is what a project gets without
that table, and anything unmatched falls into ``other``.

This module is a leaf (it imports only the trace shapes), so both
:mod:`taskgraph.config` (which validates the configured regexes) and
:mod:`taskgraph.stats` (which sums the time) can use it.
"""

from __future__ import annotations

import re
from typing import Sequence

from .trace import Call

#: Default tool-time buckets and the regexes that select them, in match order
#: (first match wins), per SPEC §9: lease waits, builds, tests, omp's ``wait``
#: tool, then everything else.
DEFAULT_BUCKETS: tuple[tuple[str, str], ...] = (
    ("lease-wait", r"\btaskgraph\b[^\n]*\blease\b"),
    ("build", r"\b(xcodebuild|swift build|cargo build|go build|gradle|make|build|compile)\b"),
    ("test", r"\b(xcodebuild test|swift test|unittest|pytest|ctest|rspec|test|tests)\b"),
    ("wait", r"^wait$"),
)

#: Bucket for tool time no pattern matched; reserved in ``[stats.buckets]``.
OTHER = "other"


def bucket_names(patterns: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Bucket names in match order, with ``other`` last."""
    return tuple([name for name, _ in patterns if name != OTHER] + [OTHER])


def bucket_for(call: Call | None, patterns: Sequence[tuple[str, str]]) -> str:
    """Bucket one tool call: first matching pattern wins, else ``other``.

    The call's command is matched when it has one, its tool name otherwise, so
    omp's ``wait`` tool lands in ``wait`` while a shell command lands in the
    regex that describes it.  :func:`taskgraph.config.StatsConfig.patterns`
    orders a project's own buckets before the defaults.
    """
    if call is None:
        return OTHER
    key = call.command if call.command is not None else call.name
    for name, pattern in patterns:
        if re.search(pattern, key):
            return name
    return OTHER
