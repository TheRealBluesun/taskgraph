"""Commit guard: refuse a task commit that looks like a build dump (SPEC §8).

A task once committed 5,833 SwiftPM build files because its output directory was
missing from ``.gitignore``; only a failed fast-forward stopped them reaching
the integration branch.  Before rebasing, :func:`taskgraph.merge` therefore
inspects the task's work commit and blocks it when it adds more than
``max_files`` files, contains a file over ``max_file_mb`` megabytes, or touches
a path under a build/output directory.

The policy is pure (SPEC §0): callers pass the added paths, the sizes of the
changed files and the configured limits, so it is unit-testable without git.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

#: SPEC §8 defaults for the ``[merge]`` table.
DEFAULT_MAX_FILES = 200
DEFAULT_MAX_FILE_MB = 5.0

#: Build/output directories blocked regardless of configuration (SPEC §8).
DEFAULT_DENY_PATHS: tuple[str, ...] = (
    ".build/",
    "build/",
    "DerivedData/",
    "node_modules/",
    "__pycache__/",
    "target/",
    "dist/",
)

#: How many offending paths a blocked reason names before it summarizes them.
REASON_PATHS = 20


def denied(path: str, deny_dirs: Iterable[str]) -> bool:
    """Return whether ``path`` lies under one of ``deny_dirs`` (SPEC §8).

    A single-segment entry (``dist``) matches any path component, so
    ``frontend/dist/app.js`` counts as under a build directory; a multi-segment
    entry (``out/gen``) matches that path or anything below it.
    """
    parts = path.split("/")
    for raw in deny_dirs:
        entry = raw.strip("/")
        if not entry:
            continue
        if "/" in entry:
            if path == entry or path.startswith(entry + "/"):
                return True
        elif entry in parts:
            return True
    return False


def problems(
    added: Sequence[str],
    sizes: Mapping[str, int],
    *,
    max_files: int,
    max_file_mb: float,
    deny_dirs: Iterable[str],
) -> list[str]:
    """Return one message per violated limit; an empty list means it passes.

    ``added`` is every path the commit adds; ``sizes`` maps every path it adds
    or modifies to its byte size (deleted paths are absent).  Each message names
    the offending paths, so a blocked task's reason says what to fix.
    """
    issues: list[str] = []

    if len(added) > max_files:
        issues.append(f"adds {len(added)} files (limit {max_files}): {_summary(added)}")

    limit = int(max_file_mb * 1024 * 1024)
    oversized = sorted(path for path, size in sizes.items() if size > limit)
    if oversized:
        named = [f"{path} ({_mb(sizes[path])})" for path in oversized]
        issues.append(f"files over {max_file_mb:g} MB: {_summary(named)}")

    blocked = sorted(path for path in sizes if denied(path, deny_dirs))
    if blocked:
        issues.append("paths under build/output dirs: " + _summary(blocked))

    return issues


def _summary(paths: Sequence[str]) -> str:
    """Join paths for a blocked reason, capping the list (SPEC §8)."""
    listed = list(paths)
    if len(listed) > REASON_PATHS:
        shown = ", ".join(listed[:REASON_PATHS])
        return f"{shown}, … (+{len(listed) - REASON_PATHS} more)"
    return ", ".join(listed)


def _mb(size: int) -> str:
    """Format a byte size as megabytes with one decimal."""
    return f"{size / 1048576:.1f} MB"
