"""Parse and update the project's Markdown task list (SPEC §2).

One task per line; every other line is ignored, so a plan can hold headings,
prose and checklists that are not tasks. Parsing is pure (text in, objects out)
and :func:`mark_done` rewrites only the one line it is asked about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A task line: "- [ ] F03 Top strip per SPEC … [deps: F01, F02] [res: mac]".
# Leading whitespace is allowed; the id is `[A-Z]+[0-9]+[a-z]?` per SPEC §2.
_LINE_RE = re.compile(
    r"^[ \t]*[-*][ \t]+\[(?P<mark>[ xX])\][ \t]+"
    r"(?P<id>[A-Z]+[0-9]+[a-z]?)(?P<rest>(?:[ \t]+.*)?)[ \t]*$"
)
_DEPS_RE = re.compile(r"\[deps:\s*([^\]]*)\]")
_RES_RE = re.compile(r"\[res:\s*([^\]]*)\]")

DONE_MARK = "x"


class PlanError(Exception):
    """The plan text cannot be updated as asked (e.g. unknown task id)."""


@dataclass(frozen=True)
class Task:
    """One parsed plan line.

    ``text`` is the task description with the ``[deps: …]`` / ``[res: …]`` tags
    removed (they are metadata, not part of the prompt), ``deps`` and ``res`` are
    the comma-separated tag values in the order written.
    """

    id: str
    text: str
    done: bool = False
    deps: tuple[str, ...] = ()
    res: tuple[str, ...] = ()


def parse(text: str) -> dict[str, Task]:
    """Return the tasks in ``text`` keyed by id, in plan order."""
    tasks: dict[str, Task] = {}
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        tid = match.group("id")
        body, deps, res = _split_tags(match.group("rest"))
        tasks[tid] = Task(
            id=tid,
            text=body,
            done=match.group("mark").lower() == DONE_MARK,
            deps=deps,
            res=res,
        )
    return tasks


def mark_done(text: str, tid: str) -> str:
    """Return ``text`` with ``tid``'s checkbox ticked; every other byte is kept.

    Raises :class:`PlanError` if no task line carries ``tid``. An already-done
    task is returned unchanged.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = _LINE_RE.match(line)
        if match is None or match.group("id") != tid:
            continue
        if match.group("mark").lower() == DONE_MARK:
            return text
        start, end = match.span("mark")
        lines[index] = line[:start] + DONE_MARK + line[end:]
        return "".join(lines)
    raise PlanError(f"plan has no task {tid!r}")


def _split_tags(rest: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Strip ``[deps: …]`` / ``[res: …]`` from ``rest``.

    Returns the description (whitespace-collapsed), the dependency ids and the
    resource names, both in the order written and with empty items dropped.
    """
    deps = _values(_DEPS_RE.search(rest))
    res = _values(_RES_RE.search(rest))
    body = _RES_RE.sub("", _DEPS_RE.sub("", rest))
    return " ".join(body.split()), deps, res


def _values(match: re.Match[str] | None) -> tuple[str, ...]:
    """Split a tag's comma-separated value, dropping blank entries."""
    if match is None:
        return ()
    return tuple(item.strip() for item in match.group(1).split(",") if item.strip())
