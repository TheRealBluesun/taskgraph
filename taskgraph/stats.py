"""Wall-time accounting from omp's JSON traces (SPEC §9).

``taskgraph stats`` answers "where does the time go".  For every task trace the
model time (the sum of the assistant messages' own ``duration``) is separated
from the tool time (the gap from an assistant message that called a tool to the
next message), and each gap is bucketed by the *first* tool call's command
(:mod:`taskgraph.buckets`, ``[stats.buckets]`` in ``taskgraph.toml``).

Parsing and summing are pure (:func:`parse_text`, :func:`summarize`) so they can
be unit-tested against a fixture trace; only :func:`collect` touches the disk
(the agent traces under the configured worktree base), and :func:`render` is
pure text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from . import agent, plan, status, trace, worktree
from .buckets import bucket_for, bucket_names
from .duration import format_seconds as format_window
from .duration import parse_seconds as parse_window
from .trace import Call, Message

#: ``<id>-<HHMMSS>.log``, as written by :func:`taskgraph.agent.start` (SPEC §6).
TRACE_NAME = re.compile(r"^(?P<id>[A-Za-z][A-Za-z0-9]*)-(?P<clock>\d{6})\.log$")


# ------------------------------------------------------------------- messages


def parse_text(text: str) -> list[Message]:
    """Parse a trace's text into its ``message_end`` entries, in order.

    Only ``message_end`` carries a complete message (``message_start`` is the
    same message twice), and only lines carrying the marker are parsed, so
    scanning a multi-megabyte trace costs one ``json.loads`` per message rather
    than per token delta.  A message without a usable timestamp is skipped.
    """
    messages: list[Message] = []
    for line in text.splitlines():
        if "message_end" not in line:
            continue
        event = trace.event(line)
        if event is None or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        start = trace.epoch(message.get("timestamp"))
        if start is None:
            continue
        messages.append(
            Message(
                role=str(message.get("role") or "?"),
                start=start,
                end=_end(start, message),
                call=_first_call(message),
            )
        )
    return messages


def parse_trace(path: Path | str) -> list[Message]:
    """Parse the trace file at ``path`` (a missing trace reports no messages)."""
    return parse_text(trace.read(path))


def _end(start: float, message: Mapping[str, object]) -> float:
    """When the message finished: ``completedAt``, else ``timestamp + duration``."""
    end = trace.epoch(message.get("completedAt"))
    if end is not None:
        return max(start, end)
    duration = message.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        return start + max(0.0, float(duration) / 1000.0)
    return start


def _first_call(message: Mapping[str, object]) -> Call | None:
    """The first tool call of a message, or ``None`` (SPEC §9 buckets by it)."""
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("toolCall", "tool_call"):
            continue
        name = str(item.get("name") or "?")
        return Call(name=name, command=trace.tool_command(item.get("arguments")))
    return None


# ------------------------------------------------------------------- summaries


@dataclass(frozen=True)
class TraceSummary:
    """One trace's split: span, model seconds, tool seconds by bucket."""

    messages: int
    wall: float
    model: float
    tool: float
    buckets: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskStats:
    """One task's traces summed, as printed in a row (``id`` may be ``total``)."""

    id: str
    traces: int
    wall: float
    model: float
    tool: float
    buckets: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Stats:
    """Everything ``taskgraph stats`` prints."""

    tasks: tuple[TaskStats, ...] = ()
    buckets: tuple[str, ...] = ()
    since: float | None = None


def summarize(
    messages: Sequence[Message],
    patterns: Sequence[tuple[str, str]],
    *,
    since: float | None = None,
    now: float | None = None,
) -> TraceSummary:
    """Split ``messages`` into model/tool time and bucket the tool time (SPEC §9).

    Model time is the assistant messages' own duration.  Tool time is the gap
    from the end of an assistant message that called a tool to the start of the
    next message; a message with no tool call ends a turn, so its gap is not
    tool time.  Overlapping timestamps (clock skew) contribute zero, never a
    negative bucket.
    """
    window = _within(messages, since, now)
    buckets = dict.fromkeys(bucket_names(patterns), 0.0)
    model = sum(m.end - m.start for m in window if m.role == "assistant")
    tool = 0.0
    for index, message in enumerate(window):
        if message.call is None or message.role != "assistant" or index + 1 >= len(window):
            continue
        gap = max(0.0, window[index + 1].start - message.end)
        tool += gap
        buckets[bucket_for(message.call, patterns)] += gap
    return TraceSummary(
        messages=len(window), wall=_span(window), model=model, tool=tool, buckets=buckets
    )


def _within(
    messages: Sequence[Message], since: float | None, now: float | None
) -> list[Message]:
    """Messages inside the ``--since`` window (everything when it is ``None``)."""
    if since is None:
        return list(messages)
    cutoff = (time.time() if now is None else now) - since
    return [message for message in messages if message.end >= cutoff]


def _span(messages: Sequence[Message]) -> float:
    """Elapsed time the messages cover: last end minus first start."""
    if not messages:
        return 0.0
    return max(message.end for message in messages) - min(message.start for message in messages)


def task_stats(tid: str, summaries: Sequence[TraceSummary]) -> TaskStats:
    """Sum one task's traces (retries and resumes are separate traces)."""
    buckets: dict[str, float] = {}
    for summary in summaries:
        for name, seconds in summary.buckets.items():
            buckets[name] = buckets.get(name, 0.0) + seconds
    return TaskStats(
        id=tid,
        traces=len(summaries),
        wall=sum(summary.wall for summary in summaries),
        model=sum(summary.model for summary in summaries),
        tool=sum(summary.tool for summary in summaries),
        buckets=buckets,
    )


def totals(snapshot: Stats) -> TaskStats:
    """Every task summed, for the table's final row."""
    buckets: dict[str, float] = {}
    for task in snapshot.tasks:
        for name, seconds in task.buckets.items():
            buckets[name] = buckets.get(name, 0.0) + seconds
    return TaskStats(
        id="total",
        traces=sum(task.traces for task in snapshot.tasks),
        wall=sum(task.wall for task in snapshot.tasks),
        model=sum(task.model for task in snapshot.tasks),
        tool=sum(task.tool for task in snapshot.tasks),
        buckets=buckets,
    )


# --------------------------------------------------------------------- collect


def traces(cfg) -> list[tuple[str, Path]]:
    """``(task id, path)`` for every agent trace under the worktree base (SPEC §6).

    Discovery is by filename (``<id>-<HHMMSS>.log``), so a trace of a task that
    has since left the plan is still counted, and ``logs/gate.log`` is ignored.
    Worktrees the merge worker already removed have no traces left to read.
    """
    found: list[tuple[str, Path]] = []
    try:
        children = sorted(entry for entry in worktree.base_dir(cfg).iterdir() if entry.is_dir())
    except OSError:
        return found
    for child in children:
        try:
            logs = sorted((child / agent.LOG_DIRNAME).glob("*.log"))
        except OSError:
            continue
        for path in logs:
            match = TRACE_NAME.match(path.name)
            if match is not None:
                found.append((match.group("id"), path))
    return found


def collect(cfg, *, since: float | None = None, now: float | None = None) -> Stats:
    """Summarize every task trace of ``cfg``'s project (SPEC §9).

    Tasks are ordered like the plan (unknown ids last, by id) so a row's place
    is stable across runs.
    """
    patterns = cfg.stats.patterns
    found: dict[str, list[TraceSummary]] = {}
    for tid, path in traces(cfg):
        summary = summarize(parse_trace(path), patterns, since=since, now=now)
        if summary.messages == 0:
            continue  # nothing in the window (or not a trace at all)
        found.setdefault(tid, []).append(summary)
    order = _task_order(cfg)
    tasks = tuple(
        task_stats(tid, found[tid])
        for tid in sorted(found, key=lambda tid: (order.get(tid, len(order)), tid))
    )
    return Stats(tasks=tasks, buckets=bucket_names(patterns), since=since)


def _task_order(cfg) -> dict[str, int]:
    """Task id → plan position, for sorting rows like the plan."""
    try:
        text = (Path(cfg.root) / cfg.plan).read_text(encoding="utf-8")
    except OSError:
        return {}
    return {tid: index for index, tid in enumerate(plan.parse(text))}


# ---------------------------------------------------------------------- render


def render(snapshot: Stats) -> str:
    """Render the per-task and total split as an aligned table (SPEC §9)."""
    header = (
        "TASK",
        "TRACES",
        "WALL",
        "MODEL",
        "TOOL",
        *[name.upper() for name in snapshot.buckets],
    )
    if not snapshot.tasks:
        if snapshot.since is None:
            return "no agent traces found"
        return f"no agent traces in the last {format_window(snapshot.since)}"
    rows = [_row(task, snapshot.buckets) for task in (*snapshot.tasks, totals(snapshot))]
    return "\n".join(status.table(header, rows))


def _row(task: TaskStats, buckets: Sequence[str]) -> tuple[str, ...]:
    """One table row: totals first, then one duration per bucket."""
    return (
        task.id,
        str(task.traces),
        status.format_age(task.wall),
        status.format_age(task.model),
        status.format_age(task.tool),
        *[status.format_age(task.buckets.get(name, 0.0)) for name in buckets],
    )
