"""Render ``taskgraph status``: one project's live picture (SPEC §9).

Read-only, and safe to run whether or not the scheduler is running: everything
shown is durable.  Running agents come from ``state.json`` (only records whose
pid is still alive — a crashed scheduler leaves dead ones behind) plus their
trace files; the merge queue is the set of tasks whose worktree holds
``progress/<id>.done`` and which are neither running nor blocked (the merge
worker only takes work off disk, SPEC §8); blocked reasons come from state; the
next runnable tasks are SPEC §3's order; lease holders are probed exactly like
``taskgraph leases`` (SPEC §5); model load is the mean ``running``/``waiting``
over the recent sample window (SPEC §4).

:func:`collect` does the I/O, :func:`render` is pure and takes a
:class:`ProjectStatus`, so the table can be unit-tested from a fixture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import agent, assign, lease, order, plan, pool, state, trace, worktree
from .config import Config

#: Trace markers that say a tool call is executing (SPEC §6/§9).
TOOL_START = agent.TOOL_START_MARK
TOOL_END = "tool_execution_end"


@dataclass(frozen=True)
class AgentStatus:
    """One running agent, as shown in the table."""

    id: str
    model: str
    age: float
    idle: float | None  # seconds since the trace last grew; None = no trace yet
    tool: str | None  # tool executing right now; None = thinking/streaming


@dataclass(frozen=True)
class ModelLoad:
    """A model's mean load over the recent sample window; ``None`` = no samples."""

    name: str
    running: float | None
    waiting: float | None


@dataclass(frozen=True)
class ProjectStatus:
    """Everything ``taskgraph status`` prints, already resolved."""

    running: tuple[AgentStatus, ...] = ()
    merging: tuple[str, ...] = ()
    blocked: tuple[tuple[str, str], ...] = ()
    runnable: tuple[str, ...] = ()
    holders: tuple[lease.Slot, ...] = ()
    load: tuple[ModelLoad, ...] = ()


# ---------------------------------------------------------------------- helpers


def format_age(seconds: float) -> str:
    """Render a duration the way ``taskgraph leases`` does: ``59s``, ``2m05s``."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-align every column but the last, which is never padded."""
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header) - 1)]
    return [
        ("  ".join(cell.ljust(width) for cell, width in zip(row[:-1], widths)) + f"  {row[-1]}")
        .rstrip()
        for row in (header, *rows)
    ]


def format_holders(slots: Sequence[lease.Slot], now: float | None = None) -> str:
    """Render live holders as an aligned table for ``taskgraph leases``."""
    if not slots:
        return "no leases held"
    header = ("RESOURCE", "SLOT", "PID", "TASK", "AGE", "COMMAND")
    rows = [
        [
            slot.resource,
            slot.name,
            str(slot.pid),
            slot.task or "-",
            format_age(slot.age(now)),
            slot.cmd,
        ]
        for slot in slots
    ]
    return "\n".join(table(header, rows))


def current_tool(log: Path | str) -> str | None:
    """Return the tool an agent is executing right now, or ``None``.

    omp writes one JSON object per line, so a tool is current while its
    ``tool_execution_start`` has no matching ``tool_execution_end``.  Only lines
    carrying either marker are parsed: an agent's own trace text that merely
    mentions the marker fails the ``type`` check, and scanning a multi-megabyte
    trace costs one ``json.loads`` per tool call, not per token delta.
    """
    open_calls: dict[str, str] = {}
    try:
        handle = Path(log).open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with handle:
        for line in handle:
            if TOOL_START not in line and TOOL_END not in line:
                continue
            event = trace.event(line)
            if event is None:
                continue
            kind = event.get("type")
            if kind == TOOL_START:
                key = str(event.get("toolCallId") or len(open_calls))
                open_calls[key] = str(event.get("toolName") or "?")
            elif kind == TOOL_END:
                open_calls.pop(str(event.get("toolCallId")), None)
    return next((name for name in reversed(open_calls.values())), None)


def _idle(log: Path | str, now: float) -> float | None:
    """Seconds since the trace last grew, or ``None`` when there is no trace."""
    seen = agent.last_activity(log)
    return None if seen <= 0.0 else max(0.0, now - seen)


def _plan_tasks(cfg: Config) -> dict[str, plan.Task]:
    """Parse the project's plan; a plan that cannot be read yields no tasks."""
    try:
        text = (Path(cfg.root) / cfg.plan).read_text(encoding="utf-8")
    except OSError:
        return {}
    return plan.parse(text)


def _model_load(cfg: Config, saved: state.State, now: float) -> tuple[ModelLoad, ...]:
    """Mean ``running``/``waiting`` per configured model over the recent window."""
    history = assign.history_from_state(saved.samples)
    loads = []
    for model in cfg.models:
        recent = pool.recent_samples(history.get(model.name, ()), now)
        if not recent:
            loads.append(ModelLoad(model.name, None, None))
            continue
        loads.append(
            ModelLoad(
                model.name,
                sum(sample.metrics.running for sample in recent) / len(recent),
                sum(sample.metrics.waiting for sample in recent) / len(recent),
            )
        )
    return tuple(loads)


# ----------------------------------------------------------------------- collect


def collect(
    cfg: Config, *, state_root: Path | str | None = None, now: float | None = None
) -> ProjectStatus:
    """Gather the live status of ``cfg``'s project (SPEC §9).

    Raises :class:`taskgraph.state.StateError` when ``state.json`` is malformed:
    a corrupt state is reported, never rendered as an empty project.
    """
    now = time.time() if now is None else now
    saved = state.load(state.state_path(cfg.root, state_root))
    tasks = _plan_tasks(cfg)
    index = {tid: position for position, tid in enumerate(tasks)}

    records = []
    for data in saved.agents:
        try:
            records.append(agent.AgentRecord.from_dict(data))
        except agent.AgentError:
            continue
    live = [record for record in records if agent.poll(record) == "running"]
    live.sort(key=lambda record: (index.get(record.id, len(tasks)), record.id))
    running_ids = {record.id for record in live}
    running = tuple(
        AgentStatus(
            id=record.id,
            model=record.model or "-",
            age=max(0.0, now - record.started),
            idle=_idle(record.log, now),
            tool=current_tool(record.log),
        )
        for record in live
    )

    # The merge worker's queue is not persisted; a finished-but-unmerged task is
    # exactly one with a `.done` marker that is neither running nor blocked.
    merging = tuple(
        tid
        for tid, task in tasks.items()
        if not task.done
        and tid not in running_ids
        and tid not in saved.blocked
        and worktree.done_path(cfg, tid).is_file()
    )
    blocked = tuple(
        (tid, " ".join(saved.blocked[tid].split()) or "-")
        for tid in sorted(saved.blocked, key=lambda tid: (index.get(tid, len(tasks)), tid))
    )
    runnable = tuple(
        task.id
        for task in order.runnable(
            tasks,
            running_ids | set(merging),
            set(saved.blocked),
            lambda task: worktree.started_rank(cfg, task.id),
        )
    )
    return ProjectStatus(
        running=running,
        merging=merging,
        blocked=blocked,
        runnable=runnable,
        holders=tuple(lease.holders(state_root)),
        load=_model_load(cfg, saved, now),
    )


# ------------------------------------------------------------------------ render


def _section(title: str, count: int, rows: Sequence[str]) -> list[str]:
    """One section: a counted header plus indented rows (``none`` when empty)."""
    body = [f"  {row}" for row in rows] if rows else ["  none"]
    return [f"{title} ({count})", *body]


def render(status: ProjectStatus, *, now: float | None = None) -> str:
    """Render ``status`` as plain text, one section per SPEC §9 entry."""
    running = table(
        ("ID", "MODEL", "AGE", "IDLE", "TOOL"),
        [
            (
                agent_status.id,
                agent_status.model,
                format_age(agent_status.age),
                "-" if agent_status.idle is None else format_age(agent_status.idle),
                agent_status.tool or "-",
            )
            for agent_status in status.running
        ],
    )
    load = table(
        ("MODEL", "RUNNING", "WAITING"),
        [
            (
                row.name,
                "-" if row.running is None else f"{row.running:.2f}",
                "-" if row.waiting is None else f"{row.waiting:.2f}",
            )
            for row in status.load
        ],
    )
    holders = format_holders(status.holders, now).splitlines() if status.holders else []
    lines = _section("running", len(status.running), running if status.running else [])
    lines += _section("merge queue", len(status.merging), list(status.merging))
    lines += _section("blocked", len(status.blocked), [f"{tid}  {why}" for tid, why in status.blocked])
    lines += _section("next runnable", len(status.runnable), list(status.runnable))
    lines += _section("leases", len(status.holders), holders)
    lines += _section("model load", len(status.load), load if status.load else [])
    return "\n".join(lines)
