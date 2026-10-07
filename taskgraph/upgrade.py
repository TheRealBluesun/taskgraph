"""Pick a running agent to restart on a freed top-tier model (SPEC §7).

When the top-priority model has a free agent slot while runnable work is done —
because only lower tiers had room when the task started, or a slot just opened —
the scheduler may restart one *young* running agent there through the normal
resume path: the worktree stays, so no work is lost, and the critical path stops
waiting behind a slow auxiliary tier.

Pure policy: no I/O and no clock, so it is unit-testable.  The scheduler applies
the result with ``agent.kill`` + the resume path and logs an ``upgrade`` event.
"""

from __future__ import annotations

from typing import Collection, Mapping, Protocol, Sequence

from . import order
from .config import ModelConfig
from .plan import Task


class Runner(Protocol):
    """The parts of a running agent :func:`pick_upgrade` reads."""

    id: str
    model: str
    started: float


def pick_upgrade(
    tasks: Mapping[str, Task],
    running: Mapping[str, Runner],
    models: Sequence[ModelConfig],
    assigned: Mapping[str, int],
    now: float,
    window: float,
    *,
    queued: Collection[str] = (),
    merging: Collection[str] = (),
    upgraded: Collection[str] = (),
    forced: Collection[str] = (),
) -> str | None:
    """Return the task id to restart on ``models[0]``, or ``None``.

    ``models`` is in priority order (config order), ``assigned`` the number of
    agents per model, ``tasks`` the parsed plan.  Nothing happens when the top
    model has no free ``agent`` slot, when ``window <= 0`` (disabled), or while
    ``queued`` work is waiting for a slot — that slot belongs to the queue.
    A task is eligible when it runs on a *lower* tier, is neither ``merging``,
    ``upgraded`` (one upgrade per task, no ping-pong) nor ``forced`` onto its
    model (a quota fallback must not be sent back to the model that refused
    it), was started at most ``window`` seconds ago, and sits on the critical
    path — its chain is the longest among the tasks still pending.  Plan order
    breaks ties, so the pick is deterministic.
    """
    if window <= 0 or not models or queued:
        return None
    top = models[0]
    if assigned.get(top.name, 0) >= top.capacity("agent"):
        return None

    chains = order.critical_paths(dict(tasks))
    pending = [tid for tid, task in tasks.items() if not task.done]
    if not pending:
        return None
    critical = max(chains.get(tid, 1) for tid in pending)
    tier = {model.name: index for index, model in enumerate(models)}

    for tid, task in tasks.items():
        record = running.get(tid)
        if record is None or task.done or tid in merging or tid in upgraded or tid in forced:
            continue
        rank = tier.get(record.model)
        if rank is None or rank <= tier[top.name]:
            continue
        if chains.get(tid, 1) != critical:
            continue
        if now - record.started > window:
            continue
        return tid
    return None
