"""Choose the next tasks to run (SPEC §3). Pure: no I/O, no clock.

The scheduler asks :func:`runnable` for the pending tasks whose dependencies are
satisfied, ordered so that work unlocking the most downstream work goes first,
then tasks already started (a restart should finish what it began), then plan
order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .plan import Task

__all__ = ["critical_paths", "runnable"]

# A caller-supplied rank, lower = further along. The scheduler maps
# worktree+notes -> 0, worktree -> 1, nothing -> 2 (SPEC §3.2).
StartedRank = Callable[[Task], int]


def critical_paths(tasks: dict[str, Task]) -> dict[str, int]:
    """Return ``chain`` for every task: ``1 + max(chain of pending dependents)``.

    Dependents are tasks naming this one in ``deps``; done dependents do not
    count. A task with no pending dependents has chain 1. A cycle is broken by
    treating the back-edge as chain 0, so this always terminates.
    """
    dependents: dict[str, list[str]] = {tid: [] for tid in tasks}
    for tid, task in tasks.items():
        for dep in task.deps:
            if dep in tasks:
                dependents[dep].append(tid)

    chains: dict[str, int] = {}
    on_stack: set[str] = set()

    def visit(tid: str) -> int:
        if tid in chains:
            return chains[tid]
        if tid in on_stack:  # back-edge in a cycle
            return 0
        on_stack.add(tid)
        best = 0
        for child in dependents[tid]:
            if not tasks[child].done:
                best = max(best, visit(child))
        on_stack.discard(tid)
        chains[tid] = best + 1
        return chains[tid]

    for tid in tasks:
        visit(tid)
    return chains


def runnable(
    tasks: dict[str, Task],
    running: Iterable[str] = (),
    blocked: Iterable[str] = (),
    started: StartedRank | None = None,
) -> list[Task]:
    """Return the tasks that may start now, in scheduling order.

    A task is skipped if it is done, already ``running``, ``blocked``, or if any
    of its dependencies is a known task that is not done (unknown dependency ids
    count as done, SPEC §2). The survivors are sorted by descending critical
    path, then by ``started`` rank (ascending, lower first; all equal when the
    callable is omitted), then by plan order.
    """
    running_ids = set(running)
    blocked_ids = set(blocked)
    chains = critical_paths(tasks)
    rank: StartedRank = started if started is not None else _no_rank
    plan_index = {tid: index for index, tid in enumerate(tasks)}

    ready = [
        task
        for tid, task in tasks.items()
        if not task.done
        and tid not in running_ids
        and tid not in blocked_ids
        and not any(dep in tasks and not tasks[dep].done for dep in task.deps)
    ]
    ready.sort(key=lambda task: (-chains[task.id], rank(task), plan_index[task.id]))
    return ready


def _no_rank(_task: Task) -> int:
    """Fallback ``started`` callable: every task compares equal."""
    return 0
