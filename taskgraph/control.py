"""Operator commands against a (possibly running) project: stop, retry (SPEC §7, §10).

``taskgraph stop`` signals *the scheduler process* and nothing else: agents are
detached process groups that deliberately survive a scheduler restart (SPEC §7),
so stopping them is a separate, explicit act (``--agents``).  ``taskgraph retry``
writes a durable request under the project's ``pending/`` directory that the
scheduler applies on its next tick (``state.request_retry``), so the command
works whether or not the scheduler is running and cannot be lost to a
``state.json`` write.

Everything here works through pids, the state file and that request directory:
no leases, no git.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import agent, plan, state
from .config import Config

#: How long the scheduler gets to exit on SIGTERM before it is SIGKILLed.
STOP_TIMEOUT = 10.0
#: How long a SIGKILLed process gets to disappear.
KILL_GRACE = 5.0
#: Poll interval while waiting for a process to exit.
POLL_SECS = 0.05


class ControlError(Exception):
    """The requested stop/retry cannot be carried out."""


@dataclass(frozen=True)
class StopResult:
    """What ``taskgraph stop`` did."""

    pid: int | None = None  # scheduler pid; ``None`` = no scheduler was running
    forced: bool = False  # the scheduler ignored SIGTERM and was SIGKILLed
    agents: tuple[str, ...] = ()  # task ids whose agents ``--agents`` stopped

    def stopped_anything(self) -> bool:
        """Return whether this call stopped a scheduler or at least one agent."""
        return self.pid is not None or bool(self.agents)


@dataclass(frozen=True)
class RetryResult:
    """What ``taskgraph retry`` queued."""

    id: str
    was_blocked: bool
    reason: str = ""  # the block the request replaces, for the operator


def stop(
    project: Path | str,
    root: Path | str | None = None,
    *,
    agents: bool = False,
    timeout: float = STOP_TIMEOUT,
    poll: float = POLL_SECS,
    kill_timeout: float = agent.STALL_KILL_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> StopResult:
    """Stop ``project``'s scheduler, and with ``agents=True`` its agents too.

    The scheduler is stopped *first* and only then are agents looked up: its
    last tick is what makes ``state.json`` list every agent it started (SPEC §7).
    Agents whose recorded pid is not alive are left alone; the records this call
    killed are dropped so a later ``taskgraph run`` restarts them (their
    worktrees are still there, so they resume).
    """
    lock = state.live_lock(project, root)
    pid: int | None = None
    forced = False
    if lock is not None:
        pid = lock.pid
        forced = stop_scheduler(lock, timeout=timeout, poll=poll, sleep=sleep)
    stopped: tuple[str, ...] = ()
    if agents:
        stopped = stop_agents(project, root, timeout=kill_timeout, sleep=sleep)
    return StopResult(pid=pid, forced=forced, agents=stopped)


def stop_scheduler(
    lock: state.Lock,
    *,
    timeout: float = STOP_TIMEOUT,
    poll: float = POLL_SECS,
    grace: float = KILL_GRACE,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """SIGTERM the scheduler named by ``lock`` and wait for it to exit.

    Returns whether SIGKILL was needed.  The scheduler's handler ends the loop
    without touching its agents, so they keep running (SPEC §7).
    """
    if not _signal(lock.pid, signal.SIGTERM):
        return False  # already gone: nothing to stop, nothing to report
    if _wait_gone(lock.pid, timeout, poll, sleep):
        return False
    _signal(lock.pid, signal.SIGKILL)
    if _wait_gone(lock.pid, grace, poll, sleep):
        return True
    raise ControlError(f"scheduler pid {lock.pid} did not exit (still alive)")


def stop_agents(
    project: Path | str,
    root: Path | str | None = None,
    *,
    timeout: float = agent.STALL_KILL_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, ...]:
    """Stop every live agent recorded for ``project`` and forget their records.

    Signal order matches the scheduler's own kills: SIGTERM to the agent's
    process group, SIGKILL after ``timeout`` (SPEC §6), so a leased build tree
    is emptied rather than orphaned.
    """
    path = state.state_path(project, root)
    try:
        saved = state.load(path)
    except state.StateError as exc:
        raise ControlError(str(exc)) from exc
    records: list[agent.AgentRecord] = []
    for data in saved.agents:
        try:
            records.append(agent.AgentRecord.from_dict(data))
        except agent.AgentError:
            continue
    live = [record for record in records if state.start_time_matches(record.pid, record.started)]
    for record in live:
        agent.kill(record, timeout=timeout, sleep=sleep)
    if live:
        stopped = {record.id for record in live}
        saved.agents = [record.as_dict() for record in records if record.id not in stopped]
        try:
            state.save(saved, path)
        except OSError as exc:
            raise ControlError(f"cannot update {path}: {exc.strerror or exc}") from exc
    return tuple(record.id for record in live)


def retry(cfg: Config, tid: str) -> RetryResult:
    """Unblock ``tid`` and give it a fresh retry budget, keeping its worktree.

    The request is written to the state dir (:func:`taskgraph.state.request_retry`)
    and applied by the scheduler on its next tick — whether it is running now or
    started later — because the worktree holds partial work the next agent must
    resume (SPEC §6/§10); clearing the retry count is what makes ``retry`` an
    actual second chance rather than an immediate re-block.  A task whose agent
    is still alive is refused: the loop would start a second agent for the same
    worktree.
    """
    project = Path(cfg.root)
    tasks = _tasks(cfg)
    task = tasks.get(tid)
    if task is None:
        raise ControlError(f"no task {tid} in {cfg.plan}")
    if task.done:
        raise ControlError(f"task {tid} is already done")

    try:
        saved = state.load(state.state_path(project))
    except state.StateError as exc:
        raise ControlError(str(exc)) from exc
    for data in saved.agents:
        try:
            record = agent.AgentRecord.from_dict(data)
        except agent.AgentError:
            continue
        if record.id == tid and state.start_time_matches(record.pid, record.started):
            raise ControlError(f"task {tid} is still running (pid {record.pid})")

    was_blocked = tid in saved.blocked
    reason = saved.blocked.get(tid, "")
    try:
        state.request_retry(project, tid)
    except OSError as exc:
        raise ControlError(f"cannot queue the retry: {exc.strerror or exc}") from exc
    return RetryResult(id=tid, was_blocked=was_blocked, reason=reason)


def _tasks(cfg: Config) -> dict[str, plan.Task]:
    """Parse the project's plan, reporting an unreadable one as a control error."""
    path = Path(cfg.root) / cfg.plan
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ControlError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return plan.parse(text)


def _signal(pid: int, sig: int) -> bool:
    """Signal ``pid``; return whether a process was there to receive it."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError as exc:  # pragma: no cover - not our process
        raise ControlError(f"cannot signal pid {pid}: {exc}") from exc
    return True


def _gone(pid: int) -> bool:
    """Return whether ``pid`` no longer runs code (gone, or an unreaped zombie)."""
    if not state.pid_alive(pid):
        return True
    return state.process_state(pid) == "Z"


def _wait_gone(
    pid: int,
    timeout: float,
    poll: float,
    sleep: Callable[[float], None],
) -> bool:
    """Wait up to ``timeout`` for ``pid`` to stop running; return whether it did."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if _gone(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(poll)
