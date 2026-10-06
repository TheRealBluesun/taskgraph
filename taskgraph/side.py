"""Side jobs: lease a model's job-class capacity (SPEC §4).

A model's KV cache is not one number.  The .10 Flash server fits ONE
long-context coding agent and still has room for at most one short vision job
(a screenshot critique), and nothing more.  So a ``[[models]]`` entry may
declare ``classes = { agent = 1, side = 1 }``; plan tasks are class ``agent``
(``sessions`` when no ``agent`` entry is given), and

    taskgraph side <class> -- <cmd…>

leases ``<class>`` capacity on one model and tells the command which model it
got through ``TASKGRAPH_MODEL``.

Model choice prefers a model whose agent is *currently blocked on a tool*
(a build, a simulator lease): its GPU is idle, so the short job rides along
without evicting a generation.  Otherwise models are tried in config order
(priority).  The capacity is the same flock semaphore as SPEC §5 — one resource
per ``model:<name>:<class>`` — so side jobs are shared by every process on the
machine and appear in ``taskgraph leases``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Callable, TextIO
from urllib.parse import quote

from . import agent, lease, leaseprocs, state, status
from .config import Config, ModelConfig
from .procsig import (
    GROUP_GRACE,
    SignalForwarder,
    block_terminating_signals,
    ignore_sigpipe,
    pending_terminating_signal,
    restore_mask,
    swallow_signal,
    terminate_group,
)

#: Environment variable telling a side command which model it was given.
MODEL_ENV = "TASKGRAPH_MODEL"

#: Exit status when the wait times out (same as GNU ``timeout``).
TIMEOUT_EXIT = lease.TIMEOUT_EXIT


class SideError(Exception):
    """A side job cannot start (no model serves the requested class)."""


def resource_name(model: str, cls: str) -> str:
    """The lease resource holding one model's capacity for one class (SPEC §4).

    The model name is percent-encoded: it may contain ``/`` or spaces (both
    illegal or ambiguous in a flat ``leases/<resource>`` directory and in the
    audit log), and two differently spelled model names must never share slots.
    """
    return f"model:{quote(model, safe='')}:{quote(cls, safe='')}"


def class_label(cls: str) -> str:
    """Announced name of a *waiting* side wrapper (which model is still open)."""
    return f"model-class:{quote(cls, safe='')}"


def ordered(
    models: Sequence[ModelConfig], cls: str, blocked: Collection[str]
) -> list[ModelConfig]:
    """Candidate models for ``cls``: blocked-agent models first.

    Only models with capacity for the class qualify; ``blocked`` names the
    models with an agent currently blocked on a tool (SPEC §4).  Ties keep
    config order, which is priority order.
    """
    return sorted(
        (model for model in models if model.capacity(cls) > 0),
        key=lambda model: model.name not in blocked,
    )


def blocked_models(saved: state.State, *, state_root: Path | str | None = None) -> set[str]:
    """Models with at least one running agent executing a tool right now (SPEC §9).

    An agent whose trace has an open ``tool_execution_start`` is inside that
    tool — a build, a simulator lease, any bash call — so its model is idle.
    """
    found: set[str] = set()
    for data in saved.agents:
        try:
            record = agent.AgentRecord.from_dict(data)
        except agent.AgentError:
            continue
        if agent.poll(record) != "running":
            continue
        if status.current_tool(record.log) is not None:
            found.add(record.model)
    return found


def _no_blocked() -> Collection[str]:
    """Default preference input: no model is known to have a blocked agent."""
    return ()


def waiting_message(cls: str, candidates: Sequence[ModelConfig]) -> str:
    """The one line printed while waiting for ``cls`` capacity (SPEC §4)."""
    names = ", ".join(model.name for model in candidates)
    return f"waiting for {cls} capacity (models: {names})"


def wait_for_model(
    models: Sequence[ModelConfig],
    cls: str,
    *,
    root: Path | str | None = None,
    task: str | None = None,
    cmd: Sequence[str] | str = (),
    poll: float = lease.POLL_SECS,
    message_secs: float | None = lease.MESSAGE_SECS,
    out: TextIO | None = None,
    timeout: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    blocked: Callable[[], Collection[str]] = _no_blocked,
    abort: Callable[[], int | None] | None = None,
) -> tuple[ModelConfig, lease.Slot] | None:
    """Wait for a free ``cls`` slot on some model, or ``None`` after ``timeout``.

    Every round re-evaluates the preference order (``blocked`` supplies the
    models whose agent is blocked on a tool), so a side job moves to whichever
    model frees first.  ``abort`` is polled each round; a signal number raises
    :class:`taskgraph.lease.WaitInterrupted`.
    """
    stream = sys.stderr if out is None else out
    base = Path(root) if root is not None else lease.state_root()
    started = clock()
    last_message: float | None = None
    while True:
        candidates = ordered(models, cls, blocked())
        for model in candidates:
            slot = lease.try_acquire(
                resource_name(model.name, cls),
                model.capacity(cls),
                root=base,
                task=task,
                cmd=cmd,
            )
            if slot is not None:
                return model, slot
        if abort is not None:
            signum = abort()
            if signum is not None:
                raise lease.WaitInterrupted(signum)
        if timeout is not None and clock() - started >= timeout:
            return None
        if message_secs is not None and (
            last_message is None or clock() - last_message >= message_secs
        ):
            print(waiting_message(cls, candidates), file=stream, flush=True)
            last_message = clock()
        sleep(poll)


def run_side(
    cls: str,
    command: Sequence[str],
    cfg: Config,
    *,
    task: str | None = None,
    state_root: Path | str | None = None,
    poll: float = lease.POLL_SECS,
    message_secs: float | None = lease.MESSAGE_SECS,
    out: TextIO | None = None,
    timeout: float | None = None,
    group_grace: float = GROUP_GRACE,
) -> int:
    """Lease ``cls`` capacity on one model, run ``command`` under it, return its status.

    The command sees ``TASKGRAPH_MODEL=<model>``.  Signal handling, the slot
    announcement (``leaseprocs``, so the stall watchdog sees a waiting wrapper)
    and process-group cleanup mirror ``taskgraph lease`` (SPEC §5/§6): the
    terminating signals are blocked and the forwarding handlers installed before
    a slot can be acquired, and the slot is released only after the command's
    whole process group is gone.  Returns 124 on timeout, 127 when the command
    cannot be executed, and 128 + *n* when it is killed by signal *n*.
    """
    base = Path(state_root) if state_root is not None else lease.state_root()
    if not ordered(cfg.models, cls, ()):
        raise SideError(
            f"no model has capacity for class '{cls}' (add e.g. "
            f"classes = {{ {cls} = 1 }} to a [[models]] entry of {cfg.path})"
        )
    stream = sys.stderr if out is None else out
    ignore_sigpipe()
    forwarder = SignalForwarder()
    forwarder.install()
    previous = block_terminating_signals()
    blocked_signals = previous is not None
    announcement: Path | None = None
    code = 1
    try:
        # Announce before waiting: omp streams nothing from a leased command, so
        # a silent agent must be explainable by a live wrapper (SPEC §6).
        announcement = leaseprocs.register_process(class_label(cls), root=base, cmd=command)
        try:
            acquired = wait_for_model(
                cfg.models,
                cls,
                root=base,
                task=task,
                cmd=command,
                poll=poll,
                message_secs=message_secs,
                out=stream,
                timeout=timeout,
                abort=pending_terminating_signal if blocked_signals else None,
                blocked=lambda: _blocked_now(cfg, base),
            )
        except lease.WaitInterrupted as exc:
            swallow_signal(exc.signum, previous)
            print(f"taskgraph side: interrupted while waiting for {cls} capacity", file=stream, flush=True)
            return 128 + exc.signum
        except KeyboardInterrupt:
            print(f"taskgraph side: interrupted while waiting for {cls} capacity", file=stream, flush=True)
            return 130
        if acquired is None:
            print(f"taskgraph side: timed out waiting for {cls} capacity", file=stream, flush=True)
            return TIMEOUT_EXIT
        model, slot = acquired
        restore_mask(previous)
        env = dict(os.environ)
        env[MODEL_ENV] = model.name
        try:
            try:
                # Own session: its pid is a process-group id, so a forwarded
                # signal reaches the whole side command tree.
                proc = subprocess.Popen(list(command), env=env, start_new_session=True)
            except OSError as exc:
                print(f"taskgraph side: cannot run {command[0]!r}: {exc}", file=stream, flush=True)
                return 127
            forwarder.attach(proc)
            code = proc.wait()
            terminate_group(proc.pid, grace=group_grace)
        finally:
            lease.release(slot, root=base)
    finally:
        if announcement is not None:
            leaseprocs.unregister_process(announcement)
        forwarder.restore()
        restore_mask(previous)
    return code if code >= 0 else 128 - code


def _blocked_now(cfg: Config, state_root: Path | str | None) -> set[str]:
    """Preference input for one wait round; a corrupt state file means no preference."""
    try:
        saved = state.load(state.state_path(cfg.root, state_root))
    except state.StateError:
        return set()
    return blocked_models(saved, state_root=state_root)
