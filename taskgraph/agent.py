"""Run and watch one coding agent per task (SPEC §6).

An agent is a detached process group (``start_new_session=True``): it survives a
scheduler restart, and killing the group stops the whole build tree.  Its
stdout+stderr go to ``<worktree>/logs/<id>-<HHMMSS>.log``, which doubles as the
trace the watchdogs inspect.

Everything that decides an agent's fate is a function of that trace file, so a
restarted scheduler can re-adopt a live agent with no in-memory history: stall
is "mtime unchanged", startup hang and quota errors are line patterns.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import leaseprocs, shims
from .config import Config, ModelConfig
from .plan import Task
from .prompt import PromptError, build_command, overlay_path, write_prompt
from .state import pid_alive

LOG_DIRNAME = "logs"

#: How long a killed agent's group gets to die on SIGTERM before SIGKILL.
STALL_KILL_TIMEOUT = 10.0

#: How long :func:`kill` waits for a SIGKILLed group to actually disappear
#: (only the reap is asynchronous; this is not a second chance to run).
SIGKILL_GRACE = 5.0

#: Startup-hang watchdog: a line *beginning* with this, in the first 20 lines,
#: while no tool has started yet, means omp never got going (SPEC §6).  Only the
#: beginning matters: an agent that echoed this phrase later must not be killed.
STARTUP_PHRASE = "Still starting after"
STARTUP_LINES = 20
TOOL_START_MARK = "tool_execution_start"

#: Case-insensitive substrings that mark a rate-limit/quota error in the trace.
QUOTA_MARKERS = ("rate limit", "rate_limit", "quota", "too many requests")


#: Popen handles for agents launched by this process, kept until reaped so a
#: running child is never garbage-collected (which would emit a ResourceWarning
#: and lose the exit status).  Adopted agents were started by an earlier
#: scheduler process and have no entry here.
_CHILDREN: dict[int, subprocess.Popen] = {}


class AgentError(Exception):
    """An agent cannot be started (bad command/overlay, unwritable worktree)."""


@dataclass
class AgentRecord:
    """The durable description of one running agent (SPEC §6).

    ``pid`` leads the agent's own process group (``pgid == pid``) because it is
    launched via ``setsid``.  ``retries`` is how many times this task's agent has
    been resumed; it lives in scheduler state so a restart does not reset it.
    """

    id: str
    pid: int
    pgid: int
    model: str
    worktree: str
    log: str
    started: float
    retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for scheduler state (SPEC §7)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentRecord":
        """Rebuild a record saved in scheduler state."""
        try:
            pid = int(data["pid"])
            tid = str(data["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentError(f"bad agent record: {data!r}") from exc
        return cls(
            id=tid,
            pid=pid,
            pgid=int(data.get("pgid") or pid),
            model=str(data.get("model", "")),
            worktree=str(data.get("worktree", "")),
            log=str(data.get("log", "")),
            started=float(data.get("started") or 0.0),
            retries=int(data.get("retries") or 0),
        )


def start(
    task: Task,
    model: ModelConfig,
    worktree: Path | str,
    cfg: Config,
    state: Any,
    *,
    resume: bool | None = None,
    now: float | None = None,
    root: Path | str | None = None,
) -> AgentRecord:
    """Launch ``task``'s agent on ``model`` in ``worktree`` (SPEC §6).

    The agent runs detached, with stdin on ``/dev/null`` and its output in
    ``<worktree>/logs/<id>-<HHMMSS>.log``; the deny shims are installed and the
    shim dir is put first on the agent's ``PATH``.  ``root`` overrides
    ``$TASKGRAPH_STATE`` (tests).  ``resume`` defaults to "this task has
    retries", which is exactly when partial work exists in the worktree.
    """
    worktree = Path(worktree)
    retries = int(state.retries.get(task.id, 0))
    if resume is None:
        resume = retries > 0
    overlay = overlay_path(cfg.agent.overlay)
    if not overlay.is_file():
        raise AgentError(f"agent overlay {overlay} does not exist")
    try:
        shims.create_shims(cfg.agent.deny, root=root)
    except shims.ShimsError as exc:
        raise AgentError(f"cannot create deny shims: {exc}") from exc
    try:
        prompt_file = write_prompt(worktree, task, cfg, resume=resume)
    except PromptError as exc:
        raise AgentError(str(exc)) from exc
    started = time.time() if now is None else float(now)
    try:
        argv = build_command(
            cfg.agent.command, model=model.name, overlay=overlay, prompt_file=prompt_file
        )
    except PromptError as exc:
        raise AgentError(str(exc)) from exc
    log = Path(worktree) / LOG_DIRNAME / (
        f"{task.id}-{time.strftime('%H%M%S', time.localtime(started))}.log"
    )
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
    except OSError as exc:
        raise AgentError(f"{log}: cannot open agent log: {exc.strerror or exc}") from exc
    env = dict(os.environ)
    env["PATH"] = shims.path_with_shims(env.get("PATH"), root=root)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=worktree,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise AgentError(f"cannot start agent {task.id}: {exc}") from exc
    finally:
        handle.close()
    _CHILDREN[proc.pid] = proc
    return AgentRecord(
        id=task.id,
        pid=proc.pid,
        pgid=proc.pid,
        model=model.name,
        worktree=os.fspath(worktree),
        log=os.fspath(log),
        started=started,
        retries=retries,
    )


def poll(record: AgentRecord) -> str:
    """Return ``"running"`` or ``"exited"`` for ``record`` (SPEC §6).

    Our own children are asked through their ``Popen`` handle, which reaps them
    *and* records the exit status — a bare ``os.waitpid`` would leave ``Popen``
    thinking the child still runs.  An agent adopted from a previous scheduler
    run has no handle here, so plain liveness decides (its real parent, the dead
    scheduler, is gone, so the kernel has already reaped it).
    """
    if record.pid <= 0:
        return "exited"
    proc = _CHILDREN.get(record.pid)
    if proc is not None:
        if proc.poll() is None:
            return "running"
        _CHILDREN.pop(record.pid, None)
        return "exited"
    return "running" if pid_alive(record.pid) else "exited"


def log_size(log: Path | str) -> int:
    """Return the trace file's size, ``0`` when it does not exist yet."""
    try:
        return Path(log).stat().st_size
    except OSError:
        return 0


def last_activity(log: Path | str, default: float = 0.0) -> float:
    """Return when the trace last grew (its mtime), or ``default`` if absent."""
    try:
        return Path(log).stat().st_mtime
    except OSError:
        return default


def stalled(log: Path | str, now: float, stall_secs: float) -> bool:
    """Return whether the trace has not grown for ``stall_secs`` (SPEC §6).

    Uses the file mtime: the log is append-only, so "size unchanged" and "mtime
    unchanged" are the same event, and mtime needs no per-tick bookkeeping — an
    agent re-adopted after a scheduler restart is judged correctly.  An absent
    trace is not a stall: the startup-hang watchdog owns that case.
    """
    seen = last_activity(log)
    if seen <= 0.0:
        return False
    return now - seen >= stall_secs


def lease_processes(
    worktree: Path | str,
    *,
    root: Path | str | None = None,
    processes: Sequence[leaseprocs.LeaseProcess] | None = None,
) -> list[leaseprocs.LeaseProcess]:
    """Live ``taskgraph lease`` wrappers whose cwd is inside ``worktree`` (SPEC §6).

    A wrapper announces itself under the state root for its whole life — while
    it waits for a slot and while it holds one — so this is where the stall
    watchdog finds the reason an agent's trace went quiet.  ``processes`` lets a
    caller reuse one scan for several agents; cwd comparison resolves symlinks.
    """
    if processes is None:
        processes = leaseprocs.active_processes(root)
    base = _real(worktree)
    return [proc for proc in processes if _inside(proc.cwd, base)]


def oldest_lease(
    record: AgentRecord,
    *,
    root: Path | str | None = None,
    processes: Sequence[leaseprocs.LeaseProcess] | None = None,
) -> leaseprocs.LeaseProcess | None:
    """The longest-running live lease wrapper in ``record``'s worktree (SPEC §6).

    ``None`` means no lease process explains the silence, so a stalled agent is
    really stalled.  The oldest is the one whose age the anomaly check compares
    against ``agent.max_lease_secs``.
    """
    return min(
        lease_processes(record.worktree, root=root, processes=processes),
        key=lambda proc: proc.since,
        default=None,
    )


def startup_hang(log: Path | str, *, head_lines: int = STARTUP_LINES) -> bool:
    """Return whether the trace shows omp never got going (SPEC §6).

    True only while no tool has started yet *and* one of the first ``head_lines``
    lines *begins* with ``Still starting after``.  The phrase anywhere else (an
    agent reading this file and echoing it) must not trigger a kill.
    """
    head: list[str] = []
    try:
        with Path(log).open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if TOOL_START_MARK in line:
                    return False
                if index < head_lines:
                    head.append(line)
    except OSError:
        return False
    return any(line.startswith(STARTUP_PHRASE) for line in head)


def group_alive(pgid: int) -> bool:
    """Return whether any process still belongs to process group ``pgid``."""
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill(
    record: AgentRecord,
    *,
    timeout: float = STALL_KILL_TIMEOUT,
    sleep: Any = None,
) -> None:
    """Stop ``record``'s process group: SIGTERM, then SIGKILL after ``timeout``.

    A process that exited but has not been reaped is still a group member and
    would keep ``killpg(…, 0)`` succeeding, so the loop reaps through
    :func:`poll` as it waits.
    """
    if record.pgid <= 0:
        return
    _signal_group(record.pgid, signal.SIGTERM)
    nap = time.sleep if sleep is None else sleep
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        poll(record)
        if not group_alive(record.pgid):
            return
        nap(0.05)
    _signal_group(record.pgid, signal.SIGKILL)
    deadline = time.monotonic() + SIGKILL_GRACE
    while time.monotonic() < deadline:
        poll(record)
        if not group_alive(record.pgid):
            return
        nap(0.05)


def quota_error(log: Path | str) -> bool:
    """Return whether the trace contains a rate-limit/quota error (SPEC §6)."""
    try:
        text = Path(log).read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(marker in text for marker in QUOTA_MARKERS)


def fallback_model(record: AgentRecord, cfg: Config) -> ModelConfig | None:
    """Return the model to restart ``record`` on after a quota error.

    ``None`` when the trace shows no quota error or the model has no ``fallback``
    configured (SPEC §1/§6); the caller counts the restart as a retry.
    """
    model = cfg.model(record.model)
    if model is None or model.fallback is None or not quota_error(record.log):
        return None
    return cfg.model(model.fallback)


def _real(path: Path | str) -> Path:
    """Resolve ``path`` best-effort (a missing worktree still compares fine)."""
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - resolve() only fails on absurd paths
        return Path(path)


def _inside(cwd: str, base: Path) -> bool:
    """Return whether ``cwd`` is ``base`` or below it, symlinks resolved."""
    if not cwd:
        return False
    return _real(cwd).is_relative_to(base)


def _signal_group(pgid: int, sig: int) -> None:
    """Best-effort signal to a process group (it may already be gone)."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
