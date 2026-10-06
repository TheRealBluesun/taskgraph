"""Announcements of live ``taskgraph lease`` wrappers (SPEC §5, §6).

``taskgraph lease`` writes one file per wrapper into ``$TASKGRAPH_STATE/procs/``
before it starts waiting for a slot and removes it when it exits.  The scheduler's
stall watchdog reads them: an agent whose trace went quiet because omp does not
stream a running leased command's output must not be killed as stalled (SPEC §6).

Announcements live outside every worktree and, like a held slot, count only while
their pid is alive: a wrapper killed by SIGKILL cannot unlink its file, so dead
pids are pruned on read.  The machinery is only ever *written* by the wrapper
process itself; readers never take the announcement as proof of ownership.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .lease import LeaseError, state_root

#: Directory (under the state root) of live lease-wrapper announcements.
PROCS_DIRNAME = "procs"


@dataclass(frozen=True)
class LeaseProcess:
    """A live ``taskgraph lease`` wrapper, as announced under ``procs/``.

    ``since`` is when the wrapper started, so :meth:`age` is how long its agent
    has been blocked on that resource — waiting time included.
    """

    pid: int
    resource: str
    cwd: str
    cmd: str
    since: float
    path: Path

    def age(self, now: float | None = None) -> float:
        """Seconds since the wrapper started (never negative)."""
        return max(0.0, (time.time() if now is None else now) - self.since)


def procs_root(root: Path | str | None = None) -> Path:
    """Return the directory of live lease-wrapper announcements."""
    return _root(root) / PROCS_DIRNAME


def register_process(
    resource: str,
    *,
    root: Path | str | None = None,
    cmd: Sequence[str] | str = (),
    cwd: str | None = None,
    pid: int | None = None,
    now: float | None = None,
) -> Path:
    """Announce this process as a ``taskgraph lease`` wrapper (SPEC §6).

    Written atomically, one ``mkstemp`` file per call (unique, so several
    wrappers in one process — tests — are not confused).  Returns the path to
    hand to :func:`unregister_process` when the wrapper exits.
    """
    directory = procs_root(root)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LeaseError(f"{directory}: cannot create process dir: {exc}") from exc
    owner = os.getpid() if pid is None else pid
    payload = {
        "pid": owner,
        "resource": resource,
        "cwd": os.getcwd() if cwd is None else cwd,
        "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
        "since": time.time() if now is None else now,
    }
    try:
        fd, name = tempfile.mkstemp(dir=directory, prefix=f"{owner}-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    except OSError as exc:
        raise LeaseError(f"{directory}: cannot announce lease process: {exc}") from exc
    return Path(name)


def unregister_process(path: Path | str) -> None:
    """Remove a wrapper's announcement (best effort; dead pids are pruned too)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def active_processes(
    root: Path | str | None = None, *, resource: str | None = None
) -> list[LeaseProcess]:
    """Announced lease wrappers whose pid is alive, oldest first (SPEC §6).

    Announcements of dead or unreadable pids are pruned here, so a crashed
    wrapper never exempts an agent from the stall watchdog.
    """
    directory = procs_root(root)
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    live: list[LeaseProcess] = []
    for path in entries:
        if path.suffix != ".json":
            continue
        data = _read_announcement(path)
        if data is None:
            unregister_process(path)
            continue
        try:
            pid = int(data.get("pid", 0))
            since = float(data.get("since", 0.0))
        except (TypeError, ValueError):
            unregister_process(path)
            continue
        if not _pid_alive(pid):
            unregister_process(path)
            continue
        name = str(data.get("resource", ""))
        if resource is not None and name != resource:
            continue
        live.append(
            LeaseProcess(
                pid=pid,
                resource=name,
                cwd=str(data.get("cwd", "")),
                cmd=str(data.get("cmd", "")),
                since=since,
                path=path,
            )
        )
    return sorted(live, key=lambda proc: proc.since)


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else state_root()


def _read_announcement(path: Path) -> dict | None:
    """Parse one announcement file; ``None`` when it is missing or malformed."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` exists (``PermissionError`` means it does)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
