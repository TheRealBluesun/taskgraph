"""Scheduler state and its single-instance lock (SPEC §7).

The scheduler keeps everything it must not lose across a restart under
``$TASKGRAPH_STATE/projects/<hash-of-project-root>/``:

* ``state.json`` — running agent records, blocked tasks (with reasons), retry
  counts, the pool's ``last_extra`` timestamps and per-model load samples.
  Small and rewritten often, so :func:`save` writes a temp file in the same
  directory and ``os.replace``s it: a crash never leaves a half-written state.
* ``lock`` — the pid (and start time) of the running scheduler.  ``taskgraph
  run`` refuses to start while that pid is alive and is the same process it was
  when the lock was written; ``taskgraph stop`` kills exactly that pid.

Process identity is ``(pid, process start time)``, never the pid alone: pids are
reused, and a restarted machine can hand a dead agent's pid to something else.
``ps -o lstart=`` gives a start time that changes when a pid is recycled, so a
record whose pid is alive but whose start time differs is treated as dead.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import lease

# ``ps`` output like ``Sun Oct  5 22:53:05 2026`` (local time, second precision).
_PS_TIME_FMT = "%a %b %d %H:%M:%S %Y"

# Two ``ps`` reads of the same process can differ by a second across DST folds.
_START_TOLERANCE = 1.0


class StateError(Exception):
    """The state file is unreadable or malformed."""


class LockError(StateError):
    """Another scheduler already holds this project's lock."""


@dataclass
class State:
    """Everything the scheduler persists (SPEC §7).

    ``agents`` holds serialized ``AgentRecord`` dicts (shape owned by
    ``taskgraph.agent``); the other fields mirror SPEC §7 verbatim.
    """

    agents: list[dict[str, Any]] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    last_extra: dict[str, float] = field(default_factory=dict)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready mapping written to ``state.json``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "State":
        """Build a :class:`State` from JSON, defaulting and ignoring unknown keys."""
        return cls(
            agents=list(data.get("agents") or []),
            blocked=dict(data.get("blocked") or {}),
            retries=dict(data.get("retries") or {}),
            last_extra=dict(data.get("last_extra") or {}),
            samples=dict(data.get("samples") or {}),
        )


# --------------------------------------------------------------------------- paths


def _state_dir(root: Path | str | None) -> Path:
    """Return the taskgraph state directory (``$TASKGRAPH_STATE`` by default)."""
    return Path(root) if root is not None else lease.state_root()


def project_key(project_root: Path | str) -> str:
    """Return the stable directory name for a project (hash of its resolved path)."""
    resolved = str(Path(project_root).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def project_dir(project_root: Path | str, root: Path | str | None = None) -> Path:
    """Return ``$TASKGRAPH_STATE/projects/<key>`` for ``project_root``."""
    return _state_dir(root) / "projects" / project_key(project_root)


def state_path(project_root: Path | str, root: Path | str | None = None) -> Path:
    """Return the project's ``state.json`` path."""
    return project_dir(project_root, root) / "state.json"


def lock_path(project_root: Path | str, root: Path | str | None = None) -> Path:
    """Return the project's single-instance lock path (SPEC §7 ``state/lock``)."""
    return project_dir(project_root, root) / "lock"


#: Suffix of an operator retry request (SPEC §10 ``taskgraph retry``).
RETRY_SUFFIX = ".retry"

#: Suffix a consumer renames a request to before applying it, so a crash
#: between renaming and applying leaves a file the next collection takes again.
CLAIM_SUFFIX = ".claimed"


def pending_dir(project_root: Path | str, root: Path | str | None = None) -> Path:
    """Return the project's ``pending/`` directory of operator requests."""
    return project_dir(project_root, root) / "pending"


def request_retry(project_root: Path | str, tid: str, root: Path | str | None = None) -> Path:
    """Write ``tid``'s retry request; it stays until the scheduler consumes it.

    ``taskgraph retry`` cannot simply edit ``state.json``: a running scheduler
    owns that file and rewrites it from memory every tick, so the edit could be
    overwritten before the loop ever notices it.  A request file is durable and
    outside anything the loop writes.
    """
    path = pending_dir(project_root, root) / f"{tid}{RETRY_SUFFIX}"
    _write_atomic(path, json.dumps({"pid": os.getpid()}) + "\n")
    return path


def take_intents(project_root: Path | str, root: Path | str | None = None) -> list[str]:
    """Return the task ids of every pending retry request, removing those files.

    Each request is renamed to ``<id>.retry.claimed`` before it is removed: one
    written while this is collecting survives for the next collection, and one
    whose consumer died between renaming and applying is taken again.
    """
    directory = pending_dir(project_root, root)
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    tids: list[str] = []
    for path in entries:
        name = path.name
        if name.endswith(RETRY_SUFFIX + CLAIM_SUFFIX):
            tid = name[: -len(RETRY_SUFFIX + CLAIM_SUFFIX)]
            claimed = path
        elif name.endswith(RETRY_SUFFIX):
            tid = name[: -len(RETRY_SUFFIX)]
            claimed = path.with_name(name + CLAIM_SUFFIX)
            try:
                os.replace(path, claimed)
            except OSError:
                continue
        else:
            continue
        tids.append(tid)
        claimed.unlink(missing_ok=True)
    return tids


# ----------------------------------------------------------------------- state file


def _write_atomic(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` via a unique temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save(state: State, path: Path | str) -> None:
    """Write ``state`` to ``path`` atomically (temp file + ``os.replace``)."""
    _write_atomic(Path(path), json.dumps(state.as_dict(), indent=2, sort_keys=True) + "\n")


def load(path: Path | str) -> State:
    """Read ``path`` into a :class:`State`; a missing file is an empty state."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return State()
    except OSError as exc:
        raise StateError(f"{path}: {exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise StateError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StateError(f"{path}: expected a JSON object")
    return State.from_dict(data)


# ------------------------------------------------------------------ process identity


def pid_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live process we can signal."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_state(pid: int) -> str | None:
    """Return ``ps``'s state letter for ``pid`` (``"Z"`` = zombie), or ``None``.

    A zombie keeps its pid but runs no code, so a caller waiting for a process
    to stop must treat it as gone — it only lingers until its parent reaps it.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    letter = out.stdout.strip()
    return letter[:1] or None


def process_start_time(pid: int) -> float | None:
    """Return ``pid``'s start time as an epoch float, or ``None`` if unknown.

    Uses ``ps -o lstart=`` (portable across macOS and Linux); a dead pid, a
    missing ``ps``, or unparseable output all report ``None``.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        parsed = time.strptime(out.stdout.strip(), _PS_TIME_FMT)
    except ValueError:
        return None
    return time.mktime(parsed)


def start_time_matches(pid: int, started: float, tolerance: float = _START_TOLERANCE) -> bool:
    """Return whether ``pid`` is alive and started at ``started`` (± ``tolerance``).

    An unknown recorded or current start time degrades to a plain liveness check,
    so a missing ``ps`` never declares a live agent dead.
    """
    if not pid_alive(pid):
        return False
    if not started:
        return True
    actual = process_start_time(pid)
    if actual is None:
        return True
    return abs(actual - started) <= tolerance


# ------------------------------------------------------------------------- lock


@dataclass(frozen=True)
class Lock:
    """The scheduler's single-instance lock: a pid plus its start time."""

    path: Path
    pid: int
    started: float

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON written into the lock file."""
        return {"pid": self.pid, "started": self.started}

    def alive(self) -> bool:
        """Return whether the lock's process is still the one that wrote it."""
        return start_time_matches(self.pid, self.started)

    def release(self) -> None:
        """Remove the lock file, but only if it is still this lock's."""
        current = _read_lock(self.path)
        if current is not None and current.pid == self.pid and current.started == self.started:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _read_lock(path: Path) -> Lock | None:
    """Parse a lock file; missing or malformed files report ``None`` (stale)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid, started = int(data["pid"]), float(data.get("started") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    return Lock(path=path, pid=pid, started=started)


def read_lock(project_root: Path | str, root: Path | str | None = None) -> Lock | None:
    """Return the project's lock as written, live or stale, or ``None``."""
    return _read_lock(lock_path(project_root, root))


def live_lock(project_root: Path | str, root: Path | str | None = None) -> Lock | None:
    """Return the project's lock only if its scheduler process is still alive."""
    lock = _read_lock(lock_path(project_root, root))
    return lock if lock is not None and lock.alive() else None


def claim_lock(project_root: Path | str, root: Path | str | None = None) -> Lock:
    """Take the project's lock for this process, clearing any stale one.

    Raises :class:`LockError` naming the live holder when another scheduler owns
    the lock (SPEC §7).
    """
    path = lock_path(project_root, root)
    holder = _read_lock(path)
    if holder is not None and holder.alive():
        raise LockError(f"{path}: scheduler already running (pid {holder.pid})")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock(path=path, pid=os.getpid(), started=process_start_time(os.getpid()) or 0.0)
    _write_atomic(path, json.dumps(lock.as_dict()))
    return lock
