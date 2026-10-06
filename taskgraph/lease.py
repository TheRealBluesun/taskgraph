"""Shared resource leases (SPEC §5): one counting semaphore per resource.

Agents run in separate worktrees and may outlive a scheduler restart, so the
semaphore cannot live in process memory.  Each slot is the file
``leases/<resource>/slot-<n>`` (``n < capacity``), created on first use and
never deleted.  A slot is *held* iff some process holds
``fcntl.flock(fd, LOCK_EX | LOCK_NB)`` on it: the lease process takes the lock,
truncates and writes ``{pid, cmd, cwd, task, since}`` into the file (status and
audit only — never used to decide ownership), keeps the fd open for the whole
lease and releases by closing it.  The kernel drops the lock the instant the
holder exits, crashes or is killed, so there is no stale state to detect and no
pid/``ps`` guessing (SPEC §5).

Waiters try each slot's lock in turn every ``POLL_SECS``.  ``taskgraph leases``
probes the same locks: a slot whose lock it can take is free, so it closes it
again and reports only the slots that stayed locked.

All paths come from ``$TASKGRAPH_STATE`` (default ``~/.taskgraph``), never from
a project dir, so agents cannot rewrite the machinery (SPEC §5).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

ENV_STATE = "TASKGRAPH_STATE"
LEASE_ENV = "TASKGRAPH_LEASE"
DEFAULT_STATE_DIR = "~/.taskgraph"

POLL_SECS = 2.0
MESSAGE_SECS = 30.0

#: Shown by readers for a held slot whose info JSON is empty or half-written.
STARTING = "starting…"

_SLOT_FILE = re.compile(r"^slot-(\d+)$")


class LeaseError(Exception):
    """A lease could not be created (e.g. the state dir is not writable)."""


class WaitInterrupted(Exception):
    """A terminating signal arrived while waiting for a slot (SPEC §5)."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class Slot:
    """One lease slot: its ``slot-<n>`` file plus the info JSON written there.

    ``fd`` is the lock-holding descriptor and is set only on slots returned by
    :func:`try_acquire`; slots probed back for display carry ``fd = None``.
    """

    resource: str
    n: int
    pid: int
    cmd: str
    cwd: str
    task: str | None
    since: float
    path: Path
    fd: int | None = field(default=None, compare=False, repr=False)

    @property
    def name(self) -> str:
        """The slot file's name, e.g. ``slot-0``."""
        return f"slot-{self.n}"

    def age(self, now: float | None = None) -> float:
        """Seconds since acquisition (never negative)."""
        return max(0.0, (time.time() if now is None else now) - self.since)


# --------------------------------------------------------------------------- paths


def state_root(env: Mapping[str, str] | None = None) -> Path:
    """Return ``$TASKGRAPH_STATE``, defaulting to ``~/.taskgraph``."""
    raw = (os.environ if env is None else env).get(ENV_STATE)
    return Path(raw).expanduser() if raw else Path(DEFAULT_STATE_DIR).expanduser()


def leases_root(root: Path | str | None = None) -> Path:
    """Return the directory holding every resource's slots."""
    return _root(root) / "leases"


def resource_dir(root: Path | str | None, resource: str) -> Path:
    """Return the slot directory of one resource."""
    return leases_root(root) / resource


def audit_path(root: Path | str | None = None) -> Path:
    """Return the append-only lease audit log."""
    return leases_root(root) / "audit.log"


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else state_root()


# ------------------------------------------------------------------------- slots


def slot_files(root: Path | str | None, resource: str) -> list[tuple[int, Path]]:
    """Existing slot files of ``resource`` as ``(n, path)``, sorted by ``n``."""
    found: list[tuple[int, Path]] = []
    try:
        entries = list(resource_dir(root, resource).iterdir())
    except OSError:
        return found
    for path in entries:
        match = _SLOT_FILE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def holders(
    root: Path | str | None = None, *, resource: str | None = None
) -> list[Slot]:
    """Slots currently held, ordered by resource then slot number.

    Ownership is probed, not read: a slot whose ``flock`` can be taken is free,
    so it is closed again and skipped (SPEC §5).  ``resource`` limits the scan
    to one resource instead of every directory under ``leases/``.
    """
    base = _root(root)
    if resource is None:
        try:
            resources = sorted(
                entry.name for entry in leases_root(base).iterdir() if entry.is_dir()
            )
        except OSError:
            return []
    else:
        resources = [resource]
    live: list[Slot] = []
    for name in resources:
        for n, path in slot_files(base, name):
            held = _held_slot(name, n, path)
            if held is not None:
                live.append(held)
    return live


def _held_slot(resource: str, n: int, path: Path) -> Slot | None:
    """Return ``path``'s slot info if its lock is taken, else ``None``.

    The probe fd is closed on both paths: ``holders`` runs every status tick.
    """
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        slot = _read_slot(resource, n, path)
        _close(fd)
        return slot
    _close(fd)
    return None


def _close(fd: int) -> None:
    """Close ``fd``, ignoring an already-closed/invalid descriptor."""
    try:
        os.close(fd)
    except OSError:
        pass


def _read_slot(resource: str, n: int, path: Path) -> Slot:
    """Best-effort parse of a slot's info JSON; unknown fields stay empty."""
    data: dict = {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            data = parsed
    except (OSError, ValueError):
        pass
    try:
        pid = int(data.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    try:
        since = float(data.get("since", 0.0))
    except (TypeError, ValueError):
        since = 0.0
    task = data.get("task")
    cmd = str(data.get("cmd") or "").strip()
    return Slot(
        resource=resource,
        n=n,
        pid=pid,
        cmd=cmd or STARTING,
        cwd=str(data.get("cwd", "")),
        task=str(task) if task is not None else None,
        since=since,
        path=path,
    )


def try_acquire(
    resource: str,
    capacity: int,
    *,
    root: Path | str | None = None,
    task: str | None = None,
    cmd: Sequence[str] | str = (),
    cwd: str | None = None,
    now: float | None = None,
    pid: int | None = None,
) -> Slot | None:
    """Claim a free slot of ``resource``, or return ``None`` when all are held.

    Tries each slot below ``capacity`` in turn: open it (creating it once, if
    needed) and take its lock without blocking.  The returned slot holds that
    lock until :func:`release` closes its fd.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    base = _root(root)
    directory = resource_dir(base, resource)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LeaseError(f"{directory}: cannot create lease dir: {exc}") from exc
    payload_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
    for n in range(capacity):
        path = directory / f"slot-{n}"
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            raise LeaseError(f"{path}: cannot open slot: {exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        try:
            slot = Slot(
                resource=resource,
                n=n,
                pid=os.getpid() if pid is None else pid,
                cmd=payload_cmd,
                cwd=os.getcwd() if cwd is None else cwd,
                task=task,
                since=time.time() if now is None else now,
                path=path,
                fd=fd,
            )
            _write_info(slot)
            audit(base, "acquire", slot)
        except BaseException:
            # The lock is ours; close it on every failure path so a caller's
            # exception (or KeyboardInterrupt) cannot leak a held slot.
            _close(fd)
            raise
        return slot
    return None


def _write_info(slot: Slot) -> None:
    """Truncate ``slot``'s file and write its info JSON (best effort)."""
    data = {
        "pid": slot.pid,
        "cmd": slot.cmd,
        "cwd": slot.cwd,
        "task": slot.task,
        "since": slot.since,
    }
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    try:
        os.lseek(slot.fd, 0, os.SEEK_SET)
        os.ftruncate(slot.fd, 0)
        written = 0
        while written < len(payload):
            written += os.write(slot.fd, payload[written:])
    except OSError:
        pass


def release(slot: Slot, *, root: Path | str | None = None) -> None:
    """Release ``slot`` by closing its lock fd (SPEC §5).

    A slot read back for display has ``fd = None`` and is left alone.  The
    audit line is written *before* the close so the log can never show a new
    holder's ``acquire`` before the previous holder's ``release``.
    """
    if slot.fd is None:
        return
    base = _root(root) if root is not None else slot.path.parents[1].parent
    audit(base, "release", slot)
    try:
        os.close(slot.fd)
    except OSError:
        pass


# ------------------------------------------------------------------------- audit


def audit(root: Path | str | None, action: str, slot: Slot) -> None:
    """Append one ``time action resource slot pid task cwd`` line (best effort)."""
    path = audit_path(root)
    line = " ".join(
        (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            action,
            slot.resource,
            slot.name,
            str(slot.pid),
            slot.task or "-",
            slot.cwd or "-",
        )
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------- waiting


def wait_message(resource: str, slots: Sequence[Slot], now: float | None = None) -> str:
    """The one-line message printed while waiting for ``resource``."""
    oldest = max(slots, key=lambda slot: slot.age(now), default=None)
    who = (oldest.task if oldest and oldest.task else None) or "?"
    age = int(oldest.age(now)) if oldest else 0
    return f"waiting for {resource} (held by {who} for {age} s)"


def wait_for_slot(
    resource: str,
    capacity: int,
    *,
    root: Path | str | None = None,
    task: str | None = None,
    cmd: Sequence[str] | str = (),
    cwd: str | None = None,
    poll: float = POLL_SECS,
    message_secs: float | None = MESSAGE_SECS,
    out: TextIO | None = None,
    timeout: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    abort: Callable[[], int | None] | None = None,
) -> Slot | None:
    """Wait for a free slot of ``resource``, or ``None`` after ``timeout``.

    Prints ``waiting for <resource> (held by <task> for <n> s)`` on entering the
    wait and then at most every ``message_secs`` (SPEC §5).  Slots are retried
    every ``poll`` seconds.  ``abort`` is polled each round; a signal number
    raises :class:`WaitInterrupted` (the CLI blocks terminating signals and
    must abort rather than swallow them).
    """
    stream = sys.stderr if out is None else out
    base = _root(root)
    started = clock()
    last_message: float | None = None
    while True:
        slot = try_acquire(resource, capacity, root=base, task=task, cmd=cmd, cwd=cwd)
        if slot is not None:
            return slot
        if abort is not None:
            signum = abort()
            if signum is not None:
                raise WaitInterrupted(signum)
        if timeout is not None and clock() - started >= timeout:
            return None
        if message_secs is not None and (
            last_message is None or clock() - last_message >= message_secs
        ):
            print(
                wait_message(resource, holders(base, resource=resource)),
                file=stream,
                flush=True,
            )
            last_message = clock()
        sleep(poll)
