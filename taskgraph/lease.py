"""Shared resource leases (SPEC §5): one counting semaphore per resource.

Agents run in separate worktrees and may outlive a scheduler restart, so the
lock cannot live in process memory.  Each held slot is the file
``leases/<resource>/slot-<n>.json``, claimed with ``O_CREAT|O_EXCL`` so two
waiters can never win the same slot.

A slot is stale when its pid is dead, or alive but not a ``taskgraph lease``
process (that is also how a reused pid is caught); any waiter may remove it.
The holder is *always* a ``taskgraph lease`` process — see :func:`taskgraph.cli.run`
for the wrapper that takes a slot, runs the command with ``TASKGRAPH_LEASE`` set,
and releases it.  All paths come from ``$TASKGRAPH_STATE`` (default
``~/.taskgraph``), never from a project dir, so agents cannot rewrite the
machinery (SPEC §5).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

ENV_STATE = "TASKGRAPH_STATE"
LEASE_ENV = "TASKGRAPH_LEASE"
DEFAULT_STATE_DIR = "~/.taskgraph"

POLL_SECS = 2.0
MESSAGE_SECS = 30.0
PS_TIMEOUT = 5.0

_SLOT_FILE = re.compile(r"^slot-(\d+)\.json$")
# A holder's command line looks like `/usr/bin/python3 .../bin/taskgraph lease …`.
_LEASE_CMD = re.compile(r"(?:^|[\s/])taskgraph(?:\.cli)?\s.*\blease\b")


class LeaseError(Exception):
    """A lease could not be created (e.g. the state dir is not writable)."""


@dataclass(frozen=True)
class Slot:
    """One held lease slot, mirroring its ``slot-<n>.json`` file."""

    resource: str
    n: int
    pid: int
    cmd: str
    cwd: str
    task: str | None
    since: float
    path: Path

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


# ------------------------------------------------------------------- process checks


def process_command(pid: int) -> str | None:
    """Return ``ps -o lstart=,command=`` for ``pid``, or ``None`` if it is gone.

    ``None`` also covers ``ps`` being unavailable or timing out: an unreadable
    holder must never look alive.
    """
    if pid <= 0:
        return None
    try:
        proc = subprocess.run(
            ["ps", "-o", "lstart=,command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip()
    return line or None


def looks_like_lease_command(cmd: str) -> bool:
    """Whether ``cmd`` (a process command line) is a ``taskgraph lease`` call."""
    return bool(_LEASE_CMD.search(cmd))


def pid_is_live_lease(pid: int) -> bool:
    """Whether ``pid`` is a live ``taskgraph lease`` process (any project)."""
    cmd = process_command(pid)
    return cmd is not None and looks_like_lease_command(cmd)


def is_stale(slot: Slot, *, alive: Callable[[int], bool] = pid_is_live_lease) -> bool:
    """Whether ``slot``'s holder is dead, or its pid now belongs to something else."""
    return not alive(slot.pid)


# ------------------------------------------------------------------------- slots


def slot_files(root: Path | str | None, resource: str) -> list[tuple[int, Path]]:
    """Existing slot files of ``resource`` as ``(n, path)``, sorted by ``n``.

    Filenames count as taken even when their JSON is unreadable or still being
    written, so a half-written slot is never handed out twice.
    """
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


def read_slots(root: Path | str | None, resource: str) -> list[Slot]:
    """Parse the readable slots of ``resource``, sorted by slot number."""
    slots: list[Slot] = []
    for n, path in slot_files(root, resource):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            task = data.get("task")
            slots.append(
                Slot(
                    resource=resource,
                    n=n,
                    pid=int(data.get("pid", 0)),
                    cmd=str(data.get("cmd", "")),
                    cwd=str(data.get("cwd", "")),
                    task=str(task) if task is not None else None,
                    since=float(data.get("since", 0.0)),
                    path=path,
                )
            )
        except (TypeError, ValueError):
            continue
    return slots


def remove_stale(
    root: Path | str | None,
    resource: str,
    *,
    alive: Callable[[int], bool] = pid_is_live_lease,
) -> list[Slot]:
    """Delete stale slots of ``resource`` and audit each removal."""
    base = _root(root)
    removed: list[Slot] = []
    for slot in read_slots(base, resource):
        if not is_stale(slot, alive=alive):
            continue
        try:
            slot.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue
        audit(base, "stale-removed", slot)
        removed.append(slot)
    return removed


def try_acquire(
    resource: str,
    capacity: int,
    *,
    root: Path | str | None = None,
    task: str | None = None,
    cmd: Sequence[str] | str = (),
    cwd: str | None = None,
    now: float | None = None,
    alive: Callable[[int], bool] = pid_is_live_lease,
    pid: int | None = None,
) -> Slot | None:
    """Claim a free slot of ``resource``, or return ``None`` when all are held.

    Stale slots are cleared first (SPEC §5: any waiter may do this), then the
    lowest-numbered free index below ``capacity`` is claimed atomically.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    base = _root(root)
    directory = resource_dir(base, resource)
    directory.mkdir(parents=True, exist_ok=True)
    remove_stale(base, resource, alive=alive)
    used = {n for n, _ in slot_files(base, resource)}
    payload_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
    for n in range(capacity):
        if n in used:
            continue
        path = directory / f"slot-{n}.json"
        slot = Slot(
            resource=resource,
            n=n,
            pid=os.getpid() if pid is None else pid,
            cmd=payload_cmd,
            cwd=os.getcwd() if cwd is None else cwd,
            task=task,
            since=time.time() if now is None else now,
            path=path,
        )
        data = {
            "pid": slot.pid,
            "cmd": slot.cmd,
            "cwd": slot.cwd,
            "task": slot.task,
            "since": slot.since,
        }
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        except OSError as exc:
            raise LeaseError(f"{path}: cannot create slot: {exc}") from exc
        try:
            os.write(fd, json.dumps(data, sort_keys=True).encode("utf-8"))
        finally:
            os.close(fd)
        audit(base, "acquire", slot)
        return slot
    return None


def release(slot: Slot, *, root: Path | str | None = None) -> None:
    """Release ``slot``; a no-op if a waiter already removed and re-claimed it."""
    base = _root(root if root is not None else slot.path.parents[1])
    try:
        data = json.loads(slot.path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and data.get("pid") != slot.pid:
        return
    try:
        slot.path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return
    audit(base, "release", slot)


def holders(
    root: Path | str | None = None, *, alive: Callable[[int], bool] = pid_is_live_lease
) -> list[Slot]:
    """Live slots of every resource, ordered by resource then slot number."""
    base = leases_root(_root(root))
    try:
        resources = sorted(entry.name for entry in base.iterdir() if entry.is_dir())
    except OSError:
        return []
    live: list[Slot] = []
    for resource in resources:
        live.extend(s for s in read_slots(base.parent, resource) if not is_stale(s, alive=alive))
    return live


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
    alive: Callable[[int], bool] = pid_is_live_lease,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Slot | None:
    """Wait for a free slot of ``resource``, or ``None`` after ``timeout``.

    Prints ``waiting for <resource> (held by <task> for <n> s)`` on entering the
    wait and then at most every ``message_secs`` (SPEC §5).
    """
    stream = sys.stderr if out is None else out
    base = _root(root)
    started = clock()
    last_message: float | None = None
    while True:
        slot = try_acquire(
            resource, capacity, root=base, task=task, cmd=cmd, cwd=cwd, alive=alive
        )
        if slot is not None:
            return slot
        if timeout is not None and clock() - started >= timeout:
            return None
        if message_secs is not None and (
            last_message is None or clock() - last_message >= message_secs
        ):
            print(
                wait_message(resource, read_slots(base, resource)),
                file=stream,
                flush=True,
            )
            last_message = clock()
        sleep(poll)

