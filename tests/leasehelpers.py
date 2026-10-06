"""Shared fixtures for the lease tests.

Not named ``test_*.py`` so unittest discovery does not collect it (the base
class has no test methods, and discovery would import it twice otherwise).
"""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from taskgraph import lease

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "taskgraph"

MINIMAL_TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "../wt"
main = "main"

[resources]
simulator = 1

[agent]
command = "true"

[[models]]
name = "m"
sessions = 1
"""


def wait_until(predicate, timeout=10.0, interval=0.02):
    """Poll ``predicate`` until true or ``timeout``; return the final result."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


class LeaseTestCase(unittest.TestCase):
    """Temp state dir (and temp project config) for one lease test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.state = self.base / "state"
        self.state.mkdir()
        lease.resource_dir(self.state, "simulator").mkdir(parents=True)
        self._held: list[lease.Slot] = []
        self.addCleanup(self._release_held)

    def _release_held(self):
        """Release every slot a test acquired in this process (once each)."""
        while self._held:
            lease.release(self._held.pop(), root=self.state)

    def acquire_slot(self, resource="simulator", capacity=1, **kwargs):
        """Take a slot in-process and release it once the test ends."""
        slot = lease.try_acquire(resource, capacity, root=self.state, **kwargs)
        if slot is not None:
            self._held.append(slot)
        return slot

    def release_slot(self, slot):
        """Release a slot taken by :meth:`acquire_slot` (exactly once)."""
        try:
            self._held.remove(slot)
        except ValueError:
            return
        lease.release(slot, root=self.state)

    def held_names(self, resource="simulator") -> list:
        """Names of the slots that are actually locked right now."""
        return [slot.name for slot in lease.holders(self.state, resource=resource)]

    def hold_fd(self, resource: str, n: int) -> int:
        """Take and keep the flock on ``slot-<n>`` writing no info JSON.

        The caller owns the returned fd and must close it exactly once.
        """
        path = lease.resource_dir(self.state, resource) / f"slot-{n}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd

    def project(self, toml=MINIMAL_TOML) -> Path:
        """Create a temp project dir with a ``taskgraph.toml`` and return it."""
        project = self.base / "project"
        project.mkdir(exist_ok=True)
        (project / "taskgraph.toml").write_text(toml)
        return project

    def env(self) -> dict:
        return {**os.environ, lease.ENV_STATE: str(self.state)}

    def spawn_lease(self, project: Path, *args: str) -> subprocess.Popen:
        """Start ``taskgraph lease`` in ``project`` and register cleanup."""
        proc = subprocess.Popen(
            [sys.executable, str(BIN), "lease", *args],
            cwd=str(project),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def kill():
            if proc.poll() is None:
                # SIGTERM first: the wrapper forwards it to the leased command's
                # own process group, so no orphan is left behind.
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

        self.addCleanup(kill)
        return proc

    def write_slot(self, resource: str, n: int, **fields) -> Path:
        """Write a slot's info JSON without locking it (a crashed holder)."""
        data = {"pid": 0, "cmd": "", "cwd": "", "task": None, "since": time.time()}
        data.update(fields)
        path = lease.resource_dir(self.state, resource) / f"slot-{n}"
        path.write_text(json.dumps(data))
        return path

    def slot_names(self, resource="simulator") -> list:
        return [path.name for _, path in lease.slot_files(self.state, resource)]

    def audit_lines(self) -> list:
        path = lease.audit_path(self.state)
        return path.read_text().splitlines() if path.exists() else []

    def max_concurrent(self, resource="simulator") -> int:
        """Peak simultaneous holders recorded in the audit log.

        ``release`` is logged before the lock is closed, so a new holder's
        ``acquire`` can never appear first: the recorded order is real order.
        """
        depth = peak = 0
        for line in self.audit_lines():
            fields = line.split()
            if len(fields) < 3 or fields[2] != resource:
                continue
            if fields[1] == "acquire":
                depth += 1
                peak = max(peak, depth)
            elif fields[1] == "release":
                depth -= 1
        return peak
