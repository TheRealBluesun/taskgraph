"""Shared fixtures for the lease tests.

Not named ``test_*.py`` so unittest discovery does not collect it (the base
class has no test methods, and discovery would import it twice otherwise).
"""

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
        """Hand-write a slot file, as a crashed holder would leave behind."""
        data = {"pid": 0, "cmd": "", "cwd": "", "task": None, "since": time.time()}
        data.update(fields)
        path = lease.resource_dir(self.state, resource) / f"slot-{n}.json"
        path.write_text(json.dumps(data))
        return path

    def slot_names(self, resource="simulator") -> list:
        return [path.name for _, path in lease.slot_files(self.state, resource)]

    def audit_lines(self) -> list:
        path = lease.audit_path(self.state)
        return path.read_text().splitlines() if path.exists() else []

    def alive(self, pid: int) -> bool:
        """Staleness check that trusts everything (used to test the slot logic)."""
        return True
