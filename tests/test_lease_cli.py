"""Tests for ``taskgraph lease`` / ``taskgraph leases`` (SPEC §5, SPEC §10).

The semaphore itself is covered in ``test_lease.py``; here we exercise the
command wrapper: exit codes, ``TASKGRAPH_LEASE``, signal forwarding, waiting,
the config lookup and the rendered holders table.
"""

import io
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from leasehelpers import BIN, MINIMAL_TOML, ROOT, LeaseTestCase, wait_until

from taskgraph import cli, lease


class FormattingTest(unittest.TestCase):
    def slot(self, **kw):
        fields = dict(
            resource="simulator",
            n=0,
            pid=4242,
            cmd="xcodebuild -scheme App",
            cwd="/w",
            task="F03",
            since=1000.0,
            path=Path("/tmp/slot-0.json"),
        )
        fields.update(kw)
        return lease.Slot(**fields)

    def test_format_age(self):
        self.assertEqual(cli.format_age(0), "0s")
        self.assertEqual(cli.format_age(59.9), "59s")
        self.assertEqual(cli.format_age(73.9), "1m13s")
        self.assertEqual(cli.format_age(125), "2m05s")
        self.assertEqual(cli.format_age(3720), "1h02m")

    def test_format_holders_table(self):
        lines = cli.format_holders([self.slot()], now=1073.0).splitlines()
        self.assertEqual(lines[0].split(), ["RESOURCE", "SLOT", "PID", "TASK", "AGE", "COMMAND"])
        for cell in ("simulator", "slot-0", "4242", "F03", "1m13s", "xcodebuild -scheme App"):
            self.assertIn(cell, lines[1])
        self.assertEqual(cli.format_holders([]), "no leases held")


class RunLeaseTest(LeaseTestCase):
    def test_returns_child_exit_code_and_releases(self):
        code = cli.run_lease("simulator", ["/bin/sh", "-c", "exit 7"], capacity=1, root=self.state)
        self.assertEqual(code, 7)
        self.assertEqual(self.slot_names(), [])
        self.assertEqual([line.split()[1] for line in self.audit_lines()], ["acquire", "release"])

    def test_exports_lease_env_to_the_child(self):
        out = self.base / "out.txt"
        code = cli.run_lease(
            "simulator",
            ["/bin/sh", "-c", f'printf %s "$TASKGRAPH_LEASE" > "{out}"'],
            capacity=1,
            root=self.state,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.read_text(), "simulator")

    def test_reports_127_when_command_is_missing(self):
        code = cli.run_lease(
            "simulator", ["/nonexistent/prog"], capacity=1, root=self.state, out=io.StringIO()
        )
        self.assertEqual(code, 127)
        self.assertEqual(self.slot_names(), [])

    def test_times_out_on_a_held_slot(self):
        project = self.project()
        holder = self.spawn_lease(project, "simulator", "--task", "F03", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.slot_names() == ["slot-0.json"]), "holder acquired")
        stream = io.StringIO()
        code = cli.run_lease(
            "simulator",
            ["/bin/sh", "-c", "true"],
            capacity=1,
            root=self.state,
            poll=0.02,
            out=stream,
            timeout=0.2,
        )
        self.assertEqual(code, cli.TIMEOUT_EXIT)
        self.assertIn("waiting for simulator (held by F03", stream.getvalue())
        self.assertEqual(self.slot_names(), ["slot-0.json"])  # the holder keeps it


class CliTest(LeaseTestCase):
    def cli(self, cwd: Path, *args: str, timeout: float = 30.0):
        return subprocess.run(
            [sys.executable, str(BIN), *args],
            cwd=str(cwd),
            env=self.env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_lease_runs_command_and_releases_the_slot(self):
        project = self.project()
        proc = self.cli(
            project, "lease", "simulator", "--task", "T07", "--", "/bin/sh", "-c", "exit 3"
        )
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.slot_names(), [])
        self.assertIn("T07", self.audit_lines()[-1])

    def test_lease_accepts_task_flag_before_and_after_the_resource(self):
        project = self.project()
        for argv in (
            ("lease", "--task", "A1", "simulator", "--", "/bin/sh", "-c", "true"),
            ("lease", "simulator", "--task", "A1", "--", "/bin/sh", "-c", "true"),
        ):
            with self.subTest(argv=argv):
                proc = self.cli(project, *argv)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("A1", self.audit_lines()[-1])

    def test_lease_rejects_unknown_resource(self):
        project = self.project()
        proc = self.cli(project, "lease", "gpu", "--", "true")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("[resources]", proc.stderr)

    def test_lease_without_config_fails_clearly(self):
        empty = self.base / "empty"
        empty.mkdir()
        proc = self.cli(empty, "lease", "simulator", "--", "true")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("taskgraph.toml", proc.stderr)

    def test_lease_without_command_is_a_usage_error(self):
        project = self.project()
        proc = self.cli(project, "lease", "simulator")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("cmd", proc.stderr)

    def test_find_config_walks_up_to_the_project_root(self):
        project = self.project()
        nested = project / "a" / "b"
        nested.mkdir(parents=True)
        self.assertEqual(cli.find_config(nested), (project / cli.CONFIG_NAME).resolve())
        self.assertIsNone(cli.find_config(self.base / "empty"))

    def test_capacity_from_config_limits_concurrent_leases(self):
        project = self.project(MINIMAL_TOML.replace("simulator = 1", "simulator = 2"))
        first = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        second = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.slot_names() == ["slot-0.json", "slot-1.json"]))
        third = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: "slot-2.json" not in self.slot_names()))
        time.sleep(0.4)
        third.terminate()
        third.wait(timeout=10)
        self.assertIn("waiting for simulator", third.stderr.read())
        for proc in (first, second):
            proc.terminate()
            proc.wait(timeout=10)
        self.assertTrue(wait_until(lambda: self.slot_names() == []))

    def group_members(self, pgid: int) -> list:
        """Pids in process group ``pgid`` (portable across macOS/Linux pgrep flags)."""
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid="], capture_output=True, text=True, timeout=10
        ).stdout
        return [
            parts[0]
            for parts in (line.split() for line in out.splitlines())
            if len(parts) >= 2 and parts[1] == str(pgid)
        ]

    def test_forwarded_signal_stops_the_whole_leased_command_tree(self):
        project = self.project()
        pidfile = self.base / "child.pid"
        holder = self.spawn_lease(
            project, "simulator", "--", "sh", "-c", f"echo $$ > {pidfile}; sleep 30"
        )

        def child_pgid() -> int:
            try:
                return int(pidfile.read_text().strip())
            except (OSError, ValueError):
                return 0

        self.assertTrue(wait_until(lambda: child_pgid() > 0), "child started")
        pgid = child_pgid()
        self.assertTrue(
            wait_until(lambda: len(self.group_members(pgid)) >= 2), "shell + sleep running"
        )
        holder.terminate()
        self.assertEqual(holder.wait(timeout=10), 128 + 15)
        self.assertTrue(wait_until(lambda: self.group_members(pgid) == []), "grandchild stopped")
        self.assertTrue(wait_until(lambda: self.slot_names() == []), "slot released")

    def test_leases_listing_shows_a_live_holder_then_nothing(self):
        project = self.project()
        holder = self.spawn_lease(project, "simulator", "--task", "F03", "--", "sleep", "30")
        try:
            self.assertTrue(wait_until(lambda: self.slot_names() == ["slot-0.json"]), "acquired")
            listed = self.cli(project, "leases")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn("simulator", listed.stdout)
            self.assertIn("slot-0", listed.stdout)
            self.assertIn("F03", listed.stdout)
        finally:
            holder.terminate()
            holder.wait(timeout=10)
        self.assertEqual(holder.returncode, 128 + 15)  # forwarded SIGTERM reaches the child
        self.assertTrue(wait_until(lambda: self.slot_names() == []), "released")
        empty = self.cli(project, "leases")
        self.assertEqual(empty.stdout.strip(), "no leases held")


if __name__ == "__main__":
    unittest.main()
