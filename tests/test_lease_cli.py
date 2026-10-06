"""Tests for ``taskgraph lease`` / ``taskgraph leases`` (SPEC §5, SPEC §10).

The semaphore itself is covered in ``test_lease.py``; here we exercise the
command wrapper: exit codes, ``TASKGRAPH_LEASE``, signal forwarding, waiting
and the config lookup. The table rendering lives in ``test_status.py``.
"""

import io
import os
import select
import signal
import subprocess
import sys
import threading
import unittest
from pathlib import Path

from leasehelpers import BIN, MINIMAL_TOML, LeaseTestCase, wait_until

from taskgraph import cli


def _pid_alive(pid: int) -> bool:
    """True while ``pid`` still exists (a reaped process returns False)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RunLeaseTest(LeaseTestCase):
    def test_returns_child_exit_code_and_releases(self):
        code = cli.run_lease("simulator", ["/bin/sh", "-c", "exit 7"], capacity=1, root=self.state)
        self.assertEqual(code, 7)
        self.assertEqual(self.held_names(), [])
        self.assertEqual(self.slot_names(), ["slot-0"])  # never deleted
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
        self.assertEqual(self.held_names(), [])
        self.assertEqual(self.slot_names(), ["slot-0"])

    def test_times_out_on_a_held_slot(self):
        project = self.project()
        holder = self.spawn_lease(project, "simulator", "--task", "F03", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0"]), "holder acquired")
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
        self.assertEqual(self.held_names(), ["slot-0"])  # the holder keeps it


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
        self.assertEqual(self.held_names(), [])
        self.assertEqual(self.slot_names(), ["slot-0"])
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
        self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0", "slot-1"]))
        third = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        # The first waiting line is printed before the first 2 s poll; wait for
        # that write rather than a fixed sleep (startup stretches under load).
        self.assertTrue(
            wait_until(lambda: bool(select.select([third.stderr], [], [], 0)[0])),
            "waiting lease wrote no message",
        )
        self.assertEqual(self.held_names(), ["slot-0", "slot-1"])
        third.terminate()
        third.wait(timeout=10)
        self.assertIn("waiting for simulator", third.stderr.read())
        for proc in (first, second):
            proc.terminate()
            proc.wait(timeout=10)
        self.assertTrue(wait_until(lambda: self.held_names() == []))

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
        self.assertTrue(wait_until(lambda: self.held_names() == []), "slot released")

    def test_leases_listing_shows_a_live_holder_then_nothing(self):
        project = self.project()
        holder = self.spawn_lease(project, "simulator", "--task", "F03", "--", "sleep", "30")
        try:
            self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0"]), "acquired")
            listed = self.cli(project, "leases")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn("simulator", listed.stdout)
            self.assertIn("slot-0", listed.stdout)
            self.assertIn("F03", listed.stdout)
        finally:
            holder.terminate()
            holder.wait(timeout=10)
        self.assertEqual(holder.returncode, 128 + 15)  # forwarded SIGTERM reaches the child
        self.assertTrue(wait_until(lambda: self.held_names() == []), "released")
        empty = self.cli(project, "leases")
        self.assertEqual(empty.stdout.strip(), "no leases held")


class SignalTest(LeaseTestCase):
    """A lease process dying by any means frees its slot (SPEC §5)."""

    def child_pgid(self, pidfile: Path) -> int:
        try:
            return int(pidfile.read_text().strip())
        except (OSError, ValueError):
            return 0

    def holder(self, project: Path, tag: str) -> subprocess.Popen:
        """Start a lease whose child writes its pgid, and wait until it holds."""
        pidfile = self.base / f"{tag}.pid"
        proc = self.spawn_lease(
            project,
            "simulator",
            "--task",
            "F03",
            "--",
            "sh",
            "-c",
            f"echo $$ > {pidfile}; sleep 30",
        )

        def kill_group():
            pgid = self.child_pgid(pidfile)
            if pgid > 0:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass

        self.addCleanup(kill_group)
        self.assertTrue(wait_until(lambda: self.child_pgid(pidfile) > 0), "child started")
        self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0"]), "slot held")
        return proc

    def assert_released(self):
        self.assertTrue(wait_until(lambda: self.held_names() == []), "slot freed")
        self.assertIsNotNone(self.acquire_slot())

    def signal_case(self, tag: str, signum: int) -> None:
        """Signal the wrapper; the slot frees only once its child tree is gone."""
        project = self.project()
        proc = self.holder(project, tag)
        pgid = self.child_pgid(self.base / f"{tag}.pid")
        os.kill(proc.pid, signum)
        self.assertEqual(proc.wait(timeout=10), 128 + int(signum))
        self.assertTrue(wait_until(lambda: self.group_members(pgid) == []), "child tree gone")
        self.assert_released()

    def test_sigterm_frees_the_slot(self):
        proc = self.holder(self.project(), "term")
        proc.terminate()
        self.assertEqual(proc.wait(timeout=10), 128 + 15)
        self.assert_released()

    def test_sigint_frees_the_slot(self):
        proc = self.holder(self.project(), "int")
        os.kill(proc.pid, signal.SIGINT)
        proc.wait(timeout=10)
        self.assert_released()

    def test_sighup_frees_the_slot(self):
        self.signal_case("hup", signal.SIGHUP)

    def test_sigquit_frees_the_slot(self):
        self.signal_case("quit", signal.SIGQUIT)

    def test_sigkill_frees_the_slot(self):
        proc = self.holder(self.project(), "kill")
        proc.kill()
        proc.wait(timeout=10)
        self.assert_released()

    def test_signal_while_waiting_aborts_without_taking_the_slot(self):
        project = self.project()
        first = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0"]), "holder acquired")
        second = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(
            wait_until(lambda: bool(select.select([second.stderr], [], [], 0)[0])),
            "waiter wrote no message",
        )
        second.terminate()
        self.assertEqual(second.wait(timeout=10), 128 + signal.SIGTERM)
        self.assertEqual(self.held_names(), ["slot-0"])  # the holder keeps it
        first.terminate()
        first.wait(timeout=10)
        self.assertTrue(wait_until(lambda: self.held_names() == []))


class GrandchildTest(LeaseTestCase):
    """A dead direct child must not release the slot while its tree still runs."""

    def kill_pid(self, pidfile: Path) -> None:
        try:
            pid = int(pidfile.read_text())
        except (OSError, ValueError):
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    def test_slot_released_only_after_the_grandchild_dies(self):
        project = self.project()
        grand = self.base / "grand.py"
        grand.write_text(
            "import os, signal, sys, time\n"
            "for s in (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM):\n"
            "    signal.signal(s, signal.SIG_IGN)\n"
            "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        direct = self.base / "direct.py"
        direct.write_text(
            "import os, subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
            "open(sys.argv[3], 'w').write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        grand_pid = self.base / "grand.pid"
        direct_pid = self.base / "direct.pid"
        result: dict = {}

        def run() -> None:
            result["code"] = cli.run_lease(
                "simulator",
                [sys.executable, str(direct), str(grand), str(grand_pid), str(direct_pid)],
                capacity=1,
                root=self.state,
                group_grace=0.8,
            )

        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(self.kill_pid, grand_pid)
        self.addCleanup(thread.join, 15)

        def pid(path: Path) -> int:
            try:
                return int(path.read_text())
            except (OSError, ValueError):
                return 0

        self.assertTrue(wait_until(lambda: pid(grand_pid) and pid(direct_pid)), "tree started")
        self.assertTrue(wait_until(lambda: self.held_names() == ["slot-0"]), "slot held")
        gp = pid(grand_pid)
        os.kill(pid(direct_pid), signal.SIGKILL)  # direct child dies, grandchild survives
        seen = {"alive_while_held": False, "released_while_alive": False}

        def track() -> bool:
            alive = _pid_alive(gp)
            held = self.held_names() == ["slot-0"]
            if alive and held:
                seen["alive_while_held"] = True
            if alive and not held:
                seen["released_while_alive"] = True
            return not alive

        self.assertTrue(wait_until(track, timeout=15), "grandchild was never killed")
        self.assertTrue(seen["alive_while_held"], "slot was not held while it lived")
        self.assertFalse(seen["released_while_alive"], "slot released before the tree was gone")
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive(), "lease wrapper did not return")
        self.assertEqual(result.get("code"), 128 + signal.SIGKILL)
        self.assertTrue(wait_until(lambda: self.held_names() == []), "slot released")


if __name__ == "__main__":
    unittest.main()
