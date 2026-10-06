"""Tests for lease-wrapper announcements (SPEC §5, §6).

A ``taskgraph lease`` wrapper announces itself under the state root for its
whole life, so the scheduler's stall watchdog can tell that an otherwise silent
agent is queued for or holding a resource.  Nothing else in taskgraph reads
these files, so they are exercised both in-process and with real wrappers.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from leasehelpers import LeaseTestCase, wait_until

from taskgraph import lease, leaseprocs


class ProcessRegistryTest(LeaseTestCase):
    """Live wrappers announce themselves so the stall watchdog sees them (SPEC §6)."""

    def test_register_and_unregister_round_trip(self):
        path = leaseprocs.register_process(
            "simulator", root=self.state, cmd=["xcodebuild", "-scheme", "App"], cwd="/tmp/x"
        )
        self.assertTrue(path.is_file())
        proc = leaseprocs.active_processes(self.state)[0]
        self.assertEqual(
            (proc.pid, proc.resource, proc.cwd, proc.cmd),
            (os.getpid(), "simulator", "/tmp/x", "xcodebuild -scheme App"),
        )
        leaseprocs.unregister_process(path)
        self.assertFalse(path.exists())
        self.assertEqual(leaseprocs.active_processes(self.state), [])

    def test_defaults_describe_the_registering_process(self):
        path = leaseprocs.register_process("simulator", root=self.state, now=100.0)
        self.addCleanup(leaseprocs.unregister_process, path)
        proc = leaseprocs.active_processes(self.state)[0]
        self.assertEqual(proc.pid, os.getpid())
        self.assertEqual(proc.cwd, os.getcwd())
        self.assertEqual(proc.age(now=130.0), 30.0)

    def test_oldest_first_and_resource_filter(self):
        newer = leaseprocs.register_process("simulator", root=self.state, now=200.0)
        older = leaseprocs.register_process("gpu", root=self.state, now=100.0)
        self.addCleanup(leaseprocs.unregister_process, newer)
        self.addCleanup(leaseprocs.unregister_process, older)
        self.assertEqual([p.since for p in leaseprocs.active_processes(self.state)], [100.0, 200.0])
        self.assertEqual(
            [p.resource for p in leaseprocs.active_processes(self.state, resource="gpu")], ["gpu"]
        )

    def test_dead_pid_is_pruned_on_read(self):
        dead = subprocess.Popen([sys.executable, "-c", ""])
        dead.wait()
        path = leaseprocs.register_process("simulator", root=self.state, pid=dead.pid)
        self.assertTrue(path.exists())
        self.assertEqual(leaseprocs.active_processes(self.state), [])
        self.assertFalse(path.exists())

    def test_malformed_or_foreign_files_are_ignored(self):
        directory = leaseprocs.procs_root(self.state)
        directory.mkdir(parents=True)
        (directory / "junk.json").write_text("{not json", encoding="utf-8")
        (directory / "README").write_text("not an announcement", encoding="utf-8")
        self.assertEqual(leaseprocs.active_processes(self.state), [])
        self.assertFalse((directory / "junk.json").exists())  # pruned
        self.assertTrue((directory / "README").exists())  # not ours, left alone

    def test_a_wrapper_announces_while_waiting_and_holding(self):
        project = self.project()
        held = self.acquire_slot()  # capacity 1: the spawned wrapper has to wait
        waiting = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: len(leaseprocs.active_processes(self.state)) == 1))
        announced = leaseprocs.active_processes(self.state)[0]
        self.assertEqual(announced.pid, waiting.pid)
        self.assertEqual(announced.resource, "simulator")
        self.assertEqual(Path(announced.cwd).resolve(), project.resolve())
        self.assertEqual(self.held_names(), ["slot-0"])  # the test still holds it

        self.release_slot(held)  # the waiting wrapper takes the slot
        self.assertTrue(
            wait_until(lambda: [s.pid for s in lease.holders(self.state)] == [waiting.pid])
        )
        self.assertEqual(len(leaseprocs.active_processes(self.state)), 1)

        waiting.terminate()
        waiting.wait(timeout=10)
        self.assertTrue(wait_until(lambda: leaseprocs.active_processes(self.state) == []))


if __name__ == "__main__":
    unittest.main()
