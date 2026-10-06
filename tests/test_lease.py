"""Tests for the shared resource lease semaphore (SPEC §5).

No network and no real user state: everything lives in a temp ``TASKGRAPH_STATE``.
Contention and staleness against *live* holders use real ``taskgraph lease``
subprocesses, because the staleness rule only trusts real lease processes.
"""

import json
import io
import os
import select
import subprocess
import sys
import unittest
from pathlib import Path

from leasehelpers import LeaseTestCase, wait_until

from taskgraph import lease


class AcquireReleaseTest(LeaseTestCase):
    def test_state_root_from_env_and_default(self):
        self.assertEqual(lease.state_root({lease.ENV_STATE: "/tmp/x"}), Path("/tmp/x"))
        self.assertEqual(lease.state_root({}), Path("~/.taskgraph").expanduser())

    def test_acquire_claims_lowest_slot_and_writes_fields(self):
        slot = lease.try_acquire(
            "simulator",
            1,
            root=self.state,
            task="F03",
            cmd=["xcodebuild", "-scheme", "App"],
            alive=self.alive,
        )
        self.assertIsNotNone(slot)
        self.assertEqual((slot.n, slot.name), (0, "slot-0"))
        self.assertEqual(slot.pid, os.getpid())
        self.assertEqual(slot.task, "F03")
        self.assertEqual(slot.cmd, "xcodebuild -scheme App")
        self.assertTrue(slot.path.is_file())
        data = json.loads(slot.path.read_text())
        self.assertEqual(sorted(data), ["cmd", "cwd", "pid", "since", "task"])
        self.assertEqual(data["cmd"], "xcodebuild -scheme App")

    def test_capacity_one_second_acquire_is_denied(self):
        self.assertIsNotNone(lease.try_acquire("simulator", 1, root=self.state, alive=self.alive))
        self.assertIsNone(lease.try_acquire("simulator", 1, root=self.state, alive=self.alive))
        self.assertEqual(self.slot_names(), ["slot-0.json"])

    def test_capacity_two_uses_two_slots_then_denies(self):
        first = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive)
        second = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive)
        self.assertEqual([first.n, second.n], [0, 1])
        self.assertIsNone(lease.try_acquire("simulator", 2, root=self.state, alive=self.alive))

    def test_release_frees_only_its_own_slot(self):
        first = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive)
        second = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive)
        lease.release(first, root=self.state)
        self.assertEqual(self.slot_names(), ["slot-1.json"])
        again = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive)
        self.assertEqual(again.n, 0)
        lease.release(second, root=self.state)
        lease.release(again, root=self.state)
        self.assertEqual(self.slot_names(), [])

    def test_release_leaves_a_reclaimed_slot_alone(self):
        slot = lease.try_acquire("simulator", 1, root=self.state, alive=self.alive)
        lease.release(slot, root=self.state)
        other = lease.try_acquire(
            "simulator", 1, root=self.state, alive=self.alive, pid=slot.pid + 1
        )
        lease.release(slot, root=self.state)  # stale handle from the first holder
        self.assertTrue(other.path.is_file())

    def test_audit_records_acquire_and_release(self):
        slot = lease.try_acquire(
            "simulator", 1, root=self.state, task="F03", cwd="/w", alive=self.alive
        )
        lease.release(slot, root=self.state)
        actions = [line.split()[1:4] for line in self.audit_lines()]
        self.assertEqual(
            actions, [["acquire", "simulator", "slot-0"], ["release", "simulator", "slot-0"]]
        )
        fields = self.audit_lines()[0].split()
        self.assertEqual(fields[4], str(os.getpid()))
        self.assertEqual(fields[5], "F03")
        self.assertEqual(fields[6], "/w")

    def test_capacity_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            lease.try_acquire("simulator", 0, root=self.state)

    def test_unreadable_slot_still_blocks_the_index(self):
        (lease.resource_dir(self.state, "simulator") / "slot-0.json").write_text("{truncated")
        self.assertIsNone(lease.try_acquire("simulator", 1, root=self.state, alive=self.alive))
        self.assertEqual(lease.read_slots(self.state, "simulator"), [])


class StaleTest(LeaseTestCase):
    def test_dead_pid_slot_is_removed_and_reacquired(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=30)
        self.write_slot("simulator", 0, pid=proc.pid, task="old")
        slot = lease.try_acquire("simulator", 1, root=self.state)  # real ps check
        self.assertIsNotNone(slot)
        self.assertEqual(slot.pid, os.getpid())
        self.assertEqual(
            [line.split()[1] for line in self.audit_lines()], ["stale-removed", "acquire"]
        )

    def test_live_pid_that_is_not_a_lease_process_is_stale(self):
        self.write_slot("simulator", 0, pid=os.getpid(), task="copycat")
        removed = lease.remove_stale(self.state, "simulator")
        self.assertEqual([slot.task for slot in removed], ["copycat"])
        self.assertEqual(lease.read_slots(self.state, "simulator"), [])

    def test_is_stale_uses_injected_alive_check(self):
        slot = lease.try_acquire("simulator", 1, root=self.state, alive=self.alive)
        self.assertFalse(lease.is_stale(slot, alive=self.alive))
        self.assertTrue(lease.is_stale(slot, alive=lambda pid: False))

    def test_process_command_for_dead_and_live_pids(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=30)
        self.assertIsNone(lease.process_command(proc.pid))
        self.assertIsNone(lease.process_command(0))
        self.assertIsNotNone(lease.process_command(os.getpid()))

    def test_looks_like_lease_command(self):
        yes = [
            "taskgraph lease simulator -- xcodebuild",
            "/usr/bin/python3 /opt/taskgraph/bin/taskgraph lease simulator -- xcodebuild -x",
            "python3 -m taskgraph.cli lease simulator -- true",
            "/Users/x/taskgraph/bin/taskgraph lease simulator -- sleep 1",
        ]
        no = [
            "taskgraph leases",
            "python3 -m taskgraph.cli leases",
            "/bin/sh -c ./dev.sh build",
            "/bin/sleep 10",
            "taskgraph run --project .",
        ]
        for cmd in yes:
            with self.subTest(cmd=cmd):
                self.assertTrue(lease.looks_like_lease_command(cmd))
        for cmd in no:
            with self.subTest(cmd=cmd):
                self.assertFalse(lease.looks_like_lease_command(cmd))


class WaitForSlotTest(LeaseTestCase):
    def test_removes_a_stale_slot_and_takes_its_place(self):
        self.write_slot("simulator", 0, pid=os.getpid(), task="copycat")
        slot = lease.wait_for_slot(
            "simulator", 1, root=self.state, task="F03", poll=0.01, message_secs=0.01
        )
        self.assertIsNotNone(slot)
        self.assertEqual((slot.n, slot.task), (0, "F03"))

    def test_timeout_returns_none_and_prints_waiting_line(self):
        self.write_slot("simulator", 0, pid=os.getpid(), task="F03")
        stream = io.StringIO()
        slot = lease.wait_for_slot(
            "simulator",
            1,
            root=self.state,
            task="T07",
            poll=0.02,
            out=stream,
            timeout=0.15,
            alive=self.alive,
        )
        self.assertIsNone(slot)
        self.assertIn("waiting for simulator (held by F03 for 0 s)", stream.getvalue())

    def test_wait_message_names_the_task_and_age(self):
        def slot(**kw):
            fields = dict(
                resource="simulator",
                n=0,
                pid=4242,
                cmd="xcodebuild",
                cwd="/w",
                task="F03",
                since=1000.0,
                path=Path("/tmp/slot-0.json"),
            )
            fields.update(kw)
            return lease.Slot(**fields)

        self.assertEqual(
            lease.wait_message("simulator", [slot()], now=1073.0),
            "waiting for simulator (held by F03 for 73 s)",
        )
        self.assertEqual(
            lease.wait_message("simulator", [slot(task=None)], now=1000.0),
            "waiting for simulator (held by ? for 0 s)",
        )
        self.assertEqual(
            lease.wait_message("simulator", [], now=1000.0),
            "waiting for simulator (held by ? for 0 s)",
        )

    def test_wait_message_reports_the_oldest_holder(self):
        def slot(n, since, task):
            return lease.Slot(
                resource="simulator",
                n=n,
                pid=100 + n,
                cmd="",
                cwd="/w",
                task=task,
                since=since,
                path=Path(f"/tmp/slot-{n}.json"),
            )

        message = lease.wait_message("simulator", [slot(1, 900.0, "old"), slot(0, 990.0, "new")], now=1000.0)
        self.assertEqual(message, "waiting for simulator (held by old for 100 s)")

    def test_holders_lists_only_live_slots(self):
        live = lease.try_acquire("simulator", 2, root=self.state, alive=self.alive, pid=os.getpid())
        self.write_slot("simulator", 1, pid=os.getpid() + 1, task="copycat")

        def alive_check(pid: int) -> bool:
            return pid == os.getpid()

        names = [slot.name for slot in lease.holders(self.state, alive=alive_check)]
        self.assertEqual(names, ["slot-0"])
        self.assertEqual(lease.holders(self.state, alive=lambda pid: False), [])
        self.assertTrue(live.path.is_file())


class ContentionTest(LeaseTestCase):
    def test_second_holder_waits_for_capacity_one(self):
        project = self.project()
        first = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.slot_names() == ["slot-0.json"]), "first holds")
        second = self.spawn_lease(project, "simulator", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.slot_names() == ["slot-0.json"]))
        # The waiting line is printed before the first 2 s poll; wait for that
        # write instead of a fixed sleep (startup stretches under load).
        self.assertTrue(
            wait_until(lambda: bool(select.select([second.stderr], [], [], 0)[0])),
            "second lease wrote no message",
        )
        second.terminate()  # no child yet: default SIGTERM disposition applies
        second.wait(timeout=10)
        self.assertIn("waiting for simulator (held by ? for", second.stderr.read())
        first.terminate()
        first.wait(timeout=10)
        self.assertTrue(wait_until(lambda: self.slot_names() == []), "slot released")


if __name__ == "__main__":
    unittest.main()
