"""Tests for the shared resource lease semaphore (SPEC §5).

No network and no real user state: everything lives in a temp ``TASKGRAPH_STATE``.
Ownership is an ``fcntl.flock`` on ``leases/<resource>/slot-<n>``, so lock
mechanics are exercised in-process, while contention and lock release on death
use real ``taskgraph lease`` subprocesses.
"""

import fcntl
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import unittest
from pathlib import Path

from leasehelpers import ROOT, LeaseTestCase, wait_until

from taskgraph import lease

#: Eight real lease processes, driven through ``cli.run_lease`` with a short
#: poll so the test does not pay the 2 s production poll for each hand-off.
DRIVER = """\
import sys
sys.path.insert(0, {root!r})
from taskgraph import cli
raise SystemExit(cli.run_lease(sys.argv[1], sys.argv[4:], capacity=int(sys.argv[2]), poll=float(sys.argv[3])))
"""

POLL = 0.05


def guard_command(*dirs: Path) -> list[str]:
    """A shell command failing with exit 9 if more than ``len(dirs)`` overlap.

    Each concurrent run claims one guard directory and holds it for a moment, so
    any breach of the lease capacity makes one of them exit non-zero.
    """
    quoted = [shlex.quote(str(d)) for d in dirs]
    lines = []
    for i, q in enumerate(quoted):
        keyword = "if" if i == 0 else "elif"
        lines.append(f"{keyword} mkdir {q} 2>/dev/null; then d={q};")
    lines.append("else exit 9; fi")
    return ["sh", "-c", " ".join(lines) + '; sleep 0.2; rmdir "$d"']


class LayoutTest(LeaseTestCase):
    def test_state_root_from_env_and_default(self):
        self.assertEqual(lease.state_root({lease.ENV_STATE: "/tmp/x"}), Path("/tmp/x"))
        self.assertEqual(lease.state_root({}), Path("~/.taskgraph").expanduser())

    def test_slot_files_are_created_once_and_never_deleted(self):
        slot = self.acquire_slot()
        self.assertEqual(self.slot_names(), ["slot-0"])
        self.release_slot(slot)
        self.assertEqual(self.slot_names(), ["slot-0"])
        self.assertEqual(lease.holders(self.state), [])


class AcquireReleaseTest(LeaseTestCase):
    def test_acquire_claims_lowest_slot_and_writes_info(self):
        slot = self.acquire_slot(task="F03", cmd=["xcodebuild", "-scheme", "App"])
        self.assertIsNotNone(slot)
        self.assertEqual((slot.n, slot.name, slot.path.name), (0, "slot-0", "slot-0"))
        self.assertEqual(slot.pid, os.getpid())
        self.assertEqual(slot.cmd, "xcodebuild -scheme App")
        self.assertIsNotNone(slot.fd)
        data = json.loads(slot.path.read_text())
        self.assertEqual(sorted(data), ["cmd", "cwd", "pid", "since", "task"])
        self.assertEqual(data["cmd"], "xcodebuild -scheme App")
        self.assertEqual(data["task"], "F03")

    def test_capacity_one_second_acquire_is_denied(self):
        self.assertIsNotNone(self.acquire_slot())
        self.assertIsNone(self.acquire_slot())

    def test_capacity_two_uses_two_slots_then_denies(self):
        first = self.acquire_slot(capacity=2)
        second = self.acquire_slot(capacity=2)
        self.assertEqual([first.n, second.n], [0, 1])
        self.assertIsNone(self.acquire_slot(capacity=2))

    def test_release_frees_only_its_own_slot(self):
        first = self.acquire_slot(capacity=2)
        second = self.acquire_slot(capacity=2)
        self.release_slot(first)
        again = self.acquire_slot(capacity=2)
        self.assertEqual(again.n, 0)
        self.release_slot(second)
        self.release_slot(again)

    def test_release_of_a_display_slot_is_a_no_op(self):
        slot = lease.Slot(
            resource="simulator",
            n=0,
            pid=1,
            cmd="",
            cwd="/w",
            task=None,
            since=0.0,
            path=lease.resource_dir(self.state, "simulator") / "slot-0",
        )
        lease.release(slot, root=self.state)
        self.assertEqual(self.audit_lines(), [])

    def test_audit_records_acquire_and_release(self):
        slot = self.acquire_slot(task="F03", cwd="/w")
        self.release_slot(slot)
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


class FlockTest(LeaseTestCase):
    def test_info_json_without_a_lock_is_not_a_holder(self):
        self.write_slot("simulator", 0, pid=os.getpid(), task="crashed")
        self.assertEqual(lease.holders(self.state), [])
        slot = self.acquire_slot(task="fresh")
        self.assertEqual(slot.n, 0)
        self.assertEqual(json.loads(slot.path.read_text())["task"], "fresh")

    def test_a_holder_that_wrote_no_json_still_blocks(self):
        fd = self.hold_fd("simulator", 0)
        try:
            self.assertIsNone(self.acquire_slot())
            held = lease.holders(self.state)
            self.assertEqual([slot.name for slot in held], ["slot-0"])
            self.assertEqual(held[0].pid, 0)  # no info was ever written
        finally:
            os.close(fd)
        self.assertIsNotNone(self.acquire_slot())

    def test_closing_the_fd_drops_the_lock(self):
        path = lease.resource_dir(self.state, "simulator") / "slot-0"
        fd = self.hold_fd("simulator", 0)
        self.assertIsNone(self.acquire_slot())
        os.close(fd)
        slot = self.acquire_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(slot.path, path)

    def test_lock_is_cross_process_and_dropped_when_the_holder_is_killed(self):
        path = lease.resource_dir(self.state, "simulator") / "slot-0"
        marker = self.base / "locked"
        script = (
            "import fcntl, os, sys, time\n"
            f"fd = os.open({str(path)!r}, os.O_RDWR | os.O_CREAT, 0o644)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            f"with open({str(marker)!r}, 'w') as fh:\n"
            "    fh.write('locked')\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", script])

        def kill():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

        self.addCleanup(kill)
        self.assertTrue(wait_until(marker.exists), "child never locked the slot")
        self.assertTrue(wait_until(lambda: lease.holders(self.state) != []))
        self.assertIsNone(self.acquire_slot())
        proc.kill()
        proc.wait(timeout=10)
        self.assertTrue(wait_until(lambda: lease.holders(self.state) == []), "kernel kept the lock")
        self.assertIsNotNone(self.acquire_slot())


class WaitForSlotTest(LeaseTestCase):
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
                path=Path("/tmp/slot-0"),
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
                path=Path(f"/tmp/slot-{n}"),
            )

        message = lease.wait_message(
            "simulator", [slot(1, 900.0, "old"), slot(0, 990.0, "new")], now=1000.0
        )
        self.assertEqual(message, "waiting for simulator (held by old for 100 s)")

    def test_timeout_returns_none_and_prints_waiting_line(self):
        holder = self.acquire_slot(task="F03")
        stream = io.StringIO()
        slot = lease.wait_for_slot(
            "simulator", 1, root=self.state, task="T07", poll=0.02, out=stream, timeout=0.2
        )
        self.assertIsNone(slot)
        self.assertIn("waiting for simulator (held by F03 for 0 s)", stream.getvalue())
        self.assertEqual(self.held_names(), ["slot-0"])
        self.release_slot(holder)

    def test_waits_until_the_holder_releases(self):
        holder = self.acquire_slot(task="F03")
        threading.Timer(0.15, self.release_slot, args=(holder,)).start()
        slot = lease.wait_for_slot(
            "simulator",
            1,
            root=self.state,
            task="T07",
            poll=0.02,
            out=io.StringIO(),
            timeout=5,
        )
        self.assertIsNotNone(slot)
        self.assertEqual(slot.task, "T07")
        self.release_slot(slot)


class ContentionTest(LeaseTestCase):
    def spawn_driver(self, capacity: int, cmd: list[str]) -> subprocess.Popen:
        """Start a real lease process running ``cmd`` under ``capacity`` slots."""
        driver = self.base / "drive.py"
        if not driver.exists():
            driver.write_text(DRIVER.format(root=str(ROOT)))
        proc = subprocess.Popen(
            [sys.executable, "-u", str(driver), "simulator", str(capacity), str(POLL), *cmd],
            cwd=str(self.base),
            env=self.env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        def kill():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

        self.addCleanup(kill)
        return proc

    def run_eight(self, capacity: int, *dirs: Path):
        """Run 8 concurrent leases of one guard command; return (code, stderr)."""
        procs = [self.spawn_driver(capacity, guard_command(*dirs)) for _ in range(8)]
        results = []
        for proc in procs:
            code = proc.wait(timeout=60)
            results.append((code, proc.stderr.read()))
            proc.stderr.close()
        return results

    def test_capacity_one_is_never_exceeded_by_eight_leases(self):
        results = self.run_eight(1, self.base / "guard-1")
        self.assertEqual([code for code, _ in results], [0] * 8, results)
        self.assertEqual(self.max_concurrent(), 1)

    def test_capacity_two_is_never_exceeded_by_eight_leases(self):
        results = self.run_eight(2, self.base / "guard-a", self.base / "guard-b")
        self.assertEqual([code for code, _ in results], [0] * 8, results)
        self.assertLessEqual(self.max_concurrent(), 2)


if __name__ == "__main__":
    unittest.main()
