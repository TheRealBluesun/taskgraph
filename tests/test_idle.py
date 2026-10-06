"""``taskgraph.idle`` unit tests (SPEC §9): the idle clock, anomaly dedup, diagnosis.

Everything runs against a temp state dir, so the lease/proc scans never touch
the user's ``~/.taskgraph``.  Agents are records only — the watch reads their
trace path, and a lease holder/waiter is a real slot/announcement under the temp
root. No omp, no model, no network.
"""

import os
import tempfile
import unittest
from pathlib import Path

from taskgraph import agent, idle, lease, leaseprocs, metrics, pool
from taskgraph.config import ModelConfig

NOW = 1_000_000.0
URL = "http://127.0.0.1:8002/metrics"


def model(name="m1"):
    return ModelConfig(name=name, sessions=1, max_agents=2, metrics=URL)


def sample(at, running=0.0, tokens=0.0):
    return pool.Sample(at, metrics.Metrics(running, 0.0, tokens))


def record(tid="T01", log="/nonexistent"):
    return agent.AgentRecord(
        id=tid,
        pid=os.getpid(),
        pgid=os.getpid(),
        model="m1",
        worktree="/w",
        log=str(log),
        started=NOW,
    )


class WatchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state_root = tmp.name
        self.events: list[tuple[str, str, str]] = []
        self.watch = idle.IdleWatch()

    def emit(self, kind, tid, detail):
        self.events.append((kind, tid, detail))

    def run_watch(self, now, samples, running=()):
        self.watch.watch([model()], running, {"m1": samples}, now, self.state_root, self.emit)

    def test_three_minutes_of_zero_running_is_reported_once(self):
        self.run_watch(NOW, [sample(NOW)], [record()])
        self.run_watch(NOW + 179.0, [sample(NOW + 179.0)], [record()])
        self.assertEqual(self.events, [])  # 179 s is not yet 3 min
        self.run_watch(NOW + 180.0, [sample(NOW + 180.0)], [record()])
        self.assertEqual([kind for kind, _, _ in self.events], ["idle"])
        self.assertEqual(self.events[0][1], "m1")
        self.run_watch(NOW + 200.0, [sample(NOW + 200.0)], [record()])
        self.assertEqual(len(self.events), 1)  # one line per idle episode

    def test_a_stale_sample_never_starts_the_clock(self):
        # A restarted scheduler finds old samples in state.json; only a sample
        # this process just recorded may start the clock.
        self.run_watch(NOW, [sample(NOW - 1000.0)], [record()])
        self.run_watch(NOW + 180.0, [], [record()])
        self.assertEqual(self.events, [])

    def test_load_resets_the_clock(self):
        self.run_watch(NOW, [sample(NOW)], [record()])
        self.run_watch(NOW + 100.0, [sample(NOW + 100.0, running=2.0)], [record()])
        self.run_watch(NOW + 200.0, [sample(NOW + 200.0)], [record()])
        self.assertEqual(self.events, [])
        self.run_watch(NOW + 380.0, [sample(NOW + 380.0)], [record()])
        self.assertEqual([kind for kind, _, _ in self.events], ["idle"])

    def test_no_assigned_agents_means_no_idle_event(self):
        self.run_watch(NOW, [sample(NOW)])
        self.run_watch(NOW + 200.0, [sample(NOW + 200.0)])
        self.assertEqual(self.events, [])

    def test_a_model_without_a_fresh_sample_is_never_idle(self):
        self.run_watch(NOW, [sample(NOW)], [record()])
        self.run_watch(NOW + 100.0, [sample(NOW)], [record()])  # scrape skipped
        self.run_watch(NOW + 200.0, [sample(NOW)], [record()])
        self.assertEqual(self.events, [])


class DiagnosisTest(unittest.TestCase):
    """The idle line names the agents' last tool call plus holders and waiters."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.state_root = str(self.base / "state")
        self.log = self.base / "T01.log"
        self.log.write_text(
            '{"type":"tool_execution_start","toolCallId":"1","toolName":"bash"}\n',
            encoding="utf-8",
        )
        os.utime(self.log, (NOW + 150.0, NOW + 150.0))  # 30 s before the idle line
        self.events: list[tuple[str, str, str]] = []

    def emit(self, kind, tid, detail):
        self.events.append((kind, tid, detail))

    def idle_line(self):
        watch = idle.IdleWatch()
        for offset in (0.0, 180.0):
            watch.watch(
                [model()],
                [record(log=self.log)],
                {"m1": [sample(NOW + offset)]},
                NOW + offset,
                self.state_root,
                self.emit,
            )
        line = next(detail for kind, _, detail in self.events if kind == "idle")
        return line, self.events

    def test_agent_note_and_no_leases(self):
        line, events = self.idle_line()
        self.assertEqual(line, "agents T01 bash 30s ago; holders none; waiters none")
        self.assertEqual([kind for kind, _, _ in events], ["idle"])

    def test_holders_and_waiters_are_listed(self):
        announcement = leaseprocs.register_process("sim", root=self.state_root, pid=os.getpid())
        self.addCleanup(leaseprocs.unregister_process, announcement)
        held = lease.try_acquire(
            "sim", 1, root=self.state_root, task="T02", now=NOW - 60.0, pid=os.getpid()
        )
        self.addCleanup(lease.release, held, root=self.state_root)

        line, events = self.idle_line()
        self.assertIn("holders sim/slot-0 task T02 4m00s", line)
        self.assertIn("waiters none", line)
        self.assertEqual([kind for kind, _, _ in events], ["idle"])

    def test_a_waiting_wrapper_is_listed_as_a_waiter(self):
        announcement = leaseprocs.register_process(
            "sim", root=self.state_root, pid=os.getpid(), now=NOW
        )
        self.addCleanup(leaseprocs.unregister_process, announcement)

        line, _ = self.idle_line()
        self.assertIn("holders none", line)
        self.assertIn(f"waiters sim pid {os.getpid()} 3m00s", line)


if __name__ == "__main__":
    unittest.main()
