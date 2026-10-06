"""``taskgraph status`` tests (SPEC §9).

Two layers: :func:`taskgraph.status.render` is pure and driven from a fixture
:class:`ProjectStatus`; :func:`taskgraph.status.collect` runs against a temp
project with a real ``state.json``, worktree ``.done`` markers and a held lease
slot, and one test drives the real ``bin/taskgraph status`` CLI. No omp, no
model, no network.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from taskgraph import config, lease, state, status

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "taskgraph"

PLAN = """\
- [ ] T01 first
- [ ] T02 second [deps: T01]
- [ ] T03 third
- [ ] T05 fifth
- [x] T04 already done
"""

TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "wt"
main = "main"

[resources]
simulator = 1

[agent]
command = "true"

[[models]]
name = "m1"
sessions = 2
metrics = "http://127.0.0.1:1/metrics"

[[models]]
name = "m2"
sessions = 1
"""


def trace_line(kind: str, call_id: str, **fields) -> str:
    """One omp JSON trace line."""
    return json.dumps({"type": kind, "toolCallId": call_id, **fields})


def slot(**kw) -> lease.Slot:
    """A lease slot fixture for the holders table."""
    fields = dict(
        resource="simulator",
        n=0,
        pid=4242,
        cmd="xcodebuild -scheme App",
        cwd="/w",
        task="T01",
        since=1000.0,
        path=Path("/tmp/slot-0"),
    )
    fields.update(kw)
    return lease.Slot(**fields)


class FormattingTest(unittest.TestCase):
    def test_format_age(self):
        self.assertEqual(status.format_age(0), "0s")
        self.assertEqual(status.format_age(59.9), "59s")
        self.assertEqual(status.format_age(73.9), "1m13s")
        self.assertEqual(status.format_age(125), "2m05s")
        self.assertEqual(status.format_age(3720), "1h02m")

    def test_format_holders_table(self):
        lines = status.format_holders([slot()], now=1073.0).splitlines()
        self.assertEqual(lines[0].split(), ["RESOURCE", "SLOT", "PID", "TASK", "AGE", "COMMAND"])
        for cell in ("simulator", "slot-0", "4242", "T01", "1m13s", "xcodebuild -scheme App"):
            self.assertIn(cell, lines[1])
        self.assertEqual(status.format_holders([]), "no leases held")


class CurrentToolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "trace.log"

    def write(self, *lines):
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_open_call_is_current(self):
        self.write(trace_line("tool_execution_start", "1", toolName="read"))
        self.assertEqual(status.current_tool(self.log), "read")

    def test_closed_call_is_not_current(self):
        self.write(
            trace_line("tool_execution_start", "1", toolName="read"),
            trace_line("tool_execution_end", "1", toolName="read"),
        )
        self.assertIsNone(status.current_tool(self.log))

    def test_parallel_calls_report_the_newest(self):
        self.write(
            trace_line("tool_execution_start", "1", toolName="read"),
            trace_line("tool_execution_start", "2", toolName="task"),
            trace_line("tool_execution_end", "1", toolName="read"),
        )
        self.assertEqual(status.current_tool(self.log), "task")

    def test_stray_end_and_junk_lines_are_ignored(self):
        self.write(
            trace_line("tool_execution_end", "9", toolName="read"),
            "not json at all: tool_execution_start",
            json.dumps({"type": "message_update", "text": "tool_execution_start"}),
            trace_line("tool_execution_start", "3", toolName="bash"),
        )
        self.assertEqual(status.current_tool(self.log), "bash")

    def test_missing_trace(self):
        self.assertIsNone(status.current_tool(self.log))


class RenderTest(unittest.TestCase):
    def test_renders_every_section(self):
        snapshot = status.ProjectStatus(
            running=(status.AgentStatus("T01", "m1", 125.0, 3.0, "bash"),),
            merging=("T02",),
            blocked=(("T03", "gate failed: boom"),),
            runnable=("T02", "T03"),
            holders=(slot(),),
            load=(status.ModelLoad("m1", 1.25, 0.0), status.ModelLoad("m2", None, None)),
        )
        lines = status.render(snapshot, now=1073.0).splitlines()

        self.assertEqual(lines[0], "running (1)")
        self.assertEqual(lines[1].split(), ["ID", "MODEL", "AGE", "IDLE", "TOOL"])
        self.assertEqual(lines[2].split(), ["T01", "m1", "2m05s", "3s", "bash"])
        self.assertEqual(lines[3], "merge queue (1)")
        self.assertEqual(lines[4], "  T02")
        self.assertEqual(lines[5], "blocked (1)")
        self.assertEqual(lines[6], "  T03  gate failed: boom")
        self.assertEqual(lines[7], "next runnable (2)")
        self.assertEqual(lines[8:10], ["  T02", "  T03"])
        self.assertEqual(lines[10], "leases (1)")
        self.assertEqual(lines[11].split(), ["RESOURCE", "SLOT", "PID", "TASK", "AGE", "COMMAND"])
        self.assertIn("simulator", lines[12])
        self.assertEqual(lines[13], "model load (2)")
        self.assertEqual(lines[14].split(), ["MODEL", "RUNNING", "WAITING"])
        self.assertEqual(lines[15].split(), ["m1", "1.25", "0.00"])
        self.assertEqual(lines[16].split(), ["m2", "-", "-"])

    def test_empty_status_says_none_in_every_section(self):
        lines = status.render(status.ProjectStatus()).splitlines()
        self.assertEqual(
            lines,
            [
                "running (0)",
                "  none",
                "merge queue (0)",
                "  none",
                "blocked (0)",
                "  none",
                "next runnable (0)",
                "  none",
                "leases (0)",
                "  none",
                "model load (0)",
                "  none",
            ],
        )


class ProjectTestCase(unittest.TestCase):
    """A temp project: config, plan, state dir and a worktree base."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.state_root = self.base / "state"
        (self.root / "PLAN.md").write_text(PLAN, encoding="utf-8")
        (self.root / "PROMPT.md").write_text("rules\n", encoding="utf-8")
        (self.root / "taskgraph.toml").write_text(TOML, encoding="utf-8")
        self.cfg = config.load(self.root / "taskgraph.toml")
        self.now = time.time()
        self._slots = []

    def tearDown(self):
        for held in self._slots:
            lease.release(held, root=self.state_root)

    def save(self, **fields):
        state.save(state.State(**fields), state.state_path(self.root, self.state_root))

    def trace(self, tid, *lines):
        """Write ``tid``'s fake trace and return its path."""
        log = self.base / f"{tid}.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return log

    def record(self, tid, log, **fields):
        """A state.json agent record for a live process (this test process)."""
        data = {
            "id": tid,
            "pid": os.getpid(),
            "pgid": os.getpid(),
            "model": "m1",
            "worktree": str(self.root / "wt" / tid),
            "log": str(log),
            "started": self.now - 125.0,
            "retries": 0,
        }
        data.update(fields)
        return data

    def done_marker(self, tid):
        """Create ``wt/<tid>/progress/<tid>.done`` (a finished agent's worktree)."""
        progress = self.root / "wt" / tid / "progress"
        progress.mkdir(parents=True, exist_ok=True)
        (progress / f"{tid}.done").write_text("", encoding="utf-8")


class CollectTest(ProjectTestCase):
    def test_running_agent_shows_model_age_and_tool(self):
        trace = self.trace("T01", trace_line("tool_execution_start", "1", toolName="bash"))
        self.save(
            agents=[self.record("T01", trace)],
            blocked={"T03": "gate failed:\nexit 1"},
        )
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now + 125.0)

        self.assertEqual([a.id for a in snapshot.running], ["T01"])
        agent_status = snapshot.running[0]
        self.assertEqual(
            (agent_status.model, agent_status.age, agent_status.tool), ("m1", 250.0, "bash")
        )
        self.assertIsNotNone(agent_status.idle)
        self.assertEqual(snapshot.blocked, (("T03", "gate failed: exit 1"),))
        # T01 runs, T02 waits on it, T03 is blocked: only T05 may start.
        self.assertEqual(snapshot.runnable, ("T05",))

    def test_done_marker_is_the_merge_queue(self):
        self.done_marker("T05")
        self.save()
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual(snapshot.merging, ("T05",))
        # A finished task waits for the merge worker, it is not started again.
        self.assertEqual(snapshot.runnable, ("T01", "T03"))

    def test_order_with_nothing_running(self):
        self.save()
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual(snapshot.running, ())
        # T01 unblocks T02, so T01 leads the critical path; then plan order.
        self.assertEqual(snapshot.runnable, ("T01", "T03", "T05"))

    def test_dead_agent_record_is_not_running(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        log = self.trace("T01", trace_line("tool_execution_start", "1", toolName="read"))
        self.save(agents=[self.record("T01", log, pid=proc.pid, pgid=proc.pid)])
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual(snapshot.running, ())
        self.assertEqual(snapshot.runnable, ("T01", "T03", "T05"))

    def test_blocked_task_with_done_marker_is_not_queued(self):
        self.done_marker("T03")
        self.save(blocked={"T03": "conflict in a.py"})
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual(snapshot.merging, ())
        self.assertEqual([tid for tid, _ in snapshot.blocked], ["T03"])

    def test_held_lease_is_listed(self):
        held = lease.try_acquire(
            "simulator", 1, root=self.state_root, task="T01", cmd=["xcodebuild"], now=self.now
        )
        self.assertIsNotNone(held)
        self._slots.append(held)
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual([s.resource for s in snapshot.holders], ["simulator"])
        self.assertIn("xcodebuild", status.render(snapshot, now=self.now))

    def test_model_load_uses_the_recent_window(self):
        def sample(at, running, waiting):
            return {
                "at": at,
                "metrics": {"running": running, "waiting": waiting, "generation_tokens": 0.0},
            }

        self.save(
            samples={
                "m1": [sample(self.now - 10.0, 2.0, 1.0), sample(self.now - 20.0, 0.0, 0.0)],
                "m2": [sample(self.now - 3600.0, 3.0, 3.0)],  # stale: outside the window
            }
        )
        snapshot = status.collect(self.cfg, state_root=self.state_root, now=self.now)
        self.assertEqual(
            [(row.name, row.running, row.waiting) for row in snapshot.load],
            [("m1", 1.0, 0.5), ("m2", None, None)],
        )


class CliTest(ProjectTestCase):
    def run_status(self, env=None):
        return subprocess.run(
            [sys.executable, str(BIN), "status"],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_prints_the_table_for_a_fixture_state(self):
        self.save(agents=[self.record("T01", self.trace("T01"))], blocked={"T02": "boom"})
        self.done_marker("T03")
        env = dict(os.environ, TASKGRAPH_STATE=str(self.state_root))
        result = self.run_status(env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("running (1)", result.stdout)
        self.assertIn("T01", result.stdout)
        self.assertIn("blocked (1)", result.stdout)
        self.assertIn("  T02  boom", result.stdout)
        self.assertIn("merge queue (1)", result.stdout)
        self.assertIn("m1", result.stdout)

    def test_malformed_state_is_reported(self):
        path = state.state_path(self.root, self.state_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        result = self.run_status(dict(os.environ, TASKGRAPH_STATE=str(self.state_root)))
        self.assertEqual(result.returncode, 2)
        self.assertIn("taskgraph status:", result.stderr)

    def test_no_config_is_reported(self):
        result = subprocess.run(
            [sys.executable, str(BIN), "status"], cwd=self.base, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no taskgraph.toml", result.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
