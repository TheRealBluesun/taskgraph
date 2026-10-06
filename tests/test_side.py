"""Tests for ``taskgraph side`` and job-class capacities (SPEC §4).

The policy is exercised in-process (``ordered`` / ``wait_for_model``) and through
the real CLI in a temp project; the capacity semaphore is a plain ``flock`` on a
slot file under a temp ``TASKGRAPH_STATE``, so nothing here needs a model or the
network.
"""

import json
import os
import select
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from leasehelpers import BIN, LeaseTestCase, wait_until

from taskgraph import agent, lease, side, state
from taskgraph.config import ModelConfig

#: Only m1 serves the ``side`` class, so a full m1 means "wait".
SIDE_TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "../wt"
main = "main"

[agent]
command = "true"

[[models]]
name = "m1"
sessions = 1
classes = { agent = 1, side = 1 }

[[models]]
name = "m2"
sessions = 1
"""

#: Both models serve ``side``; used for the blocked-agent preference.
TWO_SIDE_TOML = SIDE_TOML.replace(
    'name = "m2"\nsessions = 1', 'name = "m2"\nsessions = 1\nclasses = { agent = 1, side = 1 }'
)


def model(name, classes=None, sessions=1):
    return ModelConfig(name=name, sessions=sessions, max_agents=sessions, classes=classes or {})


class OrderedTest(unittest.TestCase):
    """The pure preference order (SPEC §4)."""

    def test_blocked_agent_models_come_first_then_config_order(self):
        models = [model("m1", {"side": 1}), model("m2", {"side": 1}), model("m3", {"side": 1})]
        self.assertEqual(
            [m.name for m in side.ordered(models, "side", {"m3"})], ["m3", "m1", "m2"]
        )
        self.assertEqual([m.name for m in side.ordered(models, "side", ())], ["m1", "m2", "m3"])

    def test_models_without_the_class_are_not_candidates(self):
        models = [model("m1", {"agent": 1}), model("m2", {"side": 1})]
        self.assertEqual([m.name for m in side.ordered(models, "side", ())], ["m2"])
        self.assertEqual(side.ordered(models, "gpu", ()), [])

    def test_resource_name_and_waiting_message(self):
        self.assertEqual(side.resource_name("m1", "side"), "model:m1:side")
        self.assertEqual(
            side.waiting_message("side", [model("m1"), model("m2")]),
            "waiting for side capacity (models: m1, m2)",
        )

    def test_model_names_with_slashes_keep_one_flat_resource(self):
        # "local-vllm-flash/Qwen3.8-Flash-Next" must not nest a slot directory
        # (invisible to `taskgraph leases`) or split the audit line.
        resource = side.resource_name("local-vllm-flash/Qwen3.8-Flash-Next", "side")
        self.assertEqual(resource, "model:local-vllm-flash%2FQwen3.8-Flash-Next:side")
        self.assertEqual(Path(resource).name, resource)


class BlockedModelsTest(unittest.TestCase):
    """Which models have an agent blocked on a tool (SPEC §4, §9)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def record(self, model_name, log):
        return agent.AgentRecord(
            id="T1",
            pid=os.getpid(),  # this test process is alive, so the agent looks running
            pgid=os.getpid(),
            model=model_name,
            worktree=str(self.base),
            log=str(log),
            started=time.time(),
        )

    def save(self, *records):
        return state.State(agents=[record.as_dict() for record in records])

    def test_open_tool_call_marks_the_model_blocked(self):
        log = self.base / "a.log"
        log.write_text(json.dumps({"type": "tool_execution_start", "toolCallId": "c", "toolName": "bash"}) + "\n")
        self.assertEqual(side.blocked_models(self.save(self.record("m2", log))), {"m2"})

    def test_finished_tool_call_and_missing_trace_do_not(self):
        log = self.base / "b.log"
        log.write_text(
            json.dumps({"type": "tool_execution_start", "toolCallId": "c", "toolName": "bash"}) + "\n"
            + json.dumps({"type": "tool_execution_end", "toolCallId": "c"}) + "\n"
        )
        self.assertEqual(side.blocked_models(self.save(self.record("m1", log))), set())
        self.assertEqual(side.blocked_models(self.save(self.record("m1", self.base / "none.log"))), set())


class WaitForModelTest(LeaseTestCase):
    """Slot acquisition, in-process (the CLI is covered below)."""

    def test_takes_the_free_slot_of_the_preferred_model(self):
        models = [model("m1", {"side": 1}), model("m2", {"side": 1})]
        chosen, slot = side.wait_for_model(
            models, "side", root=self.state, blocked=lambda: {"m2"}, poll=0.01, message_secs=None, timeout=2
        )
        self.addCleanup(lease.release, slot, root=self.state)
        self.assertEqual(chosen.name, "m2")
        self.assertEqual([s.name for s in lease.holders(self.state, resource="model:m2:side")], ["slot-0"])

    def test_falls_back_when_the_preferred_model_is_full(self):
        models = [model("m1", {"side": 1}), model("m2", {"side": 1})]
        held = self.acquire_slot(side.resource_name("m2", "side"))
        self.assertIsNotNone(held)
        chosen, slot = side.wait_for_model(
            models, "side", root=self.state, blocked=lambda: {"m2"}, poll=0.01, message_secs=None, timeout=2
        )
        self.addCleanup(lease.release, slot, root=self.state)
        self.assertEqual(chosen.name, "m1")

    def test_timeout_returns_none(self):
        models = [model("m1", {"side": 1})]
        self.acquire_slot(side.resource_name("m1", "side"))
        self.assertIsNone(
            side.wait_for_model(
                models, "side", root=self.state, poll=0.01, message_secs=None, timeout=0.05
            )
        )


class SideCliTest(LeaseTestCase):
    """``taskgraph side`` end to end in a temp project."""

    def cli(self, cwd: Path, *args: str, timeout: float = 30.0):
        return subprocess.run(
            [sys.executable, str(BIN), *args],
            cwd=str(cwd),
            env=self.env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def spawn_side(self, project: Path, *args: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, str(BIN), "side", *args],
            cwd=str(project),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def kill():
            if proc.poll() is None:
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

    def held(self, resource) -> list:
        return [slot.name for slot in lease.holders(self.state, resource=resource)]

    def write_agent(self, project: Path, model_name: str, log: Path) -> None:
        record = agent.AgentRecord(
            id="T1",
            pid=os.getpid(),
            pgid=os.getpid(),
            model=model_name,
            worktree=str(self.base),
            log=str(log),
            started=time.time(),
        )
        state.save(state.State(agents=[record.as_dict()]), state.state_path(project, self.state))

    def test_runs_command_and_exports_the_chosen_model(self):
        project = self.project(SIDE_TOML)
        out = self.base / "model.txt"
        proc = self.cli(project, "side", "side", "--task", "S1", "--", "sh", "-c",
                        f'printf %s "$TASKGRAPH_MODEL" > "{out}"')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(out.read_text(), "m1")
        self.assertEqual(self.held("model:m1:side"), [])
        self.assertEqual([p.name for _, p in lease.slot_files(self.state, "model:m1:side")], ["slot-0"])
        self.assertIn("model:m1:side", self.audit_lines()[-1])
        self.assertIn("S1", self.audit_lines()[-1])

    def test_prefers_a_model_whose_agent_is_blocked_on_a_tool(self):
        project = self.project(TWO_SIDE_TOML)
        out = self.base / "model.txt"
        for blocked, expected in ((True, "m2"), (False, "m1")):
            with self.subTest(blocked=blocked):
                log = self.base / f"agent-{blocked}.log"
                events = [{"type": "tool_execution_start", "toolCallId": "c", "toolName": "bash"}]
                if not blocked:
                    events.append({"type": "tool_execution_end", "toolCallId": "c"})
                log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
                self.write_agent(project, "m2", log)
                proc = self.cli(project, "side", "side", "--", "sh", "-c",
                                f'printf %s "$TASKGRAPH_MODEL" > "{out}"')
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(out.read_text(), expected)

    def test_capacity_one_makes_the_second_side_job_wait(self):
        project = self.project(SIDE_TOML)
        first = self.spawn_side(project, "side", "--task", "S1", "--", "sleep", "30")
        self.assertTrue(wait_until(lambda: self.held("model:m1:side") == ["slot-0"]), "holder acquired")
        second = self.spawn_side(project, "side", "--task", "S2", "--", "sleep", "30")
        self.assertTrue(
            wait_until(lambda: bool(select.select([second.stderr], [], [], 0)[0])),
            "waiting side job wrote no message",
        )
        self.assertIn("waiting for side capacity", second.stderr.readline())
        self.assertEqual(self.held("model:m1:side"), ["slot-0"])
        self.assertEqual(self.max_concurrent("model:m1:side"), 1)
        first.terminate()
        self.assertEqual(first.wait(timeout=10), 128 + 15)  # forwarded SIGTERM, slot freed
        self.assertTrue(wait_until(lambda: self.held("model:m1:side") == ["slot-0"]), "second took it")
        second.terminate()
        second.wait(timeout=10)
        self.assertTrue(wait_until(lambda: self.held("model:m1:side") == []))

    def test_unknown_class_is_a_usage_error(self):
        project = self.project(SIDE_TOML)
        proc = self.cli(project, "side", "gpu", "--", "true")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("classes", proc.stderr)

    def test_side_without_command_is_a_usage_error(self):
        project = self.project(SIDE_TOML)
        proc = self.cli(project, "side", "side")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("cmd", proc.stderr)


if __name__ == "__main__":
    unittest.main()
