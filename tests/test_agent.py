"""Tests for the agent runner (SPEC §6).

Agents are faked with shell scripts, so these tests launch real detached
process groups, real logs and real watchdogs — but never omp, a model or the
network; every started group is killed on cleanup.
"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from taskgraph import agent, prompt, worktree
from taskgraph.config import ModelConfig
from agenthelpers import AgentTestCase, make_config, make_task, wait_for


class StartTest(AgentTestCase):
    def test_start_returns_a_detached_record_and_captures_output(self):
        script = self.script('echo "trace line"\nexec sleep 30')
        record = self.launch(f"{script} {{prompt_file}}")
        self.assertEqual(record.id, "T01")
        self.assertEqual(record.model, "m1")
        self.assertEqual(record.pgid, record.pid)
        self.assertGreater(record.pid, 0)
        self.assertEqual(record.worktree, os.fspath(self.worktree))
        self.assertEqual(record.retries, 0)
        self.assertGreater(record.started, 0.0)
        self.assertRegex(Path(record.log).name, r"^T01-\d{6}\.log$")
        self.assertTrue(Path(record.log).is_file())
        self.assertEqual(os.getpgid(record.pid), record.pid)  # own session
        self.wait_log(record, "trace line")
        self.assertEqual(agent.poll(record), "running")

    def test_prompt_file_is_written_git_excluded_and_passed_to_the_agent(self):
        subprocess.run(["git", "init", "-q", "-b", "main", "."], cwd=self.worktree, check=True)
        script = self.script('echo "arg1=$1"')
        record = self.launch(f"{script} {{prompt_file}}")
        prompt_path = self.worktree / prompt.PROMPT_NAME
        self.assertEqual(prompt_path, self.worktree / ".taskgraph-prompt.md")
        self.assertIn("YOUR TASK IS **T01**", prompt_path.read_text(encoding="utf-8"))
        exclude = worktree.exclude_file(self.worktree).read_text(encoding="utf-8")
        self.assertIn(prompt.PROMPT_NAME, exclude)
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", prompt.PROMPT_NAME], cwd=self.worktree
        )
        self.assertEqual(ignored.returncode, 0)  # git itself ignores the prompt file
        self.wait_log(record, f"arg1={prompt_path}")

    def test_poll_reports_exited_after_a_short_agent_finishes(self):
        script = self.script('echo bye')
        record = self.launch(f"{script}")
        self.wait_log(record, "bye")
        wait_for(lambda: agent.poll(record) == "exited")
        self.assertEqual(agent.poll(record), "exited")

    def test_retries_and_resume_note_come_from_scheduler_state(self):
        self.state.retries["T01"] = 2
        script = self.script("echo hi")
        record = self.launch(f"{script}")
        self.assertEqual(record.retries, 2)
        text = (self.worktree / prompt.PROMPT_NAME).read_text(encoding="utf-8")
        self.assertTrue(text.startswith(prompt.RESUME_NOTE))
        self.assertIn("continue from it; do not start over.", text)

    def test_explicit_resume_false_suppresses_the_note(self):
        self.state.retries["T01"] = 1
        script = self.script("echo hi")
        self.launch(f"{script}", resume=False)
        text = (self.worktree / prompt.PROMPT_NAME).read_text(encoding="utf-8")
        self.assertFalse(text.startswith(prompt.RESUME_NOTE))

    def test_missing_overlay_is_an_error(self):
        script = self.script("echo hi")
        cfg = self.config(f"{script} {{prompt_file}}", overlay="/nonexistent/agent.yml")
        with self.assertRaises(agent.AgentError):
            agent.start(make_task(), cfg.models[0], self.worktree, cfg, self.state, root=self.state_root)

    def test_bad_deny_entry_is_an_error(self):
        script = self.script("echo hi")
        cfg = self.config(f"{script} {{prompt_file}}", deny=("/usr/bin/xcodebuild",))
        with self.assertRaises(agent.AgentError):
            agent.start(make_task(), cfg.models[0], self.worktree, cfg, self.state, root=self.state_root)

    def test_unknown_command_placeholder_is_an_error(self):
        cfg = self.config("agent {nope}")
        with self.assertRaises(agent.AgentError):
            agent.start(make_task(), cfg.models[0], self.worktree, cfg, self.state, root=self.state_root)

    def test_deny_shims_are_installed_and_first_on_the_agents_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp)
            tool = bindir / "xcodebuild"
            tool.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
            tool.chmod(0o755)
            path = os.pathsep.join([os.fspath(bindir), os.environ.get("PATH", "")])
            with mock.patch.dict(os.environ, {"PATH": path}):
                script = self.script('echo "PATH=$PATH"')
                cfg = self.config(f"{script} {{prompt_file}}", deny=("xcodebuild",))
                record = self.launch(f"{script} {{prompt_file}}", cfg=cfg)
        self.assertTrue((self.state_root / "shims" / "xcodebuild").is_file())
        first = self.wait_log(record, "PATH=").split("PATH=", 1)[1].splitlines()[0]
        self.assertEqual(first.split(os.pathsep)[0], os.fspath(self.state_root / "shims"))


class PollTest(unittest.TestCase):
    def record(self, pid: int) -> agent.AgentRecord:
        return agent.AgentRecord(
            id="T01", pid=pid, pgid=pid, model="m1", worktree="/w", log="/w/x.log", started=0.0
        )

    def test_zero_and_missing_pids_report_exited(self):
        self.assertEqual(agent.poll(self.record(0)), "exited")
        self.assertEqual(agent.poll(self.record(-1)), "exited")

    def test_dead_pid_reports_exited(self):
        # A pid that certainly never existed.
        self.assertEqual(agent.poll(self.record(2**30)), "exited")


class RecordTest(unittest.TestCase):
    def test_round_trips_through_state_json(self):
        record = agent.AgentRecord(
            id="F03", pid=12, pgid=12, model="m1", worktree="/w", log="/w/l.log", started=1.5, retries=2
        )
        self.assertEqual(agent.AgentRecord.from_dict(record.as_dict()), record)

    def test_missing_pgid_defaults_to_pid(self):
        data = {"id": "F03", "pid": 12, "model": "m1", "worktree": "/w", "log": "/l", "started": 1.0}
        self.assertEqual(agent.AgentRecord.from_dict(data).pgid, 12)

    def test_bad_record_is_reported(self):
        with self.assertRaises(agent.AgentError):
            agent.AgentRecord.from_dict({"model": "m1"})


class StallTest(AgentTestCase):
    def test_stalled_uses_the_trace_mtime(self):
        log = self.base / "trace.log"
        log.write_text("x", encoding="utf-8")
        fresh = log.stat().st_mtime
        self.assertTrue(agent.stalled(log, fresh + 10, 5))
        self.assertFalse(agent.stalled(log, fresh + 4.9, 5))
        self.assertFalse(agent.stalled(self.base / "missing.log", fresh + 100, 5))

    def test_log_size_and_last_activity_default_for_missing_files(self):
        self.assertEqual(agent.log_size(self.base / "nope.log"), 0)
        self.assertEqual(agent.last_activity(self.base / "nope.log"), 0.0)

    def test_kill_stops_a_long_running_agent_group(self):
        script = self.script("echo up\nexec sleep 30")
        record = self.launch(f"{script}")
        self.wait_log(record, "up")
        self.assertTrue(agent.group_alive(record.pgid))
        agent.kill(record, timeout=5.0)
        self.assertFalse(agent.group_alive(record.pgid))
        self.assertEqual(agent.poll(record), "exited")

    def test_kill_escalates_to_sigkill_when_sigterm_is_ignored(self):
        script = self.script("trap '' TERM\necho up\nexec sleep 30")
        record = self.launch(f"{script}")
        self.wait_log(record, "up")
        started = time.monotonic()
        agent.kill(record, timeout=0.3)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertFalse(agent.group_alive(record.pgid))

    def test_group_alive_is_false_for_a_bogus_pgid(self):
        self.assertFalse(agent.group_alive(0))
        self.assertFalse(agent.group_alive(2**30))


class StartupHangTest(AgentTestCase):
    def log(self, text: str) -> Path:
        path = self.base / "trace.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_true_while_no_tool_started_and_phrase_begins_a_head_line(self):
        self.assertTrue(agent.startup_hang(self.log("Still starting after 4s\nmore\n")))

    def test_false_once_a_tool_has_started(self):
        self.assertFalse(
            agent.startup_hang(self.log('{"type":"tool_execution_start"}\nStill starting after 4s\n'))
        )

    def test_false_when_the_phrase_is_past_the_head(self):
        self.assertFalse(agent.startup_hang(self.log("\n" * 20 + "Still starting after 1s\n")))

    def test_phrase_within_the_head_is_detected_at_the_last_line(self):
        self.assertTrue(agent.startup_hang(self.log("\n" * 19 + "Still starting after 1s\n")))

    def test_phrase_must_begin_the_line(self):
        self.assertFalse(agent.startup_hang(self.log("2026-01-01 Still starting after 1s\n")))

    def test_missing_file_is_false(self):
        self.assertFalse(agent.startup_hang(self.base / "nope.log"))


class QuotaTest(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = make_config(
            self.root,
            "agent {prompt_file}",
            models=(
                ModelConfig(name="m1", sessions=1, max_agents=2, fallback="m2"),
                ModelConfig(name="m2", sessions=1, max_agents=1),
            ),
        )

    def record(self, text: str | None) -> agent.AgentRecord:
        log = self.base / "trace.log"
        if text is not None:
            log.write_text(text, encoding="utf-8")
        return agent.AgentRecord(
            id="T01", pid=1, pgid=1, model="m1", worktree="/w", log=str(log), started=0.0
        )

    def test_quota_markers_are_case_insensitive(self):
        for text in (
            "HTTP 429: rate limit exceeded",
            "You exceeded your current quota",
            "Too many requests, slow down",
            "error: rate_limit_error",
        ):
            with self.subTest(text=text):
                self.assertTrue(agent.quota_error(self.record(text).log))

    def test_no_quota_error(self):
        self.assertFalse(agent.quota_error(self.record("all good\n").log))
        self.assertFalse(agent.quota_error(self.record(None).log))

    def test_fallback_is_only_returned_with_a_quota_error(self):
        self.assertEqual(agent.fallback_model(self.record("rate limit hit"), self.cfg).name, "m2")
        self.assertIsNone(agent.fallback_model(self.record("fine"), self.cfg))

    def test_no_fallback_configured_returns_none(self):
        cfg = make_config(self.root, "agent {prompt_file}")
        self.assertIsNone(agent.fallback_model(self.record("quota exceeded"), cfg))

    def test_unknown_model_returns_none(self):
        record = self.record("quota exceeded")
        record.model = "ghost"
        self.assertIsNone(agent.fallback_model(record, self.cfg))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
