"""``taskgraph stop`` and ``taskgraph retry`` (SPEC §7, §10).

The scheduler is a real ``bin/taskgraph run`` process (or an in-process
``Scheduler``) driving the fake agent script from the scheduler tests; the state
directory is a temp dir, so nothing here touches ``~/.taskgraph``, omp or the
network.
"""

import contextlib
import io
import json
import os
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from test_scheduler import PLAN_1, ProjectTestCase, wait_until

from taskgraph import agent, cli, prompt, state, worktree

DONE_PLAN = "- [x] T01 already done\n"


def kill_group(pgid):
    """Best-effort SIGKILL of a process group (test cleanup)."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


class ControlTestCase(ProjectTestCase):
    """The scheduler fixtures plus an in-process CLI runner and agent records."""

    def run_cli(self, argv):
        """Run a subcommand against this project; return ``(rc, stdout, stderr)``."""
        out, err = io.StringIO(), io.StringIO()
        env = {"TASKGRAPH_STATE": str(self.state_root)}
        with mock.patch.dict(os.environ, env), contextlib.chdir(self.root):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def start_scheduler(self, *, fake_sleep="30"):
        """Spawn a real ``taskgraph run`` and wait until it has started one agent."""
        self.write_toml(sessions=1)
        env = {
            **os.environ,
            "TASKGRAPH_STATE": str(self.state_root),
            "TASKGRAPH_TICK": "0.05",
            "FAKE_SLEEP": fake_sleep,
        }
        proc = self.spawn_run(env)
        wait_until(lambda: len(self.saved_state().agents) == 1, timeout=60)
        return proc, int(self.saved_state().agents[0]["pid"])

    def orphan(self, seconds="30"):
        """Spawn a detached process and reap it promptly, so its group can empty."""
        proc = subprocess.Popen(["sleep", seconds], start_new_session=True)
        threading.Thread(target=proc.wait, daemon=True).start()
        self.addCleanup(kill_group, proc.pid)
        return proc

    def record(
        self, tid="T01", *, pid, worktree_path=None, model="m1", started=None
    ) -> agent.AgentRecord:
        """Build an agent record for a live process, as the scheduler would save it."""
        if started is None:
            started = state.process_start_time(pid) or 0.0
        return agent.AgentRecord(
            id=tid,
            pid=pid,
            pgid=pid,
            model=model,
            worktree=str(worktree_path or self.base / "wt" / tid),
            log=str(self.base / f"{tid}.log"),
            started=started,
        )

    def save_state(self, saved):
        state.save(saved, state.state_path(self.root, self.state_root))


class StopTest(ControlTestCase):
    def test_without_a_scheduler_it_reports_nothing_running(self):
        rc, out, err = self.run_cli(["stop"])
        self.assertEqual(rc, 1)
        self.assertIn("no scheduler running", err)
        self.assertEqual(out, "")

    def test_a_stale_lock_is_not_a_running_scheduler(self):
        dead = subprocess.Popen(["true"])
        dead.wait()
        lock = state.lock_path(self.root, self.state_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": dead.pid, "started": 0.0}), encoding="utf-8")

        rc, _, err = self.run_cli(["stop"])
        self.assertEqual(rc, 1)
        self.assertIn("no scheduler running", err)

    def test_stops_the_scheduler_and_leaves_its_agents_running(self):
        proc, agent_pid = self.start_scheduler()
        lock = state.live_lock(self.root, self.state_root)
        self.assertIsNotNone(lock)
        self.assertEqual(lock.pid, proc.pid)

        rc, out, err = self.run_cli(["stop"])

        self.assertEqual(rc, 0, err)
        self.assertIn(f"stopped scheduler pid {proc.pid}", out)
        proc.wait(timeout=30)  # the scheduler is really gone; its handler ran
        self.assertIsNone(state.live_lock(self.root, self.state_root))
        self.assertTrue(agent.group_alive(agent_pid))  # SPEC §7: agents keep running
        self.assertEqual([int(d["pid"]) for d in self.saved_state().agents], [agent_pid])

    def test_agents_can_be_stopped_too(self):
        proc, agent_pid = self.start_scheduler()

        rc, out, err = self.run_cli(["stop", "--agents"])

        self.assertEqual(rc, 0, err)
        self.assertIn(f"stopped scheduler pid {proc.pid}", out)
        self.assertIn("stopped agents T01", out)
        proc.wait(timeout=30)
        self.assertFalse(agent.group_alive(agent_pid))
        self.assertEqual(self.saved_state().agents, [])

    def test_agents_left_by_a_crashed_scheduler_are_still_found(self):
        proc = self.orphan()
        self.save_state(state.State(agents=[self.record(pid=proc.pid).as_dict()]))

        rc, out, err = self.run_cli(["stop", "--agents"])

        self.assertEqual(rc, 0, err)
        self.assertIn("stopped agents T01", out)
        self.assertIn("no scheduler running", err)
        self.assertFalse(agent.group_alive(proc.pid))
        self.assertEqual(self.saved_state().agents, [])


class RetryTest(ControlTestCase):
    def retry_file(self, tid="T01"):
        return state.pending_dir(self.root, self.state_root) / f"{tid}{state.RETRY_SUFFIX}"

    def test_unblocks_a_blocked_task_and_resumes_its_worktree(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(sessions=1, retries=0)  # first exit without .done blocks
        sched = self.make_scheduler(cfg)
        sched.start()
        with mock.patch.dict(os.environ, {"FAKE_NO_DONE": "1"}):
            self.pump(sched, lambda: "T01" in sched.state.blocked)
            wt = worktree.path(cfg, "T01")
            self.assertTrue(wt.is_dir())  # the interrupted agent left partial work

            rc, out, err = self.run_cli(["retry", "T01"])

            self.assertEqual(rc, 0, err)
            self.assertIn("queued a retry for T01", out)
            self.assertIn("was blocked", out)
            self.assertIn("will be resumed", out)
            self.assertTrue(self.retry_file().is_file())

            sched.tick()  # the running scheduler applies the request (SPEC §10)

            self.assertNotIn("T01", sched.state.blocked)
            self.assertEqual(self.saved_state().blocked, {})
            self.assertEqual(self.saved_state().retries, {})
            self.assertFalse(self.retry_file().exists())
            text = (wt / prompt.PROMPT_NAME).read_text(encoding="utf-8")
            self.assertIn(prompt.RESUME_NOTE, text)
        events = self.events_text()
        self.assertEqual(events.count(" start T01"), 2)
        self.assertIn("start T01 m1 pid", events)
        self.assertIn("(resume)", events)
        self.assertIn("retry T01 retry requested by the operator", events)

    def test_a_request_survives_ticks_that_do_not_see_it(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(sessions=1, retries=0)
        sched = self.make_scheduler(cfg)
        sched.start()
        with mock.patch.dict(os.environ, {"FAKE_NO_DONE": "1"}):
            self.pump(sched, lambda: "T01" in sched.state.blocked)
            rc, _, err = self.run_cli(["retry", "T01"])
            self.assertEqual(rc, 0, err)

            # A tick that misses the request still rewrites state.json from
            # memory — the request is a file outside it, so nothing is lost.
            with mock.patch.object(state, "take_intents", return_value=[]):
                sched.tick()
            self.assertIn("T01", self.saved_state().blocked)
            self.assertTrue(self.retry_file().is_file())

            self.pump(sched, lambda: "T01" not in sched.state.blocked)

            self.assertFalse(self.retry_file().exists())
            self.assertIn("(resume)", self.events_text())

    def test_a_running_agent_is_not_retried(self):
        self.write_plan(PLAN_1)
        self.load_cfg(sessions=1)
        proc = self.orphan()
        self.save_state(
            state.State(agents=[self.record(pid=proc.pid).as_dict()], blocked={"T01": "stalled"})
        )

        rc, _, err = self.run_cli(["retry", "T01"])

        self.assertEqual(rc, 1)
        self.assertIn("still running", err)
        self.assertEqual(self.saved_state().blocked, {"T01": "stalled"})
        self.assertFalse(self.retry_file().exists())

    def test_unknown_task_is_refused(self):
        self.load_cfg(sessions=1)
        rc, _, err = self.run_cli(["retry", "T99"])
        self.assertEqual(rc, 1)
        self.assertIn("no task T99 in PLAN.md", err)
        self.assertFalse(self.retry_file("T99").exists())

    def test_a_done_task_is_refused(self):
        self.write_plan(DONE_PLAN)
        self.load_cfg(sessions=1)
        rc, _, err = self.run_cli(["retry", "T01"])
        self.assertEqual(rc, 1)
        self.assertIn("already done", err)

    def test_a_task_that_was_not_blocked_is_scheduled_again(self):
        self.write_plan(PLAN_1)
        self.load_cfg(sessions=1)
        self.save_state(state.State(retries={"T01": 2}))

        rc, out, err = self.run_cli(["retry", "T01"])

        self.assertEqual(rc, 0, err)
        self.assertIn("it was not blocked", out)
        self.assertTrue(self.retry_file().is_file())

    def test_without_a_config_it_fails_like_the_other_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.chdir(tmp):
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = cli.main(["retry", "T01"])
        self.assertEqual(rc, 2)
        self.assertIn("no taskgraph.toml", err.getvalue())


class RetryRequestTest(ControlTestCase):
    """The durable request one process writes and the scheduler consumes (SPEC §10)."""

    def test_a_request_is_durable_until_it_is_taken(self):
        state.request_retry(self.root, "T01", self.state_root)
        path = state.pending_dir(self.root, self.state_root) / f"T01{state.RETRY_SUFFIX}"
        self.assertTrue(path.is_file())

        self.assertEqual(state.take_intents(self.root, self.state_root), ["T01"])

        self.assertFalse(path.exists())
        self.assertEqual(state.take_intents(self.root, self.state_root), [])

    def test_repeated_requests_collapse_into_one(self):
        state.request_retry(self.root, "T01", self.state_root)
        state.request_retry(self.root, "T01", self.state_root)
        self.assertEqual(state.take_intents(self.root, self.state_root), ["T01"])

    def test_foreign_files_are_left_alone(self):
        pending = state.pending_dir(self.root, self.state_root)
        pending.mkdir(parents=True, exist_ok=True)
        (pending / "notes.txt").write_text("mine\n", encoding="utf-8")
        state.request_retry(self.root, "T02", self.state_root)

        self.assertEqual(state.take_intents(self.root, self.state_root), ["T02"])
        self.assertTrue((pending / "notes.txt").is_file())

    def test_a_request_whose_consumer_died_is_taken_again(self):
        # A consumer renames the request before applying it; a crash in between
        # must not swallow the operator's request.
        pending = state.pending_dir(self.root, self.state_root)
        pending.mkdir(parents=True, exist_ok=True)
        leftover = pending / f"T01{state.RETRY_SUFFIX}{state.CLAIM_SUFFIX}"
        leftover.write_text("", encoding="utf-8")

        self.assertEqual(state.take_intents(self.root, self.state_root), ["T01"])
        self.assertFalse(leftover.exists())

    def test_taking_from_an_absent_directory_is_empty(self):
        self.assertEqual(state.take_intents(self.root, self.state_root), [])


if __name__ == "__main__":
    unittest.main()
