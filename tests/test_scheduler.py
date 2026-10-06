"""Scheduler-loop tests (SPEC §7–§9).

Each test builds a throwaway git repository with a real ``taskgraph.toml`` and
fakes agents with a tiny shell script that writes ``progress/<id>.done`` in its
own worktree (its basename is the task id). The loop is driven in-process
(``Scheduler.tick``) except for the restart test, which runs the real
``bin/taskgraph run`` as a subprocess and kills it mid-run. No omp, no model,
no network.
"""

import contextlib
import io
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from taskgraph import (
    agent,
    assign,
    cli,
    config,
    lease,
    leaseprocs,
    metrics,
    scheduler,
    state,
    trace,
    worktree,
)

ROOT = Path(__file__).resolve().parent.parent

PLAN_3 = "- [ ] T01 first\n- [ ] T02 second\n- [ ] T03 third [deps: T01]\n"
PLAN_1 = "- [ ] T01 only\n"

FAKE_AGENT = """\
#!/bin/sh
id=$(basename "$PWD")
echo "trace $id"
sleep "${FAKE_SLEEP:-0.02}"
if [ -n "${FAKE_NO_DONE:-}" ]; then
  echo "giving up"
  exit 0
fi
mkdir -p progress
printf -- "- did %s\\n" "$id" > "progress/$id.md"
: > "progress/$id.done"
"""

TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "{worktrees}"
main = "main"
{resources}
[agent]
command = '{command}'
stall_secs = {stall_secs}
max_lease_secs = {max_lease_secs}
retries = {retries}

[[models]]
name = "m1"
sessions = {sessions}
max_agents = {max_agents}
{metrics}
"""


IDLE_AGENT = """\
#!/bin/sh
echo '{"type":"tool_execution_start","toolCallId":"1","toolName":"bash"}'
sleep "${FAKE_SLEEP:-60}"
"""


def git(*args, cwd):
    """Run ``git args`` in ``cwd`` and fail the test if it does not succeed."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def wait_until(predicate, timeout=60.0, interval=0.01):
    """Poll ``predicate`` until truthy; raise ``AssertionError`` on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(interval)


class ProjectTestCase(unittest.TestCase):
    """A temp git project, a fake agent script and an isolated state dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        git("init", "-q", "-b", "main", ".", cwd=self.root)
        git("config", "user.email", "t@example.invalid", cwd=self.root)
        git("config", "user.name", "Test", cwd=self.root)
        (self.root / "README.md").write_text("hello\n", encoding="utf-8")
        self.write_plan(PLAN_3)
        (self.root / "PROMPT.md").write_text("PROJECT RULES\n", encoding="utf-8")
        self.script = self.base / "fake-agent.sh"
        self.script.write_text(FAKE_AGENT, encoding="utf-8")
        self.script.chmod(0o755)
        self.write_toml()
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "init", cwd=self.root)
        self.state_root = self.base / "state"
        self._procs: list[subprocess.Popen] = []
        self.addCleanup(self._cleanup)

    def write_plan(self, text):
        (self.root / "PLAN.md").write_text(text, encoding="utf-8")

    def write_toml(
        self,
        *,
        sessions=2,
        max_agents=None,
        stall_secs=480,
        retries=2,
        max_lease_secs=1800,
        resources="",
        command=None,
        worktrees="../wt",
        metrics="",
    ):
        text = TOML.format(
            command=self.script if command is None else command,
            stall_secs=stall_secs,
            retries=retries,
            max_lease_secs=max_lease_secs,
            resources=resources,
            sessions=sessions,
            max_agents=sessions if max_agents is None else max_agents,
            worktrees=worktrees,
            metrics=metrics,
        )
        path = self.root / "taskgraph.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def load_cfg(self, **kwargs):
        self.write_toml(**kwargs)
        return config.load(self.root / "taskgraph.toml")

    def events_text(self):
        path = state.project_dir(self.root, self.state_root) / "events.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def saved_state(self):
        return state.load(state.state_path(self.root, self.state_root))

    def make_scheduler(self, cfg, **kwargs):
        kwargs.setdefault("state_root", self.state_root)
        kwargs.setdefault("out", io.StringIO())
        kwargs.setdefault("tick_secs", 0.03)
        sched = scheduler.Scheduler(cfg, **kwargs)
        self.addCleanup(sched.close)
        return sched

    def pump(self, sched, predicate, timeout=120.0, interval=0.01):
        """Tick the loop until ``predicate`` holds (or fail on timeout)."""
        deadline = time.monotonic() + timeout
        while True:
            sched.tick()
            if predicate():
                return
            if time.monotonic() >= deadline:
                self.fail(f"condition not met within {timeout}s; events:\n{self.events_text()}")
            time.sleep(interval)

    def plan_all_done(self):
        return "[ ]" not in (self.root / "PLAN.md").read_text(encoding="utf-8")

    def spawn_run(self, env):
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "bin" / "taskgraph"), "run", "--project", str(self.root)],
            cwd=self.base,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        return proc

    def _cleanup(self):
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        try:
            saved = self.saved_state()
        except Exception:  # pragma: no cover - cleanup must never mask a failure
            return
        for data in saved.agents:
            try:
                pid = int(data["pid"])
            except (KeyError, TypeError, ValueError):  # pragma: no cover
                continue
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


class MergeOrderTest(ProjectTestCase):
    def test_three_tasks_merge_in_dependency_order(self):
        cfg = self.load_cfg(sessions=2)
        sched = self.make_scheduler(cfg)
        sched.start()
        self.pump(sched, self.plan_all_done, timeout=120)
        sched.close()

        self.assertNotIn("[ ]", (self.root / "PLAN.md").read_text(encoding="utf-8"))
        subjects = git("log", "--format=%s", "main", cwd=self.root).stdout.splitlines()
        # git log is newest first, so T03's work commit must come before T01's.
        self.assertEqual(subjects.count("T01: work (taskgraph)"), 1)
        self.assertLess(
            subjects.index("T03: work (taskgraph)"), subjects.index("T01: work (taskgraph)")
        )
        progress = (self.root / "PROGRESS.md").read_text(encoding="utf-8")
        for tid in ("T01", "T02", "T03"):
            self.assertIn(f"## {tid}", progress)
        events = self.events_text()
        self.assertRegex(events, r"(?m)^\d{2}:\d{2}:\d{2} merged T01 ")
        self.assertIn("start T03", events)
        for tid in ("T01", "T02", "T03"):
            self.assertFalse(worktree.exists(cfg, tid))


class RestartTest(ProjectTestCase):
    def test_restart_adopts_live_agents_instead_of_restarting_them(self):
        self.write_toml(sessions=2)
        env = {
            **os.environ,
            "TASKGRAPH_STATE": str(self.state_root),
            "TASKGRAPH_TICK": "0.05",
            "FAKE_SLEEP": "30",  # fail-safe: cleanup kills the group, no test waits for it
        }
        first = self.spawn_run(env)
        wait_until(lambda: len(self.saved_state().agents) >= 2)
        pids = sorted(int(data["pid"]) for data in self.saved_state().agents)
        self.assertEqual(len(pids), 2)

        first.kill()  # a hard scheduler crash: no signal handler runs
        first.wait(timeout=30)

        self.spawn_run(env)
        wait_until(lambda: "adopt T01" in self.events_text())
        adopted = sorted(int(data["pid"]) for data in self.saved_state().agents)
        self.assertEqual(adopted, pids)  # same processes, not replacements
        events = self.events_text()
        self.assertEqual(events.count(" start T01"), 1)  # no second start


class RetryTest(ProjectTestCase):
    def test_agent_exiting_without_done_retries_then_blocks(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(sessions=1, retries=1)
        sched = self.make_scheduler(cfg)
        sched.start()
        with mock.patch.dict(os.environ, {"FAKE_NO_DONE": "1"}):
            self.pump(sched, lambda: "T01" in sched.state.blocked)
        sched.close()

        events = self.events_text()
        self.assertEqual(events.count(" start T01"), 2)
        self.assertEqual(events.count(" retry T01"), 1)
        self.assertEqual(events.count(" blocked T01"), 1)
        self.assertEqual(sched.state.retries["T01"], 1)
        self.assertFalse((worktree.path(cfg, "T01") / "progress" / "T01.done").exists())


class StallTest(ProjectTestCase):
    def test_stalled_agent_is_killed_then_blocked_with_no_retries(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(sessions=1, retries=0, stall_secs=0.2)
        sched = self.make_scheduler(cfg)
        sched.start()
        with mock.patch.dict(os.environ, {"FAKE_SLEEP": "5"}):
            self.pump(sched, lambda: "T01" in sched.state.blocked)
        sched.close()

        events = self.events_text()
        self.assertIn("stall T01", events)
        self.assertIn("blocked T01", events)
        match = re.search(r"start T01 m1 pid (\d+)", events)
        self.assertIsNotNone(match)
        self.assertFalse(agent.group_alive(int(match.group(1))))


class StallExemptTest(ProjectTestCase):
    """A silent agent waiting for a shared resource is not stalled (SPEC §6)."""

    def lease_agent(self) -> Path:
        """A fake agent that runs the real ``taskgraph lease`` and then sleeps."""
        lease_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'bin' / 'taskgraph'))}"
        script = self.base / "lease-agent.sh"
        script.write_text(
            "#!/bin/sh\n"
            'echo "trace $(basename "$PWD")"\n'
            f"{lease_cmd} lease sim -- sleep 30\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def test_waiting_on_a_lease_keeps_the_stall_watchdog_off(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(
            sessions=1,
            retries=0,
            stall_secs=2.0,
            max_lease_secs=2.0,
            resources="\n[resources]\nsim = 1\n",
            command=self.lease_agent(),
            worktrees="wt",  # inside the project, so `taskgraph lease` finds the config
        )
        # The worktree is a checkout of main and `taskgraph lease` reads the
        # config from it, so main must carry the [resources] table.
        git("add", "taskgraph.toml", cwd=self.root)
        git("commit", "-qm", "add the simulator resource", cwd=self.root)
        # Hold the only slot in-process: the agent's wrapper waits for it, which
        # is exactly the silent-by-design case that was killed as "stalled".
        held = lease.try_acquire("sim", 1, root=self.state_root, task="other")
        self.assertIsNotNone(held)
        self.addCleanup(lease.release, held, root=self.state_root)

        # TASKGRAPH_STATE reaches the agent through the environment (the
        # scheduler only injects `state_root` in-process), and the agent's own
        # `taskgraph lease` must use this test's slots, not the user's.
        with mock.patch.dict(os.environ, {"TASKGRAPH_STATE": str(self.state_root)}):
            sched = self.make_scheduler(cfg)
            sched.start()
            # Fail fast on a regression: a stall kill is the opposite of the point.
            self.pump(
                sched,
                lambda: any(
                    phrase in self.events_text()
                    for phrase in ("anomaly T01", "stall T01", "blocked T01")
                ),
            )
            events = self.events_text()
            self.assertIn("anomaly T01", events)  # lease outlived agent.max_lease_secs
            self.assertNotIn("stall T01", events)
            self.assertNotIn("blocked T01", events)
            record = sched.running["T01"]
            self.assertTrue(agent.group_alive(record.pgid))
            agent.kill(record, timeout=5)  # the wrapper forwards SIGTERM to `sleep`
        self.assertFalse(agent.group_alive(record.pgid))


class IdleWatchTest(ProjectTestCase):
    """A model with agents but no requests is reported once, with a diagnosis (SPEC §9)."""

    def test_idle_model_is_reported_once_with_a_diagnosis(self):
        self.write_plan(PLAN_1)
        script = self.base / "idle-agent.sh"
        script.write_text(IDLE_AGENT, encoding="utf-8")
        script.chmod(0o755)
        cfg = self.load_cfg(
            sessions=1,
            max_agents=2,
            stall_secs=3600,
            command=script,
            metrics='metrics = "http://127.0.0.1:1/metrics"',
        )
        clock = [time.time()]
        sched = self.make_scheduler(
            cfg, now=lambda: clock[0], sample=lambda url: metrics.Metrics(0.0, 0.0, 0.0)
        )
        sched.start()
        sched.tick()  # starts T01; the zero-running clock starts on the next tick
        record = sched.running["T01"]
        # The agent's shell needs a moment to write its first trace line.
        wait_until(lambda: trace.last_tool(record.log) == "bash", timeout=10)
        for _ in range(12):  # > IDLE_SECS at the 20 s tick
            clock[0] += 20.0
            sched.tick()
        agent.kill(record, timeout=5)

        events = self.events_text()
        self.assertEqual(events.count(" idle m1 "), 1)
        line = next(line for line in events.splitlines() if " idle m1 " in line)
        self.assertIn("agents T01 bash", line)
        self.assertIn("ago", line)
        self.assertIn("holders none", line)
        self.assertIn("waiters none", line)


class LeaseAnomalyTest(ProjectTestCase):
    """A held lease that outlives 8 min or has no lease wrapper is an anomaly (SPEC §9)."""

    def test_holder_that_never_announced_is_flagged_once(self):
        cfg = self.load_cfg(sessions=1)
        sched = self.make_scheduler(cfg, now=time.time, max_agents=0)
        sched.start()
        held = lease.try_acquire("sim", 1, root=self.state_root, task="T01")
        self.addCleanup(lease.release, held, root=self.state_root)
        sched.tick()
        sched.tick()
        events = self.events_text()
        self.assertIn(f"slot-0 held by a non-lease process (pid {os.getpid()})", events)
        self.assertEqual(events.count(" anomaly sim "), 1)

    def test_lease_held_over_eight_minutes_is_flagged_once(self):
        cfg = self.load_cfg(sessions=1)
        clock = [time.time()]
        sched = self.make_scheduler(cfg, now=lambda: clock[0], max_agents=0)
        sched.start()
        announcement = leaseprocs.register_process("sim", root=self.state_root, pid=os.getpid())
        self.addCleanup(leaseprocs.unregister_process, announcement)
        held = lease.try_acquire(
            "sim", 1, root=self.state_root, task="T02", now=clock[0] - 600.0
        )
        self.addCleanup(lease.release, held, root=self.state_root)
        sched.tick()
        events = self.events_text()
        self.assertIn("slot-0 held 10m00s by T02", events)
        self.assertEqual(events.count(" anomaly sim "), 1)
        sched.tick()
        self.assertEqual(self.events_text().count(" anomaly sim "), 1)


class ReloadTest(ProjectTestCase):
    def test_model_pool_reload_is_picked_up_and_logged(self):
        cfg = self.load_cfg(sessions=1)
        sched = self.make_scheduler(cfg)
        self.write_toml(sessions=2, max_agents=2)
        future = time.time() + 10
        os.utime(self.root / "taskgraph.toml", (future, future))
        sched.tick()
        self.assertEqual(sched.cfg.models[0].sessions, 2)
        self.assertIn("pool", self.events_text())


class RunTest(ProjectTestCase):
    def test_run_bounds_ticks_and_writes_state(self):
        self.write_plan(PLAN_1)
        cfg = self.load_cfg(sessions=1)
        sched = self.make_scheduler(cfg, tick_secs=0.01)
        self.assertEqual(sched.run(max_ticks=2), 0)
        self.assertTrue(state.state_path(self.root, self.state_root).is_file())

    def test_run_refuses_a_second_scheduler_for_the_same_project(self):
        cfg = self.load_cfg(sessions=1)
        state.claim_lock(self.root, self.state_root)
        sched = self.make_scheduler(cfg)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sched.run(max_ticks=1), 2)


class DryRunTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        (self.root / "PLAN.md").write_text(PLAN_3, encoding="utf-8")
        (self.root / "PROMPT.md").write_text("rules\n", encoding="utf-8")
        (self.root / "taskgraph.toml").write_text(
            TOML.format(
                command="true",
                stall_secs=480,
                retries=2,
                max_lease_secs=1800,
                resources="",
                sessions=1,
                max_agents=1,
                worktrees="../wt",
                metrics="",
            ),
            encoding="utf-8",
        )
        self.cfg = config.load(self.root / "taskgraph.toml")

    def test_lists_runnable_tasks_with_models_and_stops_at_capacity(self):
        lines = assign.dry_run(self.cfg, now=1_000_000.0).splitlines()
        self.assertEqual(lines[0].split("\t"), ["T01", "m1"])
        self.assertEqual(lines[1].split("\t"), ["T02", "-"])  # pool is full
        self.assertNotIn("T03", "\n".join(lines))  # dependency not done yet

    def test_max_agents_caps_the_assignment(self):
        lines = assign.dry_run(self.cfg, max_agents=0, now=1_000_000.0).splitlines()
        self.assertEqual(lines[0].split("\t"), ["T01", "-"])

    def test_cli_dry_run_prints_the_plan(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(["run", "--project", str(self.root), "--dry-run"])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertIn("T01", out.getvalue())


if __name__ == "__main__":
    unittest.main()
