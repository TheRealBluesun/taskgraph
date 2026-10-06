"""Tests for ``taskgraph.state`` (SPEC §7): atomic save/load, the single-instance
lock and the (pid, start time) process-identity helper.  Everything runs in temp
dirs with real short-lived child processes; no network, no omp.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from taskgraph import state
from taskgraph.state import LockError, State, StateError


def _spawn(sleep: float = 30.0) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({sleep})"])


def _reap(proc: subprocess.Popen) -> int:
    """Wait for ``proc`` and return its now-dead pid."""
    proc.terminate()
    proc.wait(timeout=10)
    return proc.pid


class PathsTest(unittest.TestCase):
    def test_project_key_is_stable_and_path_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir()
            b.mkdir()
            self.assertEqual(state.project_key(a), state.project_key(str(a)))
            self.assertNotEqual(state.project_key(a), state.project_key(b))

    def test_paths_live_under_projects_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, project = Path(tmp) / "state", Path(tmp) / "proj"
            project.mkdir()
            key = state.project_key(project)
            self.assertEqual(state.project_dir(project, root), root / "projects" / key)
            self.assertEqual(state.state_path(project, root), root / "projects" / key / "state.json")
            self.assertEqual(state.lock_path(project, root), root / "projects" / key / "lock")

    def test_default_root_honours_taskgraph_state_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"TASKGRAPH_STATE": tmp}):
                self.assertEqual(state.project_dir("/some/project"), Path(tmp) / "projects" / state.project_key("/some/project"))


class SaveLoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "state.json"

    def test_missing_file_is_an_empty_state(self):
        self.assertEqual(state.load(self.path), State())

    def test_round_trip_all_fields(self):
        original = State(
            agents=[{"id": "T01", "pid": 1234, "model": "local"}],
            blocked={"T02": "gate failed"},
            retries={"T03": 1},
            last_extra={"local": 100.0},
            samples={"local": [{"at": 1.0, "running": 2.0}]},
        )
        state.save(original, self.path)
        self.assertEqual(state.load(self.path), original)

    def test_save_creates_parent_dirs_and_leaves_no_temp_file(self):
        state.save(State(retries={"T01": 2}), self.path)
        self.assertTrue(self.path.is_file())
        leftovers = [p.name for p in self.path.parent.iterdir() if p.name != "state.json"]
        self.assertEqual(leftovers, [])

    def test_file_is_pretty_json_with_trailing_newline(self):
        state.save(State(retries={"T01": 2}), self.path)
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text)["retries"], {"T01": 2})

    def test_unknown_keys_are_ignored_and_missing_keys_default(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"retries": {"T01": 3}, "future": [1]}), encoding="utf-8")
        self.assertEqual(state.load(self.path), State(retries={"T01": 3}))

    def test_malformed_json_raises_state_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(StateError):
            state.load(self.path)

    def test_non_object_json_raises_state_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(StateError):
            state.load(self.path)

    def test_save_overwrites_previous_state(self):
        state.save(State(retries={"T01": 1}), self.path)
        state.save(State(retries={"T02": 5}), self.path)
        self.assertEqual(state.load(self.path).retries, {"T02": 5})


class ProcessIdentityTest(unittest.TestCase):
    def test_own_pid_is_alive_and_dead_pid_is_not(self):
        self.assertTrue(state.pid_alive(os.getpid()))
        proc = _spawn(0)
        dead = _reap(proc)
        self.assertFalse(state.pid_alive(dead))

    def test_invalid_pids_are_not_alive(self):
        self.assertFalse(state.pid_alive(0))
        self.assertFalse(state.pid_alive(-1))

    def test_start_time_of_a_live_child_matches_itself(self):
        proc = _spawn()
        try:
            started = state.process_start_time(proc.pid)
            self.assertIsNotNone(started)
            self.assertTrue(state.start_time_matches(proc.pid, started))
        finally:
            _reap(proc)

    def test_recycled_pid_with_a_different_start_time_is_not_the_same_process(self):
        proc = _spawn()
        try:
            started = state.process_start_time(proc.pid)
            self.assertIsNotNone(started)
            self.assertFalse(state.start_time_matches(proc.pid, started - 3600))
            self.assertFalse(state.start_time_matches(proc.pid, started + 3600))
        finally:
            _reap(proc)

    def test_dead_process_never_matches(self):
        proc = _spawn(0)
        dead = _reap(proc)
        self.assertIsNone(state.process_start_time(dead))
        self.assertFalse(state.start_time_matches(dead, time.time()))

    def test_unknown_start_time_degrades_to_liveness(self):
        self.assertTrue(state.start_time_matches(os.getpid(), 0.0))


class LockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "state"
        self.project = Path(self.tmp.name) / "proj"
        self.project.mkdir()
        self.path = state.lock_path(self.project, self.root)

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_claim_writes_our_pid_and_creates_the_dir(self):
        lock = state.claim_lock(self.project, self.root)
        self.assertEqual(lock.pid, os.getpid())
        self.assertTrue(self.path.is_file())
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["pid"], os.getpid())
        self.assertEqual(state.live_lock(self.project, self.root), lock)

    def test_second_claim_is_refused_while_the_holder_is_alive(self):
        state.claim_lock(self.project, self.root)
        with self.assertRaises(LockError) as ctx:
            state.claim_lock(self.project, self.root)
        self.assertIn(str(os.getpid()), str(ctx.exception))

    def test_release_removes_the_lock(self):
        lock = state.claim_lock(self.project, self.root)
        lock.release()
        self.assertFalse(self.path.exists())
        self.assertIsNone(state.read_lock(self.project, self.root))
        state.claim_lock(self.project, self.root)  # free again

    def test_release_leaves_a_foreign_lock_alone(self):
        lock = state.claim_lock(self.project, self.root)
        self._write({"pid": os.getpid() + 1, "started": lock.started})
        lock.release()
        self.assertTrue(self.path.exists())

    def test_stale_lock_from_a_dead_pid_is_taken_over(self):
        dead = _reap(_spawn(0))
        self._write({"pid": dead, "started": time.time() - 60})
        self.assertIsNone(state.live_lock(self.project, self.root))
        lock = state.claim_lock(self.project, self.root)
        self.assertEqual(lock.pid, os.getpid())

    def test_recycled_pid_with_wrong_start_time_is_stale(self):
        started = state.process_start_time(os.getpid())
        if started is None:
            self.skipTest("ps unavailable")
        self._write({"pid": os.getpid(), "started": started - 10000})
        self.assertIsNone(state.live_lock(self.project, self.root))
        self.assertEqual(state.claim_lock(self.project, self.root).pid, os.getpid())

    def test_malformed_and_pidless_lock_files_are_stale(self):
        self._write({"started": 1.0})
        self.assertIsNone(state.live_lock(self.project, self.root))
        self.path.write_text("garbage", encoding="utf-8")
        self.assertIsNone(state.read_lock(self.project, self.root))
        self.assertEqual(state.claim_lock(self.project, self.root).pid, os.getpid())


if __name__ == "__main__":
    unittest.main()
