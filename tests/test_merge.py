"""Tests for the merge queue (SPEC §8).

Every test builds a throwaway git repository in a temp dir, creates a real
worktree with real ``task/<id>`` state, and runs :func:`merge.merge` against it.
Gates are trivial shell commands (``true`` / ``false`` / a tiny script).  No
network, no omp, nothing outside the temp dir.
"""

import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from taskgraph import merge, worktree
from taskgraph.config import AgentConfig, Config, MergeConfig, ModelConfig

INITIAL_PLAN = "- [ ] T01 first task\n- [x] T00 done\n"
INITIAL_PROGRESS = "# Progress\n\n"
NOTES = "- did the thing\n- verified it\n"


def git(*args, cwd):
    """Run ``git args`` in ``cwd`` and fail the test if it does not succeed."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def make_config(root, gate="true", trailer=(), merge_config=None):
    """Build a :class:`Config` rooted at ``root`` (no I/O)."""
    return Config(
        path=Path(root) / "taskgraph.toml",
        plan="PLAN.md",
        prompt="PROMPT.md",
        gate=gate,
        worktrees="../wt",
        main="main",
        links=(),
        resources={},
        agent=AgentConfig(command="true"),
        models=(ModelConfig(name="m1", sessions=1, max_agents=1),),
        trailer=tuple(trailer),
        merge=merge_config or MergeConfig(),
    )


class MergeTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        git("init", "-q", "-b", "main", ".", cwd=self.root)
        git("config", "user.email", "t@example.invalid", cwd=self.root)
        git("config", "user.name", "Test", cwd=self.root)
        (self.root / "PLAN.md").write_text(INITIAL_PLAN, encoding="utf-8")
        (self.root / "PROGRESS.md").write_text(INITIAL_PROGRESS, encoding="utf-8")
        (self.root / "README.md").write_text("hello\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "init", cwd=self.root)

    def config(self, gate="true", trailer=()):
        return make_config(self.root, gate=gate, trailer=trailer)

    def start(self, tid="T01", gate="true", trailer=()):
        """Create a worktree with agent work, notes and a ``.done`` marker."""
        cfg = self.config(gate=gate, trailer=trailer)
        wt = worktree.create(cfg, tid)
        (wt / "work.txt").write_text("agent work\n", encoding="utf-8")
        (wt / "logs").mkdir()
        (wt / "logs" / "agent.log").write_text("noise\n", encoding="utf-8")
        (wt / "progress").mkdir()
        (wt / "progress" / f"{tid}.md").write_text(NOTES, encoding="utf-8")
        (wt / "progress" / f"{tid}.done").write_text("", encoding="utf-8")
        return cfg, wt

    def script(self, name, body):
        """Write an executable shell script and return its path as a string."""
        path = self.base / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def log(self, *args, cwd=None):
        return git("log", *args, cwd=cwd or self.root).stdout

    def branch_exists(self, name):
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
            cwd=self.root,
            capture_output=True,
        )
        return result.returncode == 0


class MergeSuccessTest(MergeTestCase):
    def test_merges_work_ticks_plan_and_appends_notes(self):
        cfg, wt = self.start(trailer=("Taskgraph-Task: T01",))
        result = merge.merge(cfg, "T01")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.attempts, 1)
        message = self.log("--format=%B", "-1")
        self.assertIn("T01: merge (taskgraph)", message)
        self.assertIn("Taskgraph-Task: T01", message)
        self.assertIn("T01: work (taskgraph)", self.log("--format=%s"))
        self.assertTrue((self.root / "work.txt").is_file())
        self.assertFalse((self.root / "logs").exists())
        self.assertEqual(
            (self.root / "PLAN.md").read_text(encoding="utf-8"),
            "- [x] T01 first task\n- [x] T00 done\n",
        )
        progress = (self.root / "PROGRESS.md").read_text(encoding="utf-8")
        self.assertIn("## T01\n\n" + NOTES, progress)
        self.assertFalse(wt.exists())
        self.assertFalse(worktree.exists(cfg, "T01"))
        self.assertFalse(self.branch_exists("task/T01"))

    def test_no_changes_or_notes_still_ticks_plan(self):
        cfg = self.config()
        worktree.create(cfg, "T01")
        result = merge.merge(cfg, "T01")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            (self.root / "PROGRESS.md").read_text(encoding="utf-8"), INITIAL_PROGRESS
        )
        self.assertIn("- [x] T01", (self.root / "PLAN.md").read_text(encoding="utf-8"))
        self.assertFalse(worktree.exists(cfg, "T01"))


class GitignoredLogsTest(MergeTestCase):
    """Regression: naming a gitignored path made ``git add`` (and the commit) fail."""

    def setUp(self):
        super().setUp()
        (self.root / ".gitignore").write_text("logs/\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "ignore logs", cwd=self.root)

    def test_stages_and_commits_when_the_project_ignores_logs(self):
        cfg, wt = self.start()
        result = merge.merge(cfg, "T01")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.attempts, 1)
        self.assertTrue((self.root / "work.txt").is_file())
        self.assertFalse((self.root / "logs").exists())
        self.assertIn("- [x] T01", (self.root / "PLAN.md").read_text(encoding="utf-8"))
        self.assertFalse(wt.exists())


class NoWorktreeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)

    def test_missing_worktree_raises(self):
        with self.assertRaises(merge.MergeError):
            merge.merge(make_config(self.base), "T01")


class MergeBlockTest(MergeTestCase):
    def test_conflict_blocks_with_paths_and_keeps_worktree(self):
        cfg = self.config()
        wt = worktree.create(cfg, "T01")
        (wt / "README.md").write_text("agent version\n", encoding="utf-8")
        (self.root / "README.md").write_text("main version\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "main edit", cwd=self.root)

        result = merge.merge(cfg, "T01")

        self.assertFalse(result.ok)
        self.assertIn("conflict in README.md", result.reason)
        self.assertTrue(wt.is_dir())
        self.assertTrue(self.branch_exists("task/T01"))
        # The failed rebase was aborted, so the branch is usable again.
        self.assertEqual(
            git("status", "--porcelain", cwd=wt).stdout, ""
        )

    def test_gate_failure_blocks_with_the_tail_of_its_output(self):
        gate = "echo build ok; echo 'boom: 2 tests failed' >&2; exit 3"
        cfg, wt = self.start(gate=gate)

        result = merge.merge(cfg, "T01")

        self.assertFalse(result.ok)
        self.assertIn("gate failed (3)", result.reason)
        self.assertIn("boom: 2 tests failed", result.reason)
        self.assertTrue(wt.is_dir())
        self.assertIn(
            "boom: 2 tests failed", (wt / "logs" / "gate.log").read_text(encoding="utf-8")
        )

    def test_main_moved_between_rebase_and_merge_is_retried(self):
        marker = self.base / "moved"
        gate = self.script(
            "gate.sh",
            f'if [ ! -f "{marker}" ]; then\n'
            f'  touch "{marker}"\n'
            f'  git -C "{self.root}" commit -q --allow-empty -m "other merge"\n'
            f"fi\n",
        )
        cfg, _ = self.start(gate=gate)
        result = merge.merge(cfg, "T01")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.attempts, 2)

    def test_main_moved_three_times_blocks(self):
        gate = self.script(
            "gate.sh",
            f'git -C "{self.root}" commit -q --allow-empty -m "other merge"\n',
        )
        cfg, wt = self.start(gate=gate)
        result = merge.merge(cfg, "T01")

        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, merge.MAX_ATTEMPTS)
        self.assertIn("main moved 3 times", result.reason)
        self.assertTrue(wt.is_dir())


class CommitGuardTest(MergeTestCase):
    def fill(self, wt, kind):
        """Put one guard offence of ``kind`` into worktree ``wt``."""
        if kind == "many":
            (wt / "gen").mkdir()
            for i in range(201):
                (wt / "gen" / f"f{i}.txt").write_text("x\n", encoding="utf-8")
        elif kind == "big":
            (wt / "big.bin").write_bytes(b"\0" * (6 * 1024 * 1024))
        else:
            (wt / ".build").mkdir()
            (wt / ".build" / "x.o").write_text("obj\n", encoding="utf-8")

    def test_each_limit_blocks_and_keeps_the_worktree(self):
        cfg = self.config()
        cases = {
            "T01": ("many", "adds 201 files (limit 200)", "gen/f0.txt"),
            "T02": ("big", "files over 5 MB", "big.bin (6.0 MB)"),
            "T03": ("build", "paths under build/output dirs", ".build/x.o"),
        }
        for tid, (kind, message, path) in cases.items():
            with self.subTest(kind=kind):
                wt = worktree.create(cfg, tid)
                self.fill(wt, kind)

                result = merge.merge(cfg, tid)

                self.assertFalse(result.ok)
                self.assertIn("commit guard", result.reason)
                self.assertIn(message, result.reason)
                self.assertIn(path, result.reason)
                self.assertTrue(wt.is_dir())  # kept, so retry can fix and resume it
                self.assertTrue(self.branch_exists(worktree.branch(tid)))

    def test_uses_the_limits_from_config(self):
        cfg, wt = self.start()

        strict = merge.merge(replace(cfg, merge=MergeConfig(max_files=2)), "T01")
        self.assertFalse(strict.ok)
        self.assertIn("limit 2", strict.reason)
        self.assertTrue(wt.is_dir())

        # The very same commit passes once the limits are back to the defaults.
        again = merge.merge(cfg, "T01")
        self.assertTrue(again.ok, again.reason)


class FailedCommitTest(MergeTestCase):
    def test_a_failing_commit_blocks_with_git_stderr_and_keeps_the_worktree(self):
        cfg, wt = self.start()
        hook = self.root / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\necho 'hook: nope' >&2\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)

        result = merge.merge(cfg, "T01")

        self.assertFalse(result.ok)
        self.assertIn("git commit failed", result.reason)
        self.assertIn("hook: nope", result.reason)
        self.assertTrue(wt.is_dir())
        self.assertTrue(self.branch_exists("task/T01"))


class QueueTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = make_config(Path(tmp.name))

    def test_processes_tasks_in_fifo_order(self):
        seen = []

        def fake(cfg, tid):
            seen.append(tid)
            return merge.MergeResult(tid, True)

        results = []
        with mock.patch.object(merge, "merge", side_effect=fake):
            q = merge.MergeQueue(self.cfg, on_result=results.append)
            self.addCleanup(q.stop)
            for tid in ("A1", "B2", "C3"):
                q.enqueue(tid)
            q.join()

        self.assertEqual(seen, ["A1", "B2", "C3"])
        self.assertEqual([r.id for r in results], ["A1", "B2", "C3"])
        self.assertEqual([r.id for r in q.results()], ["A1", "B2", "C3"])
        self.assertEqual(q.pending(), ())

    def test_one_failing_merge_is_reported_and_the_worker_survives(self):
        def fake(cfg, tid):
            if tid == "B2":
                raise merge.MergeError("worktree vanished")
            return merge.MergeResult(tid, True)

        with mock.patch.object(merge, "merge", side_effect=fake):
            q = merge.MergeQueue(self.cfg)
            self.addCleanup(q.stop)
            for tid in ("A1", "B2", "C3"):
                q.enqueue(tid)
            q.join()

        by_id = {r.id: r for r in q.results()}
        self.assertTrue(by_id["A1"].ok)
        self.assertFalse(by_id["B2"].ok)
        self.assertIn("merge error", by_id["B2"].reason)
        self.assertTrue(by_id["C3"].ok)

    def test_callback_failure_does_not_stop_the_queue(self):
        def boom(result):
            raise RuntimeError("scheduler bug")

        with mock.patch.object(merge, "merge", return_value=merge.MergeResult("A1", True)):
            q = merge.MergeQueue(self.cfg, on_result=boom)
            self.addCleanup(q.stop)
            q.enqueue("A1")
            q.enqueue("A2")
            q.join()

        self.assertEqual(len(q.results()), 2)


if __name__ == "__main__":
    unittest.main()
