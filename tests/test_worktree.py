"""Tests for the worktree manager (SPEC §1, §5, §8).

Every test builds a throwaway git repository in a temp dir and creates real
worktrees from it; branch/worktree state is asserted with real ``git`` queries.
No network, no omp, nothing outside the temp dir.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from taskgraph import worktree
from taskgraph.config import AgentConfig, Config, ModelConfig


def git(*args, cwd):
    """Run ``git args`` in ``cwd`` and fail the test if it does not succeed."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def real(path):
    """Compare paths by their real location (macOS /var -> /private/var)."""
    return os.path.realpath(path)


class WorktreeTestCase(unittest.TestCase):
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
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "init", cwd=self.root)

    def config(self, *, links=(), worktrees="../wt", main="main", root=None):
        root = Path(root) if root is not None else self.root
        return Config(
            path=root / "taskgraph.toml",
            plan="PLAN.md",
            prompt="PROMPT.md",
            gate="true",
            worktrees=worktrees,
            main=main,
            links=tuple(links),
            resources={},
            agent=AgentConfig(command="true"),
            models=(ModelConfig(name="m1", sessions=1, max_agents=1),),
        )

    def branch_exists(self, name):
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
            cwd=self.root,
            capture_output=True,
        )
        return result.returncode == 0

    def registered(self):
        listing = git("worktree", "list", "--porcelain", cwd=self.root).stdout
        return [
            line[len("worktree ") :].strip()
            for line in listing.splitlines()
            if line.startswith("worktree ")
        ]


class CreateTest(WorktreeTestCase):
    def test_branch_name_is_task_slash_id(self):
        self.assertEqual(worktree.branch("F03"), "task/F03")

    def test_create_makes_a_worktree_on_a_branch_from_main(self):
        cfg = self.config()
        wt = worktree.create(cfg, "T01")
        self.assertEqual(real(wt), real(self.base / "wt" / "T01"))
        self.assertTrue(wt.is_dir())
        self.assertEqual(
            git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt).stdout.strip(), "task/T01"
        )
        main_head = git("rev-parse", "main", cwd=self.root).stdout.strip()
        self.assertEqual(git("rev-parse", "HEAD", cwd=wt).stdout.strip(), main_head)
        self.assertEqual((wt / "README.md").read_text(encoding="utf-8"), "hello\n")

    def test_create_branches_from_the_configured_main(self):
        git("checkout", "-q", "-b", "release", cwd=self.root)
        (self.root / "VERSION").write_text("2.0\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "release only", cwd=self.root)
        git("checkout", "-q", "main", cwd=self.root)
        cfg = self.config(main="release")
        wt = worktree.create(cfg, "T01")
        self.assertEqual((wt / "VERSION").read_text(encoding="utf-8"), "2.0\n")
        self.assertFalse((wt / "taskgraph.toml").exists())

    def test_create_accepts_an_absolute_worktrees_dir(self):
        cfg = self.config(worktrees=str(self.base / "abs-wt"))
        wt = worktree.create(cfg, "T01")
        self.assertEqual(real(wt), real(self.base / "abs-wt" / "T01"))

    def test_create_creates_missing_parent_directories(self):
        cfg = self.config(worktrees="../deep/nested/wt")
        wt = worktree.create(cfg, "T01")
        self.assertTrue(wt.is_dir())

    def test_create_twice_is_an_error_and_leaves_the_first_intact(self):
        cfg = self.config()
        first = worktree.create(cfg, "T01")
        with self.assertRaises(worktree.WorktreeError):
            worktree.create(cfg, "T01")
        self.assertTrue(first.is_dir())
        self.assertEqual(
            git("rev-parse", "--abbrev-ref", "HEAD", cwd=first).stdout.strip(), "task/T01"
        )

    def test_create_unknown_main_is_an_error_with_no_leftovers(self):
        cfg = self.config(main="does-not-exist")
        with self.assertRaises(worktree.WorktreeError) as caught:
            worktree.create(cfg, "T01")
        self.assertIn("git worktree add", str(caught.exception))
        self.assertFalse(worktree.exists(cfg, "T01"))

    def test_create_reports_an_existing_branch_even_without_a_worktree(self):
        cfg = self.config()
        wt = worktree.create(cfg, "T01")
        git("worktree", "remove", "--force", os.fspath(wt), cwd=self.root)
        self.assertTrue(self.branch_exists("task/T01"))
        with self.assertRaises(worktree.WorktreeError):
            worktree.create(cfg, "T01")

    def test_create_outside_a_repository_is_an_error(self):
        plain = self.base / "plain"
        plain.mkdir()
        cfg = self.config(root=plain, worktrees="wt")
        with self.assertRaises(worktree.WorktreeError):
            worktree.create(cfg, "T01")


class LinkTest(WorktreeTestCase):
    def test_links_are_symlinked_to_the_project_file_not_copied(self):
        env = self.root / ".env"
        env.write_text("SECRET=1\n", encoding="utf-8")
        cfg = self.config(links=[".env"])
        wt = worktree.create(cfg, "T01")
        link = wt / ".env"
        self.assertTrue(link.is_symlink())
        self.assertEqual(real(link), real(env))
        self.assertEqual(link.read_text(encoding="utf-8"), "SECRET=1\n")
        env.write_text("SECRET=2\n", encoding="utf-8")
        self.assertEqual(link.read_text(encoding="utf-8"), "SECRET=2\n")
        self.assertIn(
            ".env",
            git("check-ignore", "-v", ".env", cwd=wt).stdout,
        )
        self.assertEqual(git("status", "--porcelain", cwd=wt).stdout.strip(), "")

    def test_nested_link_creates_its_parent_directories(self):
        source = self.root / "config" / "dev.env"
        source.parent.mkdir()
        source.write_text("TOKEN=abc\n", encoding="utf-8")
        cfg = self.config(links=["config/dev.env"])
        wt = worktree.create(cfg, "T01")
        link = wt / "config" / "dev.env"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.read_text(encoding="utf-8"), "TOKEN=abc\n")
        self.assertEqual(git("status", "--porcelain", cwd=wt).stdout.strip(), "")

    def test_missing_link_source_is_skipped(self):
        (self.root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        cfg = self.config(links=[".env", "missing.secret"])
        wt = worktree.create(cfg, "T01")
        self.assertTrue((wt / ".env").is_symlink())
        self.assertFalse((wt / "missing.secret").exists())

    def test_link_is_skipped_when_the_checkout_already_has_the_path(self):
        (self.root / "README.md").write_text("project copy\n", encoding="utf-8")
        cfg = self.config(links=["README.md"])
        wt = worktree.create(cfg, "T01")
        checked_out = wt / "README.md"
        self.assertFalse(checked_out.is_symlink())
        self.assertEqual(checked_out.read_text(encoding="utf-8"), "hello\n")

    def test_create_rolls_back_when_linking_fails(self):
        (self.root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        cfg = self.config(links=[".env"])
        with mock.patch("taskgraph.worktree.os.symlink", side_effect=OSError(13, "denied")):
            with self.assertRaises(worktree.WorktreeError):
                worktree.create(cfg, "T01")
        self.assertFalse(worktree.exists(cfg, "T01"))
        self.assertFalse(self.branch_exists("task/T01"))
        self.assertEqual(self.registered(), [real(self.root)])


class PresenceTest(WorktreeTestCase):
    def test_exists_is_resume_and_started_rank_track_notes(self):
        cfg = self.config()
        self.assertFalse(worktree.exists(cfg, "T01"))
        self.assertFalse(worktree.is_resume(cfg, "T01"))
        self.assertEqual(worktree.started_rank(cfg, "T01"), 2)

        wt = worktree.create(cfg, "T01")
        self.assertTrue(worktree.exists(cfg, "T01"))
        self.assertTrue(worktree.is_resume(cfg, "T01"))
        self.assertEqual(worktree.started_rank(cfg, "T01"), 1)

        notes = worktree.notes_path(cfg, "T01")
        self.assertEqual(notes, wt / "progress" / "T01.md")
        notes.parent.mkdir(parents=True)
        notes.write_text("- did things\n", encoding="utf-8")
        self.assertEqual(worktree.started_rank(cfg, "T01"), 0)

    def test_path_resolves_against_the_project_root(self):
        cfg = self.config()
        self.assertEqual(real(worktree.path(cfg, "T01")), real(self.base / "wt" / "T01"))


class RemoveTest(WorktreeTestCase):
    def test_remove_deletes_the_worktree_and_branch_and_allows_recreate(self):
        cfg = self.config()
        wt = worktree.create(cfg, "T01")
        worktree.remove(cfg, "T01")
        self.assertFalse(wt.exists())
        self.assertFalse(worktree.exists(cfg, "T01"))
        self.assertFalse(self.branch_exists("task/T01"))
        self.assertEqual(self.registered(), [real(self.root)])
        again = worktree.create(cfg, "T01")
        self.assertTrue(again.is_dir())

    def test_remove_is_idempotent(self):
        cfg = self.config()
        worktree.remove(cfg, "T99")  # never created
        wt = worktree.create(cfg, "T01")
        worktree.remove(cfg, "T01")
        worktree.remove(cfg, "T01")  # already gone
        self.assertFalse(wt.exists())

    def test_remove_handles_a_directory_deleted_by_hand(self):
        cfg = self.config()
        wt = worktree.create(cfg, "T01")
        shutil.rmtree(wt)
        worktree.remove(cfg, "T01")
        self.assertFalse(self.branch_exists("task/T01"))
        self.assertEqual(self.registered(), [real(self.root)])

    def test_remove_leaves_other_worktrees_alone(self):
        cfg = self.config()
        first = worktree.create(cfg, "T01")
        second = worktree.create(cfg, "T02")
        worktree.remove(cfg, "T01")
        self.assertFalse(first.exists())
        self.assertTrue(second.is_dir())
        self.assertTrue(self.branch_exists("task/T02"))
        self.assertIn(real(second), [real(p) for p in self.registered()])

    def test_remove_never_deletes_an_unregistered_directory(self):
        cfg = self.config()
        stray = worktree.path(cfg, "T01")
        stray.mkdir(parents=True)
        (stray / "user-data.txt").write_text("keep\n", encoding="utf-8")
        worktree.remove(cfg, "T01")
        self.assertTrue((stray / "user-data.txt").is_file())


if __name__ == "__main__":
    unittest.main()
