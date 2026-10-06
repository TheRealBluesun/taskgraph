"""Tests for the merge conflict resolver (T21).

Each test builds a throwaway git repository, puts a real worktree into a real
mid-rebase conflict, and drives :class:`taskgraph.resolve.Resolver` with a tiny
shell script standing in for the agent.  No omp, no model, no network.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from taskgraph import resolve, worktree
from taskgraph.config import AgentConfig, Config, MergeConfig, ModelConfig

MAIN_MESSAGE = "main edit"
TASK_MESSAGE = "T01: work (taskgraph)"
BASE_TEXT = "base\n"


def git(*args, cwd):
    """Run ``git args`` in ``cwd`` and fail the test if it does not succeed."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def make_config(root, *, command, resolver_prompt="RESOLVER.md", gate="true"):
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
        agent=AgentConfig(command=command),
        models=(ModelConfig(name="m1", sessions=1, max_agents=1),),
        merge=MergeConfig(resolver_prompt=resolver_prompt),
    )


class ConflictHunkTest(unittest.TestCase):
    def test_keeps_both_sides_of_every_hunk(self):
        text = (
            "a\n<<<<<<< HEAD\nours one\n=======\ntheirs one\n>>>>>>> abc (msg)\n"
            "b\n<<<<<<< HEAD\nours two\n=======\ntheirs two\n>>>>>>> abc (msg)\n"
        )
        hunks = resolve.conflict_hunks(text)
        self.assertEqual(len(hunks), 2)
        self.assertIn("ours one", hunks[0])
        self.assertIn("theirs one", hunks[0])
        self.assertIn("ours two", hunks[1])
        self.assertIn("theirs two", hunks[1])

    def test_a_text_without_markers_has_no_hunks(self):
        self.assertEqual(resolve.conflict_hunks("clean\n"), [])


class ResolverTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        git("init", "-q", "-b", "main", ".", cwd=self.root)
        git("config", "user.email", "t@example.invalid", cwd=self.root)
        git("config", "user.name", "Test", cwd=self.root)
        (self.root / "README.md").write_text(BASE_TEXT, encoding="utf-8")
        (self.root / "PLAN.md").write_text("- [ ] T01 first task\n", encoding="utf-8")
        (self.root / "RESOLVER.md").write_text("PROJECT RESOLVER RULES\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", "init", cwd=self.root)

    def script(self, name, body):
        """Write an executable shell script and return its path as a string."""
        path = self.base / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def conflict(self, tid="T01", *, command, resolver_prompt="RESOLVER.md"):
        """Leave ``tid``'s worktree stopped mid-rebase on a README conflict."""
        cfg = make_config(self.root, command=command, resolver_prompt=resolver_prompt)
        wt = worktree.create(cfg, tid)
        (wt / "README.md").write_text("task version\n", encoding="utf-8")
        git("add", "-A", cwd=wt)
        git("commit", "-qm", TASK_MESSAGE, cwd=wt)
        (self.root / "README.md").write_text("main version\n", encoding="utf-8")
        git("add", "-A", cwd=self.root)
        git("commit", "-qm", MAIN_MESSAGE, cwd=self.root)
        rebase = subprocess.run(["git", "rebase", "main"], cwd=wt, capture_output=True, text=True)
        self.assertNotEqual(rebase.returncode, 0, "the fixture must really conflict")
        return cfg, wt

    def resolver(self, script, *, model="m1"):
        return resolve.Resolver(pick_model=lambda: model)


class DossierTest(ResolverTestCase):
    def test_names_both_commits_and_both_sides(self):
        cfg, wt = self.conflict(command="true")

        text = resolve.dossier("main", wt, ["README.md"])

        self.assertIn("README.md", text)
        self.assertIn("task version", text)
        self.assertIn("main version", text)
        self.assertIn(TASK_MESSAGE, text)
        self.assertIn(MAIN_MESSAGE, text)

    def test_prompt_text_carries_the_rule_and_the_project_text(self):
        text = resolve.prompt_text("T01", "main", "PROJECT RULES", "DOSSIER")

        self.assertIn(resolve.RULE, text)
        self.assertIn("PROJECT RULES", text)
        self.assertIn("DOSSIER", text)


class ResolverRunTest(ResolverTestCase):
    def test_a_resolver_that_finishes_the_rebase_succeeds(self):
        script = self.script(
            "resolve.sh",
            "printf 'resolved by agent\\n' > README.md\n"
            "git add README.md\n"
            "git rebase --continue\n",
        )
        cfg, wt = self.conflict(command=script)

        result = self.resolver(script).resolve(cfg, "T01", wt, ["README.md"])

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.files, ("README.md",))
        self.assertEqual(result.model, "m1")
        self.assertEqual((wt / "README.md").read_text(encoding="utf-8"), "resolved by agent\n")
        ok, reason = resolve.finished(cfg, wt)
        self.assertTrue(ok, reason)

    def test_the_resolver_command_gets_the_chosen_model(self):
        script = self.script("model.sh", 'printf "%s\\n" "$1" > model.txt\n')
        cfg, wt = self.conflict(command=f"{script} {{model}}")

        result = self.resolver(script, model="m1").resolve(cfg, "T01", wt, ["README.md"])

        self.assertEqual((wt / "model.txt").read_text(encoding="utf-8"), "m1\n")
        # The script only recorded the model: the rebase is still unresolved.
        self.assertFalse(result.ok)

    def test_a_resolver_that_does_nothing_fails(self):
        script = self.script("noop.sh", "exit 0\n")
        cfg, wt = self.conflict(command=script)

        result = self.resolver(script).resolve(cfg, "T01", wt, ["README.md"])

        self.assertFalse(result.ok)
        self.assertIn("rebase is still in progress", result.reason)

    def test_a_resolver_that_aborts_the_rebase_fails(self):
        script = self.script("abort.sh", "git rebase --abort\n")
        cfg, wt = self.conflict(command=script)

        result = self.resolver(script).resolve(cfg, "T01", wt, ["README.md"])

        self.assertFalse(result.ok)
        self.assertIn("was not applied", result.reason)

    def test_no_free_model_blocks_without_running_the_agent(self):
        script = self.script("marker.sh", "touch ran.txt\n")
        cfg, wt = self.conflict(command=script)

        result = resolve.Resolver(pick_model=lambda: None).resolve(cfg, "T01", wt, ["README.md"])

        self.assertFalse(result.ok)
        self.assertIn("no model", result.reason)
        self.assertFalse((wt / "ran.txt").exists())

    def test_a_missing_prompt_file_fails_before_picking_a_model(self):
        script = self.script("marker.sh", "touch ran.txt\n")
        cfg, wt = self.conflict(command=script, resolver_prompt="missing.md")

        result = self.resolver(script).resolve(cfg, "T01", wt, ["README.md"])

        self.assertFalse(result.ok)
        self.assertIn("resolver prompt", result.reason)
        self.assertFalse((wt / "ran.txt").exists())

    def test_the_prompt_file_is_git_excluded_and_holds_the_dossier(self):
        script = self.script("noop.sh", "exit 0\n")
        cfg, wt = self.conflict(command=script)

        self.resolver(script).resolve(cfg, "T01", wt, ["README.md"])

        prompt = (wt / resolve.PROMPT_NAME).read_text(encoding="utf-8")
        self.assertIn(resolve.RULE, prompt)
        self.assertIn("task version", prompt)
        self.assertEqual(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=wt, capture_output=True, text=True
            ).stdout.count(resolve.PROMPT_NAME),
            0,
        )


if __name__ == "__main__":
    unittest.main()
