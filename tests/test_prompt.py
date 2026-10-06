"""Tests for the agent prompt file and command template (SPEC §6)."""

import tempfile
import unittest
from pathlib import Path

from taskgraph import prompt
from agenthelpers import PROJECT_PROMPT, make_config, make_task


class PromptTextTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "project"
        self.task = make_task("F03", "Top strip per SPEC")
        self.cfg = make_config(self.root, "agent {prompt_file}", plan="PLAN.md")

    def test_header_names_the_task_and_plan_and_appends_project_prompt(self):
        text = prompt.prompt_text(self.task, self.cfg.plan, PROJECT_PROMPT)
        lines = text.splitlines()
        self.assertEqual(lines[0], "YOUR TASK IS **F03** (do only this one): F03 Top strip per SPEC")
        self.assertIn("You are one of several agents working in parallel", lines[1])
        self.assertIn("- Do NOT edit PLAN.md and do not run git commands.", lines)
        self.assertIn("- Write your notes (2–5 bullets) to progress/F03.md.", lines)
        self.assertIn(
            "- When — and only when — the task is complete and verified, create the empty file "
            "progress/F03.done.",
            lines,
        )
        self.assertIn(
            "- Run long commands in the foreground and wait for them; never end your turn while "
            "a background job is running.",
            lines,
        )
        self.assertIn(
            "- Never modify or work around the shared-resource tooling (leases, shims); if a "
            "command waits for a lease, wait.",
            lines,
        )
        self.assertTrue(text.endswith("Never break the build.\n"))
        self.assertNotIn(prompt.RESUME_NOTE, text)

    def test_resume_note_is_prepended(self):
        text = prompt.prompt_text(self.task, self.cfg.plan, PROJECT_PROMPT, resume=True)
        self.assertTrue(text.startswith(prompt.RESUME_NOTE + "\n\n"))
        self.assertIn("do not start over.", text)

    def test_empty_project_prompt_still_yields_the_header(self):
        text = prompt.prompt_text(self.task, self.cfg.plan, "\n")
        self.assertEqual(text.count("YOUR TASK IS"), 1)
        self.assertFalse(text.endswith("\n\n"))


class CommandTest(unittest.TestCase):
    def test_expands_all_three_placeholders_and_splits(self):
        argv = prompt.build_command(
            "omp -p --model {model} --config {overlay} @{prompt_file}",
            model="local-27b/Qwen",
            overlay=Path("/share/omp-agent.yml"),
            prompt_file=Path("/wt/.taskgraph-prompt.md"),
        )
        self.assertEqual(
            argv,
            [
                "omp",
                "-p",
                "--model",
                "local-27b/Qwen",
                "--config",
                "/share/omp-agent.yml",
                "@/wt/.taskgraph-prompt.md",
            ],
        )

    def test_quoted_path_with_spaces_survives(self):
        argv = prompt.build_command(
            "agent --prompt '{prompt_file}'",
            model="m",
            overlay="/o.yml",
            prompt_file="/tmp/a b/prompt.md",
        )
        self.assertEqual(argv, ["agent", "--prompt", "/tmp/a b/prompt.md"])

    def test_unknown_placeholder_is_an_error(self):
        with self.assertRaises(prompt.PromptError):
            prompt.build_command("agent {nope}", model="m", overlay="/o", prompt_file="/p")

    def test_empty_command_is_an_error(self):
        with self.assertRaises(prompt.PromptError):
            prompt.build_command("   ", model="m", overlay="/o", prompt_file="/p")


class OverlayTest(unittest.TestCase):
    def test_relative_overlay_resolves_under_the_share_dir(self):
        self.assertEqual(
            prompt.overlay_path("omp-agent.yml"), prompt.share_dir() / "omp-agent.yml"
        )

    def test_absolute_overlay_is_kept(self):
        self.assertEqual(prompt.overlay_path("/etc/agent.yml"), Path("/etc/agent.yml"))

    def test_share_dir_can_be_overridden(self):
        self.assertEqual(
            prompt.overlay_path("x.yml", share="/share"), Path("/share/x.yml")
        )

    def test_shipped_default_overlay_disables_auto_background(self):
        text = (prompt.share_dir() / "omp-agent.yml").read_text(encoding="utf-8")
        self.assertEqual(text.count("enabled: false"), 2)
        self.assertIn("bash:", text)
        self.assertIn("eval:", text)


class WritePromptTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.worktree = self.base / "wt"
        self.worktree.mkdir(parents=True)
        self.cfg = make_config(self.root, "agent {prompt_file}")
        self.task = make_task("F03", "Top strip per SPEC")

    def test_writes_the_prompt_from_the_project_prompt_file(self):
        path = prompt.write_prompt(self.worktree, self.task, self.cfg)
        self.assertEqual(path, self.worktree / prompt.PROMPT_NAME)
        text = path.read_text(encoding="utf-8")
        self.assertIn("YOUR TASK IS **F03**", text)
        self.assertIn("Never break the build.", text)

    def test_explicit_project_text_overrides_the_config_file(self):
        path = prompt.write_prompt(self.worktree, self.task, self.cfg, project="OTHER\n")
        self.assertIn("OTHER", path.read_text(encoding="utf-8"))

    def test_missing_project_prompt_is_reported(self):
        self.cfg = make_config(self.root, "agent {prompt_file}", prompt="NOPE.md")
        (self.root / "NOPE.md").unlink()
        with self.assertRaises(prompt.PromptError):
            prompt.write_prompt(self.worktree, self.task, self.cfg)


class ExcludeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)

    def test_excludes_the_prompt_file_in_a_plain_repo(self):
        worktree = self.base / "repo"
        (worktree / ".git" / "info").mkdir(parents=True)
        self.assertTrue(prompt.exclude(".taskgraph-prompt.md", worktree))
        exclude = worktree / ".git" / "info" / "exclude"
        self.assertEqual(exclude.read_text(encoding="utf-8"), ".taskgraph-prompt.md\n")
        # Re-running adds nothing (idempotent) and keeps existing entries.
        self.assertFalse(prompt.exclude(".taskgraph-prompt.md", worktree))
        self.assertEqual(exclude.read_text(encoding="utf-8"), ".taskgraph-prompt.md\n")

    def test_keeps_existing_entries_and_terminates_the_file(self):
        worktree = self.base / "repo"
        info = worktree / ".git" / "info"
        info.mkdir(parents=True)
        (info / "exclude").write_text("*.log", encoding="utf-8")
        self.assertTrue(prompt.exclude(".taskgraph-prompt.md", worktree))
        self.assertEqual(
            (info / "exclude").read_text(encoding="utf-8"), "*.log\n.taskgraph-prompt.md\n"
        )

    def test_linked_worktree_gitdir_file_is_followed(self):
        # A linked worktree's .git is a file: "gitdir: <main>/.git/worktrees/<name>".
        worktree = self.base / "wt"
        worktree.mkdir()
        real = self.base / "main" / ".git" / "worktrees" / "wt"
        real.mkdir(parents=True)
        (worktree / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
        self.assertTrue(prompt.exclude(".taskgraph-prompt.md", worktree))
        self.assertTrue((real / "info" / "exclude").is_file())

    def test_no_git_metadata_is_not_an_error(self):
        worktree = self.base / "plain"
        worktree.mkdir()
        self.assertFalse(prompt.exclude(".taskgraph-prompt.md", worktree))
        self.assertFalse((worktree / ".git").exists())


class GitDirTest(unittest.TestCase):
    def test_missing_git_reports_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(prompt.git_dir(tmp))

    def test_relative_gitdir_line_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp)
            (worktree / ".git").write_text("gitdir: ../real\n", encoding="utf-8")
            self.assertEqual(prompt.git_dir(worktree), (worktree / "../real").resolve())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
