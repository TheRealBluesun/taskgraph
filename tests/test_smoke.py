"""Smoke test: the package imports and the CLI exposes every SPEC §10 subcommand."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import taskgraph
import taskgraph.cli as cli

ROOT = Path(__file__).resolve().parent.parent


class SmokeTest(unittest.TestCase):
    def test_version(self):
        self.assertEqual(taskgraph.__version__, "0.1.0")

    def test_every_subcommand_has_a_real_handler(self):
        # T19 completed the SPEC §10 surface (plus `models`/`side`): every
        # command must parse to a real handler, never the "not implemented" stub.
        argv_for = {"lease": ["r", "--", "true"], "side": ["cls", "--", "true"], "retry": ["T01"]}
        parser = cli.build_parser()
        for name in cli.SUBCOMMANDS:
            with self.subTest(command=name):
                parsed = parser.parse_args([name, *argv_for.get(name, [])])
                self.assertIsNot(parsed.func, cli._not_implemented)

    def test_run_flags_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = cli.main(["run", "--project", tmp, "--max-agents", "2", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn(cli.CONFIG_NAME, err.getvalue())

    def test_no_command_prints_help(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main([])
        self.assertEqual(rc, 2)
        self.assertIn("usage: taskgraph", out.getvalue())

    def test_entry_script_runs(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "bin" / "taskgraph"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage: taskgraph", proc.stdout)


if __name__ == "__main__":
    unittest.main()
