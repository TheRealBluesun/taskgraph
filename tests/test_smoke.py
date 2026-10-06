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

    def test_remaining_subcommands_parse_and_stub(self):
        # `lease`/`leases` are implemented (T07), `run` in T13, `stop`/`retry`
        # in T14, `status` in T15; the rest stub.
        for name in cli.SUBCOMMANDS:
            if name in {"lease", "leases", "run", "stop", "retry", "status"}:
                continue
            with self.subTest(command=name):
                argv = [name]
                if name == "retry":
                    argv += ["T01"]
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = cli.main(argv)
                self.assertEqual(rc, 1)
                self.assertEqual(err.getvalue().strip(), f"taskgraph {name}: not implemented")

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
