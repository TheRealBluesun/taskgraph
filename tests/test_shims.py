"""Tests for the deny shims (SPEC §5).

A fake ``mytool`` script in a temp dir stands in for ``xcodebuild``: we assert
the shim denies calls without ``TASKGRAPH_LEASE``, runs the real binary (with
arguments) when leased, and that real-binary resolution never picks the shim
itself, even with the shim dir first on PATH.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from taskgraph import shims

NO_LEASE = object()  # sentinel: env without TASKGRAPH_LEASE at all


def write_executable(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` (creating parents) and mark it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


class ShimTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.state = self.base / "state"
        self.shimdir = self.state / "shims"
        self.bindir = self.base / "bin"

    def make_tool(self, name: str = "mytool", body: str = 'printf "arg:%s\\n" "$@"') -> Path:
        return write_executable(self.bindir / name, f"#!/bin/sh\n{body}\n")

    def real(self, name: str = "mytool") -> str:
        return os.path.realpath(self.bindir / name)

    def path(self, extra: str = "") -> str:
        return os.pathsep.join(
            p for p in [os.fspath(self.shimdir), os.fspath(self.bindir), extra] if p
        )

    def env(self, lease=NO_LEASE) -> dict:
        env = dict(os.environ)
        env["PATH"] = self.path()
        env.pop(shims.LEASE_ENV, None)
        if lease is not NO_LEASE:
            env[shims.LEASE_ENV] = lease
        return env

    def install(self, *commands: str, resource: str | None = None, path: str | None = None) -> dict:
        """``create_shims`` with the fake bin dir (and shim dir) on PATH."""
        if path is None:
            path = self.path()
        return shims.create_shims(commands, resource=resource, root=self.state, path=path)

    def run_shim(self, name: str, *args: str, lease=NO_LEASE):
        return subprocess.run(
            [os.fspath(self.shimdir / name), *args],
            capture_output=True,
            text=True,
            env=self.env(lease),
            timeout=30,
        )


class CreateShimsTest(ShimTestCase):
    def test_creates_one_executable_shim_per_command(self):
        self.make_tool()
        self.make_tool("othertool", 'echo "real othertool"')
        created = self.install("mytool", "othertool", resource="simulator")
        self.assertEqual(set(created), {"mytool", "othertool"})
        self.assertEqual(created["mytool"], self.shimdir / "mytool")
        for path in created.values():
            self.assertTrue(path.is_file())
            self.assertTrue(os.access(path, os.X_OK), f"{path} is not executable")
        self.assertEqual(sorted(p.name for p in self.shimdir.iterdir()), ["mytool", "othertool"])

    def test_shim_execs_absolute_real_path_and_skips_the_shim_dir(self):
        self.make_tool()
        # An old shim already sits in the shim dir, first on PATH: resolution must ignore it.
        write_executable(self.shimdir / "mytool", "#!/bin/sh\nexit 3\n")
        self.install("mytool")
        script = (self.shimdir / "mytool").read_text(encoding="utf-8")
        self.assertIn(f'exec \'{self.real()}\' "$@"', script)
        self.assertNotIn(os.fspath(self.shimdir), script)
        self.assertTrue(script.startswith("#!/bin/sh\n"))
        self.assertIn(shims.MARKER, script)

    def test_missing_command_is_skipped(self):
        self.assertEqual(self.install("nope"), {})
        self.assertFalse((self.shimdir / "nope").exists())

    def test_rejects_non_command_names(self):
        for bad in ("/usr/bin/xcodebuild", "sub/xcrun", "..", "a b", "x;rm -rf /"):
            with self.subTest(name=bad):
                with self.assertRaises(shims.ShimsError):
                    self.install(bad)
        self.assertEqual(list(self.shimdir.glob("*")), [])

    def test_dedupes_repeated_commands(self):
        self.make_tool()
        self.assertEqual(list(self.install("mytool", "mytool")), ["mytool"])

    def test_rewrites_the_shim_when_the_resource_changes(self):
        self.make_tool()
        self.install("mytool", resource="simulator")
        self.install("mytool", resource="gpu")
        self.assertIn("the shared gpu", (self.shimdir / "mytool").read_text(encoding="utf-8"))

    def test_no_temp_files_are_left_behind(self):
        self.make_tool()
        self.install("mytool")
        self.assertEqual([p.name for p in self.shimdir.iterdir()], ["mytool"])

    def test_prunes_generated_shims_no_longer_denied(self):
        self.make_tool()
        self.make_tool("xcrun", 'echo "real xcrun"')
        self.install("mytool", "xcrun")
        self.assertEqual(set(self.install("mytool")), {"mytool"})
        self.assertFalse((self.shimdir / "xcrun").exists())

    def test_prune_leaves_foreign_files_alone(self):
        self.make_tool()
        self.install("mytool")
        foreign = write_executable(self.shimdir / "handmade", "#!/bin/sh\nexit 0\n")
        self.install("mytool")
        self.assertTrue(foreign.exists())

    def test_default_root_follows_taskgraph_state(self):
        self.make_tool()
        os.environ["TASKGRAPH_STATE"] = os.fspath(self.state)
        self.addCleanup(lambda: os.environ.pop("TASKGRAPH_STATE", None))
        self.assertEqual(shims.shim_dir(), self.shimdir)
        self.assertEqual(set(self.install("mytool")), {"mytool"})
        self.assertIn("the shared resource", (self.shimdir / "mytool").read_text())


class ShimBehaviourTest(ShimTestCase):
    def setUp(self):
        super().setUp()
        self.make_tool()
        self.install("mytool", resource="simulator")

    def test_denies_without_lease(self):
        proc = self.run_shim("mytool", "alpha")
        self.assertEqual(proc.returncode, shims.DENIED_EXIT)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(
            proc.stderr.strip(),
            "mytool is disabled for agents: run it through the project's lease wrapper "
            "(e.g. ./dev.sh) — it waits for the shared simulator",
        )

    def test_denies_when_lease_env_is_empty(self):
        proc = self.run_shim("mytool", lease="")
        self.assertEqual(proc.returncode, shims.DENIED_EXIT)
        self.assertIn("is disabled for agents", proc.stderr)

    def test_runs_the_real_command_with_arguments_when_leased(self):
        proc = self.run_shim("mytool", "alpha", "beta gamma", lease="simulator")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "arg:alpha\narg:beta gamma\n")

    def test_forwards_the_real_command_exit_code(self):
        self.make_tool("failing", "exit 7")
        self.install("failing")
        proc = self.run_shim("failing", lease="simulator")
        self.assertEqual(proc.returncode, 7)

    def test_resolves_through_path_with_shim_dir_first(self):
        # What an agent's shell does: look up the bare name on its PATH.
        proc = subprocess.run(
            ["/bin/sh", "-c", "mytool alpha"],
            capture_output=True,
            text=True,
            env=self.env("simulator"),
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "arg:alpha\n")

    def test_message_quotes_awkward_resource_names(self):
        self.install("mytool", resource="it's a gpu")
        proc = self.run_shim("mytool")
        self.assertEqual(proc.returncode, shims.DENIED_EXIT)
        self.assertIn("the shared it's a gpu", proc.stderr)

    def test_denial_message_defaults_to_a_generic_resource(self):
        self.assertEqual(
            shims.denial_message("xcrun"),
            "xcrun is disabled for agents: run it through the project's lease wrapper "
            "(e.g. ./dev.sh) — it waits for the shared resource",
        )
        self.assertIn("the shared simulator", shims.denial_message("xcrun", "simulator"))


class PathWithShimsTest(ShimTestCase):
    def test_shim_dir_is_first(self):
        result = shims.path_with_shims(base=f"/usr/bin{os.pathsep}/bin", root=self.state)
        self.assertEqual(result.split(os.pathsep), [os.fspath(self.shimdir), "/usr/bin", "/bin"])

    def test_existing_shim_dir_is_deduped(self):
        base = os.pathsep.join(["/usr/bin", os.fspath(self.shimdir), "/bin"])
        result = shims.path_with_shims(base=base, root=self.state)
        self.assertEqual(result.split(os.pathsep), [os.fspath(self.shimdir), "/usr/bin", "/bin"])

    def test_empty_entries_are_dropped(self):
        result = shims.path_with_shims(base=f"{os.pathsep}/usr/bin{os.pathsep}", root=self.state)
        self.assertEqual(result.split(os.pathsep), [os.fspath(self.shimdir), "/usr/bin"])

    def test_real_command_returns_none_without_the_binary(self):
        self.assertIsNone(shims.real_command("nope", self.shimdir, self.path()))


if __name__ == "__main__":
    unittest.main()
