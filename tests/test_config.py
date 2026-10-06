"""Tests for ``taskgraph.config`` (SPEC §1)."""

import tempfile
import textwrap
import unittest
from pathlib import Path

from taskgraph.config import (
    AgentConfig,
    Config,
    ConfigError,
    MergeConfig,
    ModelConfig,
    StatsConfig,
    load,
)

# The example config from SPEC §1, verbatim (comments stripped).
SPEC_TOML = """
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "./dev.sh build && ./dev.sh test"
worktrees = "../elixir-wt"
main = "main"
links = [".env"]

[resources]
simulator = 1

[agent]
command = "omp -p --mode json --auto-approve --no-session --model {model} --thinking high --max-time 120m --config {overlay} @{prompt_file}"
overlay = "omp-agent.yml"
stall_secs = 480
retries = 2
deny = ["xcodebuild", "xcrun"]

[[models]]
name = "local-vllm-27b/Qwen3.8-27B"
sessions = 3
max_agents = 5
metrics = "http://10.10.10.15:8002/metrics"

[[models]]
name = "local-vllm/Qwen3.8-Flash-Next"
sessions = 1
"""

MINIMAL_TOML = """
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "../wt"
main = "main"

[agent]
command = "fake-agent {prompt_file}"

[[models]]
name = "m1"
sessions = 2
"""


class ConfigTestBase(unittest.TestCase):
    def load_text(self, text: str) -> Config:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "taskgraph.toml"
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        return load(path)

    def assert_error(self, text: str, *needles: str) -> ConfigError:
        with self.assertRaises(ConfigError) as ctx:
            self.load_text(text)
        for needle in needles:
            self.assertIn(needle, str(ctx.exception))
        return ctx.exception


class SpecExampleTest(ConfigTestBase):
    def test_full_example(self):
        cfg = self.load_text(SPEC_TOML)
        self.assertEqual(cfg.plan, "PLAN.md")
        self.assertEqual(cfg.prompt, "PROMPT.md")
        self.assertEqual(cfg.gate, "./dev.sh build && ./dev.sh test")
        self.assertEqual(cfg.worktrees, "../elixir-wt")
        self.assertEqual(cfg.main, "main")
        self.assertEqual(cfg.links, (".env",))
        self.assertEqual(dict(cfg.resources), {"simulator": 1})
        self.assertTrue(cfg.agent.command.startswith("omp -p --mode json"))
        self.assertIn("{prompt_file}", cfg.agent.command)
        self.assertEqual(cfg.agent.overlay, "omp-agent.yml")
        self.assertEqual(cfg.agent.stall_secs, 480.0)
        self.assertEqual(cfg.agent.retries, 2)
        self.assertEqual(cfg.agent.deny, ("xcodebuild", "xcrun"))
        self.assertEqual(len(cfg.models), 2)
        first = cfg.models[0]
        self.assertEqual(first.name, "local-vllm-27b/Qwen3.8-27B")
        self.assertEqual((first.sessions, first.max_agents), (3, 5))
        self.assertEqual(first.metrics, "http://10.10.10.15:8002/metrics")
        self.assertIsNone(first.fallback)
        second = cfg.models[1]
        self.assertEqual(second.name, "local-vllm/Qwen3.8-Flash-Next")
        self.assertEqual((second.sessions, second.max_agents), (1, 1))
        self.assertIsNone(second.metrics)

    def test_root_is_config_directory_and_model_lookup(self):
        cfg = self.load_text(MINIMAL_TOML)
        self.assertEqual(cfg.root, cfg.path.parent)
        self.assertEqual(cfg.model("m1"), cfg.models[0])
        self.assertIsNone(cfg.model("nope"))


class DefaultsTest(ConfigTestBase):
    def test_optional_keys_default(self):
        cfg = self.load_text(MINIMAL_TOML)
        self.assertEqual(cfg.links, ())
        self.assertEqual(dict(cfg.resources), {})
        self.assertEqual(cfg.agent.overlay, "omp-agent.yml")
        self.assertEqual(cfg.agent.stall_secs, 480.0)
        self.assertEqual(cfg.agent.retries, 2)
        self.assertEqual(cfg.agent.deny, ())
        self.assertEqual(cfg.models[0].max_agents, 2)  # defaults to sessions
        self.assertEqual(cfg.trailer, ())

    def test_trailer_lines_are_parsed(self):
        toml = MINIMAL_TOML.replace(
            'main = "main"', 'main = "main"\ntrailer = ["Taskgraph-Task: {id}"]'
        )
        cfg = self.load_text(toml)
        self.assertEqual(cfg.trailer, ("Taskgraph-Task: {id}",))

    def test_accepts_str_path_and_relative_resolution(self):
        cfg = self.load_text(MINIMAL_TOML)
        self.assertIsInstance(load(str(cfg.path)), Config)

    def test_fallback_must_name_a_configured_model(self):
        cfg = self.load_text(
            MINIMAL_TOML
            + """
[[models]]
name = "m2"
sessions = 1
fallback = "m1"
"""
        )
        self.assertEqual(cfg.models[1].fallback, "m1")


class MergeConfigTest(ConfigTestBase):
    def test_defaults(self):
        cfg = self.load_text(MINIMAL_TOML)
        self.assertEqual(cfg.merge, MergeConfig())
        self.assertEqual(cfg.merge.max_files, 200)
        self.assertEqual(cfg.merge.max_file_mb, 5.0)
        for entry in ("build/", ".build/", "DerivedData/", "node_modules/", "target/", "dist/"):
            self.assertIn(entry, cfg.merge.deny_dirs)

    def test_merge_table_is_parsed(self):
        toml = MINIMAL_TOML + '\n[merge]\nmax_files = 12\nmax_file_mb = 1.5\ndeny_paths = ["vendor/"]\n'
        cfg = self.load_text(toml)
        self.assertEqual(cfg.merge.max_files, 12)
        self.assertEqual(cfg.merge.max_file_mb, 1.5)
        self.assertEqual(cfg.merge.deny_paths, ("vendor/",))
        self.assertIn("vendor/", cfg.merge.deny_dirs)
        self.assertIn("build/", cfg.merge.deny_dirs)  # the defaults still apply

    def test_max_files_must_be_positive(self):
        self.assert_error(MINIMAL_TOML + "\n[merge]\nmax_files = 0\n", "[merge] max_files' must be >= 1")

    def test_max_file_mb_must_be_positive(self):
        self.assert_error(MINIMAL_TOML + "\n[merge]\nmax_file_mb = 0\n", "[merge] 'max_file_mb' must be > 0")

    def test_deny_paths_must_be_an_array_of_strings(self):
        self.assert_error(
            MINIMAL_TOML + '\n[merge]\ndeny_paths = "vendor"\n',
            "[merge] deny_paths' must be an array of non-empty strings",
        )

    def test_merge_must_be_a_table(self):
        self.assert_error(
            MINIMAL_TOML.replace('main = "main"', 'main = "main"\nmerge = 3'),
            "[merge] must be a table",
        )


class StatsConfigTest(ConfigTestBase):
    def test_defaults_to_the_spec_buckets(self):
        cfg = self.load_text(MINIMAL_TOML)
        self.assertEqual(cfg.stats, StatsConfig())
        self.assertEqual([name for name, _ in cfg.stats.patterns], ["lease-wait", "build", "test", "wait"])

    def test_buckets_are_parsed_in_file_order_and_beat_the_defaults(self):
        toml = MINIMAL_TOML + '\n[stats.buckets]\nmigrations = "\\\\bmigrate\\\\b"\nlint = "ruff"\n'
        cfg = self.load_text(toml)
        self.assertEqual(cfg.stats.buckets, (("migrations", r"\bmigrate\b"), ("lint", "ruff")))
        self.assertEqual(
            [name for name, _ in cfg.stats.patterns],
            ["migrations", "lint", "lease-wait", "build", "test", "wait"],
        )

    def test_a_configured_name_replaces_the_default_of_the_same_name(self):
        cfg = self.load_text(MINIMAL_TOML + '\n[stats.buckets]\nbuild = "^gradle$"\n')
        patterns = dict(cfg.stats.patterns)
        self.assertEqual(patterns["build"], "^gradle$")
        self.assertEqual(
            [name for name, _ in cfg.stats.patterns], ["build", "lease-wait", "test", "wait"]
        )

    def test_invalid_regex_is_rejected(self):
        self.assert_error(
            MINIMAL_TOML + '\n[stats.buckets]\nbad = "([unclosed"\n',
            "[stats.buckets] 'bad' is not a valid regex",
        )

    def test_bucket_value_must_be_a_regex_string(self):
        self.assert_error(
            MINIMAL_TOML + "\n[stats.buckets]\nbuild = 3\n",
            "[stats.buckets] 'build' must be a non-empty regex",
        )

    def test_other_is_reserved(self):
        self.assert_error(
            MINIMAL_TOML + '\n[stats.buckets]\nother = "x"\n',
            "[stats.buckets] 'other' is reserved",
        )

    def test_buckets_must_be_a_table(self):
        self.assert_error(
            MINIMAL_TOML + '\n[stats]\nbuckets = ["build"]\n',
            "[stats] 'buckets' must be a table",
        )

    def test_stats_must_be_a_table(self):
        self.assert_error(
            MINIMAL_TOML.replace('main = "main"', 'main = "main"\nstats = 3'), "[stats] must be a table"
        )


class ModelCapacityTest(ConfigTestBase):
    def test_zero_capacity_disables_a_model(self):
        cfg = self.load_text(
            MINIMAL_TOML
            + """
[[models]]
name = "off"
sessions = 0
max_agents = 0
"""
        )
        self.assertEqual(cfg.model("off"), ModelConfig(name="off", sessions=0, max_agents=0))

    def test_max_agents_defaults_to_zero_sessions(self):
        cfg = self.load_text(MINIMAL_TOML.replace("sessions = 2", "sessions = 0"))
        self.assertEqual((cfg.models[0].sessions, cfg.models[0].max_agents), (0, 0))

    def test_negative_sessions_and_max_agents_rejected(self):
        self.assert_error(
            MINIMAL_TOML.replace("sessions = 2", "sessions = -1"), "sessions' must be >= 0"
        )
        self.assert_error(
            SPEC_TOML.replace("max_agents = 5", "max_agents = -1"), "max_agents' must be >= 0"
        )


class JobClassesTest(ConfigTestBase):
    """``[models.classes]`` job capacities (SPEC §4)."""

    def test_agent_class_defaults_to_sessions_and_other_classes_are_absent(self):
        cfg = self.load_text(MINIMAL_TOML)
        model = cfg.models[0]
        self.assertEqual(model.classes, {})
        self.assertEqual(model.capacity("agent"), 2)
        self.assertEqual(model.capacity("side"), 0)

    def test_classes_split_capacity_and_override_the_agent_default(self):
        toml = MINIMAL_TOML.replace("sessions = 2", "sessions = 3\nclasses = { agent = 1, side = 2 }")
        model = self.load_text(toml).models[0]
        self.assertEqual(model.classes, {"agent": 1, "side": 2})
        self.assertEqual(model.capacity("agent"), 1)
        self.assertEqual(model.capacity("side"), 2)
        self.assertEqual(model.capacity("gpu"), 0)

    def test_classes_must_be_a_table_of_non_negative_ints(self):
        self.assert_error(
            MINIMAL_TOML.replace("sessions = 2", 'sessions = 2\nclasses = ["side"]'),
            "'classes' must be a table of capacities",
        )
        self.assert_error(
            MINIMAL_TOML.replace("sessions = 2", 'sessions = 2\nclasses = { side = "one" }'),
            "'classes.side' must be an integer >= 0",
        )
        self.assert_error(
            MINIMAL_TOML.replace("sessions = 2", "sessions = 2\nclasses = { side = -1 }"),
            "'classes.side' must be an integer >= 0",
        )

    def test_zero_disables_a_class(self):
        toml = MINIMAL_TOML.replace("sessions = 2", "sessions = 2\nclasses = { agent = 0, side = 1 }")
        model = self.load_text(toml).models[0]
        self.assertEqual(model.capacity("agent"), 0)
        self.assertEqual(model.capacity("side"), 1)


class InvalidConfigTest(ConfigTestBase):
    def test_missing_required_root_key_names_key_and_file(self):
        err = self.assert_error(MINIMAL_TOML.replace('main = "main"\n', ""), "missing required key 'main'")
        self.assertIn("taskgraph.toml", str(err))

    def test_missing_agent_command(self):
        self.assert_error(
            MINIMAL_TOML.replace('command = "fake-agent {prompt_file}"', ""),
            "missing required key '[agent] command'",
        )

    def test_wrong_type_names_key(self):
        self.assert_error(SPEC_TOML.replace("sessions = 3", 'sessions = "three"'), "sessions' must be an integer")

    def test_resources_capacity_must_be_positive(self):
        self.assert_error(
            SPEC_TOML.replace("simulator = 1", "simulator = 0"),
            "[resources] 'simulator' must be a positive integer",
        )

    def test_models_must_be_present_and_non_empty(self):
        self.assert_error(MINIMAL_TOML.split("[[models]]")[0], "missing required [[models]] entries")

    def test_duplicate_model_names(self):
        self.assert_error(
            MINIMAL_TOML
            + """
[[models]]
name = "m1"
sessions = 1
""",
            "duplicate model name 'm1'",
        )

    def test_max_agents_below_sessions(self):
        self.assert_error(SPEC_TOML.replace("max_agents = 5", "max_agents = 2"), "'max_agents' (2) must be >= 'sessions' (3)")

    def test_unknown_fallback(self):
        toml = MINIMAL_TOML.replace(
            'name = "m1"\nsessions = 2',
            'name = "m1"\nsessions = 2\nfallback = "ghost"',
        )
        self.assert_error(toml, "unknown fallback 'ghost'")

    def test_trailer_must_be_an_array_of_strings(self):
        toml = MINIMAL_TOML.replace('main = "main"', 'main = "main"\ntrailer = "x"')
        self.assert_error(toml, "trailer' must be an array of non-empty strings")

    def test_max_lease_secs_must_be_a_positive_number(self):
        self.assert_error(
            MINIMAL_TOML.replace('command = "fake-agent', 'max_lease_secs = 0\ncommand = "fake-agent'),
            "[agent] 'max_lease_secs' must be > 0",
        )
        self.assert_error(
            MINIMAL_TOML.replace('command = "fake-agent', 'max_lease_secs = "soon"\ncommand = "fake-agent'),
            "[agent] max_lease_secs' must be a number",
        )

    def test_invalid_toml(self):
        self.assert_error("plan = [unclosed", "invalid TOML")

    def test_missing_file(self):
        with self.assertRaises(ConfigError) as ctx:
            load("/nonexistent/taskgraph.toml")
        self.assertIn("cannot read config", str(ctx.exception))


class DataclassTest(unittest.TestCase):
    def test_model_config_defaults(self):
        model = ModelConfig(name="m", sessions=2, max_agents=2)
        self.assertIsNone(model.metrics)
        self.assertIsNone(model.fallback)
        self.assertEqual(model.classes, {})
        self.assertEqual(model.capacity("agent"), 2)

    def test_agent_config_defaults(self):
        agent = AgentConfig(command="omp")
        self.assertEqual(agent.overlay, "omp-agent.yml")
        self.assertEqual(agent.stall_secs, 480.0)
        self.assertEqual(agent.max_lease_secs, 1800.0)
        self.assertEqual(agent.retries, 2)
        self.assertEqual(agent.deny, ())


if __name__ == "__main__":
    unittest.main()
