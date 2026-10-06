"""The shipped examples must stay valid and faithful to what they document (T18).

A stale example is a broken document: this loads every ``examples/*/taskgraph.toml``
through the real config loader and checks that the Elixir one still expresses the
facts its guide (`dev-sh-lease.md`) and the README rely on.
"""

import unittest
from pathlib import Path

from taskgraph.buckets import bucket_for
from taskgraph.config import load
from taskgraph.guard import denied
from taskgraph.trace import Call

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ELIXIR = EXAMPLES / "elixir"


class ExamplesLoadTest(unittest.TestCase):
    """Every example config must load with the current loader."""

    def test_finds_the_example_configs(self):
        configs = sorted(EXAMPLES.glob("*/taskgraph.toml"))
        self.assertTrue(configs, f"no examples/*/taskgraph.toml under {EXAMPLES}")
        for path in configs:
            with self.subTest(example=path.parent.name):
                load(path)


class ElixirExampleTest(unittest.TestCase):
    """The Elixir example config, as the guides describe it."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load(ELIXIR / "taskgraph.toml")

    def test_paths_are_project_relative(self):
        self.assertEqual(self.cfg.plan, "PLAN.md")
        self.assertEqual(self.cfg.prompt, "PROMPT.md")
        self.assertEqual(self.cfg.main, "main")
        self.assertEqual(self.cfg.worktrees, "../elixir-wt")
        self.assertEqual(self.cfg.root, ELIXIR)

    def test_gate_builds_and_tests_through_dev_sh(self):
        # The gate is what a merge must pass; the guides promise it is dev.sh's build+test.
        self.assertEqual(self.cfg.gate, "./dev.sh build && ./dev.sh test")

    def test_one_simulator_and_a_denied_xcodebuild(self):
        self.assertEqual(dict(self.cfg.resources), {"simulator": 1})
        self.assertIn("xcodebuild", self.cfg.agent.deny)
        self.assertIn("xcrun", self.cfg.agent.deny)
        # The lease wrapper (dev.sh) needs a non-empty stall budget and retries.
        self.assertGreater(self.cfg.agent.stall_secs, 0)
        self.assertGreater(self.cfg.agent.retries, 0)

    def test_models_are_in_priority_order_with_an_auxiliary_last(self):
        names = [model.name for model in self.cfg.models]
        self.assertEqual(len(names), len(set(names)))
        # A local model is tried first; the paid overflow model is last (SPEC §4).
        self.assertTrue(names[0].startswith("local-"), names)
        self.assertTrue(names[-1].startswith("deepseek/"), names)

    def test_rate_limited_local_model_falls_back_to_the_overflow_model(self):
        with_fallback = [model for model in self.cfg.models if model.fallback is not None]
        self.assertEqual(len(with_fallback), 1, "the example should show exactly one fallback")
        local = with_fallback[0]
        overflow = self.cfg.model(local.fallback)
        self.assertIsNotNone(overflow, local.fallback)
        # The fallback must be lower priority, or the pool would prefer it on its own.
        self.assertGreater(self.cfg.models.index(overflow), self.cfg.models.index(local))

    def test_oversession_rule_is_actually_configured(self):
        local = max(self.cfg.models, key=lambda model: model.max_agents)
        self.assertGreater(local.max_agents, local.sessions, "no model allows over-session agents")
        self.assertIsNotNone(local.metrics, "the over-session rule needs a metrics URL")

    def test_flash_model_splits_agent_and_side_capacity(self):
        # The .10 server fits one long-context agent and one short side job (SPEC §4).
        flash = self.cfg.model("local-vllm-flash/Qwen3.8-Flash-Next")
        self.assertIsNotNone(flash)
        self.assertEqual(flash.capacity("agent"), 1)
        self.assertEqual(flash.capacity("side"), 1)

    def test_merge_guard_blocks_the_playtest_output(self):
        self.assertIn("playtest-results", self.cfg.merge.deny_paths)
        self.assertTrue(denied("playtest-results/greedy.json", self.cfg.merge.deny_dirs))

    def test_stats_buckets_classify_the_projects_own_tools(self):
        patterns = self.cfg.stats.patterns
        self.assertEqual(bucket_for(Call("bash", "playtest.sh --player greedy"), patterns), "playtest")
        self.assertEqual(bucket_for(Call("bash", "python3 tools/genaudio.py"), patterns), "art")
        # The SPEC §9 defaults still apply (the docs promise they do).
        self.assertEqual(bucket_for(Call("bash", "xcodebuild -scheme Elixir build"), patterns), "build")

    def test_router_workers_keep_the_paid_one_last_and_capped(self):
        # SPEC §11: request-level capacity, local workers first, one paid overflow with a spend cap.
        workers = self.cfg.workers
        self.assertEqual([worker.name for worker in workers], ["flash", "27b", "deepseek"])
        self.assertEqual(workers[0].concurrency, 1, "the .10 server takes one long-context request")
        self.assertIsNotNone(workers[0].max_context, "long prompts must skip the flash worker")
        self.assertTrue(all(not worker.overflow for worker in workers[:-1]))
        paid = workers[-1]
        self.assertTrue(paid.overflow)
        self.assertEqual(paid.api_key_env, "DEEPSEEK_API_KEY")
        self.assertGreaterEqual(paid.max_requests_per_hour, 1)

    def test_the_dev_sh_guide_is_where_the_config_points(self):
        guide = ELIXIR / "dev-sh-lease.md"
        self.assertTrue(guide.is_file(), f"{guide} is missing")
        self.assertTrue(guide.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
