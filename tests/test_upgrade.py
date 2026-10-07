"""Unit tests for the upgrade pick (SPEC §7 "upgrade on free").

Pure policy: no processes, no files, no clock — every input is built here.
"""

import unittest
from dataclasses import dataclass

from taskgraph import upgrade
from taskgraph.config import ModelConfig
from taskgraph.plan import Task

TOP = ModelConfig(name="top", sessions=1, max_agents=2)
MID = ModelConfig(name="mid", sessions=2, max_agents=3)
LOW = ModelConfig(name="low", sessions=4, max_agents=4)
MODELS = (TOP, MID, LOW)

NOW = 1000.0
WINDOW = 1200.0


@dataclass(frozen=True)
class Runner:
    """The ``AgentRecord`` fields ``pick_upgrade`` reads."""

    id: str
    model: str
    started: float


def task(tid: str, *, deps=(), done=False) -> Task:
    return Task(id=tid, text=tid, done=done, deps=tuple(deps))


def pick(tasks, running, assigned=None, *, window=WINDOW, models=MODELS, **kwargs):
    return upgrade.pick_upgrade(
        tasks,
        running,
        models,
        {} if assigned is None else assigned,
        NOW,
        window,
        **kwargs,
    )


class PickUpgradeTest(unittest.TestCase):
    def test_picks_the_critical_path_task(self):
        tasks = {"T01": task("T01"), "T02": task("T02"), "T03": task("T03", deps=("T01",))}
        running = {
            "T01": Runner("T01", "low", NOW - 60.0),
            "T02": Runner("T02", "low", NOW - 30.0),
        }
        self.assertEqual(pick(tasks, running), "T01")

    def test_plan_order_breaks_ties(self):
        tasks = {"T01": task("T01"), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "low", NOW - 5.0),
            "T02": Runner("T02", "low", NOW - 600.0),
        }
        self.assertEqual(pick(tasks, running), "T01")

    def test_queued_work_keeps_the_slot(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running, queued=["T09"]))

    def test_no_upgrade_when_the_top_model_is_full(self):
        tasks = {"T01": task("T01"), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "top", NOW - 10.0),
            "T02": Runner("T02", "low", NOW - 10.0),
        }
        self.assertIsNone(pick(tasks, running, assigned={"top": 1}))

    def test_wide_top_tier_does_not_upgrade_its_own_agent(self):
        models = (ModelConfig(name="wide", sessions=2, max_agents=2), LOW)
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "wide", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running, models=models))

    def test_window_zero_disables_upgrading(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running, window=0.0))

    def test_task_older_than_the_window_is_skipped(self):
        tasks = {"T01": task("T01"), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "low", NOW - WINDOW - 1.0),
            "T02": Runner("T02", "low", NOW - WINDOW),  # boundary is inclusive
        }
        self.assertEqual(pick(tasks, running), "T02")

    def test_task_started_inside_the_window(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertEqual(pick(tasks, running), "T01")

    def test_merging_task_is_skipped(self):
        tasks = {"T01": task("T01"), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "low", NOW - 60.0),
            "T02": Runner("T02", "low", NOW - 60.0),
        }
        self.assertEqual(pick(tasks, running, merging=["T01"]), "T02")

    def test_already_upgraded_task_is_skipped(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running, upgraded=["T01"]))

    def test_forced_task_is_skipped(self):
        """A quota fallback must not go back to the model that rate-limited it."""
        tasks = {"T01": task("T01"), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "low", NOW - 60.0),
            "T02": Runner("T02", "mid", NOW - 60.0),
        }
        self.assertEqual(pick(tasks, running, forced=["T01"]), "T02")

    def test_unknown_model_is_skipped(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "ghost", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running))

    def test_done_task_is_skipped(self):
        tasks = {"T01": task("T01", done=True), "T02": task("T02")}
        running = {
            "T01": Runner("T01", "low", NOW - 10.0),
            "T02": Runner("T02", "low", NOW - 20.0),
        }
        self.assertEqual(pick(tasks, running), "T02")

    def test_no_pending_tasks(self):
        tasks = {"T01": task("T01", done=True)}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running))

    def test_no_models(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "low", NOW - 10.0)}
        self.assertIsNone(pick(tasks, running, models=()))

    def test_middle_tier_is_a_valid_source(self):
        tasks = {"T01": task("T01")}
        running = {"T01": Runner("T01", "mid", NOW - 10.0)}
        self.assertEqual(pick(tasks, running), "T01")


if __name__ == "__main__":
    unittest.main()
