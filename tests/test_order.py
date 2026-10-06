"""Tests for ``taskgraph.order`` (SPEC §3)."""

import unittest

from taskgraph.plan import parse
from taskgraph.order import critical_paths, runnable

# A1 is last in plan order yet unblocks the longest chain (A1 -> B1 -> C1);
# Z1/Z2 are leaves with chain 1.
PLAN = """\
- [ ] Z1 first leaf
- [ ] Z2 second leaf
- [ ] B1 [deps: A1]
- [ ] C1 [deps: B1]
- [ ] A1 root
"""


def ids(task_list):
    return [task.id for task in task_list]


class CriticalPathTest(unittest.TestCase):
    def test_leaf_has_chain_one(self):
        self.assertEqual(critical_paths(parse("- [ ] A1 leaf"))["A1"], 1)

    def test_chain_counts_pending_dependents(self):
        chains = critical_paths(parse(PLAN))
        self.assertEqual(chains["A1"], 3)
        self.assertEqual(chains["B1"], 2)
        self.assertEqual(chains["C1"], 1)
        self.assertEqual(chains["Z1"], 1)

    def test_done_dependent_does_not_count(self):
        chains = critical_paths(parse("- [ ] A1 root\n- [x] B1 [deps: A1]"))
        self.assertEqual(chains["A1"], 1)

    def test_dependency_on_unknown_id_is_ignored(self):
        self.assertEqual(critical_paths(parse("- [ ] A1 leaf [deps: NOPE]"))["A1"], 1)

    def test_cycle_terminates_and_back_edge_counts_zero(self):
        chains = critical_paths(parse("- [ ] A1 [deps: B1]\n- [ ] B1 [deps: A1]"))
        self.assertEqual(set(chains), {"A1", "B1"})
        self.assertEqual(max(chains.values()), 2)

    def test_self_dependency_terminates(self):
        self.assertEqual(critical_paths(parse("- [ ] A1 [deps: A1]"))["A1"], 1)


class RunnableTest(unittest.TestCase):
    def test_critical_path_beats_plan_order(self):
        tasks = parse(PLAN)
        ready = runnable(tasks)
        # A1 (chain 3) first although it is last in the plan; then the leaves.
        self.assertEqual(ids(ready)[0], "A1")
        self.assertEqual(set(ids(ready)), {"A1", "Z1", "Z2"})

    def test_plan_order_breaks_ties(self):
        self.assertEqual(ids(runnable(parse(PLAN)))[1:], ["Z1", "Z2"])

    def test_started_rank_beats_plan_order(self):
        tasks = parse(PLAN)
        # Z2 has notes (rank 0), Z1 has a worktree only (rank 1).
        rank = {"Z1": 1, "Z2": 0}
        ready = runnable(tasks, started=lambda task: rank.get(task.id, 9))
        self.assertEqual(ids(ready)[1:], ["Z2", "Z1"])

    def test_blocked_and_running_are_excluded(self):
        tasks = parse(PLAN)
        ready = runnable(tasks, running=["A1"], blocked=["Z1"])
        self.assertEqual(ids(ready), ["Z2"])

    def test_done_and_unmet_dependency_are_not_runnable(self):
        tasks = parse("- [x] A1 done\n- [ ] B1 [deps: A1]\n- [ ] C1 [deps: D1]\n- [ ] D1 pending")
        self.assertEqual(ids(runnable(tasks, running=["D1"])), ["B1"])

    def test_unknown_dependency_counts_as_done(self):
        self.assertEqual(ids(runnable(parse("- [ ] A1 [deps: NOPE]"))), ["A1"])

    def test_dependent_appears_once_its_dependency_is_done(self):
        self.assertEqual(ids(runnable(parse("- [ ] A1 root\n- [ ] B1 [deps: A1]"))), ["A1"])
        self.assertEqual(ids(runnable(parse("- [x] A1 root\n- [ ] B1 [deps: A1]"))), ["B1"])

    def test_cycled_tasks_are_never_runnable(self):
        tasks = parse("- [ ] A1 [deps: B1]\n- [ ] B1 [deps: A1]")
        self.assertEqual(runnable(tasks), [])

    def test_empty_plan_is_empty(self):
        self.assertEqual(runnable({}), [])


if __name__ == "__main__":
    unittest.main()
