"""``taskgraph models`` tests (SPEC §4): throughput per concurrency bucket.

Pure :func:`taskgraph.models.rates`/:func:`render` are driven from injected
samples; :func:`collect` reads a temp project's ``state.json``; one test drives
the real ``bin/taskgraph models`` CLI. No network.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from taskgraph import config, metrics, models, pool, state

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "taskgraph"

TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "wt"
main = "main"

[agent]
command = "true"

[[models]]
name = "m1"
sessions = 1
metrics = "http://127.0.0.1:8002/metrics"

[[models]]
name = "m2"
sessions = 1
"""


def history(rates, per=4, step=1.0):
    """A sample history whose concurrency buckets measure exactly ``rates``.

    A generation counter that grows by each bucket's own rate means the interval
    that crosses into a new bucket already carries the new rate, so the first
    bucket holds ``per - 1`` measurements and every later one ``per``, each of
    them exactly the bucket's rate.
    """
    samples, at, tokens = [], 0.0, 0.0
    for running, rate in rates:
        for _ in range(per):
            at += step
            tokens += rate * step
            samples.append(pool.Sample(at, metrics.Metrics(float(running), 0.0, tokens)))
    return samples


def raw(sample):
    """Serialize a sample the way ``state.samples`` stores it."""
    return sample.to_dict()


class RatesTest(unittest.TestCase):
    def test_means_and_peak_for_a_falling_throughput(self):
        # The 27B server from SPEC §4: 1 request 81 tok/s, 2 → 69, 3 → 49.
        buckets = models.rates(history([(1, 81.0), (2, 69.0), (3, 49.0)]))
        self.assertEqual([b.running for b in buckets], [1, 2, 3])
        self.assertEqual([round(b.tokens_per_sec, 1) for b in buckets], [81.0, 69.0, 49.0])
        self.assertEqual([b.samples for b in buckets], [3, 4, 4])
        self.assertEqual([b.peak for b in buckets], [True, False, False])

    def test_peak_can_be_above_sessions(self):
        buckets = models.rates(history([(1, 50.0), (2, 80.0)]))
        self.assertEqual([b.peak for b in buckets], [False, True])

    def test_a_tie_marks_the_lowest_concurrency(self):
        buckets = models.rates(history([(2, 60.0), (1, 60.0)]))
        self.assertEqual([b.running for b in buckets], [1, 2])
        self.assertEqual([b.peak for b in buckets], [True, False])

    def test_one_sample_is_not_enough_for_a_rate(self):
        self.assertEqual(models.rates(history([(1, 81.0)], per=1)), ())

    def test_no_samples(self):
        self.assertEqual(models.rates([]), ())


class RenderTest(unittest.TestCase):
    def test_renders_every_bucket_and_the_peak(self):
        table = models.ModelTable(
            (
                models.ModelRates(
                    "m1",
                    (
                        models.RateBucket(1, 81.0, 40, True),
                        models.RateBucket(2, 69.0, 35, False),
                    ),
                ),
                models.ModelRates("m2"),
            )
        )
        lines = models.render(table).splitlines()
        self.assertEqual(lines[0].split(), ["MODEL", "RUNNING", "TOKENS/S", "SAMPLES", "PEAK"])
        self.assertEqual(lines[1].split(), ["m1", "1", "81.0", "40", "*"])
        self.assertEqual(lines[2].split(), ["m1", "2", "69.0", "35"])
        self.assertEqual(lines[3].split(), ["m2", "-", "-", "0"])

    def test_no_models(self):
        self.assertEqual(models.render(models.ModelTable()), "no models configured")


class ProjectTestCase(unittest.TestCase):
    """A temp project with a config and a state dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.state_root = self.base / "state"
        (self.root / "PLAN.md").write_text("- [ ] T01 first\n", encoding="utf-8")
        (self.root / "PROMPT.md").write_text("rules\n", encoding="utf-8")
        (self.root / "taskgraph.toml").write_text(TOML, encoding="utf-8")
        self.cfg = config.load(self.root / "taskgraph.toml")

    def save(self, **fields):
        state.save(state.State(**fields), state.state_path(self.root, self.state_root))


class CollectTest(ProjectTestCase):
    def test_buckets_come_from_the_full_recorded_history(self):
        self.save(
            samples={
                "m1": [raw(s) for s in history([(1, 81.0), (2, 69.0)])],
                "m2": [raw(s) for s in history([(1, 40.0)], per=1)],  # one sample: no rate
            }
        )
        table = models.collect(self.cfg, state_root=self.state_root)
        self.assertEqual([row.name for row in table.models], ["m1", "m2"])
        self.assertEqual(
            [(b.running, round(b.tokens_per_sec, 1), b.peak) for b in table.models[0].buckets],
            [(1, 81.0, True), (2, 69.0, False)],
        )
        self.assertEqual(table.models[1].buckets, ())

    def test_no_recorded_samples(self):
        self.save()
        table = models.collect(self.cfg, state_root=self.state_root)
        self.assertEqual([row.buckets for row in table.models], [(), ()])


class CliTest(ProjectTestCase):
    def test_prints_the_table_for_a_fixture_state(self):
        self.save(samples={"m1": [raw(s) for s in history([(1, 81.0), (2, 69.0)])]})
        result = subprocess.run(
            [sys.executable, str(BIN), "models"],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=dict(os.environ, TASKGRAPH_STATE=str(self.state_root)),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TOKENS/S", result.stdout)
        self.assertIn("81.0", result.stdout)
        self.assertIn("*", result.stdout)
        self.assertIn("m2", result.stdout)

    def test_malformed_state_is_reported(self):
        path = state.state_path(self.root, self.state_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(BIN), "models"],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=dict(os.environ, TASKGRAPH_STATE=str(self.state_root)),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("taskgraph models:", result.stderr)

    def test_no_config_is_reported(self):
        result = subprocess.run(
            [sys.executable, str(BIN), "models"],
            cwd=self.base,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no taskgraph.toml", result.stderr)


if __name__ == "__main__":
    unittest.main()
