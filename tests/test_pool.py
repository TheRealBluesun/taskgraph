"""Tests for ``taskgraph.pool`` (SPEC §4): base slots, over-session grants, the
180 s spacing and the throughput-peak veto. No network: metrics samples are
injected, so the whole policy is deterministic.
"""

import unittest

from taskgraph.config import ModelConfig
from taskgraph.metrics import Metrics
from taskgraph.pool import (
    EXTRA_SPACING,
    MIN_BUCKET_SAMPLES,
    MIN_SAMPLES,
    Sample,
    choose_model,
    token_rates,
)

METRICS_URL = "http://127.0.0.1:8002/metrics"


def model(name="m", sessions=1, max_agents=None, metrics=METRICS_URL):
    return ModelConfig(
        name=name,
        sessions=sessions,
        max_agents=sessions if max_agents is None else max_agents,
        metrics=metrics,
    )


def sample(at, running=0.0, waiting=0.0, tokens=0.0):
    return Sample(at, Metrics(running, waiting, tokens))


def slack(count=MIN_SAMPLES, running=0.0, waiting=0.0, start=0.0, step=20.0):
    """``count`` evenly spaced samples with rising tokens (no bucket matters yet)."""
    return [sample(start + i * step, running, waiting, float(i)) for i in range(count)]


def concurrency_history(base_rate, extra_rate, per=MIN_BUCKET_SAMPLES + 5, tail=130, step=1.0):
    """History at concurrency 1 (``base_rate`` tokens/s) then 2 (``extra_rate``).

    The idle ``tail`` keeps the overall mean ``running`` below 0.5, so this
    window passes the slack gate and only the throughput veto can block; both
    concurrency buckets hold at least ``MIN_BUCKET_SAMPLES`` rates.
    """
    samples = []
    at, tokens = 0.0, 0.0
    for running, rate in ((1.0, base_rate), (2.0, extra_rate)):
        for _ in range(per):
            at += step
            tokens += rate * step
            samples.append(sample(at, running, 0.0, tokens))
    for _ in range(tail):
        at += step
        samples.append(sample(at, 0.0, 0.0, tokens))
    return samples


def choose(models, assigned=None, load=None, now=1000.0, last_extra=None):
    """Run :func:`choose_model` and return ``(chosen, last_extra_after)``."""
    extra = {} if last_extra is None else last_extra
    names = [m.name for m in models]
    return choose_model(models, assigned or {}, load or {}, now, extra), extra


class BaseSlotTest(unittest.TestCase):
    def test_free_base_slot_is_chosen(self):
        self.assertEqual(choose([model(sessions=2)], {"m": 1})[0], "m")

    def test_missing_assigned_entry_counts_as_zero(self):
        self.assertEqual(choose([model(sessions=2)], {})[0], "m")

    def test_first_model_in_config_order_wins(self):
        models = [model("first"), model("second")]
        self.assertEqual(choose(models, {})[0], "first")

    def test_base_slot_beats_over_session_slot_elsewhere(self):
        # "busy" is full at its base slot but shows slack (eligible for an extra
        # agent); "spare" still has a free base slot and must win first.
        models = [model("busy", sessions=1, max_agents=2), model("spare")]
        load = {"busy": slack()}
        self.assertEqual(choose(models, {"busy": 1, "spare": 0}, load)[0], "spare")

    def test_no_model_free_returns_none(self):
        self.assertIsNone(choose([model(max_agents=1)], {"m": 1})[0])

    def test_no_models_returns_none(self):
        self.assertIsNone(choose([], {})[0])


class OverSessionTest(unittest.TestCase):
    def test_grant_records_now_in_last_extra(self):
        chosen, extra = choose(
            [model(max_agents=2)], {"m": 1}, {"m": slack()}, now=1234.5
        )
        self.assertEqual(chosen, "m")
        self.assertEqual(extra, {"m": 1234.5})

    def test_gate_table(self):
        cases = [
            ("slack", {"count": MIN_SAMPLES, "running": 0.0}, True),
            ("too few samples", {"count": MIN_SAMPLES - 1, "running": 0.0}, False),
            ("mean at boundary", {"count": MIN_SAMPLES, "running": 0.5}, False),
            ("mean below boundary", {"count": MIN_SAMPLES, "running": 0.4}, True),
            ("waiting pending", {"count": MIN_SAMPLES, "running": 0.0, "waiting": 1.0}, False),
        ]
        for label, kwargs, expected in cases:
            with self.subTest(label):
                chosen, _ = choose(
                    [model(max_agents=2)], {"m": 1}, {"m": slack(**kwargs)}
                )
                self.assertEqual(chosen == "m", expected)

    def test_metrics_endpoint_is_required(self):
        chosen, _ = choose([model(max_agents=2, metrics=None)], {"m": 1}, {"m": slack()})
        self.assertIsNone(chosen)

    def test_no_recorded_samples_blocks(self):
        self.assertIsNone(choose([model(max_agents=2)], {"m": 1}, {})[0])

    def test_at_max_agents_blocks_even_with_slack(self):
        chosen, _ = choose([model(max_agents=2)], {"m": 2}, {"m": slack()})
        self.assertIsNone(chosen)

    def test_spacing_blocks_within_180s(self):
        last_extra = {"m": 1000.0 - EXTRA_SPACING + 0.1}
        chosen, extra = choose([model(max_agents=2)], {"m": 1}, {"m": slack()}, last_extra=last_extra)
        self.assertIsNone(chosen)
        self.assertEqual(extra, last_extra)  # a blocked call records nothing

    def test_spacing_allows_at_boundary(self):
        chosen, extra = choose(
            [model(max_agents=2)], {"m": 1}, {"m": slack()}, last_extra={"m": 1000.0 - EXTRA_SPACING}
        )
        self.assertEqual(chosen, "m")
        self.assertEqual(extra, {"m": 1000.0})

    def test_spacing_is_per_model(self):
        models = [model("a", max_agents=2), model("b", max_agents=2)]
        load = {"a": slack(), "b": slack()}
        chosen, extra = choose(models, {"a": 1, "b": 1}, load, last_extra={"a": 1000.0})
        self.assertEqual(chosen, "b")
        self.assertEqual(extra, {"a": 1000.0, "b": 1000.0})


class ThroughputVetoTest(unittest.TestCase):
    def test_veto_when_sessions_plus_one_is_slower(self):
        samples = concurrency_history(base_rate=100.0, extra_rate=60.0)
        chosen, extra = choose(
            [model(sessions=1, max_agents=2)], {"m": 1}, {"m": samples}, now=samples[-1].at
        )
        self.assertIsNone(chosen)
        self.assertEqual(extra, {})

    def test_no_veto_when_sessions_plus_one_is_faster(self):
        samples = concurrency_history(base_rate=100.0, extra_rate=120.0)
        chosen, _ = choose(
            [model(sessions=1, max_agents=2)], {"m": 1}, {"m": samples}, now=samples[-1].at
        )
        self.assertEqual(chosen, "m")

    def test_no_veto_when_buckets_lack_samples(self):
        samples = concurrency_history(base_rate=100.0, extra_rate=60.0, per=MIN_BUCKET_SAMPLES)
        chosen, _ = choose(
            [model(sessions=1, max_agents=2)], {"m": 1}, {"m": samples}, now=samples[-1].at
        )
        self.assertEqual(chosen, "m")


class TokenRatesTest(unittest.TestCase):
    def test_rate_is_filed_under_the_later_running_value(self):
        samples = [sample(0, 1.0, tokens=0), sample(1, 2.0, tokens=100), sample(2, 3.0, tokens=300)]
        self.assertEqual(token_rates(samples), {2: [100.0], 3: [200.0]})

    def test_unordered_samples_are_sorted(self):
        samples = [sample(2, 0, tokens=300), sample(0, 0, tokens=0), sample(1, 0, tokens=100)]
        self.assertEqual(token_rates(samples), {0: [100.0, 200.0]})

    def test_counter_reset_is_skipped(self):
        samples = [sample(0, 0, tokens=100), sample(1, 0, tokens=50)]
        self.assertEqual(token_rates(samples), {})

    def test_non_positive_interval_is_skipped(self):
        samples = [sample(5, 0, tokens=0), sample(5, 0, tokens=10)]
        self.assertEqual(token_rates(samples), {})

    def test_flat_counter_yields_zero_rate(self):
        self.assertEqual(token_rates([sample(0, 0, tokens=5), sample(1, 0, tokens=5)]), {0: [0.0]})

    def test_empty_and_single_sample(self):
        self.assertEqual(token_rates([]), {})
        self.assertEqual(token_rates([sample(0)]), {})


if __name__ == "__main__":
    unittest.main()
