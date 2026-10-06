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
    RECENT_WINDOW,
    Sample,
    choose_model,
    recent_samples,
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


def choose(models, assigned=None, recent=None, history=None, now=1000.0, last_extra=None):
    """Run :func:`choose_model`; ``history`` (the veto window) defaults to ``recent``."""
    extra = {} if last_extra is None else last_extra
    recent = recent or {}
    if history is None:
        history = recent
    return choose_model(models, assigned or {}, recent, history, now, extra), extra


class BaseSlotTest(unittest.TestCase):
    def test_free_base_slot_is_chosen(self):
        self.assertEqual(choose([model(sessions=2)], {"m": 1})[0], "m")

    def test_missing_assigned_entry_counts_as_zero(self):
        self.assertEqual(choose([model(sessions=2)], {})[0], "m")

    def test_first_model_in_config_order_wins(self):
        models = [model("first"), model("second")]
        self.assertEqual(choose(models, {})[0], "first")

    def test_earlier_model_slack_beats_later_base_slot(self):
        # One config-ordered pass (SPEC §4): "busy" is at its base slot but shows
        # measured slack, so it takes the extra agent before "spare", which
        # still has a free base slot.
        models = [model("busy", sessions=1, max_agents=2), model("spare")]
        chosen, extra = choose(models, {"busy": 1, "spare": 0}, {"busy": slack()})
        self.assertEqual(chosen, "busy")
        self.assertEqual(extra, {"busy": 1000.0})

    def test_zero_capacity_model_gets_no_agent(self):
        # sessions = max_agents = 0 disables a model (SPEC §1/§4).
        self.assertIsNone(choose([model(sessions=0, max_agents=0)], {"m": 5}, {"m": slack()})[0])

    def test_agent_class_capacity_overrides_sessions(self):
        # classes = { agent = 1 } caps plan tasks below sessions (SPEC §4). No
        # metrics, so the over-session branch cannot grant the extra agent.
        capped = ModelConfig(name="m", sessions=3, max_agents=3, classes={"agent": 1})
        self.assertEqual(choose([capped], {"m": 0})[0], "m")
        self.assertIsNone(choose([capped], {"m": 1})[0])

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

    def test_assigned_above_max_agents_gets_nothing(self):
        # An operator may lower max_agents at runtime; a model already over the
        # new limit gets no further agent even though it otherwise shows slack.
        chosen, _ = choose([model(max_agents=2)], {"m": 3}, {"m": slack()})
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


class RecentSlackWindowTest(unittest.TestCase):
    def test_slack_gate_ignores_stale_busy_history(self):
        # The history was busy, but the recent window is idle: the slack gate
        # reads only the recent window, so the extra agent is granted.
        now = 10_000.0
        recent = slack(count=MIN_SAMPLES, running=0.0, start=now - 8 * 20.0, step=20.0)
        history = slack(count=MIN_SAMPLES, running=8.0, start=0.0, step=20.0) + recent
        chosen, _ = choose([model(max_agents=2)], {"m": 1}, {"m": recent}, {"m": history}, now=now)
        self.assertEqual(chosen, "m")

    def test_recent_busy_window_blocks_even_with_idle_history(self):
        # The mirror image: the server was idle, but a recent burst blocks the
        # extra agent despite the calm history.
        now = 10_000.0
        recent = slack(count=MIN_SAMPLES, running=8.0, start=now - 8 * 20.0, step=20.0)
        history = slack(count=MIN_SAMPLES, running=0.0, start=0.0, step=20.0) + recent
        chosen, _ = choose([model(max_agents=2)], {"m": 1}, {"m": recent}, {"m": history}, now=now)
        self.assertIsNone(chosen)


class RecentSamplesTest(unittest.TestCase):
    def test_keeps_only_the_window_and_drops_future_samples(self):
        samples = [sample(0.0), sample(100.0), sample(200.0), sample(350.0)]
        self.assertEqual([s.at for s in recent_samples(samples, now=200.0)], [100.0, 200.0])

    def test_window_boundary_is_inclusive(self):
        samples = [sample(0.0), sample(RECENT_WINDOW)]
        self.assertEqual([s.at for s in recent_samples(samples, now=RECENT_WINDOW)], [0.0, RECENT_WINDOW])


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

    def test_veto_uses_full_history_even_when_recent_is_quiet(self):
        # Bursts at concurrency 1 and 2 then a long idle tail: the recent window
        # is quiet (slack passes) but the veto still sees the burst in history.
        history = concurrency_history(base_rate=100.0, extra_rate=60.0, per=25, tail=400, step=1.0)
        now = history[-1].at
        recent = recent_samples(history, now)
        self.assertTrue(recent)
        self.assertTrue(all(s.metrics.running == 0.0 for s in recent))
        chosen, extra = choose(
            [model(sessions=1, max_agents=2)], {"m": 1}, {"m": recent}, {"m": history}, now=now
        )
        self.assertIsNone(chosen)
        self.assertEqual(extra, {})


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
