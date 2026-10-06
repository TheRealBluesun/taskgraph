"""Assign running agents to models (SPEC §4). Pure policy: the clock, the
recorded metrics and the previous over-session grants are all passed in, so
every branch is unit-testable without a server.

:func:`choose_model` makes **one pass in config order**: a model is free when it
has a free *base* slot (``assigned < sessions``) *or* its over-session gates
pass, so an earlier model's measured slack beats a later model's base slot.
Over-session means one agent beyond ``sessions``, up to ``max_agents``, and only
while the model's own vLLM counters show real slack. Agents spend roughly half
their time blocked on tools (simulator, builds), so more agents than ``sessions``
usually raises GPU utilisation — until generation throughput starts to fall
(speculative decoding + pipeline parallel), which :func:`token_rates` records and
the throughput veto enforces.

The slack estimate reads only the *recent* window (:func:`recent_samples`,
~180 s); the throughput veto reads the caller's full *history*, so a burst that
has since gone quiet still vetoes.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .metrics import Metrics

__all__ = [
    "EXTRA_SPACING",
    "MIN_BUCKET_SAMPLES",
    "MIN_SAMPLES",
    "RECENT_WINDOW",
    "Sample",
    "choose_model",
    "recent_samples",
    "token_rates",
]

# The slack estimate needs a few minutes of history before it is trusted:
# 9 samples is ~3 min at the 20 s tick (SPEC §4).
MIN_SAMPLES = 9

# The slack gate trusts only samples from this far back (SPEC §4: "the last 9
# ticks / 180 s"); older samples still count for the throughput veto.
RECENT_WINDOW = 180.0

# Two over-session agents on one model must start at least this far apart, so a
# briefly idle server is not piled on (SPEC §4).
EXTRA_SPACING = 180.0

# A throughput bucket is only compared once it holds this many rate samples
# (SPEC §4); below it the veto abstains.
MIN_BUCKET_SAMPLES = 20


@dataclass(frozen=True)
class Sample:
    """One metrics scrape recorded by the scheduler.

    ``at`` must be on the same clock as the ``now`` passed to
    :func:`choose_model` (and to the times in ``last_extra``).
    """

    at: float
    metrics: Metrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize for scheduler state (SPEC §7)."""
        m = self.metrics
        return {
            "at": self.at,
            "metrics": {
                "running": m.running,
                "waiting": m.waiting,
                "generation_tokens": m.generation_tokens,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sample | None":
        """Rebuild a sample from state; ``None`` when the entry is malformed."""
        try:
            at = float(data["at"])
            raw = data.get("metrics") or {}
            return cls(
                at,
                Metrics(
                    float(raw.get("running", 0.0)),
                    float(raw.get("waiting", 0.0)),
                    float(raw.get("generation_tokens", 0.0)),
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


def choose_model(
    models: Sequence[ModelConfig],
    assigned: Mapping[str, int],
    recent: Mapping[str, Sequence[Sample]],
    history: Mapping[str, Sequence[Sample]],
    now: float,
    last_extra: MutableMapping[str, float],
) -> str | None:
    """Return the name of the model that should take the next agent, or ``None``.

    ``models`` is in config order and one pass decides: a model is free when it
    has a free base slot (``assigned < sessions``) or its over-session gates
    pass. So an earlier model with measured slack wins over a later model's
    still-free base slot; ties go to the earlier entry.

    ``assigned`` counts the running agents per model name; ``recent`` holds each
    model's slack window (:func:`recent_samples`); ``history`` holds its full
    recorded samples (the throughput veto); ``last_extra`` records when each
    model last received an over-session agent. When an over-session slot is
    granted here, ``last_extra[name]`` is set to ``now``.

    An over-session slot needs the model below ``max_agents``, a metrics
    endpoint, at least :data:`MIN_SAMPLES` samples in the recent window whose mean
    ``running`` is below ``sessions - 0.5`` and whose ``waiting`` never rises
    above 0, no extra agent started on it within :data:`EXTRA_SPACING`, and no
    :func:`token_rates` veto on the full history.
    """
    for model in models:
        if assigned.get(model.name, 0) < model.sessions:
            return model.name
        if _extra_free(
            model,
            assigned.get(model.name, 0),
            recent.get(model.name, ()),
            history.get(model.name, ()),
            now,
            last_extra,
        ):
            last_extra[model.name] = now
            return model.name
    return None


def recent_samples(
    samples: Sequence[Sample], now: float, window: float = RECENT_WINDOW
) -> list[Sample]:
    """The samples at most ``window`` seconds before ``now`` (SPEC §4).

    This is the slack window; the scheduler records a sample every tick, so at
    the 20 s tick it holds the last ~9 samples. Samples from the future (a clock
    skew) are excluded.
    """
    return [sample for sample in samples if 0.0 <= now - sample.at <= window]


def token_rates(samples: Sequence[Sample]) -> dict[int, list[float]]:
    """Group generation throughput by the concurrency observed when measured.

    Each consecutive pair of samples yields one rate: growth of
    ``vllm:generation_tokens_total`` over the interval, in tokens/s, filed
    under the ``running`` value of the pair's later sample. Intervals with a
    non-positive duration and counter resets (a server restart) are skipped.
    Samples are sorted by ``at`` first, so callers may record them unordered.
    """
    ordered = sorted(samples, key=lambda sample: sample.at)
    rates: dict[int, list[float]] = {}
    for before, after in zip(ordered, ordered[1:]):
        elapsed = after.at - before.at
        tokens = after.metrics.generation_tokens - before.metrics.generation_tokens
        if elapsed <= 0 or tokens < 0:
            continue
        bucket = int(round(after.metrics.running))
        rates.setdefault(bucket, []).append(tokens / elapsed)
    return rates


def _extra_free(
    model: ModelConfig,
    assigned: int,
    recent: Sequence[Sample],
    history: Sequence[Sample],
    now: float,
    last_extra: Mapping[str, float],
) -> bool:
    """Whether ``model`` may take one agent beyond its ``sessions`` base slots.

    ``recent`` is the slack window, ``history`` the full samples for the veto.
    """
    if assigned >= model.max_agents:
        return False
    if model.metrics is None:
        return False

    if len(recent) < MIN_SAMPLES:
        return False
    if statistics.fmean(sample.metrics.running for sample in recent) >= model.sessions - 0.5:
        return False
    if max((sample.metrics.waiting for sample in recent), default=0.0) != 0:
        return False

    started = last_extra.get(model.name)
    if started is not None and now - started < EXTRA_SPACING:
        return False

    return not _throughput_veto(model, history)


def _throughput_veto(model: ModelConfig, samples: Sequence[Sample]) -> bool:
    """Whether recorded data show ``sessions + 1`` slower than ``sessions``.

    Compares the mean tokens/s of the two concurrency buckets, but only once
    each holds at least :data:`MIN_BUCKET_SAMPLES` rates; otherwise it abstains
    (returns ``False``).
    """
    buckets = token_rates(samples)
    base = buckets.get(model.sessions, ())
    extra = buckets.get(model.sessions + 1, ())
    if len(base) < MIN_BUCKET_SAMPLES or len(extra) < MIN_BUCKET_SAMPLES:
        return False
    return statistics.fmean(extra) < statistics.fmean(base)
