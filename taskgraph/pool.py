"""Assign running agents to models (SPEC §4). Pure policy: the clock, the
recorded metrics and the previous over-session grants are all passed in, so
every branch is unit-testable without a server.

Agents first fill each model's *base* capacity (``sessions`` concurrent
agents) in config order. Only when no model has a free base slot does
:func:`choose_model` consider an over-session slot: a model already at
``sessions`` may take one more agent, up to ``max_agents``, but only while its
own vLLM counters show real slack. Agents spend roughly half their time blocked
on tools (simulator, builds), so more agents than ``sessions`` usually raises
GPU utilisation — until generation throughput starts to fall (speculative
decoding + pipeline parallel), which :func:`token_rates` records and the
throughput veto enforces.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from .config import ModelConfig
from .metrics import Metrics

__all__ = [
    "EXTRA_SPACING",
    "MIN_BUCKET_SAMPLES",
    "MIN_SAMPLES",
    "Sample",
    "choose_model",
    "token_rates",
]

# The slack estimate needs a few minutes of history before it is trusted:
# 9 samples is ~3 min at the 20 s tick (SPEC §4).
MIN_SAMPLES = 9

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


def choose_model(
    models: Sequence[ModelConfig],
    assigned: Mapping[str, int],
    load: Mapping[str, Sequence[Sample]],
    now: float,
    last_extra: MutableMapping[str, float],
) -> str | None:
    """Return the name of the model that should take the next agent, or ``None``.

    ``models`` is in config order (ties go to the earlier entry); ``assigned``
    counts the running agents per model name; ``load`` holds each model's
    recorded samples; ``last_extra`` records when each model last received an
    over-session agent. When an over-session slot is granted here,
    ``last_extra[name]`` is set to ``now``.

    A model is free when it has a free base slot. Otherwise it is free for one
    extra agent when it is below ``max_agents``, has a metrics endpoint and at
    least :data:`MIN_SAMPLES` samples whose mean ``running`` is below
    ``sessions - 0.5`` and whose ``waiting`` never rises above 0, no extra agent
    started on it within :data:`EXTRA_SPACING`, and :func:`token_rates` does not
    show the ``sessions + 1`` bucket outperforming ``sessions``.
    """
    for model in models:
        if assigned.get(model.name, 0) < model.sessions:
            return model.name

    for model in models:
        if _extra_free(model, assigned, load, now, last_extra):
            last_extra[model.name] = now
            return model.name
    return None


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
    assigned: Mapping[str, int],
    load: Mapping[str, Sequence[Sample]],
    now: float,
    last_extra: Mapping[str, float],
) -> bool:
    """Whether ``model`` may take one agent beyond its ``sessions`` base slots."""
    if assigned.get(model.name, 0) >= model.max_agents:
        return False
    if model.metrics is None:
        return False

    samples = load.get(model.name, ())
    if len(samples) < MIN_SAMPLES:
        return False
    if statistics.fmean(sample.metrics.running for sample in samples) >= model.sessions - 0.5:
        return False
    if max((sample.metrics.waiting for sample in samples), default=0.0) != 0:
        return False

    started = last_extra.get(model.name)
    if started is not None and now - started < EXTRA_SPACING:
        return False

    return not _throughput_veto(model, samples)


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
