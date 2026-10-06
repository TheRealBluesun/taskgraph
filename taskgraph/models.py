"""Throughput per concurrency bucket: ``taskgraph models`` (SPEC §4).

More concurrency is not always more throughput.  On the 27B server (speculative
decoding + pipeline parallel over two GPUs) total generation *fell* as
concurrency rose — 1 request 81 tok/s, 2 → 69, 3 → 49 — so ``[[models]] sessions``
must be the concurrency where total throughput peaks, not the server's request
limit.  The scheduler records one ``vllm:generation_tokens_total`` sample per
tick and :func:`taskgraph.pool.token_rates` turns consecutive samples into a
tokens/s rate filed under the concurrency observed when it was measured; this
module averages those rates per bucket and marks the peak, which is exactly the
number the operator should put in ``sessions``.

:func:`collect` reads the **full** recorded history from ``state.json`` (not the
pool policy's recent slack window) and :func:`render` is pure text.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import assign, pool, status, state
from .config import Config


@dataclass(frozen=True)
class RateBucket:
    """Mean throughput observed at one concurrency."""

    running: int
    tokens_per_sec: float
    samples: int
    peak: bool = False


@dataclass(frozen=True)
class ModelRates:
    """One configured model's throughput buckets, lowest concurrency first."""

    name: str
    buckets: tuple[RateBucket, ...] = ()


@dataclass(frozen=True)
class ModelTable:
    """Everything ``taskgraph models`` prints."""

    models: tuple[ModelRates, ...] = ()


def rates(samples: Sequence[pool.Sample]) -> tuple[RateBucket, ...]:
    """Average tokens/s per concurrency bucket and mark the peak (SPEC §4).

    A tie goes to the lowest concurrency: it reaches the same throughput with
    fewer concurrent requests, so it is the safer ``sessions``.
    """
    grouped = pool.token_rates(samples)
    if not grouped:
        return ()
    means = {running: statistics.fmean(values) for running, values in grouped.items()}
    best = max(means.values())
    peak = min(running for running, mean in means.items() if mean == best)
    return tuple(
        RateBucket(running, means[running], len(grouped[running]), running == peak)
        for running in sorted(grouped)
    )


def collect(cfg: Config, *, state_root: Path | str | None = None) -> ModelTable:
    """Build the table for every configured model from ``state.json`` (SPEC §4).

    Raises :class:`taskgraph.state.StateError` for a malformed state file, like
    ``taskgraph status``; the caller reports it instead of showing an empty table.
    """
    saved = state.load(state.state_path(cfg.root, state_root))
    history = assign.history_from_state(saved.samples)
    return ModelTable(
        tuple(ModelRates(model.name, rates(history.get(model.name, ()))) for model in cfg.models)
    )


def render(table: ModelTable) -> str:
    """Render the per-model buckets as an aligned table (SPEC §4).

    A model with no rate yet (fewer than two samples) still gets a row, so a
    silent endpoint is visible rather than absent.
    """
    rows: list[tuple[str, ...]] = []
    for model in table.models:
        if not model.buckets:
            rows.append((model.name, "-", "-", "0", ""))
            continue
        rows.extend(
            (
                model.name,
                str(bucket.running),
                f"{bucket.tokens_per_sec:.1f}",
                str(bucket.samples),
                "*" if bucket.peak else "",
            )
            for bucket in model.buckets
        )
    if not rows:
        return "no models configured"
    return "\n".join(status.table(("MODEL", "RUNNING", "TOKENS/S", "SAMPLES", "PEAK"), rows))
