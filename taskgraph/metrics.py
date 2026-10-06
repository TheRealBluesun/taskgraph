"""Parse vLLM's Prometheus metrics (SPEC §4).

vLLM exports one series per (model, engine) label set, so every counter we care
about is the sum over its series. :func:`parse` is pure text handling and is
what tests exercise; :func:`sample` is the single HTTP call, returning ``None``
for any network/transport error so a scheduler tick can simply skip the sample.
"""

from __future__ import annotations

import http.client
import math
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ["Metrics", "parse", "sample"]

# Exact metric names: series like ``vllm:generation_tokens_total_created`` must
# not be summed into the counter (Prometheus appends ``_created``).
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
GENERATION_TOKENS = "vllm:generation_tokens_total"

@dataclass(frozen=True)
class Metrics:
    """One Prometheus scrape: gauge sums plus the generation-token counter."""

    running: float = 0.0
    waiting: float = 0.0
    generation_tokens: float = 0.0


def parse(text: str) -> Metrics:
    """Sum the vLLM series in ``text`` (Prometheus text exposition format).

    Missing metrics are reported as ``0.0``. Non-finite values (``NaN``/``Inf``)
    and unparseable lines are skipped rather than poisoning the sums.
    """
    totals = {RUNNING: 0.0, WAITING: 0.0, GENERATION_TOKENS: 0.0}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = _split_sample(line)
        if name in totals and value is not None and math.isfinite(value):
            totals[name] += value
    return Metrics(totals[RUNNING], totals[WAITING], totals[GENERATION_TOKENS])


def sample(url: str, timeout: float = 4) -> Metrics | None:
    """Fetch ``url`` and parse it; ``None`` on any network/transport error.

    A skipped sample is normal (§4): a model whose endpoint is briefly down is
    simply treated as having no metrics for that tick.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (OSError, ValueError, http.client.HTTPException):
        return None
    return parse(text)


def _split_sample(line: str) -> tuple[str, float | None]:
    """Return ``(metric_name, value)`` for one exposition line.

    ``value`` is ``None`` when the line is not a ``name{labels} number`` sample.
    """
    if "{" in line:
        name, _, rest = line.partition("{")
        _, _, rest = rest.partition("}")
    else:
        name, _, rest = line.partition(" ")
    fields = rest.split()
    if not fields:
        return name, None
    try:
        return name, float(fields[0])
    except ValueError:
        return name, None
