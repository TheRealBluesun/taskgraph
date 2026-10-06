"""Admission control and streaming relay for the request router (SPEC §11).

:mod:`taskgraph.router` owns *policy* — which worker a request should go to;
this module owns the mechanics: one runtime :class:`Worker` per configured
backend, the FIFO admission queue, the per-worker request accounting behind
``/router/stats``, and a forwarder that relays an upstream response chunk by
chunk instead of buffering it (agents stream their completions, and a buffered
relay would turn every token into a stall).

A request is *admitted* while it holds one of its worker's ``concurrency``
slots; requests that found no free worker wait in a single queue, served
strictly in ``(priority, arrival)`` order, so a high-priority request is never
overtaken (SPEC §11 "FIFO queue (by task priority, then arrival)").  Only the
queue head is considered, and the candidate list is re-evaluated on every
release, because which workers are free changes as requests finish.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .config import WorkerConfig

#: Window for the busy-seconds/utilization numbers (SPEC §11: last 10 min).
STATS_WINDOW = 600.0

#: Rolling window for ``max_requests_per_hour``.
HOUR = 3600.0

#: Bytes per relay read; large enough to keep big non-streaming bodies cheap.
CHUNK = 65536

#: Headers of our own connection/framing that must not be copied to the other side.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",
        "expect",
    }
)


class GateError(Exception):
    """A request could not be relayed to its upstream."""


@dataclass(frozen=True)
class WorkerStats:
    """One worker's numbers for ``GET /router/stats`` (SPEC §11)."""

    name: str
    model: str
    upstream: str
    concurrency: int
    in_flight: int
    queued: int
    admitted: int
    busy_seconds: float
    utilization: float


class Worker:
    """Runtime state of one ``[[workers]]`` backend."""

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.name = config.name
        self.upstream = config.upstream
        self.model = config.model
        self.concurrency = config.concurrency
        self.max_context = config.max_context
        self.overflow = config.overflow
        self.api_key_env = config.api_key_env
        self.max_requests_per_hour = config.max_requests_per_hour
        self.in_flight = 0
        self.admitted = 0
        self._spans: deque[tuple[float, float]] = deque()
        self._active: list[float] = []
        self._accepted: deque[float] = deque()

    def free(self) -> bool:
        """True when the worker has a free request slot right now."""
        return self.in_flight < self.concurrency

    def fits(self, tokens: int) -> bool:
        """True when an estimated ``tokens`` prompt is within ``max_context``."""
        return self.max_context is None or tokens <= self.max_context

    def hourly_full(self, now: float) -> bool:
        """True when an overflow worker has spent its ``max_requests_per_hour``."""
        if self.max_requests_per_hour is None:
            return False
        while self._accepted and self._accepted[0] <= now - HOUR:
            self._accepted.popleft()
        return len(self._accepted) >= self.max_requests_per_hour

    def busy_seconds(self, now: float, window: float = STATS_WINDOW) -> float:
        """Seconds this worker spent serving requests within the last ``window``."""
        floor = now - window
        total = 0.0
        for started in self._active:
            total += now - max(started, floor)
        for start, end in self._spans:
            if end > floor:
                total += end - max(start, floor)
        return max(0.0, total)

    def utilization(self, now: float, window: float = STATS_WINDOW) -> float:
        """Fraction of this worker's request slots that were busy in the window."""
        busy = self.busy_seconds(now, window)
        return min(1.0, busy / (window * self.concurrency))

    # Only the pool's lock may call these: the counters are shared state.
    def _take(self, now: float) -> None:
        self.in_flight += 1
        self.admitted += 1
        self._active.append(now)
        self._accepted.append(now)

    def _release(self, started: float, now: float) -> None:
        self.in_flight -= 1
        try:
            self._active.remove(started)
        except ValueError:  # pragma: no cover - every take has one release
            pass
        self._spans.append((started, now))
        floor = now - STATS_WINDOW
        while self._spans and self._spans[0][1] <= floor:
            self._spans.popleft()


@dataclass(frozen=True)
class Ticket:
    """An admitted request: the worker holding one of its slots."""

    worker: Worker
    started: float


@dataclass(frozen=True)
class _Pending:
    priority: int
    seq: int
    candidates: Callable[[], Sequence[Worker]]

    @property
    def key(self) -> tuple[int, int]:
        return (self.priority, self.seq)


class Pool:
    """FIFO admission over a set of workers, plus the numbers for the stats page.

    ``candidates()`` is the router's policy hook: called with the pool lock
    held, it returns the workers this request may use, in preference order
    (affinity first, then config priority), excluding the ones it may not use
    at all (context too large, hourly cap reached, missing api key).  The pool
    picks the first free candidate; when none is free or the list is empty the
    request waits.
    """

    def __init__(
        self,
        workers: Sequence[Worker],
        *,
        clock: Callable[[], float] = time.monotonic,
        window: float = STATS_WINDOW,
    ) -> None:
        self.workers = list(workers)
        self.window = window
        self._clock = clock
        self._cond = threading.Condition()
        self._pending: list[_Pending] = []
        self._seq = 0

    def next_seq(self) -> int:
        """A monotonically increasing arrival number for FIFO tie-breaking."""
        with self._cond:
            self._seq += 1
            return self._seq

    def now(self) -> float:
        """The pool's clock: the ``now`` for hourly caps and stats windows."""
        return self._clock()

    def acquire(
        self,
        priority: int,
        seq: int,
        candidates: Callable[[], Sequence[Worker]],
        *,
        timeout: float | None = None,
    ) -> Ticket | None:
        """Wait for a free slot on a worker ``candidates()`` offers, or ``None``.

        ``timeout`` bounds the wait in seconds (``None`` waits for as long as
        the client does).  Requests leave the queue in ``(priority, seq)``
        order; a request whose candidates are all full blocks the ones behind
        it, which is what "by task priority, then arrival" means.
        """
        with self._cond:
            entry = _Pending(priority, seq, candidates)
            self._pending.append(entry)
            deadline = None if timeout is None else self._clock() + timeout
            try:
                while True:
                    head = min(self._pending, key=lambda pending: pending.key)
                    if head is entry:
                        for worker in candidates():
                            if worker.free():
                                now = self._clock()
                                worker._take(now)
                                self._pending.remove(entry)
                                return Ticket(worker=worker, started=now)
                    if deadline is not None:
                        remaining = deadline - self._clock()
                        if remaining <= 0:
                            return None
                        self._cond.wait(remaining)
                    else:
                        self._cond.wait()
            finally:
                if entry in self._pending:
                    self._pending.remove(entry)

    def release(self, ticket: Ticket) -> None:
        """Give back the slot ``ticket`` holds and wake the head of the queue."""
        with self._cond:
            ticket.worker._release(ticket.started, self._clock())
            self._cond.notify_all()

    def stats(self) -> list[WorkerStats]:
        """Per-worker in-flight/queued/admitted/busy-seconds/utilization."""
        now = self._clock()
        with self._cond:
            queued = {worker.name: 0 for worker in self.workers}
            for pending in self._pending:
                candidates = pending.candidates()
                if candidates:
                    queued[candidates[0].name] += 1
            return [
                WorkerStats(
                    name=worker.name,
                    model=worker.model,
                    upstream=worker.upstream,
                    concurrency=worker.concurrency,
                    in_flight=worker.in_flight,
                    queued=queued[worker.name],
                    admitted=worker.admitted,
                    busy_seconds=round(worker.busy_seconds(now, self.window), 3),
                    utilization=round(worker.utilization(now, self.window), 4),
                )
                for worker in self.workers
            ]


def upstream_url(worker: Worker, path: str) -> str:
    """The worker's upstream URL for a request path."""
    return worker.upstream + path


def forward(
    worker: Worker,
    path: str,
    body: bytes | None,
    headers: Mapping[str, str],
    *,
    timeout: float,
):
    """Send one request to ``worker`` and return the open upstream response.

    ``urllib`` raises :class:`urllib.error.HTTPError` for a 4xx/5xx upstream
    reply; that object *is* the response, so it is returned too — a worker's
    error (e.g. a context-length 400) must reach the agent unchanged.
    """
    request = urllib.request.Request(
        upstream_url(worker, path), data=body, headers=dict(headers), method="POST"
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc
    except urllib.error.URLError as exc:
        raise GateError(f"{worker.name}: upstream unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        # A socket timeout is an OSError, not a URLError: without this the
        # caller would leak the request slot on a hung upstream.
        raise GateError(f"{worker.name}: upstream timed out after {timeout:g}s") from exc


def read_chunk(response, size: int = CHUNK) -> bytes:
    """Read what the upstream has produced now, without waiting for ``size`` bytes.

    ``read(n)`` blocks until ``n`` bytes or EOF, which would hold every SSE
    chunk hostage to the next one; ``read1`` returns as soon as anything is
    available.
    """
    read1 = getattr(response, "read1", None)
    if read1 is not None:
        return read1(size)
    return response.read(size)  # pragma: no cover - http.client always has read1


def relay(response, out, size: int = CHUNK) -> int:
    """Stream ``response`` to the binary ``out`` as it arrives; bytes written."""
    total = 0
    try:
        while True:
            chunk = read_chunk(response, size)
            if not chunk:
                return total
            out.write(chunk)
            out.flush()
            total += len(chunk)
    finally:
        response.close()


def response_headers(response) -> list[tuple[str, str]]:
    """The upstream headers worth forwarding (hop-by-hop framing removed)."""
    return [
        (name, value)
        for name, value in response.headers.items()
        if name.lower() not in HOP_BY_HOP
    ]
