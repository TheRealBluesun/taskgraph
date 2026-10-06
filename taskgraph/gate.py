"""Admission control for the request router (SPEC §11).

:mod:`taskgraph.router` owns *policy* — which worker a request should go to;
:mod:`taskgraph.relay` owns the conversation with a worker once a request has a
slot.  This module owns the queue between the two: one runtime :class:`Worker`
per configured backend, admission in ``(priority, arrival)`` order, and the
per-worker accounting behind ``/router/stats``.

A request is *admitted* while it holds one of its worker's ``concurrency``
slots; requests that found no free worker wait in a single queue, served in
``(priority, arrival)`` order, so a high-priority request is never overtaken
(SPEC §11 "FIFO queue (by task priority, then arrival)").  The first waiting
request that *can* start is admitted, not strictly the head: a request whose
candidate workers are all busy (or all unusable) must not deadlock the ones
behind it.  Candidate lists are re-evaluated on every release *and* on a poll
timer, because which workers are usable changes as requests finish and as an
``overflow`` worker's hourly cap rolls over.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import WorkerConfig

#: Window for the busy-seconds/utilization numbers (SPEC §11: last 10 min).
STATS_WINDOW = 600.0

#: Rolling window for ``max_requests_per_hour``.
HOUR = 3600.0

#: How long a queued request sleeps before re-checking its candidates; bounds
#: how late an ``overflow`` worker whose hourly cap rolled over is picked up.
ADMIT_POLL = 1.0

#: Listen backlog for the router's HTTP server.  A full backlog makes a
#: connecting agent wait in the kernel (or see a refused connection), so it is
#: far larger than the maximum number of request slots.
REQUEST_QUEUE_SIZE = 128

#: Relay outcomes counted per worker (``/router/stats``); see ``relay.classify``.
DONE = "done"
ERROR = "error"


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
    done: int
    errors: int
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
        self.done = 0
        self.errors = 0
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


def _never() -> bool:
    """Default ``cancel`` for a request that cannot be withdrawn."""
    return False


@dataclass
class _Pending:
    priority: int
    seq: int
    candidates: Callable[[], Sequence[Worker]]
    cancel: Callable[[], bool] = _never
    #: Set by :meth:`Pool._admit_one` when this request is granted a slot; the
    #: waiting thread picks it up from its own entry (it is woken by the grant).
    ticket: Ticket | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.priority, self.seq)


class Pool:
    """FIFO admission over a set of workers, plus the numbers for the stats page.

    ``candidates()`` is the router's policy hook: called with the pool lock
    held, it returns the workers this request may use, in preference order
    (affinity first, then config priority), excluding the ones it may not use
    at all (context too large, hourly cap reached, missing api key).  The pool
    picks the first waiting request (by priority, then arrival) that has a free
    candidate; a request with nothing free waits without blocking the ones
    behind it, and is dropped when its ``cancel`` says the client is gone.
    """

    def __init__(
        self,
        workers: Sequence[Worker],
        *,
        clock: Callable[[], float] = time.monotonic,
        window: float = STATS_WINDOW,
        poll: float = ADMIT_POLL,
    ) -> None:
        self.workers = list(workers)
        self.window = window
        self.poll = poll
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
        cancel: Callable[[], bool] | None = None,
    ) -> Ticket | None:
        """Wait for a free slot on a worker ``candidates()`` offers, or ``None``.

        ``timeout`` bounds the wait in seconds (``None`` waits for as long as
        the client does).  Requests leave the queue in ``(priority, seq)``
        order, but the queue is *scanned*: the first waiting request that can
        start is admitted, so one whose candidates are all busy cannot deadlock
        the ones behind it.  The scan runs on every release and once per
        ``poll``, so a candidate that becomes usable with time (an ``overflow``
        worker whose hourly cap rolled over) is picked up without a wakeup.
        ``cancel`` is consulted before admitting: a request whose client hung
        up gives up its place instead of holding it.
        """
        entry = _Pending(priority, seq, candidates, cancel or _never)
        with self._cond:
            self._pending.append(entry)
            deadline = None if timeout is None else self._clock() + timeout
            try:
                while True:
                    # ``_admit_one`` may grant the slot to *any* waiting request
                    # (that is what keeps an unservable request from blocking
                    # the queue), so the grant is picked up from our own entry.
                    self._admit_one()
                    if entry.ticket is not None:
                        return entry.ticket
                    if entry.cancel() or not self._queued(entry):
                        return None
                    if deadline is None:
                        self._cond.wait(self.poll)
                    else:
                        remaining = deadline - self._clock()
                        if remaining <= 0:
                            return None
                        self._cond.wait(min(remaining, self.poll))
            finally:
                self._forget(entry)

    def _admit_one(self) -> bool:
        """Grant a free slot to the first waiting request that can start.

        Scanning the whole queue (by priority, then arrival) is what keeps a
        request whose candidates are all busy from deadlocking the ones behind
        it.  A request whose client hung up is dropped instead of admitted.
        """
        for entry in sorted(self._pending, key=lambda pending: pending.key):
            if entry.cancel():
                self._forget(entry)
                continue
            for worker in entry.candidates():
                if worker.free():
                    now = self._clock()
                    worker._take(now)
                    entry.ticket = Ticket(worker=worker, started=now)
                    self._forget(entry)
                    return True
        return False

    def waiting(self) -> int:
        """How many requests are queued right now.

        Counts every queued request, including one whose candidates are all
        unusable at the moment (an ``overflow`` worker over its hourly cap) and
        which therefore has no worker to be attributed to in :meth:`stats`.
        """
        with self._cond:
            return len(self._pending)

    def _queued(self, entry: _Pending) -> bool:
        """True while ``entry`` is still waiting (compared by identity)."""
        return any(pending is entry for pending in self._pending)

    def _forget(self, entry: _Pending) -> None:
        """Drop ``entry`` from the queue and wake the waiters behind it."""
        for index, pending in enumerate(self._pending):
            if pending is entry:
                del self._pending[index]
                self._cond.notify_all()
                return

    def release(self, ticket: Ticket) -> None:
        """Give back the slot ``ticket`` holds and re-scan the queue."""
        with self._cond:
            ticket.worker._release(ticket.started, self._clock())
            self._cond.notify_all()

    def record_outcome(self, worker: Worker, outcome: str) -> None:
        """Count one relay outcome for ``/router/stats`` (``DONE``/``ERROR``)."""
        with self._cond:
            if outcome == DONE:
                worker.done += 1
            else:
                worker.errors += 1

    def stats(self) -> list[WorkerStats]:
        """Per-worker in-flight/queued/admitted/done/errors/busy/utilization."""
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
                    done=worker.done,
                    errors=worker.errors,
                    busy_seconds=round(worker.busy_seconds(now, self.window), 3),
                    utilization=round(worker.utilization(now, self.window), 4),
                )
                for worker in self.workers
            ]
