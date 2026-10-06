"""The request router: one OpenAI-compatible endpoint over many workers (SPEC §11).

Work (plan tasks) and workers (model backends) are separate pools.  Every agent
points its OpenAI client at ``taskgraph router``; each generation request is
dispatched to the best worker with a free slot:

1. **affinity** — the worker that served this agent's previous request, if it
   has a free slot (a cold 60k-token prefill costs 30–60 s, so keeping the
   server's prefix cache warm beats priority).
2. else the highest-priority worker (``[[workers]]`` order) with a free slot.
3. else wait in the queue of :class:`taskgraph.gate.Pool`.

Agent identity (the ``X-Taskgraph-Agent`` header, else a hash of the request's
first system and first user message), the prompt-size estimate and the
wait-queue priority come from :mod:`taskgraph.request`.  The router rewrites the
request's ``model`` field to the worker's, injects the worker's ``Authorization``
from ``api_key_env`` (never logged), skips a worker whose ``max_context`` is
smaller than the estimate or whose api key is unset, keeps ``overflow`` workers
for when every local worker is full (up to ``max_requests_per_hour``), and fails
over to the next candidate when an upstream is unreachable.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence, TextIO

from . import gate, relay
from .request import Exchange, estimate_tokens, identity, parse_json, priority, reply

#: Default bound on the upstream socket operations (not on a whole stream).
DEFAULT_UPSTREAM_TIMEOUT = 600.0


class Router:
    """Policy on top of :class:`taskgraph.gate.Pool` (SPEC §11)."""

    def __init__(
        self,
        workers: Sequence[gate.Worker],
        *,
        queue_timeout: float = 0.0,
        upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
        log: TextIO | None = None,
    ) -> None:
        self.pool = gate.Pool(workers)
        self.queue_timeout = queue_timeout
        self.upstream_timeout = upstream_timeout
        self.log = sys.stderr if log is None else log
        self._lock = threading.Lock()
        self._last: dict[str, str] = {}
        self._switches: dict[str, int] = {}
        self._hits = 0
        self._opportunities = 0
        self._warned: set[str] = set()

    # ------------------------------------------------------------- policy

    def api_key(self, worker: gate.Worker) -> str | None:
        """The ``Authorization`` value for ``worker``, or ``None`` to pass the client's.

        A worker whose ``api_key_env`` is unset in this process is not a
        candidate: it could only answer 401.  The value itself is never logged.
        """
        if worker.api_key_env is None:
            return None
        value = os.environ.get(worker.api_key_env)
        if not value:
            with self._lock:
                first = worker.name not in self._warned
                self._warned.add(worker.name)
            if first:
                print(
                    f"taskgraph router: worker {worker.name}: ${worker.api_key_env} is unset;"
                    " skipping it",
                    file=self.log,
                    flush=True,
                )
            return None
        return f"Bearer {value}"

    def _usable(self, worker: gate.Worker, tokens: int, now: float) -> bool:
        """True when this request may use ``worker`` at all right now."""
        if not worker.fits(tokens) or worker.hourly_full(now):
            return False
        if worker.api_key_env is not None and not os.environ.get(worker.api_key_env):
            self.api_key(worker)  # logs the missing-key warning once
            return False
        return True

    def _serveable(self, tokens: int) -> bool:
        """True when some worker could *ever* serve this request (SPEC §11).

        A missing api key or a too-small ``max_context`` makes a worker useless
        for this request forever, so the request is answered 503 at once
        instead of queueing behind a slot that will never come.  An hourly cap
        only makes it wait: the cap rolls over.
        """
        for worker in self.pool.workers:
            if worker.api_key_env is None or os.environ.get(worker.api_key_env):
                if worker.fits(tokens):
                    return True
            else:
                self.api_key(worker)  # log the missing-key warning once
        return False

    def _candidates(
        self, affinity: str | None, tokens: int, tried: set[str]
    ) -> Callable[[], Sequence[gate.Worker]]:
        """The preference-ordered workers for one request (SPEC §11 1 + 2).

        Affinity first, then config priority: local workers in configuration
        order, then ``overflow`` workers, which therefore only see traffic
        while every local worker is full.  Workers already ``tried`` (an
        unreachable upstream) are skipped by the failover loop.
        """

        def candidates() -> Sequence[gate.Worker]:
            now = self.pool.now()
            usable = [
                w
                for w in self.pool.workers
                if w.name not in tried and self._usable(w, tokens, now)
            ]
            ordered = [w for w in usable if w.name == affinity]
            ordered += [w for w in usable if not w.overflow and w.name != affinity]
            ordered += [w for w in usable if w.overflow and w.name != affinity]
            return ordered

        return candidates

    def affinity(self, agent: str) -> str | None:
        """The worker that served ``agent``'s previous request, if any."""
        with self._lock:
            return self._last.get(agent)

    def _record(self, agent: str, worker: gate.Worker) -> None:
        """Update affinity hits and per-agent worker switches for ``/router/stats``."""
        with self._lock:
            previous = self._last.get(agent)
            if previous is not None:
                self._opportunities += 1
                if previous == worker.name:
                    self._hits += 1
                else:
                    self._switches[agent] = self._switches.get(agent, 0) + 1
            self._last[agent] = worker.name

    # ------------------------------------------------------------ requests

    def exchange(
        self,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        cancel: Callable[[], bool] | None = None,
    ) -> Exchange:
        """Admit a request to a worker and send it upstream.

        Blocks while every worker the request may use is full; ``finish`` must
        be called once the returned response has been relayed, to free the slot.
        ``cancel`` is consulted while the request waits, so a client that hung
        up gives up its queue place.  An unreachable upstream fails over to the
        next candidate; the slot is released on every failure path.
        """
        if not relay.path_ok(path):
            return reply(400, f"invalid request path: {path!r}")
        payload = parse_json(body)
        if payload is None or not isinstance(payload.get("messages"), list):
            return reply(400, "request body must be a JSON object with a 'messages' list")
        tokens = estimate_tokens(payload)

        if not any(worker.fits(tokens) for worker in self.pool.workers):
            return reply(413, f"prompt of ~{tokens} tokens is too large for every worker")
        if not self._serveable(tokens):
            return reply(
                503, "no worker can serve this request right now", "server_error", retry_after=30
            )

        agent = identity(headers, body)
        tried: set[str] = set()
        candidates = self._candidates(self.affinity(agent), tokens, tried)
        seq = self.pool.next_seq()
        priority_ = priority(headers)
        timeout = self.queue_timeout if self.queue_timeout > 0 else None
        last: gate.GateError | None = None

        while len(tried) < len(self.pool.workers):
            ticket = self.pool.acquire(
                priority_, seq, candidates, timeout=timeout, cancel=cancel
            )
            if ticket is None:
                return reply(503, "all workers are busy", "server_error", retry_after=1)
            keep = False
            try:
                upstream_headers = self._forward_headers(headers, ticket.worker)
                upstream_body = self._rewrite(payload, body, ticket.worker)
                response = relay.forward(
                    ticket.worker,
                    path,
                    upstream_body,
                    upstream_headers,
                    timeout=self.upstream_timeout,
                )
                # Affinity is recorded only once the worker accepted the
                # request: a worker we never reached must not attract the
                # agent's next request.
                self._record(agent, ticket.worker)
                keep = True
                return Exchange(
                    status=response.status,
                    headers=relay.response_headers(response),
                    response=response,
                    ticket=ticket,
                    worker=ticket.worker,
                )
            except gate.GateError as exc:
                tried.add(ticket.worker.name)  # unreachable: try the next candidate
                last = exc
            except Exception as exc:
                return reply(502, f"{ticket.worker.name}: {exc}", "server_error")
            finally:
                if not keep:
                    self.pool.release(ticket)
        return reply(
            502,
            str(last) if last is not None else "no worker accepted the request",
            "server_error",
        )

    def passthrough(self, path: str, headers: Mapping[str, str]) -> Exchange:
        """Forward a non-generation request (``GET /v1/models``) to a worker.

        SPEC §11's capacity protects *generation* requests, so a listing call
        takes no slot.  The highest-priority worker (config order) serves it,
        failing over to the next when an upstream is unreachable.
        """
        if not relay.path_ok(path):
            return reply(400, f"invalid request path: {path!r}")
        for worker in self.pool.workers:
            if worker.api_key_env is not None and not os.environ.get(worker.api_key_env):
                self.api_key(worker)  # logs the missing-key warning once
                continue
            try:
                response = relay.forward(
                    worker,
                    path,
                    None,
                    self._forward_headers(headers, worker),
                    timeout=self.upstream_timeout,
                    method="GET",
                )
            except gate.GateError:
                continue
            except Exception as exc:
                return reply(502, f"{worker.name}: {exc}", "server_error")
            return Exchange(
                status=response.status,
                headers=relay.response_headers(response),
                response=response,
                worker=worker,
            )
        return reply(503, "no worker can serve this request right now", "server_error")

    def finish(self, exchange: Exchange, outcome: str = gate.DONE) -> None:
        """Release the slot an :class:`Exchange` holds and count its outcome.

        Safe to call twice (the second call counts nothing new).
        """
        if exchange.ticket is not None:
            self.pool.release(exchange.ticket)
            exchange.ticket = None
        if exchange.worker is not None:
            worker, exchange.worker = exchange.worker, None
            self.pool.record_outcome(worker, outcome)

    def _forward_headers(
        self, headers: Mapping[str, str], worker: gate.Worker
    ) -> dict[str, str]:
        """Transport headers for the upstream request (client framing dropped)."""
        out = {name: value for name, value in headers.items() if name.lower() not in relay.HOP_BY_HOP}
        key = self.api_key(worker)
        if key is not None:
            out["Authorization"] = key
        return out

    def _rewrite(self, payload: Any, body: bytes, worker: gate.Worker) -> bytes:
        """Rewrite the request's ``model`` field to the worker's model (SPEC §11)."""
        if payload is None or "model" not in payload:
            return body
        payload["model"] = worker.model
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # --------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        """The ``GET /router/stats`` document (SPEC §11)."""
        with self._lock:
            agents = [
                {"agent": agent, "worker": worker, "switches": self._switches.get(agent, 0)}
                for agent, worker in sorted(self._last.items())
            ]
            hits, opportunities = self._hits, self._opportunities
        return {
            "workers": [asdict(worker) for worker in self.pool.stats()],
            "affinity": {
                "hits": hits,
                "opportunities": opportunities,
                "rate": round(hits / opportunities, 4) if opportunities else None,
            },
            "agents": agents,
            "window_seconds": self.pool.window,
        }
