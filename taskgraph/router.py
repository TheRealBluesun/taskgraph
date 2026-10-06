"""The request router: one OpenAI-compatible endpoint over many workers (SPEC §11).

Work (plan tasks) and workers (model backends) are separate pools.  Every agent
points its OpenAI client at ``taskgraph router``; each generation request is
dispatched to the best worker with a free slot:

1. **affinity** — the worker that served this agent's previous request, if it
   has a free slot (a cold 60k-token prefill costs 30–60 s, so keeping the
   server's prefix cache warm beats priority).
2. else the highest-priority worker (``[[workers]]`` order) with a free slot.
3. else wait in the FIFO queue of :class:`taskgraph.gate.Pool`.

Agent identity is the ``X-Taskgraph-Agent`` header when present, else a hash of
the request's first system and first user message, so an agent that cannot set
headers still keeps its affinity.  The router rewrites the request's ``model``
field to the worker's, injects the worker's ``Authorization`` from
``api_key_env`` (never logged), skips a worker whose ``max_context`` is smaller
than the estimated prompt (chars/3.5), and keeps ``overflow`` workers for when
every local worker is full, up to their ``max_requests_per_hour``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence, TextIO

from . import gate

#: Header naming the agent behind a request (SPEC §11 "affinity").
AGENT_HEADER = "X-Taskgraph-Agent"

#: Optional header: a lower number is served sooner from the wait queue.
PRIORITY_HEADER = "X-Taskgraph-Priority"

#: Estimated prompt tokens per character of prompt text (SPEC §11).
CHARS_PER_TOKEN = 3.5

#: Default bound on the upstream socket operations (not on a whole stream).
DEFAULT_UPSTREAM_TIMEOUT = 600.0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup."""
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _json(body: bytes) -> Any:
    """Parse a request body as JSON, or ``None`` when it is not an object."""
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _text(content: Any) -> str:
    """The plain text of a message content (string or list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _text(part.get("text", "") if isinstance(part, dict) else part) for part in content
        )
    return ""


def estimate_tokens(body: Any) -> int:
    """Estimated prompt tokens of a request (SPEC §11: chars/3.5).

    Counts the text of ``messages[].content`` plus a raw ``prompt``; a
    non-chat body has no estimate.
    """
    if not isinstance(body, dict):
        return 0
    chars = sum(
        len(_text(message.get("content", "")))
        for message in body.get("messages") or ()
        if isinstance(message, dict)
    )
    chars += len(_text(body.get("prompt")))
    if chars <= 0:
        return 0
    return math.ceil(chars / CHARS_PER_TOKEN)


def _first_text(body: Any, role: str) -> str:
    """The text of the first message with ``role``, or ``""``."""
    for message in body.get("messages") or ():
        if isinstance(message, dict) and message.get("role") == role:
            return _text(message.get("content", ""))
    return ""


def identity(headers: Mapping[str, str], body: bytes) -> str:
    """Who a request is from: the agent header, else a hash of its first messages.

    The fallback keeps affinity for agents that cannot set headers: two
    requests from the same conversation (same system + first user message)
    hash alike.  A body that is not JSON hashes as bytes.
    """
    named = _header(headers, AGENT_HEADER)
    if named and named.strip():
        return named.strip()
    parsed = _json(body)
    if parsed is None:
        payload = body
    else:
        canon = json.dumps(
            [_first_text(parsed, "system"), _first_text(parsed, "user")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = canon.encode("utf-8")
    return "anon:" + hashlib.sha256(payload).hexdigest()[:16]


def priority(headers: Mapping[str, str]) -> int:
    """The wait-queue priority from ``X-Taskgraph-Priority`` (lower = sooner)."""
    raw = _header(headers, PRIORITY_HEADER)
    if raw is None:
        return 0
    try:
        return int(raw.strip())
    except ValueError:
        return 0


def error_body(status: int, message: str, kind: str = "invalid_request_error") -> bytes:
    """An OpenAI-shaped error body."""
    return json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8")


@dataclass
class Exchange:
    """One proxied request: the reply to send, and the slot to release after."""

    status: int
    headers: list[tuple[str, str]]
    response: Any | None = None
    body: bytes = b""
    ticket: gate.Ticket | None = None
    worker: gate.Worker | None = None


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

    def _candidates(
        self, affinity: str | None, tokens: int
    ) -> Callable[[], Sequence[gate.Worker]]:
        """The preference-ordered workers for one request (SPEC §11 1 + 2).

        Affinity first, then config priority: local workers in configuration
        order, then ``overflow`` workers, which therefore only see traffic
        while every local worker is full.
        """

        def candidates() -> Sequence[gate.Worker]:
            now = self.pool.now()
            usable = [w for w in self.pool.workers if self._usable(w, tokens, now)]
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

    def exchange(self, path: str, headers: Mapping[str, str], body: bytes) -> Exchange:
        """Admit a request to a worker and send it upstream.

        Blocks while every worker the request may use is full; ``finish`` must
        be called once the returned response has been relayed, to free the slot.
        """
        payload = _json(body)
        tokens = estimate_tokens(payload)
        agent = identity(headers, body)
        candidates = self._candidates(self.affinity(agent), tokens)

        if not any(worker.fits(tokens) for worker in self.pool.workers):
            return Exchange(
                status=413,
                headers=[("Content-Type", "application/json")],
                body=error_body(413, f"prompt of ~{tokens} tokens is too large for every worker"),
            )
        if not candidates():
            return Exchange(
                status=503,
                headers=[("Content-Type", "application/json"), ("Retry-After", "30")],
                body=error_body(
                    503, "no worker can serve this request right now", "server_error"
                ),
            )

        timeout = self.queue_timeout if self.queue_timeout > 0 else None
        ticket = self.pool.acquire(
            priority(headers), self.pool.next_seq(), candidates, timeout=timeout
        )
        if ticket is None:
            return Exchange(
                status=503,
                headers=[("Content-Type", "application/json"), ("Retry-After", "1")],
                body=error_body(503, "all workers are busy", "server_error"),
            )
        self._record(agent, ticket.worker)

        try:
            upstream_headers = self._forward_headers(headers, ticket.worker)
            upstream_body = self._rewrite(payload, body, ticket.worker)
            response = gate.forward(
                ticket.worker,
                path,
                upstream_body,
                upstream_headers,
                timeout=self.upstream_timeout,
            )
        except gate.GateError as exc:
            self.pool.release(ticket)
            return Exchange(
                status=502,
                headers=[("Content-Type", "application/json")],
                body=error_body(502, str(exc), "server_error"),
            )
        return Exchange(
            status=response.status,
            headers=gate.response_headers(response),
            response=response,
            ticket=ticket,
            worker=ticket.worker,
        )

    def finish(self, exchange: Exchange) -> None:
        """Release the slot an :class:`Exchange` holds (safe to call twice)."""
        if exchange.ticket is not None:
            self.pool.release(exchange.ticket)
            exchange.ticket = None

    def _forward_headers(
        self, headers: Mapping[str, str], worker: gate.Worker
    ) -> dict[str, str]:
        """Transport headers for the upstream request (client framing dropped)."""
        out = {name: value for name, value in headers.items() if name.lower() not in gate.HOP_BY_HOP}
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
