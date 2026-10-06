"""Reading a proxied OpenAI request: its body, its sender, its priority.

Pure functions plus the :class:`Exchange` record that travels from
:class:`taskgraph.router.Router` to :mod:`taskgraph.proxyserver`.  Agent
identity (the affinity key), the prompt-size estimate that picks a worker, and
the wait-queue priority all come from the request headers and body, so they
live here instead of inside the router's policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from . import gate

#: Header naming the agent behind a request (SPEC §11 "affinity").
AGENT_HEADER = "X-Taskgraph-Agent"

#: Optional header: a lower number is served sooner from the wait queue.
PRIORITY_HEADER = "X-Taskgraph-Priority"

#: Estimated prompt tokens per character of prompt text (SPEC §11).
CHARS_PER_TOKEN = 3.5


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup."""
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def parse_json(body: bytes) -> Any:
    """Parse a request body as a JSON object, or ``None`` when it is not one."""
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
    parsed = parse_json(body)
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


def reply(
    status: int,
    message: str,
    kind: str = "invalid_request_error",
    retry_after: int | None = None,
) -> Exchange:
    """A locally generated reply (no upstream involved)."""
    headers = [("Content-Type", "application/json")]
    if retry_after is not None:
        headers.append(("Retry-After", str(retry_after)))
    return Exchange(status=status, headers=headers, body=error_body(status, message, kind))
