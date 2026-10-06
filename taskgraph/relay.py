"""Upstream transport for the request router: forward, stream, classify.

:mod:`taskgraph.gate` owns *admission* (which request runs on which worker
slot); this module owns the conversation with a worker once a request has a
slot: building the upstream URL, forwarding the request, relaying the response
chunk by chunk, and classifying how the relay ended.  Agents stream their
completions, and a buffered relay would turn every token into a stall.

Request paths are validated (:func:`path_ok`) and redirects are never followed:
the router injects a worker's ``Authorization`` header, and neither an odd path
nor a redirect may carry it to another host.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Mapping
from urllib.parse import urlsplit

from .gate import DONE, ERROR, GateError, Worker

#: Bytes per relay read; large enough to keep big non-streaming bodies cheap.
CHUNK = 65536

#: Request paths the router forwards: only OpenAI-style API paths.
PATH_PREFIX = "/v1/"

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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a followed redirect could carry the api key elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


#: One shared opener for every worker request.  Built without the default
#: redirect handler, so a 3xx from an upstream is returned to the agent as-is
#: instead of being followed (possibly to another host, with our Authorization
#: header attached).
_OPENER = urllib.request.build_opener(_NoRedirect)


def path_ok(path: str) -> bool:
    """True when ``path`` is a safe ``/v1/`` request path.

    The upstream URL is built by concatenating ``worker.upstream`` and the
    path, so a path that is not an absolute ``/v1/`` path — ``//host``, a
    control character, a ``#`` fragment, a backslash — could rewrite the
    request target and carry an injected api key to another host.  Only
    ``/v1/`` paths are proxied.
    """
    if not path.startswith(PATH_PREFIX):
        return False
    rest = path[len(PATH_PREFIX) :]
    if not rest or rest[0] in "/?#":
        return False
    if any(ord(char) <= 0x20 or char in "#\\" for char in path):
        return False
    # ``..`` segments would let a client reach anything else on the worker.
    return ".." not in rest.split("?", 1)[0].split("/")


def upstream_url(worker: Worker, path: str) -> str:
    """The worker's upstream URL for a request path.

    ``path`` must be a ``/v1/`` path (:func:`path_ok`); the netloc check is the
    last line of defence so no caller can turn the concatenation into a request
    to a different host.
    """
    if not path_ok(path):
        raise GateError(f"invalid request path: {path!r}")
    url = worker.upstream + path
    if urlsplit(url).netloc != urlsplit(worker.upstream).netloc:
        raise GateError(f"request path escapes {worker.upstream!r}: {path!r}")
    return url


def forward(
    worker: Worker,
    path: str,
    body: bytes | None,
    headers: Mapping[str, str],
    *,
    timeout: float,
    method: str = "POST",
):
    """Send one request to ``worker`` and return the open upstream response.

    ``urllib`` raises :class:`urllib.error.HTTPError` for a 4xx/5xx upstream
    reply; that object *is* the response, so it is returned too — a worker's
    error (e.g. a context-length 400) must reach the agent unchanged.  A 3xx is
    returned the same way: redirects are never followed.
    """
    request = urllib.request.Request(
        upstream_url(worker, path), data=body, headers=dict(headers), method=method
    )
    try:
        return _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc
    except urllib.error.URLError as exc:
        raise GateError(f"{worker.name}: upstream unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        # A socket timeout is an OSError, not a URLError: without this the
        # caller would leak the request slot on a hung upstream.
        raise GateError(f"{worker.name}: upstream timed out after {timeout:g}s") from exc


def classify(completed: bool, error: BaseException | None = None) -> str:
    """The outcome of one relayed request: ``DONE`` or ``ERROR``.

    A client that hangs up *after* the whole reply was written leaves us a dead
    socket, not a failed request — counting those as errors (a live gate logged
    309 of them) hides the real ones.
    """
    if error is None or (completed and isinstance(error, OSError)):
        return DONE
    return ERROR


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
