"""The router's HTTP surface (SPEC §11): one endpoint, one thread per request.

:mod:`taskgraph.router` decides *where* a request goes; this module speaks HTTP
to the agents and to the workers.  Requests are relayed as they stream (the
whole point of a coding agent's endpoint is that tokens arrive while the
completion is still being generated), so responses are framed with
``Connection: close`` instead of a length: the client reads until the socket
closes, which is exactly what an SSE client expects.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO

from . import gate
from .config import Config
from .router import DEFAULT_UPSTREAM_TIMEOUT, Router, error_body

#: Default listen address — the router is a local proxy, not a public server.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class _Handler(BaseHTTPRequestHandler):
    """One request per thread: the router blocks on admission, it never spins."""

    protocol_version = "HTTP/1.1"
    server_version = "taskgraph-router/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Never log headers: a client's Authorization must not land in a log.
        if self.server.log is not None:
            print(f"{self.log_date_time_string()} {fmt % args}", file=self.server.log, flush=True)

    def log_error(self, fmt: str, *args: Any) -> None:
        self.log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?")[0] == "/router/stats":
            body = json.dumps(self.server.router.stats(), indent=2).encode("utf-8") + b"\n"
            self._reply(200, [("Content-Type", "application/json")], body)
        else:
            self._reply(404, [("Content-Type", "application/json")], error_body(404, "unknown path"))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self._read_body()
        if body is None:
            return
        exchange = self.server.router.exchange(self.path, self.headers, body)
        try:
            self.send_response(exchange.status)
            for name, value in [*exchange.headers, ("Connection", "close")]:
                self.send_header(name, value)
            self.end_headers()
            self.close_connection = True
            if exchange.response is not None:
                gate.relay(exchange.response, self.wfile)
            elif exchange.body:
                self.wfile.write(exchange.body)
            self.wfile.flush()
        except OSError:
            # The agent went away or the upstream timed out mid-stream; the
            # slot is released below either way.
            pass
        finally:
            self.server.router.finish(exchange)

    def _read_body(self) -> bytes | None:
        """Read the request body, or answer an error and return ``None``."""
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            self._reply(
                501,
                [("Content-Type", "application/json")],
                error_body(501, "chunked request bodies are not supported"),
            )
            return None
        raw = self.headers.get("Content-Length")
        if raw is None:
            self._reply(
                411,
                [("Content-Type", "application/json")],
                error_body(411, "a request body is required"),
            )
            return None
        try:
            length = int(raw)
        except ValueError:
            self._reply(
                400, [("Content-Type", "application/json")], error_body(400, "bad Content-Length")
            )
            return None
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            self._reply(
                400,
                [("Content-Type", "application/json")],
                error_body(400, "truncated request body"),
            )
            return None
        return body

    def _reply(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        """A complete, locally generated reply."""
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)
        self.wfile.flush()


class RouterServer(ThreadingHTTPServer):
    """The router's HTTP server: one thread per request, no request buffering."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], router: Router, *, log: TextIO | None = None):
        self.router = router
        self.log = log
        super().__init__(address, _Handler)


def run_router(
    config: Config,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    queue_timeout: float = 0.0,
    upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
    log: TextIO | None = None,
) -> int:
    """Serve ``taskgraph router`` until SIGTERM/SIGINT; returns the exit status."""
    stream = sys.stderr if log is None else log
    if not config.workers:
        print(
            f"taskgraph router: no [[workers]] configured in {config.path}",
            file=stream,
            flush=True,
        )
        return 2
    router = Router(
        [gate.Worker(worker) for worker in config.workers],
        queue_timeout=queue_timeout,
        upstream_timeout=upstream_timeout,
        log=stream,
    )
    try:
        server = RouterServer((host, port), router, log=stream)
    except OSError as exc:
        print(f"taskgraph router: cannot listen on {host}:{port}: {exc}", file=stream, flush=True)
        return 2

    actual = server.server_address[1]
    print(
        f"taskgraph router: listening on http://{host}:{actual}"
        f" ({len(router.pool.workers)} workers)",
        file=stream,
        flush=True,
    )
    stop = threading.Event()
    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: stop.set())
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, name="router", daemon=True
    )
    thread.start()
    try:
        while not stop.wait(0.2):
            pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
