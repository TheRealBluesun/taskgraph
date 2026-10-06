"""Fake OpenAI-compatible upstreams for the router tests.

Not named ``test_*.py`` so unittest discovery does not collect it.  Nothing
here talks to a real model or the network: each upstream is a loopback HTTP
server that records what it was sent, can hold a response open (so a test can
prove per-worker concurrency), and can stream SSE chunks only when the test
releases them one by one (so a test can prove the relay does not buffer).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeUpstream:
    """A tiny loopback OpenAI server that records every request it receives."""

    def __init__(self, name: str = "up", *, chunks: int = 0, status: int = 200):
        self.name = name
        self.chunks = chunks
        self.status = status
        self.requests: list[dict] = []
        self.answers: list[str] = []
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.chunk_gates = [threading.Event() for _ in range(chunks)]
        self._release = threading.Event()
        self._release.set()
        handler = type(
            f"Handler_{name}",
            (_Handler,),
            {"upstream": self},
        )
        self.server = _Server(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    # ------------------------------------------------------------------ life

    def start(self) -> "FakeUpstream":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------- behaviour

    def pause(self) -> None:
        """Hold every future response until :meth:`resume`."""
        self._release.clear()

    def resume(self) -> None:
        self._release.set()

    def release_chunk(self, index: int) -> None:
        """Let chunk ``index`` (and every earlier one) be written."""
        for gate in self.chunk_gates[: index + 1]:
            gate.set()

    def set_chunks(self, count: int) -> None:
        """Stream ``count`` gated SSE chunks instead of a single JSON reply."""
        self.chunks = count
        self.chunk_gates = [threading.Event() for _ in range(count)]

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.requests)

    def seen(self) -> list[tuple[str, str | None]]:
        """``(path, model)`` for every request received, in arrival order."""
        return [(request["path"], (request["body"] or {}).get("model")) for request in self.snapshot()]

    # -------------------------------------------------- called from the handler

    def begin(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def end(self) -> None:
        with self.lock:
            self.active -= 1

    def record(self, path: str, headers: dict, body: dict | None) -> None:
        with self.lock:
            self.requests.append({"path": path, "headers": dict(headers), "body": body})

    def answer(self, model: str, authorization: str | None) -> bytes:
        payload = {
            "id": f"cmpl-{self.name}",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }
        with self.lock:
            self.answers.append(model)
        return json.dumps(payload).encode("utf-8")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the test output clean
        pass

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        upstream = self.upstream
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        upstream.record(self.path, dict(self.headers), body)
        upstream.begin()
        try:
            upstream._release.wait(timeout=30)
            if body is not None and body.get("stream") and upstream.chunks:
                self._stream(upstream, body)
                return
            payload = upstream.answer(
                (body or {}).get("model", ""), self.headers.get("Authorization")
            )
            self.send_response(upstream.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        finally:
            upstream.end()

    def _stream(self, upstream: FakeUpstream, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True
        for index in range(upstream.chunks):
            upstream.chunk_gates[index].wait(timeout=30)
            frame = {
                "id": f"cmpl-{upstream.name}",
                "object": "chat.completion.chunk",
                "model": body.get("model", ""),
                "choices": [{"index": 0, "delta": {"content": f"part{index}"}}],
            }
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
