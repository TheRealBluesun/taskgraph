"""Tests for the request router (SPEC §11).

Three fake upstreams is the house style here: two locals and one paid
``overflow`` worker.  Nothing leaves loopback.
"""

import http.client
import io
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
import unittest
from pathlib import Path

from leasehelpers import BIN, wait_until
from upstreams import FakeUpstream

from taskgraph import gate, proxyserver, router
from taskgraph.config import WorkerConfig

PATH = "/v1/chat/completions"
CHAT = {"model": "client-model", "messages": [{"role": "user", "content": "hi"}]}


def worker(name, upstream, model=None, **kw):
    return gate.Worker(WorkerConfig(name=name, upstream=upstream, model=model or name, **kw))


class PureTest(unittest.TestCase):
    """Token estimation, agent identity and queue priority."""

    def test_estimate_tokens_is_chars_over_three_point_five(self):
        body = {"messages": [{"role": "user", "content": "a" * 35}]}
        self.assertEqual(router.estimate_tokens(body), 10)
        parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a" * 7}]}]}
        self.assertEqual(router.estimate_tokens(parts), 2)
        self.assertEqual(router.estimate_tokens({"prompt": "a" * 70}), 20)
        self.assertEqual(router.estimate_tokens({}), 0)
        self.assertEqual(router.estimate_tokens(None), 0)

    def test_identity_prefers_the_agent_header(self):
        headers = {router.AGENT_HEADER.lower(): "  T01  "}
        self.assertEqual(router.identity(headers, b'{"messages": []}'), "T01")

    def test_identity_hashes_the_first_system_and_user_messages(self):
        first = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
        same = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                             {"role": "assistant", "content": "later"}]}
        other = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u2"}]}
        one = router.identity({}, json.dumps(first).encode())
        self.assertTrue(one.startswith("anon:"))
        self.assertEqual(one, router.identity({}, json.dumps(same).encode()))
        self.assertNotEqual(one, router.identity({}, json.dumps(other).encode()))
        self.assertNotEqual(one, router.identity({}, b"not json"))
        self.assertEqual(router.identity({}, b"not json"), router.identity({}, b"not json"))

    def test_priority_header(self):
        self.assertEqual(router.priority({}), 0)
        self.assertEqual(router.priority({router.PRIORITY_HEADER: " -3 "}), -3)
        self.assertEqual(router.priority({router.PRIORITY_HEADER: "later"}), 0)


class DirectTest(unittest.TestCase):
    """Router policy without the HTTP layer (``exchange`` in-process)."""

    def setUp(self):
        self.upstreams: list[FakeUpstream] = []

    def upstream(self, name="up", **kw) -> FakeUpstream:
        upstream = FakeUpstream(name, **kw).start()
        self.upstreams.append(upstream)
        self.addCleanup(upstream.stop)
        return upstream

    def make_router(self, workers, **kw) -> router.Router:
        kw.setdefault("upstream_timeout", 20)
        self.router = router.Router(workers, **kw)
        return self.router

    def send(self, payload=None, *, agent="A", headers=None, body=None, expect=None):
        """One request through ``exchange``; returns ``(status, body, headers)``."""
        raw = body if body is not None else json.dumps(payload if payload is not None else CHAT).encode()
        sent = {"Content-Type": "application/json"}
        if agent:
            sent[router.AGENT_HEADER] = agent
        sent.update(headers or {})
        exchange = self.router.exchange(PATH, sent, raw)
        try:
            if expect is not None:
                self.assertEqual(exchange.status, expect, exchange.body)
            out = io.BytesIO()
            if exchange.response is not None:
                gate.relay(exchange.response, out)
            return exchange.status, exchange.body + out.getvalue(), dict(exchange.headers)
        finally:
            self.router.finish(exchange)

    def hold(self, agent, *, path=PATH):
        """Start a request that blocks in a paused upstream; returns (thread, exchanges)."""
        exchanges: list[router.Exchange] = []
        payload = json.dumps(CHAT).encode()
        headers = {"Content-Type": "application/json", router.AGENT_HEADER: agent}

        def run():
            exchanges.append(self.router.exchange(path, headers, payload))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        return thread, exchanges

    def test_affinity_beats_config_priority(self):
        first, second = self.upstream("w0"), self.upstream("w1")
        r = self.make_router([worker("w0", first.url, "m0"), worker("w1", second.url, "m1")])
        self.send(CHAT, agent="A", expect=200)
        self.assertEqual(first.seen(), [(PATH, "m0")])

        # Fill w0 with another agent so A cannot use it.
        first.pause()
        thread, held = self.hold("B")
        wait_until(lambda: r.pool.stats()[0].in_flight == 1)
        self.send(CHAT, agent="A", expect=200)
        self.assertEqual(second.seen(), [(PATH, "m1")])

        first.resume()
        thread.join(timeout=5)
        r.finish(held[0])
        held[0].response.close()
        wait_until(lambda: r.pool.stats()[0].in_flight == 0)

        # w0 is free and first in config order, but A's affinity is w1.
        self.send(CHAT, agent="A", expect=200)
        self.assertEqual(len(second.seen()), 2)  # A's 2nd and 3rd requests
        self.assertEqual(len(first.seen()), 2)  # A's 1st request and B's held one

        stats = r.stats()
        self.assertEqual(stats["affinity"], {"hits": 1, "opportunities": 2, "rate": 0.5})
        agents = {entry["agent"]: entry for entry in stats["agents"]}
        self.assertEqual(agents["A"], {"agent": "A", "worker": "w1", "switches": 1})

    def test_overflow_is_used_only_when_every_local_is_full(self):
        local1, local2, paid = self.upstream("l1"), self.upstream("l2"), self.upstream("paid")
        r = self.make_router(
            [
                worker("l1", local1.url, concurrency=1),
                worker("l2", local2.url, concurrency=1),
                worker("paid", paid.url, model="m-paid", overflow=True, max_requests_per_hour=1),
            ],
            queue_timeout=0.2,
        )
        self.send(CHAT, agent="A", expect=200)
        self.assertEqual(paid.seen(), [])

        local1.pause()
        local2.pause()
        held = [self.hold("X"), self.hold("Y")]
        wait_until(lambda: [w.in_flight for w in r.pool.workers] == [1, 1, 0])

        self.send(CHAT, agent="Z", expect=200)
        self.assertEqual(len(paid.seen()), 1)

        # The paid worker's hourly cap is spent, so this request may not use it:
        # it waits for a local instead (and gives up here because the test bound
        # the wait).
        status, body, _ = self.send(CHAT, agent="W")
        self.assertEqual(status, 503)
        self.assertIn("all workers are busy", body.decode())
        self.assertEqual(len(paid.seen()), 1)
        self.assertEqual(r.pool.stats()[2].queued, 0)

        # A local frees: the next request uses it, never the capped overflow.
        local1.resume()
        local2.resume()
        for thread, exchanges in held:
            thread.join(timeout=5)
            r.finish(exchanges[0])
            exchanges[0].response.close()
        wait_until(lambda: [w.in_flight for w in r.pool.workers] == [0, 0, 0])
        self.send(CHAT, agent="V", expect=200)
        self.assertEqual(len(paid.seen()), 1)
        self.assertTrue(local1.seen() or local2.seen())

    def test_max_context_skips_a_worker_and_rejects_impossible_prompts(self):
        small, big = self.upstream("small"), self.upstream("big")
        r = self.make_router(
            [
                worker("small", small.url, concurrency=1, max_context=10),
                worker("big", big.url, concurrency=1, max_context=100),
            ]
        )
        long_prompt = {"model": "c", "messages": [{"role": "user", "content": "x" * 70}]}
        self.send(long_prompt, agent="A", expect=200)
        self.assertEqual(small.seen(), [])
        self.assertEqual(len(big.seen()), 1)

        huge = {"model": "c", "messages": [{"role": "user", "content": "x" * 400}]}
        status, body, _ = self.send(huge, agent="B")
        self.assertEqual(status, 413)
        self.assertIn("too large for every worker", body.decode())

    def test_model_rewrite_and_api_key_injection(self):
        upstream = self.upstream("w")
        os.environ["TASKGRAPH_TEST_KEY"] = "s3cret-value"
        self.addCleanup(os.environ.pop, "TASKGRAPH_TEST_KEY", None)
        log = io.StringIO()
        r = self.make_router(
            [worker("w", upstream.url, model="upstream-model", api_key_env="TASKGRAPH_TEST_KEY")],
            log=log,
        )
        self.send(CHAT, agent="A", headers={"Authorization": "Bearer client-key"}, expect=200)
        request = upstream.snapshot()[0]
        self.assertEqual(request["body"]["model"], "upstream-model")
        self.assertEqual(request["headers"]["Authorization"], "Bearer s3cret-value")
        self.assertNotIn("s3cret-value", json.dumps(r.stats()))
        self.assertNotIn("s3cret-value", log.getvalue())

    def test_a_client_key_passes_through_when_the_worker_has_none(self):
        upstream = self.upstream("w")
        self.make_router([worker("w", upstream.url, model="m")])
        self.send(CHAT, agent="A", headers={"Authorization": "Bearer client-key"}, expect=200)
        self.assertEqual(upstream.snapshot()[0]["headers"]["Authorization"], "Bearer client-key")

    def test_a_worker_with_a_missing_key_is_skipped_and_warned_once(self):
        upstream = self.upstream("w")
        os.environ.pop("TASKGRAPH_MISSING_KEY", None)
        log = io.StringIO()
        self.make_router([worker("w", upstream.url, api_key_env="TASKGRAPH_MISSING_KEY")], log=log)
        status, _, _ = self.send(CHAT, agent="A")
        self.assertEqual(status, 503)
        self.assertEqual(upstream.seen(), [])
        self.assertEqual(log.getvalue().count("TASKGRAPH_MISSING_KEY"), 1)
        self.send(CHAT, agent="A")
        self.assertEqual(log.getvalue().count("TASKGRAPH_MISSING_KEY"), 1)

    def test_an_unreachable_upstream_is_a_502(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.make_router([worker("off", f"http://127.0.0.1:{port}")])
        status, body, _ = self.send(CHAT, agent="A")
        self.assertEqual(status, 502)
        self.assertIn("upstream unreachable", body.decode())

    def test_queue_timeout_reports_503_and_frees_the_queue(self):
        upstream = self.upstream("w")
        r = self.make_router([worker("w", upstream.url, concurrency=1)], queue_timeout=0.1)
        upstream.pause()
        thread, held = self.hold("A")
        wait_until(lambda: r.pool.stats()[0].in_flight == 1)
        status, body, headers = self.send(CHAT, agent="B")
        self.assertEqual(status, 503)
        self.assertEqual(headers["Retry-After"], "1")
        self.assertIn("all workers are busy", body.decode())
        self.assertEqual(r.pool.stats()[0].queued, 0)
        upstream.resume()
        thread.join(timeout=5)
        r.finish(held[0])
        held[0].response.close()

    def test_error_paths_do_not_leak_slots(self):
        upstream = self.upstream("w")
        r = self.make_router([worker("w", upstream.url, concurrency=1, max_context=5)])
        for _ in range(3):
            self.send({"messages": [{"role": "user", "content": "x" * 400}]}, agent="A")
        self.assertEqual(r.pool.stats()[0].in_flight, 0)
        self.send(CHAT, agent="A", expect=200)


class HttpTest(unittest.TestCase):
    """The same policy through the real HTTP server, with concurrent clients."""

    def setUp(self):
        self.upstream = FakeUpstream("w").start()
        self.addCleanup(self.upstream.stop)
        self.router = router.Router(
            [worker("w", self.upstream.url, model="m", concurrency=2)], upstream_timeout=30
        )
        self.server = proxyserver.RouterServer(("127.0.0.1", 0), self.router)
        self.port = self.server.server_address[1]
        thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=5)

        self.addCleanup(stop)

    def post(self, payload=None, *, agent="A", headers=None, timeout=15):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        sent = {"Content-Type": "application/json"}
        if agent:
            sent[router.AGENT_HEADER] = agent
        sent.update(headers or {})
        body = json.dumps(payload if payload is not None else CHAT).encode()
        try:
            connection.request("POST", PATH, body=body, headers=sent)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def raw(self, data: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(data)
            chunks = []
            while True:
                part = sock.recv(4096)
                if not part:
                    return b"".join(chunks)
                chunks.append(part)

    def test_proxies_a_request_and_rewrites_the_model(self):
        status, headers, body = self.post(CHAT, agent="A")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(body)["model"], "m")  # the upstream's reply model
        self.assertEqual(self.upstream.snapshot()[0]["body"]["model"], "m")

    def test_never_exceeds_the_worker_concurrency(self):
        self.upstream.pause()
        results = []
        lock = threading.Lock()

        def client(agent):
            status, _, body = self.post(CHAT, agent=agent, timeout=30)
            with lock:
                results.append((agent, status, body))

        threads = [
            threading.Thread(target=client, args=(f"T{i}",), daemon=True) for i in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            self.addCleanup(thread.join, 5)

        wait_until(
            lambda: (self.router.pool.stats()[0].in_flight, self.router.pool.stats()[0].queued)
            == (2, 4)
        )
        stats = self.router.pool.stats()[0]
        self.assertEqual(stats.in_flight, 2)
        self.assertEqual(stats.queued, 4)
        self.assertEqual(self.upstream.max_active, 2)  # never 3

        self.upstream.resume()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(len(results), 6)
        self.assertTrue(all(status == 200 for _, status, _ in results), results)
        self.assertEqual(self.upstream.max_active, 2)
        self.assertEqual(len(self.upstream.seen()), 6)
        self.assertEqual(self.router.pool.stats()[0].admitted, 6)
        self.assertEqual(self.router.pool.stats()[0].in_flight, 0)

    def test_stats_endpoint(self):
        self.post(CHAT, agent="A")
        status, body = self.get("/router/stats")
        self.assertEqual(status, 200)
        document = json.loads(body)
        self.assertEqual(document["window_seconds"], gate.STATS_WINDOW)
        self.assertEqual(document["affinity"], {"hits": 0, "opportunities": 0, "rate": None})
        entry = document["workers"][0]
        self.assertEqual(entry["name"], "w")
        self.assertEqual(entry["concurrency"], 2)
        self.assertEqual(entry["admitted"], 1)
        self.assertEqual(entry["in_flight"], 0)
        self.assertIn("busy_seconds", entry)
        self.assertIn("utilization", entry)
        self.assertEqual(document["agents"], [{"agent": "A", "worker": "w", "switches": 0}])

    def test_unknown_path_is_404(self):
        status, body = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertIn("unknown path", body.decode())

    def test_a_body_without_content_length_is_411(self):
        response = self.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertIn(b"411", response.split(b"\r\n", 1)[0])

    def test_chunked_request_bodies_are_rejected(self):
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
        )
        response = self.raw(request)
        self.assertIn(b"501", response.split(b"\r\n", 1)[0])
        self.assertEqual(self.upstream.seen(), [])


class CliTest(unittest.TestCase):
    """``taskgraph router`` end to end through the real entry point."""

    TOML = """\
    plan = "PLAN.md"
    prompt = "PROMPT.md"
    gate = "true"
    worktrees = "../wt"
    main = "main"

    [agent]
    command = "fake-agent"

    [[models]]
    name = "m"
    sessions = 1

    [[workers]]
    name = "w"
    upstream = "{upstream}"
    model = "router-served"
    concurrency = 1
    """

    def start_cli(self, toml: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "taskgraph.toml"
        path.write_text(textwrap.dedent(toml), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(BIN), "router", "--project", tmp.name, "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp.name,
        )
        self.addCleanup(self.kill, process)
        return process

    def kill(self, process):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def listening_port(self, process) -> int:
        deadline = time.monotonic() + 10
        line = ""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stderr], [], [], 0.5)
            if ready:
                line = process.stderr.readline()
                if line:
                    break
        match = re.search(r"listening on http://127\.0\.0\.1:(\d+)", line)
        self.assertIsNotNone(match, f"no listening line: {line!r}")
        return int(match.group(1))

    def test_serves_stats_and_stops_on_sigterm(self):
        upstream = FakeUpstream("w").start()
        self.addCleanup(upstream.stop)
        process = self.start_cli(self.TOML.replace("{upstream}", upstream.url))
        port = self.listening_port(process)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            PATH,
            body=json.dumps(CHAT).encode(),
            headers={"Content-Type": "application/json", router.AGENT_HEADER: "T01"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["model"], "router-served")
        connection.close()

        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 0)

    def test_refuses_without_workers(self):
        toml = self.TOML.replace("{upstream}", "http://127.0.0.1:1").split("[[workers]]")[0]
        process = self.start_cli(toml)
        _, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 2)
        self.assertIn("no [[workers]] configured", stderr)


if __name__ == "__main__":
    unittest.main()
