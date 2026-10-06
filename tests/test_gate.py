"""Tests for the router's admission pool and streaming relay (SPEC §11).

Everything runs against loopback fake upstreams (``tests/upstreams.py``): no
model, no omp, no network.
"""

import io
import json
import socket
import threading
import time
import unittest

from leasehelpers import wait_until
from upstreams import FakeUpstream

from taskgraph import gate
from taskgraph.config import WorkerConfig


def worker(name, *, upstream="http://127.0.0.1:9", model=None, **kw):
    return gate.Worker(WorkerConfig(name=name, upstream=upstream, model=model or name, **kw))


class FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def tick(self, secs):
        self.now += secs

    def set(self, now):
        self.now = now


class PoolTest(unittest.TestCase):
    """FIFO admission, priority, timeouts and the stats numbers."""

    def run_request(self, pool, name, admitted, tickets, *, priority=0, candidates=None, timeout=None):
        """Start one blocked request in a thread; returns the thread."""
        seq = pool.next_seq()  # arrival order is decided here, by the caller
        candidates = candidates or (lambda: [pool.workers[0]])

        def run():
            ticket = pool.acquire(priority, seq, candidates, timeout=timeout)
            if ticket is not None:
                admitted.append(name)
                tickets[name] = ticket

        thread = threading.Thread(target=run, name=f"req-{name}", daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        return thread

    def test_served_in_arrival_order(self):
        pool = gate.Pool([worker("a", concurrency=1)])
        admitted, tickets = [], {}
        self.run_request(pool, "A", admitted, tickets)
        wait_until(lambda: pool.stats()[0].in_flight == 1)
        self.run_request(pool, "B", admitted, tickets)
        wait_until(lambda: pool.stats()[0].queued == 1)
        self.run_request(pool, "C", admitted, tickets)
        wait_until(lambda: pool.stats()[0].queued == 2)

        pool.release(tickets["A"])
        wait_until(lambda: "B" in tickets)
        pool.release(tickets["B"])
        wait_until(lambda: "C" in tickets)
        pool.release(tickets["C"])
        self.assertEqual(admitted, ["A", "B", "C"])

    def test_a_higher_priority_request_overtakes_earlier_ones(self):
        pool = gate.Pool([worker("a", concurrency=1)])
        admitted, tickets = [], {}
        holder = pool.acquire(0, pool.next_seq(), lambda: [pool.workers[0]])
        self.run_request(pool, "low", admitted, tickets, priority=0)
        wait_until(lambda: pool.stats()[0].queued == 1)
        self.run_request(pool, "high", admitted, tickets, priority=-5)
        wait_until(lambda: pool.stats()[0].queued == 2)

        pool.release(holder)
        wait_until(lambda: "high" in tickets)
        self.assertEqual(admitted, ["high"])
        pool.release(tickets["high"])
        wait_until(lambda: "low" in tickets)
        self.assertEqual(admitted, ["high", "low"])
        pool.release(tickets["low"])

    def test_the_head_blocks_a_request_another_worker_could_serve(self):
        small = worker("small", concurrency=1, max_context=10)
        big = worker("big", concurrency=1)
        pool = gate.Pool([small, big])
        head_holder = pool.acquire(0, pool.next_seq(), lambda: [big])
        admitted, tickets = [], {}
        # The head only fits on ``big`` (busy); the request behind it could use
        # the free ``small``, but FIFO says it waits.
        self.run_request(pool, "head", admitted, tickets, candidates=lambda: [big])
        self.run_request(pool, "next", admitted, tickets, candidates=lambda: [small, big])
        wait_until(lambda: pool.stats()[1].queued == 1)
        self.assertEqual(pool.stats()[0].queued, 1)
        self.assertEqual(small.in_flight, 0)

        pool.release(head_holder)
        wait_until(lambda: "head" in tickets)
        self.assertEqual(tickets["head"].worker.name, "big")
        pool.release(tickets["head"])
        wait_until(lambda: "next" in tickets)
        self.assertEqual(tickets["next"].worker.name, "small")
        pool.release(tickets["next"])
        self.assertEqual(admitted, ["head", "next"])

    def test_timeout_returns_none_and_clears_the_queue_entry(self):
        pool = gate.Pool([worker("a", concurrency=1)])
        holder = pool.acquire(0, pool.next_seq(), lambda: [pool.workers[0]])
        admitted, tickets = [], {}
        thread = self.run_request(pool, "waiter", admitted, tickets, timeout=0.05)
        thread.join(timeout=5)
        self.assertEqual(admitted, [])
        self.assertEqual(pool.stats()[0].in_flight, 1)
        self.assertEqual(pool.stats()[0].queued, 0)
        pool.release(holder)

    def test_stats_count_in_flight_queued_admitted_and_busy_seconds(self):
        clock = FakeClock(100.0)
        fast = worker("fast", concurrency=1)
        free = worker("free", concurrency=1)
        pool = gate.Pool([fast, free], clock=clock, window=10.0)
        first = pool.acquire(0, pool.next_seq(), lambda: [fast])
        clock.tick(5.0)
        pool.release(first)
        second = pool.acquire(0, pool.next_seq(), lambda: [fast])
        stats = pool.stats()
        self.assertEqual((stats[0].in_flight, stats[0].admitted), (1, 2))
        self.assertAlmostEqual(stats[0].busy_seconds, 5.0)
        self.assertAlmostEqual(stats[0].utilization, 0.5)
        self.assertEqual((stats[1].in_flight, stats[1].admitted), (0, 0))

        # A queued request is attributed to the worker it would use first.
        holder = pool.acquire(0, pool.next_seq(), lambda: [free])
        admitted, tickets = [], {}
        self.run_request(pool, "waiter", admitted, tickets, candidates=lambda: [free, fast])
        wait_until(lambda: pool.stats()[1].queued == 1)
        self.assertEqual(pool.stats()[1].queued, 1)
        pool.release(holder)
        wait_until(lambda: admitted == ["waiter"])
        pool.release(tickets["waiter"])
        pool.release(second)

    def test_busy_seconds_forget_the_distant_past(self):
        clock = FakeClock(0.0)
        pl = gate.Pool([worker("a")], clock=clock, window=10.0)
        ticket = pl.acquire(0, pl.next_seq(), lambda: [pl.workers[0]])
        clock.tick(4.0)
        pl.release(ticket)
        clock.tick(100.0)
        stats = pl.stats()
        self.assertEqual(stats[0].busy_seconds, 0.0)
        self.assertEqual(stats[0].utilization, 0.0)


class RelayTest(unittest.TestCase):
    """Forwarding and streaming against a loopback upstream."""

    def setUp(self):
        self.up = FakeUpstream("up").start()
        self.addCleanup(self.up.stop)

    def test_response_status_headers_and_body_are_preserved(self):
        self.up.status = 429
        response = gate.forward(
            worker("up", upstream=self.up.url, model="m"),
            "/v1/chat/completions",
            b'{"model": "x"}',
            {"Content-Type": "application/json"},
            timeout=5,
        )
        out = io.BytesIO()
        written = gate.relay(response, out)
        self.assertEqual(response.status, 429)
        headers = dict(gate.response_headers(response))
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Content-Length", headers)
        self.assertEqual(written, len(out.getvalue()))
        self.assertEqual(json.loads(out.getvalue())["model"], "x")

    def test_unreachable_upstream_raises_gate_error(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with self.assertRaises(gate.GateError) as ctx:
            gate.forward(
                worker("off", upstream=f"http://127.0.0.1:{port}", model="m"),
                "/v1/chat/completions",
                b"{}",
                {},
                timeout=2,
            )
        self.assertIn("upstream unreachable", str(ctx.exception))

    def test_a_hung_upstream_times_out(self):
        self.up.pause()
        with self.assertRaises(gate.GateError) as ctx:
            gate.forward(
                worker("up", upstream=self.up.url, model="m"),
                "/v1/chat/completions",
                b"{}",
                {},
                timeout=0.2,
            )
        self.assertIn("timed out", str(ctx.exception))

    def test_chunks_are_relayed_as_they_arrive(self):
        self.up.set_chunks(2)
        response = gate.forward(
            worker("up", upstream=self.up.url, model="m"),
            "/v1/chat/completions",
            b'{"model": "m", "stream": true}',
            {"Content-Type": "application/json"},
            timeout=10,
        )
        seen = b""

        def first_chunk():
            nonlocal seen
            seen += gate.read_chunk(response, 4096)
            return b"part0" in seen

        # A buffering relay would block here until the upstream finished.
        self.up.release_chunk(0)
        self.assertTrue(wait_until(first_chunk, timeout=5), seen)
        self.assertNotIn(b"part1", seen)

        self.up.release_chunk(1)
        rest = b""
        deadline = time.monotonic() + 5
        while b"[DONE]" not in seen + rest and time.monotonic() < deadline:
            rest += gate.read_chunk(response, 4096)
        response.close()
        self.assertIn(b"part1", seen + rest)
        self.assertIn(b"[DONE]", rest)


if __name__ == "__main__":
    unittest.main()
