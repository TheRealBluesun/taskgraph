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

from taskgraph import gate, relay
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

    def run_request(
        self, pool, name, admitted, tickets, *, priority=0, candidates=None, timeout=None, cancel=None
    ):
        """Start one blocked request in a thread; returns the thread."""
        seq = pool.next_seq()  # arrival order is decided here, by the caller
        candidates = candidates or (lambda: [pool.workers[0]])

        def run():
            ticket = pool.acquire(priority, seq, candidates, timeout=timeout, cancel=cancel)
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

    def test_a_busy_request_does_not_block_one_an_earlier_worker_can_serve(self):
        small = worker("small", concurrency=1, max_context=10)
        big = worker("big", concurrency=1)
        pool = gate.Pool([small, big])
        head_holder = pool.acquire(0, pool.next_seq(), lambda: [big])
        admitted, tickets = [], {}
        # The head only fits on ``big`` (busy); the request behind it can use
        # the free ``small``, and is admitted rather than deadlocked behind it.
        self.run_request(pool, "head", admitted, tickets, candidates=lambda: [big])
        self.run_request(pool, "next", admitted, tickets, candidates=lambda: [small, big])
        wait_until(lambda: "next" in tickets)
        self.assertEqual(tickets["next"].worker.name, "small")
        self.assertEqual(admitted, ["next"])
        self.assertEqual(pool.stats()[0].in_flight, 1)  # small
        self.assertEqual(pool.stats()[1].queued, 1)  # head still waits for big

        pool.release(head_holder)
        wait_until(lambda: "head" in tickets)
        self.assertEqual(tickets["head"].worker.name, "big")
        pool.release(tickets["head"])
        pool.release(tickets["next"])
        self.assertEqual(admitted, ["next", "head"])

    def test_a_request_that_is_busy_elsewhere_never_loses_its_slot(self):
        # Every waiter must get exactly the slot granted to *it*: the pool
        # admits on behalf of another entry, so a grant must not be dropped.
        pool = gate.Pool([worker("a", concurrency=1)])
        admitted, tickets = [], {}
        holder = pool.acquire(0, pool.next_seq(), lambda: [pool.workers[0]])
        for name in ("A", "B", "C"):
            self.run_request(pool, name, admitted, tickets)
        wait_until(lambda: pool.stats()[0].queued == 3)
        for name in ("A", "B", "C"):
            pool.release(holder)
            wait_until(lambda name=name: name in tickets)
            holder = tickets[name]
        self.assertEqual(admitted, ["A", "B", "C"])
        self.assertEqual(pool.stats()[0].queued, 0)

    def test_a_cancelled_request_leaves_the_queue_and_does_not_block_it(self):
        pool = gate.Pool([worker("a", concurrency=1)], poll=0.05)
        holder = pool.acquire(0, pool.next_seq(), lambda: [pool.workers[0]])
        gone = threading.Event()
        admitted, tickets = [], {}
        self.run_request(
            pool,
            "gone",
            admitted,
            tickets,
            timeout=5,
            cancel=lambda: gone.is_set(),
        )
        wait_until(lambda: pool.stats()[0].queued == 1)
        gone.set()
        wait_until(lambda: pool.stats()[0].queued == 0)
        pool.release(holder)
        wait_until(lambda: pool.stats()[0].in_flight == 0)
        self.assertEqual(admitted, [])

    def test_a_candidate_that_becomes_usable_on_the_poll_timer_is_admitted(self):
        # An ``overflow`` worker whose hourly cap rolls over becomes usable
        # with no release to wake the queue; the poll timer must pick it up.
        pool = gate.Pool([worker("a", concurrency=1)], poll=0.05)
        allowed = threading.Event()
        admitted, tickets = [], {}
        self.run_request(
            pool,
            "later",
            admitted,
            tickets,
            timeout=5,
            candidates=lambda: [pool.workers[0]] if allowed.is_set() else [],
        )
        wait_until(lambda: pool.waiting() == 1)
        allowed.set()
        wait_until(lambda: "later" in tickets, timeout=5)
        self.assertEqual(admitted, ["later"])
        pool.release(tickets["later"])

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

    def test_record_outcome_counts_done_and_errors(self):
        pool = gate.Pool([worker("a"), worker("b")])
        first, second = pool.workers
        pool.record_outcome(first, gate.DONE)
        pool.record_outcome(first, gate.DONE)
        pool.record_outcome(first, gate.ERROR)
        stats = pool.stats()
        self.assertEqual((stats[0].done, stats[0].errors), (2, 1))
        self.assertEqual((stats[1].done, stats[1].errors), (0, 0))

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
        response = relay.forward(
            worker("up", upstream=self.up.url, model="m"),
            "/v1/chat/completions",
            b'{"model": "x"}',
            {"Content-Type": "application/json"},
            timeout=5,
        )
        out = io.BytesIO()
        written = relay.relay(response, out)
        self.assertEqual(response.status, 429)
        headers = dict(relay.response_headers(response))
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Content-Length", headers)
        self.assertEqual(written, len(out.getvalue()))
        self.assertEqual(json.loads(out.getvalue())["model"], "x")

    def test_unreachable_upstream_raises_gate_error(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with self.assertRaises(relay.GateError) as ctx:
            relay.forward(
                worker("off", upstream=f"http://127.0.0.1:{port}", model="m"),
                "/v1/chat/completions",
                b"{}",
                {},
                timeout=2,
            )
        self.assertIn("upstream unreachable", str(ctx.exception))

    def test_a_hung_upstream_times_out(self):
        self.up.pause()
        with self.assertRaises(relay.GateError) as ctx:
            relay.forward(
                worker("up", upstream=self.up.url, model="m"),
                "/v1/chat/completions",
                b"{}",
                {},
                timeout=0.2,
            )
        self.assertIn("timed out", str(ctx.exception))

    def test_only_v1_paths_are_allowed(self):
        for good in ("/v1/models", "/v1/chat/completions", "/v1/chat/completions?stream=1"):
            self.assertTrue(relay.path_ok(good), good)
        for bad in (
            "/",
            "/admin",
            "/v1",
            "/v1/",
            "//evil.example/v1/x",
            "http://evil.example/v1/x",
            "/v1/x\r\nHost: evil",
            "/v1/x y",
            "/v1/x#frag",
            "/v1/..\\..",
            "/V1/x",
        ):
            self.assertFalse(relay.path_ok(bad), bad)

    def test_upstream_url_refuses_a_path_that_would_change_the_host(self):
        up = worker("up", upstream="http://127.0.0.1:9/v1-base", model="m")
        self.assertEqual(relay.upstream_url(up, "/v1/models"), "http://127.0.0.1:9/v1-base/v1/models")
        for bad in ("//evil.example/v1/x", "/admin", "/v1/x\r\n"):
            with self.assertRaises(relay.GateError):
                relay.upstream_url(up, bad)

    def test_redirects_are_not_followed(self):
        other = FakeUpstream("other").start()
        self.addCleanup(other.stop)
        self.up.redirect_to = other.url + "/v1/chat/completions"
        response = relay.forward(
            worker("up", upstream=self.up.url, model="m"),
            "/v1/chat/completions",
            b'{"model": "m"}',
            {"Content-Type": "application/json", "Authorization": "Bearer secret"},
            timeout=5,
        )
        self.assertEqual(response.status, 302)
        self.assertEqual(dict(relay.response_headers(response))["Location"], self.up.redirect_to)
        response.close()
        self.assertEqual(other.seen(), [])  # the injected key never travelled

    def test_classify_calls_a_reset_after_the_reply_done(self):
        self.assertEqual(relay.classify(True), gate.DONE)
        self.assertEqual(relay.classify(False), gate.DONE)  # no exception: done
        self.assertEqual(relay.classify(True, BrokenPipeError()), gate.DONE)
        self.assertEqual(relay.classify(False, ConnectionResetError()), gate.ERROR)
        self.assertEqual(relay.classify(True, ValueError("boom")), gate.ERROR)

    def test_chunks_are_relayed_as_they_arrive(self):
        self.up.set_chunks(2)
        response = relay.forward(
            worker("up", upstream=self.up.url, model="m"),
            "/v1/chat/completions",
            b'{"model": "m", "stream": true}',
            {"Content-Type": "application/json"},
            timeout=10,
        )
        seen = b""

        def first_chunk():
            nonlocal seen
            seen += relay.read_chunk(response, 4096)
            return b"part0" in seen

        # A buffering relay would block here until the upstream finished.
        self.up.release_chunk(0)
        self.assertTrue(wait_until(first_chunk, timeout=5), seen)
        self.assertNotIn(b"part1", seen)

        self.up.release_chunk(1)
        rest = b""
        deadline = time.monotonic() + 5
        while b"[DONE]" not in seen + rest and time.monotonic() < deadline:
            rest += relay.read_chunk(response, 4096)
        response.close()
        self.assertIn(b"part1", seen + rest)
        self.assertIn(b"[DONE]", rest)


if __name__ == "__main__":
    unittest.main()
