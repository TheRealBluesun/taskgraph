"""``taskgraph stats`` tests (SPEC §9).

Three layers: the trace helpers and the pure split (:func:`taskgraph.stats.parse_text`,
:func:`taskgraph.stats.summarize`) run on fixture text shaped like a real omp
trace; :func:`taskgraph.stats.collect` runs against a temp project with fake
traces under the worktree base; and one test drives the real ``bin/taskgraph
stats`` CLI. No omp, no model, no network.
"""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from taskgraph import buckets, config, stats
from taskgraph.trace import Call, Message

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "taskgraph"

#: A realistic epoch base: omp writes epoch **milliseconds** (~1.79e12).
BASE = 1_791_253_000.0

PLAN = """\
- [ ] T01 first
- [ ] T02 second
"""

TOML = """\
plan = "PLAN.md"
prompt = "PROMPT.md"
gate = "true"
worktrees = "wt"
main = "main"

[agent]
command = "true"

[[models]]
name = "m1"
sessions = 1
"""


def line(
    role: str = "assistant",
    *,
    start: float = 1000.0,
    duration: float | None = None,
    completed: float | None = None,
    tools: tuple = (),
    timestamp: object = ...,
) -> str:
    """One ``message_end`` trace line; times in seconds, written as epoch ms."""
    message: dict = {"role": role, "content": [{"type": "text", "text": "…"}]}
    for index, tool in enumerate(tools):
        if isinstance(tool, Call):
            name = tool.name
            args = {"command": tool.command} if tool.command is not None else {}
        else:
            name, args = tool if isinstance(tool, tuple) else (tool, {})
        message["content"].append(
            {"type": "toolCall", "id": f"call_{index}", "name": name, "arguments": args}
        )
    message["timestamp"] = int(start * 1000)
    if timestamp is not ...:
        message["timestamp"] = timestamp
    if duration is not None:
        message["duration"] = duration * 1000.0
    if completed is not None:
        message["completedAt"] = int(completed * 1000)
    return json.dumps({"type": "message_end", "message": message})


def bash(command: str) -> Call:
    """A ``bash`` tool call fixture (what :func:`taskgraph.stats.parse_text` yields)."""
    return Call("bash", command)


class TraceHelperTest(unittest.TestCase):
    def test_epoch_normalises_ms_seconds_and_iso(self):
        self.assertEqual(stats.trace.epoch(1_791_253_037_166), 1_791_253_037.166)
        self.assertEqual(stats.trace.epoch(1_791_253_037), 1_791_253_037.0)
        self.assertEqual(stats.trace.epoch("1970-01-01T00:00:10+00:00"), 10.0)
        self.assertEqual(stats.trace.epoch("1970-01-01T00:00:10Z"), 10.0)
        self.assertEqual(stats.trace.epoch("1791253037"), 1_791_253_037.0)

    def test_epoch_rejects_junk(self):
        for value in (None, True, "", "not a time", {}, []):
            with self.subTest(value=value):
                self.assertIsNone(stats.trace.epoch(value))

    def test_tool_command_only_for_shell_tools(self):
        self.assertEqual(stats.trace.tool_command({"command": "make test"}), "make test")
        self.assertIsNone(stats.trace.tool_command({}))
        self.assertIsNone(stats.trace.tool_command({"command": ""}))
        self.assertIsNone(stats.trace.tool_command({"command": 3}))
        self.assertIsNone(stats.trace.tool_command("make test"))


class ParseTextTest(unittest.TestCase):
    def test_message_end_lines_only(self):
        text = "\n".join(
            [
                json.dumps({"type": "session", "timestamp": "2026-10-06T02:17:16.699Z"}),
                json.dumps({"type": "message_start", "message": {"role": "assistant"}}),
                line(start=BASE + 1000.0, duration=2.0, tools=(bash("make"),)),
                "not json at all, message_end",
                json.dumps({"type": "tool_execution_start", "toolName": "bash"}),
                line(role="toolResult", start=BASE + 1003.0),
                line(role="assistant", start=BASE + 1004.0, duration=1.0, tools=(bash("ls"),)),
            ]
        )
        messages = stats.parse_text(text)
        self.assertEqual(
            [(m.role, m.start, m.end) for m in messages],
            [
                ("assistant", BASE + 1000.0, BASE + 1002.0),
                ("toolResult", BASE + 1003.0, BASE + 1003.0),
                ("assistant", BASE + 1004.0, BASE + 1005.0),
            ],
        )
        self.assertEqual(messages[0].call, Call("bash", "make"))
        self.assertIsNone(messages[1].call)

    def test_completed_at_wins_over_the_duration(self):
        text = line(start=BASE + 1000.0, duration=2.0, completed=BASE + 1005.0)
        self.assertEqual(stats.parse_text(text)[0].end, BASE + 1005.0)

    def test_tool_call_without_a_command_keeps_its_name(self):
        text = line(tools=(("wait", {"timeout": 300}),))
        self.assertEqual(stats.parse_text(text)[0].call, Call("wait", None))

    def test_message_without_timestamp_is_skipped(self):
        self.assertEqual(stats.parse_text(line(timestamp=None)), [])

    def test_junk_timestamps_are_skipped(self):
        self.assertEqual(stats.parse_text(line(timestamp="soon")), [])

    def test_missing_trace_is_empty(self):
        self.assertEqual(stats.parse_trace("/nonexistent/trace.log"), [])


class BucketTest(unittest.TestCase):
    def test_default_buckets(self):
        cases = [
            (bash("taskgraph lease simulator -- xcodebuild -scheme App"), "lease-wait"),
            (bash("./dev.sh build && ./dev.sh test"), "build"),
            (bash("python3 -m unittest discover -s tests -q 2>&1 | tail -8"), "test"),
            (Call("wait", None), "wait"),
            (Call("edit", None), "other"),
            (bash("git status"), "other"),
            (None, "other"),
        ]
        for call, expected in cases:
            with self.subTest(call=call):
                self.assertEqual(buckets.bucket_for(call, buckets.DEFAULT_BUCKETS), expected)

    def test_first_match_wins_in_config_order(self):
        patterns = (("tests", r"\btests?\b"), ("builds", r"\bbuild\b"))
        self.assertEqual(buckets.bucket_for(bash("./dev.sh build && ./dev.sh test"), patterns), "tests")

    def test_bucket_names_put_other_last(self):
        self.assertEqual(
            buckets.bucket_names((("a", "x"), ("other", "y"), ("b", "z"))), ("a", "b", "other")
        )


class SummarizeTest(unittest.TestCase):
    def summarize(self, messages, **kw):
        return stats.summarize(messages, buckets.DEFAULT_BUCKETS, **kw)

    def test_model_and_tool_time_split(self):
        messages = [
            Message("user", 1000.0, 1000.0),
            Message("assistant", 1000.0, 1002.0, call=bash("make")),
            Message("toolResult", 1010.0, 1010.0),
            Message("assistant", 1010.0, 1015.0, call=Call("wait", None)),
            Message("toolResult", 1020.0, 1020.0),
        ]
        summary = self.summarize(messages)
        self.assertEqual(summary.messages, 5)
        self.assertEqual(summary.model, 7.0)  # 2 + 5, user/toolResult have none
        self.assertEqual(summary.tool, 13.0)  # to 1010 from 1002, to 1020 from 1015
        self.assertEqual(summary.buckets["build"], 8.0)
        self.assertEqual(summary.buckets["wait"], 5.0)
        self.assertEqual(summary.buckets["other"], 0.0)
        self.assertEqual(summary.wall, 20.0)

    def test_only_the_first_tool_call_buckets_the_gap(self):
        messages = [
            Message("assistant", 1000.0, 1000.0, call=bash("make")),
            Message("toolResult", 1003.0, 1003.0),
            Message("assistant", 1004.0, 1004.0, call=bash("unittest")),
            Message("toolResult", 1005.0, 1005.0),
        ]
        summary = self.summarize(messages)
        self.assertEqual(summary.buckets["build"], 3.0)
        self.assertEqual(summary.buckets["other"], 0.0)

    def test_a_message_without_tool_calls_has_no_tool_time(self):
        messages = [Message("assistant", 1000.0, 1010.0), Message("user", 5000.0, 5000.0)]
        summary = self.summarize(messages)
        self.assertEqual(summary.tool, 0.0)
        self.assertEqual(sum(summary.buckets.values()), 0.0)
        self.assertEqual(summary.model, 10.0)

    def test_parallel_results_use_the_next_message(self):
        messages = [
            Message("assistant", 1000.0, 1000.0, call=bash("make")),
            Message("toolResult", 1002.0, 1002.0),
            Message("toolResult", 1002.0, 1002.0),
        ]
        self.assertEqual(self.summarize(messages).tool, 2.0)

    def test_overlapping_timestamps_contribute_zero(self):
        messages = [
            Message("assistant", 1000.0, 1010.0, call=bash("make")),
            Message("toolResult", 1005.0, 1005.0),
        ]
        summary = self.summarize(messages)
        self.assertEqual(summary.tool, 0.0)
        self.assertEqual(summary.buckets["build"], 0.0)

    def test_window_drops_older_messages(self):
        messages = [
            Message("assistant", 0.0, 10.0, call=bash("make")),
            Message("toolResult", 10.0, 10.0),
            Message("assistant", 1000.0, 1005.0, call=bash("unittest")),
            Message("toolResult", 1008.0, 1008.0),
        ]
        summary = self.summarize(messages, since=100.0, now=1100.0)
        self.assertEqual(summary.messages, 2)
        self.assertEqual(summary.model, 5.0)
        self.assertEqual(summary.buckets["test"], 3.0)
        self.assertEqual(summary.buckets["build"], 0.0)

    def test_empty_trace(self):
        summary = self.summarize([])
        self.assertEqual((summary.messages, summary.wall, summary.model, summary.tool), (0, 0.0, 0.0, 0.0))
        self.assertEqual(sum(summary.buckets.values()), 0.0)


class WindowTest(unittest.TestCase):
    def test_parses_units(self):
        self.assertEqual(stats.parse_window("6h"), 21600.0)
        self.assertEqual(stats.parse_window("30m"), 1800.0)
        self.assertEqual(stats.parse_window("90s"), 90.0)
        self.assertEqual(stats.parse_window("2d"), 172800.0)
        self.assertEqual(stats.parse_window("45"), 45.0)
        self.assertEqual(stats.parse_window("1.5h"), 5400.0)

    def test_rejects_junk(self):
        for text in ("", "h", "6w", "-1h", "later", "1h30"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    stats.parse_window(text)

    def test_format_window(self):
        self.assertEqual(stats.format_window(21600.0), "6h")
        self.assertEqual(stats.format_window(90.0), "90s")
        self.assertEqual(stats.format_window(86400.0), "1d")
        self.assertEqual(stats.format_window(None), "all time")


class AggregateTest(unittest.TestCase):
    def test_task_and_total_rows_sum_traces(self):
        one = stats.TraceSummary(2, 10.0, 4.0, 5.0, {"test": 5.0, "other": 0.0})
        two = stats.TraceSummary(3, 20.0, 6.0, 8.0, {"test": 1.0, "other": 7.0})
        task = stats.task_stats("T01", [one, two])
        self.assertEqual((task.id, task.traces, task.wall, task.model, task.tool), ("T01", 2, 30.0, 10.0, 13.0))
        self.assertEqual(task.buckets, {"test": 6.0, "other": 7.0})
        total = stats.totals(stats.Stats(tasks=(task,)))
        self.assertEqual((total.id, total.traces, total.tool), ("total", 2, 13.0))


class ProjectTestCase(unittest.TestCase):
    """A temp project whose worktree base holds fake agent traces."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        (self.root / "PLAN.md").write_text(PLAN, encoding="utf-8")
        (self.root / "PROMPT.md").write_text("rules\n", encoding="utf-8")
        self.write_config(TOML)
        self.now = time.time()

    def write_config(self, text: str):
        (self.root / "taskgraph.toml").write_text(text, encoding="utf-8")
        self.cfg = config.load(self.root / "taskgraph.toml")

    def trace(self, tid: str, *lines: str, clock: str = "010101") -> Path:
        """Write ``tid``'s fake trace under the worktree base."""
        directory = self.root / "wt" / tid / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tid}-{clock}.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class CollectTest(ProjectTestCase):
    def test_discovers_and_orders_traces_like_the_plan(self):
        self.trace(
            "T02",
            line(start=BASE + 2000.0, duration=1.0, tools=(bash("unittest"),)),
            line(role="toolResult", start=BASE + 2004.0),
        )
        self.trace(
            "T01",
            line(start=BASE + 1000.0, duration=2.0, tools=(bash("make"),)),
            line(role="toolResult", start=BASE + 1003.0),
        )
        snapshot = stats.collect(self.cfg, since=None)
        self.assertEqual([task.id for task in snapshot.tasks], ["T01", "T02"])
        self.assertEqual(snapshot.tasks[0].buckets["build"], 1.0)
        self.assertEqual(snapshot.tasks[1].buckets["test"], 3.0)
        self.assertIn("total", stats.render(snapshot))

    def test_two_traces_of_one_task_are_summed(self):
        self.trace("T01", line(start=BASE + 1000.0, duration=1.0), clock="010101")
        self.trace("T01", line(start=BASE + 2000.0, duration=2.0), clock="020202")
        snapshot = stats.collect(self.cfg, since=None)
        self.assertEqual(len(snapshot.tasks), 1)
        self.assertEqual((snapshot.tasks[0].traces, snapshot.tasks[0].model), (2, 3.0))

    def test_gate_log_and_foreign_names_are_ignored(self):
        self.trace("T01", line(start=BASE + 1000.0, duration=1.0))
        logs = self.root / "wt" / "T01" / "logs"
        (logs / "gate.log").write_text("not a trace\n", encoding="utf-8")
        (logs / "notes.txt").write_text("x\n", encoding="utf-8")
        self.assertEqual([tid for tid, _ in stats.traces(self.cfg)], ["T01"])

    def test_missing_worktree_base_is_empty(self):
        self.assertEqual(stats.traces(self.cfg), [])
        self.assertEqual(stats.collect(self.cfg, since=None).tasks, ())

    def test_window_excludes_old_traces(self):
        self.trace("T01", line(start=self.now - 40_000.0, completed=self.now - 40_000.0))
        self.assertEqual(stats.collect(self.cfg, since=6 * 3600, now=self.now).tasks, ())
        self.assertEqual(len(stats.collect(self.cfg, since=None, now=self.now).tasks), 1)

    def test_configured_buckets_are_used(self):
        self.write_config(
            TOML
            + '\n[stats.buckets]\nmigrations = "\\\\bmigrate\\\\b"\n'
        )
        self.trace(
            "T01",
            line(start=self.now - 60.0, duration=1.0, tools=(bash("./manage.py migrate"),)),
            line(role="toolResult", start=self.now - 52.0),
        )
        snapshot = stats.collect(self.cfg, since=None)
        self.assertEqual(
            snapshot.buckets,
            ("migrations", "lease-wait", "build", "test", "wait", "other"),
        )
        self.assertEqual(snapshot.tasks[0].buckets["migrations"], 7.0)


    def test_a_configured_bucket_overrides_the_default_of_the_same_name(self):
        self.write_config(TOML + '\n[stats.buckets]\nbuild = "^gradle$"\n')
        self.trace(
            "T01",
            line(start=self.now - 60.0, duration=1.0, tools=(bash("make"),)),
            line(role="toolResult", start=self.now - 50.0),
        )
        snapshot = stats.collect(self.cfg, since=None)
        self.assertEqual(snapshot.tasks[0].buckets["build"], 0.0)
        self.assertEqual(snapshot.tasks[0].buckets["other"], 9.0)


class RenderTest(unittest.TestCase):
    def snapshot(self) -> stats.Stats:
        task = stats.TaskStats(
            "T01", 2, 47.0, 18.0, 29.0, {"test": 1500.0, "other": 4.0, "build": 0.0}
        )
        return stats.Stats(tasks=(task,), buckets=("test", "build", "other"), since=21600.0)

    def test_table_has_a_column_per_bucket_and_a_total_row(self):
        lines = stats.render(self.snapshot()).splitlines()
        self.assertEqual(
            lines[0].split(),
            ["TASK", "TRACES", "WALL", "MODEL", "TOOL", "TEST", "BUILD", "OTHER"],
        )
        self.assertEqual(lines[1].split(), ["T01", "2", "47s", "18s", "29s", "25m00s", "0s", "4s"])
        self.assertEqual(lines[2].split(), ["total", "2", "47s", "18s", "29s", "25m00s", "0s", "4s"])

    def test_empty_snapshot_names_the_window(self):
        self.assertEqual(
            stats.render(stats.Stats(since=21600.0)), "no agent traces in the last 6h"
        )
        self.assertEqual(stats.render(stats.Stats()), "no agent traces found")


class CliTest(ProjectTestCase):
    def run_stats(self, *argv, env=None):
        return subprocess.run(
            [sys.executable, str(BIN), "stats", *argv],
            cwd=self.root,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_prints_the_table(self):
        self.trace(
            "T01",
            line(start=time.time() - 60.0, duration=2.0, tools=(bash("make"),)),
            line(role="toolResult", start=time.time() - 30.0),
        )
        result = self.run_stats("--since", "6h")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0].split()[0], "TASK")
        self.assertEqual(lines[1].split()[0], "T01")
        self.assertEqual(lines[2].split()[0], "total")
        self.assertEqual(lines[2].split()[2], "30s")

    def test_since_is_defaulted_and_checked(self):
        self.assertIn("no agent traces in the last 6h", self.run_stats().stdout)
        bad = self.run_stats("--since", "6w")
        self.assertEqual(bad.returncode, 2)
        self.assertIn("invalid window", bad.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
