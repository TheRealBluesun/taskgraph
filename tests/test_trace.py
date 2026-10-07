"""``taskgraph.trace`` helpers (SPEC §6/§9): the shared trace line reader.

Only the parts without another test home live here — ``stats`` and ``status``
test the parsing they build on top of these functions. No omp, no network.
"""

import json
import tempfile
import unittest
from pathlib import Path

from taskgraph import trace


def tool(name: str, call_id: str = "1") -> str:
    """One ``tool_execution_start`` trace line, as omp writes it."""
    return json.dumps(
        {"type": "tool_execution_start", "toolCallId": call_id, "toolName": name}
    )


class LastToolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "trace.log"

    def write(self, *lines):
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_the_most_recent_start_wins(self):
        self.write(tool("read"), tool("bash", "2"))
        self.assertEqual(trace.last_tool(self.log), "bash")

    def test_an_echo_of_the_phrase_is_not_a_tool(self):
        self.write(
            "not json at all: tool_execution_start",
            json.dumps({"type": "message_update", "text": "tool_execution_start"}),
            json.dumps({"type": "tool_execution_end", "toolName": "read"}),
        )
        self.assertIsNone(trace.last_tool(self.log))

    def test_a_missing_trace_is_none(self):
        self.assertIsNone(trace.last_tool(self.log))

    def test_a_start_without_a_name_reports_a_placeholder(self):
        self.write(json.dumps({"type": "tool_execution_start", "toolCallId": "1"}))
        self.assertEqual(trace.last_tool(self.log), "?")


def call(name: str, args=None, call_id: str = "1") -> str:
    """One ``tool_execution_start`` line with arguments, as omp writes it."""
    entry = {"type": "tool_execution_start", "toolCallId": call_id, "toolName": name}
    if args is not None:
        entry["args"] = args
    return json.dumps(entry)


def signature(line: str) -> str:
    parsed = trace.event(line)
    assert parsed is not None, line
    return trace.call_signature(parsed)


class CallSignatureTest(unittest.TestCase):
    def test_the_call_id_is_ignored(self):
        self.assertEqual(
            signature(call("bash", {"command": "ls"}, "call_1")),
            signature(call("bash", {"command": "ls"}, "call_2")),
        )

    def test_arguments_identify_the_call(self):
        self.assertNotEqual(
            signature(call("bash", {"command": "ls"})),
            signature(call("bash", {"command": "ls -l"})),
        )

    def test_argument_key_order_does_not_matter(self):
        self.assertEqual(
            signature(call("read", {"path": "SPEC.md", "limit": 10})),
            signature(call("read", {"limit": 10, "path": "SPEC.md"})),
        )

    def test_the_tool_name_identifies_the_call(self):
        self.assertNotEqual(signature(call("read")), signature(call("write")))

    def test_a_call_without_arguments_has_a_signature(self):
        self.assertTrue(signature(call("wait")))


class RepeatedTailTest(unittest.TestCase):
    def ls(self, call_id: str = "1") -> str:
        return call("bash", {"command": "ls"}, call_id)

    def test_exactly_window_identical_calls_are_a_loop(self):
        self.assertTrue(trace.repeated_tail([self.ls(str(n)) for n in range(4)], 4))

    def test_fewer_calls_than_the_window_are_not_a_loop(self):
        self.assertFalse(trace.repeated_tail([self.ls(str(n)) for n in range(3)], 4))

    def test_only_the_last_window_matters(self):
        lines = [call("read", {"path": "SPEC.md"})] + [self.ls(str(n)) for n in range(3)]
        self.assertTrue(trace.repeated_tail(lines, 3))
        self.assertFalse(trace.repeated_tail(lines, 4))

    def test_same_tool_with_moving_arguments_is_not_a_loop(self):
        lines = [call("bash", {"command": f"ls {n}"}) for n in range(5)]
        self.assertFalse(trace.repeated_tail(lines, 3))

    def test_non_tool_lines_are_ignored(self):
        lines = [
            "plain prose",
            "not json: tool_execution_start",
            json.dumps({"type": "message_end", "text": "tool_execution_start"}),
            json.dumps({"type": "tool_execution_end", "toolName": "bash"}),
            self.ls("a"),
            self.ls("b"),
            self.ls("c"),
        ]
        self.assertTrue(trace.repeated_tail(lines, 3))

    def test_zero_or_negative_window_disables_the_check(self):
        lines = [self.ls(str(n)) for n in range(40)]
        self.assertFalse(trace.repeated_tail(lines, 0))
        self.assertFalse(trace.repeated_tail(lines, -1))

    def test_no_calls_is_not_a_loop(self):
        self.assertFalse(trace.repeated_tail([], 3))


if __name__ == "__main__":
    unittest.main()
