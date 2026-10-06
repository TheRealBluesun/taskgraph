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


if __name__ == "__main__":
    unittest.main()
