"""Tests for ``taskgraph.metrics`` (SPEC §4). No network: ``sample`` is exercised
through ``file://`` URLs so urllib's fetch path is real but never opens a socket.
"""

import tempfile
import unittest
from pathlib import Path

from taskgraph.metrics import Metrics, parse, sample

# Shaped like a real vLLM scrape: per-(model, engine) series, comments, an
# unrelated metric and the ``_created`` companion of the counter.
FIXTURE = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen3.8-27B",engine="0"} 3.0
vllm:num_requests_running{model_name="Qwen3.8-27B",engine="1"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting.
vllm:num_requests_waiting{model_name="Qwen3.8-27B",engine="0"} 1.0
vllm:num_requests_waiting{model_name="Qwen3.8-27B",engine="1"} 0.0
# HELP vllm:generation_tokens_total Generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="Qwen3.8-27B",engine="0"} 12000.0
vllm:generation_tokens_total{model_name="Qwen3.8-27B",engine="1"} 3456.0
vllm:generation_tokens_total_created{model_name="Qwen3.8-27B"} 1.699e9
vllm:prompt_tokens_total{model_name="Qwen3.8-27B"} 999.0
"""


class ParseTest(unittest.TestCase):
    def test_sums_each_series(self):
        m = parse(FIXTURE)
        self.assertEqual(m.running, 5.0)
        self.assertEqual(m.waiting, 1.0)
        self.assertEqual(m.generation_tokens, 15456.0)

    def test_created_and_unrelated_series_are_ignored(self):
        # 999 prompt tokens and 1.699e9 created-value must not leak in.
        self.assertEqual(parse(FIXTURE).generation_tokens, 15456.0)

    def test_labels_with_spaces_do_not_confuse_value(self):
        m = parse('vllm:num_requests_running{a="b c d"} 7\n')
        self.assertEqual(m.running, 7.0)

    def test_labelless_series_is_counted(self):
        m = parse("vllm:num_requests_running 4\n")
        self.assertEqual(m.running, 4.0)

    def test_trailing_timestamp_is_ignored(self):
        m = parse("vllm:num_requests_waiting{} 2 1699000000000\n")
        self.assertEqual(m.waiting, 2.0)

    def test_missing_metrics_default_to_zero(self):
        self.assertEqual(parse(""), Metrics())
        self.assertEqual(parse("# only a comment\n"), Metrics())

    def test_non_finite_and_malformed_values_are_skipped(self):
        text = (
            "vllm:num_requests_running 1\n"
            "vllm:num_requests_running NaN\n"
            "vllm:num_requests_running -Inf\n"
            "vllm:num_requests_running not-a-number\n"
            "vllm:num_requests_running\n"
        )
        self.assertEqual(parse(text).running, 1.0)

    def test_whitespace_and_blank_lines_tolerated(self):
        m = parse("\n  vllm:num_requests_running{engine=\"0\"}   3   \n\n")
        self.assertEqual(m.running, 3.0)


class SampleTest(unittest.TestCase):
    def test_sample_reads_and_parses_a_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.txt"
            path.write_text(FIXTURE, encoding="utf-8")
            metrics = sample(path.as_uri())
        self.assertEqual(metrics, Metrics(5.0, 1.0, 15456.0))

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.txt"
            self.assertIsNone(sample(missing.as_uri()))

    def test_malformed_url_returns_none(self):
        self.assertIsNone(sample("not a url"))


if __name__ == "__main__":
    unittest.main()
