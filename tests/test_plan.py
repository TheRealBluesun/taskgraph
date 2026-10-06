"""Tests for ``taskgraph.plan`` (SPEC §2)."""

import unittest

from taskgraph.plan import PlanError, Task, mark_done, parse

PLAN = """\
# PLAN

Some prose that mentions F01 and is not a task.

- [ ] F01 Top strip [deps: F00, F02] [res: mac]
- [x] F02 Backend
- [ ] LX01a Linux portability [res: mac, simulator]
* [ ] ABC12z Suffix letter

Not a task: - [ ] Z99 should be ignored
- a plain bullet, no checkbox
"""


class ParseTest(unittest.TestCase):
    def test_parses_only_task_lines_in_plan_order(self):
        tasks = parse(PLAN)
        self.assertEqual(list(tasks), ["F01", "F02", "LX01a", "ABC12z"])

    def test_deps_and_res_tags(self):
        tasks = parse(PLAN)
        self.assertEqual(tasks["F01"].deps, ("F00", "F02"))
        self.assertEqual(tasks["F01"].res, ("mac",))
        self.assertEqual(tasks["LX01a"].deps, ())
        self.assertEqual(tasks["LX01a"].res, ("mac", "simulator"))

    def test_text_excludes_tags(self):
        self.assertEqual(parse(PLAN)["F01"].text, "Top strip")
        self.assertEqual(parse(PLAN)["LX01a"].text, "Linux portability")

    def test_done_and_undone(self):
        tasks = parse(PLAN)
        self.assertFalse(tasks["F01"].done)
        self.assertTrue(tasks["F02"].done)

    def test_ids_with_suffix_letter(self):
        tasks = parse(PLAN)
        self.assertEqual(tasks["LX01a"].id, "LX01a")
        self.assertEqual(tasks["ABC12z"].id, "ABC12z")

    def test_task_without_description_or_tags(self):
        task = parse("- [ ] Z9")["Z9"]
        self.assertEqual(task, Task(id="Z9", text="", done=False))

    def test_empty_and_trailing_comma_tags(self):
        task = parse("- [ ] A1 x [deps: F01,] [res: ]")["A1"]
        self.assertEqual(task.deps, ("F01",))
        self.assertEqual(task.res, ())
        self.assertEqual(task.text, "x")

    def test_lowercase_id_is_not_a_task(self):
        self.assertEqual(parse("- [ ] f03 lowercase"), {})

    def test_extra_whitespace_is_collapsed(self):
        task = parse("  -   [ ]   F05   two   spaces  [deps:  F01 ,  F02 ]  ")["F05"]
        self.assertEqual(task.text, "two spaces")
        self.assertEqual(task.deps, ("F01", "F02"))


class MarkDoneTest(unittest.TestCase):
    def test_changes_only_the_matching_line(self):
        before = "# PLAN\n\n- [ ] F01 first\n- [ ] F02 second\n- [x] F03 third\n"
        after = mark_done(before, "F02")
        self.assertEqual(
            after,
            "# PLAN\n\n- [ ] F01 first\n- [x] F02 second\n- [x] F03 third\n",
        )

    def test_preserves_missing_trailing_newline(self):
        self.assertEqual(mark_done("- [ ] F01 x", "F01"), "- [x] F01 x")

    def test_preserves_crlf(self):
        self.assertEqual(mark_done("- [ ] F01 x\r\n- [ ] F02 y\r\n", "F02"),
                         "- [ ] F01 x\r\n- [x] F02 y\r\n")

    def test_already_done_is_a_no_op(self):
        text = "- [x] F01 first\n"
        self.assertEqual(mark_done(text, "F01"), text)

    def test_id_mentioned_in_prose_is_untouched(self):
        text = "F01 is done.\n- [ ] F01 real\n"
        self.assertEqual(mark_done(text, "F01"), "F01 is done.\n- [x] F01 real\n")

    def test_unknown_id_raises(self):
        with self.assertRaises(PlanError):
            mark_done("- [ ] F01 x\n", "F99")

    def test_round_trip_with_parse(self):
        text = "- [ ] F01 [deps: F00] [res: mac]\n- [ ] F02\n"
        self.assertEqual(list(parse(mark_done(text, "F01"))), ["F01", "F02"])
        self.assertTrue(parse(mark_done(text, "F01"))["F01"].done)


if __name__ == "__main__":
    unittest.main()
