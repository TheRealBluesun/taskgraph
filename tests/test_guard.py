"""Tests for the commit-guard policy (SPEC §8).

Pure table-driven tests: no git, no temp dirs.  ``merge`` supplies the paths and
sizes from a real worktree in ``tests/test_merge.py``.
"""

import unittest

from taskgraph import guard


def problems(added=(), sizes=None, *, max_files=200, max_file_mb=5.0, deny_dirs=()):
    """Call :func:`guard.problems` with the usual defaults."""
    return guard.problems(
        list(added),
        dict(sizes or {}),
        max_files=max_files,
        max_file_mb=max_file_mb,
        deny_dirs=deny_dirs,
    )


class ProblemsTest(unittest.TestCase):
    def test_a_small_normal_commit_passes(self):
        self.assertEqual(
            problems(["src/a.py"], {"src/a.py": 10}, deny_dirs=guard.DEFAULT_DENY_PATHS), []
        )

    def test_an_empty_commit_passes(self):
        self.assertEqual(problems(deny_dirs=guard.DEFAULT_DENY_PATHS), [])

    def test_more_than_max_files_blocks_and_names_paths(self):
        added = [f"gen/f{i}.txt" for i in range(201)]
        issues = problems(added, {p: 1 for p in added})
        self.assertEqual(len(issues), 1)
        self.assertIn("adds 201 files (limit 200)", issues[0])
        self.assertIn("gen/f0.txt", issues[0])

    def test_exactly_max_files_passes(self):
        added = [f"gen/f{i}.txt" for i in range(200)]
        self.assertEqual(problems(added, {p: 1 for p in added}), [])

    def test_a_file_over_the_limit_blocks_with_its_size(self):
        issues = problems([], {"big.bin": 6 * 1024 * 1024})
        self.assertEqual(len(issues), 1)
        self.assertIn("files over 5 MB", issues[0])
        self.assertIn("big.bin (6.0 MB)", issues[0])

    def test_exactly_the_limit_passes(self):
        self.assertEqual(problems([], {"big.bin": 5 * 1024 * 1024}), [])

    def test_a_modified_file_is_size_checked_too(self):
        issues = problems([], {"vendor/blob": 7 * 1024 * 1024})
        self.assertIn("vendor/blob", issues[0])

    def test_a_build_directory_path_blocks(self):
        issues = problems([], {".build/x.o": 1}, deny_dirs=guard.DEFAULT_DENY_PATHS)
        self.assertEqual(len(issues), 1)
        self.assertIn("paths under build/output dirs", issues[0])
        self.assertIn(".build/x.o", issues[0])

    def test_a_configured_deny_path_blocks(self):
        issues = problems([], {"vendor/generated.c": 1}, deny_dirs=("vendor/",))
        self.assertIn("vendor/generated.c", issues[0])

    def test_a_build_prefixed_file_is_not_a_build_directory(self):
        self.assertEqual(
            problems([], {"build.py": 1, "src/builder/x.c": 1}, deny_dirs=guard.DEFAULT_DENY_PATHS),
            [],
        )

    def test_every_violated_limit_is_reported(self):
        added = [".build/x.o", "big.bin"]
        issues = problems(added, {".build/x.o": 1, "big.bin": 9 * 1024 * 1024},
                          max_files=1, deny_dirs=guard.DEFAULT_DENY_PATHS)
        self.assertEqual(len(issues), 3)

    def test_a_long_list_is_summarized(self):
        added = [f"gen/f{i}.txt" for i in range(201)]
        issues = problems(added, {p: 1 for p in added}, max_files=0)
        self.assertIn(f"(+{201 - guard.REASON_PATHS} more)", issues[0])
        self.assertNotIn("gen/f200.txt", issues[0])


class DeniedTest(unittest.TestCase):
    def test_a_component_at_any_depth_matches(self):
        self.assertTrue(guard.denied("frontend/dist/app.js", ["dist/"]))
        self.assertTrue(guard.denied("DerivedData/x/y", guard.DEFAULT_DENY_PATHS))

    def test_a_single_segment_entry_needs_a_whole_component(self):
        self.assertFalse(guard.denied("src/distribution/x.c", ["dist"]))
        self.assertFalse(guard.denied("rebuild/x.c", ["build/"]))

    def test_a_multi_segment_entry_matches_that_path_and_below(self):
        self.assertTrue(guard.denied("out/gen/a.c", ["out/gen"]))
        self.assertTrue(guard.denied("out/gen", ["out/gen"]))
        self.assertFalse(guard.denied("out/generator/a.c", ["out/gen"]))

    def test_blank_entries_are_ignored(self):
        self.assertFalse(guard.denied("a/b", ["", "/"]))


if __name__ == "__main__":
    unittest.main()
