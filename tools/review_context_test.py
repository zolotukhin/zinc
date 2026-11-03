"""Regression tests for large evidence artifacts and privileged review input."""

from pathlib import Path
import random
import unittest

from review_context import build_review_context


def patch(path, body):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}\n"


class ReviewContextTests(unittest.TestCase):
    def test_large_published_evidence_does_not_hide_source_or_tests(self):
        evidence = patch("site/public/zinc/validation/measurement.json", "+0.123456789\n" * 20000)
        source = patch("src/compute/dmmv.zig", "+select_only_loaded_shapes();")
        tests = patch("src/compute/projection_specializations.zig", '+test "matrix input widths" {}')
        context = build_review_context(evidence + source + tests)
        self.assertLessEqual(len(context), 120000)
        self.assertIn(source, context)
        self.assertIn(tests, context)
        self.assertLess(context.index(source), context.index("diff --git a/site/"))
        self.assertIn("measurement.json [generated/evidence]", context)
        self.assertIn("Patch truncated for review", context)

    def test_large_source_files_share_context_with_later_tests(self):
        diff = patch("src/first.zig", "+first\n" * 1000)
        diff += patch("src/second.zig", "+second\n" * 1000)
        diff += patch("tests/last.test.ts", "+assert(recoveryWorks);\n" * 1000)
        context = build_review_context(diff, limit=3000, per_file_limit=1500)
        self.assertLessEqual(len(context), 3000)
        for name in ("src/first.zig", "src/second.zig", "tests/last.test.ts"):
            self.assertIn("diff --git a/" + name, context)
        self.assertIn("+assert(recoveryWorks);", context)

    def test_source_is_not_needlessly_clipped_when_all_code_fits(self):
        source = patch("src/large.zig", "+source line\n" * 6000)
        tests = patch("tests/small.test.ts", "+assert(allPaths);")
        evidence = patch("site/public/zinc/validation/run.json", "+data\n" * 30000)
        context = build_review_context(evidence + source + tests)
        self.assertLessEqual(len(context), 120000)
        self.assertIn(source, context)
        self.assertIn(tests, context)

    def test_configuration_is_reviewed_before_prose_and_generated_data(self):
        diff = patch("docs/plan.md", "+prose")
        diff += patch("config/release/targets.json", '+{"release":true}')
        diff += patch(".github/workflows/release.yml", "+permissions: read-all")
        diff += patch("AGENTS.md", "+instructions")
        diff += patch("site/src/data/zinc-performance.json", "+measurements")
        context = build_review_context(diff)
        start = context.index("Patches are ordered")
        patches = context[start:]
        self.assertLess(patches.index("diff --git a/config/"), patches.index("diff --git a/docs/"))
        self.assertLess(patches.index("diff --git a/.github/"), patches.index("diff --git a/docs/"))
        self.assertLess(patches.index("diff --git a/AGENTS.md"), patches.index("diff --git a/docs/"))
        self.assertLess(patches.index("diff --git a/docs/"), patches.index("diff --git a/site/"))

    def test_rename_binary_and_quoted_paths_remain_visible(self):
        renamed = 'diff --git "a/old name.zig" "b/new name.zig"\nsimilarity index 100%\nrename from old name.zig\nrename to new name.zig\n'
        binary = "diff --git a/assets/image.png b/assets/image.png\nBinary files differ\n"
        context = build_review_context(renamed + binary)
        self.assertIn("new name.zig [code/config]", context)
        self.assertIn(renamed, context)
        self.assertIn(binary, context)

    def test_unfamiliar_headers_and_unstructured_input_are_not_silently_lost(self):
        malformed = 'diff --git "unclosed\n+important change\n'
        self.assertIn(malformed, build_review_context(malformed))
        context = build_review_context("raw content " * 1000, limit=600)
        self.assertLessEqual(len(context), 600)
        self.assertIn("no changed-file inventory", context)
        self.assertIn("Review input truncated", context)

    def test_huge_inventory_is_bounded_and_explicit(self):
        diff = "".join(patch(f"src/file-{n}.zig", "+value") for n in range(1000))
        context = build_review_context(diff, limit=2000)
        self.assertLessEqual(len(context), 2000)
        self.assertIn("1000 files", context)
        self.assertIn("Inventory truncated", context)
        self.assertIn("patches may be truncated or omitted", context)
        inventory = context.split("Patches are ordered", 1)[0]
        self.assertTrue(all(line.endswith("]") for line in inventory.splitlines() if line.startswith("- ")))

    def test_mixed_priority_groups_never_exceed_the_whole_input_budget(self):
        rng = random.Random(28)
        for limit in (512, 768, 2048, 8192):
            for cap in (128, 512, 120000):
                for _ in range(10):
                    paths = ("src/main.zig", "tests/recovery.test.ts", "docs/plan.md",
                             "site/public/zinc/validation/run.json")
                    diff = "".join(patch(path, "+data\n" * rng.randrange(1, 1500)) for path in paths)
                    context = build_review_context(diff, limit=limit, per_file_limit=cap)
                    self.assertLessEqual(len(context), limit)
                    self.assertIn("patches may be truncated or omitted", context)

    def test_unusable_limits_fail_explicitly(self):
        for limit in (0, 511):
            with self.assertRaises(ValueError):
                build_review_context("", limit=limit)
        with self.assertRaises(ValueError):
            build_review_context("", per_file_limit=0)

    def test_privileged_workflow_loads_helper_from_trusted_base_only(self):
        workflow = (Path(__file__).resolve().parent.parent / ".github/workflows/pr-review.yml").read_text()
        checkout = workflow.split("- name: Review PR with GLM-5.2", 1)[0]
        self.assertIn("repository: " + "$" + "{{ github.repository }}", checkout)
        self.assertIn("ref: " + "$" + "{{ github.event.repository.default_branch }}", checkout)
        self.assertIn("path: review-base", checkout)
        self.assertIn("persist-credentials: false", checkout)
        self.assertNotIn("head.sha", checkout)
        self.assertNotIn("pull_request.base", checkout)
        self.assertIn("python3 -I", workflow)
        self.assertIn("working-directory: " + "$" + "{{ runner.temp }}", workflow)
        self.assertIn("/review-base/tools/review_context.py", workflow)
        self.assertIn("build_review_context(diff)", workflow)
        self.assertNotIn("diff[:diff_limit]", workflow)


if __name__ == "__main__":
    unittest.main()
