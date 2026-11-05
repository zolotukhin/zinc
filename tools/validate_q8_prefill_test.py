from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import validate_q8_prefill as validator
from validate_q8_prefill import PREFILL_TOKEN_COUNTS, validated_logits


def record(status="ok", maximum="0.000100", nonfinite=0, tolerance="0.001000", count=33):
    return (f"warning: prefillBatched validate[{status}]: last-token logits max_abs_diff={maximum} "
            f"at idx=7 (ref=1.0000 batched=1.0001) nonfinite={nonfinite} tol={tolerance} n_tokens={count}")


class LogitRegression(unittest.TestCase):
    def test_fixture_straddles_tile_and_both_intel_chunk_boundaries(self):
        self.assertEqual(len(PREFILL_TOKEN_COUNTS), len(set(PREFILL_TOKEN_COUNTS)))
        for boundary in (32, 96, 384):
            self.assertTrue({boundary - 1, boundary, boundary + 1}.issubset(PREFILL_TOKEN_COUNTS))
        self.assertTrue({1, 2}.issubset(PREFILL_TOKEN_COUNTS))

    def test_full_vocabulary_comparison_accepts_only_the_intended_boundary(self):
        self.assertEqual(validated_logits(record(), 33)["prompt_tokens"], 33)
        self.assertEqual(validated_logits(record(maximum="0"), 33)["maximum_absolute_logit_difference"], 0)

    def test_serial_fallback_and_duplicate_comparisons_cannot_pass(self):
        for log in ("Prefill complete", record() + "\n" + record()):
            with self.assertRaises(ValueError):
                validated_logits(log, 33)

    def test_mismatched_shape_nonfinite_and_relaxed_tolerance_fail(self):
        for log in (record(count=32), record(status="exceeded"), record(nonfinite=1),
                    record(maximum="nan"), record(maximum="inf"), record(maximum="-0.1"),
                    record(maximum="0.001001"), record(tolerance="nan"), record(tolerance="0.01")):
            with self.subTest(log=log), self.assertRaises(ValueError):
                validated_logits(log, 33)


class FailureReport(unittest.TestCase):
    def run_failing_fixture(self, failure, phase="case"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.touch()
            worker = root / "worker.json"
            worker.write_text(json.dumps({"models": {"qwen3-0.6b-q8-0": str(model)},
                                          "device": 0, "lock_directory": str(root / "lock")}))
            output = root / "report.json"

            @contextmanager
            def fake_process(command, log, work, env):
                count = len(command[command.index("--prompt") + 1].split())
                log.write_text("KV cache: f32\n" + record(count=count, status="exceeded" if phase == "case" and count == 2 else "ok"))

                def wait(timeout):
                    if phase == "external" and count == 2:
                        raise failure
                    if phase == "case" and count == 2 and not isinstance(failure, ValueError):
                        raise failure
                    return 0

                yield SimpleNamespace(wait=wait)

            @contextmanager
            def reservation():
                yield
                if phase == "cleanup":
                    raise failure

            fixture_policy = {"models": [{"id": "qwen3-0.6b-q8-0", "size_bytes": 0,
                                           "sha256": "0" * 64, "context": 2048}]}
            with patch.object(sys, "argv", ["validate", "--zinc-tree", str(root), "--worker-config", str(worker),
                                             "--output", str(output)]), \
                    patch.object(sys, "platform", "linux"), \
                    patch.object(validator, "install_signal_handlers"), \
                    patch.object(validator.signal, "signal"), \
                    patch.object(validator, "policy", return_value=fixture_policy), \
                    patch.object(validator, "sha256", return_value="0" * 64), \
                    patch.object(validator, "PREFILL_TOKEN_COUNTS", (1, 2, 3)), \
                    patch.object(validator, "worker_reservation", side_effect=lambda _: reservation()), \
                    patch.object(validator, "idle", side_effect=failure if phase == "setup" else None), \
                    patch.object(validator.gpu, "output", return_value=""), \
                    patch.object(validator.gpu, "token_ids", side_effect=[[0], [0, 1], [0, 1, 2]]), \
                    patch.object(validator.gpu, "hardware_evidence", return_value={"gpu": "fixture"}), \
                    patch.object(validator.gpu, "process", side_effect=fake_process):
                with self.assertRaises(type(failure)):
                    validator.main()
            return json.loads(output.read_text())

    def test_mid_loop_failure_preserves_passed_cases_and_identifies_boundary(self):
        report = self.run_failing_fixture(ValueError())
        self.assertFalse(report["passed"])
        self.assertEqual([case["prompt_tokens"] for case in report["cases"]], [1])
        self.assertEqual(report["failure"], {
            "prompt_tokens": 2, "error_type": "Q8RegressionError",
            "reason": "Q8 batched prefill differs from the serial full-vocabulary reference",
        })

    def test_process_failure_does_not_publish_private_command_or_path(self):
        report = self.run_failing_fixture(subprocess.TimeoutExpired(["/private/worker/bin/zinc"], 360))
        self.assertEqual(report["failure"], {
            "prompt_tokens": 2, "error_type": "TimeoutExpired", "reason": "TimeoutExpired",
        })
        self.assertNotIn("/private/worker", json.dumps(report))

    def test_generic_value_error_does_not_publish_private_message(self):
        report = self.run_failing_fixture(ValueError("/private/worker/command"), phase="external")
        self.assertEqual(report["failure"], {
            "prompt_tokens": 2, "error_type": "ValueError", "reason": "ValueError",
        })
        self.assertNotIn("/private/worker", json.dumps(report))

    def test_interrupt_is_explicit_and_keeps_validation_incomplete(self):
        for error in (KeyboardInterrupt(), SystemExit(130)):
            with self.subTest(error=type(error).__name__):
                report = self.run_failing_fixture(error, phase="external")
                self.assertFalse(report["passed"])
                self.assertTrue(report["interrupted"])
                self.assertEqual([case["prompt_tokens"] for case in report["cases"]], [1])
                self.assertEqual(report["failure"]["prompt_tokens"], 2)
                self.assertEqual(report["failure"]["error_type"], type(error).__name__)

    def test_setup_failure_has_no_fabricated_token_boundary(self):
        report = self.run_failing_fixture(OSError("/private/worker/lock"), phase="setup")
        self.assertFalse(report["passed"])
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["failure"], {
            "prompt_tokens": None, "error_type": "OSError", "reason": "OSError",
        })

    def test_reservation_cleanup_failure_cannot_leave_a_successful_report(self):
        report = self.run_failing_fixture(OSError("/private/worker/lock"), phase="cleanup")
        self.assertFalse(report["passed"])
        self.assertNotIn("cleanup", report)
        self.assertEqual([case["prompt_tokens"] for case in report["cases"]], [1, 2, 3])
        self.assertEqual(report["failure"], {
            "prompt_tokens": None, "error_type": "OSError", "reason": "OSError",
        })


if __name__ == "__main__":
    unittest.main()
