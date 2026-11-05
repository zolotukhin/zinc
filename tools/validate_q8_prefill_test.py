import unittest

from validate_q8_prefill import validated_logits


def record(status="ok", maximum="0.000100", nonfinite=0, tolerance="0.001000", count=33):
    return (f"warning: prefillBatched validate[{status}]: last-token logits max_abs_diff={maximum} "
            f"at idx=7 (ref=1.0000 batched=1.0001) nonfinite={nonfinite} tol={tolerance} n_tokens={count}")


class LogitRegression(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
