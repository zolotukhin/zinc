"""CPU acceptance guards: no compiler, model, runtime, or GPU is opened."""
import copy
import unittest
from unittest.mock import Mock, patch

import intel_native_probe as native


class NativeProbeTest(unittest.TestCase):
    def fixture(self):
        return {"passed": True, "backend": "level_zero", "device_type": "gpu", "vendor_id": 0x8086,
                "device_name": "Intel GPU", "driver_version": "test", "cases": 78,
                "elements_checked": 13406292, "guards_checked": 9984, "max_absolute_error": 0}

    def test_complete_exact_gpu_result(self):
        self.assertIs(native.validate_probe(self.fixture()).get("passed"), True)

    def test_cpu_opencl_wrong_vendor_and_incomplete_runs_cannot_pass(self):
        for key, values in {"passed": [False, 1, "true"], "backend": ["opencl", "vulkan", None],
                            "device_type": ["cpu", None], "vendor_id": [0x1002, True, "32902"],
                            "cases": [1, 77, 79, 78.0, True], "elements_checked": [0, 13406291],
                            "guards_checked": [0, 9983], "device_name": ["", " ", None],
                            "driver_version": ["", None],
                            "max_absolute_error": [None, True, -1, 0.00001, float("nan"), float("inf")]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    row = copy.deepcopy(self.fixture()); row[key] = value
                    native.validate_probe(row)

    def test_missing_fields_cannot_pass(self):
        for key in self.fixture():
            with self.subTest(key=key), self.assertRaises(ValueError):
                row = self.fixture(); del row[key]
                native.validate_probe(row)

    def test_busy_and_failed_process_inspection_stop_the_run(self):
        for code in (0, 2):
            with self.subTest(code=code), patch.object(native.subprocess, "run", return_value=Mock(returncode=code)):
                with self.assertRaisesRegex(ValueError, "worker is busy"):
                    native.idle()
        with patch.object(native.subprocess, "run", return_value=Mock(returncode=1)) as run:
            native.idle()
            self.assertEqual(run.call_count, 5)


if __name__ == "__main__":
    unittest.main()
