"""CPU fixtures: no model, Intel runtime, or GPU is opened."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
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
            with self.subTest(code=code), patch.object(native.subprocess, "run", return_value=Mock(returncode=code)), \
                    patch.object(native.Path, "glob", side_effect=AssertionError("must stop before DRI inspection")) as nodes, \
                    patch.object(native.shutil, "which", side_effect=AssertionError("must stop before fuser discovery")) as fuser:
                with self.assertRaisesRegex(ValueError, "worker is busy"):
                    native.idle()
                nodes.assert_not_called(); fuser.assert_not_called()

    def test_idle_worker_allows_the_probe(self):
        with patch.object(native.subprocess, "run", return_value=Mock(returncode=1, stderr="")) as run, \
                patch.object(native.Path, "glob", return_value=[native.Path("/dev/dri/renderD128")]), \
                patch.object(native.shutil, "which", return_value="/usr/bin/fuser"):
            native.idle()
            self.assertEqual(run.call_count, len(native.BUSY_PROCESSES) + 1)
            self.assertEqual(run.call_args.args[0][0], "fuser")

    def test_render_node_clients_and_incomplete_inspection_stop_the_probe(self):
        for code, stderr in ((0, ""), (2, ""), (1, "Cannot inspect process")):
            responses = [Mock(returncode=1)] * len(native.BUSY_PROCESSES) + [Mock(returncode=code, stderr=stderr)]
            with self.subTest(code=code, stderr=stderr), patch.object(native.subprocess, "run", side_effect=responses), \
                    patch.object(native.Path, "glob", return_value=[native.Path("/dev/dri/renderD128")]), \
                    patch.object(native.shutil, "which", return_value="/usr/bin/fuser"):
                with self.assertRaisesRegex(ValueError, "busy or render-node"):
                    native.idle()


class NativeProbeJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        source = root / "quote.cpp"
        source.write_text('#include "intel_probe_json.h"\n#include <iostream>\n#include <iterator>\n'
                          'int main() { const std::string value((std::istreambuf_iterator<char>(std::cin)), '
                          'std::istreambuf_iterator<char>()); std::cout << intel_probe_json_quote(value); }\n')
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("CPU JSON serialization fixtures require the build host's C++ compiler")
        cls.binary = root / "quote"
        subprocess.run([compiler, "-std=c++17", "-I", str(Path(__file__).parent), str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, timeout=60)

    def decode(self, value):
        result = subprocess.run([str(self.binary)], input=value, capture_output=True, check=True, timeout=10)
        return json.loads(result.stdout.decode("utf-8"))

    def test_all_ascii_controls_quotes_backslashes_and_del_round_trip(self):
        value = bytes(range(128))
        self.assertEqual(self.decode(value), value.decode("ascii"))

    def test_valid_unicode_device_names_are_preserved(self):
        for value in ("Intel GPU", "é — 中文 🧪", "\U0010ffff"):
            with self.subTest(value=value):
                self.assertEqual(self.decode(value.encode("utf-8")), value)

    def test_invalid_truncated_overlong_and_surrogate_utf8_remain_valid_json(self):
        for value in (b"\xff", b"\xc0\xaf", b"\xe2\x82", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\x80", b"\xf0(\x8c\xbc"):
            with self.subTest(value=value):
                result = self.decode(b"before" + value + b"after")
                self.assertTrue(result.startswith("before") and result.endswith("after"))
                self.assertIn("\ufffd", result)

if __name__ == "__main__":
    unittest.main()
