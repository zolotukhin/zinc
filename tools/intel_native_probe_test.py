"""CPU fixtures: no model, Intel runtime, or GPU is opened."""
import copy
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import intel_native_probe as native


class NativeProbeTest(unittest.TestCase):
    @staticmethod
    def fixture():
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


class NativeProbeRunnerTest(unittest.TestCase):
    """Mock every GPU entrypoint; only the shared owner test runs a CPU child."""

    def execute(self, result=None, *, raw=None, exit_code=0, failure=None, busy=False,
                changed_binary=False, observation_error=None, startup_error=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); probe = root / "fixture-probe"; probe.write_bytes(b"probe fixture")
            output = root / "result.json"
            args = SimpleNamespace(probe=probe, output=output, reservation=root / "worker-reservation")
            result = NativeProbeTest.fixture() if result is None else result
            raw = json.dumps(result).encode() if raw is None else raw
            private = (str(root / "operator") + ": private driver diagnostics").encode()
            if startup_error is not None: raw, private = b"", b""
            active, events = [], []
            child = Mock(pid=123)
            child.poll.return_value = None
            def wait(timeout):
                self.assertEqual(timeout, 180)
                if failure is not None: raise failure
                child.poll.return_value = exit_code
                return exit_code
            child.wait.side_effect = wait
            @contextlib.contextmanager
            def reservation(path):
                self.assertEqual(path, args.reservation)
                active.append("reservation"); events.append("lock")
                try: yield
                finally:
                    self.assertEqual(active, ["reservation"])
                    active.clear(); events.append("unlock")
            @contextlib.contextmanager
            def process(command, log, cwd, env, *, stdout):
                self.assertEqual(active, ["reservation"])
                self.assertEqual(command, [str(probe)])
                self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
                self.assertFalse(json.loads(output.read_text())["passed"])
                stdout.write(raw); log.write_bytes(private)
                if startup_error is not None: raise startup_error
                active.append("child"); events.append("start")
                try: yield child
                finally:
                    active.pop(); child.poll.return_value = exit_code; events.append("reap")
                    if changed_binary: probe.write_bytes(b"different fixture")
            previous_umask = os.umask(0o027)
            original_sha256 = native.sha256
            def observed_sha256(path):
                if observation_error is not None and Path(path).name == "stdout.log": raise observation_error
                return original_sha256(path)
            try:
                with patch.object(native, "worker_reservation", side_effect=reservation), \
                        patch.object(native, "idle", side_effect=ValueError("busy worker") if busy else None), \
                        patch.object(native.gpu, "process", side_effect=process) as launch, \
                        patch.object(native, "sha256", side_effect=observed_sha256), \
                        patch.dict(os.environ, {"ONEAPI_DEVICE_SELECTOR": "level_zero:gpu"}):
                    caught = None
                    try: native.run_probe(args)
                    except BaseException as error: caught = error
                    self.assertEqual(os.umask(0o027), 0o027)
                    saved = json.loads(output.read_text())
                    self.assertEqual(active, [])
                    self.assertEqual(events, ["lock", "unlock"] if busy or startup_error is not None else ["lock", "start", "reap", "unlock"])
                    self.assertNotIn(str(root), json.dumps(saved))
                    self.assertNotIn("private driver diagnostics", json.dumps(saved))
                    self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(len(list(root.glob("intel-native-probe-*"))), 1)
                    logs = next(root.glob("intel-native-probe-*"))
                    self.assertEqual(logs.stat().st_mode & 0o777, 0o700)
                    if busy: launch.assert_not_called()
                    else:
                        if startup_error is None: self.assertTrue(saved["owned_process_reaped"])
                        else: self.assertNotIn("owned_process_reaped", saved)
                        self.assertEqual((logs / "stdout.log").read_bytes(), raw)
                        self.assertEqual((logs / "stderr.log").read_bytes(), private)
                        if observation_error is None:
                            for name, data in (("stdout", raw), ("stderr", private)):
                                self.assertEqual(saved["diagnostics"][name], {
                                    "sha256": original_sha256(logs / (name + ".log")), "bytes": len(data)})
                    return saved, caught
            finally: os.umask(previous_umask)

    def test_success_requires_the_complete_probe_unchanged_binary_and_cleanup(self):
        result = {**NativeProbeTest.fixture(), "operator_path": "private-extra-field"}
        report, error = self.execute(result)
        self.assertIsNone(error)
        self.assertTrue(report["passed"])
        self.assertEqual(report["validation_stage"], "complete")
        self.assertEqual(report["probe"], NativeProbeTest.fixture())
        self.assertFalse(report["release_qualification"])
        self.assertFalse(report["native_zinc_backend"])
        report, error = self.execute(changed_binary=True)
        self.assertIsInstance(error, ValueError)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error_type"], "ValueError")

    def test_nonzero_incomplete_and_malformed_results_preserve_failed_reports_and_private_logs(self):
        for options in ({"exit_code": 1}, {"result": {**NativeProbeTest.fixture(), "cases": 77}},
                        {"raw": b"not JSON"}, {"raw": b"\xff"}, {"raw": b"[]"},
                        {"raw": json.dumps(NativeProbeTest.fixture()).encode("utf-16")},
                        {"raw": b"x" * (native.MAX_PROBE_JSON_BYTES + 1)}):
            with self.subTest(options=list(options)):
                report, error = self.execute(**options)
                self.assertIsInstance(error, ValueError)
                self.assertFalse(report["passed"])
                self.assertEqual(report["validation_stage"], "result")
                if options.get("exit_code") == 1:
                    self.assertEqual(report["exit_code"], 1)
                    self.assertEqual(report["probe"], NativeProbeTest.fixture())

    def test_nonfinite_failures_keep_json_safe_fields_without_passing_the_probe(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            report, error = self.execute({**NativeProbeTest.fixture(), "max_absolute_error": value})
            self.assertIsInstance(error, ValueError)
            self.assertFalse(report["passed"])
            self.assertIsNone(report["probe"]["max_absolute_error"])
            json.dumps(report, allow_nan=False)

    def test_timeout_and_cancellation_preserve_the_primary_failure_after_reap(self):
        for failure in (subprocess.TimeoutExpired("private command", 180), KeyboardInterrupt(), SystemExit(129)):
            with self.subTest(failure=type(failure).__name__):
                report, error = self.execute(failure=failure)
                self.assertIs(error, failure)
                self.assertFalse(report["passed"])
                self.assertEqual(report["validation_stage"], "running")
                self.assertEqual(report["error_type"], type(failure).__name__)
                self.assertEqual(report["probe"], NativeProbeTest.fixture())
                self.assertNotIn("private command", json.dumps(report))

    def test_diagnostic_observation_failure_does_not_replace_an_earlier_failure(self):
        failure = subprocess.TimeoutExpired("private command", 180)
        report, error = self.execute(failure=failure, observation_error=OSError("private log path"))
        self.assertIs(error, failure)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error_type"], "TimeoutExpired")
        self.assertEqual(report["observation_error"], "OSError")
        report, error = self.execute(observation_error=OSError("private log path"))
        self.assertIsInstance(error, ValueError)
        self.assertFalse(report["passed"])
        self.assertEqual(report["validation_stage"], "observations")
        self.assertNotIn("private log path", json.dumps(report))

    def test_startup_error_keeps_a_failed_report_and_never_claims_a_started_child(self):
        failure = OSError("private executable path")
        report, error = self.execute(startup_error=failure)
        self.assertIs(error, failure)
        self.assertFalse(report["passed"])
        self.assertEqual(report["validation_stage"], "startup")
        self.assertNotIn("owned_process_reaped", report)

    def test_cli_unwinds_the_owned_runner_on_ssh_disconnect(self):
        handlers = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); probe = root / "fixture"; probe.touch()
            arguments = ["intel_native_probe.py", "--probe", str(probe), "--output", str(root / "result"),
                         "--reservation", str(root / "reservation")]
            def run(args): handlers[native.signal.SIGHUP](native.signal.SIGHUP, None)
            with patch.object(native.sys, "argv", arguments), patch.object(native.sys, "platform", "linux"), \
                    patch.object(native, "install_signal_handlers"), patch.object(native, "run_probe", side_effect=run), \
                    patch.object(native.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})):
                with self.assertRaises(SystemExit) as caught: native.main()
                self.assertEqual(caught.exception.code, 128 + native.signal.SIGHUP)

    def test_busy_worker_records_the_failure_without_starting_a_probe(self):
        report, error = self.execute(busy=True)
        self.assertIsInstance(error, ValueError)
        self.assertFalse(report["passed"])
        self.assertEqual(report["validation_stage"], "reservation")
        self.assertEqual(report["diagnostics"], {})

    def test_retry_needs_a_fresh_output_path_and_preserves_prior_success_or_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "result.json"
            args = SimpleNamespace(output=output, probe=root / "fixture", reservation=root / "reservation")
            previous_umask = os.umask(0o027)
            try:
                for passed in (True, False):
                    old = json.dumps({"passed": passed, "retained": "earlier attempt"})
                    output.write_text(old)
                    with patch.object(native, "worker_reservation") as reserve, \
                            patch.object(native.gpu, "process") as launch:
                        with self.assertRaises(FileExistsError): native.run_probe(args)
                        self.assertEqual(os.umask(0o027), 0o027)
                        reserve.assert_not_called(); launch.assert_not_called()
                    self.assertEqual(output.read_text(), old)
            finally: os.umask(previous_umask)

    def test_shared_process_owner_can_separate_machine_json_from_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stdout = root / "stdout"; stderr = root / "stderr"
            code = "import sys; print('result'); print('diagnostic', file=sys.stderr)"
            with stdout.open("wb") as stream:
                with native.gpu.process([sys.executable, "-c", code], stderr, root, dict(os.environ), stdout=stream) as child:
                    self.assertEqual(child.wait(timeout=10), 0)
            self.assertIsNotNone(child.poll())
            self.assertEqual(stdout.read_text(), "result\n")
            self.assertEqual(stderr.read_text(), "diagnostic\n")


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
