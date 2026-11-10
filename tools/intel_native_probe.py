#!/usr/bin/env python3
"""Remote-only Level Zero/SYCL bring-up. Does not qualify a ZINC backend."""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile

import release_gpu as gpu
from release_lib import install_signal_handlers, require, sha256, worker_reservation, write_json

WIDTHS = (1, 31, 32, 63, 64, 65, 255, 256, 257, 1023, 1024, 65537, 1048583)
EXPECTED_CASES = len(WIDTHS) * 2 * 3
EXPECTED_ELEMENTS = sum(WIDTHS) * 2 * 2 * 3
BUSY_PROCESSES = ("zinc", "llama-server", "llama-cli", "llama-bench", "zinc-sycl-probe", "icpx", "cmake")
PROBE_FIELDS = ("passed", "backend", "device_type", "vendor_id", "device_name", "driver_version",
                "cases", "elements_checked", "guards_checked", "max_absolute_error")
MAX_PROBE_JSON_BYTES = 64 * 1024


def validate_probe(result):
    require(isinstance(result, dict) and result.get("passed") is True, "native probe failed")
    require(result.get("backend") == "level_zero" and result.get("device_type") == "gpu"
            and type(result.get("vendor_id")) is int and result["vendor_id"] == 0x8086,
            "probe did not execute on an Intel Level Zero GPU")
    for name in ("device_name", "driver_version"):
        require(isinstance(result.get(name), str) and result[name].strip(), "missing native device identity")
    for name, expected in (("cases", EXPECTED_CASES), ("elements_checked", EXPECTED_ELEMENTS),
                           ("guards_checked", EXPECTED_CASES * 32 * 4)):
        require(type(result.get(name)) is int and result[name] == expected, "incomplete native probe: " + name)
    error = result.get("max_absolute_error")
    require(type(error) in (int, float) and math.isfinite(error) and error == 0,
            "native probe arithmetic did not match the exact reference")
    return result


def idle():
    for name in BUSY_PROCESSES:
        result = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        require(result.returncode == 1, "GPU worker is busy or its processes cannot be inspected")
    nodes = list(Path("/dev/dri").glob("renderD*"))
    require(nodes and shutil.which("fuser"), "render-node inspection is unavailable")
    clients = subprocess.run(["fuser", *map(str, nodes)], capture_output=True, text=True)
    require(clients.returncode == 1 and not clients.stderr.strip(), "GPU worker is busy or render-node inspection is incomplete")


def record_probe_result(path, report):
    with path.open("rb") as stream:
        data = stream.read(MAX_PROBE_JSON_BYTES + 1)
    result = None
    if len(data) <= MAX_PROBE_JSON_BYTES:
        try:
            result = json.loads(data.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeError) as error:
            report["result_decode_error"] = type(error).__name__
    else:
        report["result_decode_error"] = "OutputLimitExceeded"
    if isinstance(result, dict):
        # Only documented device/math fields enter the shareable JSON.
        # Nonfinite numbers remain invalid and get a JSON-safe marker.
        report["probe"] = {key: (None if type(value) is float and not math.isfinite(value) else value)
                           for key, value in result.items() if key in PROBE_FIELDS
                           and type(value) in (str, int, float, bool, type(None))}
    elif "result_decode_error" not in report:
        report["result_decode_error"] = "UnexpectedResultType"
    return result


def run_probe(args):
    """Retain failed bring-up without publishing driver stderr or worker paths."""
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "passed": False, "release_qualification": False,
              "native_zinc_backend": False, "kernel": platform.release(), "validation_stage": "reservation",
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    # Never overwrite a passing or failed earlier attempt. Exclusive creation
    # also prevents simultaneous callers from sharing the same report file.
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    work, child = None, None
    try:
        work = Path(tempfile.mkdtemp(prefix="intel-native-probe-", dir=args.output.parent))
        report["probe_sha256"] = sha256(args.probe)
        environment = dict(os.environ)
        environment.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
        # Keep the reservation until the shared owner has stopped the complete
        # child group, including cancellation and descendants of a failed JIT.
        with worker_reservation(args.reservation):
            idle()
            report["validation_stage"] = "startup"
            write_json(args.output, report)
            with (work / "stdout.log").open("wb") as stdout:
                with gpu.process([str(args.probe)], work / "stderr.log", work, environment, stdout=stdout) as child:
                    report["validation_stage"] = "running"
                    write_json(args.output, report)
                    report["exit_code"] = child.wait(timeout=180)
            report["validation_stage"] = "result"
            # Files retain complete diagnostics even when output is oversized
            # or the process times out. Parse a bounded result only after reap.
            result = record_probe_result(work / "stdout.log", report)
            require(report["exit_code"] == 0, "native probe process failed; inspect private worker diagnostics")
            validate_probe(result)
            require(sha256(args.probe) == report["probe_sha256"], "native probe binary changed during validation")
        report.update(passed=True, validation_stage="complete")
    except BaseException as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        primary_error = "error_type" in report
        observation_error = None
        try:
            if child is not None:
                report["owned_process_reaped"] = child.poll() is not None
                require(report["owned_process_reaped"], "native probe child was not reaped")
            if work is not None:
                # A runtime may hang during teardown after printing its result.
                # Retain that math evidence without accepting the timed-out run.
                if (work / "stdout.log").is_file() and "probe" not in report and "result_decode_error" not in report:
                    record_probe_result(work / "stdout.log", report)
                report["diagnostics"] = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                                         for name in ("stdout", "stderr") if (path := work / (name + ".log")).is_file()}
        except Exception as error:
            observation_error = error
            report.update(passed=False, observation_error=type(error).__name__)
            if not primary_error:
                report.update(validation_stage="observations", error_type="ValueError")
        report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output, report)
        # A vanished /proc/log file cannot replace an earlier timeout or cancel.
        # An otherwise successful run still needs its cleanup/diagnostic proof.
        if observation_error is not None and not primary_error:
            raise ValueError("native probe cleanup or diagnostic observation unavailable") from observation_error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh report path; earlier attempts are preserved")
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "run only on the remote Linux GPU worker")
    require(args.probe.is_absolute() and args.probe.is_file(), "probe must be an absolute executable path")
    install_signal_handlers()
    def disconnected(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGHUP, disconnected)
    print(json.dumps(run_probe(args)))


if __name__ == "__main__":
    main()
