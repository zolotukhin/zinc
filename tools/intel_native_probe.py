#!/usr/bin/env python3
"""Remote-only Level Zero/SYCL bring-up. Does not qualify a ZINC backend."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from release_lib import install_signal_handlers, require, sha256, worker_reservation, write_json

WIDTHS = (1, 31, 32, 63, 64, 65, 255, 256, 257, 1023, 1024, 65537, 1048583)
EXPECTED_CASES = len(WIDTHS) * 2 * 3
EXPECTED_ELEMENTS = sum(WIDTHS) * 2 * 2 * 3


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
    for name in ("zinc", "llama-server", "llama-cli", "llama-bench", "zinc-sycl-probe"):
        result = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        require(result.returncode == 1, "GPU worker is busy or its processes cannot be inspected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "run only on the remote Linux GPU worker")
    require(args.probe.is_absolute() and args.probe.is_file(), "probe must be an absolute executable path")
    install_signal_handlers()
    # Reuse the worker's existing reservation path across inference harnesses.
    with worker_reservation(args.reservation):
        idle()
        environment = dict(os.environ)
        environment.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
        process = subprocess.Popen([str(args.probe)], env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=180)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        require(process.returncode == 0, "native probe process failed; inspect private worker diagnostics")
        probe = validate_probe(json.loads(stdout))
        report = {"schema_version": 1, "passed": True, "release_qualification": False,
                  "native_zinc_backend": False, "kernel": platform.release(),
                  "probe_sha256": sha256(args.probe), "probe": probe}
        write_json(args.output, report)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
