#!/usr/bin/env python3
"""Run every selected profile for one worker role, owning and reaping each child."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import release_gpu as gpu
from release_lib import ROOT, install_signal_handlers, release_policy, require


def check_group(role, directory, output):
    profiles = [p for p in release_policy()["gpu_profiles"] if p["runner_variable"] == role]
    require(profiles, "unknown release worker role")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    # A previous report must never silently join this qualification attempt.
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="zinc-gpu-group-") as temporary:
        work = Path(temporary)
        for profile in profiles:
            print("Checking packaged profile: " + profile["id"], flush=True)
            command = [sys.executable, str(ROOT / "tools/release_gpu.py"), "--directory", str(directory),
                       "--profile", profile["id"], "--output", str(output / (profile["id"] + ".json"))]
            with gpu.process(command, work / "private-validation.log", work, dict(os.environ)) as child:
                try:
                    result = child.wait(timeout=150 * 60)
                except subprocess.TimeoutExpired:
                    raise ValueError("packaged GPU profile timed out after 150 minutes: " + profile["id"]) from None
                require(result == 0, "packaged GPU profile failed: " + profile["id"])


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        check_group(args.role, args.directory, args.output)
    except Exception as error:
        detail = str(error) if type(error) is ValueError else type(error).__name__
        raise SystemExit("Packaged worker group failed: " + detail)


if __name__ == "__main__":
    main()
