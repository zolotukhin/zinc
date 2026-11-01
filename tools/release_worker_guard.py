#!/usr/bin/env python3
"""Runner-admin hook. Install read-only outside the runner's writable directory."""

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

GPU_NAMES = {"zinc", "llama-server", "llama-cli", "llama-bench"}


def validate_context(policy, env):
    def require(ok, reason):
        if not ok:
            raise ValueError(reason)
    repository = policy.get("repository")
    require(isinstance(repository, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", repository), "invalid administrator policy")
    require(isinstance(policy.get("actors"), list) and policy["actors"] and
            all(isinstance(actor, str) and actor for actor in policy["actors"]), "invalid actor policy")
    require(env.get("GITHUB_REPOSITORY") == repository, "repository is not allowed")
    require(env.get("GITHUB_REF") == "refs/heads/main", "only main is allowed")
    require(env.get("GITHUB_ACTOR") in policy.get("actors", []), "actor is not allowed")
    require(re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")), "source identity is missing")
    require(re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")), "run identity is missing")
    workflows = policy.get("workflows", {})
    require(isinstance(workflows, dict), "invalid workflow policy")
    rule = workflows.get(env.get("GITHUB_WORKFLOW_REF"))
    require(isinstance(rule, dict), "workflow is not allowed")
    require(all(isinstance(rule.get(key), list) and rule[key] and all(isinstance(value, str) and value for value in rule[key])
                for key in ("events", "jobs")), "invalid job policy")
    require(env.get("GITHUB_EVENT_NAME") in rule.get("events", []), "event is not allowed")
    require(env.get("GITHUB_JOB") in rule.get("jobs", []), "job is not allowed")


def owned_gpu_processes(run_id):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or (entry / "comm").read_text().strip() not in GPU_NAMES:
                continue
            env = dict(item.split(b"=", 1) for item in (entry / "environ").read_bytes().split(b"\0") if b"=" in item)
            if env.get(b"GITHUB_RUN_ID") == run_id.encode():
                # Pin the start time as well as the PID before sending a signal.
                start = (entry / "stat").read_text().rsplit(")", 1)[1].split()[19]
                found.append((int(entry.name), start))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def cleanup(run_id):
    owned = owned_gpu_processes(run_id)
    for sig, delay in ((signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        for pid, start in owned:
            try:
                current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                if current == start:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass
        deadline = time.monotonic() + delay
        while owned_gpu_processes(run_id) and time.monotonic() < deadline:
            time.sleep(0.1)
    if owned_gpu_processes(run_id):
        raise ValueError("owned GPU processes did not stop")
    print("ZINC worker cleanup: no owned GPU process remains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    try:
        policy = json.loads(Path(args.policy).read_text())
        validate_context(policy, os.environ)
        if args.cleanup:
            cleanup(os.environ["GITHUB_RUN_ID"])
        else:
            print("ZINC worker accepted a trusted main release job")
    except Exception as error:
        # Do not echo received context, private paths or subprocess arguments.
        reason = str(error) if type(error) is ValueError else "administrator policy could not be checked"
        raise SystemExit("ZINC worker rejected job: " + reason)


if __name__ == "__main__":
    main()
