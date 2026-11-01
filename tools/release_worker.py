#!/usr/bin/env python3
"""Check a configured worker using preview bytes. Never certifies a release."""

import argparse
import base64
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import release_gpu as gpu
from release_lib import (
    ROOT, archive_name, comparison_proves_ancestor, extract_archive, install_signal_handlers, policy,
    probe_binary, read_json, reference_agreement, require, sha256, source_sha,
    worker_reservation, write_json,
)

SCHEMA = "zinc-worker-readiness-v1"


def profile_for(identifier):
    profiles = [p for p in policy()["gpu_profiles"] if p["id"] == identifier]
    require(len(profiles) == 1, "unknown worker profile")
    return profiles[0]


def preview_source(run):
    require(isinstance(run, dict), "preview run metadata is invalid")
    require(isinstance(run.get("repository"), dict) and run["repository"].get("full_name") == "zolotukhin/zinc", "preview repository differs")
    require(isinstance(run.get("path"), str) and run["path"].split("@")[0] == ".github/workflows/release-preview.yml", "not a packaging preview")
    require(run.get("event") in ("push", "workflow_dispatch") and run.get("head_branch") == "main", "preview is not trusted main")
    require(run.get("status") == "completed" and run.get("conclusion") == "success", "preview did not pass")
    require(type(run.get("id")) is int and run["id"] > 0, "preview run identity is missing")
    return source_sha(run.get("head_sha"))


def resolve_preview(profile, run, variables, expected_sha, comparison=None, preview_policy_digest=None):
    commit, validation_commit = preview_source(run), source_sha(expected_sha)
    if commit != validation_commit:
        # Reuse diagnostic inputs under current checks and the same policy.
        # The archive keeps its own source identity, even if engine code changed;
        # this does not assert equivalence to a build of validation_commit.
        require(comparison_proves_ancestor(comparison, commit), "preview source must be a verified ancestor of this workflow's main commit")
        require(preview_policy_digest == sha256(ROOT / "config/release/targets.json"), "preview policy differs; build a new packaging preview")
    labels = json.loads(variables.get(profile["runner_variable"], "null"))
    require(isinstance(labels, list) and labels and all(isinstance(label, str) and
            re.fullmatch(r"self-hosted|linux|x64|macos|arm64|zinc-[a-z0-9-]{1,60}", label, re.I) for label in labels),
            "worker assignment must use generic role labels")
    return {"commit": commit, "validation_commit": validation_commit, "version": f"0.1.0-dev.{run['id']}", "runner": labels,
            "target": profile["target"], "profile": profile["id"], "preview_run": str(run["id"]),
            "runner_variable": profile["runner_variable"]}


def github_json(path):
    request = urllib.request.Request("https://api.github.com/repos/zolotukhin/zinc/" + path,
                                     headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def policy_digest_from_response(response):
    require(isinstance(response, dict) and response.get("type") == "file" and
            response.get("path") == "config/release/targets.json" and response.get("encoding") == "base64" and
            isinstance(response.get("content"), str), "could not read the preview's frozen policy")
    try:
        data = base64.b64decode("".join(response["content"].split()), validate=True)
    except ValueError:
        raise ValueError("could not decode the preview's frozen policy") from None
    return hashlib.sha256(data).hexdigest()


def validation_identity(expected_sha):
    actual = source_sha(gpu.output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip())
    require(actual == source_sha(expected_sha), "validation checkout differs from the frozen workflow commit")
    require(not gpu.output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"]).strip(), "validation checkout is dirty")
    return actual


def resolve(args):
    require(re.fullmatch(r"[1-9][0-9]*", args.preview_run), "preview run must be a numeric ID")
    validation_commit = validation_identity(source_sha(os.environ["GITHUB_SHA"]))
    profile = profile_for(args.profile)
    run = github_json("actions/runs/" + args.preview_run)
    commit = preview_source(run)
    require(str(run.get("id")) == args.preview_run, "preview run identity differs")
    comparison, digest = None, None
    if commit != validation_commit:
        comparison = github_json("compare/" + commit + "..." + validation_commit + "?per_page=1")
        require(comparison_proves_ancestor(comparison, commit), "preview source must be a verified ancestor of this workflow's main commit")
        digest = policy_digest_from_response(github_json("contents/config/release/targets.json?ref=" + commit))
    result = resolve_preview(profile, run, json.loads(os.environ.get("RELEASE_VARIABLES", "{}")), validation_commit, comparison, digest)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in result.items():
            stream.write(key + "=" + (json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value) + "\n")
    print(json.dumps({key: result[key] for key in ("commit", "version", "profile")}))


def idle():
    checks = [subprocess.run(["pgrep", "-x", name], capture_output=True) for name in gpu_names()]
    require(all(item.returncode in (0, 1) for item in checks), "could not inspect GPU processes")
    require(all(item.returncode == 1 for item in checks), "GPU worker is busy; no process was interrupted")


def gpu_names():
    return ("zinc", "llama-server", "llama-cli", "llama-bench")


@contextlib.contextmanager
def reservation(path):
    with worker_reservation(path):
        idle()
        yield


def model_pins(profile, config):
    result = []
    for identifier in profile["models"]:
        expected = next(m for m in policy()["models"] if m["id"] == identifier)
        path = Path(config["models"][identifier]).expanduser().resolve()
        require(path.is_file() and path.stat().st_size == expected["size_bytes"] and sha256(path) == expected["sha256"],
                "model pin differs: " + identifier)
        result.append({"id": identifier, "sha256": expected["sha256"], "context": expected["context"]})
    return result


def check(args):
    profile = profile_for(args.profile)
    config_path = Path(os.environ["ZINC_RELEASE_WORKER_CONFIG"]).expanduser().resolve()
    require(ROOT not in config_path.parents, "worker configuration must be outside the checkout")
    config = gpu.worker_for_profile(read_json(config_path), profile["id"])
    require(type(config.get("device")) is int and config["device"] >= 0, "worker device index is missing")
    report = {"schema_version": SCHEMA, "release_qualification": False, "profile": profile["id"],
              "source_commit": source_sha(args.commit), "preview_version": args.version,
              "policy_sha256": sha256(ROOT / "config/release/targets.json"), "passed": False}
    stage = "validation identity"
    try:
        actual = validation_identity(getattr(args, "validation_commit", None))
        report["validation_source_commit"] = actual
        report["validation_helpers_sha256"] = {name: sha256(ROOT / "tools" / name)
                                                for name in ("release_worker.py", "release_gpu.py", "release_lib.py", "release.py")}
        stage = "reservation"
        with reservation(Path(config["lock_directory"]).expanduser().resolve()), tempfile.TemporaryDirectory(prefix="zinc-worker-check-") as temporary:
            work = Path(temporary)
            stage = "package"
            archive = Path(args.directory) / archive_name(args.version, profile["target"])
            tree = extract_archive(archive, work / "unpack", args.version, profile["target"], args.commit)
            report["archive_sha256"] = sha256(archive)
            probe_binary(tree, args.version, profile["target"], args.commit)
            env = gpu.packaged_environment()
            stage = "hardware"
            idle()
            report["hardware"] = gpu.hardware_evidence(gpu.output([str(tree / "bin/zinc"), "--check", "-d", str(config["device"])], env, timeout=360), profile["family"], config["device"])
            idle()
            stage = "reference"
            reference, checkout = Path(config["reference_bin"]), Path(config["reference_source"])
            backend = next(t["backend"] for t in policy()["targets"] if t["id"] == profile["target"])
            reference_backend = gpu.reference_device_backend(config.get("reference_args"), backend)
            expected = policy()["reference"]["commit"]
            require(gpu.output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).strip() == expected, "reference source is not pinned")
            require(not gpu.output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"]).strip(), "reference source is dirty")
            reference_snapshot = gpu.reference_artifacts(reference)
            require(expected[:7] in gpu.output([str(reference / "llama-server"), "--version"], gpu.reference_environment(reference, env)), "reference executable differs from policy")
            report["reference"] = {"commit": expected, "device_backend": reference_backend,
                                   "generation": policy()["reference"]["generation"], **reference_snapshot}
            stage = "model_pins"
            report["models"] = model_pins(profile, config)
            if args.exercise_models:
                if profile["target"].endswith("vulkan"):
                    stage = "kernel_reference"
                    idle()
                    gpu.kernel_reference(tree, reference, checkout, config["device"], work, env)
                for model in report["models"]:
                    idle()
                    model_work = work / model["id"]
                    model_work.mkdir()
                    stage = model["id"]
                    path = Path(config["models"][model["id"]])
                    print("Checking pinned model: " + model["id"], flush=True)
                    gpu.tokenizer_parity(tree, reference, path, model_work, env)
                    samples = gpu.inference_parity(tree, reference, path, model["context"], config["device"], config["reference_args"], model_work, env)
                    if model.get("sliding_window"):
                        model["sliding_window_reference"] = gpu.sliding_window_reference(tree, reference, path, model["context"],
                            config["device"], config["reference_args"], model_work, env, model["sliding_window"])
                    if profile["target"] == "linux-x86_64-rocm" and model["id"] == "gemma4-31b-q4k-m":
                        model["bounded_prefill"] = gpu.bounded_prefill_parity(tree, path, model["context"], config["device"], model_work, env, samples)
                    browser_chat = next(t for t in policy()["targets"] if t["id"] == profile["target"])["interfaces"]["browser_chat"]
                    check_prefill = profile["target"].endswith("vulkan") and model["id"] == "qwen36-35b-a3b-q4k-xl"
                    prefill = gpu.api_checks(tree, path, model["context"], config["device"], model_work, env, browser_chat, samples, check_prefill)
                    if check_prefill:
                        require(prefill and prefill.get("passed") is True, "prefill cancellation evidence is missing")
                        model["prefill_cancellation"] = prefill
                    model["agreement"] = reference_agreement([sample["comparison"] for _, sample in samples])
                    model["api_passed"] = True
                    idle()
            stage = "cleanup"
            idle()
            require(gpu.reference_artifacts(reference) == reference_snapshot, "reference artifacts changed during GPU validation")
        report.update(passed=True, models_exercised=args.exercise_models, cleanup="no GPU process remains")
    except BaseException as error:
        report.update(passed=False, failed_stage=stage, error=str(error) if type(error) is ValueError else type(error).__name__)
        raise
    finally:
        write_json(args.output, report)
    print("Worker readiness passed; this report cannot certify a release")


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("resolve")
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--preview-run", required=True)
    run = commands.add_parser("check")
    for name in ("profile", "directory", "version", "commit", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--exercise-models", action="store_true")
    run.add_argument("--validation-commit", required=True)
    args = parser.parse_args()
    try:
        (resolve if args.command == "resolve" else check)(args)
    except Exception as error:
        reason = str(error) if type(error) is ValueError else type(error).__name__ + ": inspect private worker configuration"
        raise SystemExit("Worker readiness failed: " + reason)


if __name__ == "__main__":
    main()
