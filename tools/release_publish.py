#!/usr/bin/env python3
"""Review and promote an attested candidate. Never build or replace release bytes."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from release import aggregate, verify_candidate
from release_lib import ROOT, policy, read_json, require, sha256, source_sha, write_json

REPOSITORY = "zolotukhin/zinc"
WORKFLOW = ".github/workflows/release-candidate.yml"
PUBLIC_FILES = {"release-manifest.json", "validation.json", "release-policy.json", "SHA256SUMS.txt", "install.sh", "release-notes.md", "provenance.jsonl"}


def gh(args, body=None, optional=False):
    result = subprocess.run(["gh", *args], input=None if body is None else json.dumps(body), capture_output=True, text=True, timeout=300)
    if optional and result.returncode and "HTTP 404" in result.stderr:
        return None
    require(result.returncode == 0, "GitHub request failed: " + args[0])
    return result.stdout


def api(path, body=None, optional=False):
    args = ["api", "repos/" + REPOSITORY + "/" + path]
    if body is not None:
        args += ["--method", "POST", "--input", "-"]
    text = gh(args, body, optional)
    return None if text is None else json.loads(text)


def validate_run(run, run_id):
    require(str(run.get("id")) == str(run_id) and run.get("path") == WORKFLOW, "wrong candidate workflow/run")
    require(run.get("event") == "workflow_dispatch" and run.get("head_branch") == "main", "publish only manually frozen main candidates")
    require(run.get("head_repository", {}).get("full_name") == REPOSITORY, "fork candidate is forbidden")
    require(run.get("status") == "completed" and run.get("conclusion") == "success", "candidate workflow has not passed")
    return source_sha(run.get("head_sha"))


def validate_environment(environment):
    rules = environment.get("protection_rules", []) if environment else []
    require(any(rule.get("type") == "required_reviewers" and rule.get("reviewers") for rule in rules),
            "configure required reviewers on the release environment before publishing; an unprotected/missing environment cannot approve a release")


def validate_catalog(catalog, release_version):
    for model in policy()["models"]:
        entry = re.search(r'\.id = "' + re.escape(model["id"]) + r'",([\s\S]*?)(?=\n    \},)', catalog)
        require(entry is not None, "release model is missing from the shipped catalog")
        text = entry.group(1)
        require('/resolve/' + model["revision"] + '/' + model["file"] + '?download=true' in text
                and '.sha256 = "' + model["sha256"] + '"' in text, "shipped model pin differs from release policy")
        if model["id"] == "qwen3-0.6b-q8-0" and "-" not in release_version.split("+")[0]:
            require(".status = .supported" in text and not re.search(r"\.tested_profiles\s*=\s*&\.\{\s*\}", text),
                    "stable release requires the starter model to be qualified and promoted in the catalog; first run a nightly/RC, review the evidence, then build a new final-version candidate")


def verify_remote_assets(directory, tag):
    directory = Path(directory)
    files = {p.name for p in directory.iterdir()}
    with tempfile.TemporaryDirectory(prefix="zinc-upload-audit-") as temporary:
        gh(["release", "download", tag, "--repo", REPOSITORY, "--dir", temporary])
        require({p.name for p in Path(temporary).iterdir()} == files, "uploaded asset inventory mismatch")
        require(all(sha256(Path(temporary) / name) == sha256(directory / name) for name in files), "uploaded bytes differ; release is not modified")


def verify_bundle(directory, run_id, run):
    directory = Path(directory)
    workflow_sha = validate_run(run, run_id)
    data = verify_candidate(directory)
    require(data.get("candidate_run") == str(run_id) and data.get("workflow_sha") == workflow_sha, "candidate belongs to a different workflow invocation")
    expected = PUBLIC_FILES | {a["filename"] for a in data["artifacts"]}
    require({p.name for p in directory.iterdir()} == expected and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()), "unexpected candidate assets")
    require(sha256(directory / "release-policy.json") == data["policy_sha256"], "published policy differs from validated policy")
    lines = (directory / "SHA256SUMS.txt").read_text().splitlines()
    require(lines == [a["sha256"] + "  " + a["filename"] for a in data["artifacts"]], "candidate checksums differ from manifest")
    validation = read_json(directory / "validation.json")
    with tempfile.TemporaryDirectory(prefix="zinc-gate-") as temporary:
        reports = Path(temporary) / "reports"
        reports.mkdir()
        for index, report in enumerate(validation.get("profiles", [])):
            write_json(reports / f"{index}.json", report)
        # Aggregate into a separate directory, preserving original release bytes.
        gate = Path(temporary) / "candidate"
        gate.mkdir()
        for name in ["release-manifest.json", *[a["filename"] for a in data["artifacts"]]]:
            shutil.copyfile(directory / name, gate / name)
        aggregate(SimpleNamespace(directory=str(gate), results=str(reports), version=data["version"], commit=data["commit"]))
        require(read_json(gate / "validation.json") == validation, "validation report differs from recomputed gates")
    for name in sorted(expected - {"provenance.jsonl"}):
        gh(["attestation", "verify", str(directory / name), "--bundle", str(directory / "provenance.jsonl"),
            "--repo", REPOSITORY, "--signer-workflow", REPOSITORY + "/" + WORKFLOW,
            "--source-digest", workflow_sha, "--source-ref", "refs/heads/main", "--deny-self-hosted-runners"])
    for commit in (data["commit"], workflow_sha):
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True, capture_output=True)
    return data


def publish(directory, data, run_id):
    directory = Path(directory)
    tag = "v" + data["version"]
    marker = f"<!-- zinc-candidate-run: {run_id}; manifest: {sha256(directory / 'release-manifest.json')} -->"
    ref = api("git/ref/tags/" + tag, optional=True)
    existing = api("releases/tags/" + tag, optional=True)
    if existing:
        require(marker in existing.get("body", ""), "refusing to mutate an existing/published release")
        require(ref and ref.get("object", {}).get("type") == "commit" and ref["object"]["sha"] == data["commit"], "existing tag/source mismatch")
        if existing.get("draft") is False:
            verify_remote_assets(directory, tag)
            print("This candidate is already published; all existing bytes match")
            return
    if ref:
        require(ref.get("object", {}).get("type") == "commit" and ref["object"]["sha"] == data["commit"] and existing,
                "tag already exists without this exact candidate draft; inspect it manually")
    else:
        api("git/refs", {"ref": "refs/tags/" + tag, "sha": data["commit"]})
    if not existing:
        existing = api("releases", {"tag_name": tag, "name": "ZINC " + tag,
                                    "body": (directory / "release-notes.md").read_text() + "\n" + marker,
                                    "draft": True, "prerelease": "-" in data["version"].split("+")[0]})
    files = sorted(p.name for p in directory.iterdir())
    present = {asset["name"] for asset in existing.get("assets", [])}
    require(present.issubset(files), "draft contains unexpected assets")
    # Resume only this candidate's draft; no --clobber and no overwrite endpoint.
    for name in files:
        if name not in present:
            gh(["release", "upload", tag, str(directory / name), "--repo", REPOSITORY])
    verify_remote_assets(directory, tag)
    # gh uses PATCH here and keeps the existing asset bytes.
    gh(["release", "edit", tag, "--repo", REPOSITORY, "--draft=false"])
    print("Published https://github.com/" + REPOSITORY + "/releases/tag/" + tag)


def download_public(directory, data):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(PUBLIC_FILES | {a["filename"] for a in data["artifacts"]}):
        url = "https://github.com/" + REPOSITORY + "/releases/download/v" + data["version"] + "/" + name
        with urllib.request.urlopen(url, timeout=180) as response, (destination / name).open("xb") as stream:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                require(size <= 128 * 1024 * 1024, "published asset exceeds its size limit")
                stream.write(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--public-download")
    args = parser.parse_args()
    require(re.fullmatch(r"[1-9][0-9]*", args.run), "invalid candidate run ID")
    require(os.environ.get("GITHUB_REF") == "refs/heads/main" and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY,
            "publisher must run from the project's main workflow")
    run = api("actions/runs/" + args.run)
    validate_run(run, args.run)
    validate_environment(api("environments/release", optional=True))
    directory = Path(args.directory)
    if not directory.exists():
        gh(["run", "download", args.run, "--repo", REPOSITORY, "--name", "release-candidate", "--dir", str(directory)])
    data = verify_bundle(directory, args.run, run)
    catalog = subprocess.check_output(["git", "show", data["commit"] + ":src/model/catalog.zig"], cwd=ROOT, text=True)
    validate_catalog(catalog, data["version"])
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("version=" + data["version"] + "\ncommit=" + data["commit"] + "\n")
            stream.write("matrix=" + json.dumps({"include": policy()["targets"]}, separators=(",", ":")) + "\n")
    summary = f"Candidate [{args.run}](https://github.com/{REPOSITORY}/actions/runs/{args.run}): **v{data['version']}**, source `{data['commit']}`. All {len(data['required_profiles'])} GPU profiles passed.\n\n"
    summary += "\n".join("- `" + a["filename"] + "` — `" + a["sha256"] + "`" for a in data["artifacts"]) + "\n"
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(summary)
    if args.publish:
        publish(directory, data, args.run)
    if args.public_download:
        download_public(args.public_download, data)
        for file in directory.iterdir():
            require(sha256(Path(args.public_download) / file.name) == sha256(file), "public download differs from candidate")
        print("Every public download matches the tested candidate bytes")
    print("Candidate provenance, source, archive checksums and hardware evidence verified")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        sys.exit("release publication: " + str(error))
