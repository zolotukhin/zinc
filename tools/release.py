#!/usr/bin/env python3
"""Build-independent release packaging and candidate gates (Python stdlib only)."""

import argparse
import contextlib
import datetime
import gzip
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from release_lib import (
    ROOT, archive_name, check_metadata, extract_archive, inventory, policy,
    probe_binary, read_json, require, sha256, source_sha, target, version, write_json,
)


def compatibility_version(value):
    return tuple(int(n) for n in value.split("."))


def command_output(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, f"release inspection failed: {args[0]}")
    return result.stdout


def inspect_binary(binary, spec):
    """Reject a wrong architecture, excessive glibc floor or build-host libraries."""
    with binary.open("rb") as stream:
        head = stream.read(64)
    if spec["backend"] == "metal":
        require(len(head) >= 8 and head[:4] == b"\xcf\xfa\xed\xfe" and struct.unpack_from("<I", head, 4)[0] == 0x100000C,
                "expected an ARM64 Mach-O executable")
        loads = command_output(["otool", "-l", str(binary)])
        minimums = re.findall(r"\bminos\s+(\d+(?:\.\d+)+)", loads)
        require(minimums and max(map(compatibility_version, minimums)) <= compatibility_version(spec["minimum_os"]),
                "Mach-O deployment target exceeds the release minimum")
        libraries = [line.strip().split(" (", 1)[0] for line in command_output(["otool", "-L", str(binary)]).splitlines()[1:]]
        require(all(lib.startswith(("/usr/lib/", "/System/Library/")) for lib in libraries), "non-system Mach-O dependency")
        return {"libraries": libraries, "minimum_os": max(minimums, key=compatibility_version)}
    require(len(head) >= 20 and head[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<H", head, 18)[0] == 62,
            "expected an x86_64 little-endian ELF executable")
    symbols = command_output(["readelf", "--version-info", str(binary)])
    versions = re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)", symbols)
    require(versions, "cannot determine the glibc requirement")
    maximum = max(versions, key=compatibility_version)
    require(compatibility_version(maximum) <= compatibility_version(spec["minimum_glibc"]), "ELF requires a newer glibc than the release minimum")
    dynamic = command_output(["readelf", "-d", str(binary)])
    libraries = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic)
    base = re.compile(r"(?:lib(?:c|m|dl|rt|pthread|gcc_s|stdc\+\+)\.so\.[0-9.]+|ld-linux-x86-64\.so\.2)")
    backend = re.compile(r"libvulkan\.so\.1" if spec["backend"] == "vulkan" else r"lib(?:amdhip64|hiprtc|hipblas)\.so\.[0-9.]+")
    require(all(base.fullmatch(lib) or backend.fullmatch(lib) for lib in libraries), "unexpected ELF runtime dependency")
    required = ["libvulkan.so.1"] if spec["backend"] == "vulkan" else ["libamdhip64.so.", "libhiprtc.so.", "libhipblas.so."]
    require(all(any(lib.startswith(prefix) for lib in libraries) for prefix in required), "missing backend runtime dependency")
    rpaths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]+)\]", dynamic)
    require(all(part.startswith(("$ORIGIN", "/opt/rocm")) for value in rpaths for part in value.split(":")), "build-host ELF runtime path")
    return {"libraries": sorted(libraries), "required_glibc": maximum}


def package(args):
    release_version = version(args.version)
    commit = source_sha(args.commit or command_output(["git", "rev-parse", "HEAD"]).strip())
    identifier = args.target
    if identifier in ("linux-x86_64", "macos-aarch64"):
        identifier += "-" + (args.backend or ("metal" if identifier.startswith("macos") else "vulkan"))
    spec = target(identifier)
    require(not args.backend or args.backend == spec["backend"], "target/backend mismatch")
    prefix = Path(args.prefix).resolve()
    binary = prefix / "bin/zinc"
    require(binary.is_file() and not binary.is_symlink() and os.access(binary, os.X_OK), "missing executable bin/zinc")
    asset_policy = read_json(prefix / "share/zinc/asset-manifest.json")
    require(asset_policy.get("schema_version") == 1 and asset_policy.get("backend") == spec["backend"], "build asset manifest does not match backend")
    assets = asset_policy.get("files")
    require(isinstance(assets, list) and len(assets) == len(set(assets)), "invalid build asset inventory")
    require(bool(assets) or spec["backend"] == "rocm", "empty shader inventory")
    audit = inspect_binary(binary, spec)
    output = Path(args.dist).resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = archive_name(release_version, identifier)
    archive = output / name
    require(not archive.exists(), "refusing to overwrite an existing release archive")
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    require(timestamp >= 0, "invalid SOURCE_DATE_EPOCH")
    with tempfile.TemporaryDirectory(prefix="zinc-package-") as temporary:
        tree = Path(temporary) / name.removesuffix(".tar.gz")
        (tree / "bin").mkdir(parents=True)
        shutil.copyfile(binary, tree / "bin/zinc")
        (tree / "bin/zinc").chmod(0o755)
        tokenizer = prefix / "bin/zinc-tokenize"
        require(tokenizer.is_file() and not tokenizer.is_symlink() and os.access(tokenizer, os.X_OK), "missing packaged tokenizer tool")
        shutil.copyfile(tokenizer, tree / "bin/zinc-tokenize")
        (tree / "bin/zinc-tokenize").chmod(0o755)
        for relative in ["share/zinc/asset-manifest.json", *assets]:
            require(isinstance(relative, str) and relative.startswith("share/zinc/") and ".." not in relative.split("/"), "unsafe asset inventory path")
            source = prefix / relative
            require(source.is_file() and not source.is_symlink(), "missing installed runtime asset")
            dest = tree / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        for relative in ["README.md", "LICENSE"]:
            shutil.copyfile(ROOT / relative, tree / relative)
        signing = {"status": "unsigned"}
        if args.notary_receipt:
            require(spec["backend"] == "metal", "notarization is only valid for Metal")
            receipt = read_json(args.notary_receipt)
            require(receipt.get("status") == "Accepted" and receipt.get("binary_sha256") == sha256(binary), "notarization receipt does not match binary")
            for executable in (binary, tokenizer):
                require(receipt.get("binaries", {}).get(executable.name) == sha256(executable), "notarization receipt is missing an executable")
                subprocess.run(["codesign", "--verify", "--strict", str(executable)], check=True, capture_output=True, timeout=30)
                details = subprocess.run(["codesign", "-dv", "--verbose=4", str(executable)], capture_output=True, text=True, timeout=30)
                require(details.returncode == 0 and "Authority=Developer ID Application:" in details.stderr and "flags=0x10000(runtime)" in details.stderr,
                        "release binary lacks a Developer ID hardened-runtime signature")
            signing = {"status": "notarized", "submission_id": receipt["id"], "binary_sha256": sha256(binary)}
        metadata = {
            "schema_version": 1, "name": "zinc", "version": release_version, "tag": "v" + release_version,
            "commit": commit, "target": identifier, "zig_target": spec["zig_target"], "cpu": spec["cpu"],
            "backend": spec["backend"], "optimize": "ReleaseFast", "minimum_os": spec["minimum_os"],
            "minimum_glibc": spec["minimum_glibc"], "runtime": spec["runtime"], "audit": audit, "signing": signing,
            "built_at": datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat(),
        }
        write_json(tree / "VERSION.json", metadata)
        values = {
            "format": "1", "version": release_version, "commit": commit, "target": identifier,
            "backend": spec["backend"], "minimum_os": spec["minimum_os"], "minimum_glibc": spec["minimum_glibc"],
            "binary_sha256": sha256(tree / "bin/zinc"),
        }
        (tree / "INSTALL-METADATA").write_text("".join(f"{key}={value}\n" for key, value in values.items()))
        (tree / "FILES.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in inventory(tree).items()))
        check_metadata(tree, release_version, identifier, commit)
        probe_binary(tree, release_version, identifier, commit)
        # Normalize unsigned staging metadata. Developer ID timestamps remain
        # intentional signed content; every published candidate keeps its bytes.
        staged_archive = Path(temporary) / name
        with staged_archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=timestamp) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                for path in [tree, *sorted(tree.rglob("*"))]:
                    info = bundle.gettarinfo(str(path), str(path.relative_to(tree.parent)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = timestamp
                    info.mode = 0o755 if path.is_dir() or path.parent == tree / "bin" else 0o644
                    if path.is_file():
                        with path.open("rb") as source:
                            bundle.addfile(info, source)
                    else:
                        bundle.addfile(info)
        # Copy via a temporary file on the output filesystem, then publish the
        # complete archive without replacing an existing candidate.
        with tempfile.NamedTemporaryFile(dir=output, prefix=".archive-", delete=False) as raw:
            pending = Path(raw.name)
            try:
                with staged_archive.open("rb") as source:
                    shutil.copyfileobj(source, raw)
                raw.flush()
                os.fsync(raw.fileno())
                os.link(pending, archive)
            finally:
                pending.unlink(missing_ok=True)
    print(archive)


def verify(args):
    with tempfile.TemporaryDirectory(prefix="zinc-verify-") as temporary:
        tree = extract_archive(args.archive, temporary, args.version, args.target, args.commit)
        if args.probe:
            probe_binary(tree, args.version, args.target, read_json(tree / "VERSION.json")["commit"])
    print("Package verified")


def matrix(args):
    data = policy()
    if args.kind == "build":
        result = {"include": data["targets"]}
    else:
        variables = json.loads(os.environ.get("RELEASE_VARIABLES", "{}"))
        missing = [p["runner_variable"] for p in data["gpu_profiles"] if not variables.get(p["runner_variable"])]
        require(not missing, "GPU runner configuration missing: " + ", ".join(sorted(set(missing))))
        rows = []
        for profile in data["gpu_profiles"]:
            labels = json.loads(variables[profile["runner_variable"]])
            require(isinstance(labels, list) and labels and all(isinstance(s, str) and s for s in labels), "runner variable must be a JSON label array")
            rows.append({**profile, "runner": labels})
        result = {"include": rows}
    print(json.dumps(result, separators=(",", ":")))


def resolve(args):
    release_version = version(args.version)
    require(args.ref in ("main", "refs/heads/main") or re.fullmatch(r"[0-9a-f]{40}", args.ref), "candidate source must be main or a full SHA")
    ref = "origin/main" if args.ref in ("main", "refs/heads/main") else args.ref
    commit = source_sha(command_output(["git", "rev-parse", "--verify", ref + "^{commit}"]).strip())
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], check=True, capture_output=True)
    require(not command_output(["git", "status", "--porcelain"]).strip(), "release source checkout is dirty")
    with contextlib.redirect_stdout(io.StringIO()) as output:
        matrix(argparse.Namespace(kind="gpu"))
    gpu_matrix = output.getvalue().strip()
    values = {
        "version": release_version, "commit": commit,
        "epoch": command_output(["git", "show", "-s", "--format=%ct", commit]).strip(),
        "build_matrix": json.dumps({"include": policy()["targets"]}, separators=(",", ":")),
        "gpu_matrix": gpu_matrix,
    }
    require(values["epoch"].isdigit(), "invalid source timestamp")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for key, value in values.items():
                stream.write(key + "=" + value + "\n")
    # Runner labels remain in job outputs, not the public release report.
    print(json.dumps({"version": release_version, "commit": commit, "epoch": values["epoch"]}))


def manifest(args):
    release_version, commit = version(args.version), source_sha(args.commit)
    directory = Path(args.directory)
    data = policy()
    artifacts = []
    for spec in data["targets"]:
        archive = directory / archive_name(release_version, spec["id"])
        with tempfile.TemporaryDirectory(prefix="zinc-manifest-") as temporary:
            tree = extract_archive(archive, temporary, release_version, spec["id"], commit)
            meta = read_json(tree / "VERSION.json")
            require(isinstance(meta.get("audit"), dict) and meta["audit"].get("libraries"), "missing binary dependency audit")
            require(not spec["signing_required"] or meta.get("signing", {}).get("status") == "notarized", "Metal candidate must be signed and notarized")
            artifacts.append({"target": spec["id"], "filename": archive.name, "sha256": sha256(archive),
                              "size_bytes": archive.stat().st_size, "binary_sha256": sha256(tree / "bin/zinc"),
                              "signing": meta.get("signing"), "audit": meta["audit"]})
    expected = {a["filename"] for a in artifacts}
    require({p.name for p in directory.glob("*.tar.gz")} == expected, "unexpected or duplicate candidate archive")
    result = {"schema_version": 1, "version": release_version, "commit": commit,
              "candidate_run": os.environ.get("GITHUB_RUN_ID"), "workflow_sha": os.environ.get("GITHUB_SHA"),
              "policy_sha256": sha256(ROOT / "config/release/targets.json"),
              "artifacts": artifacts,
              "required_profiles": [{k: p[k] for k in ("id", "target", "family")} for p in data["gpu_profiles"]]}
    write_json(directory / "release-manifest.json", result)
    (directory / "SHA256SUMS.txt").write_text("".join(a["sha256"] + "  " + a["filename"] + "\n" for a in artifacts))
    print("Candidate manifest and archive checksums written")


def verify_candidate(directory, expected_version=None, expected_commit=None):
    directory = Path(directory)
    data = read_json(directory / "release-manifest.json")
    require(data.get("schema_version") == 1, "unsupported candidate manifest")
    version(data.get("version"))
    source_sha(data.get("commit"))
    require(expected_version is None or data["version"] == expected_version, "candidate version mismatch")
    require(expected_commit is None or data["commit"] == expected_commit, "candidate source mismatch")
    require(data.get("policy_sha256") == sha256(ROOT / "config/release/targets.json"), "release policy differs from candidate")
    expected_profiles = [{k: p[k] for k in ("id", "target", "family")} for p in policy()["gpu_profiles"]]
    require(data.get("required_profiles") == expected_profiles, "candidate hardware policy mismatch")
    artifacts = data.get("artifacts", [])
    require({a.get("target") for a in artifacts} == {t["id"] for t in policy()["targets"]} and len(artifacts) == 3,
            "candidate must contain exactly three targets")
    for artifact in artifacts:
        spec = target(artifact["target"])
        require(artifact.get("filename") == archive_name(data["version"], spec["id"]), "candidate archive name mismatch")
        path = directory / artifact["filename"]
        require(path.is_file() and not path.is_symlink() and sha256(path) == artifact.get("sha256"), "candidate archive digest mismatch")
        require(path.stat().st_size == artifact.get("size_bytes"), "candidate archive size mismatch")
        require(not spec["signing_required"] or artifact.get("signing", {}).get("status") == "notarized", "candidate is not notarized")
    return data


def aggregate(args):
    directory = Path(args.directory)
    data = verify_candidate(directory, args.version, args.commit)
    artifacts = {a["target"]: a for a in data["artifacts"]}
    reports = []
    seen = set()
    required_checks = {"package", "identity", "gpu", "model_hash", "tokenizer_parity", "inference_parity", "http", "cleanup"}
    for path in sorted(Path(args.results).glob("*.json")):
        report = read_json(path)
        require(report.get("schema_version") == 1, "unsupported GPU report schema")
        identifier = report.get("profile")
        require(identifier not in seen, "duplicate GPU report")
        seen.add(identifier)
        spec = next((p for p in data["required_profiles"] if p["id"] == identifier), None)
        require(spec is not None, "unexpected GPU report")
        require(all(report.get(k) == data[k] for k in ("version", "commit", "policy_sha256")), "stale or mismatched GPU report")
        artifact = artifacts[spec["target"]]
        require(report.get("target") == spec["target"] and report.get("family") == spec["family"], "wrong GPU family or backend")
        require(report.get("archive_sha256") == artifact["sha256"] and report.get("binary_sha256") == artifact["binary_sha256"], "GPU tested different artifact bytes")
        checks = report.get("checks", {})
        require(report.get("options") == {"parallel": 1, "kv_cache": "f32", "mtp": "backend-default", "greedy": True}, "untested or missing runtime option policy")
        require(required_checks.issubset(checks) and all(value is True for value in checks.values()), "GPU checks failed or were skipped")
        require(not spec["target"].endswith("vulkan") or checks.get("kernel_reference") is True, "missing packaged Vulkan numerical reference checks")
        profile = next(p for p in policy()["gpu_profiles"] if p["id"] == identifier)
        models = report.get("models", [])
        require(len(models) == len(profile["models"]) and {m.get("id") for m in models} == set(profile["models"]), "missing model correctness evidence")
        for model in models:
            expected = next(m for m in policy()["models"] if m["id"] == model["id"])
            require(model.get("sha256") == expected["sha256"] and model.get("context") == expected["context"] and model.get("passed") is True,
                    "model digest/context differs from policy")
        reference = report.get("reference", {})
        require(reference.get("commit") == policy()["reference"]["commit"] and all(re.fullmatch(r"[0-9a-f]{64}", reference.get(key, "")) for key in ("server_sha256", "tokenizer_sha256")),
                "missing pinned reference evidence")
        require(all(report.get("hardware", {}).get(key) for key in ("gpu", "os", "driver")), "missing public hardware/driver evidence")
        reports.append(report)
    require(seen == {p["id"] for p in data["required_profiles"]}, "required GPU reports are missing")
    result = {"schema_version": 1, "version": data["version"], "commit": data["commit"],
              "policy_sha256": data["policy_sha256"], "manifest_sha256": sha256(directory / "release-manifest.json"),
              "status": "passed", "profiles": reports}
    write_json(directory / "validation.json", result)
    print(f"All {len(reports)} required GPU profiles passed")


def install_smoke(args):
    """Install the actual archive from outside the source tree, CPU only."""
    release_version, identifier = version(args.version), target(args.target)["id"]
    archive = Path(args.directory).resolve() / archive_name(release_version, identifier)
    with tempfile.TemporaryDirectory(prefix="zinc install smoke ") as temporary:
        root = Path(temporary)
        downloads = root / "downloads" / ("v" + release_version)
        downloads.mkdir(parents=True)
        shutil.copyfile(archive, downloads / archive.name)
        (downloads / "SHA256SUMS.txt").write_text(sha256(archive) + "  " + archive.name + "\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ZINC_")}
        env.update({"ZINC_VERSION": "v" + release_version, "ZINC_BACKEND": target(identifier)["backend"],
                    "ZINC_BASE_URL": downloads.parent.as_uri(), "ZINC_INSTALL_DIR": str(root / "installed files"),
                    "ZINC_BIN_DIR": str(root / "commands")})
        for attempt in range(3):
            subprocess.run(["bash", str(ROOT / "scripts/install.sh")], cwd=root, env=env, check=True, capture_output=True, timeout=120)
            installed = root / "installed files/current"
            probe_binary(installed, release_version, identifier, read_json(installed / "VERSION.json")["commit"])
            if attempt == 1:
                (root / "commands/zinc").unlink()
        require((root / "commands/zinc").resolve() == (installed / "bin/zinc").resolve(), "installer did not repair executable link")
    print("Actual archive: clean install, idempotent install and link repair passed (CPU only)")


def finalize(args):
    directory = Path(args.directory)
    data = verify_candidate(directory)
    validation = read_json(directory / "validation.json")
    require(validation.get("status") == "passed" and validation.get("manifest_sha256") == sha256(directory / "release-manifest.json"), "candidate is not validated")
    shutil.copyfile(ROOT / "scripts/install.sh", directory / "install.sh")
    shutil.copyfile(ROOT / "config/release/targets.json", directory / "release-policy.json")
    notes = [f"# ZINC v{data['version']}", "", "Metal, Vulkan and ROCm binaries built from `" + data["commit"] + "`.", "",
             "Download the backend for your hardware. Models and GPU drivers are installed separately. CUDA remains experimental and source-only.", "",
             "| Archive | Runtime |", "| --- | --- |"]
    for artifact in data["artifacts"]:
        notes.append(f"| `{artifact['filename']}` | {target(artifact['target'])['runtime']} |")
    notes.extend(["", "Verify archive hashes with `SHA256SUMS.txt`. The signed provenance in `provenance.jsonl` covers the archives, installer and validation report.", "",
                  "```bash", "# Download and inspect install.sh from this release, then:", f"ZINC_VERSION=v{data['version']} bash install.sh", "# Linux ROCm: ZINC_BACKEND=rocm ZINC_VERSION=v" + data["version"] + " bash install.sh", "zinc --check", "zinc model list --all", "```", "",
                  "Installer defaults: Metal on ARM64 macOS 15+, Vulkan on glibc Linux x86_64 (2.35+); ROCm is explicit (glibc 2.39+, ROCm 7.2.4). Intel Macs, musl, Linux ARM64 and native Windows have no 0.1.0 archive.", "",
                  "The installer verifies and executes the new binary before atomic activation. To restore the previous install: `bash install.sh --rollback`. Models remain in their existing cache.", "",
                  "For a first Metal/Vulkan run:", "", "```bash", "zinc model pull qwen3-0.6b-q8-0", 'zinc --model-id qwen3-0.6b-q8-0 --chat --prompt "Explain a ring buffer."', "```", "",
                  "For ROCm, use `qwen35-9b-q4k-m` in both commands. Metal/Vulkan can also launch the browser UI with `zinc chat --model-id qwen3-0.6b-q8-0`.", "",
                  "## Validated hardware and models", "", "Each row ran the exact archive bytes against the pinned llama.cpp reference, including tokenization, generation, streaming/cancellation, and the listed context limit.", "",
                  "| Profile | Hardware | Models (context tokens) |", "| --- | --- | --- |"])
    for report in validation["profiles"]:
        models = ", ".join("`" + m["id"] + "` (" + str(m["context"]) + ")" for m in report["models"])
        notes.append(f"| {report['profile']} | {report['hardware']['gpu']} | {models} |")
    notes.extend(["", "Only the combinations in `validation.json` are certified. The small Qwen3 starter is scoped to Metal/Vulkan; ROCm starts with Qwen3.5 9B. Larger models require sufficient GPU/unified memory. Apple chip variants still require their factory minimum macOS version.", "",
                  "Metal/Vulkan include the browser chat UI. ROCm exposes CLI/HTTP chat and completion routes; its standalone server has no built-in browser UI or hot model switching.", "",
                  "Release checks use greedy decoding, one slot, f32 KV and each backend's default MTP setting. They do not establish performance claims or certify every MTP/context/concurrency configuration. Alternative KV formats and other architecture/backend combinations require separate validation.", "",
                  "Source-only experimental backends and development features are outside this binary support promise. See `release-policy.json` for model revisions, hashes, the pinned reference commit and mandatory profiles.", ""])
    (directory / "release-notes.md").write_text("\n".join(notes))
    print("Validated installer, policy and release notes frozen for attestation")


def upstream_check(_args):
    # Verify download identities without downloading any model weights or
    # opening a GPU. Cache presence must not hide a broken upstream pin.
    for model in policy()["models"]:
        require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]), "invalid upstream repository")
        url = "https://huggingface.co/api/models/" + model["repository"] + "/revision/" + model["revision"] + "?blobs=true"
        with urllib.request.urlopen(url, timeout=45) as response:
            data = json.load(response)
        file = next((f for f in data.get("siblings", []) if f.get("rfilename") == model["file"]), None)
        require(file and file.get("size") == model["size_bytes"] and file.get("lfs", {}).get("sha256") == model["sha256"], "upstream model pin is unavailable or differs: " + model["id"])
        require(data.get("cardData", {}).get("license") == model["license"], "upstream model license metadata differs: " + model["id"])
        print(model["id"] + ": immutable upstream file/hash/size/license verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--version", required=True)
    pack.add_argument("--target", required=True)
    pack.add_argument("--backend", choices=["metal", "vulkan", "rocm"])
    pack.add_argument("--commit")
    pack.add_argument("--prefix", default="zig-out")
    pack.add_argument("--dist", default="dist")
    pack.add_argument("--notary-receipt")
    pack.set_defaults(run=package)
    check = sub.add_parser("verify")
    check.add_argument("--archive", required=True)
    check.add_argument("--version", required=True)
    check.add_argument("--target", required=True)
    check.add_argument("--commit")
    check.add_argument("--probe", action="store_true")
    check.set_defaults(run=verify)
    mat = sub.add_parser("matrix")
    mat.add_argument("kind", choices=["build", "gpu"])
    mat.set_defaults(run=matrix)
    resolver = sub.add_parser("resolve")
    resolver.add_argument("--version", required=True)
    resolver.add_argument("--ref", default="main")
    resolver.set_defaults(run=resolve)
    frozen = sub.add_parser("manifest")
    frozen.add_argument("--version", required=True)
    frozen.add_argument("--commit", required=True)
    frozen.add_argument("--directory", required=True)
    frozen.set_defaults(run=manifest)
    gate = sub.add_parser("aggregate")
    gate.add_argument("--version", required=True)
    gate.add_argument("--commit", required=True)
    gate.add_argument("--directory", required=True)
    gate.add_argument("--results", required=True)
    gate.set_defaults(run=aggregate)
    smoke = sub.add_parser("install-smoke")
    smoke.add_argument("--directory", required=True)
    smoke.add_argument("--version", required=True)
    smoke.add_argument("--target", required=True)
    smoke.set_defaults(run=install_smoke)
    frozen_notes = sub.add_parser("finalize")
    frozen_notes.add_argument("--directory", required=True)
    frozen_notes.set_defaults(run=finalize)
    upstream = sub.add_parser("upstream-check")
    upstream.set_defaults(run=upstream_check)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, f"release: {error}\n")


if __name__ == "__main__":
    main()
