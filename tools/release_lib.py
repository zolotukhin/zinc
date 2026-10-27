"""Shared release policy and archive validation. Development/CI only; stdlib only."""

import hashlib
import contextlib
import gzip
import json
import math
import os
import re
import signal
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
SAFE_PATH = re.compile(r"[A-Za-z0-9_./+-]+")
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 4096


def install_signal_handlers():
    """Unwind owned subprocess/keychain cleanup on normal Actions cancellation."""
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version(value):
    require(isinstance(value, str) and SEMVER.fullmatch(value), "invalid semantic version")
    return value


def source_sha(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "source commit must be a full lowercase SHA")
    return value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def policy(path=None):
    data = read_json(path or ROOT / "config/release/targets.json")
    require(data.get("schema_version") == 1, "unsupported release policy")
    ids = [t["id"] for t in data["targets"]]
    require(len(ids) == len(set(ids)) == 3, "release requires three unique targets")
    require({t["backend"] for t in data["targets"]} == {"vulkan", "metal", "rocm"}, "unexpected release backends")
    profiles = data["gpu_profiles"]
    require(len(profiles) == len({p["id"] for p in profiles}) > 0, "duplicate or empty GPU policy")
    require(all(p["target"] in ids for p in profiles), "unknown GPU artifact target")
    models = data["models"]
    model_ids = {m["id"] for m in models}
    require(len(model_ids) == len(models) > 0, "duplicate/empty model policy")
    for model in models:
        source_sha(model["revision"])
        require(DIGEST.fullmatch(model["sha256"]) and model["size_bytes"] > 0 and model["context"] >= 2048,
                "model policy requires pinned bytes and a context limit")
    require(all(p.get("models") and set(p["models"]).issubset(model_ids) and len(set(p["models"])) == len(p["models"]) for p in profiles),
            "GPU profile is missing its required model matrix")
    source_sha(data["reference"]["commit"])
    generation = data["reference"].get("generation", {})
    require(set(generation) == {"batch_size", "micro_batch", "parallel", "kv_cache", "flash_attention"} and
            type(generation["batch_size"]) is int and type(generation["micro_batch"]) is int and
            1 <= generation["micro_batch"] <= generation["batch_size"] <= 4096 and
            type(generation["parallel"]) is int and generation["parallel"] == 1 and
            generation["kv_cache"] == "f32" and type(generation["flash_attention"]) is bool,
            "invalid reference generation policy")
    agreement = data["reference"].get("agreement", {})
    require(set(agreement) == {"cases", "tokens_per_case", "min_exact_fraction", "max_logprob_gap", "max_near_ties_per_case"} and
            type(agreement["cases"]) is int and agreement["cases"] > 0 and
            type(agreement["tokens_per_case"]) is int and 1 <= agreement["tokens_per_case"] <= 20 and
            type(agreement["min_exact_fraction"]) in (int, float) and 0.95 <= agreement["min_exact_fraction"] <= 1 and
            type(agreement["max_logprob_gap"]) in (int, float) and 0 <= agreement["max_logprob_gap"] <= 0.05 and
            type(agreement["max_near_ties_per_case"]) is int and 0 <= agreement["max_near_ties_per_case"] <= 1,
            "invalid reference agreement policy")
    require(data["toolchain"]["zig"] == (ROOT / ".zig-version").read_text().strip(), "release compiler differs from repository pin")
    source_sha(data["toolchain"]["vulkan_headers"]["commit"])
    require(DIGEST.fullmatch(data["toolchain"]["vulkan_headers"]["sha256"]), "Vulkan headers must be checksum-pinned")
    return data


def reference_agreement(cases, expected_cases=None):
    """Validate and summarize measured choices; an accepted tie is never exact."""
    limits = policy()["reference"]["agreement"]
    require(isinstance(cases, list) and len(cases) == (limits["cases"] if expected_cases is None else expected_cases),
            "missing reference agreement cases")
    total, exact = 0, 0
    for case in cases:
        require(isinstance(case, dict) and set(case) == {"tokens", "near_ties"} and
                type(case["tokens"]) is int and 0 < case["tokens"] <= limits["tokens_per_case"] and
                isinstance(case["near_ties"], list) and len(case["near_ties"]) <= limits["max_near_ties_per_case"],
                "invalid reference agreement case")
        for tie in case["near_ties"]:
            require(isinstance(tie, dict) and set(tie) == {"position", "chosen_id", "reference_id", "rank", "logprob_gap"} and
                    all(type(tie[key]) is int for key in ("position", "chosen_id", "reference_id", "rank")) and
                    0 <= tie["position"] < case["tokens"] and tie["chosen_id"] >= 0 and tie["reference_id"] >= 0 and
                    tie["chosen_id"] != tie["reference_id"] and tie["rank"] == 2 and
                    type(tie["logprob_gap"]) in (int, float) and math.isfinite(tie["logprob_gap"]) and
                    0 <= tie["logprob_gap"] <= limits["max_logprob_gap"], "reference choice exceeds near-tie policy")
        total += case["tokens"]
        exact += case["tokens"] - len(case["near_ties"])
    require(total > 0 and exact / total >= limits["min_exact_fraction"], "reference exact-choice agreement is too low")
    return {"cases": cases, "total_tokens": total, "exact_tokens": exact, "near_tie_tokens": total - exact}


def verify_reference_agreement(data):
    require(isinstance(data, dict) and set(data) == {"cases", "total_tokens", "exact_tokens", "near_tie_tokens"} and
            all(type(data[key]) is int for key in ("total_tokens", "exact_tokens", "near_tie_tokens")),
            "missing reference agreement measurements")
    require(data == reference_agreement(data["cases"]), "reference agreement totals differ from measured cases")


def target(identifier, data=None):
    found = [t for t in (data or policy())["targets"] if t["id"] == identifier]
    require(len(found) == 1, f"unsupported release target: {identifier}")
    return found[0]


def archive_name(release_version, identifier):
    return f"zinc-v{version(release_version)}-{target(identifier)['id']}.tar.gz"


def safe_member(name, root):
    require(SAFE_PATH.fullmatch(name) is not None, "unsafe archive member characters")
    parts = PurePosixPath(name).parts
    require(not name.startswith("/") and parts and parts[0] == root, "unexpected archive root")
    require(all(p not in ("..", ".", "") for p in name.rstrip("/").split("/")), "unsafe archive member path")


def inventory(tree):
    tree = Path(tree)
    found = {}
    for path in sorted(tree.rglob("*")):
        require(not path.is_symlink(), "release trees cannot contain symbolic links")
        if path.is_file():
            relative = path.relative_to(tree).as_posix()
            require(SAFE_PATH.fullmatch(relative) is not None, "unsafe release filename")
            found[relative] = sha256(path)
        else:
            require(path.is_dir(), "unsupported release file type")
    return found


def verify_files(tree):
    tree = Path(tree)
    lines = (tree / "FILES.sha256").read_text().splitlines()
    expected = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        require(separator and DIGEST.fullmatch(digest) and SAFE_PATH.fullmatch(name), "invalid file inventory")
        require(name not in expected and name not in ("FILES.sha256", ".archive-sha256"), "duplicate/self-referencing file inventory")
        require(not name.startswith("/") and ".." not in name.split("/"), "unsafe inventory path")
        expected[name] = digest
    actual = inventory(tree)
    actual.pop("FILES.sha256", None)
    actual.pop(".archive-sha256", None)
    require(expected == actual, "release file inventory mismatch")
    return actual


def check_metadata(tree, release_version, identifier, commit=None):
    t = target(identifier)
    meta = read_json(Path(tree) / "VERSION.json")
    require(meta.get("schema_version") == 1 and meta.get("name") == "zinc", "unsupported package metadata")
    require(meta.get("version") == version(release_version) and meta.get("tag") == "v" + release_version, "package version mismatch")
    require(meta.get("target") == identifier and meta.get("backend") == t["backend"], "package target/backend mismatch")
    require(meta.get("optimize") == "ReleaseFast" and meta.get("cpu") == t["cpu"], "package build mode mismatch")
    require(meta.get("minimum_os") == t["minimum_os"] and meta.get("minimum_glibc") == t["minimum_glibc"], "package compatibility mismatch")
    source_sha(meta.get("commit"))
    if commit is not None:
        require(meta["commit"] == source_sha(commit), "package source commit mismatch")
    expected = {
        "format": "1", "version": meta["version"], "commit": meta["commit"],
        "target": identifier, "backend": t["backend"],
        "minimum_os": t["minimum_os"], "minimum_glibc": t["minimum_glibc"],
        "binary_sha256": sha256(Path(tree) / "bin/zinc"),
    }
    receipt = {}
    for line in (Path(tree) / "INSTALL-METADATA").read_text().splitlines():
        key, sep, value = line.partition("=")
        require(sep and key not in receipt, "invalid installer metadata")
        receipt[key] = value
    require(receipt == expected, "installer metadata mismatch")
    files = verify_files(tree)
    require("bin/zinc-tokenize" in files, "missing packaged tokenizer tool")
    assets = read_json(Path(tree) / "share/zinc/asset-manifest.json")
    require(assets.get("backend") == t["backend"] and assets.get("schema_version") == 1, "asset manifest backend mismatch")
    required = assets.get("files")
    require(isinstance(required, list) and len(required) == len(set(required)), "invalid asset manifest")
    require(all(isinstance(f, str) and f in files for f in required), "missing runtime asset")
    actual_assets = {f for f in files if f.startswith("share/zinc/shaders/")}
    require(set(required) == actual_assets, "runtime asset inventory mismatch")
    require(bool(required) or t["backend"] == "rocm", "empty runtime shader set")
    expected_files = {"bin/zinc", "bin/zinc-tokenize", "VERSION.json", "INSTALL-METADATA", "README.md", "LICENSE", "share/zinc/asset-manifest.json", *required}
    require(set(files) == expected_files, "unexpected packaged file")
    return meta


@contextlib.contextmanager
def bounded_tar(archive):
    # Bound the entire decompressed stream before tarfile parses PAX/long-name
    # metadata, not only the declared sizes of ordinary file members.
    with gzip.open(archive, "rb") as source, tempfile.TemporaryFile() as raw:
        total = 0
        for block in iter(lambda: source.read(1024 * 1024), b""):
            total += len(block)
            require(total <= MAX_UNPACKED_BYTES, "release archive expands beyond its limit")
            raw.write(block)
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode="r:") as bundle:
            yield bundle


def extract_archive(archive, destination, release_version, identifier, commit=None):
    archive = Path(archive)
    require(archive.is_file() and archive.stat().st_size <= MAX_ARCHIVE_BYTES, "release archive missing or too large")
    root = archive_name(release_version, identifier).removesuffix(".tar.gz")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    require(not any(destination.iterdir()), "archive destination must be empty")
    with bounded_tar(archive) as bundle:
        members = []
        total_bytes = 0
        seen = set()
        for member in bundle:
            members.append(member)
            total_bytes += member.size
            require(len(members) <= MAX_MEMBERS, "invalid archive member count")
            require(total_bytes <= MAX_UNPACKED_BYTES, "release archive expands beyond its limit")
            safe_member(member.name, root)
            require(member.name.rstrip("/") not in seen, "duplicate archive member")
            seen.add(member.name.rstrip("/"))
            require(member.isdir() or member.isreg(), "archive links/devices are forbidden")
            require(not member.mode & 0o7000, "special archive permissions are forbidden")
        require(members, "empty release archive")
        # Every member was checked before extracting anything. Do not follow links.
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                path.chmod(0o755 if member.name in (root + "/bin/zinc", root + "/bin/zinc-tokenize") else 0o644)
    tree = destination / root
    check_metadata(tree, release_version, identifier, commit)
    return tree


def probe_binary(tree, release_version, identifier, commit):
    t = target(identifier)
    result = subprocess.run([str(Path(tree).resolve() / "bin/zinc"), "--version"], cwd=Path(tree).parent,
                            capture_output=True, text=True, timeout=30,
                            env={k: v for k, v in os.environ.items() if not k.startswith("ZINC_")})
    require(result.returncode == 0, "packaged binary --version failed")
    lines = result.stdout.splitlines()
    require(lines and lines[0] == "zinc " + release_version, "binary version mismatch")
    fields = {}
    for line in lines[1:]:
        require(": " in line, "invalid binary identity output")
        key, value = line.split(": ", 1)
        require(key not in fields, "duplicate binary identity field")
        fields[key] = value
    require(fields.get("commit") == commit and fields.get("backends") == t["backend"], "binary commit/backend mismatch")
    # Zig 0.17 spells this mode `fast`; the public build flag retains the
    # ReleaseFast alias. Earlier tagged binaries report the long spelling.
    require(fields.get("optimize") in ("fast", "ReleaseFast"), "binary is not ReleaseFast")
    expected_target = "aarch64-macos-none" if t["backend"] == "metal" else "x86_64-linux-gnu"
    require(fields.get("target") == expected_target, "binary CPU/OS target mismatch")
    return fields
