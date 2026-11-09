"""Isolate the native Intel baseline; never change ZINC/oracle loader paths."""
import hashlib
import json
import os
from pathlib import Path
import re

from release_lib import require, sha256

COMPONENTS = ("dpcpp_linux-64", "intel-sycl-rt", "mkl-devel-dpcpp", "onednn", "onedpl-devel")


def stack_manifest(prefix):
    prefix = Path(prefix).resolve()
    metadata = prefix / "conda-meta"
    require(metadata.is_dir(), "SYCL prefix needs its installed package metadata")
    packages = {}
    for path in metadata.glob("*.json"):
        row = json.loads(path.read_text())
        name = row.get("name")
        require(isinstance(name, str) and name not in packages, "duplicate or invalid native package identity")
        packages[name] = {key: row.get(key) for key in ("name", "version", "build", "sha256", "channel")}
    for name in COMPONENTS:
        require(name in packages and isinstance(packages[name]["version"], str)
                and re.fullmatch(r"[a-f0-9]{64}", packages[name].get("sha256") or ""),
                "native package identity is missing: " + name)
    rows = [packages[name] for name in sorted(packages)]
    return {"packages": rows, "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def environment(prefix, base):
    prefix = Path(prefix).resolve()
    require((prefix / "bin/icpx").is_file() and (prefix / "lib").is_dir(), "native compiler/runtime prefix is incomplete")
    # Inherited tuning can silently disable graphs/XMX or choose another device.
    # Start from documented defaults; only this baseline gets the new libraries.
    result = {key: value for key, value in base.items() if not key.startswith(("GGML_", "SYCL_", "ONEAPI_", "UR_", "ZE_"))}
    result["PATH"] = str(prefix / "bin") + os.pathsep + result.get("PATH", os.defpath)
    result["LD_LIBRARY_PATH"] = str(prefix / "lib")
    result["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
    result["MKLROOT"] = str(prefix)
    return result


def runtime_evidence(pid, prefix):
    """Prove which SYCL/UR libraries the live model server actually loaded."""
    return runtime_evidence_from_maps(Path(f"/proc/{pid}/maps").read_text(), prefix)


def runtime_evidence_from_maps(text, prefix):
    prefix = Path(prefix).resolve()
    paths = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw = fields[5]
        name = Path(raw.removesuffix(" (deleted)")).name
        if not re.match(r"lib(?:sycl(?:[-.]|$)|ur_|mkl_sycl|dnnl)", name):
            continue
        require(not raw.endswith(" (deleted)"), "native runtime changed while the server was running")
        path = Path(raw).resolve()
        require(prefix in path.parents and path.is_file(), "baseline loaded an Intel runtime outside its selected prefix")
        require(name not in paths or paths[name] == path, "baseline loaded conflicting native runtime libraries")
        paths[name] = path
    require(any(name.startswith("libsycl.so") for name in paths)
            and any(name.startswith("libur_adapter_level_zero") for name in paths),
            "baseline did not load the selected SYCL and Level Zero runtime")
    return {"runtime": "level_zero", "library_sha256": {name: sha256(path) for name, path in sorted(paths.items())}}
