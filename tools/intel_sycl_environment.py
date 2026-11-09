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


def build_manifest(directory, source):
    """Retain compile-time precision without publishing CMake's private paths."""
    cache = Path(directory).resolve().parent / "CMakeCache.txt"
    require(cache.is_file(), "SYCL reference needs its original CMake build cache")
    values = {}
    for line in cache.read_text().splitlines():
        if line and not line.startswith(("//", "#")) and "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    require(Path(values.get("CMAKE_HOME_DIRECTORY", "")).resolve() == Path(source).resolve(),
            "SYCL build cache differs from its recorded source")
    expected = {"CMAKE_BUILD_TYPE": "Release", "GGML_SYCL": "ON", "GGML_VULKAN": "OFF", "GGML_CUDA": "OFF",
                "GGML_METAL": "OFF", "GGML_NATIVE": "OFF", "GGML_SYCL_HOST_MEM_FALLBACK": "OFF",
                "GGML_SYCL_SUPPORT_XMX": "ON", "GGML_SYCL_GRAPH": "ON", "GGML_SYCL_DNN": "ON"}
    require(all(values.get(key) == value for key, value in expected.items()), "SYCL reference build settings differ")
    require(values.get("GGML_SYCL_F16") in ("ON", "OFF"), "SYCL reference precision is missing")
    return {"cache_sha256": sha256(cache), "settings": {**expected, "GGML_SYCL_F16": values["GGML_SYCL_F16"]}}


def environment(prefix, base, graphs=False):
    prefix = Path(prefix).resolve()
    require((prefix / "bin/icpx").is_file() and (prefix / "lib").is_dir(), "native compiler/runtime prefix is incomplete")
    # Inherited tuning can silently disable graphs/XMX or choose another device.
    # Start from documented defaults; only this baseline gets the new libraries.
    result = {key: value for key, value in base.items() if not key.startswith(("GGML_", "SYCL_", "ONEAPI_", "UR_", "ZE_"))}
    result["PATH"] = str(prefix / "bin") + os.pathsep + result.get("PATH", os.defpath)
    result["LD_LIBRARY_PATH"] = str(prefix / "lib")
    result["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
    require(type(graphs) is bool, "SYCL graph selection must be explicit")
    result["GGML_SYCL_ENABLE_GRAPH"] = "1" if graphs else "0"
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
