"""Isolate the native Intel baseline; never change ZINC/oracle loader paths."""
import hashlib
import ctypes
import json
import os
from pathlib import Path
import re
import sys

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
    result = {key: value for key, value in base.items() if not key.startswith(("GGML_", "SYCL_", "ONEAPI_", "UR_", "ZE_", "ZES_", "OCL_", "OPENCL_"))}
    result["PATH"] = str(prefix / "bin") + os.pathsep + result.get("PATH", os.defpath)
    result["LD_LIBRARY_PATH"] = str(prefix / "lib")
    result["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
    # Conda's OpenCL loader otherwise finds only its bundled CPU ICD. oneDNN
    # maps a Level Zero device to the matching GPU OpenCL device for GEMM/SDPA.
    # Keep the host's existing GPU ICDs scoped to this reference environment.
    result["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"
    require(type(graphs) is bool, "SYCL graph selection must be explicit")
    result["GGML_SYCL_ENABLE_GRAPH"] = "1" if graphs else "0"
    result["MKLROOT"] = str(prefix)
    return result


def opencl_driver_evidence(text=None):
    if text is None:
        text = Path("/proc/self/maps").read_text()
    libraries = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw = fields[5]
        name = Path(raw.removesuffix(" (deleted)")).name
        if not re.match(r"lib(?:igdrcl|igc(?:[-.]|$)|igdfcl|iga64)", name):
            continue
        require(not raw.endswith(" (deleted)"), "OpenCL GPU driver changed during discovery")
        path = Path(raw).resolve()
        require(path.is_file(), "OpenCL GPU driver file is missing")
        digest = sha256(path)
        require(name not in libraries or libraries[name] == digest, "conflicting OpenCL GPU drivers")
        libraries[name] = digest
    require(any(name.startswith("libigdrcl") for name in libraries), "official Intel GPU OpenCL driver was not loaded")
    return libraries


def opencl_gpu_probe(library):
    """Run in an owned child so enumeration cannot retain driver fds in Python."""
    library = Path(library).resolve()
    api = ctypes.CDLL(str(library))
    uint, pointer = ctypes.c_uint, ctypes.c_void_p
    api.clGetPlatformIDs.argtypes = [uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    api.clGetDeviceIDs.argtypes = [pointer, ctypes.c_uint64, uint, ctypes.POINTER(pointer), ctypes.POINTER(uint)]
    api.clGetDeviceInfo.argtypes = [pointer, uint, ctypes.c_size_t, pointer, ctypes.POINTER(ctypes.c_size_t)]
    for function in (api.clGetPlatformIDs, api.clGetDeviceIDs, api.clGetDeviceInfo):
        function.restype = ctypes.c_int
    count = uint()
    require(api.clGetPlatformIDs(0, None, ctypes.byref(count)) == 0 and 0 < count.value <= 64,
            "native OpenCL loader has no usable platforms")
    platforms = (pointer * count.value)()
    require(api.clGetPlatformIDs(count, platforms, None) == 0, "OpenCL platform enumeration failed")
    devices = []
    for platform in platforms:
        count = uint()
        status = api.clGetDeviceIDs(platform, 4, 0, None, ctypes.byref(count))  # CL_DEVICE_TYPE_GPU
        if status == -1:  # CL_DEVICE_NOT_FOUND: a CPU-only platform is insufficient.
            continue
        require(status == 0 and 0 < count.value <= 64, "OpenCL GPU enumeration failed")
        handles = (pointer * count.value)()
        require(api.clGetDeviceIDs(platform, 4, count, handles, None) == 0, "OpenCL GPU handles are missing")
        for handle in handles:
            vendor = uint()
            require(api.clGetDeviceInfo(handle, 0x1001, ctypes.sizeof(vendor), ctypes.byref(vendor), None) == 0,
                    "OpenCL GPU vendor is missing")
            if vendor.value != 0x8086:
                continue
            row = {"backend": "opencl", "device_type": "gpu", "vendor_id": vendor.value}
            # Read names/versions only: never request device UUIDs or PCI IDs.
            for name, field in (("device_name", 0x102B), ("driver_version", 0x102D)):
                size = ctypes.c_size_t()
                require(api.clGetDeviceInfo(handle, field, 0, None, ctypes.byref(size)) == 0 and 1 < size.value <= 4096,
                        "OpenCL GPU identity is missing")
                data = ctypes.create_string_buffer(size.value)
                require(api.clGetDeviceInfo(handle, field, size, data, None) == 0, "OpenCL GPU identity read failed")
                row[name] = data.value.decode("utf-8", errors="strict")
            devices.append(row)
    require(len(devices) == 1, "native OpenCL loader must expose exactly one Intel GPU; CPU discovery is insufficient")
    return {**devices[0], "library_sha256": sha256(library),
            "driver_library_sha256": opencl_driver_evidence()}


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
        native = re.match(r"lib(?:sycl(?:[-.]|$)|ur_|mkl_sycl|dnnl|OpenCL)", name)
        driver = re.match(r"lib(?:igdrcl|igc(?:[-.]|$)|igdfcl|iga64)", name)
        if not (native or driver):
            continue
        require(not raw.endswith(" (deleted)"), "native runtime changed while the server was running")
        path = Path(raw).resolve()
        require(path.is_file() and (not native or prefix in path.parents), "baseline loaded an Intel runtime outside its selected prefix")
        require(name not in paths or paths[name] == path, "baseline loaded conflicting native runtime libraries")
        paths[name] = path
    require(any(name.startswith("libsycl.so") for name in paths)
            and any(name.startswith("libur_adapter_level_zero") for name in paths),
            "baseline did not load the selected SYCL and Level Zero runtime")
    return {"runtime": "level_zero", "library_sha256": {name: sha256(path) for name, path in sorted(paths.items())}}


if __name__ == "__main__":
    require(sys.platform.startswith("linux") and len(sys.argv) == 3 and sys.argv[1] == "--probe-opencl",
            "OpenCL discovery runs only as an owned remote benchmark child")
    print(json.dumps(opencl_gpu_probe(sys.argv[2])))
