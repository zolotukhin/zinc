"""CPU fixtures for native-baseline isolation; no Intel SDK or GPU is used."""
import json
import ctypes
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import intel_sycl_environment as sycl


class SyclEnvironmentTest(unittest.TestCase):
    def test_native_loader_environment_is_scoped_and_discards_inherited_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory); (prefix / "bin").mkdir(); (prefix / "lib").mkdir()
            (prefix / "bin/icpx").write_bytes(b"fixture")
            base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/old", "GGML_SYCL_ENABLE_XMX": "0",
                    "SYCL_DEVICE_FILTER": "cpu", "ONEAPI_DEVICE_SELECTOR": "opencl:cpu", "ZE_AFFINITY_MASK": "other",
                    "ZINC_MTP": "0", "OCL_ICD_FILENAMES": "cpu.icd", "OPENCL_VENDOR_PATH": "/old-vendors"}
            original = dict(base); result = sycl.environment(prefix, base)
            self.assertEqual(base, original)
            self.assertEqual(result["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
            self.assertEqual(result["LD_LIBRARY_PATH"], str(prefix.resolve() / "lib"))
            self.assertTrue(result["PATH"].startswith(str(prefix.resolve() / "bin")))
            for key in ("GGML_SYCL_ENABLE_XMX", "SYCL_DEVICE_FILTER", "ZE_AFFINITY_MASK", "OCL_ICD_FILENAMES", "OPENCL_VENDOR_PATH"):
                self.assertNotIn(key, result)
            self.assertEqual(result["OCL_ICD_VENDORS"], "/etc/OpenCL/vendors")
            self.assertEqual(result["ZINC_MTP"], "0")
            self.assertEqual(result["GGML_SYCL_ENABLE_GRAPH"], "0")
            self.assertEqual(result["GGML_SYCL_ENABLE_DNN"], "1")
            self.assertEqual(result["GGML_SYCL_FA_ONEDNN"], "1")
            self.assertEqual(sycl.environment(prefix, base, graphs=True)["GGML_SYCL_ENABLE_GRAPH"], "1")
            with self.assertRaisesRegex(ValueError, "graph selection"):
                sycl.environment(prefix, base, graphs="0")

    def test_native_controls_are_explicit_independent_and_do_not_change_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory); (prefix / "bin").mkdir(); (prefix / "lib").mkdir()
            (prefix / "bin/icpx").write_bytes(b"fixture")
            base = {"GGML_SYCL_ENABLE_DNN": "0", "GGML_SYCL_FA_ONEDNN": "0"}
            for dnn, attention in ((False, False), (False, True), (True, False), (True, True)):
                result = sycl.environment(prefix, base, dnn=dnn, onednn_attention=attention)
                self.assertEqual(result["GGML_SYCL_ENABLE_DNN"], "1" if dnn else "0")
                self.assertEqual(result["GGML_SYCL_FA_ONEDNN"], "1" if attention else "0")
                self.assertEqual(result["GGML_SYCL_ENABLE_GRAPH"], "0")
            self.assertEqual(sycl.environment(prefix, base)["GGML_SYCL_ENABLE_DNN"], "1")
            for options in ({"dnn": "0"}, {"onednn_attention": "1"}, {"dnn": None}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "explicit booleans"):
                    sycl.environment(prefix, base, **options)

    def test_onednn_math_cache_and_tracing_cannot_leak_into_a_default_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory); (prefix / "bin").mkdir(); (prefix / "lib").mkdir()
            (prefix / "bin/icpx").write_bytes(b"fixture")
            tuning = {"ONEDNN_DEFAULT_FPMATH_MODE": "ANY", "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "0",
                      "ONEDNN_VERBOSE": "all", "DNNL_DEFAULT_FPMATH_MODE": "BF16",
                      "DNNL_PRIMITIVE_CACHE_CAPACITY": "0", "DNNL_VERBOSE": "1"}
            base = {"PATH": "/usr/bin", "ZINC_MTP": "0", "OMP_NUM_THREADS": "4", **tuning}
            original = dict(base)
            clean = sycl.environment(prefix, {key: value for key, value in base.items() if key not in tuning})
            self.assertEqual(sycl.environment(prefix, base), clean)
            self.assertEqual(base, original)
            # A separately labelled diagnostic may opt in after isolation; it
            # must not alter subsequent defaults or the caller's environment.
            diagnostic = sycl.environment(prefix, base)
            diagnostic["ONEDNN_VERBOSE"] = "1"
            self.assertNotIn("ONEDNN_VERBOSE", sycl.environment(prefix, base))

    def probe(self, *, gpu_count=1, vendor=0x8086, platform_count=1):
        def platforms(count, data, actual):
            if actual is not None: actual._obj.value = platform_count
            if data is not None:
                for index in range(platform_count): data[index] = index + 1
            return 0
        def devices(platform, kind, count, data, actual):
            self.assertEqual(kind, 4, "only GPU devices may qualify")
            if gpu_count == 0: return -1
            if actual is not None: actual._obj.value = gpu_count
            if data is not None:
                for index in range(gpu_count): data[index] = index + 10
            return 0
        fields = []
        def info(device, field, size, data, actual):
            fields.append(field)
            self.assertIn(field, (0x1001, 0x102B, 0x102D), "never query UUIDs or PCI identifiers")
            if field == 0x1001:
                data._obj.value = vendor
            else:
                text = b"Intel test GPU\0" if field == 0x102B else b"driver test\0"
                if actual is not None: actual._obj.value = len(text)
                if data is not None: ctypes.memmove(data, text, len(text))
            return 0
        api = Mock(clGetPlatformIDs=Mock(side_effect=platforms), clGetDeviceIDs=Mock(side_effect=devices), clGetDeviceInfo=Mock(side_effect=info))
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libOpenCL.so.1"; library.write_bytes(b"fixture")
            with patch.object(sycl.ctypes, "CDLL", return_value=api), \
                    patch.object(sycl, "opencl_driver_evidence", return_value={"libigdrcl.so": "a" * 64}):
                result = sycl.opencl_gpu_probe(library)
            self.assertNotIn(directory, json.dumps(result))
        return result, fields

    def test_opencl_readiness_requires_a_real_intel_gpu_and_avoids_unique_identifiers(self):
        result, fields = self.probe()
        self.assertEqual(result["device_type"], "gpu")
        self.assertEqual(result["vendor_id"], 0x8086)
        self.assertEqual(result["device_name"], "Intel test GPU")
        self.assertRegex(result["library_sha256"], r"^[a-f0-9]{64}$")
        for options in ({"gpu_count": 0}, {"gpu_count": 2}, {"vendor": 0x10DE}, {"platform_count": 0}, {"platform_count": 65}):
            with self.subTest(options=options), self.assertRaises(ValueError): self.probe(**options)

    def test_live_opencl_driver_hashes_hide_paths_and_detect_replacements(self):
        with tempfile.TemporaryDirectory(prefix="driver maps ") as directory:
            root = Path(directory); driver = root / "libigdrcl.so"; driver.write_bytes(b"driver-v1")
            line = "00001000-00002000 r-xp 00000000 00:00 0 " + str(driver)
            before = sycl.opencl_driver_evidence(line)
            self.assertEqual(set(before), {"libigdrcl.so"}); self.assertNotIn(directory, json.dumps(before))
            driver.write_bytes(b"driver-v2")
            self.assertNotEqual(before, sycl.opencl_driver_evidence(line))
            with self.assertRaisesRegex(ValueError, "changed during"):
                sycl.opencl_driver_evidence(line + " (deleted)")
            with self.assertRaisesRegex(ValueError, "was not loaded"):
                sycl.opencl_driver_evidence("")

    def test_build_identity_distinguishes_fp16_and_rejects_fallback_or_other_sources(self):
        with tempfile.TemporaryDirectory(prefix="native build ") as directory:
            root = Path(directory); binary = root / "bin"; binary.mkdir(); source = root / "source"; source.mkdir()
            values = {"CMAKE_HOME_DIRECTORY": str(source), "CMAKE_BUILD_TYPE": "Release", "GGML_SYCL": "ON",
                      "GGML_VULKAN": "OFF", "GGML_CUDA": "OFF", "GGML_METAL": "OFF", "GGML_NATIVE": "OFF",
                      "GGML_SYCL_HOST_MEM_FALLBACK": "OFF", "GGML_SYCL_SUPPORT_XMX": "ON",
                      "GGML_SYCL_GRAPH": "ON", "GGML_SYCL_DNN": "ON", "GGML_SYCL_F16": "OFF"}
            def write():
                (root / "CMakeCache.txt").write_text("\n".join(key + ":STRING=" + value for key, value in values.items()))
            write(); before = sycl.build_manifest(binary, source)
            self.assertNotIn(directory, json.dumps(before))
            values["GGML_SYCL_F16"] = "ON"; write(); after = sycl.build_manifest(binary, source)
            self.assertNotEqual(before, after)
            self.assertEqual(after["settings"]["GGML_SYCL_F16"], "ON")
            values["GGML_SYCL_HOST_MEM_FALLBACK"] = "ON"; write()
            with self.assertRaisesRegex(ValueError, "build settings differ"):
                sycl.build_manifest(binary, source)
            values["GGML_SYCL_HOST_MEM_FALLBACK"] = "OFF"; values["CMAKE_HOME_DIRECTORY"] = str(root / "other"); write()
            with self.assertRaisesRegex(ValueError, "recorded source"):
                sycl.build_manifest(binary, source)

    def test_explicit_stack_lock_contains_only_public_archive_urls_with_sha256(self):
        lock = Path(__file__).resolve().parent.parent / "config/intel/oneapi-2026.2-linux-64.explicit.txt"
        lines = [line for line in lock.read_text().splitlines() if line and not line.startswith("#")]
        self.assertEqual(lines.pop(0), "@EXPLICIT")
        self.assertEqual(len(lines), 42)
        self.assertEqual(len(lines), len(set(lines)))
        for line in lines:
            url = urlsplit(line)
            self.assertEqual(url.scheme, "https")
            self.assertIn(url.netloc, ("software.repos.intel.com", "conda.anaconda.org"))
            self.assertEqual(url.query, "")
            self.assertRegex(url.fragment, r"^[a-f0-9]{64}$")
            self.assertNotIn("/home/", url.path)
            self.assertNotIn("/Users/", url.path)

    def test_package_manifest_is_complete_order_independent_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory); metadata = prefix / "conda-meta"; metadata.mkdir()
            for index, name in enumerate(reversed(sycl.COMPONENTS)):
                (metadata / f"{index}.json").write_text(json.dumps({"name": name, "version": "2026.2.0", "build": "fixture", "sha256": "a" * 64}))
            before = sycl.stack_manifest(prefix)
            self.assertEqual([r["name"] for r in before["packages"]], sorted(sycl.COMPONENTS))
            path = metadata / "0.json"; row = json.loads(path.read_text()); row["sha256"] = "b" * 64; path.write_text(json.dumps(row))
            self.assertNotEqual(before["sha256"], sycl.stack_manifest(prefix)["sha256"])
            path.unlink()
            with self.assertRaisesRegex(ValueError, "package identity is missing"):
                sycl.stack_manifest(prefix)

    def fixture_maps(self, prefix):
        lines = []
        for name in ("libsycl.so.9", "libur_adapter_level_zero_v2.so.0"):
            path = prefix / name; path.write_bytes(name.encode())
            lines.append("00001000-00002000 r-xp 00000000 00:00 0 " + str(path))
        return "\n".join(lines)

    def test_live_maps_prove_both_native_libraries_and_hide_private_paths(self):
        with tempfile.TemporaryDirectory(prefix="native maps ") as directory:
            prefix = Path(directory); text = self.fixture_maps(prefix)
            result = sycl.runtime_evidence_from_maps(text, prefix)
            self.assertEqual(result["runtime"], "level_zero")
            self.assertEqual(set(result["library_sha256"]), {"libsycl.so.9", "libur_adapter_level_zero_v2.so.0"})
            self.assertNotIn(str(prefix), json.dumps(result))
            for line in text.splitlines():
                with self.subTest(line=line), self.assertRaisesRegex(ValueError, "SYCL and Level Zero"):
                    sycl.runtime_evidence_from_maps(line, prefix)

    def test_old_or_deleted_runtime_cannot_be_labeled_as_the_new_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = root / "selected"; prefix.mkdir(); old = root / "old"; old.mkdir()
            text = self.fixture_maps(old)
            with self.assertRaisesRegex(ValueError, "outside its selected prefix"):
                sycl.runtime_evidence_from_maps(text, prefix)
            text = self.fixture_maps(prefix)
            with self.assertRaisesRegex(ValueError, "changed while"):
                sycl.runtime_evidence_from_maps(text + " (deleted)", prefix)


class OneDnnDevelopmentTest(unittest.TestCase):
    """These file/CMake fixtures never load a library or initialize a GPU."""

    def sdk(self, root):
        prefix = root / "SDK with spaces"; metadata = prefix / "conda-meta"; metadata.mkdir(parents=True)
        for name in sycl.COMPONENTS:
            row = {"name": name, "version": "2026.1.0" if name.startswith("onednn") else "2026.2.0",
                   "build": "intel_617" if name.startswith("onednn") else "fixture", "sha256": "a" * 64}
            (metadata / (name + ".json")).write_text(json.dumps(row))
        headers = prefix / "include/oneapi/dnnl"; headers.mkdir(parents=True)
        (headers / "dnnl_version.h").write_text("#define DNNL_VERSION_MAJOR 3\n#define DNNL_VERSION_MINOR 13\n#define DNNL_VERSION_PATCH 2\n")
        (headers / "dnnl_config.h").write_text("#define DNNL_GPU_RUNTIME DNNL_RUNTIME_SYCL\n#define DNNL_GPU_VENDOR DNNL_VENDOR_INTEL\n")
        (prefix / "lib").mkdir(); (prefix / "lib/libdnnl.so.3.13").write_bytes(b"CPU fixture, never load this file")
        (prefix / "lib/libdnnl.so").symlink_to("libdnnl.so.3.13")
        return prefix

    def test_runtime_only_or_mismatched_development_packages_fail_before_generation(self):
        for mutation in ("missing", "version", "build", "checksum", "both-builds-missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                prefix = self.sdk(Path(directory)); path = prefix / "conda-meta/onednn-devel.json"
                row = json.loads(path.read_text())
                if mutation == "both-builds-missing":
                    for name in ("onednn", "onednn-devel"):
                        item = prefix / ("conda-meta/" + name + ".json")
                        data = json.loads(item.read_text()); data.pop("build"); item.write_text(json.dumps(data))
                elif mutation == "missing": path.unlink()
                else:
                    row[mutation if mutation != "checksum" else "sha256"] = "different"
                    path.write_text(json.dumps(row))
                with self.assertRaises(ValueError): sycl.prepare_onednn_cmake(prefix)
                self.assertFalse((prefix / "lib/cmake/dnnl").exists())

    def test_cpu_only_wrong_abi_missing_and_external_files_cannot_be_advertised_as_a_gpu_sdk(self):
        for mutation in ("cpu", "other-vendor", "wrong-abi", "missing-header", "duplicate-version", "external-library", "external-header"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); prefix = self.sdk(root)
                header = prefix / "include/oneapi/dnnl/dnnl_version.h"
                config = prefix / "include/oneapi/dnnl/dnnl_config.h"
                if mutation == "cpu": config.write_text(config.read_text().replace("DNNL_RUNTIME_SYCL", "DNNL_RUNTIME_NONE"))
                elif mutation == "other-vendor": config.write_text(config.read_text().replace("DNNL_VENDOR_INTEL", "DNNL_VENDOR_ANY"))
                elif mutation == "wrong-abi": header.write_text(header.read_text().replace("MINOR 13", "MINOR 11"))
                elif mutation == "missing-header": header.unlink()
                elif mutation == "duplicate-version": header.write_text(header.read_text() + "#define DNNL_VERSION_MINOR 11\n")
                elif mutation == "external-library":
                    outside = root / "libdnnl.so.3.13"; outside.write_bytes(b"host fixture")
                    library = prefix / "lib/libdnnl.so"; library.unlink(); library.symlink_to(outside)
                elif mutation == "external-header":
                    outside = root / "host-header"; outside.write_text(header.read_text()); header.unlink(); header.symlink_to(outside)
                with self.assertRaises(ValueError): sycl.prepare_onednn_cmake(prefix)
                self.assertFalse((prefix / "lib/cmake/dnnl").exists())

    def test_adapter_is_relocatable_idempotent_and_keeps_hashes_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = self.sdk(root)
            before = sycl.stack_manifest(prefix)
            first = sycl.prepare_onednn_cmake(prefix)
            self.assertIn("not an Intel-supplied CMake config", first["origin"])
            self.assertEqual(first, sycl.prepare_onednn_cmake(prefix))
            self.assertEqual(first["development"]["version"], "3.13.2")
            self.assertEqual(first["development"]["library_name"], "libdnnl.so.3.13")
            self.assertEqual(before, sycl.stack_manifest(prefix))
            config = prefix / "lib/cmake/dnnl/dnnl-config.cmake"
            self.assertNotIn(str(root), config.read_text())
            self.assertNotIn(str(root), json.dumps(first))
            header = prefix / "include/oneapi/dnnl/dnnl_version.h"
            header.write_text(header.read_text().replace("PATCH 2", "PATCH 3"))
            old = config.read_bytes()
            with self.assertRaisesRegex(ValueError, "preserve the existing"):
                sycl.prepare_onednn_cmake(prefix)
            self.assertEqual(config.read_bytes(), old)

    def test_existing_and_external_cmake_configs_are_preserved(self):
        for mutation in ("existing", "external-directory", "external-file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); prefix = self.sdk(root); location = prefix / "lib/cmake/dnnl"
                outside = root / "outside"; outside.mkdir()
                (prefix / "lib/cmake").mkdir()
                if mutation == "external-directory": location.symlink_to(outside)
                else:
                    location.mkdir()
                    path = location / "dnnl-config.cmake"
                    if mutation == "existing": path.write_text("# existing package configuration\n")
                    else:
                        target = outside / "dnnl-config.cmake"; target.write_text("# preserve me\n"); path.symlink_to(target)
                with self.assertRaises(ValueError): sycl.prepare_onednn_cmake(prefix)
                self.assertEqual(list(outside.iterdir()), [] if mutation != "external-file" else [outside / "dnnl-config.cmake"])

    def test_cmake_resolves_the_selected_gpu_headers_and_library_without_host_fallback(self):
        cmake = shutil.which("cmake")
        self.assertIsNotNone(cmake, "CMake is required for native SDK compatibility fixtures")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = self.sdk(root); sycl.prepare_onednn_cmake(prefix)
            source = root / "consumer"; source.mkdir(); build = root / "build"
            (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(OneDnnAdapterFixture NONE)
find_package(DNNL CONFIG REQUIRED NO_DEFAULT_PATH PATHS "''' + (prefix / "lib/cmake/dnnl").as_posix() + '''")
if(NOT DNNL_GPU_VENDOR STREQUAL "INTEL" OR NOT DNNL_VERSION STREQUAL "3.13.2")
  message(FATAL_ERROR "Incorrect package identity")
endif()
get_target_property(location DNNL::dnnl IMPORTED_LOCATION_RELEASE)
get_target_property(headers DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${location}\\n${headers}\\n")
''')
            result = subprocess.run([cmake, "-S", str(source), "-B", str(build)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / "selected.txt").read_text().splitlines(), [str(prefix / "lib/libdnnl.so"), str(prefix / "include")])

    def test_qualified_build_rejects_implicit_or_host_oneDNN_development_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = self.sdk(root); sycl.prepare_onednn_cmake(prefix)
            source = root / "source"; source.mkdir(); build = root / "build"; binary = build / "bin"; binary.mkdir(parents=True)
            values = {"CMAKE_HOME_DIRECTORY": str(source), "CMAKE_BUILD_TYPE": "Release", "GGML_SYCL": "ON",
                      "GGML_VULKAN": "OFF", "GGML_CUDA": "OFF", "GGML_METAL": "OFF", "GGML_NATIVE": "OFF",
                      "GGML_SYCL_HOST_MEM_FALLBACK": "OFF", "GGML_SYCL_SUPPORT_XMX": "ON",
                      "GGML_SYCL_GRAPH": "ON", "GGML_SYCL_DNN": "ON", "GGML_SYCL_F16": "OFF"}
            for config in (None, str(root / "host-dnnl"), str(prefix / "lib/cmake/dnnl")):
                data = dict(values)
                if config is not None: data["DNNL_DIR"] = config
                (build / "CMakeCache.txt").write_text("\n".join(k + ":STRING=" + v for k, v in data.items()))
                if config != str(prefix / "lib/cmake/dnnl"):
                    with self.assertRaisesRegex(ValueError, "inside the selected SDK"):
                        sycl.build_manifest(binary, source, prefix)
                else:
                    report = sycl.build_manifest(binary, source, prefix)
                    self.assertEqual(report["onednn_development"]["version"], "3.13.2")
                    self.assertNotIn(str(root), json.dumps(report))
            config = prefix / "lib/cmake/dnnl/dnnl-config.cmake"
            original = config.read_text()
            config.write_text(original.replace('${_ZINC_DNNL_PREFIX}/include', '/host-sdk/include'))
            with self.assertRaisesRegex(ValueError, "prefix-only configuration"):
                sycl.build_manifest(binary, source, prefix)
            config.write_text(original)
            outside = root / "host-config.cmake"; outside.write_text(config.read_text())
            config.unlink(); config.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "missing or external"):
                sycl.build_manifest(binary, source, prefix)

    def test_live_runtime_must_match_the_recorded_development_library_even_within_the_same_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self.sdk(Path(directory)); development = sycl.onednn_development(prefix)
            text = SyclEnvironmentTest().fixture_maps(prefix)
            library = (prefix / "lib/libdnnl.so").resolve()
            text += "\n00002000-00003000 r-xp 00000000 00:00 0 " + str(library)
            self.assertEqual(sycl.runtime_evidence_from_maps(text, prefix, development)["library_sha256"][library.name], development["library_sha256"])
            with self.assertRaisesRegex(ValueError, "differs from the selected"):
                sycl.runtime_evidence_from_maps(text.rsplit("\n", 1)[0], prefix, development)
            library.write_bytes(b"same filename, changed runtime")
            with self.assertRaisesRegex(ValueError, "differs from the selected"):
                sycl.runtime_evidence_from_maps(text, prefix, development)

    def test_malformed_snapshot_cannot_skip_live_library_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self.sdk(Path(directory)); text = SyclEnvironmentTest().fixture_maps(prefix)
            for development in ({}, [], {"library_name": "libdnnl.so.3.13"},
                                {"library_sha256": "a" * 64},
                                {"library_name": "host-library", "library_sha256": "a" * 64},
                                {"library_name": "libdnnl.so.3.13", "library_sha256": "invalid"}):
                with self.subTest(development=development), self.assertRaisesRegex(ValueError, "recorded oneDNN"):
                    sycl.runtime_evidence_from_maps(text, prefix, development)


if __name__ == "__main__":
    unittest.main()
