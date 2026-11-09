"""CPU fixtures for native-baseline isolation; no Intel SDK or GPU is used."""
import json
import ctypes
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
            self.assertEqual(sycl.environment(prefix, base, graphs=True)["GGML_SYCL_ENABLE_GRAPH"], "1")
            with self.assertRaisesRegex(ValueError, "graph selection"):
                sycl.environment(prefix, base, graphs="0")

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
        self.assertEqual(len(lines), 41)
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


if __name__ == "__main__":
    unittest.main()
