"""CPU fixtures for native-baseline isolation; no Intel SDK or GPU is used."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlsplit

import intel_sycl_environment as sycl


class SyclEnvironmentTest(unittest.TestCase):
    def test_native_loader_environment_is_scoped_and_discards_inherited_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory); (prefix / "bin").mkdir(); (prefix / "lib").mkdir()
            (prefix / "bin/icpx").write_bytes(b"fixture")
            base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/old", "GGML_SYCL_ENABLE_XMX": "0",
                    "SYCL_DEVICE_FILTER": "cpu", "ONEAPI_DEVICE_SELECTOR": "opencl:cpu", "ZE_AFFINITY_MASK": "other",
                    "ZINC_MTP": "0"}
            original = dict(base); result = sycl.environment(prefix, base)
            self.assertEqual(base, original)
            self.assertEqual(result["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
            self.assertEqual(result["LD_LIBRARY_PATH"], str(prefix.resolve() / "lib"))
            self.assertTrue(result["PATH"].startswith(str(prefix.resolve() / "bin")))
            for key in ("GGML_SYCL_ENABLE_XMX", "SYCL_DEVICE_FILTER", "ZE_AFFINITY_MASK"):
                self.assertNotIn(key, result)
            self.assertEqual(result["ZINC_MTP"], "0")
            self.assertEqual(result["GGML_SYCL_ENABLE_GRAPH"], "0")
            self.assertEqual(sycl.environment(prefix, base, graphs=True)["GGML_SYCL_ENABLE_GRAPH"], "1")
            with self.assertRaisesRegex(ValueError, "graph selection"):
                sycl.environment(prefix, base, graphs="0")

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
