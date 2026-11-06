import gzip
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from check_repository_privacy import inspect_file, inspect_text, read_private_hosts
from image_metadata import PNG_SIGNATURE, clean_image, orientation_exif, read_orientation

ROOT = Path(__file__).resolve().parent.parent


def segment(marker, payload):
    return bytes((255, marker)) + struct.pack(">H", len(payload) + 2) + payload


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


class ImageMetadataTests(unittest.TestCase):
    def test_jpeg_keeps_orientation_color_and_both_progressive_scans(self):
        scan = segment(0xDA, b"scan-header") + b"pixel\xff\x00data\xff\xd0"
        exif = orientation_exif(6)
        # Extra TIFF bytes stand for the old thumbnail/GPS/private sub-IFDs.
        source = (b"\xff\xd8" + segment(0xE1, exif + b"private GPS thumbnail") +
                  segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00private XMP") +
                  segment(0xED, b"Photoshop IPTC") + segment(0xFE, b"comment") +
                  segment(0xE2, b"ICC_PROFILE\x00color") + segment(0xEE, b"Adobe color") +
                  scan + segment(0xE1, b"late private XMP") + scan + b"\xff\xd9private trailer")
        result = clean_image(source)
        self.assertEqual(result, b"\xff\xd8" + segment(0xE1, exif) +
                         segment(0xE2, b"ICC_PROFILE\x00color") + segment(0xEE, b"Adobe color") +
                         scan + scan + b"\xff\xd9")
        self.assertEqual(read_orientation(exif), 6)
        self.assertEqual(clean_image(result), result)

    def test_big_endian_orientation(self):
        exif = b"Exif\x00\x00" + struct.pack(">2sHIHHHIHI", b"MM", 42, 8, 1, 274, 3, 1, 8, 0)
        self.assertEqual(read_orientation(exif), 8)

    def test_png_preserves_pixels_and_color(self):
        safe = PNG_SIGNATURE + chunk(b"IHDR", b"header") + chunk(b"iCCP", b"color")
        pixels = chunk(b"IDAT", b"compressed pixels") + chunk(b"IEND", b"")
        private = b"".join(chunk(k, b"private") for k in (b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"))
        self.assertEqual(clean_image(safe + private + pixels), safe + pixels)

    def test_gif_keeps_animation_and_drops_comments(self):
        header = b"GIF89a" + struct.pack("<HHBBB", 1, 1, 0, 0, 0)
        loop = b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
        control = b"\x21\xf9\x04\x00\x01\x00\x00\x00"
        frame = b"\x2c" + struct.pack("<HHHHB", 0, 0, 1, 1, 0) + b"\x02\x02\x44\x01\x00"
        comment = b"\x21\xfe\x07private\x00"
        clean = header + loop + control + frame + b";"
        self.assertEqual(clean_image(header + loop + comment + control + frame + b";"), clean)
        self.assertEqual(clean_image(clean), clean)

    def test_malformed_images_fail_closed(self):
        for data in (b"\xff\xd8\xff\xe1\xff\xff", PNG_SIGNATURE, b"GIF89a", b"unknown"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                clean_image(data)


class RepositoryPrivacyTests(unittest.TestCase):
    def test_private_details_and_encoded_paths_are_redacted(self):
        host = ".".join(map(str, (10, 23, 45, 67)))
        text = "\n".join(("/Users/" + "private-person/work", "C:\\Users\\" + "private-person\\work",
                          "/home%2F" + "private-person/work", host, "GPU-" + "01234567-89ab-cdef-0123-456789abcdef",
                          "worker.example." + "ts.net", "fd12" + ":3456::1"))
        issues = inspect_text("notes.md", text)
        self.assertEqual(len(issues), 7)
        output = repr(issues)
        for value in (host, "private-person", "worker.example", "01234567"):
            self.assertNotIn(value, output)

    def test_public_urls_system_paths_versions_and_loopback_are_valid(self):
        self.assertFalse(inspect_text("docs.md", "https://github.com/zolotukhin/zinc /opt/rocm /root/models/test.gguf 127.0.0.1 0.0.0.0 0.17.0 192.0.2.1"))
        self.assertFalse(inspect_text("example.test.ts", "/home/" + "alice/work /Users/example/work"))
        self.assertTrue(inspect_text("docs.md", "/home/" + "alice/work"))

    def test_compressed_evidence_is_not_exempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result.json.gz").write_bytes(gzip.compress(json.dumps({"path": "/Users/" + "private-person/work"}).encode()))
            self.assertEqual(inspect_file(root, "result.json.gz"), {("result.json.gz", 1, "personal home path")})
            (root / "bad.gz").write_bytes(b"invalid gzip")
            self.assertTrue(inspect_file(root, "bad.gz"))

    def test_private_env_is_read_without_executing_or_disclosing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("export ZINC_HOST='worker.example.invalid' # private\nZINC_PORT=2222\nZINC_USER=example\n")
            hosts = read_private_hosts(path)
            issues = inspect_text("notes.md", "ssh worker.example.invalid true", hosts)
            self.assertEqual(issues, {("notes.md", 1, "configured private endpoint")})

    def test_image_metadata_and_credential_filenames_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_bytes(b"\xff\xd8" + segment(0xE1, b"private XMP") + b"\xff\xd9")
            self.assertIn(("photo.jpg", 1, "identifying image metadata"), inspect_file(root, "photo.jpg"))
            (root / ".env").write_text("EXAMPLE=value")
            self.assertIn((".env", 1, "credential file"), inspect_file(root, ".env"))
            (root / ".env.local").write_text("EXAMPLE=value")
            self.assertIn((".env.local", 1, "credential file"), inspect_file(root, ".env.local"))
            (root / ".env.example").write_text("EXAMPLE=value")
            self.assertFalse(inspect_file(root, ".env.example"))


class CudaConfigurationTests(unittest.TestCase):
    def select(self, **variables):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ZINC_GPU", "CUDA_VISIBLE_DEVICES"))}
        env.update(variables)
        return subprocess.run(["bash", "-uc", 'source "$1"; zinc_select_cuda_device || exit; printf "%s" "$CUDA_VISIBLE_DEVICES"',
                               "test", str(ROOT / "scripts/lib/cuda_target.sh")], env=env, capture_output=True, text=True)

    def test_missing_device_or_unconfigured_role_fails_before_a_workload(self):
        for env in ({}, {"ZINC_GPU": "4090"}, {"ZINC_GPU": "5090"}, {"ZINC_GPU": "0,1"}):
            self.assertNotEqual(self.select(**env).returncode, 0)

    def test_explicit_ordinals_roles_and_environment_are_supported(self):
        identifier = "GPU-" + "01234567-89ab-cdef-0123-456789abcdef"
        for env, expected in (({"ZINC_GPU": "0"}, "0"), ({"CUDA_VISIBLE_DEVICES": "1"}, "1"),
                              ({"ZINC_GPU": "4090", "ZINC_GPU_4090": identifier}, identifier),
                              ({"ZINC_GPU": "5090", "ZINC_GPU_5090": "2"}, "2")):
            result = self.select(**env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)


@unittest.skipUnless(os.environ.get("GITLEAKS_BINARY"), "set GITLEAKS_BINARY to verify credential scanner exceptions")
class CredentialScannerTests(unittest.TestCase):
    def test_checksum_exceptions_do_not_hide_realistic_keys_on_the_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Synthetic hex key has the same shape as a checksum, but a
            # credential field must still be detected on this minified line.
            key = hashlib.sha256(b"synthetic credential scanner regression").hexdigest()
            (root / "evidence.json").write_text(json.dumps({"tokenizer_sha256": key, "api_key": key}))
            report = root / "report.json"
            run = subprocess.run([os.environ["GITLEAKS_BINARY"], "dir", str(root), "--config", str(ROOT / ".gitleaks.toml"),
                                  "--redact=100", "--report-format=json", "--report-path", str(report), "--no-banner"], capture_output=True)
            self.assertEqual(run.returncode, 1)
            findings = json.loads(report.read_text())
            self.assertEqual(len(findings), 1)
            self.assertIn("api_key", findings[0]["Match"])
            self.assertNotIn(key, report.read_text())


if __name__ == "__main__":
    unittest.main()
