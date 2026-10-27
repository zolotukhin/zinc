"""Release contract tests. Fixtures execute CPU-only C programs, never a GPU/model."""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import release
import release_gpu
import release_lib
import release_publish
import release_sign
from release_lib import (
    ROOT, archive_name, check_metadata, extract_archive, inventory, policy,
    read_json, sha256, target, version, write_json,
)

COMMIT = "1" * 40
NATIVE = "macos-aarch64-metal" if sys.platform == "darwin" else "linux-x86_64-vulkan"


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zinc release test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def tree(self, release_version="9.8.7", exit_code=0):
        spec = target(NATIVE)
        tree = self.root / "release" / ("v" + release_version) / archive_name(release_version, NATIVE).removesuffix(".tar.gz")
        (tree / "bin").mkdir(parents=True)
        report = (f"zinc {release_version}\ncommit: {COMMIT}\n"
                  f"target: {'aarch64-macos-none' if sys.platform == 'darwin' else 'x86_64-linux-gnu'}\n"
                  f"optimize: fast\nbackends: {spec['backend']}\n")
        source = self.root / "fixture.c"
        source.write_text("#include <stdio.h>\nint main(void) { fputs(" + json.dumps(report) + f", stdout); return {exit_code}; }}\n")
        flags = ["-mmacosx-version-min=15.0", "-framework", "Metal", "-framework", "Foundation"] if sys.platform == "darwin" else ["-Wl,--no-as-needed", "-lvulkan"]
        subprocess.run(["cc", str(source), "-o", str(tree / "bin/zinc"), *flags], check=True, capture_output=True)
        shutil.copy2(tree / "bin/zinc", tree / "bin/zinc-tokenize")
        asset = "share/zinc/shaders/metal/fixture.metal" if sys.platform == "darwin" else "share/zinc/shaders/fixture.spv"
        (tree / asset).parent.mkdir(parents=True)
        (tree / asset).write_text("fixture shader\n")
        write_json(tree / "share/zinc/asset-manifest.json", {"schema_version": 1, "backend": spec["backend"], "files": [asset]})
        for name in ["README.md", "LICENSE"]:
            (tree / name).write_text(name + "\n")
        meta = {"schema_version": 1, "name": "zinc", "version": release_version, "tag": "v" + release_version,
                "commit": COMMIT, "target": NATIVE, "backend": spec["backend"], "cpu": "baseline",
                "optimize": "ReleaseFast", "minimum_os": spec["minimum_os"], "minimum_glibc": spec["minimum_glibc"]}
        write_json(tree / "VERSION.json", meta)
        receipt = {"format": "1", "version": release_version, "commit": COMMIT, "target": NATIVE,
                   "backend": spec["backend"], "minimum_os": spec["minimum_os"], "minimum_glibc": spec["minimum_glibc"],
                   "binary_sha256": sha256(tree / "bin/zinc")}
        (tree / "INSTALL-METADATA").write_text("".join(f"{k}={v}\n" for k, v in receipt.items()))
        self.refresh(tree)
        return tree

    def refresh(self, tree):
        if (tree / "FILES.sha256").exists():
            (tree / "FILES.sha256").unlink()
        (tree / "FILES.sha256").write_text("".join(f"{h}  {f}\n" for f, h in inventory(tree).items()))

    def bundle(self, tree, extras=(), duplicate_checksum=False):
        archive = tree.parent / (tree.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(tree, arcname=tree.name)
            for member in extras:
                stream.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
        line = sha256(archive) + "  " + archive.name + "\n"
        (tree.parent / "SHA256SUMS.txt").write_text(line * (2 if duplicate_checksum else 1))
        return archive

    def install(self, release_version="9.8.7", *args, **overrides):
        env = {**os.environ, "ZINC_VERSION": "v" + release_version,
               "ZINC_BASE_URL": (self.root / "release").as_uri(),
               "ZINC_INSTALL_DIR": str(self.root / "installation"), "ZINC_BIN_DIR": str(self.root / "commands")}
        env.pop("ZINC_BACKEND", None)
        env.update(overrides)
        return subprocess.run(["bash", str(ROOT / "scripts/install.sh"), *args], cwd=self.root,
                              env=env, capture_output=True, text=True, timeout=60)

    def assert_installed(self, expected):
        self.assertEqual((self.root / "installation/current").resolve().name,
                         archive_name(expected, NATIVE).removesuffix(".tar.gz"))

    def test_semver_rejects_paths_shell_text_and_numeric_prerelease_zeroes(self):
        for value in ["../0.1.0", "v0.1.0", "01.0.0", "1.2", "0.1.0-01", "0.1.0\n", "$(touch marker)"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                version(value)
        for value in ["0.1.0", "0.1.0-rc.1", "0.1.0+build-01", "1.2.3-0"]:
            self.assertEqual(version(value), value)

    def test_policy_has_three_backends_and_nine_required_hardware_profiles(self):
        data = policy()
        self.assertEqual({x["backend"] for x in data["targets"]}, {"metal", "vulkan", "rocm"})
        self.assertEqual(len(data["gpu_profiles"]), 9)

    def test_rocm_archive_resolves_the_standard_runtime_without_shell_setup(self):
        binary = self.root / "fixture-elf"
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (62).to_bytes(2, "little")
        binary.write_bytes(header)
        needed = "\n".join(f"(NEEDED) Shared library: [{name}]" for name in
                           ["libc.so.6", "libamdhip64.so.7", "libhiprtc.so.7", "libhipblas.so.3"])
        for paths, valid in [(None, False), ("/opt/rocm/lib", True),
                             ("$ORIGIN:/opt/rocm/lib", True), ("/tmp/toolkit/lib", False)]:
            dynamic = needed + (f"\n(RUNPATH) Library runpath: [{paths}]" if paths else "")
            def inspect(command):
                return "GLIBC_2.39" if "--version-info" in command else dynamic
            with self.subTest(paths=paths), patch.object(release, "command_output", side_effect=inspect):
                if valid:
                    self.assertEqual(release.inspect_binary(binary, target("linux-x86_64-rocm"))["runtime_paths"], [paths])
                else:
                    with self.assertRaises(ValueError):
                        release.inspect_binary(binary, target("linux-x86_64-rocm"))

    def test_missing_gpu_configuration_fails_instead_of_skipping(self):
        with patch.dict(os.environ, {"RELEASE_VARIABLES": "{}"}):
            with self.assertRaisesRegex(ValueError, "configuration missing"):
                release.matrix(type("Args", (), {"kind": "gpu"})())

    def test_package_and_verify_outside_checkout_with_real_native_executable(self):
        tree = self.tree()
        # Package the fixture as an install prefix, so the real asset lookup and
        # architecture/version checks run. It does not initialize any GPU.
        result = subprocess.run(["bash", str(ROOT / "scripts/package_release.sh"), "--version", "9.8.7",
                                 "--target", NATIVE, "--prefix", str(tree), "--dist", str(self.root / "dist"),
                                 "--commit", COMMIT], cwd=self.root,
                                env={**os.environ, "SOURCE_DATE_EPOCH": "1700000000"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.root / "dist" / archive_name("9.8.7", NATIVE)
        extracted = extract_archive(archive, self.root / "extract", "9.8.7", NATIVE, COMMIT)
        self.assertEqual(check_metadata(extracted, "9.8.7", NATIVE)["commit"], COMMIT)
        self.assertTrue((extracted / "bin/zinc").stat().st_mode & 0o111)

    def test_package_refuses_source_identity_mismatch_and_existing_archive(self):
        tree = self.tree()
        args = ["python3", str(ROOT / "tools/release.py"), "package", "--version", "9.8.7", "--target", NATIVE,
                "--prefix", str(tree), "--dist", str(self.root / "dist"), "--commit"]
        wrong = subprocess.run([*args, "2" * 40], capture_output=True, text=True)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("commit/backend", wrong.stderr)

    def test_archive_links_traversal_and_duplicates_are_rejected_before_extraction(self):
        tree = self.tree()
        for index, (name, kind) in enumerate([("../escape", tarfile.REGTYPE),
                                            (tree.name + "/linked", tarfile.SYMTYPE),
                                            (tree.name + "/hard", tarfile.LNKTYPE),
                                            (tree.name + "/VERSION.json", tarfile.REGTYPE)]):
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/outside"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            archive = self.bundle(tree, [member])
            dest = self.root / ("unsafe-" + str(index))
            with self.subTest(name=name), self.assertRaises(ValueError):
                extract_archive(archive, dest, "9.8.7", NATIVE)
            self.assertFalse(any(dest.iterdir()))

    def test_file_inventory_detects_missing_and_modified_runtime_assets(self):
        tree = self.tree()
        shader = next((tree / "share/zinc/shaders").rglob("fixture.*"))
        shader.write_text("modified")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            check_metadata(tree, "9.8.7", NATIVE)
        shader.unlink()
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            check_metadata(tree, "9.8.7", NATIVE)

    def test_installer_native_install_spaces_idempotence_and_bin_link_repair(self):
        tree = self.tree()
        self.bundle(tree)
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assert_installed("9.8.7")
        (self.root / "commands/zinc").unlink()
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue((self.root / "commands/zinc").is_symlink())
        self.assertFalse((self.root / "installation/previous").exists())

    def test_upgrade_and_explicit_rollback_preserve_model_cache(self):
        self.bundle(self.tree("9.8.7"))
        self.assertEqual(self.install("9.8.7").returncode, 0)
        cache = self.root / "models/user-model.gguf"
        cache.parent.mkdir()
        cache.write_text("untouched")
        self.bundle(self.tree("9.8.8"))
        update = self.install("9.8.8")
        self.assertEqual(update.returncode, 0, update.stderr)
        self.assert_installed("9.8.8")
        rollback = self.install("9.8.8", "--rollback")
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        self.assert_installed("9.8.7")
        self.assertEqual(cache.read_text(), "untouched")

    def test_non_executable_download_never_replaces_working_current(self):
        self.bundle(self.tree("9.8.7"))
        self.assertEqual(self.install("9.8.7").returncode, 0)
        self.bundle(self.tree("9.8.8", exit_code=42))
        failure = self.install("9.8.8")
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("--version failed", failure.stderr)
        self.assert_installed("9.8.7")
        self.assertFalse((self.root / "installation/.install.lock").exists())

    def test_wrong_archive_version_is_rejected(self):
        tree = self.tree()
        metadata = read_json(tree / "VERSION.json")
        metadata["version"] = "9.8.6"
        write_json(tree / "VERSION.json", metadata)
        self.refresh(tree)
        self.bundle(tree)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata disagree", result.stderr)

    def test_missing_duplicate_or_corrupt_archive_checksums_fail(self):
        tree = self.tree()
        archive = self.bundle(tree, duplicate_checksum=True)
        duplicate = self.install()
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("duplicate checksum", duplicate.stderr)
        (tree.parent / "SHA256SUMS.txt").write_text("a" * 64 + "  other.tar.gz\n")
        self.assertNotEqual(self.install().returncode, 0)
        (tree.parent / "SHA256SUMS.txt").write_text("a" * 64 + "  " + archive.name + "\n")
        corrupt = self.install()
        self.assertNotEqual(corrupt.returncode, 0)
        self.assertIn("checksum mismatch", corrupt.stderr)

    def test_installer_rejects_archive_links_traversal_and_duplicate_members(self):
        tree = self.tree()
        for name, kind in [("../escape", tarfile.REGTYPE), (tree.name + "/share/link", tarfile.SYMTYPE),
                           (tree.name + "/share/hard", tarfile.LNKTYPE), (tree.name + "/VERSION.json", tarfile.REGTYPE)]:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = kind, "/outside"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            self.bundle(tree, [member])
            with self.subTest(name=name):
                self.assertNotEqual(self.install().returncode, 0)
                self.assertFalse((self.root / "installation/current").exists())
        self.assertFalse((self.root / "escape").exists())

    def test_installer_fails_when_current_points_at_an_unmanaged_directory(self):
        self.bundle(self.tree())
        (self.root / "installation/current").mkdir(parents=True)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged", result.stderr)

    def test_installer_does_not_adopt_an_unmanaged_current_symlink(self):
        self.bundle(self.tree())
        outside = self.root / "outside installation"
        outside.mkdir()
        (outside / "keep").write_text("keep this installation")
        (self.root / "installation").mkdir()
        (self.root / "installation/current").symlink_to(outside)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged", result.stderr)
        self.assertEqual((self.root / "installation/current").resolve(), outside.resolve())
        self.assertEqual((outside / "keep").read_text(), "keep this installation")

    def test_explicit_prerelease_and_build_metadata_version(self):
        self.bundle(self.tree("9.8.7-rc.1+build-01"))
        result = self.install("9.8.7-rc.1+build-01")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unconfigured_backend_and_invalid_version_do_not_execute_shell(self):
        result = self.install(ZINC_BACKEND="cuda")
        self.assertNotEqual(result.returncode, 0)
        result = self.install(ZINC_VERSION="v0.1.0;touch marker")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "marker").exists())

    def candidate(self):
        directory, reports = self.root / "candidate", self.root / "reports"
        directory.mkdir()
        reports.mkdir()
        artifacts = []
        for spec in policy()["targets"]:
            archive = directory / archive_name("9.8.7", spec["id"])
            archive.write_bytes(("CPU-only gate fixture: " + spec["id"]).encode())
            artifacts.append({"target": spec["id"], "filename": archive.name, "size_bytes": archive.stat().st_size,
                              "sha256": sha256(archive), "binary_sha256": "a" * 64,
                              "signing": {"status": "notarized" if spec["backend"] == "metal" else "unsigned"}})
        data = {"schema_version": 1, "version": "9.8.7", "commit": COMMIT, "candidate_run": "123", "workflow_sha": COMMIT,
                "policy_sha256": sha256(ROOT / "config/release/targets.json"), "artifacts": artifacts,
                "required_profiles": [{k: p[k] for k in ("id", "target", "family")} for p in policy()["gpu_profiles"]]}
        write_json(directory / "release-manifest.json", data)
        (directory / "SHA256SUMS.txt").write_text("".join(a["sha256"] + "  " + a["filename"] + "\n" for a in artifacts))
        for profile in policy()["gpu_profiles"]:
            artifact = next(a for a in artifacts if a["target"] == profile["target"])
            models = [dict(id=m["id"], sha256=m["sha256"], context=m["context"], passed=True,
                           reference_agreement=release_lib.reference_agreement([{"tokens": 16, "near_ties": []} for _ in range(3)]))
                      for m in policy()["models"] if m["id"] in profile["models"]]
            report = {"schema_version": 1, "profile": profile["id"], "target": profile["target"], "family": profile["family"],
                      **{key: data[key] for key in ("version", "commit", "policy_sha256")},
                      "archive_sha256": artifact["sha256"], "binary_sha256": artifact["binary_sha256"], "models": models,
                      "hardware": {"gpu": "CPU fixture, no GPU was executed", "os": "fixture", "driver": "fixture"},
                      "options": {"parallel": 1, "kv_cache": "f32", "mtp": "backend-default", "greedy": True},
                      "reference": {"commit": policy()["reference"]["commit"], "generation": policy()["reference"]["generation"],
                                    "server_sha256": "b" * 64, "tokenizer_sha256": "c" * 64},
                      "checks": {name: True for name in ("package", "identity", "gpu", "model_hash", "tokenizer_parity", "inference_parity", "http", "cleanup", "kernel_reference")}}
            write_json(reports / (profile["id"] + ".json"), report)
        return directory, reports

    def aggregate(self, directory, reports):
        with contextlib.redirect_stdout(io.StringIO()):
            release.aggregate(SimpleNamespace(directory=directory, results=reports, version="9.8.7", commit=COMMIT))

    def test_every_policy_model_is_pinned_in_the_managed_catalog(self):
        catalog = (ROOT / "src/model/catalog.zig").read_text()
        for model in policy()["models"]:
            with self.subTest(model=model["id"]):
                entry = re.search(r'\.id = "' + re.escape(model["id"]) + r'",([\s\S]*?)(?=\n    \},)', catalog)
                self.assertIsNotNone(entry)
                self.assertIn('/resolve/' + model["revision"] + '/' + model["file"] + '?download=true', entry.group(1))
                self.assertIn('.sha256 = "' + model["sha256"] + '"', entry.group(1))
                self.assertIn('.size_bytes = ' + format(model["size_bytes"], '_') + ',', entry.group(1))

    def test_complete_nine_profile_evidence_is_accepted(self):
        directory, reports = self.candidate()
        self.aggregate(directory, reports)
        data = read_json(directory / "validation.json")
        self.assertEqual(data["status"], "passed")
        self.assertEqual(len(data["profiles"]), 9)

    def test_missing_duplicate_or_skipped_gpu_reports_block_a_candidate(self):
        directory, reports = self.candidate()
        report = reports / "metal-apple7.json"
        saved = report.read_text()
        report.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.aggregate(directory, reports)
        report.write_text(saved)
        duplicate = reports / "duplicate.json"
        duplicate.write_text(saved)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.aggregate(directory, reports)
        duplicate.unlink()
        data = json.loads(saved)
        data["checks"]["cleanup"] = False
        write_json(report, data)
        with self.assertRaisesRegex(ValueError, "failed or were skipped"):
            self.aggregate(directory, reports)

    def test_report_cannot_substitute_source_artifact_family_model_context_or_reference(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-rdna4.json"
        saved = path.read_text()
        changes = [lambda d: d.update(commit="2" * 40), lambda d: d.update(archive_sha256="0" * 64),
                   lambda d: d.update(binary_sha256="0" * 64), lambda d: d.update(family="rdna3"),
                   lambda d: d["models"][0].update(sha256="0" * 64), lambda d: d["models"][0].update(context=1),
                   lambda d: d["models"].pop(), lambda d: d["reference"].update(commit="2" * 40),
                   lambda d: d["models"][0].pop("reference_agreement"),
                   lambda d: d["models"][0]["reference_agreement"].update(exact_tokens=1),
                   lambda d: d["reference"].pop("generation"), lambda d: d["reference"]["generation"].update(micro_batch=2),
                   lambda d: d["checks"].pop("kernel_reference"), lambda d: d["hardware"].pop("driver")]
        for index, change in enumerate(changes):
            data = json.loads(saved)
            change(data)
            write_json(path, data)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.aggregate(directory, reports)

    def test_unsigned_metal_or_changed_archive_cannot_be_promoted(self):
        directory, _ = self.candidate()
        path = directory / "release-manifest.json"
        data = read_json(path)
        data["artifacts"][0]["signing"]["status"] = "unsigned"
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "not notarized"):
            release.verify_candidate(directory)
        data["artifacts"][0]["signing"]["status"] = "notarized"
        write_json(path, data)
        (directory / data["artifacts"][1]["filename"]).write_bytes(b"different archive")
        with self.assertRaisesRegex(ValueError, "digest"):
            release.verify_candidate(directory)

    def test_publisher_accepts_only_successful_manual_main_runs(self):
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        self.assertEqual(release_publish.validate_run(run, "123"), COMMIT)
        for key, value in [("event", "pull_request"), ("event", "schedule"), ("head_branch", "feature"), ("conclusion", "skipped"),
                           ("status", "in_progress"), ("path", ".github/workflows/test.yml"), ("head_repository", {"full_name": "someone/fork"})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                release_publish.validate_run({**run, key: value}, "123")

    def test_unprotected_environment_is_not_an_approval(self):
        for environment in [None, {}, {"protection_rules": []}, {"protection_rules": [{"type": "wait_timer"}]},
                            {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]}]:
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "required reviewers"):
                release_publish.validate_environment(environment)
        release_publish.validate_environment({"protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User"}]}]})

    def test_publisher_never_replaces_an_existing_published_release(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        with patch.object(release_publish, "api", side_effect=[{"object": {"type": "commit", "sha": COMMIT}}, {"draft": False}]) as api, patch.object(release_publish, "gh") as gh:
            with self.assertRaisesRegex(ValueError, "existing/published"):
                release_publish.publish(directory, data, "123")
            self.assertEqual(api.call_count, 2)
            gh.assert_not_called()

    def test_published_candidate_rerun_is_read_only_when_all_bytes_match(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        marker = f"<!-- zinc-candidate-run: 123; manifest: {sha256(directory / 'release-manifest.json')} -->"
        existing = {"draft": False, "body": marker}
        def command(args):
            self.assertEqual(args[:2], ["release", "download"])
            destination = Path(args[args.index("--dir") + 1])
            for path in directory.iterdir():
                shutil.copyfile(path, destination / path.name)
        with patch.object(release_publish, "api", side_effect=[{"object": {"type": "commit", "sha": COMMIT}}, existing]) as api, patch.object(release_publish, "gh", side_effect=command) as gh, contextlib.redirect_stdout(io.StringIO()):
            release_publish.publish(directory, data, "123")
            self.assertEqual(api.call_count, 2)
            self.assertEqual(gh.call_count, 1)

    def test_stable_catalog_requires_reviewed_starter_qualification(self):
        catalog = (ROOT / "src/model/catalog.zig").read_text()
        # Exercise both qualification states even after the real starter is
        # promoted; this contract must not lock the current experimental state.
        match = re.search(r'(\.id = "qwen3-0.6b-q8-0",[\s\S]*?)(?=\n    \},)', catalog)
        entry = re.sub(r'\.status = \.[a-z]+', '.status = .experimental', match.group(1))
        unqualified = catalog[:match.start()] + entry + catalog[match.end():]
        release_publish.validate_catalog(unqualified, "0.1.0-rc.1")
        with self.assertRaisesRegex(ValueError, "starter model to be qualified"):
            release_publish.validate_catalog(unqualified, "0.1.0")
        qualified = unqualified.replace(".status = .experimental", ".status = .supported", 1).replace(".tested_profiles = &.{}", '.tested_profiles = &.{"apple-silicon"}', 1)
        release_publish.validate_catalog(qualified, "0.1.0")

    def test_archive_decompression_limit_applies_before_tar_metadata_parsing(self):
        tree = self.tree()
        archive = self.bundle(tree)
        with patch.object(release_lib, "MAX_UNPACKED_BYTES", 512):
            with self.assertRaisesRegex(ValueError, "expands beyond"):
                extract_archive(archive, self.root / "bounded", "9.8.7", NATIVE)
        self.assertEqual(list((self.root / "bounded").iterdir()), [])

    def test_extra_executable_is_rejected_even_with_a_matching_inventory(self):
        tree = self.tree()
        (tree / "bin/unexpected").write_text("extra executable")
        self.refresh(tree)
        with self.assertRaisesRegex(ValueError, "unexpected packaged file"):
            check_metadata(tree, "9.8.7", NATIVE)

    def test_publisher_recomputes_reports_and_verifies_every_attested_file(self):
        directory, reports = self.candidate()
        self.aggregate(directory, reports)
        with contextlib.redirect_stdout(io.StringIO()):
            release.finalize(SimpleNamespace(directory=directory))
        (directory / "provenance.jsonl").write_text("CPU fixture: signature verification is mocked\n")
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        with patch.object(release_publish, "gh", return_value="") as gh, patch.object(release_publish.subprocess, "run"), contextlib.redirect_stdout(io.StringIO()):
            release_publish.verify_bundle(directory, "123", run)
            self.assertEqual(gh.call_count, 9)
            for call in gh.call_args_list:
                self.assertIn("--deny-self-hosted-runners", call.args[0])
                self.assertIn("--source-digest", call.args[0])
        validation = read_json(directory / "validation.json")
        validation["profiles"][0]["models"][0]["context"] = 1
        write_json(directory / "validation.json", validation)
        with self.assertRaisesRegex(ValueError, "context"):
            release_publish.verify_bundle(directory, "123", run)

    def test_gpu_family_assertions_reject_software_and_wrong_devices(self):
        good = "Selected GPU: #0 AMD Radeon AI PRO R9700 (RADV GFX1201)\nGPU tuning: amd_rdna4 | 32768 MB VRAM\nVerdict: READY [OK]\n"
        self.assertIn("R9700", release_gpu.gpu_evidence(good, "rdna4"))
        self.assertEqual(release_gpu.gpu_evidence("GPU family: apple7 [OK]\nVerdict: READY [OK]", "apple7"), "apple7")
        rocm = "  Device: AMD Radeon AI PRO R9700 [gfx1201] [OK]\n  Compute target: gfx1201 [OK]\nVerdict: READY [OK]\n"
        self.assertEqual(release_gpu.gpu_evidence(rocm, "gfx1201"), "AMD Radeon AI PRO R9700")
        for text, family in [(good, "rdna3"), (good.replace("R9700", "llvmpipe"), "rdna4"),
                             (good + "requested #1 out of range", "rdna4"), (good + "[FAIL]", "rdna4"),
                             (rocm.replace("[gfx1201]", "[gfx1100]"), "gfx1201"),
                             (rocm.replace("Compute target: gfx1201", "Compute target: gfx1100"), "gfx1201")]:
            with self.subTest(family=family), self.assertRaises(ValueError):
                release_gpu.gpu_evidence(text, family)

    def test_api_uses_the_loaded_model_instead_of_switching_to_an_unloaded_catalog_entry(self):
        self.assertEqual(release_gpu.active_model_id([{"id": "unloaded", "active": False}, {"id": "loaded", "active": True}]), "loaded")
        self.assertEqual(release_gpu.active_model_id([{"id": "rocm-loaded.gguf"}]), "rocm-loaded.gguf")
        for data in [[], [{"id": "unloaded", "active": False}], [{"id": "one"}, {"id": "two"}]]:
            with self.assertRaises(ValueError):
                release_gpu.active_model_id(data)

    def test_http_reference_gate_rejects_early_stops_and_uses_the_exact_reference_prompt(self):
        samples = [("<rendered reference prompt>", {"content": "The complete reference continuation.\nSecond line."})]
        body = {"model": "loaded", "prompt": "unrelated", "temperature": 0}
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "The complete reference"}]}) as request:
            with self.assertRaisesRegex(ValueError, "full HTTP reference"):
                release_gpu.reference_http_parity(1, body, samples)
            self.assertEqual(request.call_args.args[2]["prompt"], samples[0][0])
            self.assertEqual(request.call_args.args[2]["model"], "loaded")
            self.assertFalse(request.call_args.args[2]["stream"])
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": samples[0][1]["content"]}]}):
            release_gpu.reference_http_parity(1, body, samples)
        samples[0][1].update(validated_content="A qualified alternate.", comparison={"near_ties": [{"rank": 2}]})
        for content in [samples[0][1]["content"], "A qualified alternate."]:
            with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": content}]}):
                release_gpu.reference_http_parity(1, body, samples)
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "A qualified"}]}), self.assertRaises(ValueError):
            release_gpu.reference_http_parity(1, body, samples)
        samples[0][1].pop("comparison")
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "A qualified alternate."}]}), self.assertRaises(ValueError):
            release_gpu.reference_http_parity(1, body, samples)

    def test_cli_reference_gate_checks_all_backend_token_formats_and_complete_text(self):
        expected = {"content": "First line.\n  Second line. ",
                    "completion_probabilities": [{"id": token} for token in [791, 315, 198]]}
        for label in ["Generated tokens (3): { 791, 315, 198 }",
                      "Output tokens (3): { 791, 315, 198 }",
                      "Output tokens (3): first20={ 791, 315, 198 }"]:
            for output in ["Output (3 tokens): ", "Output text: "]:
                text = "debug(zinc): " + label + "\ninfo(zinc): " + output + expected["content"] + "\n"
                with self.subTest(label=label, output=output):
                    release_gpu.cli_reference_parity(text, expected)
                for bad in [text.replace("315, 198", "315, 199"),
                            text.replace("(3):", "(4):"), text.replace("791, 315, 198", "791"),
                            text.replace("  Second line. ", "Second line."),
                            text.replace("\n  Second line. ", ""), text.replace(expected["content"], "First")]:
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        release_gpu.cli_reference_parity(bad, expected)
        with self.assertRaises(ValueError):
            release_gpu.cli_reference_parity("topk: top=791\nOutput text: First line.\n", expected)

    def test_reference_precision_is_fixed_after_worker_flags_and_cli_keeps_gpu_argmax(self):
        commands, environments = [], []
        @contextlib.contextmanager
        def fake_process(command, log, _cwd, env):
            commands.append(command)
            environments.append(env)
            if Path(command[0]).name == "zinc":
                Path(log).write_text("debug(zinc): Generated tokens (1): { 760 }\ninfo(zinc): Output (1 tokens): The\n")
            yield SimpleNamespace(wait=lambda **_kwargs: 0)
        expected = {"content": "The", "completion_probabilities": [{"id": 760}]}
        worker_flags = ["--ubatch-size", "512", "--cache-type-k", "f16", "--flash-attn", "on"]
        with patch.object(release_gpu, "PROMPTS", ["fixture"]), patch.object(release_gpu, "process", fake_process), \
             patch.object(release_gpu, "ready"), patch.object(release_gpu, "json_request", side_effect=[{"prompt": "rendered"}, expected]):
            samples = release_gpu.inference_parity(self.root / "package", self.root / "reference", self.root / "model",
                                                  4096, 0, worker_flags, self.root, {})
        self.assertEqual(samples, [("rendered", expected)])
        for flag, value in [("--ubatch-size", "1"), ("--batch-size", "512"), ("--parallel", "1"),
                            ("--cache-type-k", "f32"), ("--cache-type-v", "f32"), ("--flash-attn", "off")]:
            self.assertEqual([commands[0][i + 1] for i, item in enumerate(commands[0][:-1]) if item == flag][-1], value)
        self.assertIn("--debug", commands[1])
        self.assertNotIn("ZINC_LOG_TOPK", environments[1])
        self.assertNotIn("ZINC_FORCE_CPU_ARGMAX", environments[1])

    def test_reference_agreement_records_a_bounded_tie_without_counting_it_exact(self):
        cases = [{"tokens": 16, "near_ties": []} for _ in range(3)]
        tie = {"position": 11, "chosen_id": 36337, "reference_id": 3618, "rank": 2, "logprob_gap": 0.032}
        cases[1]["near_ties"].append(tie)
        report = release_lib.reference_agreement(cases)
        self.assertEqual((report["exact_tokens"], report["total_tokens"], report["near_tie_tokens"]), (47, 48, 1))
        release_lib.verify_reference_agreement(report)
        for change in [lambda data: data[1]["near_ties"][0].update(rank=3),
                       lambda data: data[1]["near_ties"][0].update(logprob_gap=0.051),
                       lambda data: data[1]["near_ties"][0].update(logprob_gap=float("nan")),
                       lambda data: data[1]["near_ties"][0].update(position=16),
                       lambda data: data[1]["near_ties"][0].update(chosen_id=3618),
                       lambda data: data[0]["near_ties"].append(dict(tie)),
                       lambda data: data[1]["near_ties"].append(dict(tie)),
                       lambda data: data.pop()]:
            data = json.loads(json.dumps(cases));change(data)
            with self.assertRaises(ValueError):
                release_lib.reference_agreement(data)
        report["exact_tokens"] = 48
        with self.assertRaisesRegex(ValueError, "totals"):
            release_lib.verify_reference_agreement(report)

    def test_reference_near_tie_rejects_unrelated_choices_and_large_probability_gaps(self):
        expected = {"completion_probabilities": [{"id": token} for token in range(100, 116)]}
        expected["completion_probabilities"][11]["top_logprobs"] = [{"id": 111, "logprob": -1.0}, {"id": 999, "logprob": -1.032}]
        actual = list(range(100, 116));actual[11] = 999
        tie = release_gpu.reference_near_tie(actual, expected)
        self.assertEqual((tie["position"], tie["rank"]), (11, 2))
        self.assertAlmostEqual(tie["logprob_gap"], 0.032)
        for selected, low in [(998, -1.032), (999, -1.051), (999, float("nan")), (999, -0.9)]:
            data = json.loads(json.dumps(expected));data["completion_probabilities"][11]["top_logprobs"][1]["logprob"] = low
            tokens = actual.copy();tokens[11] = selected
            with self.assertRaises(ValueError):
                release_gpu.reference_near_tie(tokens, data)
        with self.assertRaises(ValueError):
            release_gpu.reference_near_tie(actual[:-1], expected)

    def test_near_tie_branch_requires_matching_prompt_all_suffix_tokens_and_complete_detokenization(self):
        actual = list(range(100, 116));actual[11] = 999
        tie = {"position": 11, "chosen_id": 999, "reference_id": 111, "rank": 2, "logprob_gap": 0.032}
        content = "A complete alternate continuation.\n Second line. "
        log = "info(zinc): Output text: " + content + "\n"
        expected = {"content": "Original reference remains visible"}
        replies = [{"tokens": [1, 2]}, {"completion_probabilities": [{"id": token} for token in actual[12:]]}, {"content": content}]
        with patch.object(release_gpu, "output", return_value="[1, 2]\n"), \
             patch.object(release_gpu, "json_request", side_effect=replies) as request:
            release_gpu.qualify_reference_branch(1, self.root, self.root / "model", "rendered", expected, actual, log, tie, self.root, {})
        self.assertEqual(request.call_args_list[1].args[2]["prompt"], [1, 2] + actual[:12])
        self.assertEqual(request.call_args_list[1].args[2]["n_predict"], 4)
        self.assertEqual(expected["content"], "Original reference remains visible")
        self.assertEqual(expected["validated_content"], content)
        for native, wrong in [("[9]\n", replies), ("[1, 2]\n", [replies[0], {"completion_probabilities": [{"id": 7}]}]),
                              ("[1, 2]\n", [*replies[:2], {"content": content.rstrip()}])]:
            data = {"content": "original"}
            with patch.object(release_gpu, "output", return_value=native), patch.object(release_gpu, "json_request", side_effect=wrong), self.assertRaises(ValueError):
                release_gpu.qualify_reference_branch(1, self.root, self.root / "model", "rendered", data, actual, log, tie, self.root, {})
            self.assertNotIn("validated_content", data)

    def test_backend_device_indices_can_differ_but_the_physical_reservation_cannot(self):
        base = {"device": 1, "lock_directory": "<shared-reservation>", "models": {"one": "<model>"},
                "profiles": {"rocm-rdna4": {"device": 0, "reference_bin": "<HIP-reference>"}}}
        self.assertEqual(release_gpu.worker_for_profile(base, "vulkan-rdna4")["device"], 1)
        rocm = release_gpu.worker_for_profile(base, "rocm-rdna4")
        self.assertEqual(rocm["device"], 0)
        self.assertEqual(rocm["lock_directory"], base["lock_directory"])
        self.assertEqual(rocm["models"], base["models"])
        base["profiles"]["rocm-rdna4"]["lock_directory"] = "<different-reservation>"
        with self.assertRaisesRegex(ValueError, "shared reservation"):
            release_gpu.worker_for_profile(base, "rocm-rdna4")

    def test_gpu_process_is_reaped_on_test_failure(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with release_gpu.process([sys.executable, "-c", "import time; time.sleep(120)"], self.root / "child.log", self.root, os.environ) as child:
                raise RuntimeError("fixture failure")
        self.assertIsNotNone(child.poll())

    def test_signing_timeouts_do_not_expose_secret_arguments(self):
        command = ["security", "-p", "synthetic-test-password"]
        with patch.object(release_sign.subprocess, "run", side_effect=subprocess.TimeoutExpired(command, 1)):
            with self.assertRaisesRegex(ValueError, "timed out: security") as caught:
                release_sign.run(command, timeout=1)
        self.assertNotIn("synthetic-test-password", str(caught.exception))

    def test_gpu_process_is_reaped_when_controller_receives_sigterm(self):
        script = ("import sys,time,os; sys.path.insert(0,sys.argv[1]); import release_gpu; "
                  "release_gpu.install_signal_handlers()\n"
                  "with release_gpu.process([sys.executable,'-c','import time;time.sleep(120)'], sys.argv[2], '.', os.environ) as child:\n"
                  " print(child.pid,flush=True)\n time.sleep(120)\n")
        controller = subprocess.Popen([sys.executable, "-c", script, str(ROOT / "tools"), str(self.root / "signal-child.log")],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            child_pid = int(controller.stdout.readline())
            controller.terminate()
            controller.communicate(timeout=10)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
