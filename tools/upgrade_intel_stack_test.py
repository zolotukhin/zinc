"""CPU fixtures; no package manager, network, root operation, or GPU is used."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

import intel_native_probe as native

spec = importlib.util.spec_from_file_location("upgrade_intel_stack", Path(__file__).parents[1] / "scripts/upgrade_intel_stack.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


class UpgradeTest(unittest.TestCase):
    def targets(self):
        return {"kernel_stable": "7.3.3", "oneapi_current": "2026.2.0",
                "arch": {name: {"version": "7.3.3.arch1-1" if name == "linux" else
                                   "2026.2.0-1" if name == "intel-oneapi-toolkit" else "1.0.0-1"}
                         for name in upgrade.PACKAGES}}

    def test_default_plan_never_applies(self):
        with patch.object(upgrade, "check_host"), patch.object(upgrade, "latest", return_value=self.targets()), \
                patch.object(upgrade, "installed", return_value={}), patch.object(upgrade, "apply") as apply, \
                patch("sys.stdout", new_callable=io.StringIO), patch.object(upgrade.os, "umask"):
            upgrade.main([])
            apply.assert_not_called()

    def test_apply_is_explicit_and_distro_lag_is_never_silently_accepted(self):
        with patch.object(upgrade.os, "geteuid", return_value=1000), patch.object(upgrade, "idle") as idle:
            with self.assertRaisesRegex(ValueError, "administrator"):
                upgrade.apply(self.targets(), False)
            idle.assert_not_called()
        targets = self.targets(); targets["arch"]["linux"]["version"] = "7.2.14.arch1-1"
        with patch.object(upgrade.os, "geteuid", return_value=0), patch.object(upgrade, "idle") as idle:
            with self.assertRaisesRegex(ValueError, "lags upstream"):
                upgrade.apply(targets, False)
            idle.assert_not_called()

    def test_version_and_both_upstream_lags_are_detected(self):
        self.assertEqual(upgrade.version("1:2.42.6-1"), (2, 42, 6))
        self.assertEqual(upgrade.lagging(self.targets()), {})
        targets = self.targets(); targets["arch"]["linux"]["version"] = "7.2.14.arch1-1"
        targets["arch"]["intel-oneapi-toolkit"]["version"] = "2026.0.1.27+build-2"
        self.assertEqual(set(upgrade.lagging(targets)), {"kernel", "oneapi"})
        for value in (None, "latest", "", "$(uname -r)"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                upgrade.version(value)

    def test_fallback_rewrites_only_exact_boot_paths_and_preserves_arguments(self):
        original = "menuentry 'current' {\n linux /vmlinuz-linux root=UUID=example quiet\n initrd /intel-ucode.img /initramfs-linux.img\n}\nmenuentry 'LTS' {\n linux /vmlinuz-linux-lts root=UUID=example\n initrd /initramfs-linux-lts.img\n}\n"
        result = upgrade.rewrite_grub(original, {"/vmlinuz-linux": "/saved/vmlinuz-linux",
                                                "/initramfs-linux.img": "/saved/initramfs-linux.img"})
        self.assertIn("linux /saved/vmlinuz-linux root=UUID=example quiet", result)
        self.assertIn("initrd /intel-ucode.img /saved/initramfs-linux.img", result)
        self.assertIn("linux /vmlinuz-linux-lts root=UUID=example", result)
        self.assertIn("initrd /initramfs-linux-lts.img", result)
        self.assertNotIn("/saved/vmlinuz-linux-lts", result)
        with self.assertRaisesRegex(ValueError, "every saved boot image"):
            upgrade.rewrite_grub(original, {"/absent.img": "/saved/absent.img"})

    def test_official_package_resolution_handles_lowercase_repository_names(self):
        def response(url):
            if "kernel.org" in url:
                return {"releases": [{"moniker": "mainline", "version": "7.4-rc1"},
                                     {"moniker": "stable", "version": "7.3.3"}]}
            name = url.rsplit("=", 1)[1]
            return {"results": [{"pkgname": name, "repo": "core", "arch": "x86_64", "pkgver": "1.0", "pkgrel": "1",
                                 "epoch": 1 if name == "intel-graphics-compiler" else 0},
                                {"pkgname": name, "repo": "core-testing", "arch": "x86_64", "pkgver": "9.0", "pkgrel": "1"}]}
        with patch.object(upgrade, "json_url", side_effect=response):
            result = upgrade.latest()
            self.assertEqual(result["kernel_stable"], "7.3.3")
            self.assertEqual(set(result["arch"]), set(upgrade.PACKAGES))
            self.assertEqual(result["arch"]["intel-graphics-compiler"]["version"], "1:1.0-1")
            self.assertTrue(all(v["version"] == "1.0-1" for k, v in result["arch"].items() if k != "intel-graphics-compiler"))

    def test_ambiguous_official_packages_fail_closed(self):
        def response(url):
            if "kernel.org" in url:
                return {"releases": [{"moniker": "stable", "version": "7.3.3"}]}
            row = {"pkgname": "linux", "repo": "core", "arch": "x86_64", "pkgver": "7.3.3", "pkgrel": "1"}
            return {"results": [row, row]}
        with patch.object(upgrade, "json_url", side_effect=response), self.assertRaisesRegex(ValueError, "ambiguous"):
            upgrade.latest()

    def test_transaction_verification_uses_arch_comparator_for_firmware_and_epochs(self):
        targets = self.targets()
        current = {name: row["version"] for name, row in targets["arch"].items()}
        current["linux-firmware-intel"] = "20261019-1"
        current["intel-graphics-compiler"] = "1:2.42.6-1"
        with patch.object(upgrade, "command", return_value=Mock(stdout="0\n")) as compare:
            self.assertTrue(upgrade.packages_ready(current, targets))
            self.assertEqual(compare.call_count, len(upgrade.PACKAGES))
            self.assertTrue(all(call.args[0][0] == "vercmp" for call in compare.call_args_list))
            self.assertIn(["vercmp", "20261019-1", targets["arch"]["linux-firmware-intel"]["version"]],
                          [call.args[0] for call in compare.call_args_list])
        with patch.object(upgrade, "command", return_value=Mock(stdout="-1\n")):
            self.assertFalse(upgrade.packages_ready(current, targets))
        with patch.object(upgrade, "command", return_value=Mock(stdout="invalid")), self.assertRaises(ValueError):
            upgrade.packages_ready(current, targets)

    def test_installer_and_probe_keep_the_standalone_busy_policy_in_sync(self):
        self.assertEqual(upgrade.BUSY_PROCESSES, native.BUSY_PROCESSES)
        self.assertIn("icpx", upgrade.BUSY_PROCESSES)
        self.assertIn("cmake", upgrade.BUSY_PROCESSES)

    def test_failed_command_keeps_signature_or_bootloader_diagnostics(self):
        result = Mock(returncode=2, stderr="signature verification failed\n", stdout="")
        with patch.object(upgrade.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "exit 2.*") as raised:
                upgrade.command(["pacman-key", "--verify", "fixture.sig", "fixture"])
            self.assertIn("signature verification failed", str(raised.exception))
            self.assertIs(upgrade.command(["pacman-key"], check=False), result)

    def test_cached_rollback_archive_still_requires_signature_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root / "cache"; cache.mkdir(); dest = root / "saved"; dest.mkdir()
            name = "linux-7.2.7.arch1-1-x86_64.pkg.tar.zst"
            (cache / name).write_bytes(b"package"); (cache / (name + ".sig")).write_bytes(b"signature")
            with patch.object(upgrade, "PACKAGE_CACHE", cache), patch.object(upgrade, "download") as download, \
                    patch.object(upgrade, "command") as verify:
                self.assertEqual(upgrade.archive_package("linux", "1:7.2.7.arch1-1", dest), name)
                download.assert_not_called()
                verify.assert_called_once_with(["pacman-key", "--verify", str(dest / name) + ".sig", str(dest / name)])
            self.assertEqual((dest / name).read_bytes(), b"package")

    def test_downloaded_archive_404_fallback_checks_the_final_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dest = root / "saved"; dest.mkdir()
            def download(url, target):
                if "x86_64" in target.name:
                    raise urllib.error.HTTPError(url, 404, "missing", {}, None)
                target.write_bytes(b"signature" if target.name.endswith(".sig") else b"package")
            with patch.object(upgrade, "PACKAGE_CACHE", root / "missing"), \
                    patch.object(upgrade, "download", side_effect=download) as downloaded, patch.object(upgrade, "command") as verify:
                name = upgrade.archive_package("linux-firmware-intel", "20261019-1", dest)
                self.assertEqual(name, "linux-firmware-intel-20261019-1-any.pkg.tar.zst")
                self.assertEqual(downloaded.call_count, 4)
                verify.assert_called_once_with(["pacman-key", "--verify", str(dest / name) + ".sig", str(dest / name)])

    def test_missing_signature_does_not_leave_the_unverified_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dest = root / "saved"; dest.mkdir()
            def download(url, target):
                if target.name.endswith(".zst.sig"):
                    raise urllib.error.HTTPError(url, 404, "missing", {}, None)
                target.write_bytes(b"downloaded")
            with patch.object(upgrade, "PACKAGE_CACHE", root / "missing"), \
                    patch.object(upgrade, "download", side_effect=download), patch.object(upgrade, "command"):
                name = upgrade.archive_package("linux", "7.2.7.arch1-1", dest)
            self.assertTrue(name.endswith(".xz"))
            self.assertFalse(any(p.name.endswith(".zst") for p in dest.iterdir()))

    def test_all_404s_and_bad_signatures_stop_before_an_archive_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dest = root / "saved"; dest.mkdir()
            error = urllib.error.HTTPError("https://archive.archlinux.org/", 404, "missing", {}, None)
            with patch.object(upgrade, "PACKAGE_CACHE", root / "missing"), \
                    patch.object(upgrade, "download", side_effect=error) as download, patch.object(upgrade, "command") as verify:
                with self.assertRaisesRegex(ValueError, "cannot preserve"):
                    upgrade.archive_package("linux", "7.2.7.arch1-1", dest)
                self.assertEqual(download.call_count, 4); verify.assert_not_called()
            def downloaded(url, target): target.write_bytes(b"untrusted")
            with patch.object(upgrade, "PACKAGE_CACHE", root / "missing"), \
                    patch.object(upgrade, "download", side_effect=downloaded), \
                    patch.object(upgrade, "command", side_effect=ValueError("bad signature")):
                with self.assertRaisesRegex(ValueError, "bad signature"):
                    upgrade.archive_package("linux", "7.2.7.arch1-1", dest)
            self.assertEqual(list(dest.iterdir()), [])

    def test_archive_validation_and_transport_errors_never_fall_through(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for name, value in (("../linux", "7.2.7-1"), ("..", "7.2.7-1"), ("linux", "../7.2.7"), ("linux", "::::")):
                with self.subTest(name=name, value=value), patch.object(upgrade, "download") as download:
                    with self.assertRaisesRegex(ValueError, "invalid package"):
                        upgrade.archive_package(name, value, dest)
                    download.assert_not_called()
            with patch.object(upgrade, "PACKAGE_CACHE", dest / "missing"), \
                    patch.object(upgrade, "download", side_effect=urllib.error.URLError("transport unavailable")) as download:
                with self.assertRaises(urllib.error.URLError):
                    upgrade.archive_package("linux", "7.2.7-1", dest)
                self.assertEqual(download.call_count, 1)

    def test_download_is_atomic_and_removes_partial_files_on_read_failure(self):
        class Broken(io.BytesIO):
            def read(self, count=-1):
                if self.tell(): raise OSError("interrupted download")
                return super().read(3)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"; target.write_bytes(b"original")
            with patch.object(upgrade.urllib.request, "urlopen", return_value=Broken(b"partial")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    upgrade.download("https://archive.archlinux.org/package", target)
            self.assertEqual(target.read_bytes(), b"original")
            self.assertFalse(target.with_name("package.partial").exists())
            with patch.object(upgrade.urllib.request, "urlopen", return_value=io.BytesIO(b"complete")):
                upgrade.download("https://archive.archlinux.org/package", target)
            self.assertEqual(target.read_bytes(), b"complete")

    def test_transaction_keeps_primary_failure_and_reports_both_recovery_errors(self):
        with patch.object(upgrade.subprocess, "run", return_value=Mock(returncode=41)), \
                patch.object(upgrade, "restore_old_modules", side_effect=OSError("copy failed")) as restore, \
                patch.object(upgrade, "regenerate_grub", side_effect=ValueError("GRUB failed")) as grub, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaisesRegex(ValueError, "pacman update failed with exit code 41"):
                upgrade.run_transaction(Path("saved"), Path("modules"), Path("boot"))
            restore.assert_called_once_with(Path("saved"), Path("modules"), Path("boot"), True)
            grub.assert_called_once(); self.assertIn("copy failed", stderr.getvalue()); self.assertIn("GRUB failed", stderr.getvalue())

    def test_transaction_cancellation_propagates_and_success_cannot_hide_recovery_failure(self):
        interruption = KeyboardInterrupt()
        with patch.object(upgrade.subprocess, "run", side_effect=interruption), \
                patch.object(upgrade, "finish_transaction", return_value=[]) as finish:
            with self.assertRaises(KeyboardInterrupt) as raised:
                upgrade.run_transaction(Path("saved"), Path("modules"), Path("boot"))
            self.assertIs(raised.exception, interruption)
            self.assertIs(finish.call_args.args[-1], True)
        with patch.object(upgrade.subprocess, "run", return_value=Mock(returncode=0)), \
                patch.object(upgrade, "finish_transaction", return_value=[("regenerate GRUB", OSError("write failed"))]):
            with self.assertRaisesRegex(ValueError, "update finished but recovery failed.*write failed"):
                upgrade.run_transaction(Path("saved"), Path("modules"), Path("boot"))

    def test_modules_are_restored_even_when_unowned_files_keep_the_old_directory_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); modules = root / "old"; modules.mkdir()
            (modules / "dkms-unowned").write_bytes(b"keep")
            with patch.object(upgrade, "command", return_value=Mock(returncode=0, stdout="linux /new/kernel/module.ko\n")) as command:
                upgrade.restore_old_modules(root / "saved", modules, root / "boot", False)
                self.assertEqual(command.call_args.args[0][0], "cp")
            self.assertTrue((modules / "dkms-unowned").is_file())

    def test_same_release_success_keeps_installed_modules_and_changed_kernel_stops_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); saved = root / "saved"; saved.mkdir(); modules = root / "old"
            (saved / "vmlinuz").write_bytes(b"original"); boot = root / "boot"; boot.write_bytes(b"original")
            listing = Mock(returncode=0, stdout="linux " + str(modules) + "/kernel/module.ko\n")
            with patch.object(upgrade, "command", return_value=listing) as command:
                upgrade.restore_old_modules(saved, modules, boot, False)
                command.assert_called_once_with(["pacman", "-Ql", "linux"], check=False)
            with patch.object(upgrade, "command", return_value=listing) as command:
                upgrade.restore_old_modules(saved, modules, boot, True)
                self.assertEqual(command.call_args.args[0][0], "cp")
            boot.write_bytes(b"new kernel, same release")
            with patch.object(upgrade, "command", return_value=listing) as command:
                with self.assertRaisesRegex(ValueError, "different image"):
                    upgrade.restore_old_modules(saved, modules, boot, False)
                self.assertEqual(command.call_count, 1)

    def test_unknown_package_ownership_cannot_clobber_modules(self):
        with patch.object(upgrade, "command", return_value=Mock(returncode=2, stderr="database unreadable")) as command:
            with self.assertRaisesRegex(ValueError, "cannot determine"):
                upgrade.restore_old_modules(Path("saved"), Path("modules"), Path("boot"), True)
            self.assertEqual(command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
