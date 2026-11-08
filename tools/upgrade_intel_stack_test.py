"""CPU fixtures; no package manager, network, root operation, or GPU is used."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

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
            return {"results": [{"pkgname": name, "repo": "core", "arch": "x86_64", "pkgver": "1.0", "pkgrel": "1"},
                                {"pkgname": name, "repo": "core-testing", "arch": "x86_64", "pkgver": "9.0", "pkgrel": "1"}]}
        with patch.object(upgrade, "json_url", side_effect=response):
            result = upgrade.latest()
            self.assertEqual(result["kernel_stable"], "7.3.3")
            self.assertEqual(set(result["arch"]), set(upgrade.PACKAGES))
            self.assertTrue(all(v["version"] == "1.0-1" for v in result["arch"].values()))

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


if __name__ == "__main__":
    unittest.main()
