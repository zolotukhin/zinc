#!/usr/bin/env python3
"""Administrator tool for an Arch Intel Xe worker. Default: inspect, never update.

Run on the remote worker, not on the developer workstation. --apply performs a
normal full Arch transaction, preserves a separate GRUB kernel fallback, and
does not reboot. No Ubuntu repositories or unsigned kernel builds are added.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

PACKAGES = ("linux", "linux-headers", "linux-firmware-intel", "intel-compute-runtime",
            "intel-graphics-compiler", "intel-gmmlib", "level-zero-loader", "intel-oneapi-toolkit")
BACKUP_PACKAGES = PACKAGES + ("linux-firmware", "mesa", "vulkan-intel")
ONEAPI_CURRENT = "2026.2.0"  # Intel's 2026-10-22 release; checked 2026-11-07.
ENVIRONMENT = {**os.environ, "LC_ALL": "C"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(args, check=True):
    result = subprocess.run(args, capture_output=True, text=True, env=ENVIRONMENT, timeout=120)
    require(not check or result.returncode == 0, "command failed: " + shlex.join(args))
    return result


def json_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ZINC-Intel-stack-upgrade"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def version(value):
    require(isinstance(value, str), "version is missing")
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value.split(":")[-1])
    require(match is not None, "unsupported version: " + value)
    return tuple(int(x or 0) for x in match.groups())


def latest():
    releases = json_url("https://www.kernel.org/releases.json")["releases"]
    stable = next(r["version"] for r in releases if r.get("moniker") == "stable" and not r.get("iseol"))
    packages = {}
    for name in PACKAGES:
        # Firmware packages moved between repositories; search the official API
        # rather than assuming an Ubuntu package name or a stale mirror version.
        data = json_url("https://archlinux.org/packages/search/json/?name=" + name)
        rows = [r for r in data["results"] if r["pkgname"] == name and r["repo"].lower() in ("core", "extra")
                and r["arch"] in ("x86_64", "any")]
        require(len(rows) == 1, "official Arch package is missing or ambiguous: " + name)
        packages[name] = {"version": rows[0]["pkgver"] + "-" + rows[0]["pkgrel"], "repo": rows[0]["repo"]}
    return {"kernel_stable": stable, "oneapi_current": ONEAPI_CURRENT, "arch": packages}


def lagging(targets):
    return {name: {"arch": targets["arch"][package]["version"], "upstream": upstream}
            for name, package, upstream in (("kernel", "linux", targets["kernel_stable"]),
                                             ("oneapi", "intel-oneapi-toolkit", targets["oneapi_current"]))
            if version(targets["arch"][package]["version"]) < version(upstream)}


def installed():
    return dict(line.split(maxsplit=1) for line in command(["pacman", "-Q"]).stdout.splitlines())


def packages_ready(current, targets):
    for name in PACKAGES:
        if name not in current:
            return False
        # Firmware uses dates; Arch also has epochs and package-release suffixes.
        # The package manager's comparator, not kernel-version arithmetic, owns these.
        comparison = command(["vercmp", current[name], targets["arch"][name]["version"]]).stdout.strip()
        require(re.fullmatch(r"-?\d+", comparison), "invalid package version comparison")
        if int(comparison) < 0:
            return False
    return True


def check_host():
    require(sys.platform.startswith("linux") and platform.machine() == "x86_64", "requires the remote x86_64 Linux worker")
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    require(release.get("ID", "").strip('"') == "arch", "this script supports Arch Linux only")
    require(Path("/boot/grub/grub.cfg").is_file() and Path("/etc/grub.d").is_dir(), "requires the worker's existing GRUB installation")
    xe = [p for p in Path("/sys/class/drm").glob("renderD*/device")
          if (p / "vendor").read_text().strip() == "0x8086" and (p / "driver").resolve().name == "xe"]
    require(bool(xe), "an Intel GPU using the Xe kernel driver was not found")
    require(re.fullmatch(r"[A-Za-z0-9_.+-]+", platform.release()), "unsafe kernel release")


def idle():
    require(not Path("/var/lib/pacman/db.lck").exists(), "another package transaction is active")
    for name in ("zinc", "llama-server", "llama-cli", "llama-bench", "zinc-sycl-probe", "icpx", "cmake"):
        require(command(["pgrep", "-x", name], check=False).returncode == 1, "worker is busy: " + name)
    nodes = list(Path("/dev/dri").glob("renderD*"))
    require(bool(nodes), "no render node")
    # A quiet inference process list does not prove that another GPU user is idle.
    require(shutil.which("fuser") is not None, "install psmisc for GPU-user inspection")
    require(command(["fuser", *map(str, nodes)], check=False).returncode == 1,
            "a render node is open; stop and inspect its owner before updating")


def rewrite_grub(snapshot, replacements):
    counts = {source: 0 for source in replacements}
    output = []
    for line in snapshot.splitlines(keepends=True):
        if re.match(r"^\s*(?:linux|linuxefi|initrd|initrdefi)\s", line):
            for source, target in replacements.items():
                # Do not rewrite /vmlinuz-linux-lts or similarly prefixed paths.
                pattern = r"(?<![\w/.-])" + re.escape(source) + r"(?![\w/.-])"
                line, changed = re.subn(pattern, lambda _: target, line)
                counts[source] += changed
        output.append(line)
    require(all(counts.values()), "the existing GRUB config does not reference every saved boot image")
    return "".join(output)


def download(url, target):
    temporary = target.with_name(target.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def archive_package(name, package_version, destination):
    require(re.fullmatch(r"[a-z0-9+_.-]+", name), "invalid package name")
    require(re.fullmatch(r"[A-Za-z0-9:+_.-]+", package_version), "invalid package version")
    cache = Path("/var/cache/pacman/pkg")
    archive_version = package_version.split(":")[-1]
    for arch in ("x86_64", "any"):
        for suffix in ("zst", "xz"):
            filename = f"{name}-{archive_version}-{arch}.pkg.tar.{suffix}"
            target = destination / filename
            if (cache / filename).is_file() and (cache / (filename + ".sig")).is_file():
                shutil.copy2(cache / filename, target)
                shutil.copy2(cache / (filename + ".sig"), target.with_name(filename + ".sig"))
            else:
                base = "https://archive.archlinux.org/packages/" + name[0] + "/" + name + "/"
                try:
                    download(base + urllib.parse.quote(filename), target)
                    download(base + urllib.parse.quote(filename + ".sig"), target.with_name(filename + ".sig"))
                except urllib.error.HTTPError as error:
                    if error.code == 404:
                        target.unlink(missing_ok=True)
                        continue
                    raise
            # Read the system trust store; never import an arbitrary signing key.
            command(["pacman-key", "--verify", str(target) + ".sig", str(target)])
            return filename
    raise ValueError("cannot preserve a signed rollback archive for " + name + " " + package_version)


def regenerate_grub():
    candidate = Path("/boot/grub/.zinc-upgrade-grub.cfg")
    try:
        command(["grub-mkconfig", "-o", str(candidate)])
        command(["grub-script-check", str(candidate)])
        os.chmod(candidate, 0o600)
        candidate.replace("/boot/grub/grub.cfg")
    finally:
        candidate.unlink(missing_ok=True)


def apply(targets, allow_lag):
    require(os.geteuid() == 0, "--apply requires an administrator: sudo python3 <script> --apply")
    require(allow_lag or not lagging(targets),
            "Arch still lags upstream. Wait for matching packages, or explicitly choose --allow-distro-lag; this does not install upstream's latest kernel/toolkit")
    idle()
    release = platform.release()
    modules = Path("/usr/lib/modules") / release
    require(modules.is_dir() and (modules / "vmlinuz").is_file(), "running kernel files are unavailable")
    boot = Path("/boot")
    require(hashlib.sha256((modules / "vmlinuz").read_bytes()).digest() == hashlib.sha256((boot / "vmlinuz-linux").read_bytes()).digest(),
            "installed boot kernel differs from the running kernel; reboot and inspect before upgrading")
    require(shutil.disk_usage(boot).free >= 512 * 1024**2, "at least 512 MiB free is required on /boot for the saved kernel")
    require(shutil.disk_usage("/var/lib").free >= 12 * 1024**3, "at least 12 GiB free is required for rollback inputs")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path("/var/lib/zinc/intel-stack-backups") / stamp
    backup.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = installed()
    (backup / "packages-before.json").write_text(json.dumps(state, indent=2) + "\n")
    (backup / "targets.json").write_text(json.dumps(targets, indent=2) + "\n")
    archives = backup / "packages"; archives.mkdir(mode=0o700)
    for name in BACKUP_PACKAGES:
        if name in state:
            archive_package(name, state[name], archives)
    saved_modules = backup / "modules" / release
    shutil.copytree(modules, saved_modules, symlinks=True)
    shutil.copy2(boot / "grub/grub.cfg", backup / "grub-before.cfg")
    shutil.copytree("/var/lib/pacman/local", backup / "pacman-local", symlinks=True)
    saved_boot = boot / ("zinc-intel-fallback-" + stamp)
    saved_boot.mkdir(mode=0o700)
    replacements = {}
    for name in ("vmlinuz-linux", "initramfs-linux.img"):
        require((boot / name).is_file(), "missing boot image: " + name)
        shutil.copy2(boot / name, saved_boot / name)
        old = command(["grub-mkrelpath", str(boot / name)]).stdout.strip()
        new = command(["grub-mkrelpath", str(saved_boot / name)]).stdout.strip()
        replacements[old] = new
    for name in ("initramfs-linux-fallback.img", "intel-ucode.img", "amd-ucode.img"):
        if (boot / name).is_file():
            old = command(["grub-mkrelpath", str(boot / name)]).stdout.strip()
            if old in (backup / "grub-before.cfg").read_text():
                shutil.copy2(boot / name, saved_boot / name)
                replacements[old] = command(["grub-mkrelpath", str(saved_boot / name)]).stdout.strip()
    snapshot_name = "zinc-intel-saved-" + stamp + ".cfg"
    (boot / "grub" / snapshot_name).write_text(rewrite_grub((backup / "grub-before.cfg").read_text(), replacements))
    generator = Path("/etc/grub.d") / ("42_zinc_intel_fallback_" + stamp)
    entries = [f"menuentry 'ZINC saved working kernel {release} ({stamp})' {{", f"  configfile $prefix/{snapshot_name}", "}"]
    generator.write_text("#!/bin/sh\nset -eu\nprintf '%s\\n' " + " ".join(shlex.quote(s) for s in entries) + "\n")
    generator.chmod(0o755)
    command(["grub-script-check", str(boot / "grub" / snapshot_name)])
    regenerate_grub()
    print("Saved boot fallback and signed package archives: " + str(backup), flush=True)
    print("Review pacman's full transaction. This script never reboots.", flush=True)
    try:
        # Arch partial upgrades are unsupported: refresh and upgrade together.
        result = subprocess.run(["pacman", "-Syu", "--needed", *PACKAGES], env=ENVIRONMENT)
    finally:
        # Package removal may delete the old module tree, but the saved kernel
        # must still be able to load matching modules after a rollback boot.
        # Unowned/DKMS files can keep an otherwise emptied directory alive.
        # Restore its contents too, rather than testing directory existence.
        modules.mkdir(exist_ok=True)
        command(["cp", "-a", "--remove-destination", str(saved_modules) + "/.", str(modules) + "/"])
        regenerate_grub()
    require(result.returncode == 0, "package update failed; inspect pacman output and the saved fallback")
    after = installed()
    (backup / "packages-after.json").write_text(json.dumps(after, indent=2) + "\n")
    require(packages_ready(after, targets),
            "the selected mirror did not provide the planned packages; inspect the transaction before rebooting")
    print("Update complete. Reboot deliberately with: systemctl reboot")
    print("After reboot, run this script again without --apply, then repeat ZINC/native GPU qualification.")
    print("The saved kernel is in GRUB's ZINC fallback menu. Userspace rollback needs review of the full pacman transaction; do not blindly downgrade a subset of system libraries.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="administrator-only full Arch update; never reboots")
    parser.add_argument("--allow-distro-lag", action="store_true", help="explicitly accept newest Arch packages even if upstream kernel/oneAPI is newer")
    args = parser.parse_args(argv)
    os.umask(0o077)
    check_host()
    targets = latest()
    current = installed()
    report = {"mode": "apply" if args.apply else "plan", "running_kernel": platform.release(),
              "installed": {name: current.get(name) for name in BACKUP_PACKAGES}, "targets": targets,
              "distro_lags_upstream": lagging(targets), "automatic_reboot": False,
              "command": ["pacman", "-Syu", "--needed", *PACKAGES]}
    print(json.dumps(report, indent=2), flush=True)
    if args.apply:
        apply(targets, args.allow_distro_lag)
    else:
        print("Plan only: no packages, boot files, drivers, or kernel were changed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, StopIteration, KeyError) as error:
        print("Stopped: " + str(error), file=sys.stderr)
        raise SystemExit(1)
