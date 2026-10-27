#!/usr/bin/env python3
"""Build a release target on a disposable CI worker. Never runs inference."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from release_lib import ROOT, policy, require, source_sha, target, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = target(args.target)
    version(args.version)
    source_sha(args.commit)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(actual == args.commit, "build checkout does not match the frozen source commit")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=True)
    epoch = subprocess.check_output(["git", "show", "-s", "--format=%ct", args.commit], cwd=ROOT, text=True).strip()
    require(epoch.isdigit(), "invalid source timestamp")
    output = Path(args.output).resolve()
    prefix = output / "prefix"
    prefix.mkdir(parents=True, exist_ok=True)
    common = ["-Doptimize=ReleaseFast", "-Dtarget=" + spec["zig_target"], "-Dcpu=" + spec["cpu"],
              "-Dbackend=" + spec["backend"], "-Dversion=" + args.version, "-Dcommit=" + args.commit]
    if spec["backend"] == "metal":
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        subprocess.run([str(ROOT / "scripts/zig.sh"), "build", "tokenize-tool", *common, "--prefix", str(prefix)],
                       cwd=ROOT, env={**os.environ, "SDKROOT": sdk}, check=True)
        subprocess.run([str(ROOT / "scripts/zig.sh"), "build", *common, "--prefix", str(prefix)],
                       cwd=ROOT, env={**os.environ, "SDKROOT": sdk}, check=True)
    else:
        require(os.environ.get("GITHUB_ACTIONS") == "true", "Linux release builds require a disposable CI worker")
        if spec["backend"] == "vulkan":
            subprocess.run(["bash", str(ROOT / "scripts/setup_release_shaderc.sh")], check=True)
            subprocess.run([str(ROOT / "scripts/zig.sh"), "build", "release-shaders", "-Dbackend=vulkan", "--prefix", str(prefix)], cwd=ROOT, check=True)
        zig = Path(subprocess.check_output(["sh", "-c", "command -v zig"], text=True).strip()).resolve().parent
        # Export only the frozen Git tree: no credentials, local models, caches,
        # checkout metadata or untracked files enter the build container.
        source = tempfile.TemporaryDirectory(prefix="zinc-release-source-")
        source_archive = Path(source.name) / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output=" + str(source_archive), args.commit], cwd=ROOT, check=True)
        command = ["docker", "run", "--rm", "-i", "--platform", "linux/amd64",
                   "-v", str(source_archive) + ":/input/source.tar:ro", "-v", str(prefix) + ":/out",
                   "-v", str(zig) + ":/opt/zinc-zig:ro",
                   "-e", "PATH=/opt/zinc-zig:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                   "-e", "ZINC_RELEASE_BACKEND=" + spec["backend"], "-e", "ZINC_RELEASE_TARGET=" + spec["zig_target"],
                   "-e", "ZINC_RELEASE_VERSION=" + args.version, "-e", "ZINC_RELEASE_COMMIT=" + args.commit,
                   "-e", "SOURCE_DATE_EPOCH=" + epoch,
                   "-e", "ZINC_VULKAN_HEADERS_URL=" + policy()["toolchain"]["vulkan_headers"]["url"],
                   "-e", "ZINC_VULKAN_HEADERS_SHA=" + policy()["toolchain"]["vulkan_headers"]["sha256"],
                   "-e", "ZINC_OUTPUT_UID=" + str(os.getuid()), "-e", "ZINC_OUTPUT_GID=" + str(os.getgid()),
                   spec["image"], "bash", "-seu"]
        script = r'''
export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update
apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends python3 binutils ca-certificates curl
if [ "$ZINC_RELEASE_BACKEND" = rocm ]; then
  apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 install -y --no-install-recommends hipblas-dev=3.2.0.70204-93~24.04 rocwmma-dev=2.2.0.70204-93~24.04
  export LD_LIBRARY_PATH=/opt/rocm/lib
else
  apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends libvulkan-dev
  export ZINC_VULKAN_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
  # Keep the old loader/glibc ABI while compiling the extension declarations
  # needed by current shaders. These headers add no runtime dependency.
  curl -fsSL --connect-timeout 15 --max-time 120 --retry 3 "$ZINC_VULKAN_HEADERS_URL" -o /tmp/vulkan-headers.tar.gz
  printf '%s  %s\n' "$ZINC_VULKAN_HEADERS_SHA" /tmp/vulkan-headers.tar.gz | sha256sum -c -
  mkdir /tmp/vulkan-headers
  tar -xzf /tmp/vulkan-headers.tar.gz -C /tmp/vulkan-headers --strip-components=1
  cp -a /tmp/vulkan-headers/include/. /usr/include/
fi
# Build caches are disposable and source remains read-only.
mkdir /tmp/source
tar -xf /input/source.tar -C /tmp/source
cd /tmp/source
export ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache
./scripts/zig.sh build -Dtarget="$ZINC_RELEASE_TARGET" -Dcpu=baseline -Dbackend="$ZINC_RELEASE_BACKEND" -Dshaders=false -Doptimize=ReleaseFast -Dversion="$ZINC_RELEASE_VERSION" -Dcommit="$ZINC_RELEASE_COMMIT" --prefix /out
./scripts/zig.sh build tokenize-tool -Dtarget="$ZINC_RELEASE_TARGET" -Dcpu=baseline -Dbackend="$ZINC_RELEASE_BACKEND" -Dshaders=false -Doptimize=ReleaseFast --prefix /out
python3 tools/release.py package --version "$ZINC_RELEASE_VERSION" --target "linux-x86_64-$ZINC_RELEASE_BACKEND" --commit "$ZINC_RELEASE_COMMIT" --prefix /out --dist /out/archives
python3 tools/release.py install-smoke --version "$ZINC_RELEASE_VERSION" --target "linux-x86_64-$ZINC_RELEASE_BACKEND" --directory /out/archives
chown -R "$ZINC_OUTPUT_UID:$ZINC_OUTPUT_GID" /out
'''
        try:
            subprocess.run(command, input=script, text=True, check=True)
        finally:
            source.cleanup()
    if spec["backend"] == "metal":
        # Signing is a separate protected job. CI packages are deliberately
        # unsigned and cannot pass the final candidate manifest gate.
        subprocess.run([sys.executable, str(ROOT / "tools/release.py"), "package", "--version", args.version,
                        "--target", args.target, "--commit", args.commit, "--prefix", str(prefix),
                        "--dist", str(prefix / "archives")], env={**os.environ, "SOURCE_DATE_EPOCH": epoch}, check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/release.py"), "install-smoke", "--version", args.version,
                        "--target", args.target, "--directory", str(prefix / "archives")], check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit("release build: " + str(error))
