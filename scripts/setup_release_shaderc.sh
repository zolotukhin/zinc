#!/usr/bin/env bash
# Disposable Ubuntu CI worker only. Install the minimal, versioned shaderc
# package rather than the GUI-heavy Vulkan SDK metapackage.
set -euo pipefail
[[ "${GITHUB_ACTIONS:-}" = true || "${ZINC_RELEASE_SETUP:-}" = 1 ]] || {
  echo 'Set ZINC_RELEASE_SETUP=1 only on a disposable build worker.' >&2
  exit 1
}
. /etc/os-release
case "$ID-$VERSION_ID" in ubuntu-22.04|ubuntu-24.04) ;; *) echo 'Unsupported shader build image' >&2; exit 1 ;; esac
sudo apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update
sudo apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends ca-certificates curl gnupg libvulkan-dev
key="$(mktemp)"
trap 'rm -f "$key"' EXIT
curl -fsSL --connect-timeout 15 --max-time 120 --retry 3 https://packages.lunarg.com/lunarg-signing-key-pub.asc -o "$key"
sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/zinc-lunarg.gpg "$key"
printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/zinc-lunarg.gpg] https://packages.lunarg.com/vulkan/1.4.313 %s main\n' "$VERSION_CODENAME" | sudo tee /etc/apt/sources.list.d/zinc-lunarg.list >/dev/null
sudo apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 update
sudo apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 install -y --no-install-recommends "shaderc=2025.2~rc1-1lunarg${VERSION_ID}-1"
glslc --version
