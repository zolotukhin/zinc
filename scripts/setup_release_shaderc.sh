#!/usr/bin/env bash
# Disposable Ubuntu CI worker only. Install the minimal, versioned shaderc
# package rather than the GUI-heavy Vulkan SDK metapackage.
set -euo pipefail

shaderc_require_disposable() {
  [[ "${GITHUB_ACTIONS:-}" = true || "${ZINC_RELEASE_SETUP:-}" = 1 ]] || {
    echo 'Set ZINC_RELEASE_SETUP=1 only on a disposable build worker.' >&2
    return 1
  }
}

shaderc_codename() {
  . /etc/os-release
  case "$ID-$VERSION_ID" in ubuntu-24.04) ;; *) echo 'Unsupported shader build image' >&2; return 1 ;; esac
  printf '%s\n' "$VERSION_CODENAME"
}

shaderc_write_sources() {
  # Hosted-runner mirror lists stalled for 40 minutes before source tests.
  # Keep authentication and updates, using the official HTTPS archives directly.
  cat > "$1" <<'SOURCES'
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu noble main universe
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu noble-updates main universe
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://security.ubuntu.com/ubuntu noble-security main universe
SOURCES
}

shaderc_apt() {
  local sources="$1"
  shift
  # Both overrides are required: runner image repositories must not supplement
  # this temporary list. Incomplete index downloads must fail, not use old data.
  sudo env DEBIAN_FRONTEND=noninteractive apt-get \
    -o "Dir::Etc::sourcelist=$sources" \
    -o "Dir::Etc::sourceparts=$(dirname -- "$sources")/empty-sources" \
    -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
    -o Acquire::Languages=none -o DPkg::Lock::Timeout=60 \
    -o APT::Update::Error-Mode=any "$@"
}

shaderc_install() {
  shaderc_require_disposable
  local script_dir shaderc_version codename sources key
  codename="$(shaderc_codename)"
  script_dir="$(CDPATH= cd -- "$(dirname -- "$1")" && pwd)"
  shaderc_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["toolchain"]["shaderc"])' "$script_dir/../config/release/targets.json")"
  # Keep this child-process variable alive for the EXIT trap after return.
  shaderc_workspace="$(mktemp -d)"
  # APT's unprivileged download user must be able to read the public sources.
  chmod 755 "$shaderc_workspace"
  trap 'rm -rf -- "$shaderc_workspace"' EXIT
  trap 'exit 143' TERM
  mkdir "$shaderc_workspace/empty-sources"
  sources="$shaderc_workspace/sources.list"
  key="$shaderc_workspace/lunarg.asc"
  shaderc_write_sources "$sources"
  shaderc_apt "$sources" update
  shaderc_apt "$sources" install -y --no-install-recommends ca-certificates curl gnupg libvulkan-dev
  curl -fsSL --connect-timeout 15 --max-time 120 --retry 3 https://packages.lunarg.com/lunarg-signing-key-pub.asc -o "$key"
  sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/zinc-lunarg.gpg "$key"
  printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/zinc-lunarg.gpg] https://packages.lunarg.com/vulkan/1.4.313 %s main\n' "$codename" >> "$sources"
  shaderc_apt "$sources" update
  shaderc_apt "$sources" install -y --no-install-recommends "shaderc=$shaderc_version"
  glslc --version
}

shaderc_bounded_install() {
  # Per-socket APT timeouts did not bound mirror-list failover. Bound the entire
  # setup and its children too; a stalled repository must not hold CI for hours.
  timeout --signal=TERM --kill-after=30s 600s \
    bash -c 'source "$1"; shaderc_install "$1"' -- "$1"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  shaderc_require_disposable
  shaderc_bounded_install "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
fi
