#!/usr/bin/env bash
# ZINC installer: downloads the prebuilt release for this machine, verifies
# its checksum, and installs it under the user's home directory.
#
#   curl -fsSL https://raw.githubusercontent.com/zolotukhin/zinc/main/scripts/install.sh | bash
#
# Environment overrides:
#   ZINC_VERSION      Release tag to install (default: latest, e.g. v0.1.0)
#   ZINC_INSTALL_DIR  Install root (default: ~/.local/share/zinc)
#   ZINC_BIN_DIR      Symlink target dir for the zinc binary (default: ~/.local/bin)
#   ZINC_BASE_URL     Alternate download base URL (mirrors / testing). Assets are
#                     expected at $ZINC_BASE_URL/<tag>/<asset>.
set -euo pipefail

REPO="zolotukhin/zinc"

log() { printf 'zinc-install: %s\n' "$*"; }
fail() {
  printf 'zinc-install: error: %s\n' "$*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

has_file_matching() {
  find "$1" -maxdepth 1 -type f -name "$2" -print -quit | grep -q .
}

# ── Platform detection ────────────────────────────────────────
os="$(uname -s)"
arch="$(uname -m)"
case "${os}-${arch}" in
  Linux-x86_64) target="linux-x86_64" ;;
  Darwin-arm64) target="macos-aarch64" ;;
  Darwin-x86_64)
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
      fail "this shell is running under Rosetta translation on Apple Silicon. Run it natively instead, e.g.: arch -arm64 zsh -c \"\$(curl -fsSL https://raw.githubusercontent.com/${REPO}/main/scripts/install.sh)\""
    fi
    fail "no prebuilt binary for ${os}/${arch}. Build from source instead: https://github.com/${REPO}#start-here"
    ;;
  *)
    fail "no prebuilt binary for ${os}/${arch}. Build from source instead: https://github.com/${REPO}#start-here"
    ;;
esac

# ── Resolve release tag ───────────────────────────────────────
tag="${ZINC_VERSION:-latest}"
if [ "$tag" = "latest" ]; then
  # Follow the /releases/latest redirect and read the tag off the final URL.
  # Avoids a jq dependency and GitHub API rate limits.
  effective_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")" ||
    fail "could not reach github.com to resolve the latest release"
  tag="${effective_url##*/}"
  case "$tag" in
    v*) ;;
    *) fail "no published release found for ${REPO} (resolved '${tag}'). Pass ZINC_VERSION=vX.Y.Z or build from source." ;;
  esac
fi

asset="zinc-${tag}-${target}.tar.gz"
base_url="${ZINC_BASE_URL:-https://github.com/${REPO}/releases/download}"
asset_url="${base_url}/${tag}/${asset}"
sums_url="${base_url}/${tag}/SHA256SUMS.txt"

install_root="${ZINC_INSTALL_DIR:-${HOME}/.local/share/zinc}"
bin_dir="${ZINC_BIN_DIR:-${HOME}/.local/bin}"

log "installing zinc ${tag} (${target})"

# ── Download and verify ───────────────────────────────────────
tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

log "downloading ${asset}"
curl -fsSL -o "${tmp}/${asset}" "$asset_url" ||
  fail "download failed: ${asset_url}"
curl -fsSL -o "${tmp}/SHA256SUMS.txt" "$sums_url" ||
  fail "checksum download failed: ${sums_url}"

expected="$(awk -v asset="$asset" '$2 == asset { print $1; found = 1; exit } END { exit found ? 0 : 1 }' "${tmp}/SHA256SUMS.txt")" ||
  fail "no checksum entry for ${asset} in SHA256SUMS.txt"
[ -n "$expected" ] || fail "no checksum entry for ${asset} in SHA256SUMS.txt"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "${tmp}/${asset}" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "${tmp}/${asset}" | awk '{print $1}')"
fi
[ "$expected" = "$actual" ] || fail "checksum mismatch for ${asset}: expected ${expected}, got ${actual}"
log "checksum verified"

# ── Install ───────────────────────────────────────────────────
tar -xzf "${tmp}/${asset}" -C "$tmp" --no-same-owner
tree_name="zinc-${tag}-${target}"
[ -d "${tmp}/${tree_name}" ] || fail "unexpected archive layout: missing ${tree_name}/"
[ -x "${tmp}/${tree_name}/bin/zinc" ] || fail "unexpected archive layout: missing bin/zinc"
case "$target" in
  linux-x86_64)
    shader_dir="${tmp}/${tree_name}/share/zinc/shaders"
    [ -d "$shader_dir" ] || fail "unexpected archive layout: missing share/zinc/shaders/"
    has_file_matching "$shader_dir" "*.spv" || fail "unexpected archive layout: no SPIR-V shaders in share/zinc/shaders/"
    ;;
  macos-aarch64)
    shader_dir="${tmp}/${tree_name}/share/zinc/shaders/metal"
    [ -d "$shader_dir" ] || fail "unexpected archive layout: missing share/zinc/shaders/metal/"
    has_file_matching "$shader_dir" "*.metal" || fail "unexpected archive layout: no Metal shaders in share/zinc/shaders/metal/"
    ;;
esac

mkdir -p "$install_root" "$bin_dir"
rm -rf "${install_root:?}/${tree_name}"
mv "${tmp}/${tree_name}" "${install_root}/${tree_name}"
# The binary resolves its shaders relative to its real path
# (../share/zinc/shaders), and selfExePath follows symlinks, so a symlinked
# binary keeps working. 'current' gives a stable path across upgrades.
ln -sfn "${install_root}/${tree_name}" "${install_root}/current"
ln -sf "${install_root}/current/bin/zinc" "${bin_dir}/zinc"

installed_version="$("${bin_dir}/zinc" --version 2>/dev/null | head -1 || true)"
log "installed: ${installed_version:-zinc ${tag}} -> ${bin_dir}/zinc"

case ":${PATH}:" in
  *":${bin_dir}:"*) ;;
  *)
    log "note: ${bin_dir} is not on your PATH. Add it, e.g.:"
    log "  export PATH=\"${bin_dir}:\$PATH\""
    ;;
esac

log "next steps: 'zinc --check' to verify your GPU, 'zinc model list' to pick a model"
