#!/usr/bin/env bash
# Install verified ZINC releases without a source checkout or language runtime.
# ZINC_VERSION=v0.1.0 ZINC_BACKEND=rocm bash scripts/install.sh
# bash scripts/install.sh --rollback
# Overrides: ZINC_INSTALL_DIR, ZINC_BIN_DIR, ZINC_BASE_URL (mirrors/testing).
set -euo pipefail
export LC_ALL=C
unset TAR_OPTIONS TAPE
REPO="zolotukhin/zinc"
log() { printf 'zinc-install: %s\n' "$*"; }
fail() { printf 'zinc-install: error: %s\n' "$*" >&2; exit 1; }
for tool in curl tar gzip head awk find od tr cmp; do
  command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done
if command -v sha256sum >/dev/null 2>&1; then
  digest() { sha256sum "$1" | awk '{print $1}'; }
else
  command -v shasum >/dev/null 2>&1 || fail "sha256sum or shasum is required"
  digest() { shasum -a 256 "$1" | awk '{print $1}'; }
fi
semver() {
  [[ "$1" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-([0-9A-Za-z-]+)(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]] || return 1
  # Numeric prerelease identifiers cannot have leading zeroes.
  local core="${1%%+*}" pre part
  pre="${core#*-}"
  if [ "$pre" != "$core" ]; then
    pre="${pre%%+*}"
    while [ -n "$pre" ]; do
      part="${pre%%.*}"
      [[ "$part" =~ ^0[0-9]+$ ]] && return 1
      [ "$part" != "$pre" ] || break
      pre="${pre#*.}"
    done
  fi
}
version_at_least() { awk -v actual="$1" -v required="$2" 'BEGIN { n=split(actual,a,"."); m=split(required,b,"."); for(i=1;i<=n||i<=m;i++){if(a[i]+0>b[i]+0)exit 0;if(a[i]+0<b[i]+0)exit 1}exit 0 }'; }
operation=install
if [ "$#" -gt 0 ]; then
  [ "$#" -eq 1 ] && [ "$1" = --rollback ] || fail "usage: install.sh [--rollback]"
  operation=rollback
fi
os="$(uname -s)"
arch="$(uname -m)"
case "$os-$arch" in
  Linux-x86_64) platform=linux-x86_64; backend="${ZINC_BACKEND:-vulkan}" ;;
  Darwin-arm64) platform=macos-aarch64; backend="${ZINC_BACKEND:-metal}" ;;
  Darwin-x86_64)
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
      fail "Rosetta translation detected; run the installer from a native arm64 shell"
    fi
    fail "Intel Macs do not have a supported binary release" ;;
  *) fail "no binary release for $os/$arch" ;;
esac
case "$platform-$backend" in
  macos-aarch64-metal|linux-x86_64-vulkan|linux-x86_64-rocm) ;;
  *) fail "unsupported backend '$backend' for $platform" ;;
esac
target="$platform-$backend"
install_root="${ZINC_INSTALL_DIR:-$HOME/.local/share/zinc}"
bin_dir="${ZINC_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$install_root" "$bin_dir"
install_root="$(CDPATH= cd -- "$install_root" && pwd -P)"
bin_dir="$(CDPATH= cd -- "$bin_dir" && pwd -P)"
lock="$install_root/.install.lock"
mkdir "$lock" 2>/dev/null || fail "another installer is active or left $lock; inspect it before retrying"
printf '%s\n' "$$" > "$lock/pid"
staging=""
created_bin=0
completed=0
cleanup() {
  if [ "$completed" != 1 ] && [ "$created_bin" = 1 ] && [ -L "$bin_dir/zinc" ] && [ "$(readlink "$bin_dir/zinc")" = "$install_root/current/bin/zinc" ]; then
    rm -f "$bin_dir/zinc"
  fi
  [ -z "$staging" ] || rm -rf -- "$staging"
  rm -f "$lock/pid"
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
staging="$(mktemp -d "$install_root/.staging.XXXXXXXX")"
# mv treats a destination symlink to a directory differently on GNU/BSD.
replace_link() {
  case "$os" in Linux) mv -fT "$1" "$2" ;; Darwin) mv -fh "$1" "$2" ;; esac
}
atomic_link() {
  local destination="$1" value="$2" prepared
  prepared="$(mktemp "$install_root/.link.XXXXXXXX")"
  rm -f "$prepared"
  ln -s "$value" "$prepared"
  if ! replace_link "$prepared" "$destination"; then rm -f "$prepared"; return 1; fi
}
check_pointer() {
  local pointer="$1" value
  [ -e "$pointer" ] || [ -L "$pointer" ] || return 0
  [ -L "$pointer" ] || fail "refusing to replace unmanaged path: $pointer"
  value="$(readlink "$pointer")"
  case "$value" in "$install_root"/zinc-v*) ;; *) fail "refusing unmanaged installation pointer: $pointer" ;; esac
  [ "$value" = "$install_root/$(basename "$value")" ] && [ ! -L "$value" ] || fail "refusing unmanaged installation pointer: $pointer"
}
check_pointer "$install_root/current"
check_pointer "$install_root/previous"
if [ -e "$bin_dir/zinc" ] || [ -L "$bin_dir/zinc" ]; then
  [ -L "$bin_dir/zinc" ] && [ "$(readlink "$bin_dir/zinc")" = "$install_root/current/bin/zinc" ] || fail "refusing to replace unmanaged $bin_dir/zinc"
fi
read_metadata() {
  local tree="$1" line key value seen='|'
  meta_format='' meta_version='' meta_commit='' meta_target='' meta_backend='' meta_os='' meta_glibc='' meta_binary=''
  [ -f "$tree/INSTALL-METADATA" ] && [ ! -L "$tree/INSTALL-METADATA" ] || fail "missing installer metadata"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[a-z_0-9]+=[A-Za-z0-9_.+-]+$ ]] || fail "invalid installer metadata"
    key="${line%%=*}"; value="${line#*=}"
    case "$seen" in *"|$key|"*) fail "duplicate installer metadata" ;; esac
    seen="$seen$key|"
    case "$key" in
      format) meta_format="$value" ;; version) meta_version="$value" ;;
      commit) meta_commit="$value" ;; target) meta_target="$value" ;;
      backend) meta_backend="$value" ;; minimum_os) meta_os="$value" ;;
      minimum_glibc) meta_glibc="$value" ;; binary_sha256) meta_binary="$value" ;;
      *) fail "unknown installer metadata field" ;;
    esac
  done < "$tree/INSTALL-METADATA"
  [ "$meta_format" = 1 ] && semver "v$meta_version" || fail "invalid package format/version"
  [[ "$meta_commit" =~ ^[0-9a-f]{40}$ ]] && [[ "$meta_binary" =~ ^[0-9a-f]{64}$ ]] || fail "invalid package digest"
  [ "$meta_target" = "$platform-$meta_backend" ] || fail "package architecture/backend mismatch"
  case "$meta_target" in macos-aarch64-metal|linux-x86_64-vulkan|linux-x86_64-rocm) ;; *) fail "unsupported package target" ;; esac
  [ -n "$meta_os" ] && [ -n "$meta_glibc" ] || fail "missing compatibility metadata"
}
check_compatibility() {
  if [ "$os" = Darwin ]; then
    [[ "$meta_os" =~ ^[0-9]+(\.[0-9]+)+$ ]] || fail "invalid macOS minimum"
    version_at_least "$(sw_vers -productVersion)" "$meta_os" || fail "requires macOS $meta_os or newer"
    [ "$meta_glibc" = none ] || fail "invalid Mac compatibility metadata"
  else
    [ "$meta_os" = linux ] || fail "invalid Linux compatibility metadata"
    local libc
    libc="$(getconf GNU_LIBC_VERSION 2>/dev/null)" || fail "this download requires glibc; musl/Alpine is not supported"
    [[ "$meta_glibc" =~ ^[0-9]+\.[0-9]+$ ]] || fail "invalid glibc minimum"
    version_at_least "${libc#glibc }" "$meta_glibc" || fail "requires glibc $meta_glibc or newer"
  fi
}
verify_tree() {
  local tree="$1" line hash relative count=0 actual_count
  read_metadata "$tree"
  check_compatibility
  [ -f "$tree/FILES.sha256" ] && [ -f "$tree/VERSION.json" ] && [ -x "$tree/bin/zinc" ] || fail "incomplete release layout"
  awk -F'"' -v v="$meta_version" -v c="$meta_commit" -v t="$meta_target" -v b="$meta_backend" -v o="$meta_os" -v g="$meta_glibc" '
    BEGIN {expected["version"]=v;expected["commit"]=c;expected["target"]=t;expected["backend"]=b;expected["minimum_os"]=o;expected["minimum_glibc"]=g}
    /^  "(version|commit|target|backend|minimum_os|minimum_glibc)":/ {
      key=$2;line="  \""key"\": \""expected[key]"\"";
      if(($0!=line&&$0!=line",")||seen[key]++)exit 1;count++
    }
    END {if(count!=6)exit 1}
  ' "$tree/VERSION.json" || fail "JSON and installer metadata disagree"
  [ -z "$(find "$tree" -type l -print)" ] || fail "links in installed release are forbidden"
  : > "$staging/seen-files"
  while IFS= read -r line || [ -n "$line" ]; do
    hash="${line%%  *}"; relative="${line#*  }"
    [[ "$hash" =~ ^[0-9a-f]{64}$ ]] && [[ "$relative" =~ ^[A-Za-z0-9_./+-]+$ ]] || fail "invalid file inventory"
    [ "$line" = "$hash  $relative" ] || fail "invalid inventory separator"
    case "/$relative/" in *'/../'*|*'/./'*|*'//'*) fail "unsafe inventory path" ;; esac
    case "$relative" in FILES.sha256|.archive-sha256) fail "self-referencing file inventory" ;; esac
    printf '%s\n' "$relative" >> "$staging/seen-files"
    [ -f "$tree/$relative" ] && [ "$(digest "$tree/$relative")" = "$hash" ] || fail "file checksum mismatch: $relative"
    count=$((count + 1))
  done < "$tree/FILES.sha256"
  awk 'seen[$0]++{exit 1}' "$staging/seen-files" || fail "duplicate inventory entry"
  actual_count="$(find "$tree" -type f ! -name FILES.sha256 ! -name .archive-sha256 | wc -l | tr -d ' ')"
  [ "$count" -eq "$actual_count" ] && [ "$count" -gt 0 ] || fail "release file inventory mismatch"
  [ "$(digest "$tree/bin/zinc")" = "$meta_binary" ] || fail "binary digest mismatch"
  local magic machine output
  magic="$(od -An -tx1 -N8 "$tree/bin/zinc" | tr -d ' \n')"
  if [ "$os" = Darwin ]; then
    [ "$magic" = cffaedfe0c000001 ] || fail "expected an ARM64 Mach-O executable"
  else
    machine="$(od -An -tx1 -j18 -N2 "$tree/bin/zinc" | tr -d ' \n')"
    [[ "$magic" = 7f454c460201* ]] && [ "$machine" = 3e00 ] || fail "expected an x86_64 ELF executable"
  fi
  output="$("$tree/bin/zinc" --version 2>"$staging/version-error")" || fail "binary --version failed; check the backend libraries and OS requirements"
  printf '%s\n' "$output" | awk -v v="$meta_version" -v c="$meta_commit" -v b="$meta_backend" -v p="$platform" '
    NR==1 {if($0!="zinc "v)exit 1}
    /^commit: / {if($0!="commit: "c)exit 1;commit++}
    /^backends: / {if($0!="backends: "b)exit 1;backend++}
    /^optimize: / {if($0!="optimize: ReleaseFast"&&$0!="optimize: fast")exit 1;optimize++}
    /^target: / {t=(p=="macos-aarch64"?"aarch64-macos-none":"x86_64-linux-gnu");if($0!="target: "t)exit 1;target++}
    END {if(commit!=1||backend!=1||optimize!=1||target!=1)exit 1}
  ' || fail "binary version identity does not match package metadata"
}
activate() {
  local tree="$1" old_current='' old_previous='' bin_prepared
  [ ! -L "$install_root/current" ] || old_current="$(readlink "$install_root/current")"
  [ ! -L "$install_root/previous" ] || old_previous="$(readlink "$install_root/previous")"
  if [ ! -L "$bin_dir/zinc" ]; then
    bin_prepared="$(mktemp "$bin_dir/.zinc-link.XXXXXXXX")"
    rm -f "$bin_prepared"
    ln -s "$install_root/current/bin/zinc" "$bin_prepared"
    if ! replace_link "$bin_prepared" "$bin_dir/zinc"; then rm -f "$bin_prepared"; fail "cannot install binary symlink"; fi
    created_bin=1
  fi
  if [ -n "$old_current" ]; then atomic_link "$install_root/previous" "$old_current" || fail "cannot save rollback pointer"; fi
  if ! atomic_link "$install_root/current" "$tree"; then
    if [ -n "$old_previous" ]; then atomic_link "$install_root/previous" "$old_previous" || true; else rm -f "$install_root/previous"; fi
    fail "cannot activate installation"
  fi
  completed=1
}
if [ "$operation" = rollback ]; then
  [ -L "$install_root/previous" ] || fail "no previous installation is available"
  previous="$(readlink "$install_root/previous")"
  case "$previous" in "$install_root"/zinc-v*) ;; *) fail "invalid rollback location" ;; esac
  [ "$previous" = "$install_root/$(basename "$previous")" ] || fail "invalid rollback location"
  [ -d "$previous" ] && [ ! -L "$previous" ] || fail "previous installation is missing"
  verify_tree "$previous"
  [ "$(basename "$previous")" = "zinc-v$meta_version-$meta_target" ] || fail "rollback identity mismatch"
  activate "$previous"
  log "rolled back to zinc $meta_version ($meta_backend)"
  exit 0
fi
tag="${ZINC_VERSION:-latest}"
fetch() { curl -fsSL --connect-timeout 15 --max-time 300 --retry 2 --max-filesize 134217728 "$@"; }
if [ "$tag" = latest ]; then
  effective="$(fetch -I -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest")" || fail "cannot resolve the latest stable release"
  case "$effective" in "https://github.com/$REPO/releases/tag/"*) tag="${effective##*/}" ;; *) fail "no published stable release found" ;; esac
  case "${tag%%+*}" in *-*) fail "latest resolved to a prerelease; select it explicitly" ;; esac
fi
semver "$tag" || fail "invalid release version"
name="zinc-$tag-$target"
asset="$name.tar.gz"
base="${ZINC_BASE_URL:-https://github.com/$REPO/releases/download}"
log "downloading $tag ($backend)"
fetch -o "$staging/$asset" "$base/$tag/$asset" || fail "release download failed"
fetch -o "$staging/SHA256SUMS.txt" "$base/$tag/SHA256SUMS.txt" || fail "checksum download failed"
expected="$(awk -v name="$asset" 'NF==2&&$2==name {value=$1;n++} END{if(n!=1)exit 1;print value}' "$staging/SHA256SUMS.txt")" || fail "missing or duplicate checksum entry for $asset"
[[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "invalid archive checksum"
[ "$(digest "$staging/$asset")" = "$expected" ] || fail "archive checksum mismatch"
log "checksum verified"
# Bound decompression before listing/extracting, including highly compressed payloads.
if ! gzip -dc "$staging/$asset" | head -c 536870913 > "$staging/archive.tar"; then fail "invalid or oversized compressed archive"; fi
[ "$(wc -c < "$staging/archive.tar" | tr -d ' ')" -le 536870912 ] || fail "archive expands beyond its size limit"
tar -tf "$staging/archive.tar" | awk 'NR>4096{exit 1}{sub(/\/$/,"");if(seen[$0]++)exit 1;print}' > "$staging/names" || fail "invalid or duplicate archive entries"
while IFS= read -r entry; do
  [[ "$entry" =~ ^[A-Za-z0-9_./+-]+$ ]] || fail "unsafe archive filename"
  case "/$entry/" in *'/../'*|*'/./'*|*'//'*) fail "unsafe archive path" ;; esac
  case "$entry" in
    "$name"|"$name/bin"|"$name/bin/zinc"|"$name/bin/zinc-tokenize"|"$name/README.md"|"$name/LICENSE"|"$name/VERSION.json"|"$name/INSTALL-METADATA"|"$name/FILES.sha256"|"$name/share"|"$name/share/zinc"|"$name/share/zinc/asset-manifest.json"|"$name/share/zinc/shaders"|"$name/share/zinc/shaders/metal") ;;
    "$name/share/zinc/shaders/"*.spv|"$name/share/zinc/shaders/metal/"*.metal) ;;
    *) fail "unexpected archive entry" ;;
  esac
done < "$staging/names"
tar -tvf "$staging/archive.tar" | awk '{t=substr($1,1,1);if((t!="-"&&t!="d")||$1~/[sStT]/)exit 1}' || fail "archive links or special files are forbidden"
mkdir "$staging/unpack"
tar -xf "$staging/archive.tar" -C "$staging/unpack" --no-same-owner
[ -d "$staging/unpack/$name" ] || fail "unexpected archive layout"
verify_tree "$staging/unpack/$name"
[ "v$meta_version" = "$tag" ] && [ "$meta_target" = "$target" ] && [ "$meta_backend" = "$backend" ] || fail "package identity mismatch"
case "$backend" in
  metal) [ -n "$(find "$staging/unpack/$name/share/zinc/shaders/metal" -type f -name '*.metal' -print -quit)" ] || fail "missing Metal shaders" ;;
  vulkan) [ -n "$(find "$staging/unpack/$name/share/zinc/shaders" -maxdepth 1 -type f -name '*.spv' -print -quit)" ] || fail "missing Vulkan shaders" ;;
esac
if [ -e "$install_root/$name" ] || [ -L "$install_root/$name" ]; then
  [ ! -L "$install_root/$name" ] && [ -f "$install_root/$name/.archive-sha256" ] && [ "$(cat "$install_root/$name/.archive-sha256")" = "$expected" ] || fail "an installation with different bytes already uses this version"
  verify_tree "$install_root/$name"
else
  printf '%s\n' "$expected" > "$staging/unpack/$name/.archive-sha256"
  mv "$staging/unpack/$name" "$install_root/$name"
fi
if [ -L "$install_root/current" ] && [ "$(readlink "$install_root/current")" = "$install_root/$name" ]; then
  if [ ! -L "$bin_dir/zinc" ]; then ln -s "$install_root/current/bin/zinc" "$bin_dir/zinc"; fi
  completed=1
else
  activate "$install_root/$name"
fi
log "installed zinc $meta_version ($backend) -> $bin_dir/zinc"
case ":$PATH:" in *":$bin_dir:"*) ;; *) log "add $bin_dir to your PATH" ;; esac
log "next: zinc --check; zinc model list"
