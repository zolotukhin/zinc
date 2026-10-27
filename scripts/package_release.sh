#!/usr/bin/env bash
# Build-host wrapper; installed ZINC does not require Python.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "$script_dir/../tools/release.py" package "$@"
