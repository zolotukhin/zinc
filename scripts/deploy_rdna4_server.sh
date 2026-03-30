#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env in repo root" >&2
  exit 1
fi

source .env

REMOTE_DIR="${ZINC_REMOTE_DIR:-/root/zinc}"
MODEL_PATH="${ZINC_REMOTE_MODEL:-/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
PORT="${ZINC_SERVER_PORT:-9090}"
LOG_PATH="${ZINC_REMOTE_LOG:-/tmp/zinc_${PORT}.log}"

do_sync=1
do_build=1
do_restart=1
do_healthcheck=1

usage() {
  cat <<EOF
Usage: scripts/deploy_rdna4_server.sh [options]

Options:
  --no-sync         Skip rsync to the remote node
  --no-build        Skip remote ReleaseFast zig build
  --no-restart      Skip restarting the remote server
  --no-healthcheck  Skip final /health check
  --help            Show this help

Environment overrides:
  ZINC_REMOTE_DIR    Remote checkout path (default: $REMOTE_DIR)
  ZINC_REMOTE_MODEL  Remote model path (default: $MODEL_PATH)
  ZINC_SERVER_PORT   Remote server port (default: $PORT)
  ZINC_REMOTE_LOG    Remote log path (default: $LOG_PATH)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync)
      do_sync=0
      ;;
    --no-build)
      do_build=0
      ;;
    --no-restart)
      do_restart=0
      ;;
    --no-healthcheck)
      do_healthcheck=0
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

ssh_target="${ZINC_USER}@${ZINC_HOST}"
ssh_cmd=(ssh -p "$ZINC_PORT" "$ssh_target")

if (( do_sync )); then
  echo "==> Syncing source to $ssh_target:$REMOTE_DIR"
  rsync -az --delete \
    --exclude '.git' \
    --exclude '.zig-cache' \
    --exclude 'zig-out' \
    --exclude 'node_modules' \
    --exclude '.DS_Store' \
    --exclude 'site' \
    -e "ssh -p $ZINC_PORT" \
    . "$ssh_target:$REMOTE_DIR/"
fi

if (( do_build )); then
  echo "==> Building on remote node"
  "${ssh_cmd[@]}" "cd '$REMOTE_DIR' && zig build -Doptimize=ReleaseFast"
fi

if (( do_restart )); then
  echo "==> Restarting server on port $PORT"
  "${ssh_cmd[@]}" "pid=\$(ss -ltnp | sed -n 's/.*:${PORT} .*pid=\\([0-9][0-9]*\\).*/\\1/p' | head -n1); if [ -n \"\$pid\" ]; then kill -KILL \"\$pid\" 2>/dev/null || true; sleep 1; fi; cd '$REMOTE_DIR' && nohup env RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc --model '$MODEL_PATH' --port '$PORT' >'$LOG_PATH' 2>&1 </dev/null & echo \$!"
fi

if (( do_healthcheck )); then
  echo "==> Health check"
  "${ssh_cmd[@]}" "curl -fsS http://127.0.0.1:${PORT}/health"
fi
