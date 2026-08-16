#!/usr/bin/env bash
# Catalog decode-perf measurement across the NVIDIA node's GPUs (RTX 5090 + 4090).
#
# Drives `dbg_cuda gen` for each catalog model — which runs that arch's REAL
# decodeStep (qwen35/qwen36 = ForwardCuda async CUstream/CUevent ring; gemma4 =
# ForwardGemma, currently sync-per-layer) — for NGEN greedy tokens and reports
# the steady-state tok/s the harness prints. The qwen path exercises the async
# ring; the gemma path is sync-per-layer (no ring ported yet) = the headroom.
#
# The CUDA kernels are NVRTC-compiled at runtime for the *visible* device's
# compute capability, so ONE build runs on either GPU — we just flip
# CUDA_VISIBLE_DEVICES per GPU. 5090 = sm_120 (Blackwell), 4090 = sm_89 (Ada).
#
# Run on the box from ~/workspace/zinc. GPU UUIDs are supplied through the
# environment; do not commit machine-specific UUIDs.
#
# Env overrides:
#   ZINC_GPUS    space list of targets to test   (default: "5090 4090")
#   ZINC_GPU_5090 CUDA_VISIBLE_DEVICES value for the 5090 target
#   ZINC_GPU_4090 CUDA_VISIBLE_DEVICES value for the 4090 target
#   ZINC_MODELS  catalog gguf dir                 (default: ~/workspace/models)
#   ZINC_NGEN    greedy tokens per run            (default: 160)
#   ZINC_RUNS    runs per model (report best)     (default: 2)
#   ZIG          zig binary                        (default: ~/zig-0.15.2/zig)
#   ZINC_ONLY    space list of model names to run (default: all)
set -u

# --- GPU registry: name -> CUDA_VISIBLE_DEVICES selector.
# Keep actual UUIDs in .env or the shell environment, not in tracked files.
declare -A GPU_UUID=(
  [5090]="${ZINC_GPU_5090:-}"
  [4090]="${ZINC_GPU_4090:-}"
)
declare -A GPU_DESC=(
  [5090]="sm_120 Blackwell, 32 GB, 1792 GB/s"
  [4090]="sm_89 Ada, 24 GB, 1008 GB/s"
)

GPUS=${ZINC_GPUS:-"5090 4090"}
ZIG=${ZIG:-$HOME/zig-0.15.2/zig}
MD=${ZINC_MODELS:-$HOME/workspace/models}
NGEN=${ZINC_NGEN:-160}
RUNS=${ZINC_RUNS:-2}
ONLY=${ZINC_ONLY:-}
cd "$HOME/workspace/zinc" || { echo "no ~/workspace/zinc"; exit 1; }

# --- catalog ---
NAMES=(qwen35-9b      qwen38-27b      qwen36-35b-a3b  gemma4-31b      gemma4-26b)
PATHS=(
  "$HOME/workspace/Qwen3.5-9B-Q4_K_M.gguf"
  "$MD/Qwen3.8-27B-Q4_K_M.gguf"
  "$MD/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  "$MD/gemma-4-31B-it-Q4_K_M.gguf"
  "$MD/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)
ENGINE=(ForwardCuda-async ForwardCuda-async ForwardCuda-async ForwardGemma-sync ForwardGemma-sync)

# --- build once (NVRTC compiles per-device at runtime) ---
echo "building cuda-dbg (one build runs on every GPU)..."
"$ZIG" build cuda-dbg -Dbackend=cuda -Dshaders=false >/tmp/perf.build 2>&1 \
  || { echo "BUILD FAIL"; grep -E 'error:' /tmp/perf.build | head; exit 1; }
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary"; exit 1; }
echo "  binary: $ZBIN"

# best (max) of a whitespace-separated numeric list, or empty
best() { tr ' ' '\n' <<<"$1" | grep -E '^[0-9.]+$' | sort -gr | head -1; }

declare -A RESULT  # "gpu|name" -> best tok/s

run_one_gpu() {
  local gpu="$1" uuid="$2"
  export CUDA_VISIBLE_DEVICES="$uuid"
  printf '\n=== RTX %s decode tok/s  (%s)  |  %s-tok greedy, best of %s ===\n' \
    "$gpu" "${GPU_DESC[$gpu]}" "$NGEN" "$RUNS"
  printf '  %-15s %-18s  %s\n' "model" "engine" "tok/s (best [all runs])"
  local i nm m eng rates v line b r
  for i in "${!NAMES[@]}"; do
    nm="${NAMES[$i]}"; m="${PATHS[$i]}"; eng="${ENGINE[$i]}"
    if [ -n "$ONLY" ] && ! grep -qw "$nm" <<<"$ONLY"; then continue; fi
    if [ ! -f "$m" ]; then printf '  %-15s %-18s  MISSING (%s)\n' "$nm" "$eng" "$m"; continue; fi
    rates=""
    for r in $(seq 1 "$RUNS"); do
      line=$(timeout 360 "$ZBIN" gen 1,2,3 "$NGEN" "$m" 2>&1 | grep -E 'tok/s' | tail -1)
      v=$(sed -E 's/.*[=:] *([0-9.]+) *tok.*/\1/' <<<"$line")
      [ -n "$v" ] && rates="$rates $v"
    done
    b=$(best "$rates"); b=${b:-"-"}
    RESULT["$gpu|$nm"]="$b"
    printf '  %-15s %-18s  %-8s [%s ]\n' "$nm" "$eng" "$b" "$rates"
  done
}

for gpu in $GPUS; do
  uuid="${GPU_UUID[$gpu]:-}"
  [ -n "$uuid" ] || { echo "missing selector for GPU '$gpu'; set ZINC_GPU_${gpu}=<cuda-visible-device>"; continue; }
  run_one_gpu "$gpu" "$uuid"
done

# --- side-by-side comparison when both GPUs ran ---
if grep -qw 5090 <<<"$GPUS" && grep -qw 4090 <<<"$GPUS"; then
  printf '\n=== 5090 vs 4090 decode (best tok/s, %s-tok greedy) ===\n' "$NGEN"
  printf '  %-15s %-18s %10s %10s %9s\n' "model" "engine" "5090" "4090" "5090/4090"
  for i in "${!NAMES[@]}"; do
    nm="${NAMES[$i]}"; eng="${ENGINE[$i]}"
    if [ -n "$ONLY" ] && ! grep -qw "$nm" <<<"$ONLY"; then continue; fi
    a="${RESULT["5090|$nm"]:-}"; b="${RESULT["4090|$nm"]:-}"
    ratio="-"
    if [[ "$a" =~ ^[0-9.]+$ && "$b" =~ ^[0-9.]+$ ]]; then
      ratio=$(awk -v x="$a" -v y="$b" 'BEGIN{ if(y>0) printf "%.2fx", x/y; else print "-" }')
    fi
    printf '  %-15s %-18s %10s %10s %9s\n' "$nm" "$eng" "${a:--}" "${b:--}" "$ratio"
  done
fi
echo "=== done ==="
