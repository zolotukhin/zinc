#!/usr/bin/env bash
# Wait for the scoreboard to finish (free the 4090), then nsys-profile ONE prefill
# of gemma-31b (dense, the cuBLAS+round-trip path) and qwen35-9b to see where
# prefill time actually goes: dequant_q4k_to_f16 (the round-trip) vs cublas_hgemm
# vs f32_to_f16 vs attention/SSM/norms. Confirms whether the fused-GEMM is the lever.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib/cuda_target.sh"
zinc_select_cuda_device || exit 1
ZBIN=$HOME/zinc-main-meas/zig-out/bin/zinc
NSYS=/usr/local/cuda/bin/nsys
WAIT_PID=${1:-0}

if [ "$WAIT_PID" != 0 ]; then
  echo "waiting for scoreboard pid $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
  echo "scoreboard done; starting profile $(date -u +%FT%TZ)"
fi

S="The memory hierarchy of a modern graphics processor determines how quickly a large language model processes its prompt and generates new tokens. "
P=""; for i in $(seq 1 45); do P="$P$S"; done

for tag in gemma4-31b qwen35-9b; do
  case "$tag" in
    gemma4-31b) M=$HOME/workspace/models/gemma-4-31B-it-Q4_K_M.gguf ;;
    qwen35-9b)  M=$HOME/workspace/models/Qwen3.5-9B-Q4_K_M.gguf ;;
  esac
  echo "============================================================"
  echo "PROFILE $tag (prefill, -n 1)"
  echo "============================================================"
  "$NSYS" profile --stats=true -o /tmp/prof_$tag --force-overwrite true \
    "$ZBIN" -m "$M" --prompt "$P" --raw -n 1 2>&1 \
    | grep -A 22 "CUDA GPU Kernel Summary\|cuda_gpu_kern_sum\|Time (%)" | head -30
  echo
done
echo "### PROFILE DONE $(date -u +%FT%TZ)"
