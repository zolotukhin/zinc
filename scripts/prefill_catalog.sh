#!/usr/bin/env bash
# Catalog PREFILL-perf + head-skip correctness across the NVIDIA node's GPUs.
#
# Production prefill builds the KV cache for every prompt token, but only the
# LAST token needs logits (its argmax seeds generation). `prefillStep` skips
# the vocab-sized LM head on the T-1 prompt-internal tokens — pure waste
# otherwise, and a larger share for the big-vocab / small-active-forward models.
# This harness runs each catalog model over a long synthetic prompt as a
# counterbalanced baseline (`decodeStep` every token) vs head-skip
# (ZINC_PREFILL_SKIP=1 -> `prefillStep`) A/B, reports each prefill tok/s, and
# ASSERTS the generated tokens are byte-identical. The skip is output-preserving
# by construction, so the GEN_IDS match IS the merge gate; a mismatch is a bug.
#
# IMPORTANT — counterbalancing: each model is run as ABBA over ZINC_ROUNDS pairs
# (B S S B by default), so the head-skip path is NOT always 2nd. Without this,
# run-order thermal/boost drift (worse on the slow 26-31B loads) masquerades as
# a regression — a real bug found 2026-06-12 when a naive B-then-S harness
# showed gemma "-11%" for a strictly-fewer-ops change.
#
# Companion to scripts/perf_catalog.sh (decode). Run on the box from an isolated
# checkout. GPU selectors are supplied through the environment; do not commit
# machine-specific UUIDs.
# `dbg_cuda gen` prints "PREFILL: T tokens in Xs = Y tok/s" and "GEN_IDS:...".
#
# Env overrides:
#   ZINC_GPUS    space list of targets   (default "4090")
#   ZINC_GPU_4090 CUDA_VISIBLE_DEVICES value for the 4090 target
#   ZINC_GPU_5090 CUDA_VISIBLE_DEVICES value for the 5090 target
#   ZINC_MODELS  catalog gguf dir        (default ~/workspace/models)
#   ZINC_PROMPT  synthetic prompt length (default 250)
#   ZINC_NGEN    gen tokens for GEN_IDS  (default 8)
#   ZINC_ROUNDS  ABBA pairs per model    (default 2 -> 4 runs: B S S B)
#   ZINC_ONLY    space list of names     (default all)
#   ZINC_AB      which path to A/B vs baseline: headskip|batched (default headskip)
#                  headskip -> ZINC_PREFILL_SKIP=1 (the merged LM-head-skip win)
#                  batched  -> ZINC_BATCHED_PREFILL=1 (Effort 24 batched-GEMM
#                              prefill; gemma dense only — qwen/MoE fall back to
#                              per-token so the "S" path == baseline there)
#   ZIG          zig binary              (default ~/zig-0.15.2/zig)
set -u

declare -A GPU_UUID=(
  [5090]="${ZINC_GPU_5090:-}"
  [4090]="${ZINC_GPU_4090:-}"
)
GPUS=${ZINC_GPUS:-"4090"}
ZIG=${ZIG:-$HOME/zig-0.15.2/zig}
MD=${ZINC_MODELS:-$HOME/workspace/models}
PROMPT_LEN=${ZINC_PROMPT:-250}
NGEN=${ZINC_NGEN:-8}
ROUNDS=${ZINC_ROUNDS:-2}
ONLY=${ZINC_ONLY:-}
MODE=${ZINC_AB:-headskip}
case "$MODE" in
  headskip) S_ENV="ZINC_BATCHED_PREFILL=0 ZINC_PREFILL_SKIP=1"; S_LABEL="headskip" ;;
  batched)  S_ENV="ZINC_BATCHED_PREFILL=1"; S_LABEL="batched"
            # Cycle 11: ZINC_BATCHED_TC=1 also routes dense Q4_K GEMMs through the
            # fp16 tensor-core kernel (NOT byte-identical → expect GEN_IDS may
            # differ; use validate_catalog as its gate, this is for perf only).
            [ "${ZINC_BATCHED_TC:-0}" = "1" ] && { S_ENV="$S_ENV ZINC_BATCHED_TC=1"; S_LABEL="batched+tc"; }
            # Cycle 17: opt into the wider 128x64 M-tile low-shared Q4_K TC kernel.
            [ "${ZINC_BATCHED_TC_M128_LOWSMEM:-0}" = "1" ] && S_ENV="$S_ENV ZINC_BATCHED_TC_M128_LOWSMEM=1"
            # Cycle 18: opt into token-GROUPED routed experts (byte-identical → GEN_IDS
            # must still match; this knob is for measuring the L2-reuse perf delta).
            [ "${ZINC_BATCHED_EXPERTS_GROUPED:-0}" = "1" ] && { S_ENV="$S_ENV ZINC_BATCHED_EXPERTS_GROUPED=1"; S_LABEL="$S_LABEL+grouped"; }
            # Cycle 19: share one f32→f16 activation recast across same-input GEMMs
            # (byte-identical → GEN_IDS must still match; this knob measures the win).
            [ "${ZINC_BATCHED_TC_SHAREA:-0}" = "1" ] && { S_ENV="$S_ENV ZINC_BATCHED_TC_SHAREA=1"; S_LABEL="$S_LABEL+sharea"; }
            # Cycle 21: norm/GeGLU producers emit fp16 directly into act_f16, dropping
            # the per-GEMM recast (byte-identical → GEN_IDS must still match; perf knob).
            [ "${ZINC_BATCHED_TC_NORMF16:-0}" = "1" ] && { S_ENV="$S_ENV ZINC_BATCHED_TC_NORMF16=1"; S_LABEL="$S_LABEL+normf16"; } ;;
  *) echo "unknown ZINC_AB '$MODE' (want headskip|batched)"; exit 1 ;;
esac
DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$DIR" || { echo "no repo dir $DIR"; exit 1; }

NAMES=(qwen35-9b      qwen38-27b      qwen36-35b-a3b  gemma4-31b      gemma4-26b)
PATHS=(
  "$HOME/workspace/Qwen3.5-9B-Q4_K_M.gguf"
  "$MD/Qwen3.8-27B-Q4_K_M.gguf"
  "$MD/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  "$MD/gemma-4-31B-it-Q4_K_M.gguf"
  "$MD/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)

echo "building cuda-dbg (one build runs on every GPU)..."
"$ZIG" build cuda-dbg -Dbackend=cuda -Dshaders=false >/tmp/pf.build 2>&1 \
  || { echo "BUILD FAIL"; grep -E 'error:' /tmp/pf.build | head; exit 1; }
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary"; exit 1; }
echo "  binary: $ZBIN  (ABBA x${ROUNDS} per model)"
PROMPT=$(seq -s, 1 "$PROMPT_LEN")

pf_of() { sed -E 's/.* = ([0-9.]+) tok.*/\1/' <<<"$1"; }
run_one() { # $1 = B|S -> echoes "<tok/s>|<GEN_IDS line>"
  local o
  # Effort 25: batched prefill is now the DEFAULT for gemma, so the baseline arm
  # must explicitly opt OUT (ZINC_BATCHED_PREFILL=0) to be the true per-token path
  # — otherwise "B" would also run batched and the A/B would show ~0% gain.
  if [ "$1" = "B" ]; then o=$(env ZINC_BATCHED_PREFILL=0 timeout 600 "$ZBIN" gen "$PROMPT" "$NGEN" "$2" 2>&1)
  else o=$(env $S_ENV timeout 600 "$ZBIN" gen "$PROMPT" "$NGEN" "$2" 2>&1); fi
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

fails=0
for gpu in $GPUS; do
  uuid="${GPU_UUID[$gpu]:-}"; [ -n "$uuid" ] || { echo "missing selector for GPU '$gpu'; set ZINC_GPU_${gpu}=<cuda-visible-device>"; continue; }
  export CUDA_VISIBLE_DEVICES="$uuid"
  printf '\n=== RTX %s prefill tok/s  |  %s-tok prompt, %s vs baseline (ABBA x%s) ===\n' "$gpu" "$PROMPT_LEN" "$S_LABEL" "$ROUNDS"
  printf '  %-15s %10s %10s %7s   %s\n' "model" "baseline" "$S_LABEL" "gain" "correctness"
  for i in "${!NAMES[@]}"; do
    nm="${NAMES[$i]}"; m="${PATHS[$i]}"
    [ -n "$ONLY" ] && ! grep -qw "$nm" <<<"$ONLY" && continue
    [ -f "$m" ] || { printf '  %-15s   MISSING (%s)\n' "$nm" "$m"; continue; }
    bsum=0; bn=0; ssum=0; sn=0; bg=""; sg=""
    for r in $(seq 1 "$ROUNDS"); do
      if (( r % 2 == 1 )); then order="B S"; else order="S B"; fi
      for w in $order; do
        res=$(run_one "$w" "$m"); v=${res%%|*}; g=${res#*|}
        if [ "$w" = "B" ]; then bsum=$(awk -v a="$bsum" -v b="${v:-0}" 'BEGIN{print a+b}'); bn=$((bn+1)); [ -z "$bg" ] && bg="$g"
        else ssum=$(awk -v a="$ssum" -v b="${v:-0}" 'BEGIN{print a+b}'); sn=$((sn+1)); [ -z "$sg" ] && sg="$g"; fi
      done
    done
    bpf=$(awk -v s="$bsum" -v n="$bn" 'BEGIN{ if(n>0) printf "%.2f", s/n }')
    spf=$(awk -v s="$ssum" -v n="$sn" 'BEGIN{ if(n>0) printf "%.2f", s/n }')
    gain=$(awk -v b="${bpf:-0}" -v s="${spf:-0}" 'BEGIN{ if(b>0) printf "+%.0f%%", (s/b-1)*100; else print "-" }')
    if [ -n "$bg" ] && [ "$bg" = "$sg" ]; then ok="PASS (identical)"; else ok="*** FAIL: tokens differ ***"; fails=$((fails+1)); fi
    printf '  %-15s %10s %10s %7s   %s\n' "$nm" "${bpf:--}" "${spf:--}" "$gain" "$ok"
  done
done
echo ""
[ "$fails" -eq 0 ] && echo "=== prefill_catalog: ALL PASS ($S_LABEL output-preserving) ===" \
                   || echo "=== prefill_catalog: $fails FAILURE(S) — DO NOT MERGE ==="
exit "$fails"
