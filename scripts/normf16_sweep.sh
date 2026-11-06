#!/usr/bin/env bash
# Cycle 21 diagnostic: does emitting the norm/GeGLU activations as fp16 DIRECTLY
# into act_f16 (ZINC_BATCHED_TC_NORMF16 — rms_norm_f16 for the attn/FFN pre-norms,
# geglu_f16 for ffn_down) — dropping the per-GEMM f32→fp16 recast launch ENTIRELY
# across the layer — beat the per-GEMM recast on the TC path? Both arms run
# ZINC_BATCHED_PREFILL=1 ZINC_BATCHED_TC=1 (TC f16-A path active, so the recast
# EXISTS to remove); only ZINC_BATCHED_TC_NORMF16 differs. Asserts GEN_IDS
# byte-identical (the normf16 correctness claim — the producer __float2half's the
# SAME f32 value f32_to_f16 would, so act_f16 bits are identical) and reports
# prefill tok/s per arm. Swept across T because the recast scales with T·K, so a
# sub-floor win at T=250 (cf. cycle 19 shared-A +2.6%) may clear the floor at
# T=750/1500. ABBA-counterbalanced so boost drift isn't read as a delta.
#
# Env: ZINC_TS (space list of prompt lengths, default "250 750 1500"),
#      ZINC_ROUNDS (ABBA pairs, default 2), ZINC_GPU (4090 role; requires ZINC_GPU_4090),
#      ZINC_MODEL (gguf, default gemma-4-31B dense — the clean TC signal),
#      ZINC_NGEN (default 8).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib/cuda_target.sh"
: "${ZINC_GPU:=4090}"
zinc_select_cuda_device || exit 1
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_MODEL:-$MD/gemma-4-31B-it-Q4_K_M.gguf}
TS=${ZINC_TS:-"250 750 1500"}
ROUNDS=${ZINC_ROUNDS:-2}
NGEN=${ZINC_NGEN:-8}
DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$DIR"
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary (build first)"; exit 1; }
echo "binary: $ZBIN   model: $(basename "$MODEL")   GPU: configured CUDA device   ABBA x$ROUNDS"

pf_of() { sed -E 's/.* = ([0-9.]+) tok.*/\1/' <<<"$1"; }
# A = TC per-GEMM recast (no normf16), B = TC normf16 (producers emit fp16)
run_one() { # $1=A|B  $2=prompt -> "<tok/s>|<GEN_IDS>"
  local env_extra="ZINC_BATCHED_PREFILL=1 ZINC_BATCHED_TC=1"
  [ "$1" = "B" ] && env_extra="$env_extra ZINC_BATCHED_TC_NORMF16=1"
  local o; o=$(env $env_extra timeout 900 "$ZBIN" gen "$2" "$NGEN" "$MODEL" 2>&1)
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

printf '\n  %-6s %12s %12s %8s   %s\n' "T" "no-normf16" "normf16" "gain" "correctness"
for T in $TS; do
  PROMPT=$(seq -s, 1 "$T")
  declare -a AV=() BV=(); g0=""; ok="identical"
  for ((r=0;r<ROUNDS;r++)); do
    for arm in A B B A; do
      res=$(run_one "$arm" "$PROMPT"); v=${res%%|*}; g=${res#*|}
      [ -z "$g0" ] && g0="$g"; [ "$g" != "$g0" ] && ok="MISMATCH"
      [ "$arm" = A ] && AV+=("$v") || BV+=("$v")
    done
  done
  mean() { local s=0 n=0; for x in "$@"; do s=$(echo "$s+$x"|bc -l); n=$((n+1)); done; echo "scale=2;$s/$n"|bc -l; }
  am=$(mean "${AV[@]}"); bm=$(mean "${BV[@]}")
  gain=$(echo "scale=1;($bm-$am)/$am*100"|bc -l)
  printf '  %-6s %12s %12s %7s%%   %s\n' "$T" "$am" "$bm" "$gain" "$ok"
done
