#!/usr/bin/env bash
# Cycle 19 diagnostic: does sharing ONE f32→f16 activation recast across the
# same-input GEMMs (ZINC_BATCHED_TC_SHAREA — attn Q/K/V from b.norm, FFN gate/up
# from b.ffn_norm) beat the per-GEMM recast on the TC path? Both arms run
# ZINC_BATCHED_PREFILL=1 ZINC_BATCHED_TC=1 (so the TC f16-A path is active and
# the recast EXISTS to share); only ZINC_BATCHED_TC_SHAREA differs. Asserts
# GEN_IDS byte-identical (the shared-A correctness claim — same __float2half
# bits, act_f16 reused stream-ordered) and reports prefill tok/s per arm.
# ABBA-counterbalanced so boost drift doesn't masquerade as a delta.
#
# Env: ZINC_TS (space list of prompt lengths, default "250"),
#      ZINC_ROUNDS (ABBA pairs, default 2), ZINC_GPU (4090 role; requires ZINC_GPU_4090),
#      ZINC_MODEL (gguf, default gemma-4-31B dense — the main beneficiary),
#      ZINC_NGEN (default 8).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib/cuda_target.sh"
: "${ZINC_GPU:=4090}"
zinc_select_cuda_device || exit 1
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_MODEL:-$MD/gemma-4-31B-it-Q4_K_M.gguf}
TS=${ZINC_TS:-"250"}
ROUNDS=${ZINC_ROUNDS:-2}
NGEN=${ZINC_NGEN:-8}
DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$DIR"
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary (build first)"; exit 1; }
echo "binary: $ZBIN   model: $(basename "$MODEL")   GPU: configured CUDA device   ABBA x$ROUNDS"

pf_of() { sed -E 's/.* = ([0-9.]+) tok.*/\1/' <<<"$1"; }
# A = TC no-sharea (per-GEMM recast), B = TC sharea (shared recast)
run_one() { # $1=A|B  $2=prompt -> "<tok/s>|<GEN_IDS>"
  local env_extra="ZINC_BATCHED_PREFILL=1 ZINC_BATCHED_TC=1"
  [ "$1" = "B" ] && env_extra="$env_extra ZINC_BATCHED_TC_SHAREA=1"
  local o; o=$(env $env_extra timeout 900 "$ZBIN" gen "$2" "$NGEN" "$MODEL" 2>&1)
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

printf '\n  %-6s %12s %12s %8s   %s\n' "T" "no-sharea" "sharea" "gain" "correctness"
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
