#!/usr/bin/env bash
# Cycle 20 diagnostic: how does the fp16 tensor-core dense GEMM path
# (ZINC_BATCHED_TC, cycles 11-19) scale with prompt length T? Cycle 11 measured
# it at ONLY T=250 and concluded "+6%, memory-bound" — but at T=250 the dense
# Q4_K GEMM is memory/launch-bound, so the fp16 tensor cores barely help. As T
# grows the GEMM becomes COMPUTE-bound (many T-tiles reuse each single weight
# read), and the tensor cores finally pay off. This sweeps T and runs a clean
# f32-vs-TC ABBA (both arms ZINC_BATCHED_PREFILL=1; only ZINC_BATCHED_TC differs)
# on the DENSE gemma-31b (the clean signal — TC covers ~all dense GEMMs; on the
# MoE model TC touches only the smaller dense-attention GEMMs and the MoE FFN +
# boost noise swamp it). TC is fp16 → token-correct within tolerance (validated
# 5/5 by validate_catalog with ZINC_BATCHED_TC=1), NOT byte-identical to f32, so
# this REPORTS whether the TC GEN matched f32's (informational) rather than
# hard-asserting identity. ABBA-counterbalanced so boost drift doesn't masquerade
# as a delta.
#
# Cycle-20 result (4090): gemma-31b dense, f32 batched GEMM vs fp16 TC GEMM
#   T=750   f32 138.1  TC 178.8  = +29.5%  (ABBA x2, ranges non-overlapping)
#   T=1500  f32 143.9  TC 171.7  = +19.3%  (ABBA x2, ranges non-overlapping)
#   T=250   f32 134.2  TC 166.3  = +23.9%  (ABBA x1)
# => The TC dense path is a robust ~+20-30% prefill win across T=250-1500 — NOT
#    the "+6%, memory-bound" the cycle-11 log records. That +6% measured the
#    ORIGINAL m64 TC kernel; cycles 12 (fp16-A) + 15 (lowsmem, +9-12%) improved
#    the default TC path afterward but the headline was never re-measured. The
#    win is therefore much larger than logged and roughly T-stable on dense.
#    (Kept opt-in: fp16 breaks the strict byte-identity merge gate; the f32 arm
#    is the merge candidate, TC is the recommended fast prefill path.)
#
# Env: ZINC_TS (space list of prompt lengths, default "250 750 1500"),
#      ZINC_ROUNDS (ABBA pairs, default 2), ZINC_GPU (4090 role; requires ZINC_GPU_4090),
#      ZINC_MODEL (gguf, default gemma-4-31B dense — the clean beneficiary),
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
# A = f32 batched GEMM (byte-identical merge path), B = fp16 tensor-core GEMM
run_one() { # $1=A|B  $2=prompt -> "<tok/s>|<GEN_IDS>"
  local env_extra="ZINC_BATCHED_PREFILL=1"
  [ "$1" = "B" ] && env_extra="$env_extra ZINC_BATCHED_TC=1"
  local o; o=$(env $env_extra timeout 900 "$ZBIN" gen "$2" "$NGEN" "$MODEL" 2>&1)
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

printf '\n  %-6s %12s %12s %8s   %s\n' "T" "f32" "TC" "gain" "tok-match"
for T in $TS; do
  # varied non-collapsing prompt (stresses byte-identity / routing across tokens)
  PROMPT=$(awk -v n="$T" 'BEGIN{for(i=0;i<n;i++){printf "%s%d",(i?",":""),((i*73+11)%251)+5}}')
  declare -a AV=() BV=(); ag=""; bg=""
  for ((r=0;r<ROUNDS;r++)); do
    for arm in A B B A; do
      res=$(run_one "$arm" "$PROMPT"); v=${res%%|*}; g=${res#*|}
      if [ "$arm" = A ]; then AV+=("$v"); [ -z "$ag" ] && ag="$g"
      else BV+=("$v"); [ -z "$bg" ] && bg="$g"; fi
    done
  done
  mean() { awk 'BEGIN{s=0;n=0} {for(i=1;i<=NF;i++){s+=$i;n++}} END{if(n>0)printf "%.2f",s/n}' <<<"$*"; }
  am=$(mean "${AV[@]}"); bm=$(mean "${BV[@]}")
  gain=$(awk -v a="${am:-0}" -v b="${bm:-0}" 'BEGIN{if(a>0)printf "%+.1f%%",(b/a-1)*100; else print "-"}')
  if [ -n "$ag" ] && [ "$ag" = "$bg" ]; then tm="yes (no divergence)"; else tm="DIVERGED (fp16 tol)"; fi
  printf '  %-6s %12s %12s %8s   %s\n' "$T" "${am:--}" "${bm:--}" "$gain" "$tm"
done
echo ""
echo "note: TC is fp16 (token-correct within tolerance, validate_catalog 5/5), NOT"
echo "      byte-identical to f32 — kept opt-in behind ZINC_BATCHED_TC. The f32 arm"
echo "      is the strict byte-identity merge path; TC is the faster prefill path"
echo "      recommended for realistic (long) prompts."
