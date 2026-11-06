#!/usr/bin/env bash
# Cycle 19 diagnostic: does the cycle-18 token-GROUPED routed-expert path
# (ZINC_BATCHED_EXPERTS_GROUPED) beat the default cycle-8 _batched matvecs at
# LARGER T? Cycle 18 found it in-noise at T=250 and hypothesized the L2-reuse
# win only emerges above the boost floor at T>>250 (more (token,slot) work-items
# per expert => longer L2 residency). This sweeps T and runs a CLEAN grouped-vs-
# batched ABBA (BOTH arms ZINC_BATCHED_PREFILL=1; only the grouped toggle differs)
# on gemma-26b MoE, asserting GEN_IDS byte-identical (the cycle-18 correctness
# claim) and reporting prefill tok/s per arm.
#
# Env: ZINC_TS (space list of prompt lengths, default "250 750 1500"),
#      ZINC_ROUNDS (ABBA pairs, default 2), ZINC_GPU (4090 role; requires ZINC_GPU_4090),
#      ZINC_MODEL (gguf, default gemma-4-26B), ZINC_NGEN (default 8).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib/cuda_target.sh"
: "${ZINC_GPU:=4090}"
zinc_select_cuda_device || exit 1
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_MODEL:-$MD/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}
TS=${ZINC_TS:-"250 750 1500"}
ROUNDS=${ZINC_ROUNDS:-2}
NGEN=${ZINC_NGEN:-8}
DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$DIR"
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary (build first)"; exit 1; }
echo "binary: $ZBIN   model: $(basename "$MODEL")   GPU: configured CUDA device   ABBA x$ROUNDS"

pf_of() { sed -E 's/.* = ([0-9.]+) tok.*/\1/' <<<"$1"; }
# A = batched (default), B = grouped
run_one() { # $1=A|B  $2=prompt -> "<tok/s>|<GEN_IDS>"
  local env_extra="ZINC_BATCHED_PREFILL=1"
  [ "$1" = "B" ] && env_extra="$env_extra ZINC_BATCHED_EXPERTS_GROUPED=1"
  local o; o=$(env $env_extra timeout 900 "$ZBIN" gen "$2" "$NGEN" "$MODEL" 2>&1)
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

printf '\n  %-6s %12s %12s %8s   %s\n' "T" "batched" "grouped" "gain" "correctness"
fails=0
for T in $TS; do
  # varied non-collapsing prompt (also stresses byte-identity across routing)
  PROMPT=$(awk -v n="$T" 'BEGIN{for(i=0;i<n;i++){printf "%s%d",(i?",":""),((i*73+11)%251)+5}}')
  asum=0; an=0; bsum=0; bn=0; ag=""; bg=""
  for r in $(seq 1 "$ROUNDS"); do
    if (( r % 2 == 1 )); then order="A B"; else order="B A"; fi
    for w in $order; do
      res=$(run_one "$w" "$PROMPT"); v=${res%%|*}; g=${res#*|}
      if [ "$w" = "A" ]; then asum=$(awk -v a="$asum" -v b="${v:-0}" 'BEGIN{print a+b}'); an=$((an+1)); [ -z "$ag" ] && ag="$g"
      else bsum=$(awk -v a="$bsum" -v b="${v:-0}" 'BEGIN{print a+b}'); bn=$((bn+1)); [ -z "$bg" ] && bg="$g"; fi
    done
  done
  apf=$(awk -v s="$asum" -v n="$an" 'BEGIN{if(n>0)printf "%.2f",s/n}')
  bpf=$(awk -v s="$bsum" -v n="$bn" 'BEGIN{if(n>0)printf "%.2f",s/n}')
  gain=$(awk -v a="${apf:-0}" -v b="${bpf:-0}" 'BEGIN{if(a>0)printf "%+.1f%%",(b/a-1)*100; else print "-"}')
  if [ -n "$ag" ] && [ "$ag" = "$bg" ]; then ok="PASS (identical)"; else ok="*** FAIL: GEN differ ***"; fails=$((fails+1)); fi
  printf '  %-6s %12s %12s %8s   %s\n' "$T" "${apf:--}" "${bpf:--}" "$gain" "$ok"
done
echo ""
[ "$fails" -eq 0 ] && echo "=== grouped_sweep: ALL byte-identical ===" || echo "=== grouped_sweep: $fails FAIL ==="
exit "$fails"
