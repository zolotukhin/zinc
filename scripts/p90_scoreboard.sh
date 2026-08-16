#!/usr/bin/env bash
# p90 prefill + decode scoreboard: ZINC (main) vs llama.cpp on ONE pinned GPU.
# One zinc process per run (long prompt -> prefill, then N_GEN decode), parse both.
# Report p90 + median over RUNS. llama-bench at the matched pp<T>/tg<N_GEN>.
# Line-buffered (stdbuf) so progress is visible live.
set -u
GPU_UUID=${GPU_UUID:-GPU-e59a6fce-1961-bafe-927c-06c0149f2370}   # 4090
export CUDA_VISIBLE_DEVICES=$GPU_UUID ZINC_GPU=$GPU_UUID
ZBIN=${ZBIN:-$HOME/zinc-main-meas/zig-out/bin/zinc}
MD=${MD:-$HOME/workspace/models}
LB=${LB:-$HOME/workspace/llama.cpp/build/bin/llama-bench}
RUNS=${RUNS:-5}
LLAMA_REPS=${LLAMA_REPS:-5}
N_GEN=${N_GEN:-160}
ONLY=${ONLY:-}

# Build a deterministic ~512-token neutral prompt (no external file).
SENT="The memory hierarchy of a modern graphics processor determines how quickly a large language model can process its prompt and generate new tokens during inference. "
PROMPT=""; for i in $(seq 1 40); do PROMPT="$PROMPT$SENT"; done

NAMES=(qwen35-9b qwen38-27b qwen36-35b-a3b gemma4-31b gemma4-26b)
declare -A GGUF=(
  [qwen35-9b]="$MD/Qwen3.5-9B-Q4_K_M.gguf"
  [qwen38-27b]="$MD/Qwen3.8-27B-Q4_K_M.gguf"
  [qwen36-35b-a3b]="$MD/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  [gemma4-31b]="$MD/gemma-4-31B-it-Q4_K_M.gguf"
  [gemma4-26b]="$MD/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
)
[ -x "$ZBIN" ] || { echo "no zinc binary $ZBIN"; exit 1; }

pctl() { local p=${1:-90}; sort -g | awk -v p="$p" '{a[++n]=$1} END{if(n==0){print"NA";exit} i=int((n-1)*p/100+0.5)+1; if(i>n)i=n; printf "%.1f",a[i]}'; }
median() { sort -g | awk '{a[++n]=$1} END{if(n==0){print"NA";exit} if(n%2)printf"%.1f",a[(n+1)/2]; else printf"%.1f",(a[n/2]+a[n/2+1])/2}'; }

echo "### p90 scoreboard GPU=$GPU_UUID RUNS=$RUNS N_GEN=$N_GEN $(date -u +%FT%TZ)"
printf '%-16s | %-22s | %-22s\n' "model" "ZINC prefill p90/med t/s" "ZINC decode p90/med t/s"
declare -A ZTOK
for name in "${NAMES[@]}"; do
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
  g="${GGUF[$name]}"; [ -f "$g" ] || { echo "$name MISSING $g"; continue; }
  pre=(); dec=(); ntok=""
  for r in $(seq 1 "$RUNS"); do
    out=$("$ZBIN" -m "$g" --prompt "$PROMPT" --raw -n "$N_GEN" 2>&1)
    y=$(printf '%s\n' "$out" | grep -oE 'Prefill complete: [0-9]+ tokens in [0-9.]+ ms \([0-9.]+ tok/s\)' | grep -oE '\([0-9.]+ tok' | grep -oE '[0-9.]+')
    t=$(printf '%s\n' "$out" | grep -oE 'Prefill complete: [0-9]+ tokens' | grep -oE '[0-9]+')
    z=$(printf '%s\n' "$out" | grep -oE 'Generated [0-9]+ tokens in [0-9.]+ ms . [0-9.]+ tok/s' | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+')
    [ -n "$y" ] && pre+=("$y"); [ -n "$z" ] && dec+=("$z"); [ -n "$t" ] && ntok="$t"
  done
  ZTOK[$name]="$ntok"
  printf '%-16s | %-22s | %-22s\n' "$name" \
    "$(printf '%s\n' "${pre[@]}"|pctl 90) / $(printf '%s\n' "${pre[@]}"|median) (T=$ntok)" \
    "$(printf '%s\n' "${dec[@]}"|pctl 90) / $(printf '%s\n' "${dec[@]}"|median)"
done

echo
echo "### llama-bench (same GPU+gguf, -r $LLAMA_REPS): pp<T> + tg$N_GEN (avg±std)"
for name in "${NAMES[@]}"; do
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
  g="${GGUF[$name]}"; [ -f "$g" ] || continue
  T="${ZTOK[$name]:-512}"
  "$LB" -m "$g" -ngl 99 -p "$T" -n "$N_GEN" -r "$LLAMA_REPS" 2>/dev/null | grep -E "pp|tg" | sed "s/^/$name /"
done
echo "### DONE $(date -u +%FT%TZ)"
