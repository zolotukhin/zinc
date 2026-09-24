#!/bin/bash
# Check ZINC against llama.cpp on the same GGUF before publishing numbers:
#   1. tokenizer parity: zinc-tokenize vs llama-tokenize --ids over a built-in
#      corpus (prose, code, numbers, accents, CJK, emoji, Cyrillic, URLs,
#      whitespace runs) plus any --prompt-file;
#   2. output parity: llama-server renders each chat prompt (thinking off), both
#      engines greedily continue the exact same text, and the first token and
#      the first line of the continuation are compared.
#
# ZINC comparing equal to itself proves nothing: a Gemma 4 tokenizer that
# merged by score and a gate/up kernel that skipped half of every Q4_K block
# both stayed "token-identical" run to run for months. Run this on the machine
# that holds the GPU, with no other server using it.
#
# usage: tools/llama_parity.sh --zinc-dir DIR --llama-dir DIR [--device N]
#            [--port P] [--prompt-file F]... [--tokenizer-only] MODEL.gguf...
#   --zinc-dir   ZINC install dir: bin/zinc, bin/zinc-tokenize (zig build
#                tokenize-tool), share/zinc/shaders for Vulkan builds
#   --llama-dir  llama.cpp bin dir: llama-server, llama-tokenize
#   --device     ZINC -d device index (default 0)
#   --llama-args extra llama-server flags, e.g. "--device Vulkan0 -ngl 99"
# Exit status: 0 when every check matched, 1 otherwise.
set -u

ZINC_DIR=""
LLAMA_DIR=""
DEVICE=0
PORT=${ZINC_PARITY_PORT:-19415}
LLAMA_ARGS="-ngl 99"
TOKENIZER_ONLY=0
PROMPT_FILES=()
MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --zinc-dir) ZINC_DIR=$2; shift 2 ;;
    --llama-dir) LLAMA_DIR=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --llama-args) LLAMA_ARGS=$2; shift 2 ;;
    --prompt-file) PROMPT_FILES+=("$2"); shift 2 ;;
    --tokenizer-only) TOKENIZER_ONLY=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done
if [ -z "$ZINC_DIR" ] || [ -z "$LLAMA_DIR" ] || [ ${#MODELS[@]} -eq 0 ]; then
  sed -n '2,26p' "$0"; exit 2
fi
for bin in "$ZINC_DIR/bin/zinc-tokenize" "$LLAMA_DIR/llama-tokenize"; do
  [ -x "$bin" ] || { echo "missing $bin" >&2; exit 2; }
done
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0

# --- Corpus -------------------------------------------------------------------
python3 - "$WORK" <<'PY'
import sys, os
texts = [
    "Explain how a refrigerator works.",
    "Write a long story about a lighthouse keeper.",
    "The quick brown fox jumps over the lazy dog.",
    "Internationalization and localization are often abbreviated i18n and l10n.",
    "Supercalifragilisticexpialidocious antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "for (int i = 0; i < 1024; ++i) { buf[i] = (uint8_t)(i * 31 ^ 0x5A); }",
    "   leading spaces, trailing spaces   ",
    "Multiple\n\n\nnewlines\n\nand\ttabs\t\there.",
    "Numbers: 3.14159, 2,718,281, 1e-9, 0xDEADBEEF, 42nd, 1st, 2026-09-23.",
    "Émile Zola écrivit « J'accuse…! » en 1898 — naïve café façade.",
    "日本語のテキストと中文文本以及한국어 텍스트.",
    "Emoji test: \U0001F680\U0001F525\U0001F44D\U0001F3FD \U0001F468‍\U0001F469‍\U0001F467‍\U0001F466 and ✓ ✗ ★.",
    "Привет, как дела? Это тест токенизатора.",
    "URLs like https://example.com/path?query=value&x=1#frag and emails like a.b@c.io",
    "The mitochondria is the powerhouse of the cell; photosynthesis converts light into chemical energy.",
    "Refrigerators, thermodynamics, compressors, evaporators, condensers, refrigerant.",
    "Don't, won't, can't, shouldn't've, y'all'd've, it's, O'Reilly.",
    "ALL CAPS SENTENCE WITH ACRONYMS LIKE NASA, GPU, RDNA4, and CUDA.",
    "mixedCaseIdentifiers snake_case_names kebab-case-names SCREAMING_SNAKE",
]
os.makedirs(os.path.join(sys.argv[1], "corpus"), exist_ok=True)
for i, t in enumerate(texts):
    open(os.path.join(sys.argv[1], "corpus", f"t{i:02d}.txt"), "w").write(t)
PY
i=0
for f in "${PROMPT_FILES[@]}"; do cp "$f" "$WORK/corpus/p$(printf %02d $i).txt"; i=$((i + 1)); done

CHAT_PROMPTS=("Write a long story about a lighthouse keeper." "Explain how a refrigerator works." "List three uses of copper.")
for f in "${PROMPT_FILES[@]}"; do CHAT_PROMPTS+=("$(cat "$f")"); done

for M in "${MODELS[@]}"; do
  name=$(basename "$M" .gguf)
  echo "== $name"

  # --- 1. Tokenizer parity ------------------------------------------------------
  pass=0; bad=0
  for f in "$WORK"/corpus/*.txt; do
    z=$("$ZINC_DIR/bin/zinc-tokenize" "$M" "$f" 2>&1 | tail -1 | tr -d ' ')
    l=$("$LLAMA_DIR/llama-tokenize" -m "$M" -f "$f" --ids --log-disable 2>/dev/null | tail -1 | tr -d ' ')
    if [ "$z" = "$l" ]; then pass=$((pass + 1)); else bad=$((bad + 1)); echo "  tokenizer DIFF $(basename "$f")"; echo "    zinc  ${z:0:200}"; echo "    llama ${l:0:200}"; fi
  done
  echo "  tokenizer: $pass match, $bad differ"
  [ $bad -eq 0 ] || fail=1
  [ $TOKENIZER_ONLY -eq 1 ] && continue

  # --- 2. Output parity -----------------------------------------------------------
  # shellcheck disable=SC2086
  "$LLAMA_DIR/llama-server" -m "$M" $LLAMA_ARGS --host 127.0.0.1 --port "$PORT" -c 4096 > "$WORK/llama.log" 2>&1 < /dev/null &
  lp=$!
  for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  n=0
  for pr in "${CHAT_PROMPTS[@]}"; do
    python3 -c "import json,sys; print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'chat_template_kwargs':{'enable_thinking':False}}))" "$pr" > "$WORK/req.json"
    curl -s "http://127.0.0.1:$PORT/apply-template" -H 'Content-Type: application/json' -d @"$WORK/req.json" | jq -j .prompt > "$WORK/prompt$n.txt"
    python3 -c "import json,sys; print(json.dumps({'prompt':open(sys.argv[1]).read(),'n_predict':16,'temperature':0,'n_probs':2,'cache_prompt':False}))" "$WORK/prompt$n.txt" > "$WORK/comp.json"
    curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @"$WORK/comp.json" > "$WORK/llama$n.json"
    n=$((n + 1))
  done
  kill "$lp" 2>/dev/null; wait "$lp" 2>/dev/null
  sleep 3

  for ((k = 0; k < n; k++)); do
    PT="$(cat "$WORK/prompt$k.txt"; printf x)"; PT="${PT%x}"
    out=$(ZINC_LOG_TOPK=1 ZINC_SHADER_DIR="$ZINC_DIR/share/zinc/shaders" "$ZINC_DIR/bin/zinc" -m "$M" -d "$DEVICE" --prompt "$PT" -n 16 2>&1)
    ztok=$(printf '%s\n' "$out" | grep -m1 "topk:" | sed -E 's/.*top=([0-9]+).*/\1/')
    ztxt=$(printf '%s\n' "$out" | grep -m1 -E "Output text:|Output \([0-9]+ tokens\):" | sed -E 's/.*Output text: //; s/.*Output \([0-9]+ tokens\): //')
    ltok=$(jq -r '.completion_probabilities[0].id // empty' "$WORK/llama$k.json")
    ltxt=$(jq -r '.content' "$WORK/llama$k.json" | head -1)
    zc=$(printf '%s' "$ztxt" | tr -d ' '); lc=$(printf '%s' "$ltxt" | tr -d ' ')
    # ZINC logs its text up to the first newline and llama's first line is
    # compared, so either may be a prefix of the other.
    prefix_match=0
    if [ -n "$zc" ] && [ -n "$lc" ]; then
      if [ "${lc#"$zc"}" != "$lc" ] || [ "${zc#"$lc"}" != "$zc" ]; then prefix_match=1; fi
    fi
    if [ -n "$ztok" ] && [ "$ztok" != "$ltok" ]; then
      verdict="FIRST-TOKEN DIFF (zinc $ztok, llama $ltok)"; fail=1
    elif [ $prefix_match -eq 0 ]; then
      verdict="TEXT DIFF"; fail=1
    else
      verdict="match"
    fi
    echo "  prompt $k: $verdict"
    echo "    llama: ${ltxt:0:80}"
    echo "    zinc : ${ztxt:0:80}"
  done
done
exit $fail
