#!/usr/bin/env bash
# Effort 28 increment 3 (3b) — CUDA multi-tenant HTTP/SSE serving gate.
#
# Proves the new `zinc` CUDA server (one GPU worker thread + per-connection
# SSE handlers; src/server/cuda_serve.zig + runCudaServe in main.zig) delivers
# each concurrent request its OWN token stream, token-identical to the isolated
# single-sequence path. Three comparisons, all token-level exact (the server runs
# in ZINC_SERVE_DEBUG_IDS=1 mode: prompts are raw comma-separated token ids and
# each SSE event payload is the generated token id):
#
#   GATE A  concurrent-N streams == the same prompts run one-at-a-time (B=1)
#           through the server  → transport + concurrency isolation + slot reuse.
#   GATE B  server B=1 streams == `dbg_cuda serve` SERVE_SEQ streams (already
#           gated == isolated decodeStep) → the HTTP engine == the proven engine.
#
# Combined with scripts/validate_catalog.sh (decodeStep == llama.cpp) the chain
# is: concurrent HTTP == isolated == llama-correct.
#
# Run ON the box from the repo root. 5090-pinned by default. nohup + poll a log.
set -u
GPU=${ZINC_GPU:?Set ZINC_GPU to the target device}
ZIG=${ZIG:-./scripts/zig.sh}
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_SERVE_MODEL:-$MD/gemma-4-31B-it-Q4_K_M.gguf}
PORT=${ZINC_SERVE_PORT:-8189}
# Local loopback host to reach the server. The <worker> box is net-fenced to
# 127.99/16, so plain 127.0.0.1 is dropped — use a 127.99.x address (the server
# binds 0.0.0.0 so any local IP reaches it). Override for an unfenced box.
HOST=${ZINC_SERVE_HOST:-127.99.0.1}
NG=${ZINC_SERVE_NG:-10}
NSLOTS=${ZINC_SERVE_NSLOTS:-2}
CTX=${ZINC_SERVE_CTX:-1024}
# Mixed-length raw-token-id prompts; '|' separates sequences, ',' within.
SEQS=${ZINC_SERVE_SEQS:-"651,2134,573,1496|1024,53100,108|7611,1492,573,2134,235336|236774,236775,236776"}
# Budget-only stop by default (no token equals u32-max) so both server and the
# reference run the full NG; override to a real mid-stream token to also exercise
# EOS eviction under concurrency.
EOS=${ZINC_SCHED_EOS:-4294967295}
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-/tmp/e28gc}
T=/tmp/zinc_serve_gate; mkdir -p "$T"; rm -f "$T"/*.out "$T"/*.log 2>/dev/null
IFS='|' read -ra PROMPTS <<< "$SEQS"
N=${#PROMPTS[@]}

echo "=== build zinc (main server binary) ==="
$ZIG build -Dbackend=cuda -Dshaders=false >"$T/build_zinc.log" 2>&1 || { echo "ZINC BUILD FAIL"; grep -E 'error:' "$T/build_zinc.log" | head; exit 1; }
ZINC=$(ls -t zig-out/bin/zinc 2>/dev/null | head -1)
[ -x "$ZINC" ] || { echo "no zinc binary"; exit 1; }
echo "zinc md5: $(md5sum "$ZINC" | cut -d' ' -f1)"

echo "=== build cuda-dbg (reference) ==="
$ZIG build cuda-dbg -Dbackend=cuda -Dshaders=false >"$T/build_dbg.log" 2>&1 || true
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary"; exit 1; }
echo "cuda-dbg md5: $(md5sum "$ZBIN" | cut -d' ' -f1)"

# --- helper: read one SSE stream, emit comma-joined token ids -----------------
post_ids() { # $1=prompt-ids  -> stdout: "id,id,id"
  curl -N -s --max-time 120 -X POST "http://$HOST:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$1\",\"max_tokens\":$NG}" \
  | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' | paste -sd, -
}

# --- start the server ---------------------------------------------------------
echo "=== start server (port $PORT, slots $NSLOTS, ctx $CTX, eos $EOS) ==="
CUDA_VISIBLE_DEVICES=$GPU ZINC_GPU=$GPU ZINC_SERVE_DEBUG_IDS=1 ZINC_SCHED_EOS=$EOS \
  "$ZINC" -m "$MODEL" -p "$PORT" -c "$CTX" --parallel "$NSLOTS" -n "$NG" \
  >"$T/server.log" 2>&1 &
SRV=$!
trap '[ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT
# wait for listen (gemma-31b uploads ~17GB + nvrtc compile → can exceed 2 min)
ready=0
for _ in $(seq 1 360); do
  if curl -s --max-time 3 "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "server died during startup:"; tail -20 "$T/server.log"; exit 1; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "server not ready in time:"; tail -20 "$T/server.log"; exit 1; }
echo "server ready (pid $SRV)"

# --- sequential B=1 references (one request in flight at a time) ---------------
echo "=== sequential B=1 (isolated through the server) ==="
declare -a SEQOUT
for j in "${!PROMPTS[@]}"; do
  SEQOUT[$j]=$(post_ids "${PROMPTS[$j]}")
  echo "  SEQ$j: ${SEQOUT[$j]}"
done

# --- concurrent N (all requests in flight at once) ----------------------------
# wait ONLY on the curl jobs, not the long-lived server background job ($SRV).
echo "=== concurrent N=$N (real multi-tenant) ==="
cpids=()
for j in "${!PROMPTS[@]}"; do
  ( post_ids "${PROMPTS[$j]}" >"$T/c$j.out" ) &
  cpids+=($!)
done
wait "${cpids[@]}"
declare -a CONOUT
for j in "${!PROMPTS[@]}"; do
  CONOUT[$j]=$(cat "$T/c$j.out")
  echo "  CONC$j: ${CONOUT[$j]}"
done

# --- dbg_cuda serve reference (proven == isolated decodeStep) ------------------
echo "=== dbg_cuda serve reference ==="
CUDA_VISIBLE_DEVICES=$GPU ZINC_GPU=$GPU ZINC_SCHED_EOS=$EOS \
  "$ZBIN" serve "$SEQS" "$NG" "$NSLOTS" "$MODEL" >"$T/dbgserve.log" 2>&1 || true
declare -a REFOUT
for j in "${!PROMPTS[@]}"; do
  REFOUT[$j]=$(sed -n "s/^SERVE_SEQ$j(.*)://p" "$T/dbgserve.log" | awk '{print $1}')
  echo "  REF$j:  ${REFOUT[$j]}"
done
grep -E 'SERVE_GATE' "$T/dbgserve.log" || echo "  (no SERVE_GATE line — see $T/dbgserve.log)"

# --- gates --------------------------------------------------------------------
passA=1; passB=1
for j in "${!PROMPTS[@]}"; do
  [ -n "${CONOUT[$j]}" ] && [ "${CONOUT[$j]}" = "${SEQOUT[$j]}" ] || { passA=0; echo "  GATE A DIFF seq$j: conc='${CONOUT[$j]}' seq='${SEQOUT[$j]}'"; }
  [ -n "${REFOUT[$j]}" ] && [ "${SEQOUT[$j]}" = "${REFOUT[$j]}" ] || { passB=0; echo "  GATE B DIFF seq$j: server='${SEQOUT[$j]}' ref='${REFOUT[$j]}'"; }
done
echo "=== SERVE_HTTP_GATE_A:$([ $passA = 1 ] && echo PASS || echo FAIL) (concurrent==sequential, N=$N nslots=$NSLOTS) ==="
echo "=== SERVE_HTTP_GATE_B:$([ $passB = 1 ] && echo PASS || echo FAIL) (server-B1==dbg-serve-reference) ==="
[ $passA = 1 ] && [ $passB = 1 ] && echo "SERVE_HTTP_GATE:PASS" || { echo "SERVE_HTTP_GATE:FAIL"; exit 1; }
