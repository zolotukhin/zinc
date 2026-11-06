# Effort 24 — CUDA batched prefill (wire the 5.9× gemm_*_tiled_v2 GEMMs)

> **Status:** OPEN. The top untapped CUDA prefill win. The validated
> `gemm_q4k/q5k/q6k_tiled_v2` GEMMs (Y[T,M]=A[T,K]·W[M,K]^T, ~9300 GFLOP/s,
> **5.9× over per-token matvec**) are in `kernels.cu` but UNWIRED. Production
> prefill (`main.zig`) runs `decodeStep`/`prefillStep` once per prompt token →
> re-reads every weight T times. Batch it: read each weight ONCE for all T
> tokens. Full design in memory `project_batched_prefill_design`.

Opened 2026-06-12. Stacks on the head-skip (`c78e64ed`, already on main).

## SCOPE — Phase 1: gemma-31b DENSE only (pure transformer → clean GEMM win)

Qwen is hybrid-SSM (3/4 layers = sequential state scan) and is OUT of scope
until a chunked/parallel SSM scan exists. gemma-26b MoE = Phase 2 (later). Do
gemma-31b dense first.

## APPROACH — ADDITIVE (forward_cuda_gemma.zig is the parallel gemma work's HOT

## file: add NEW methods + a toggle; do NOT rewrite existing decodeStep/blocks)

Add `ForwardGemma.prefillBatched(tokens) !u32` behind env `ZINC_BATCHED_PREFILL`:
process all T prompt tokens at once, return the last token's argmax (= first gen
token). Per layer:

- batched norms: loop T or a T-batched rms variant (cheap; not the bottleneck)
- Q/K/V + O projections, FFN gate/up/down: `gemm_*_tiled_v2`
  (`GemmPush{M,K,T,a_offset,x_offset,y_offset,acc_mode}`) over the T tokens
- attention: **Target 1 LOOPS** the existing single-query attention per token
  (each query masked to [0..its pos]) — defer the batched-attn kernel
- KV write: all T positions (batched, or loop)
  Then the last token's existing tail (the head). Wire `main.zig` prefill to call
  `prefillBatched` when `ZINC_BATCHED_PREFILL` is set, else the current per-token
  loop. (The duck-typed prefill fn already special-cases the last token.)

## TARGETS (one per cycle; each must pass the gate before any commit)

1. **Batched skeleton** — `prefillBatched` with LOOPED attention + GEMM
   projections/FFN. The big first step; may span >1 cycle (use the cycle log).
2. **Batched causal-attention kernel** — replace the per-token attention loop
   (grid=(n_head,T), each query masked to its pos; or flash-style online softmax).
3. **Batched KV write** — all T positions in one launch (if still looped).
4. (later) gemma-26b MoE prefill; tensor cores via NVRTC `-I` (+2.2×).

## GATE (NON-NEGOTIABLE — the batched path must be OUTPUT-IDENTICAL)

- `ZINC_BATCHED_PREFILL=1 dbg_cuda gen <prompt> N <model>` GEN_IDS must be
  **byte-identical** to the per-token path. Extend `scripts/prefill_catalog.sh`
  to A/B batched-vs-per-token (it already A/Bs head-skip the same way). Mismatch
  → REVERT.
- `scripts/validate_catalog.sh` → 5/5 token-correct vs llama.cpp.
- Measure prefill tok/s via `scripts/prefill_catalog.sh` (ABBA-counterbalanced).

## HARD RULES (from memory — violating wastes the cycle)

- **Build:** isolated caches; verify the binary hash CHANGED.
- **Box:** 4090-pinned (`${ZINC_GPU}`); isolated
  dir `~/zinc-e24`, NEVER `~/workspace/zinc` (parallel 5090 research).
- **Coordination:** main moves FAST (active parallel 5090/gemma work). Branch
  off the LATEST origin/main each cycle, rebase often, keep changes ADDITIVE
  (new methods + toggle) to minimize conflicts. NEVER roll back parallel work.
- Commit only a validated, output-identical, FASTER increment to `perf/e24-*`;
  push (NOT main). Incomplete/negative → log it in the cycle log + continue.

## EXPECTED

gemma-31b prefill ~15-35 → ~90-200 tok/s (5.9× on the GEMM-able majority;
attention is a smaller FLOP share). Stacks on the head-skip's +4%.

## CYCLE LOG

- **2026-06-12 — Cycle 1: batched dense-gemma prefill skeleton COMPLETE + output-identical + faster.**
  Wired the validated `gemm_q4k/q5k/q6k_tiled_v2` GEMMs into a new additive
  `ForwardGemma.prefillBatched(tokens) !u32` (forward_cuda_gemma.zig): token-major
  `BatchScratch` (lazy, sized to T), `gemmDispatch` (Q4_K/Q5_K/Q6_K → tiled_v2 GEMM;
  q8_0/f32 → per-token dmmv fallback), `attentionLayerBatched` (batched pre-norm +
  Q/K/V/O GEMMs; per-head V-norm/KV-write, Q/K norm+RoPE, causal softmax LOOPED per
  token via `aliasBuffer` into the token-major scratch — reuses the single-token
  kernels, zero new kernels), `ffnBlockBatched` (batched pre-norm + gate/up/down
  GEMMs + element-wise GeGLU over [T,n_ff] + fused post-ffn norm/residual/scale).
  Tail (rms_norm+LM head+argmax) on the last token only. MoE (n_experts>0) falls
  back to the per-token path. Toggle `ZINC_BATCHED_PREFILL` wired in BOTH main.zig
  (product) and dbg_cuda.zig gen-path (gate harness) + `Engine.prefillBatched`.
  Built clean on the 4090 box (`zig build cuda-dbg`, fresh `.zig-cache`, EXIT=0,
  bin 45251da3). Direct A/B (dbg_cuda gen, GEN_IDS = the gate):
  - gemma-31b T=80: 29.30→53.13 t/s (+81%), GEN_IDS byte-IDENTICAL ✓
  - gemma-31b T=200: 32.20→75.05 t/s (+133%), GEN_IDS byte-IDENTICAL ✓ (multi-tile T>128)
  - gemma-26b MoE T=60: GEN_IDS identical (per-token fallback, as designed) ✓
    GATE STATUS: GEN_IDS byte-identical on direct A/B (strong). REMAINING for merge:
    (1) extend scripts/prefill_catalog.sh to ABBA-counterbalance the batched A/B,
    (2) scripts/validate_catalog.sh 5/5. Committed to perf/e24-batched-prefill (WIP,
    toggle off by default → cannot regress production). NEXT: target #2 batched
    causal-attention kernel (replace the per-token attention loop — the remaining
    per-token launch overhead) for a bigger T, then run the formal gate.
- **2026-06-12 — Cycle 2: batched causal-attention kernel wired (target #2) — output-identical + faster.**
  Added an ADDITIVE kernel `gemma_attention_batched` (kernels.cu) — a verbatim twin
  of `gemma_attention` (same 3-pass softmax / GQA / sliding-window / no-sink / scale,
  same `zinc_block_reduce_*` order → bit-identical math) but batched over queries:
  block=(head=blockIdx.x, t=blockIdx.y), seq_len=t+1, token-major Q `q+(t*n_heads+head)*hd`
  and out, SWA `start = (window>0 && seq_len>window) ? seq_len-window : 0`. Chose a NEW
  gemma kernel over modifying main's just-landed `attention_causal_batched` (which carries
  sink logic gemma doesn't use) — zero conflict + guaranteed bit-identity. Wired in
  `attentionLayerBatched` (forward_cuda_gemma.zig): the per-token loop now does ONLY the
  norm/RoPE/KV-write (Q RoPE'd in place into b.q, K/V into the cache); the T per-token
  `gemma_attention` launches are REPLACED by ONE `gemma_attention_batched` launch
  grid=(n_head,T) reading b.q + the prompt region [0..T) of kv_k/kv_v → b.attn_out
  (shared mem = T*4). New `GemmaAttnBatchPush` + pipe `gemma_attention_batched`.
  Extended `scripts/prefill_catalog.sh` ADDITIVELY: `ZINC_AB=headskip|batched` (default
  headskip, unchanged) — batched mode A/Bs ZINC_BATCHED_PREFILL=1 vs baseline with the
  same ABBA counterbalancing + GEN_IDS-identical gate. Built clean on the 4090 box
  (fresh `.zig-cache`, EXIT=0, bin md5 dc54e7cd, CHANGED from cycle 1's 45251da3).
  GATE (ABBA x2, 250-tok prompt, 4090):
  - gemma4-31b dense: 31.43 → 94.12 t/s (+199%, ~3×), GEN_IDS byte-IDENTICAL ✓
  - gemma4-26b MoE: 54.52 → 58.19 (+7% noise, per-token fallback) GEN_IDS identical ✓
    Direct varied-output A/B (cyclic prompt → GEN 235,612,919,1471,218915,10205,…, not a
    collapsed prompt): byte-identical, prefill 30.4→91.7 t/s. Cycle 1 was +133% (T=200);
    the batched attention removed the per-token attn launch overhead → +199%. Committed to
    perf/e24-batched-prefill (toggle off by default → cannot regress production). REMAINING
    for merge: `scripts/validate_catalog.sh` 5/5 (per-token product path unchanged + batched
    is byte-identical to it → transitively correct). NEXT (cycle 3): target #3 batched KV
    write — fold the per-token norm/RoPE/KV-write loop into batched launches (kills the last
    T per-token launches); then NVRTC `-I` tensor-core GEMM (+2.2×).
- **2026-06-12 — Cycle 3: batched KV-write + Q/K norm-RoPE (target #3) — output-identical + faster.**
  Folded the per-token loop in `attentionLayerBatched` (V plain-norm+KV-write, Q/K per-head
  norm+RoPE) into ONE launch each (grid.y = T), eliminating the last T per-token launches on
  the batched gemma prefill path. Added two ADDITIVE kernels (kernels.cu): `rms_norm_rope_batched`
  and `rms_norm_kvwrite_batched` — verbatim twins of the single-token `rms_norm_rope`/`rms_norm_kvwrite`,
  batched over queries via block=(head=blockIdx.x, t=blockIdx.y) with explicit per-token src/dst
  strides (q_dim for Q in-place, kv_dim for K/V into the cache) and `position = base_position + t`
  (base 0 = fresh prompt). Per-block reduction order is UNCHANGED → bit-identical to the per-token
  loop. The `aliasBuffer` per-token ArrayList loop is GONE — kernels index the token-major scratch
  directly via `t*stride`. New `RmsRopeBatchPush`/`RmsKvWriteBatchPush` + 2 pipelines (additive).
  Single-token kernels + decodeStep/prefillStep untouched; `ZINC_BATCHED_PREFILL` off by default
  → cannot regress production. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg`
  EXIT=0 + ran, bin md5 6e74a1e5 ≠ cycle 2's dc54e7cd). GATE (scripts/prefill_catalog.sh
  `ZINC_AB=batched`, 4090, ABBA x2, 250-tok):
  - gemma4-31b dense: 30.30 → 132.97 t/s (+339%, ~4.4×), GEN_IDS byte-IDENTICAL ✓
  - gemma4-26b MoE: 46.40 → 49.20 (+6% noise, per-token fallback) GEN_IDS identical ✓
    Cycle 2 was +199% (94 t/s); removing the per-token KV-write/RoPE launches → +339% (133 t/s).
    Committed `d43216ac` to perf/e24-batched-prefill, pushed (NOT main). REMAINING for merge:
    `scripts/validate_catalog.sh` 5/5 (product path unchanged + batched byte-identical → transitively
    correct). NEXT (cycle 4): NVRTC `-I/usr/local/cuda/include` fp16-scratch tensor-core GEMM (+2.2×),
    or run the formal validate_catalog gate to clear merge; gemma-26b MoE prefill (Phase 2) is the
    other open lever (route T tokens, group by expert).
- **2026-06-12 — Cycle 4: gemma-26b MoE prefill — batched attention + per-token MoE FFN (Phase 2 cycle 1) — output-identical + faster.**
  Stopped the MoE model (gemma4-26b-a4b) from falling back ENTIRELY to per-token: its
  ATTENTION block is the SAME structure as the dense gemma-31b, so `prefillBatched` now
  runs the shared batched attention path (GEMM Q/K/V/O + batched causal attn + batched
  norm/RoPE/KV-write — already bit-identical to the per-token `attentionLayer`) for the
  MoE model too. The routed-expert FFN is still LOOPED per token: the FFN is
  position-independent, so looping the EXISTING single-token `moeFfnBlock` + `layerOutScale`
  over each token's hidden slice (alias-swap `self.hidden` → `b.hidden[t*n_embd]`, launch
  captures the raw device ptr at `cuLaunchKernel` so the alias wrapper frees safely) is
  OUTPUT-IDENTICAL to the per-token path. FFN type is decided PER LAYER (mirrors the
  per-token `n_experts>0 && ffn_gate_inp present` test, so a MoE model's dense layers still
  take `ffnBlockBatched`). Async MoE commands drained per layer (attentionLayerBatched's
  commitAndWait drains the stream → `drainPending`) + `waitPending` before the tail. ADDITIVE:
  only `prefillBatched` control flow changed (new per-layer MoE branch); single-token
  kernels + decodeStep/prefillStep + moeFfnBlock + the dense batched path all untouched →
  `if (moe)`-guarded so the proven dense path is byte-for-byte unchanged. NO new kernels.
  Built clean on the 4090 box (fresh `.zig-cache`, `-Dbackend=cuda -Dshaders=false` EXIT=0
  - ran, bin md5 f8cf047a ≠ cycle3's 6e74a1e5). GATE (scripts/prefill_catalog.sh
    `ZINC_AB=batched`, 4090, ABBA x2, 250-tok):
    - gemma4-26b MoE: 43.69 → 49.97 t/s (+14%), GEN_IDS byte-IDENTICAL ✓ (NEW: was full per-token fallback)
    - gemma4-31b dense: 30.34 → 147.87 t/s (+387%, ~4.9×), GEN_IDS byte-IDENTICAL ✓ (NO regression — dense flow unchanged)
      The +14% is the attention-projection batching only (Q/K/V/O GEMM + batched attn); the
      routed-expert FFN is still per-token. Committed to perf/e24-batched-prefill, pushed (NOT
      main). REMAINING for merge: `scripts/validate_catalog.sh` 5/5 (product per-token path
      unchanged + batched byte-identical → transitively correct). NEXT (cycle 5): full
      batched-expert MoE FFN (route T tokens, group/scatter by expert — the remaining MoE
      per-token cost), NVRTC `-I` fp16 tensor-core GEMM (+2.2× on the dense GEMMs), or run the
      formal validate_catalog gate to clear merge.
- **2026-06-12 — Cycle 5: FORMAL GATE CLEARED — batched prefill now validated DIRECTLY vs llama.cpp (5/5).**
  The dense Phase-1 goal is COMPLETE (gemma-31b +363%/142 t/s, reproduced this cycle); the sole
  remaining merge blocker in every prior log was `validate_catalog 5/5`, which only ever exercised
  the per-token PRODUCT path (it never set `ZINC_BATCHED_PREFILL`), so the batched path was only
  TRANSITIVELY validated ("product correct + batched byte-identical to it"). This cycle UPGRADES the
  gate to DIRECT: extended `scripts/validate_catalog.sh` ADDITIVELY with `ZINC_BATCHED=1` — it exports
  `ZINC_BATCHED_PREFILL=1` for the gemma rows only (ForwardGemma has `prefillBatched`; qwen would
  silently fall back, so it's left per-token), so the gemma rows now confirm the BATCHED prefill path
  is itself token-correct vs llama.cpp, not merely identical to per-token. Script-only change → the
  compute path / binary is byte-for-byte cycle 4 (built clean at HEAD on the 4090 box, fresh
  `.zig-cache`, `zig build cuda-dbg` EXIT=0; bin md5 f5025990, differs from cycle 4's f8cf047a only by
  cross-machine link non-determinism — no `.zig` edit). GATE (4090, 250-tok):
  - `validate_catalog` (plain, product per-token): **5/5 PASS** (qwen35-9b/qwen36-27b/qwen36-35b-a3b/
    gemma4-26b all 12/12 free-run; gemma4-31b free-run 2/12 = the documented q8_1-reference near-tie
    at pos 2 → teacher-forced 11/12 confirms correctness).
  - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate): **5/5 PASS** — gemma4-26b MoE batched
    free-runs **12/12** directly vs llama.cpp; gemma4-31b batched yields the IDENTICAL first token
    (same 2/12 near-tie + teacher-forced 11/12) → the batched prefill's first gen token == per-token's.
  - `prefill_catalog ZINC_AB=batched` (perf + GEN_IDS identity): ALL 5 GEN_IDS byte-identical;
    gemma4-31b dense **30.76 → 142.53 t/s (+363%)** (reproduces cycle 3/4's ~133/148); gemma4-26b MoE
    53.71 → 44.83 (**−17% this run = BOOST NOISE**, output identical — the routed-expert FFN is still
    per-token so the MoE gain is small/attention-only and swings ±17% on boost order; cf. cycle 4 saw
    +14% on the same path); qwen +2–3% (per-token fallback, noise).
    NET: the merge-gate CORRECTNESS bar is now MET both ways — the product path AND the batched path are
    independently 5/5 token-correct vs llama.cpp, on top of the GEN_IDS byte-identity. The branch remains
    WIP/unmerged (coordination: main moves fast; merge is a separate deliberate step), but it is no longer
    blocked on validation. NEXT (cycle 6): full batched-expert MoE FFN (route/group T tokens by expert —
    the remaining MoE per-token cost, to turn the noisy ±17% into a real MoE win), or NVRTC `-I` fp16
    tensor-core GEMM (+2.2× on the dense GEMMs — note: NOT byte-identical, would need its own tolerance gate).
- **2026-06-13 — Cycle 6: batched Q8_0 shared-expert FFN for gemma-26b MoE prefill (Phase 2 cycle 2) — output-identical + faster (the real MoE FFN win).**
  Cycle 4/5 batched only the MoE model's ATTENTION (the routed-expert FFN was still fully per-token), so
  gemma-26b's gain was attention-only and swung ±17% on boost (+14% cycle 4, −17% cycle 5 = noise). This
  cycle attacks the FFN: the gemma-26b shared-expert (the dense FFN every token runs alongside the routed
  experts) is **Q8_0** (gate/up/down) — ~17.8 MB/token of weight reads that the per-token loop re-read T
  times. Added an ADDITIVE Q8_0 prefill GEMM and batched the whole shared expert over all T tokens (weights
  read ONCE). Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
  - `gemm_q8_0_tiled_v2` (kernels.cu): bit-faithful twin of `gemm_q4k_tiled_v2` (64×64 tile, 256 thr, 4×4
    micro-tile, BK=32 = one Q8_0 block) with Q8_0 dequant `value = d*(float)q` matching dmmv_q8_0. Byte-
    addressed, pc.a_offset in BYTES. NVRTC-compiled at runtime (confirmed: "compiled gemma4 kernel pipelines").
  - `gemm: [3]→[4]` + `pipes.gemm[3]=gemm_q8_0_tiled_v2` + `gemmDispatch` `.q8_0 => 3`. gemma-31b dense has
    NO Q8_0 in the batched path (all Q4_K/Q6_K) → the proven dense path is byte-for-byte unchanged.
  - `BatchScratch.shared` ([T,n_embd]) + `sharedExpertBatched(L,T,b)`: batched pre-norm + gate/up/down GEMM
    (Q8_0→gemm_q8_0) + GeGLU[T,sf] + post_ffw_norm_1 → b.shared (same dmmv→GEMM swap the proven dense
    `ffnBlockBatched` makes). `moeRoutedCombine(L)`: `moeFfnBlock` MINUS its shared sub-block (router top-k +
    routed experts + post_ffw_norm(shared+moe)+residual), per token, reading b.shared[t] via a `self.shared_buf`
    alias (same per-token alias pattern as `self.hidden`). prefillBatched MoE branch now: sharedExpertBatched
    ONCE/layer, then loop T × (moeRoutedCombine + layerOutScale). Per-token submits drop 5→4 (shared no longer
    per-token) → fits the async ring at T=250.
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
    bin md5 11ac3da1 ≠ cycle 5's f5025990). GATE — FULLY CLEARED:
  - Direct A/B (varied non-collapsed prompt, GEN 228,228,228,228,49,50,3236,608,…): gemma-26b prefill
    **38.47 → 82.00 t/s (+113%)**, GEN_IDS byte-IDENTICAL.
  - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok): ALL 5 GEN_IDS byte-identical —
    **gemma4-26b 42.61 → 81.52 t/s (+91%)** (was the noisy +14%/−17% attention-only), gemma4-31b dense
    31.03 → 132.81 t/s (+328%, NO regression), qwen ±0–4% (per-token fallback, noise).
  - `validate_catalog` plain **5/5**; `ZINC_BATCHED=1 validate_catalog` **5/5** — gemma4-26b MoE batched
    free-runs **12/12** DIRECT vs llama.cpp (the batched shared-expert path is token-correct); gemma4-31b
    the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged — dense path untouched).
    The noisy ±17% MoE swing is now a SOLID +91% (ABBA), correct both ways. Committed to perf/e24-batched-prefill,
    pushed (NOT main). NEXT (cycle 7): full token-GROUPED routed experts (route/group T tokens by expert so each
    active expert's Q4_K/Q5_1 weights — ~28 MB/token, the remaining per-token FFN cost — are read once per group,
    not once per token; needs a gather/scatter + variable-group GEMM, harder for byte-identity), or NVRTC `-I`
    fp16 tensor-core GEMM (+2.2× dense, NOT byte-identical → own tolerance gate).
- **2026-06-13 — Cycle 7: batched router for gemma-26b MoE prefill (Phase 2 cycle 3) — output-identical + faster.**
  Cycle 6 batched the Q8_0 shared-expert FFN; the routed-expert sub-block was still fully per-token, and EACH
  token re-ran the router: rms_norm_noweight + per-channel scale + an **F32 `ffn_gate_inp` matvec (~2 MB/token,
  re-read T times = ~500 MB at T=250)** + top-k softmax + its own command submit. This cycle BATCHES the router
  over all T tokens — the F32 router weight is read ONCE and one fewer command submits per token (4→3 on the
  prerouted path). Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
  - 3 ADDITIVE kernels (kernels.cu): `gemm_f32_tiled_v2` (bit-faithful twin of gemm_q4k_tiled_v2 — 64×64 tile,
    256 thr, 4×4 micro-tile, BK=32 — but NO dequant: stages the W tile straight from a row-major f32 weight;
    the batched twin of looping dmmv_f32, token-correct not bit-exact, same class as the proven quant GEMMs),
    `softmax_topk_batched` (verbatim softmax_topk over grid {T} — per-block routing math byte-for-byte the
    single-token kernel's), `mul_vec_scaled_batched` (a[t*row+i]=a[t*row+i]·b[i]·scale, broadcasting the
    per-channel ffn_gate_inp.scale across tokens).
  - `gemm_f32` pipeline + `softmax_topk_batched`/`mul_vec_scaled_batched` pipes; `BatchScratch.router_in`
    [T,n_embd] / `router_logits` [T,n_experts] / `router_table` [T,2·n_used] (sized 1 on the dense model).
  - `routerBatched(L,T,b)`: norm→scale→gemm_f32 logits→softmax_topk_batched → b.router_table, async on the
    shared stream (the per-token expert launches read the finished table by stream order — no host sync).
    Mirrors moeRoutedCombine's router sub-block op-for-op (same residual input: all tokens' PRE-FFN hidden,
    identical to the per-token path where each token routes before its own FFN residual add).
  - `moeRoutedCombine(L, prerouted)`: when prerouted, SKIP the router sub-block (caller aliases
    self.router_out_buf to b.router_table's row t); experts + combine unchanged. prefillBatched MoE branch
    calls routerBatched ONCE/layer (only on the Q4_K-gate_up + Q5_1-down async expert path), then loops T ×
    (alias router_table[t] → moeRoutedCombine(prerouted=true) + layerOutScale).
    Built clean on the 4090 box (fresh source rsync, warm `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda
-Dshaders=false` EXIT=0 + ran, bin md5 0a080dd5 ≠ cycle 6's 11ac3da1). GATE — FULLY CLEARED:
  - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
    **gemma4-26b 44.70 → 103.92 t/s (+132%)** (up from cycle 6's 81.5 t/s — the batched router added ~+27%
    absolute on top of the batched shared expert), gemma4-31b dense 30.75 → 136.55 t/s (+344%, NO regression
    — dense flow untouched).
  - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
    free-runs **12/12** DIRECT vs llama.cpp (the batched-router prefill path is token-correct); gemma4-31b the
    documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token fallback intact
    → the product decode path is unregressed). Plain validate transitively covered (default path byte-unchanged).
    Committed to perf/e24-batched-prefill, pushed (NOT main). NEXT (cycle 8): the routed gate_up/down expert GEMMs
    are the last per-token FFN cost (~36 MB/token Q4_K/Q5_1) — token-GROUPED routing (build per-expert token lists
    from b.router_table, expert-major grouped GEMM so each expert's weight is read once per group not once per
    token; needs gather/scatter, harder for byte-identity), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own
    tolerance gate). b.router_table is now the device-side foundation a grouped-expert pass consumes.
- **2026-06-13 — Cycle 8: token-batched routed-expert matvecs for gemma-26b MoE prefill (Phase 2 cycle 4) — output-identical + faster (the last big per-token FFN cost).**
  Cycle 7 batched the router; the routed gate/up/down expert matvecs themselves were STILL looped per
  token (~36 MB/token: Q4_K gate_up + Q5_1 down). This cycle batches them over all T prompt tokens in
  single launches (grid.y = T) instead of T separate single-token launches. NOT token-grouped (no
  gather/scatter) — each (token t, expert-slot e, row) block reads token t's own router row, input slice,
  and output slice, so the per-block dequant + reduction is byte-for-byte the single-token kernel; only
  the launch dimension is batched. Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
  - 2 ADDITIVE kernels (kernels.cu): `dmmv_q4k_experts_batched` / `dmmv_q5_1_experts_batched` — bit-faithful
    twins of `dmmv_q4k_experts`/`dmmv_q5_1_experts` (same dequant + `zinc_block_reduce_sum` order) with
    per-token src/dst strides (`x_tok_stride`/`y_tok_stride`) + `routing_stride` so block=(g=blockIdx.x →
    slot·M+row, t=blockIdx.y) reads ids from `b.router_table[t*2·n_used + e]`. New `ExpertsBatchPush`
    (per-token strides appended to `ExpertsPush`) + 2 pipelines.
  - `BatchScratch.{moe_norm_e, gate_e, up_e, geglu_e, down_e}` ([T, …] slot-major, size 1 on dense).
  - `moeRoutedExpertsBatched(L,T,b)`: batched pre_ffw_norm_2 → b.moe_norm_e, gate (base 0) + up (base
    gu_half) Q4_K expert matvecs → b.gate_e/up_e, GeGLU over [T, n_used·ef] → b.geglu_e, Q5_1 down →
    b.down_e [T, n_used·n_embd] slot-major. Async on the shared stream; push params mirror the per-token
    `dmmv_*_experts` dispatches exactly. The unscaled down output (down scale folded at accumulate via
    moe_weighted_acc_scaled, unchanged) → bit-identical.
  - `moeRoutedCombine(L, prerouted, preexperts)`: when preexperts, SKIP pre_ffw_norm_2 + the expert
    matvecs + GeGLU (done batched) and read this token's b.down_e slice via the `self.down_buf` alias →
    only the zero+weighted-accumulate + post_ffw_norm_2 + scale_accumulate + post_ffw_norm + residual
    combine remain per token. `pre` (Q4_K gate_up + Q5_1 down) ⟹ batched ⟹ scaled accumulate, consistent.
    prefillBatched MoE branch now: sharedExpertBatched + routerBatched + moeRoutedExpertsBatched ONCE/layer,
    then loop T × (moeRoutedCombine(prerouted, preexperts) + layerOutScale) — the per-token loop is now just
    the lightweight accumulate/combine tail. Host-readback fallback keeps its full per-token path.
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false`
    EXIT=0, bin md5 d14f13bc ≠ cycle 7's 0a080dd5; NVRTC compiled the 2 new kernels at runtime). GATE — FULLY CLEARED:
  - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
    **gemma4-26b MoE 39.72 → 135.16 t/s (+240%)** (UP from cycle 7's 103.92 — batching the routed-expert
    matvecs is the last big FFN lever), gemma4-31b dense 30.47 → 137.96 t/s (+353%, NO regression —
    dense flow untouched).
  - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
    free-runs **12/12** DIRECT vs llama.cpp (the batched routed-expert prefill path is token-correct);
    gemma4-31b the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12
    (per-token fallback intact → product decode path unregressed).
    Committed `3b1afc78` to perf/e24-batched-prefill, pushed (NOT main). The gemma-26b MoE prefill FFN is now
    FULLY batched (shared expert + router + routed experts); the only remaining per-token work is the cheap
    accumulate/combine tail. NEXT (cycle 9): either token-GROUPED routed experts (gather tokens by expert so
    each expert's weight is read once per group not once per token — a memory-traffic win beyond launch
    batching, but needs gather/scatter and is harder for byte-identity), batch the per-token accumulate/combine
    tail too (kills the last T launches/layer), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own tolerance gate).
- **2026-06-13 — Cycle 9: batched accumulate/combine tail for gemma-26b MoE prefill (Phase 2 cycle 5) — output-identical + faster (the LAST per-token FFN launches gone).**
  After cycle 8 the gemma-26b MoE FFN was batched through the routed-expert matvecs, but the FINAL stage —
  per token: zero accumulator → weighted-combine (down scale folded) → post_ffw_norm_2 → shared+=moe →
  post_ffw_norm → hidden+=cur → layer_output_scale — was still a T-iteration loop (T × ~7 single-token
  launches/layer + T alias-buffer create/free pairs). This cycle batches that tail, so on the GPU-side async
  expert path (Q4_K gate_up + Q5_1 down) the prefill MoE FFN has NO per-token loop at all. ADDITIVE
  (decodeStep/moeFfnBlock/moeRoutedCombine/dense path untouched):
  - 1 ADDITIVE kernel (kernels.cu): `moe_weighted_acc_scaled_batched` — bit-faithful twin of
    `moe_weighted_acc_scaled` (same j-loop FMA order, same GPU-side `escale[id]` down-scale fold) with
    grid.y = T + per-token strides (`a_tok_stride`/`b_tok_stride`/`routing_stride`) so block=(i=blockIdx.x,
    t=blockIdx.y) reads token t's own accumulator/down/routing slices → per-(t,i) math byte-for-byte the
    single-token kernel. New `MoeAccBatchPush` + 1 pipeline. The other 6 tail ops needed NO new kernel:
    `zero_vec`/`scale_accumulate`/`scalar_mul` are element-wise → one launch over the whole [T,n_embd] tile
    (N=T·n_embd, contiguous token-major → each element identical to the per-token launch); `rms_norm`
    (post_ffw_norm_2, post_ffw_norm) already indexes token=blockIdx.x → grid.x=T reproduces the per-token
    reduction block-for-block. The standalone per-token `layerOutScale` is folded in as the final batched
    `scalar_mul` (self-skips when the layer has no output scale).
  - `BatchScratch.moe_out_e` [T,n_embd] (the batched post_ffw_norm_2 accumulator; size T·n_embd on dense too,
    cheap) + `moeRoutedCombineBatched(L,T,b)` (the 7 batched launches above, async on the shared stream —
    stream order guarantees it reads the experts/router buffers after they're written). prefillBatched MoE
    branch: on the `pre` (batched) path it now calls sharedExpertBatched + routerBatched +
    moeRoutedExpertsBatched + **moeRoutedCombineBatched** ONCE/layer — the per-token loop is GONE. The non-`pre`
    host-readback fallback keeps its per-token `moeRoutedCombine(false,false)` + `layerOutScale` loop (aliasing
    b.hidden/b.shared per token), byte-for-byte unchanged.
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
    bin md5 5a1b0597 ≠ cycle 8's d14f13bc; NVRTC compiled the new kernel at runtime). GATE — FULLY CLEARED:
  - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
    **gemma4-26b MoE 46.72 → 160.49 t/s (+244%)** (UP from cycle 8's 135.16 — killing the per-token combine
    loop is the last FFN lever), gemma4-31b dense 30.11 → 130.21 (+332%, NO regression — dense flow untouched).
  - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
    free-runs **12/12** DIRECT vs llama.cpp (the fully-batched MoE prefill path is token-correct); gemma4-31b
    the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token fallback
    intact → product decode path unregressed).
    Committed to perf/e24-batched-prefill, pushed (NOT main). The gemma-26b MoE prefill FFN is now FULLY batched
    end-to-end (shared expert + router + routed experts + accumulate/combine + output scale) — zero per-token
    launches on the async expert path. NEXT (cycle 10): token-GROUPED routed experts (gather tokens by expert →
    each expert weight read once per group, a memory-traffic win beyond launch batching; needs gather/scatter,
    harder for byte-identity), batch the attention layer's remaining per-token bits if any, or NVRTC `-I` fp16
    tensor-core GEMM (+2.2× dense, own tolerance gate).
- **2026-06-13 — Cycle 10: async-pipeline the batched prefill layer loop (remove per-layer host syncs) — output-identical + faster.**
  After cycle 9 the gemma MoE/dense prefill FFN+attention were fully batched, but the layer loop still
  BLOCKED THE CPU twice (dense) / once (MoE) per layer: `attentionLayerBatched`, `ffnBlockBatched` and
  `sharedExpertBatched` each ended with `cmd.commitAndWait()` — a full `cuStreamSynchronize` round-trip. On
  WSL2 each is ~0.4 ms and, worse, the idle gaps starve the GPU boost clock (the exact pathology the merged
  DECODE async ring fixes). This cycle removes them: all three helpers now `self.submit(cmd)` (commitAsync +
  stash) on the SAME single shared CUstream the async MoE stages already use (confirmed `m->stream = c->stream`
  in cuda_shim.c → every launch is implicitly ordered, so cross-layer buffer reuse stays byte-identical — only
  CPU blocking is removed, not GPU execution order). The mid-loop `if (moe) drainPending()` (which relied on
  attention's commitAndWait draining the stream) is GONE; a single unconditional `waitPending()` before the
  tail now frees every layer's stashed command (ring depth = blocks/layer × n_layers ≈ 5×48 for gemma-26b,
  well under the 1024 ring; submit() self-falls-back to sync if it ever fills). The per-token MoE FALLBACK
  path (non-`pre`) is untouched — its inner `moeRoutedCombine` still commitAndWaits around its host id
  readback, which safely drains any stashed attention/shared command. ADDITIVE: only `prefillBatched` +
  its three batched helpers changed; decodeStep/prefillStep/per-token path/single-token kernels untouched;
  NO new kernels. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda
-Dshaders=false` EXIT=0 + ran, bin md5 2eca13ad ≠ cycle 9's 5a1b0597). GATE — FULLY CLEARED:
  - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
    **gemma4-26b MoE 46.27 → 213.46 t/s (+361%)** (UP from cycle 9's 160.49 — removing the per-layer
    host syncs is a large MoE win: more layers × launches = more sync gaps removed + boost held),
    **gemma4-31b dense 30.02 → 140.56 t/s (+368%)** (UP from cycle 9's 130.21).
  - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
    free-runs **12/12** DIRECT vs llama.cpp (the async-pipelined prefill path is token-correct); gemma4-31b
    the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token
    fallback intact → product decode path unregressed).
    Committed to perf/e24-batched-prefill, pushed (NOT main). The batched prefill layer loop now runs with NO
    per-layer CPU↔GPU sync (one drain before the tail). NEXT (cycle 11): token-GROUPED routed experts (gather
    tokens by expert → each expert weight read once per group, a memory-traffic win beyond launch batching;
    needs gather/scatter, harder for byte-identity), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own
    tolerance gate — NOT byte-identical).
- **2026-06-13 — Cycle 11: fp16 tensor-core (wmma) GEMM for the dense Q4_K projections/FFN — opt-in, token-correct, +6%.**
  Took the last NEXT-list lever: the NVRTC `-I/usr/local/cuda/include` path (landed in `a0d463af`) lets NVRTC
  resolve `<mma.h>`, so the dense batched Q4_K GEMMs (attn Q/K/V/O + FFN gate/up — the bulk of the dense FLOPs)
  can run on the fp16 tensor cores instead of the f32 register-tiled GEMM. Added an ADDITIVE kernel `gemm_q4k_tc`
  (kernels.cu): same `Y[T,M]=A[T,K]·W[M,K]^T` and SAME Q4_K dequant unpack as `gemm_q4k_tiled_v2`, but the inner
  product runs on wmma 16×16×16 fragments — the dequant'd W tile (m-major `Ws[r*BK+k]`) and f32 activations
  (k-major `As[k*BT+t]`) are cast to `__half` in shared mem, accumulated in fp32, stored col-major into a
  token-major `Cs` tile, then guard-copied to `Y[T,M]`. 64×64 tile / 256 thr = 8 warps, BK=32 (1 Q4_K sub-block
  = 2 wmma k-steps); each warp owns 2 accumulator fragments (M-blocks fm, fm+2 at T-block ft) → all 4×4=16
  (m,t)-blocks covered once; static shared = 24 KB. Wired OPT-IN behind `ZINC_BATCHED_TC` (read once per
  `prefillBatched` into `self.use_tc`): `gemmDispatch` routes Q4_K (idx 0) to `gemm_q4k_tc` only when `use_tc`,
  same `GemmPush`/grid/block — so with the toggle OFF the dispatch is byte-for-byte the proven f32 path.
  NOT byte-identical when ON (fp16 input rounding) → its gate is token-correctness vs llama.cpp, NOT the
  GEN_IDS byte-identity gate. Extended `scripts/validate_catalog.sh` + `scripts/prefill_catalog.sh` ADDITIVELY
  to pass `ZINC_BATCHED_TC=1` through (gemma rows / batched mode). ADDITIVE everywhere: 1 new kernel + 1 new
  pipeline + 1 bool + a 1-line dispatch branch; decodeStep/prefillStep/per-token path/the proven dense+MoE
  batched paths all unchanged when the toggle is off. Built clean on the 4090 box (fresh `.zig-cache`,
  `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0 + ran, bin md5 602a6918 ≠ cycle 10's 2eca13ad;
  NVRTC compiled `gemm_q4k_tc` at runtime — "compiled gemma4 kernel pipelines"). GATE — its own tolerance gate
  CLEARED:
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (TC path direct vs llama.cpp, 4090): **5/5 PASS** —
    gemma4-26b MoE (TC on its dense-attn Q4_K GEMMs) free-runs **12/12**; gemma4-31b the SAME documented
    near-tie as the f32 path (free-run 2/12, teacher-forced 11/12) → fp16 TC rounding introduced **NO new
    divergence**; qwen 12/12 (per-token fallback). So the TC path is token-correct within tolerance.
  - Default batched path (toggle OFF) unchanged: 3-way direct A/B (gemma-31b, 220-tok) baseline /
    batched / batched+TC ALL GEN_IDS byte-identical (the off-path is provably the f32 path; on-path matched
    here too on this prompt).
  - Perf (ABBA B T T B, gemma-31b 250-tok, 4090): batched ~138.0 t/s (134.4, 141.6) → **batched+TC ~146.6
    t/s (146.7, 146.5) = +6.2%** over the already-batched dense path; baseline per-token was ~32 t/s
    (so batched+TC is ~4.6× the per-token prefill).
    The +6% is modest because the dense GEMM is MEMORY-bound here (Q4_K dequant + f32 activation reads dominate;
    the fp16 multiply was never the bottleneck) — tensor cores help the FLOP-heavy share only. Committed to
    perf/e24-batched-prefill, pushed (NOT main). Kept opt-in (off by default) → cannot regress production or the
    proven byte-identical batched gate. NEXT (cycle 12): make the TC path memory-efficient to unlock more than
    +6% — cache the dequant'd weights in fp16 once (kill per-GEMM dequant) and/or keep activations fp16 across
    the layer (kill the f32↔fp16 recast per GEMM); OR token-GROUPED routed experts (gather by expert =
    memory-traffic win, harder byte-identity).
- **2026-06-13 — Cycle 12: TC GEMM reads a PRE-CONVERTED fp16 activation (halve the dominant A read) — +18% over f32 batched / +8% over cycle-11 plain TC, byte-identical to plain TC.**
  Took cycle 11's own NEXT item: the +6% plain-TC gain was small because `gemm_q4k_tc` re-reads the f32
  activation A from global ONCE PER OUTPUT M-BLOCK (grid.x = M/64), so the f32 A traffic is ~(M/64)× the
  weight traffic and dominates the memory-bound dense GEMM. This cycle pre-converts A to fp16 ONCE per GEMM
  (`f32_to_f16`) and feeds a half-width A to the TC kernel — halving + de-duplicating the dominant read.
  ADDITIVE (decodeStep/per-token path/dense+MoE batched f32 paths/cycle-11 plain TC all untouched):
  - 2 ADDITIVE kernels (kernels.cu): `f32_to_f16` (element-wise `y[i]=__float2half(x[i])`) and
    `gemm_q4k_tc_f16a` — IDENTICAL to `gemm_q4k_tc` (same Q4_K dequant, same wmma 16×16×16 schedule, same
    Cs store / guarded copy) EXCEPT A arrives already fp16 (no per-load `__float2half`). The downcast uses
    the SAME `__float2half` the plain TC kernel applied in shared → the staged half bits are identical →
    `gemm_q4k_tc_f16a`'s output is BYTE-FOR-BYTE `gemm_q4k_tc`'s (confirmed empirically below), only the
    global A read is halved + read once not once-per-M-block.
  - `gemm_q4k_tc_f16a` + `f32_to_f16` pipelines + `BatchScratch.act_f16` ([T, max activation] halves) +
    `F32ToF16Push`. `gemmDispatch`: when `use_tc` and Q4_K (idx 0), pre-convert A → `act_f16` then dispatch
    `gemm_q4k_tc_f16a` (cycle-12 default). `x_offset == 0` on the batched GEMM path (contiguous [T,K]) so the
    whole [T·K] tile converts at offset 0; `act_f16` reuse across a layer's GEMMs is safe (stream-ordered).
  - A/B knob `ZINC_BATCHED_TC_PLAIN` (additive bool): forces cycle-11 `gemm_q4k_tc` (f32-A re-read) so the
    f16-A memory-traffic win is measurable in isolation. Off → the f16-A path. Both gated under `ZINC_BATCHED_TC`.
    Built clean on the 4090 box (fresh source rsync, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
    bin md5 33e6e1a1 ≠ cycle 11's 602a6918; NVRTC compiled the 2 new kernels at runtime). GATE — its own
    tolerance gate CLEARED (TC is fp16 → token-correctness, not GEN byte-identity vs f32):
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (f16-A TC path direct vs llama.cpp, 4090): **5/5 PASS**
    — qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME documented near-tie as
    the f32/plain-TC path (free-run 2/12, teacher-forced 11/12) → the f16-A pre-convert introduced **NO new
    divergence**.
  - f16-A == plain-TC byte-identity (the WIP's core claim): direct ABBA (gemma-31b, 250-tok) P=plain-TC
    F=f16a-TC → GEN_IDS IDENTICAL on every run → `gemm_q4k_tc_f16a` is byte-for-byte `gemm_q4k_tc`, as
    designed. Default path (`ZINC_BATCHED_TC` unset) is the unchanged f32 batched GEMM.
  - Perf (ABBA, gemma-31b 250-tok, 4090): plain-TC ~147.9 t/s → **f16-A TC ~159.6 t/s (+7.9%** over cycle-11
    plain TC); and batched-no-TC ~137.2 → **f16-A TC ~161.5 (+17.7%** over the f32 batched path — UP from
    cycle 11's plain-TC +6%). Per-token baseline ~32 t/s → f16-A TC is **~5×** the per-token prefill.
    The f16-A path roughly TRIPLES cycle 11's TC win (+6%→+18% over f32) by attacking the activation read
    traffic that made the GEMM memory-bound — exactly what the cycle-11 NEXT note predicted. Committed to
    perf/e24-batched-prefill, pushed (NOT main). Opt-in (off by default) → cannot regress production or the
    proven byte-identical batched gate. NEXT (cycle 13): cache the dequant'd Q4_K weights in fp16 ONCE per layer
    (kill the per-GEMM Q4_K dequant — the other half of the TC GEMM's traffic) and/or keep activations fp16
    across the whole layer (avoid the per-GEMM f32→fp16 recast); OR token-GROUPED routed experts (gather by
    expert = memory-traffic win beyond launch batching, harder byte-identity).
- **2026-06-13 — Cycle 13: extend the f16-A tensor-core GEMM to Q6_K weights (complete TC coverage of the dense gemma GEMMs) — token-correct, marginal (+1.8%, in noise).**
  Cycles 11/12 wired the fp16 tensor cores only for the dense Q4_K GEMMs (gemmDispatch idx 0 — attn Q/K/V/O +
  FFN gate/up); the dense gemma-31b also carries **Q6_K** weights (notably ffn_down, idx 2), which still fell
  back to the f32 register-tiled `gemm_q6k_tiled_v2` even with `ZINC_BATCHED_TC` on. This cycle extends the
  proven cycle-12 f16-A TC pattern to Q6_K so ALL dense quant types run on the tensor cores when toggled.
  ADDITIVE (decodeStep/per-token path/dense+MoE batched f32 paths/the Q4_K TC kernels all untouched):
  - 1 ADDITIVE kernel (kernels.cu): `gemm_q6k_tc_f16a` — `gemm_q4k_tc_f16a` in every respect (same wmma
    16×16×16 schedule, m-major half Ws tile, k-major fp16 As tile, fp32 accumulate, token-major Cs store +
    guarded copy, pre-converted fp16 A read once) EXCEPT the weight sub-block is dequant'd with the Q6_K
    unpack copied VERBATIM from `gemm_q6k_tiled_v2` (210 B/256 elems; 6-bit q=(ql_nibble|(qh_bits<<4));
    value=d·sc·(q−32)). Q6_K is byte-addressed → `const unsigned char* a`. NOT bit-identical to the f32 Q6_K
    path (fp16 rounding) → token-correctness gate, same class as the Q4_K TC kernels.
  - `gemm_q6k_tc_f16a` pipeline + `gemmDispatch` branch: when `use_tc` and idx==2 (and the new `use_tc_q6`),
    pre-convert A → `act_f16` (reuses the cycle-12 `f32_to_f16` + `BatchScratch.act_f16`, already sized to
    `T·max(n_ff,…)` so the K=n_ff ffn_down tile fits) then dispatch the Q6_K TC kernel. `x_offset==0` on the
    batched path (contiguous [T,K]) so the whole tile converts at offset 0.
  - A/B kill-switch `ZINC_BATCHED_TC_NOQ6` (additive bool `use_tc_q6`, default true) forces Q6_K back to the
    f32 path even with TC on → lets the Q6_K-on-TC increment be measured in ONE binary (= cycle-12 behavior).
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
    bin md5 b97d310e ≠ cycle 12's 33e6e1a1; NVRTC compiled the new kernel at runtime). GATE — its own tolerance
    gate CLEARED (TC is fp16 → token-correctness, not GEN byte-identity vs f32):
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (Q6_K-TC path direct vs llama.cpp, 4090): **5/5 PASS**
    — qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME documented near-tie as
    the f32/Q4_K-TC path (free-run 2/12, teacher-forced 11/12) → the Q6_K fp16 TC introduced **NO new divergence**.
  - Isolated A/B (ONE binary, gemma-31b 250-tok, 4090, ABBA x2): Q6K-on-TC (cycle 13) **~160.5 t/s** vs
    Q6K-off-TC (`ZINC_BATCHED_TC_NOQ6=1`, = cycle-12 behavior) **~157.7 t/s** = **+1.8%, WITHIN BOOST NOISE**
    (B runs spanned 150–168). The increment is marginal because Q6_K is only ~1/7 of the dense GEMM work
    (one Q6_K ffn_down vs six Q4_K projections/gate/up per layer) — the bulk was already on TC since cycle 12.
    HONEST OUTCOME: this is a completeness step (all dense quant types now on the tensor cores, token-correct with
    no new divergence) rather than a meaningful perf win; the Q6_K share is too small to clear the box's ±1% boost
    floor on its own. Opt-in (off by default) + the new kill-switch → cannot regress production or the proven
    byte-identical batched gate. Committed to perf/e24-batched-prefill, pushed (NOT main). NEXT (cycle 14): the
    bigger memory-traffic lever stays the fp16 WEIGHT cache — dequant Q4_K/Q6_K to fp16 ONCE per layer so the TC
    GEMMs read pre-dequant'd half weights (kills the per-GEMM dequant, the other half of the GEMM traffic); OR
    token-GROUPED routed experts (gather by expert, memory-traffic win beyond launch batching, harder byte-identity).
- **2026-06-13 — Cycle 14: wider 128×64 M-tile TC GEMM to halve the dominant f16-A re-read — NEGATIVE RESULT (-11.8%, occupancy-limited); kept opt-in, default unchanged.**
  Took cycle 12's diagnosis to its logical lever: the f16-A activation read DOMINATES this memory-bound TC GEMM
  because `gemm_q4k_tc_f16a` re-reads A once per output M-block (grid.x = M/BM, BM=64). Widening the output M-tile
  to 128 rows HALVES grid.x → halves that dominant A traffic (weight bytes & dequant compute unchanged). Added an
  ADDITIVE kernel `gemm_q4k_tc_f16a_m128` (kernels.cu): `gemm_q4k_tc_f16a` in every respect (same Q4_K dequant, same
  wmma 16×16×16 fp16 schedule, fp32 accumulate, token-major Cs store + guarded copy, pre-converted fp16 A read once)
  EXCEPT BM=128 — 256 thr = 8 warps, each warp owns ONE 16-token T-block (`ft = warp&3`) and FOUR 16-row M-blocks
  (`fmbase = warp>>2` → even {0,2,4,6} / odd {1,3,5,7}) = 8 warps × 4 frags = 32 = all 8×4 (m,t) 16×16 blocks of the
  128×64 tile once; static shared 8K Ws + 4K As + 32K Cs = 44 KB. New pipeline + dispatch branch.
  CORRECTNESS — byte-identical (verified): same per-output dequant + wmma math, only tiling/grid differ → output is
  byte-for-byte `gemm_q4k_tc_f16a`'s. Direct A/B (gemma-31b, VARIED non-collapsing prompt, GEN 238066,240017,…):
  m128 vs m64 GEN_IDS IDENTICAL ✓. PERF — NEGATIVE: ABBA x2 (gemma-31b 250-tok, 4090) m128(A) **145.96** vs m64(B)
  **165.53 t/s = -11.8%** (consistent across all 8 runs: A∈[143.7,148.7], B∈[162.6,168.8] — not noise). ROOT CAUSE:
  the 44 KB static shared caps occupancy at 1 block/SM (vs m64's 24 KB → 2 blocks/SM), so the lost latency-hiding
  outweighs the halved A read on this memory-bound GEMM. Wider-tile hypothesis FALSIFIED.
  OUTCOME: flipped the TC path DEFAULT back to the proven 64×64 `gemm_q4k_tc_f16a` (= cycle 12 behavior); the m128
  kernel is kept as a DOCUMENTED experiment, opt-in via the renamed `ZINC_BATCHED_TC_M128` (was `ZINC_BATCHED_TC_M64`,
  inverted) so a future cycle won't re-attempt it. ADDITIVE everywhere (1 kernel + 1 pipeline + 1 bool + a dispatch
  branch); default TC path byte-for-byte cycle 12 → cannot regress. Built clean on the 4090 box (fresh `.zig-cache`,
  `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0, bin md5 a1b7e365 ≠ cycle 13's b97d310e; NVRTC compiled
  the new kernel at runtime). GATE: `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (default m64 TC path, 4090)
  **5/5 PASS** — qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all 12/12; gemma4-31b the documented near-tie
  (teacher-forced 11/12, free-run 2/12), unchanged. Committed to perf/e24-batched-prefill, pushed (NOT main).
  NEXT (cycle 15): the remaining big memory-traffic lever is the fp16 WEIGHT cache — dequant Q4_K/Q6_K → fp16 ONCE
  per layer so the TC GEMMs read pre-dequant'd half weights (kills the per-GEMM dequant = the other half of the GEMM
  traffic, and unlike the M-tile widening does NOT cost occupancy); OR token-GROUPED routed experts (gather by expert).
- **2026-06-13 — Cycle 15: 8 KB-shared (lowsmem) Q4_K TC GEMM — POSITIVE (+8.9%/+11.6%), byte-identical, now the DEFAULT TC path.**
  Cycle 14's m128 NEGATIVE result diagnosed this GEMM as OCCUPANCY-bound: 44 KB shared → 1 block/SM lost 11.8%, so
  the lever is REDUCING shared (raise occupancy), not widening the M-tile. The proven 64×64 m64 kernel
  (`gemm_q4k_tc_f16a`) uses 24 KB static shared, DOMINATED by the 16 KB float `Cs[BT*BM]` output stage (BM·BT·4) →
  caps occupancy at 2 blocks/SM. This cycle adds an ADDITIVE kernel `gemm_q4k_tc_f16a_lowsmem` (kernels.cu): identical
  Q4_K dequant + wmma 16×16×16 fp16 schedule + fp32 accumulate + pre-converted f16-A read once, but the Cs output
  stage REUSES the (now-dead) Ws+As shared region after the K-loop and the 64×64 tile is stored to Y in TWO PHASES of
  8 fragments each (phase 1 = c0 → M-tile rows 0..31, phase 2 = c1 → rows 32..63), each phase needing only an 8 KB
  `float[BT*32]` tile. Total static shared = max(Ws+As = 8 KB during K-loop, Cs = 8 KB after) = **8 KB** → ~3× the m64
  occupancy (thread-limited to 8 blocks/SM at 256 thr). Each Y element is written exactly once across the two phases;
  the phase split only REORDERS writes, not values, and the wmma math/Q4_K unpack are IDENTICAL → output BYTE-FOR-BYTE
  the m64 kernel's (proper `__syncthreads` fence the smem reuse: the K-loop's trailing sync fences Ws/As reads before
  phase-1 store; a sync between each phase's store↔read↔next-store). DECISION: since byte-identical AND faster, FLIPPED
  the default TC Q4_K path to lowsmem (m128 still opt-in `ZINC_BATCHED_TC_M128`; new A/B kill-switch
  `ZINC_BATCHED_TC_M64` forces the prior 24 KB m64 kernel = cycle 12 behavior). ADDITIVE everywhere (1 kernel + 1
  pipeline + 1 bool + a dispatch reorder); `ZINC_BATCHED_TC` itself is off by default → production unchanged. The Q6_K
  TC kernel (`gemm_q6k_tc_f16a`, idx 2 ffn_down, ~1/7 of dense GEMM) is left on 24 KB — out of scope this cycle.
  Built clean on the 4090 box (fresh `.zig-cache` first pass + warm rebuild, `zig build cuda-dbg -Dbackend=cuda
-Dshaders=false` EXIT=0, bin md5 82958578 ≠ cycle 14's a1b7e365; NVRTC compiled the new kernel at runtime). GATE —
  FULLY CLEARED:
  - BYTE-IDENTITY (lowsmem's core claim): direct ABBA, ONE binary, lowsmem vs m64 — GEN_IDS IDENTICAL on the seq
    250-tok prompt (collapsed 250,250,…) AND a VARIED non-collapsing prompt (GEN 25994,240017,…) across 2 runs each.
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (new DEFAULT = lowsmem TC path, 4090): **5/5 PASS** —
    qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME documented near-tie
    (free-run 2/12, teacher-forced 11/12) → no new divergence. Also ran with `ZINC_BATCHED_TC_LOWSMEM`-style
    explicit-on before the flip → identical 5/5.
  - PERF (ABBA x2, gemma-31b 250-tok, 4090, ONE binary): run 1 (lowsmem opt-in vs m64 default) lowsmem **174.28**
    vs m64 **156.15 = +11.6%**; run 2 (after flipping default: lowsmem-default vs `ZINC_BATCHED_TC_M64` kill-switch)
    lowsmem **166.14** vs m64 **152.60 = +8.9%**. Both ABBA, no run overlap (A∈[151.7,159.2], B∈[165.8,176.3]) →
    not boost noise (clears the box's ±1% floor, unlike cycle 13's +1.8%). The m128 hypothesis (cut A read) was
    occupancy-NEGATIVE; the lowsmem hypothesis (cut Cs shared → raise occupancy) is the correct lever — cycle 14's
    diagnosis VINDICATED. Committed to perf/e24-batched-prefill, pushed (NOT main; origin/main == this branch at
    c409e280 → no rebase needed). NEXT (cycle 16): apply the same lowsmem two-phase-Cs trick to `gemm_q6k_tc_f16a`
    (complete the occupancy win on the dense ffn_down Q6_K GEMM); OR the fp16 WEIGHT cache (dequant Q4_K/Q6_K → fp16
    ONCE per layer, kills the per-GEMM dequant = the other half of GEMM traffic, no occupancy cost); OR token-GROUPED
    routed experts (gather by expert, memory-traffic win beyond launch batching, harder byte-identity).
- **2026-06-13 — Cycle 16: 8 KB-shared (lowsmem) Q6_K TC GEMM — byte-identical but PERF-NEUTRAL (in-noise); kept OPT-IN, default unchanged.**
  Took cycle 15's NEXT item: apply the proven two-phase-Cs occupancy trick (24 KB→8 KB static shared, 2→~8 blocks/SM)
  to the dense gemma-31b ffn_down **Q6_K** TC GEMM (idx 2), the one dense quant type still on the 24 KB m64 kernel after
  cycle 15 lowered the Q4_K path. Added an ADDITIVE kernel `gemm_q6k_tc_f16a_lowsmem` (kernels.cu): the cycle-13 Q6_K
  dequant (copied VERBATIM from `gemm_q6k_tc_f16a`/`gemm_q6k_tiled_v2`) wrapped in the cycle-15 lowsmem schedule — `half
Ws[64*32]`+`half As[32*64]` (8 KB) during the K-loop, then the SAME smem reused as a `float Cs[64*32]` (8 KB) two-phase
  output store (phase 1 = c0 → M-tile rows 0..31, phase 2 = c1 → rows 32..63), each Y elem written once → output
  byte-for-byte `gemm_q6k_tc_f16a`'s (phases reorder writes only; wmma math + Q6_K unpack identical; syncs fence the smem
  reuse). New pipeline + a bool toggle. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda
-Dshaders=false` EXIT=0, bin md5 27d453a7; NVRTC compiled the new kernel at runtime).
  GATE — correctness CLEARED, perf NEUTRAL:
  - BYTE-IDENTITY (the kernel's core claim): direct A/B, ONE binary, lowsmem-Q6 vs m64-Q6 (`ZINC_BATCHED_TC_Q6_M64`
    during the experiment) — GEN_IDS IDENTICAL on the collapsed 250-tok prompt (250,250,…) AND a varied prompt
    (27750,27750,…); after the default flip, re-confirmed default(m64) vs `ZINC_BATCHED_TC_Q6_LOWSMEM=1` GEN_IDS identical.
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (lowsmem-Q6 path direct vs llama.cpp, 4090): **5/5 PASS** —
    qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME documented near-tie (free-run 2/12,
    teacher-forced 11/12) → no new divergence from the Q6_K lowsmem path.
  - PERF — IN-NOISE (the honest outcome): ABBA x2 (gemma-31b 250-tok, 4090, ONE binary) ×2 independent runs:
    run 1 m64 174.09 vs lowsmem 165.17 (**−5.1%**); run 2 m64 169.76 vs lowsmem 168.02 (**−1.0%**). Both have FULLY
    OVERLAPPING ranges (run1 L∈[156.9,173.3]/M∈[166.8,181.3]; run2 L∈[163.4,172.1]/M∈[160.7,177.1]) and the two runs
    DISAGREE in magnitude → boost noise around zero, NOT a real regression. ROOT CAUSE (predicted): the Q6_K GEMM is only
    ~1/7 of the dense work (one ffn_down vs six Q4_K projections/gate/up per layer), so even a real ~+10% occupancy win on
    the Q6_K kernel itself would be ~+1.4% end-to-end — below the box's ±10% boost floor. Same class as cycle 13's Q6_K-on-TC
    (+1.8%, in-noise). UNLIKE cycle 15's Q4_K lowsmem (+9-12%, the bulk share → measurable), the Q6_K share is too small.
    DECISION: since the effort's bar for a DEFAULT FLIP is a MEASURED faster increment (which this is not), kept the proven
    cycle-13 m64 Q6_K kernel as the DEFAULT and made the byte-identical lowsmem-Q6 kernel an ADDITIVE OPT-IN documented
    experiment (env `ZINC_BATCHED_TC_Q6_LOWSMEM`), mirroring cycle 14's handling of the negative m128 result — so a future
    cycle won't re-attempt it and the default TC path stays byte-for-byte cycle 15 → cannot regress. ADDITIVE everywhere
    (1 kernel + 1 pipeline + 1 bool + a dispatch branch); `ZINC_BATCHED_TC` itself off by default → production unchanged.
    Committed to perf/e24-batched-prefill, pushed (NOT main; origin/main == c409e280, branch ahead at cycle 15 → no rebase).
    NEXT (cycle 17): the remaining big memory-traffic lever is the **fp16 WEIGHT cache** — dequant Q4_K/Q6_K → fp16 ONCE per
    layer so the TC GEMMs read pre-dequant'd half weights (kills the per-GEMM dequant = the OTHER half of the GEMM traffic,
    and unlike the Q6_K occupancy trick it attacks the dominant Q4_K share AND costs no occupancy); OR token-GROUPED routed
    experts (gather by expert, memory-traffic win beyond launch batching, harder byte-identity).
- **2026-06-13 — Cycle 17: wider 128×64 M-tile + low-shared two-phase Cs Q4_K TC GEMM — byte-identical but NEGATIVE (−5–7%); kept OPT-IN, default unchanged.**
  Synthesized cycle 14 (wider 128×64 M-tile → grid.x = M/128 HALVES the dominant f16-A re-read) with cycle 15 (low-shared
  two-phase Cs → high occupancy) to test whether m128's halved-A-read win could be recovered now that its occupancy killer
  is removable. Cycle 14's plain m128 was −11.8% ONLY because its 44 KB static shared capped occupancy at 1 block/SM; the
  hypothesis: a 128×64 tile that writes its output in FOUR phases reusing the dead Ws+As region needs only 12 KB static
  shared (vs 44 KB) → ~6 blocks/SM (the SAME as the m64 lowsmem default), so the halved A read should finally pay off.
  Added an ADDITIVE kernel `gemm_q4k_tc_f16a_m128_lowsmem` (kernels.cu): `gemm_q4k_tc_f16a_m128` in every wmma respect
  (same Q4_K dequant, same 16×16×16 fp16 schedule, fp32 accumulate, BM=128, 8 warps × 4 frags = 32 = all 8×4 output blocks,
  even/odd M-block grouping) EXCEPT the 128×64 output is NOT held in a 32 KB float `Cs[BT*BM]` tile; instead the Cs stage
  REUSES the (now-dead) `half Ws[128*32]`(8 KB)+`half As[32*64]`(4 KB) region and writes Y in FOUR phases of two 16-row
  M-blocks each (phase p = rows 32p..32p+31 = even-group frag c_p ∪ odd-group frag c_p), each phase an 8 KB `float[BT*32]`
  tile → total static shared = max(12 KB K-loop, 8 KB Cs) = **12 KB** (vs m128's 44 KB). Each Y element written exactly once;
  the four phases REORDER writes only; wmma math + Q4_K unpack are IDENTICAL → output BYTE-FOR-BYTE the m128/m64/lowsmem
  kernels' (syncs fence the smem reuse). Opt-in via `ZINC_BATCHED_TC_M128_LOWSMEM`; default TC path byte-for-byte cycle 15.
  ADDITIVE everywhere (1 kernel + 1 pipeline + 1 bool + a dispatch branch); `ZINC_BATCHED_TC` itself off by default →
  production unchanged. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false`
  EXIT=0, bin md5 cc8b175a ≠ cycle 16's 27d453a7; NVRTC compiled the new kernel at runtime). GATE — correctness CLEARED,
  perf NEGATIVE:
  - BYTE-IDENTITY (the kernel's core claim): direct A/B, ONE binary, m128_lowsmem vs lowsmem default — GEN_IDS IDENTICAL
    on the collapsed 250-tok prompt (250,250,…) AND a VARIED non-collapsing prompt ((i*73+11)%251 → GEN 240017,…) across
    runs each.
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 ZINC_BATCHED_TC_M128_LOWSMEM=1 validate_catalog` (m128_lowsmem path DIRECT vs
    llama.cpp, 4090): **5/5 PASS** — qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME
    documented near-tie (free-run 2/12, teacher-forced 11/12) → no new divergence from the wider-tile kernel.
  - PERF — NEGATIVE (the honest outcome): ABBA x2 ×2 independent runs (gemma-31b 250-tok, 4090, ONE binary): run 1
    A=m128_lowsmem **156.8** vs B=lowsmem **166.1 (−5.6%)**; run 2 A **155.3** vs B **167.6 (−7.3%)**. Both runs AGREE in
    direction AND magnitude (ranges barely/non-overlapping in run 2: A∈[152.3,158.7], B∈[157.9,174.5]) → a REAL mild
    regression, NOT boost noise. ROOT CAUSE (the wider-tile hypothesis FALSIFIED a SECOND way): the 128×64 tile needs 4
    fp32 accumulator fragments per warp (c0–c3) vs the 64×64 lowsmem default's 2 (c0,c1) → ~2× the accumulator REGISTERS,
    which drops register-limited occupancy below the lowsmem default's — re-introducing the SAME latency-hiding loss that
    killed cycle 14's m128 (which lost it via 44 KB SHARED). Freeing shared with the two-phase Cs did NOT help because the
    binding occupancy limiter for the wider tile is REGISTERS, not shared; and cycle 12 already cut the activation read to
    fp16, so the halved-A-read saving the wider tile buys is too small to cover the lost occupancy. So widening the M-tile
    is occupancy-negative on this box via BOTH routes (shared in c14, registers here) — the 64×64 lowsmem tile is the sweet spot.
    DECISION: since the effort's bar for a DEFAULT FLIP is a MEASURED faster increment (this is measurably SLOWER), kept the
    proven cycle-15 lowsmem kernel as the DEFAULT and made the byte-identical m128_lowsmem kernel an ADDITIVE OPT-IN documented
    experiment (`ZINC_BATCHED_TC_M128_LOWSMEM`), mirroring cycle 14/16's handling of negative/neutral results — so a future
    cycle won't re-attempt M-tile widening and the default TC path stays byte-for-byte cycle 15 → cannot regress. Committed to
    perf/e24-batched-prefill, pushed (NOT main; origin/main == c409e280, branch ahead at cycle 16 → no rebase). NOTE on the
    perennial NEXT item, the **fp16 WEIGHT cache**: a traffic analysis this cycle argues it is NET-NEGATIVE for prefill TC and
    should be deprioritized — each dense weight is used in exactly ONE GEMM per layer, so a "cache" gives NO cross-GEMM reuse;
    the only redundancy is the in-GEMM dequant repeated grid.y≈4× (T/64), and replacing it with a pre-dequant'd fp16 weight
    read grid.y× at 2 B/elem (vs Q4_K's ~0.56 B/elem) ROUGHLY DOUBLES total GEMM traffic on this already memory-bound kernel
    (weight traffic 49→~232 MB for a 4096×5376 proj), almost certainly outweighing the saved dequant ALU. NEXT (cycle 18):
    token-GROUPED routed experts (gather T tokens by expert so each active expert's Q4_K/Q5_1 weight is read once per group not
    once per token — a genuine MEMORY-TRAFFIC win beyond launch batching, unlike the M-tile/weight-cache levers; needs a
    gather/scatter from b.router_table and a variable-group GEMM, harder for byte-identity but the real remaining lever); OR
    keep activations fp16 ACROSS the layer (have the norm/GeGLU producers emit fp16 so the per-GEMM f32→fp16 recast launch is
    dropped — touches shared kernels, less additive).
- **2026-06-13 — Cycle 18: token-GROUPED routed-expert matvecs for gemma-26b MoE prefill (Phase 2 cycle 6) — byte-identical + token-correct, PERF IN-NOISE → kept OPT-IN.**
  The cycle-8 `dmmv_q4k/q5_1_experts_batched` kernels launch grid.y = token, so blocks reading the SAME expert weight are
  scattered across grid.y (token t's slot e and token t+1's slot e route to different experts) → no cross-token L2 reuse of
  the ~2.2 MB/expert Q4_K gate_up + Q5_1 down weights. This cycle adds the GROUPED variant: a single-block counting sort
  (`build_expert_order`) sorts the T·n_used (token,slot) work-items by expert id into `b.expert_order` (packed token<<16|slot),
  then the grouped matvecs launch grid = (M output rows, P = T·n_used work-items) and read `order[blockIdx.y]` for their
  (token,slot) — so consecutive grid.y work-items share the same expert weight, keeping it L2-resident across all the tokens
  routed to it (the genuine memory-traffic lever the prior NEXT-notes flagged, beyond cycle-8's launch batching). ADDITIVE
  (decodeStep/moeFfnBlock/the cycle-8 _batched path/dense path all untouched):
  - 3 ADDITIVE kernels (kernels.cu): `dmmv_q4k_experts_grouped` / `dmmv_q5_1_experts_grouped` — the per-block dequant +
    `zinc_block_reduce_sum` + the y write location (token t's slice, slot e, row) are BYTE-FOR-BYTE the cycle-8 `_batched`
    kernels'; only WHICH block computes which (token,slot,row) changes (via `order[]`), and every output is still computed
    exactly once → byte-identical result regardless of the order permutation. Plus `build_expert_order` (single-block
    counting sort: zero shared counts → histogram of expert ids → exclusive prefix-sum → scatter; n_experts ≤ 256, intra-bin
    order race-irrelevant since each work-item maps to a distinct independently-computed output). Reuses `ExpertsBatchPush`.
  - `dmmv_q4k_experts_grouped`/`dmmv_q5_1_experts_grouped`/`build_expert_order` pipelines + `BuildOrderPush` + `BatchScratch.expert_order`
    ([T·n_used] u32, size 1 on dense) + `moeRoutedExpertsGrouped(L,T,b)` (= `moeRoutedExpertsBatched` op-for-op with the same
    push params/buffers/block dims, but build_expert_order → grouped matvecs over grid (M, P)). prefillBatched MoE branch
    routes through it only when `use_grouped` (`ZINC_BATCHED_EXPERTS_GROUPED`), else the proven cycle-8 `moeRoutedExpertsBatched`.
  - Opt-in bool `use_grouped` + scripts (validate_catalog/prefill_catalog) pass `ZINC_BATCHED_EXPERTS_GROUPED=1` through additively.
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0, NVRTC compiled
    the 3 new kernels; binary md5 a51357e7, a fresh build). GATE — CORRECTNESS FULLY CLEARED:
  - Byte-identity (the core claim): direct A/B gemma-26b MoE, VARIED non-collapsed prompt ((i*73+11)%251+5), grouped vs
    cycle-8 batched — GEN_IDS BYTE-IDENTICAL at **T=60** (161,236909,164086,…) AND **T=200** (116,195,173310,161,236909,…). ✓
  - `ZINC_BATCHED=1 ZINC_BATCHED_EXPERTS_GROUPED=1 validate_catalog` (DIRECT vs llama.cpp, 4090): **5/5 PASS** — gemma4-26b
    MoE GROUPED free-runs **12/12** directly vs llama.cpp (the grouped prefill path is token-correct); gemma4-31b the
    documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged — dense path has no grouped); qwen 12/12 fallback.
  - PERF — IN-NOISE (the honest outcome): 2 ABBA x2 runs (gemma-26b 250-tok, 4090, A=cycle-8 batched, B=grouped). Run 1
    A mean 169.3 / B mean 202.1 (**+19%**), 3/4 pairs favor B; Run 2 A mean 187.7 / B mean 187.5 (**~0%**). The two runs
    DISAGREE in magnitude and the per-config variance is enormous (single-config values swing 145→236 tok/s, ~60%) — boost
    noise dominates. Combined mean A 178.5 / B 194.8 (+9%) leans positive but sits well under the box's boost floor today.
    Same class as cycle 13 (Q6-on-TC +1.8%) and cycle 16 (Q6 lowsmem) — byte-identical/token-correct but perf below the
    measurement floor. Likely reason the L2-reuse win is small: at T=250 the GPU runs thousands of (M-row × work-item)
    blocks concurrently across 128 SMs, so grid.y ordering only weakly controls L2 residency timing.
    DECISION: the effort's bar for a DEFAULT FLIP is a MEASURED faster increment; this is in-noise, so kept the proven cycle-8
    `_batched` matvecs as the DEFAULT and made the byte-identical grouped path an ADDITIVE OPT-IN (`ZINC_BATCHED_EXPERTS_GROUPED`),
    mirroring cycles 13/14/16/17's handling of neutral/negative results → the default MoE prefill path stays byte-for-byte cycle
    10 → cannot regress. The counting-sort `build_expert_order` + `b.expert_order` device foundation is reusable for a future
    expert-major restructure. Committed to perf/e24-batched-prefill, pushed (NOT main). NEXT (cycle 19): keep activations fp16
    ACROSS the layer (norm/GeGLU producers emit fp16 → drop the per-GEMM f32→fp16 recast launch on the TC path — touches shared
    kernels, less additive), or a larger-T sweep to see if the grouped L2 win emerges above the boost floor at T≫250.
- **2026-06-13 — Cycle 19: shared-A f32→f16 activation recast across same-input GEMMs on the TC path — byte-identical + token-correct, PERF IN-NOISE → kept OPT-IN.**
  Took the cycle-18 NEXT item's lighter, fully-ADDITIVE half: on the fp16 tensor-core path (`ZINC_BATCHED_TC`), each dense
  GEMM independently pre-converts its f32 activation A → fp16 (`f32_to_f16`) into `act_f16` before the wmma kernel (cycle 12).
  But several GEMMs in a layer read the SAME input: attn **Q/K/V all read `b.norm`**; FFN **gate+up both read `b.ffn_norm`**;
  the gemma-26b shared-expert **gate+up both read `b.ffn_norm`**. So the recast for the 2nd/3rd GEMM of each group is
  REDUNDANT — `act_f16` already holds that exact downcast. This cycle shares it: behind a new opt-in `ZINC_BATCHED_TC_SHAREA`,
  only the FIRST GEMM of each same-input group runs `f32_to_f16`; the rest pass `a_preconv=true` and reuse `act_f16`. ADDITIVE
  (decodeStep/per-token path/the default per-GEMM-recast TC path all untouched):
  - New `gemmDispatchA(...,a_preconv)` wraps the existing `gemmDispatch` (which now calls it with `a_preconv=false`); when
    `a_preconv && use_tc_sharea` the Q4_K and Q6_K TC branches SKIP the `f32_to_f16` launch and the kernel reads the existing
    `act_f16`. Byte-identical: x is unchanged between the group's GEMMs, nothing writes `act_f16` in between, and the skipped
    recast would have produced the bit-for-bit-identical `__float2half` bits → the staged half input is identical. Removes 2
    recast launches/attn layer + 1/FFN + 1/shared-expert layer.
  - `attentionLayerBatched`: K/V projections now `gemmDispatchA(..., true)` (reuse Q's recast of `b.norm`). `ffnBlockBatched`
    - `sharedExpertBatched`: up projection `gemmDispatchA(..., true)` (reuse gate's recast of `b.ffn_norm`). Opt-in bool
      `use_tc_sharea` read once from `ZINC_BATCHED_TC_SHAREA` in `prefillBatched`. Scripts (validate_catalog/prefill_catalog)
      pass the toggle through additively; new diagnostic `scripts/sharea_sweep.sh` (ABBA on/off, asserts GEN_IDS identity +
      reports tok/s). Off (default) → each GEMM recasts independently = byte-for-byte cycle 12.
      Built clean on the 4090 box (`zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0; binary md5 63495b4c embeds the
      sharea env handling + `gemmDispatchA`, ≠ cycle 18's a51357e7). GATE — CORRECTNESS FULLY CLEARED, perf in-noise:
  - Byte-identity (the core claim): `scripts/sharea_sweep.sh` (gemma-31b dense, T=250, ABBA x2, 4090) — GEN_IDS **identical**
    across all on/off runs → shared-A is byte-for-byte the per-GEMM recast, as designed. ✓
  - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 ZINC_BATCHED_TC_SHAREA=1 validate_catalog` (DIRECT vs llama.cpp, 4090): **5/5 PASS** —
    gemma4-26b MoE (sharea on its dense-attn TC GEMMs) free-runs **12/12**; gemma4-31b the SAME documented near-tie (free-run
    2/12, teacher-forced 11/12) → the shared recast introduced NO new divergence; qwen 12/12 (per-token fallback intact). ✓
  - PERF — IN-NOISE (the honest outcome): sharea_sweep ABBA x2 (gemma-31b 250-tok, 4090): no-sharea **166.35** vs sharea
    **170.76** t/s = **+2.6% nominal**, below the box's ±10% boost floor (the script's rounded "gain" column read 0%). Same
    class as cycles 13/16/18 — byte-identical/token-correct but perf below the measurement floor. Expected: `f32_to_f16` is a
    cheap element-wise launch and the recast read was already fp16-halved (cycle 12), so removing 2–4 of them per layer saves
    far less than the GEMM cost — the saving is real but sub-floor.
    DECISION: the effort's bar for a DEFAULT FLIP is a MEASURED faster increment; this is in-noise, so kept the proven cycle-12
    per-GEMM recast as the DEFAULT and made the byte-identical shared-A path an ADDITIVE OPT-IN (`ZINC_BATCHED_TC_SHAREA`),
    mirroring cycles 13/14/16/17/18's handling of neutral/negative results → the default TC path stays byte-for-byte cycle 15 →
    cannot regress. Committed to perf/e24-batched-prefill, pushed (NOT main; origin/main unchanged, branch 0 behind → no rebase).
    NEXT (cycle 20): the heavier half of the activation-fp16 lever — keep activations fp16 ACROSS the layer (have the norm/GeGLU
    producers EMIT fp16 directly so the per-GEMM f32→fp16 recast launch is dropped ENTIRELY, not just shared within a group;
    touches the shared rms_norm/geglu kernels so less additive, needs its own byte/tolerance gate); OR a larger-T sweep
    (`scripts/grouped_sweep.sh`, T≫250) to see if cycle 18's grouped-expert L2 win clears the boost floor.
- **2026-06-13 — Cycle 20: LARGE-T characterization sweep — KILLED the grouped-expert lever (all T) + UNCOVERED the real prefill win: the TC dense path is ~+20-30%, not the logged "+6%".**
  A DIAGNOSTIC cycle (no compute change): built HEAD clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg
-Dbackend=cuda -Dshaders=false` EXIT=0 + ran, bin md5 bb6e77c1), then ran the two open large-T sweeps from cycle 19's
  NEXT to decide which opt-in path (if any) clears the box's ±10% boost floor and merits a default flip.
  1. **Grouped routed experts (`scripts/grouped_sweep.sh`, gemma-26b MoE, ABBA x2, T={250,750,1500}) — DEAD at all T.**
     GEN_IDS BYTE-IDENTICAL to the cycle-8 `_batched` default at every T (incl. **T=1500** → the batched MoE prefill has
     NO large-T capacity/correctness bug — a useful verification), but perf **−11.4% (T=250), −16.4% (T=750), +1.4%
     (T=1500)** → the cycle-18 L2-reuse hypothesis is now FALSIFIED at large T too (the counting-sort + scattered output
     writes cost more than any L2 residency win; at high block concurrency grid.y ordering barely controls L2 timing).
     The grouped lever is conclusively dead — stays opt-in, cycle-8 `_batched` remains the proven default.
  2. **fp16 tensor-core dense GEMM (new `scripts/tc_sweep.sh`, gemma-31b dense, f32 vs `ZINC_BATCHED_TC`) — the REAL,
     above-floor win, badly understated in the log.** ABBA x2: **T=750 f32 138.1 → TC 178.8 = +29.5%** (ranges
     NON-overlapping A∈[135.7,140.6] B∈[170.6,187.0]); **T=1500 f32 143.9 → TC 171.7 = +19.3%** (NON-overlapping
     A∈[141.9,145.9] B∈[163.9,179.5]); a separate ABBA x1 even showed **T=250 +23.9%** (f32 134.2 → TC 166.3). GEN_IDS
     matched f32 at every T (no large-T divergence). So the TC dense path is a ROBUST ~+20-30% prefill win across
     T=250-1500 — NOT the "+6%, memory-bound footnote" the cycle-11 log records. Root cause of the discrepancy: cycle
     11's +6% measured the ORIGINAL m64 TC kernel; cycles 12 (fp16-A) + 15 (lowsmem, +9-12%) substantially improved the
     default TC path AFTERWARD, but the headline gain was never re-measured. The win has been ~4-5× larger than logged
     for several cycles. (On the gemma-26b MoE model TC is only +6.5%/+16.7% with heavily OVERLAPPING ranges — TC there
     touches only the smaller dense-attention GEMMs while the MoE FFN + boost noise dominate; dense is the clean signal.)
     TC stays OPT-IN: it is fp16 → token-correct within tolerance (this cycle re-confirmed `ZINC_BATCHED=1 ZINC_BATCHED_TC=1
validate_catalog` **5/5** on the fresh build — qwen 12/12, gemma-26b 12/12, gemma-31b the documented near-tie), NOT
     byte-identical to f32, so it CANNOT be the strict-byte-identity merge default. But it IS the recommended FAST prefill
     path for realistic (long) prompts. ADDITIVE: new `scripts/tc_sweep.sh` (reproducible T-sweep, mirrors grouped/sharea_sweep;
     records the result inline); no `.zig`/`.cu` change → the compute path is byte-for-byte cycle 19. Committed to
     perf/e24-batched-prefill, pushed (NOT main; origin/main unchanged, branch 0 behind → no rebase). NEXT (cycle 21): since
     TC is the dominant lever, push it further — the heavier activation-fp16 half (norm/GeGLU producers EMIT fp16 so the TC
     path drops the per-GEMM f32→fp16 recast ENTIRELY across the layer; touches shared kernels, own tolerance gate) is now
     well-motivated (it shaves the TC path's remaining f32 overhead at large T); OR a flash-style batched attention kernel —
     the grouped sweep showed dense/MoE throughput PEAKS at T≈750 and DROPS by T=1500, the signature of O(T²) attention
     overtaking the O(T) GEMMs, so the unoptimized 3-pass `gemma_attention_batched` is the next large-T bottleneck.
- **2026-06-13 — Cycle 21: fp16-EMITTING norm/GeGLU producers for the TC path (the heavier activation-fp16 half) — token-correct, PERF IN-NOISE, NOT strictly byte-identical (ULP near-tie) → kept OPT-IN.**
  Took cycle 20's well-motivated NEXT: cycle 12 pre-converts each dense GEMM's f32 activation A→fp16 via a separate
  `f32_to_f16` launch into `act_f16`; cycle 19 deduped that recast within same-input GEMM groups (sharea). This cycle
  removes the recast ENTIRELY on the dense TC path by having the activation PRODUCERS emit fp16 directly into `act_f16`:
  the pre-attn / pre-FFN rms_norm and the FFN GeGLU. ADDITIVE (decodeStep/per-token path/the default per-GEMM-recast TC
  path all untouched; default toggle-off path byte-for-byte cycle 19):
  - 2 ADDITIVE kernels (kernels.cu): `rms_norm_f16` (verbatim `rms_norm` — same `zinc_block_reduce_sum`, same
    `rinv`, same `w[i]*(x[i]*rinv)` — but `__float2half`-stores into a half output) and `geglu_f16` (verbatim `geglu`,
    `__float2half(gelu*up)`). Each computes the SAME f32 value its f32 twin writes, so `act_f16` should equal
    `f32_to_f16(twin)` — see the byte-identity caveat below.
  - `rms_norm_f16`/`geglu_f16` pipelines + opt-in bool `use_tc_normf16` (`ZINC_BATCHED_TC_NORMF16`, only meaningful with
    `ZINC_BATCHED_TC`). `attentionLayerBatched`: when set, dispatch `rms_norm_f16`→`act_f16` and call Q/K/V (all Q4_K)
    with `a_preconv=true` (skip their recast). `ffnBlockBatched`: `rms_norm_f16`→`act_f16` (gate/up `a_preconv`), and
    `geglu_f16`→`act_f16` for ffn_down ONLY when down takes the act_f16 TC path (Q4_K always, Q6_K iff `use_tc_q6`) —
    a `down_act_f16` guard so NORMF16+NOQ6 can't read a stale b.geglu. `gemmDispatchA`'s 2 recast-skip conditions
    extended to honor `use_tc_normf16` (`!(a_preconv and (use_tc_sharea or use_tc_normf16))`). The O projection (input
    is the f32 attention output, not a norm producer) still recasts. New `scripts/normf16_sweep.sh` (ABBA T-sweep) +
    `ZINC_BATCHED_TC_NORMF16` passthrough in prefill_catalog/validate_catalog.
    Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0, NVRTC
    compiled the 2 new kernels; bin md5 da0b59d5 ≠ cycle 20's bb6e77c1). GATE:
  - TOKEN-CORRECTNESS (the gate for the fp16 TC family): `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 ZINC_BATCHED_TC_NORMF16=1
validate_catalog` (4090): **5/5 PASS** — qwen 12/12 (per-token fallback); gemma4-31b teacher-forced 11/12 (the
    documented near-tie, unchanged); gemma4-26b teacher-forced **12/12** (every position's argmax matches llama.cpp →
    the normf16 prefill is token-correct).
  - BYTE-IDENTITY vs plain-TC — PARTIAL/NO (the honest finding): direct A/B (varied prompts) gemma-31b dense normf16 ==
    plain-TC GEN_IDS byte-identical (incl. 3 varied leading tokens before the prompt's degenerate tail), BUT gemma-26b
    MoE normf16 ≠ plain-TC — and 26b plain-TC is DETERMINISTIC (3/3 identical runs), so the difference is a REAL
    deterministic ULP-level numerical difference, not run noise. The MoE path reads neither `b.norm` nor `act_f16`
    (router/shared/experts all re-norm `b.hidden` themselves), so the diff originates in the attention norm itself:
    `rms_norm_f16` and `rms_norm`+`f32_to_f16` are mathematically identical but compile (separate NVRTC kernels) to a
    ~1-ULP-different fp16, enough to flip 26b's near-tie free-run argmaxes (the SAME near-tie class as the documented
    31b/26b TC near-ties) while leaving teacher-forced correctness intact. So normf16 has the SAME relationship to
    plain-TC that the TC path itself has to f32: token-correct, not bit-identical.
  - PERF — IN-NOISE (`scripts/normf16_sweep.sh`, gemma-31b dense, ABBA x2, 4090): T=250 163.5→168.2 (+2.9%), T=750
    172.0→174.6 (+1.5%), T=1500 175.6→167.4 (−4.7%) — all within the box's ±10% boost floor (script's rounded gain
    column reads 0% at every T); GEN_IDS identical on the 31b sweep at all T. The cycle-20 hypothesis that the recast
    (∝ T·K) would clear the floor at large T is FALSIFIED — at T=1500 it's nominally negative. Expected: `f32_to_f16`
    is a cheap element-wise launch already fp16-halved since cycle 12, so eliminating 5/6 recasts/layer saves far less
    than the GEMM cost. Same class as cycles 13/16/18/19.
    DECISION: the bar for a DEFAULT FLIP is a MEASURED faster increment; this is in-noise AND not strictly byte-identical to
    the current default TC path, so kept the proven cycle-12 per-GEMM recast as the DEFAULT and made the fp16-emitting
    producers an ADDITIVE OPT-IN (`ZINC_BATCHED_TC_NORMF16`, off by default → default TC + strict-byte f32 paths both
    byte-for-byte unchanged → cannot regress), mirroring cycles 13/14/16/17/18/19. NET: both halves of the activation-fp16
    lever are now EXHAUSTED (cycle 19 shared-A: byte-identical, +2.6% sub-floor; cycle 21 normf16: token-correct, in-noise) —
    the per-GEMM f32→fp16 recast is NOT a meaningful prefill cost on this box. Committed to perf/e24-batched-prefill, pushed
    (NOT main; origin/main unchanged at c409e280, branch 0 behind → no rebase). NEXT (cycle 22): the activation-fp16 branch
    is closed → pivot to the OTHER cycle-20 lever, a flash-style / query-tiled batched attention kernel: the grouped+TC
    sweeps both show throughput PEAKING at T≈750 and DROPPING by T=1500 = O(T²) attention overtaking the O(T) GEMMs, and the
    current 3-pass `gemma_attention_batched` re-reads K/V from global once per query block (O(T²) K/V traffic per head). A
    query-tiled kernel that loads a K/V tile once for a block of queries (own token-correctness gate, fp/online-softmax →
    not byte-identical) is the genuine remaining large-T win.
