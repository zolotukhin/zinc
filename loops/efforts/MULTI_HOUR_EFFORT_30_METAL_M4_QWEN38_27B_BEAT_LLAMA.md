# Effort 30 — Qwen 3.8 27B: beat latest llama.cpp on decode AND prefill (Metal, M4 Max) — STATUS: DECODE + SHORT PREFILL AHEAD, LONG PREFILL 7–10% BEHIND

Model `qwen38-27b-q4k-m` (Qwen3.8-27B Q4_K_M, `qwen35` dense hybrid: 64 layers,
48 DeltaNet SSM layers + 16 full-attention layers, dense FFN 5120→17408,
lm-head Q6_K 248320×5120). Per decoded token the GPU streams ~16.1 GB of
weights (17.1 GB file minus the embedding table and the unused NextN block).

Baseline (2026-09-02, main `04f4cd9b`, 48-token suite core prompt, 96 greedy
tokens, `zinc --chat` CLI): prefill 112 tok/s, decode **17.20 tok/s**
(58.1 ms/tok, ~255 GB/s effective). llama.cpp b67a17c1 (2026-09-03 master,
`llama-bench -fa 1` on the same GGUF): pp48 152 / pp128 210 / pp512 226 tok/s,
tg96 **21.20 tok/s** (47.2 ms/tok). The Aug-15 site data (llama-server b607)
recorded llama.cpp at 104 prefill / 23.4 decode; the prefill figure there is a
server-mode artifact — raw llama.cpp prefill is much faster than that.

## Method

- Byte accounting from the GGUF header (pure-python parser, no numpy) — dense
  FFN is 66% of decode bytes (gate/up Q4_K 100 MB/layer, down Q6_K 73 MB/layer),
  SSM projections 20%, lm-head 6%.
- `ZINC_METAL_KERNEL_TIMING=1 --profile`: destructive per-dispatch CPU-side
  sync probe. Absolute numbers are inflated by ~100 µs/dispatch, but the
  *ranking by total* is reliable and pointed straight at both wins below.
- `zinc-bench-metal-shapes --case <27B case>`: exact-shape serialized kernel
  GB/s.
- llama.cpp kernel parity: a 60-line ggml program (scratchpad
  `ggml_mv_bench.cpp`) timing `ggml_mul_mat` on the exact 27B shapes with
  12–24 distinct weight copies (defeats the SLC), concurrency disabled.
- The `--profile` per-slot line `decode async decode chunk slot N ... avg_gpu_ms
  ... eff_GiB/s` is the in-situ bandwidth truth for decode.

## Landed

- **Decode +40%: route the 27B's F32 SSM alpha/beta tails through
  `dmmv_f32_dual_small`.** `canUseQwenSsmF32AlphaBetaDual` was hard-wired to
  the 35B-A3B shape (32×2048), so the 27B's 48×5120 tails fell through to the
  generic SPIR-V-translated `dmmv_f32`: one thread per row walking K=5120
  serially in a single 64-thread threadgroup, ~0.5 ms per launch, 96 launches
  per token. The tails overlap the qkv+gate matvec but the conv/delta barrier
  waits for them, so ~0.35 ms per SSM layer was exposed. Widened the gate
  (`ZINC_METAL_QWEN27B_SSM_TAIL_F32_DUAL`, default on): **17.20 → 24.00 tok/s**
  (41.7 ms/tok), slot-0 effective bandwidth 223 → 347 GiB/s, output
  token-identical. Prefill unaffected (496-tok: 138.5 vs 139.0 tok/s). Commit
  `e5b66a3c`.
- After that fix decode is kernel-bound: slot 1 moves 1.69 GiB in 4.68 ms
  (388 GB/s), the same rate the big matvecs reach in isolation (Q4_K gate/up
  dual 387 GB/s, Q6_K down 414, Q6_K qkv 391, lm-head 456). Dispatch/barrier
  overhead is no longer measurable in the slot timings.
- Prefill: `ssm_delta_net_prefill_warp` (register-resident state, one
  simdgroup per state row, zero barriers per token — written 2026-06 for the
  35B, never dispatched) wired in behind `canUseSsmDeltaNetPrefillWarp`
  (`ZINC_METAL_SSM_DELTA_NET_PREFILL_WARP`, default on) for any
  `head_v_dim == d_state == 128` shape with ≤512 tokens per pass, with a
  27B-shape (48 heads / 16 groups) unit test against the CPU reference. The
  27B previously took the block kernel's generic scalar path: 2.87 ms/layer at
  48 tokens (138 ms of a 430 ms prefill) and 9.2 ms/layer at 165 tokens
  (144 calls ≈ 1.3 s of a 3.6 s 496-token prefill). The kernel had never been
  compiled in a production build: its signature mixed a `uint3`
  `threadgroup_position_in_grid` with a scalar `thread_position_in_threadgroup`,
  which MSL rejects — fixed by taking the thread index as `uint3`.
  **48-token prefill 435.6 → 302.4 ms (110 → 158.7 tok/s); 496-token prefill
  3557–3635 → 2358–2376 ms (137 → 209 tok/s).** Greedy output identical on both
  prompts; top-5 logits agree to 2–3 decimals (reduction-order noise).
  Decode unchanged (23.9 at 48 tokens; 21.0–21.6 after the 496-token prefill,
  both ways). With the scan gone the 496-token prefill is pure GEMM:
  `gemm_q4k_gate_up_swiglu` 192 × 5.43 ms (58.8 GFLOP/call → 10.8 TFLOPS),
  `gemm_q4k` 480 × 1.53 ms, `gemm_q6k` 192 × 2.15 ms, `gemm_q5k` 144 × 1.12 ms.

- Long prompts as one pass: `qwen35_dense27b_queued_prefill_max_tokens`
  192 → 512, scratch buffers and the layer-major materialization capacity
  (`qwen_ssm_projection_prefill_max_tokens`) 256 → 512 via
  `batched_prefill_scratch_tokens`; 9B/MoE single-pass ceilings stay at 256
  (`queuedPrefillSinglePassMaxTokens`). First attempt raised only the scratch
  size: a 496-token pass then materialized 256 tokens and token-major-
  replayed the other 240 (211k dispatches, 14.7 s vs 2.7 s) — the
  materialization capacity is the real limit. With both raised, paired A/B
  on a loaded box (env override 192 vs default 512): 292 tokens 1687 → 1632 ms
  (+3.4%), 496 tokens 2696 → 2576 ms (+4.7%), 48 tokens unchanged; greedy
  output identical on all three, top-5 logits agree to 2 decimals. Smaller
  than the ggml N-scaling suggested: ZINC's N48-tile GEMMs gain little from
  wider N, so the win is mostly the saved weight re-streams.

## Suite methodology fix: llama-server prompt cache

The first official run (`tools/performance_suite.mjs --target metal --models
qwen38-27b-q4k-m --llama-server <b67a17c1>`, artifact
`/tmp/zinc-perf/zinc-performance-metal-qwen38-20260902-232703.json`) recorded
llama.cpp at 23.4 tok/s prefill on the 48-token core prompt with
`prefill_tokens: 4` of 87: the local launcher never passed
`--no-cache-prompt` (the remote RDNA launcher does), so after the warmup
request llama-server reused the cached KV prefix and only the trailing
template tokens were prefetched. The Aug-15 publication had the same
artifact in milder form (45 of 87 tokens). ZINC's CLI runs always prefill
the whole prompt, so the published prefill ratio was not a comparison.
Fixed in `launchLocalLlamaServer` (mirrors the remote launch: `-b 4096 -ub
1024 --flash-attn on --no-cache-prompt`) with a guard test. Decode in that
run: ZINC 22.5 vs llama-server 18.7 tok/s (core), 20.0 vs 18.7
(context-medium), 19.7 vs 18.7 (context-long), 19.2 vs 18.8
(decode-extended) — server-mode llama.cpp decodes ~12% slower than its own
`llama-bench` (21.2), which is the suite's established method.

## Kernel parity vs llama.cpp b67a17c1 (exact 27B shapes, GB/s, serialized)

| shape | ZINC | ggml Metal |
|---|---:|---:|
| dense gate+up Q4_K (2×17408×5120) | 387 (dual kernel) | 434 (single 34816-row matvec) |
| dense down Q6_K (5120×17408) | 414 | 424 |
| ssm qkv Q6_K (10240×5120) | 391 | 410 |
| ssm out Q5_K (5120×6144) | — | 328 |
| lm-head Q6_K (248320×5120) | 456 | 375 |

f32/q8_0 matvecs stream at ~500 GB/s on this box (llama.cpp's own perf
cases), so the K-quant kernels on both sides sit ~15–20% under the DRAM
ceiling; the remaining decode headroom is inside the Q4_K/Q6_K dequant
kernels, not in dispatch structure.

## Dead ends (do not retry)

- Trusting the background-task exit code for `zig build` — the wrapper reports
  0 even when the build failed; check the log's own `exit=` line and the
  binary timestamp before measuring (cost one wasted A/B here).
- The `qwen27b_ssm_q4q4` microbench case reports 185 GB/s for the SSM
  Q4/Q4 pair; in-situ slot timings show the pair cannot be that slow (the
  rest of the slot would exceed the DRAM ceiling). Treat that case's byte
  accounting as suspect, not the kernel.

## Where it stands (2026-09-02 late, three commits on `perf/metal-qwen27b-decode`)

Interleaved same-conditions comparison, llama.cpp b67a17c1 `llama-bench -fa 1`
vs ZINC CLI, two rounds each within ±1% (box loaded by WindowServer + a Codex
service at ~30% CPU each, which costs both engines ~25% vs the quiet numbers
quoted above — the ratios are what to read):

| | llama.cpp | ZINC | ZINC/llama |
|---|---:|---:|---:|
| decode tg96 | 16.1–16.4 | 17.5 | +7–9% |
| prefill 48 tok | 132–133 | 138–139 | +4.5% |
| prefill ~300 tok | 194 | 180 | −7% |
| prefill ~500 tok | 213 | 192 | −10% |

Quiet-box reference from earlier in the session: decode 24.0 vs 21.2
(+13%), pp48 159 vs 152 (+4.5%), pp512 209 vs 226 (−8%, before the
single-pass change).

## Still open

- Prefill vs raw llama.cpp after the delta-net route: **159 vs 152 tok/s at
  48 tokens (ahead), 209 vs 226 at ~500 tokens (~8% behind)**. The remaining
  bucket is the N48-tile GEMM family. Two structural inefficiencies on the
  long prompt: (a) a 496-token prompt runs as three ~165-token passes
  (`qwen35_dense27b_queued_prefill_max_tokens = 192`), each re-streaming all
  16 GB of weights (~40 ms/pass) and padding N=165 to 4×48 tile columns (16%
  wasted MMA); (b) the fused gate/up GEMM sits at ~10.8 TFLOPS effective while
  llama.cpp's pp512 implies ~11.6 end-to-end — measure its mul_mm on the exact
  shape before touching the kernel.
- Long-prompt prefill (≥~250 tokens) is the one place llama.cpp is still
  ahead, by 7–10%. Its `mul_mm` gains ~17% going from N=165 to N=512 on the
  exact gate shape; ZINC's N48-tile `gemm_q4k_gate_up_swiglu` / `gemm_q4k` /
  `gemm_q6k` do not scale with N the same way (496-token single pass only
  bought +4.7%). The next lever is a wide-N GEMM tile (or a port of the
  current ggml mul_mm tiling) for N ≥ 128 — a kernel effort, not a routing
  one. Use `ggml_mm_bench <N>` (scratchpad) vs
  `zinc-bench-metal-shapes --case qwen27b_prefill_tail_hot` for the paired
  microbench.
- Official numbers: `tools/performance_suite.mjs --target metal --models
  qwen38-27b-q4k-m --llama-server /private/tmp/llama-latest-67a17c/build/bin/llama-server --skip-local-build`
  from this worktree, on a quiet box (the 23:27 run is invalid on the
  llama.cpp side for the prompt-cache reason above; the suite fix is in
  `c9256b04`).
