# Effort 11 — Flatten the RDNA4 decode-with-context curve

## TL;DR  (2026-04-27, after the manual cycles 71/72/73 on flash_attn.comp)

**The user-visible problem.** On Qwen 3 8B Q4_K_M and Qwen 3.6 35B-A3B
through the chat UI, decode tok/s drops 4–5x as KV cache grows past
1k tokens. Concrete CLI measurements on Qwen 3 8B with the
flash_attn-staged binary (commits `6ece0a8` + `f68b7d7`):

| context length | decode tok/s | ms/token | shape vs L=5 |
|---:|---:|---:|---|
| 5 (empty) | 80.5 | 12.4 | baseline |
| 466 | 35.2 | 28.4 | 2.3× drop |
| 1162 | 17.0 | 58.8 | 4.7× drop |
| 2325 | **GPU hang** | — | — |

The user reports this is unacceptable and notes that other engines
(referenced: Claude Opus 4.6 inference) do not exhibit this slope.
Opus runs on H100-class hardware with continuous batching across
many users, so true flatness is unrealistic on consumer AMD; the
realistic target is **decode at L=1500 ≥ 60% of decode at L=5**, i.e.
~50 tok/s instead of the current 17. That cuts the user-visible
slope from 4.7× to ~1.6× across the same range.

**The GPU hang is the priority.** The test at L=2325 hung the GPU
(`amdgpu: ring comp_1.0.1 timeout`, `device wedged, but recovered
through reset`). Reproducible. The cause is not yet rooted out —
manual cycle 73's rescale-fusion was reverted and the hang persists
on the Q-stage + page_id cache combination. Need a clean-bisect
cycle to confirm whether one of the manual flash_attn changes is
guilty or whether the hang predates them.

## What landed in the manual session before this effort opened

Three commits on `flash_attn.comp` and `flash_attn_batched.comp`:

| commit | shader change | LDS added | validated |
|---|---|---:|---|
| `6ece0a8` | Stage Q in `shared float s_q[512]` once per workgroup; remove per-K-iter `q_data[]` re-reads from inner dot | +2 KB | bit-correct on Qwen 3 8B; decode L=5 went 54 → 82 tok/s |
| `f68b7d7` | Cache `page_ids[]` per block in `shared uint s_page_ids_block[256]`; both Phase 1 and Phase 4 read shared instead of global | +1 KB | bit-correct; prefill L=466 went 183 → 195 tok/s |
| `539b2aa` | Fuse Phase 4 rescale into V-accumulation; mirror Q-stage and page_id cache into `flash_attn_batched.comp` | none | bit-correct at L=466; **suspected guilty in the GPU hang at L=2325 but not confirmed** |

Total LDS budget on each shader is now ~6.5 KB per WG, well inside
RDNA4's 64 KB envelope.

The L=466 measurement loop ran 3 trials with the full stack:

```
Prefill: 187 / 193 / 200 tok/s   (median 193, +5–9% vs Q-stage-only)
Decode:  35.88 / 34.79 / 35.21 tok/s   (median 35, stable)
```

## Cycle 0 contract

This effort is scored on **decode tok/s at L=1500**, not on empty-context
decode and not on prefill. Specifically:

- Model: `Qwen3-8B-Q4_K_M.gguf` (the smaller dense model is the
  target because it pinpoints the attention-only bucket; the 35B
  hybrid muddies the picture with SSM and MoE contributions).
- Prompt: `/tmp/long2k.txt` on the RDNA node — a ~1162-token excerpt
  of a single English narrative. The agent may extend the prompt to
  reach exactly L=1500 if it wants, but must not change the prompt
  shape (English narrative, single conversational turn, no chat
  template overhead).
- Decode: `-n 32 --raw`. The 32-token cap keeps each cycle's run
  under 5 seconds while still amortizing per-token measurement
  noise.
- Build: `zig build -Doptimize=ReleaseFast`.
- Vulkan env: `RADV_PERFTEST=coop_matrix`.
- Node state: no other process holding the GPU.

Keep rules:

1. Coherence on `Qwen3-8B-Q4_K_M.gguf` AND `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
   must pass at L=1500 with greedy first-10-token agreement against the
   pre-change binary. **Do not** test at L ≥ 2300 until the cycle 1
   bisect resolves whether the manual session's commits are guilty of
   the GPU hang.
2. Decode at L=5 must stay within 5% of the pre-change rate (no
   regressing the empty-context fast path while chasing long-context).
3. Prefill on `Qwen3-8B-Q4_K_M.gguf` at L=466 must not regress more
   than 2% (the Q-stage + page_id cache wins are real and should not
   be undone by a future cycle).
4. No GPU hang at the chosen test length. Any cycle that produces
   `amdgpu: ring comp_*.*.* timeout` in dmesg fails the keep gate
   regardless of measured tok/s.

## Phase budget at L=1162  (cycle 0, the data point we have)

```
Per-token decode time at L=1162:  58.8 ms
Per-token decode time at L=5:     12.4 ms
Slope:                            ~40 µs per added KV token
```

For Qwen 3 8B (36 layers, head_dim=128, n_kv_heads=8) the bandwidth
floor on attention reads at L=1162 is:

```
  layer KV read = 2 * 1162 * 8 * 128 * 4 = 9.5 MB
  36 layers * 9.5 MB = 342 MB per decoded token
  342 MB / 576 GB/s   = 0.59 ms
```

So the pure attention bandwidth floor at L=1162 is ~0.6 ms.
Measured per-token at L=1162 is 58.8 ms. The L-dependent component
is ~46 ms (= 58.8 - 12.4). That is **77× the bandwidth floor**.

Conclusion: attention at L=1162 is not bandwidth-bound on this
hardware. Something in the per-block flash_attn loop is running
massively slower than its memory-traffic ceiling. The first cycle
of this effort must answer where that 77× factor goes.

## Hypotheses to prove or kill, in priority order

### H1 — Phase 4 V-accumulation is FMA-throughput-bound, not BW-bound

Phase 4 inner loop reads `s_scores[i]` (LDS), `v_data[v_base + d]`
(global), and computes a 2-MAC fused multiply-add. For
head_dim=128 on a 64-thread WG, each thread does
`(head_dim/64) * block_len = 2 * 256 = 512` MACs per block. At
L=1162 that's 5 blocks × 512 MAC = 2,560 MACs per WG per token. With
32 query heads dispatched in parallel and 64 CUs, 32 WGs fit
roughly twice on the device, so this should be ~1 µs of compute per
WG, ~30 µs across the layer's 32 query heads in 36 layers, ~1 ms
total per decoded token.

If H1 is true, the 46 ms is something else (dispatch overhead,
serialization, or a pathological cache miss pattern). Use Vulkan
timestamps via `recordTimestamp` around the flash_attn dispatch of
one layer at L=1162 to measure the kernel time directly. If that
kernel is 1.3 ms × 36 = 46 ms, attention is the bottleneck and the
bottleneck is *not* bandwidth.

### H2 — Per-token dispatch overhead grows with L because the page table is re-bound

Each decode iteration re-builds the descriptor sets for flash_attn,
binding the KV cache and page table buffers. As the page table
grows, the binding cost grows. Test by recording the time between
"start of decode iteration" and "first dispatch issued" via
`PerformanceCounters` or wallclock. If the gap scales with L, the
host-side binding is the culprit and the fix is push-descriptors or
pre-bound descriptor caches.

### H3 — Cache thrashing in V reads as L grows

Phase 4 reads V[i, d] for i ∈ [0, block_len) and d ∈ {tid, tid+64}.
For block_len=256 and head_dim=128, that's 32k V reads per WG per
block, scattered across 256 cache lines (V[i, :] is one row, 128 *
4 = 512 bytes = 4 cache lines per i). RDNA4's L1 is 16 KB per CU.
At long L, the L1 fills with V tiles and the working set spills to
L2. Profile `radv_perftest=cache` or vendor counters for L1/L2 hit
rate during decode at different L values.

### H4 — flash_attn is not the bottleneck; some other L-dependent op is

ZINC's `kv_cache_write.comp` runs once per decoded token. RoPE on
the new token runs once. Norms run once per layer. None of those
are L-dependent. But the K projection writes into the page-indexed
KV cache via the page table. If the host re-allocates pages or
re-uploads the page table per token, that scales with L on the host
side.

Confirm or kill by running with `ZINC_PREFILL_PROFILE=1` (it works
for decode too — check the path). The phase output will show which
bucket grows with L.

## Concrete single-cycle attacks  (in landing order, lowest risk first)

### Step 1 — Bisect the GPU hang at L=2325

Build three binaries (or use git checkout + zig rebuild):

- A: `5858b4f` (pre-flash_attn-fixes baseline). Test at L=2325.
- B: `6ece0a8` (Q-stage only). Test at L=2325.
- C: `f68b7d7` (Q-stage + page_id cache). Test at L=2325.
- D: `539b2aa` (current head — all three changes). Test at L=2325
  again to reproduce.

If A hangs, the bug predates this session and Effort 11 should
file a separate bug-fix cycle. If A passes and B/C/D hang, find the
guilty commit and either revert or fix.

### Step 2 — Add Vulkan timestamps around flash_attn

Wrap the flash_attn dispatch in one decode iteration with
`vkCmdWriteTimestamp` before/after, read back at end-of-step.
Attribute the L-dependent ms to a specific kernel. This is
infrastructure work; accept even if flat.

### Step 3 — Stage the V-tile cooperatively (the biggest hypothesis-aligned win)

llama.cpp's flash_attn co-loads K and V into shared memory once per
block, with all threads in the WG broadcast-reading the staged
copy. Our shader currently has each thread independently read its
own (i, d) pair from `v_data[]`. Total V reads per block per WG
are unchanged either way (each element read once), but staging
into LDS lets the inner loop hit shared memory at LDS latency
(~10 cycles) instead of L1/L2 (~50–200 cycles for misses).

LDS budget: a full V tile is `block_len * head_dim * 4 = 256 * 128 *
4 = 128 KB`, far over the 64 KB per-WG envelope. Chunk it: stage
`CHUNK * head_dim * 4` bytes at a time, process one chunk's worth
of (i, d) pairs, repeat. With CHUNK=32 the chunk is 16 KB; total
LDS would be ~22 KB after our existing 6.5 KB, which fits.

This is a structural change that needs careful validation. Land it
behind `ZINC_FA_V_TILE=1` first.

### Step 4 — Multi-Q-per-WG (the matmul reformulation)

The deeper fix from llama.cpp is processing Br Q rows per WG, so K
and V tile reads are amortized across Br queries. For decode this
is Br=1 by nature, but the same change unlocks much higher prefill
throughput later. This is multi-cycle and worth attacking only
after Steps 1–3 confirm the cheaper wins.

### Step 5 — Phase 3+4 register-resident cohesion (deferred from manual)

Keep `Pf[r] = exp(score - max)` in registers across the V-acc loop
instead of round-tripping through `s_scores[]`. This requires each
thread owning a (score_range, dim_range) tile rather than the
current disjoint partitions. Multi-cycle. Worth pricing out
separately once the cheaper wins are in.

## Known flat — do not re-attempt without new evidence

These were attempted in the manual session before Effort 11 opened
and either landed (kept) or were tried-and-reverted:

- **Q staging** — landed (`6ece0a8`). Don't revert.
- **page_ids per-block cache** — landed (`f68b7d7`). Don't revert.
- **rescale + V-acc fusion** — landed (`539b2aa`) but is the leading
  suspect for the L=2325 GPU hang. Step 1 of this effort
  re-evaluates whether to keep, fix, or revert.
- **Naive block-tile-staging into 64 KB LDS** — won't fit; only
  chunked V staging is viable (Step 3).

## Reference implementations

- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
  — the scalar fallback path (RDNA's path when coopmat is not used).
  Read the Q shared-memory staging loop at lines 44–90 (we already
  ported a simpler version of this) and the *fused exp+V loop* at
  lines 355–384. The latter is Step 5's reference.
- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
  — KHR cooperative_matrix variant. RDNA3+ supports KHR coopmat;
  R9700 may benefit. Compare the wmma tile shape to our scalar
  per-thread layout.
- `/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:2854-2928`
  — host-side flash_attn tuning parameters. Look for `block_rows`,
  `block_cols`, the `gqa_ratio` collapse, and the `limit_occupancy_shmem`
  RDNA-specific dummy LDS allocation.
- `/Users/stepan/Workspace/zinc/src/shaders/dmmv_q4k_moe_kpar.comp`
  — wave64 K-parallel pattern. The same idea applies if Step 4
  wants to subdivide a row across multiple threads with subgroupAdd
  reduction. Read the inner loop and the THREADS_PER_ROW math.

## Success criteria

This effort is succeeding when:

- Decode at L=1500 on `Qwen3-8B-Q4_K_M.gguf` is **≥ 50 tok/s**
  (target = 60% of empty-context decode rate).
- The L=2325 GPU hang is rooted out and either fixed or has a
  documented workaround.
- Coherence sweep is green at L=1500 across `Qwen3-8B-Q4_K_M.gguf`
  and `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.
- Empty-context decode and L=466 prefill both stay within 5% of
  the pre-effort rates.

## Non-goals

- Do not chase prefill numbers under this effort. Effort 6 owns
  prefill on hybrid models; this is a decode-curve effort.
- Do not weaken coherence to keep a faster-but-wrong path.
- Do not test at L ≥ 2300 until Step 1 resolves the GPU hang.
- Do not invent new shader families. Each cycle should change one
  thing in `flash_attn.comp` (or `flash_attn_batched.comp`) and
  measure the L=1500 decode delta.

## Likely files

- `loops/efforts/MULTI_HOUR_EFFORT_11_RDNA_DECODE_LONG_CONTEXT.md`
- `loops/optimize_perf.ts`
- `src/shaders/flash_attn.comp`
- `src/shaders/flash_attn_batched.comp`
- `src/compute/attention.zig`
- `src/compute/forward.zig` (only if Step 4 needs dispatch-side
  changes)
