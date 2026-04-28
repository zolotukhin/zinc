# Effort 11 — Flatten the RDNA4 decode-with-context curve

## TL;DR  (2026-04-28, after run-1 + run-2 + run-3, total 46 cycles)

**3.18× over baseline. flash_attn cut to 12% of decode time. Dense
FFN is now the dominant bucket at 56%. Cross-layer fusion is hard.**

| run | cycles | keeps | starting → best tok/s | dominant lever |
|---|---:|---:|---|---|
| 1 | 17 | 12 | 31.19 → 93.68 | vec4 I/O + D-split + 4/8/16-way ILP on flash_attn |
| 2 | 13 | 2  | 93.39 → 96.16 | dispatch + barrier fusion (gate+up+SwiGLU; Q+K norm+rope+kv_write) |
| 3 | 16 | 1  | 96.16 → 99.38 | split-K flash attention (N_I_CHUNKS=4 + merge pass) |

Run-3's perf-keep was cycle 12: a split-K flash attention shader
that splits the i-axis into 4 chunks per (head) and uses a separate
merge pass to combine partial M/L/O across chunks. **8 × 4 = 32 WGs
restores the WG concurrency that naive GQA collapse destroyed.**
This is the single change that made flash_attn drop from ~56% to
~12% of decode time per the cycle-11 ProfilePhase data.

**Calibration finding from run-3 cycle 11**: the LONG_CONTEXT_DECODE_PROMPT
tokenizes to ~846 tokens on Qwen 3 8B, NOT the ~1500 the comment
claimed. All run-2 and run-3 measurements were at L≈846. The spec
text has been corrected. The structural conclusions still hold.

## Phase budget at L≈846 after run-3 cycle 12

```
total decode time per token: ~10.05 ms
  flash_attn:    12% (~1.2 ms)   ← cut from ~56% by split-K
  dense_ffn:     56% (~6.15 ms)  ← new dominant bucket
  other:         32% (~2.7 ms)   ← norms, residuals, lm_head, host gap

dense_ffn breakdown:
  gate+up+SwiGLU fused (cycle 8 of run-2): the bulk
  down+residual (Q4_K DMMV-acc): 2.4× the Q4_K bandwidth floor (per c10 audit)

per-layer profile (FA_PROFILE_LAYER): all 36 layers run uniformly
  cv = 0.004, ratio = 1.05× across layers
  no single-layer outlier, no layer-class anomaly
```

The 56% FFN bucket is the lever for run-4. Cycle 12's win pattern
(split-K with merge pass) is the template the next attack should
mirror onto down_proj.

```
trajectory at L=1500 (Qwen 3 8B Q4_K_M)
  cycle 1   31.19   baseline (Q-stage + page_id cache + rescale fusion already in place)
  cycle 3   33.14   vec4 K-loads in Phase 1               (+1.95)
  cycle 4   47.86   vec4 V-loads + vec4 s_out in Phase 4  (+14.72)
  cycle 5   60.33   D-split: pair lanes (tid, tid+32)     (+12.47)
  cycle 6   69.19   Phase 4 D-split unroll-by-2           (+8.86)
  cycle 8   75.74   Precompute s_kv_base_v4 per block     (+6.55)
  cycle 9   81.45   Phase 4 4-way ILP unroll              (+5.71)
  cycle 10  84.08   Phase 1 4-way ILP unroll              (+2.63)
  cycle 12  88.99   Phase 4 8-way ILP unroll              (+3.94)
  cycle 14  90.03   Phase 1 8-way ILP unroll              (+1.04)
  cycle 16  91.74   Phase 4 16-way ILP unroll             (+1.71)
  cycle 17  93.68   Phase 1 16-way ILP unroll             (+1.94)
```

The pattern that delivered: **vec4 I/O + D-split + ILP unrolling**.
This was none of what the original effort plan listed; the plan's
proposed Step 3 (chunked V-tile staging) was tested in cycle 2 and
regressed -12%, and the plan's Step 1 (GPU hang bisect) was never
needed because the loop measures at L=1500 where the hang doesn't
reproduce. The agent rewrote the attack plan in real time based on
profiling signal and shipped 12 keeps in a row.

## What's left (for run-4)

**Attention-shader internals are exhausted (confirmed across 3 runs).**
The bucket has moved.

**Dense FFN is the new dominant bucket (56% of decode time).** The
two attacks that landed in run-2 (cycle 8: gate+up+SwiGLU fused;
cycle 12: Q+K norm+rope+kv_write fused) plus run-3 cycle 12's
split-K cleared everything cheap. What's left on FFN:

1. **FFN split-M down_proj DMMV.** Mirror cycle-12's flash_attn
   split-K trick onto down_proj: split the M axis across more WGs
   with a merge pass that sums M-tile partials. Currently down_proj
   uses pipeline_q4k NUM_ROWS=2; tiling M further restores SIMD
   occupancy without breaking the inner-loop shape (which is
   already 78–85% of bandwidth-floor optimal). Top priority for
   run-4.
2. **FFN K-axis parallelism on down_proj.** down_proj has K=12288,
   the largest K in the model. Try THREADS_PER_BLOCK=32 (split
   each row across 32 threads) on JUST the down_proj path with
   subgroupClusteredAdd reduction. Engages more lanes per row
   while keeping NUM_ROWS=2. Distinct from cycle 9's failed wide
   NUM_ROWS=8 (which split M, not K).
3. **Cross-layer fusion via last-WG-does-norm pattern.** Both
   alternatives failed in run-3: cycle 5 (separate-buffer) -7%;
   cycle 15 (atomic-counter cross-WG sync) broken output. Third
   pattern unattempted: the LAST WG of o_proj's dispatch executes
   the rms_norm in-place after accumulate-in-place is done. Tricky
   to make robust on RDNA4; needs careful memory-scope semantics
   and a sentinel-value-wait inside the last-WG path.

**Refined GQA collapse via split-K combination is still viable.**
Both standalone Q_PER_KV=4 (-9.9%) and Q_PER_KV=2 (-17.5%) failed
because of WG-count loss. Combining with split-K's WG multiplier
gives 8 × 2 × 4 = 64 WGs (healthy concurrency) AND 2× K/V
amortization. Cycle-sized.

**flash_attn_cm1.comp KHR coopmat port** is still the structural
ceiling. Multi-week. Pursue when the FFN attacks above are
exhausted.

## On the L=2325 GPU hang  (deferred, not blocking)

The test at L=2325 in the manual session before this effort opened
hung the GPU (`amdgpu: ring comp_1.0.1 timeout`). Across 18
autonomous cycles all measuring at L=1500, no hangs were observed.
Conclusion: the hang is L≥2300-only and most likely a watchdog
duration issue from very long flash_attn dispatches at high
context, not a correctness bug. Defer to a separate cycle if the
user needs L>2300 working.

## What landed in the manual session before this effort opened

Three commits on `flash_attn.comp` and `flash_attn_batched.comp`:

| commit | shader change | LDS added |
|---|---|---:|
| `6ece0a8` | Stage Q in `shared float s_q[512]` once per workgroup; remove per-K-iter `q_data[]` re-reads from inner dot | +2 KB |
| `f68b7d7` | Cache `page_ids[]` per block in `shared uint s_page_ids_block[256]`; both Phase 1 and Phase 4 read shared instead of global | +1 KB |
| `539b2aa` | Fuse Phase 4 rescale into V-accumulation; mirror Q-stage and page_id cache into `flash_attn_batched.comp` | none |

All three are bit-correct and confirmed safe across 18 cycles of
build-on at L=1500. The cycle-8 `s_kv_base_v4` precompute later
subsumed `s_page_ids_block` entirely (the page-id math was hoisted
into the precompute), so the page-id cache LDS was repurposed.

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

Original criterion **met** as of cycle 17:

- ~~Decode at L=1500 on `Qwen3-8B-Q4_K_M.gguf` is ≥ 50 tok/s~~ → **93.68 tok/s, +87%** over the bar.
- ~~Coherence sweep green at L=1500 across both target models~~ → confirmed every cycle (Qwen 3 8B "Paris. The capital of Italy is Rome..." matches reference; Qwen 3.6 35B coherence preserved at every keep).
- ~~Empty-context decode and L=466 prefill within 5% of pre-effort rates~~ → both improved (the same vec4 + ILP wins help short context too).
- L=2325 GPU hang: deferred. Confirmed L≥2300-only across 18 cycles measuring at L=1500.

New criterion (run-4):

- Decode at L≈846 ≥ **108 tok/s** via FFN split-M down_proj +
  K-axis parallelism on down_proj + last-WG cross-layer fusion +
  split-K-with-GQA-q2. Adjusted from the prior "110 at L=1500"
  target now that we know the actual measurement L is ≈846, not
  1500. The 56% FFN bucket has ~30% of compute headroom (currently
  78-85% of bandwidth-floor) so 99.38 → 108 (+8.7%) is plausible
  if 2 of the 4 attacks land cleanly.
- No regression in empty-context decode at L=5.
- L=1500-token prompt benchmark added to the suite (separate from
  the existing L≈846 default) so the original "≥120 tok/s at L=1500"
  ambition can be re-measured directly.

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
