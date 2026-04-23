# Doubling RDNA4 prefill on ZINC: a correctness bug, a K-parallel shader, and a 72→143 tok/s story

*Session raw material, 2026-04-23. Measured on AMD Radeon AI PRO R9700
(RADV gfx1201, 32 GB, 576 GB/s HBM). Model: Qwen3-8B Q4_K_M, dense
architecture, 36 layers, 32 Q-heads / 8 KV-heads, hidden_dim=4096.*

## TL;DR

- Baseline: `ZINC_BATCHED_PREFILL` was a no-op. Every chat prefill ran
  the per-token `prefillBatch` path at ~72 tok/s on a 105-token prompt.
- Turning the gate on naively produced output gibberish —
  `![](https://upload.wikimedia.org/wikipedia/commons/...)` instead of
  "Paris."
- After two real fixes (one sampler bug, one shader shape change)
  `ZINC_BATCHED_PREFILL=1` is a **2× speedup**: 143 tok/s on the same
  prompt, same model, same GPU. Default path for supported models.

| path | prefill (3-run median) |
|---|---:|
| per-token `prefillBatch`                    |  72.3 tok/s |
| batched, serial-over-K (original shader)    |  61.5 tok/s |
| **batched, K-parallel wave64 (this work)**  | **143.1 tok/s** |

Same 105-token prompt, `ReleaseFast`, no profile overhead, no other
environment tweaks. Output text in every row: *"The capital of France
is **Paris**"*.

## Why this mattered in the first place

For Qwen3.6-35B-A3B on the same box, ZINC decode is already at
llama.cpp parity (~27 tok/s). Prefill is not: 26 tok/s vs llama.cpp's
123 tok/s — a 4.7× gap. For a typical chat session that translates
directly to several extra seconds on time-to-first-token. The goal
going into the session was simply "make Qwen3.6-35B prefill much
faster". The punchline is that Qwen3.6 is MoE+SSM so its prefill gate
(`canUseBatchedPrefillRdna`) rejects it outright — but diagnosing why
the nominally-shipped batched prefill wasn't helping the dense models
either surfaced two unrelated bugs that took Qwen3-8B from 72 to 143
tok/s without even touching the MoE/SSM path.

## The landscape going in

`src/compute/forward.zig` has two entry points for prefill on Vulkan:

- `prefillBatch(state, prompt_tokens)` — per-token path. For each
  prompt token, it runs the full decode graph (attention → flash attn
  → projections → FFN → residual), appending one KV entry per token.
  This is the "cold, proven" path that all the main regression tests
  exercise.
- `prefillBatched(state, prompt_tokens)` — batched path. Processes the
  whole prompt in one command buffer, using:
  - `dmmv_q4k_batch.comp` + `dmmv_q6k_batch.comp` for projections
    (read each weight row once, accumulate against all N input
    columns).
  - `rope_batched.comp` for RoPE on N positions in one dispatch.
  - `kv_cache_write_batched.comp` to write N KV entries at once.
  - `flash_attn_batched.comp` with a causal mask that gives query `t`
    visibility of positions `[0, seq_start + t]`.

On paper this saves the per-token weight re-reads that dominate
prefill bandwidth cost (Qwen3-8B, 7 Q4_K projections per layer, 36
layers ≈ 207 MiB of weights read 105 times in the per-token path,
only once in the batched path — a ~100× bandwidth reduction floor).

The `MULTI_HOUR_EFFORT_8_RDNA_BATCHED_PREFILL.md` design doc described
this path as "foundation shipped". The reality turned out to be
harder.

## Bug 1: the batched path was dead code on Vulkan

First surprise from trying to actually call `prefillBatched` from the
CLI and server paths:

```
src/compute/forward.zig:7525:36: error: no field named 'position'
    in struct 'compute.forward.InferenceEngine'
```

The body of `prefillBatched`, written as a port of the Metal version,
had this:

```zig
if (state.position != self.position) {
    return self.prefillBatch(state, prompt_tokens);
}
...
self.position = base_token + n_tokens;
state.position = self.position;
```

Metal's `InferenceEngine` has a `position: u32` field. Vulkan's
doesn't — on Vulkan `state.position` is authoritative. But nobody was
actually *calling* `prefillBatched` on Vulkan — the server path in
`src/server/routes.zig` and the CLI's `generate()` in
`src/compute/forward.zig` both went to `prefillBatch` directly. Zig's
lazy compilation never hit the dead body, so the error stayed hidden
through five commits of batched-prefill effort.

**Fix (`10c1737`):** drop the `self.position` check, replace the dual
assignment with `state.position = base_token + n_tokens`. Update the
regression test string-marker check.

Then wire the CLI `generate()` (`ed1e60d`) and server chat handler
(earlier attempt) to call `prefillBatched` so `ZINC_BATCHED_PREFILL=1`
could actually take effect.

## Bug 2: `recordBatchDispatchPush` only knew about Q4_K

`canUseBatchedPrefillRdna` accepted `q4_k` or `q6_k` for projections.
But the function the batched body uses to actually dispatch:

```zig
pub fn recordBatchDispatchPush(...) !void {
    const pip = switch (quant_type) {
        .q4_k => if (self.pipeline_q4k_batch) |*p| p
                 else return error.UnsupportedQuantType,
        else => return error.UnsupportedQuantType,
    };
    ...
}
```

…only knew how to pick a Q4_K pipeline. For every Q4_K_M checkpoint
in the catalog (Qwen3-8B, Gemma4-31B, etc.) `ffn_down` and `attn_v`
are quantized to Q6_K. The gate said yes, the dispatch said
`UnsupportedQuantType`, the prompt aborted before producing any logits.

Two paths out:

- Tighten the gate to Q4_K-only on projections. Correct but useless —
  no Q4_K_M model would ever reach the batched path.
- Write a `dmmv_q6k_batch.comp` shader and plumb it in.

I did both in sequence: `e1428e2` tightened the gate to keep
correctness while Q6_K was missing, then `24a837b` added the Q6_K
batched shader (below), and `328faa1` re-opened the gate.

### The Q6_K batched shader

Mirrors `dmmv_q4k_batch.comp`'s structure. A Q6_K super-block is 210
bytes: 128 bytes of ql (low 4 bits for every element), 64 bytes of qh
(high 2 bits, 4 elements per byte), 16 signed int8 scales, and a
half-precision `d`. Decode formula: `w[i] = d · signed_scale[i/16] ·
((ql[i] | (qh[i/4] << 4)) − 32)`.

Per super-block, one thread owns one output row, walks all 16 scales,
decodes 4 q-values per byte, and accumulates a dot-product against
`num_cols` input vectors read from a column-major `X[K, num_cols]`
layout. Register array `sums[MAX_COLS]` (MAX_COLS=32) fits easily
inside the wave64 VGPR budget on gfx1201.

This unblocked Qwen3-8B Q4_K_M prefill to actually execute the batched
body.

## Bug 3: the garbage-tokens bug was in the *sampler*

Opening the gate finally let Qwen3-8B run through `prefillBatched`.
Output:

```
Prompt : "The capital of France is Paris, and the capital of Italy
         is Rome, ..."
Expected: "The capital of France is **Paris**..."
Got     : "![](https://upload.wikimedia.org/wikipedia/commons/..."
```

Coherent-looking tokens, wildly unrelated to the prompt. First
instinct: the batched body has a subtle math bug — wrong causal mask,
RoPE position off by one, KV layout mismatch, something.

Instead of poking around the orchestration for hours I added
`ZINC_BATCHED_PREFILL=validate` (ported from the Metal version in
`5cc4f15`). Validate mode runs the batched path, snapshots the
last-token logits, resets state to a fresh request, runs the per-token
`prefillBatch` as reference, and diffs. The first run printed:

```
warn(forward): prefillBatched validate[ok]: last-token logits
    max_abs_diff=0.000000 at idx=0 (ref=14.0986 batched=14.0986)
    tol=0.001000 n_tokens=17
```

**Max absolute diff: 0.000000, across a 151,936-wide vocab.** The
batched path was producing bit-identical logits to the per-token
path. The forward math wasn't wrong at all. The bug had to be
*after* the forward pass.

That shifted the search to the sampler. `sampleGreedy` in
`src/compute/forward.zig` reads like this:

```zig
pub fn sampleGreedy(self: *const InferenceEngine) u32 {
    if (self.argmax.pipeline != null
        and self.argmax_descriptor_set != null) {
        const token_ptr: [*]const u32 =
            @ptrCast(@alignCast(self.argmax_result_staging.mapped.?));
        return token_ptr[0];
    }
    // CPU fallback scanning self.logits_staging.mapped
    ...
}
```

`argmax_result_staging` is a host-visible buffer that gets populated
by a two-phase GPU argmax shader the per-token `runDecodeStep`
dispatches at the end of every decode token. `prefillBatched`'s tail
did the LM head and copied logits to `logits_staging`, but never ran
the argmax shader. So the first decode step after prefill sampled
from whatever happened to be sitting in `argmax_result_staging` — on
a freshly-loaded engine, typically zeroes, hence the `!` (token 0)
leading every gibberish completion.

**Fix (`419e929`):** record `argmax.record` + `vkCmdCopyBuffer(argmax_result_buf
→ argmax_result_staging)` into the batched command buffer right after
the LM head, mirroring `runDecodeStep`'s tail exactly. 20 new lines.

After this fix the output was *correct*, but — and this is the important
part — still not faster than per-token:

| n_tokens | per-token | batched (correct, slow) |
|---:|---:|---:|
|  17 | 42 tok/s | 23 tok/s |
| 105 | 74 tok/s | 62 tok/s |

Batched correctly produced Paris, and was 17–45% slower at it. Which
brings us to the last and biggest piece.

## Why batched-serial lost to per-token

The existing `dmmv_q4k_batch.comp` has a very simple shape:

```glsl
// Dispatch: ((M+63)/64, 1, 1). 64 threads per workgroup,
// each thread owns ONE output row.
void main() {
    uint row = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x;
    float sums[MAX_COLS];
    for (uint c = 0u; c < num_cols; c++) sums[c] = 0.0;

    for (uint blk = 0u; blk < blocks_per_row; blk++) {
        // Decode Q4_K block for this row...
        for (uint sp = 0u; sp < 4u; sp++) {
            for (uint e = 0u; e < 32u; e++) {
                float w = decode(blk, sp, e);  // one weight value
                for (uint c = 0u; c < num_cols; c++) {
                    sums[c] += w * x_data[c*K + blk*256 + ...];
                }
            }
        }
    }
    for (uint c = 0u; c < num_cols; c++)
        y_data[c*M + row] = sums[c];
}
```

One thread per output row. Each thread walks *all* of K serially.
That's fundamentally the shape of a classic mul_mat_vec, except it
accumulates `num_cols` partial sums in registers so one pass over the
weight row amortizes across up to 32 input vectors.

The per-token path (`dmmv_q4k_moe_kpar.comp` and friends) uses a
completely different shape:

```glsl
// Dispatch: ((M+1)/2, ..., 1). 64 threads per workgroup,
// 16 threads cooperate on each Q4_K block, 4 blocks in parallel.
// Each thread handles 16 elements of a row; subgroupAdd reduces
// across all 64 lanes at the end.
layout(local_size_x = 64) in;
...
for (uint i = ix; i < blocks_per_row; i += 4u) {
    // Decode just THIS thread's stripe of block i.
    // Accumulate one partial sum.
}
sum = subgroupAdd(sum);  // reduce across 64 lanes
if (tid == 0u) y_data[row] = sum;
```

For K=4096, the serial batched shader does 4096 element-level updates
per thread. The K-parallel per-token shader does 4096/64 ≈ 64. On
gfx1201 with no matrix cores, the K-parallel layout is 25–60× faster
per row even before the subgroupAdd — which itself is free on wave64
as a single hardware op.

So the batched shader was optimizing the wrong axis. It saved weight
re-reads (good, L2-aligned) but gave up per-row parallelism (bad,
fundamental arithmetic throughput). On a memory-bound problem the
weight savings would win; on gfx1201 for these shapes the problem
isn't memory-bound at the shader level — it's arithmetic-bound.

## The fix: K-parallel batched shader

`dmmv_q4k_batch_kpar.comp` combines both properties:

- One workgroup per output row — same dispatch granularity as the
  per-token kpar shader (`(M, 1, 1)` instead of `((M+63)/64, 1, 1)`).
- 16 threads cooperate on each Q4_K block, using the same bit-layout
  walk as `dmmv_q4k_moe_kpar.comp` (same `v_im`, `v_in`, `l0`,
  `sb_a..sb_d`, `decode_scale/decode_min` gymnastics).
- Each thread still accumulates `sums[MAX_COLS]` in registers.
- At the end, one `subgroupAdd` per column reduces across all 64 lanes.

Per thread, per block iteration:

```glsl
vec4 q0_lo = unpack_nibbles_lo(qs_u32_0);
vec4 q0_hi = unpack_nibbles_hi(qs_u32_0);
vec4 q1_lo = unpack_nibbles_lo(qs_u32_1);
vec4 q1_hi = unpack_nibbles_hi(qs_u32_1);

// ...decode sb_a..sb_d scales/mins (same pattern as kpar)...

for (uint c = 0u; c < num_cols; c++) {
    uint col_base_v4 = x_global_base_v4 + c * x_col_stride_v4;
    vec4 by0 = x_v4[col_base_v4 + b_idx];
    vec4 by1 = x_v4[col_base_v4 + b_idx + 8u];
    vec4 by2 = x_v4[col_base_v4 + b_idx2];
    vec4 by3 = x_v4[col_base_v4 + b_idx2 + 8u];

    float partial = dot(vec4(factor0)*q0_lo - vec4(bias0), by0)
                  + dot(vec4(factor1)*q0_hi - vec4(bias1), by1)
                  + dot(vec4(factor2)*q1_lo - vec4(bias2), by2)
                  + dot(vec4(factor3)*q1_hi - vec4(bias3), by3);
    sums[c] += partial;
}
```

The output write loop:

```glsl
for (uint c = 0u; c < num_cols; c++) {
    float reduced = subgroupAdd(sums[c]);
    if (tid == 0u)
        y_data[y_offset / 4u + c * M + row] = reduced;
}
```

Every weight nibble is decoded exactly once per 64-thread workgroup
(i.e. once per output row per batch chunk). The `num_cols` inner loop
burns registers (32 × `vec4` loads of X per iteration) but that's
still well inside gfx1201's VGPR budget, so occupancy stays healthy.

### Plumbing

- `pipeline_q4k_batch_kpar: ?Pipeline` on `DmmvDispatch`, loaded at
  init with `push_desc_wave64_options` so the pipeline sees 64-thread
  subgroups.
- `use_q4k_batch_kpar: bool` on `InferenceEngine`, set from
  `ZINC_Q4K_BATCH_KPAR` at init. **Default on** when the pipeline is
  loaded; `ZINC_Q4K_BATCH_KPAR=0` falls back to the serial shader for
  A/B comparison.
- `dispatchProjectionBatched` in `src/compute/forward.zig` now
  branches: kpar when the flag is set and the tensor is Q4_K,
  serial-over-K (via `recordBatchDispatchPush`) otherwise. The Q6_K
  tensors still go through the serial shader because there's no Q6_K
  kpar variant yet — but Qwen3-8B still sees the 2× win since ~70% of
  projections are Q4_K.

## Measurements

All runs on the R9700, `ReleaseFast` build, `qwen3-8b-q4k-m` Q4_K_M
checkpoint. 105-token prompt. 3-run medians unless noted.

### Prefill throughput

| path | tok/s |
|---|---:|
| per-token (ZINC_BATCHED_PREFILL unset) |  72.3 |
| batched, serial-over-K (kpar OFF)       |  61.5 |
| **batched + kpar (new default)**        | **143.1** |

### End-to-end wall time, 105 prompt tokens + 8 generated

| path | total |
|---|---:|
| per-token             | 1.59 s |
| batched serial        | 1.78 s |
| **batched + kpar**    | **0.88 s** |

### Validate-mode diff (shows the earlier forward pass was always correct)

```
warn(forward): prefillBatched validate[ok]: last-token logits
    max_abs_diff=0.000000 at idx=0 (ref=14.0986 batched=14.0986)
    tol=0.001000 n_tokens=17
```

Max absolute diff across 151,936 vocab entries: `0.000000`. Every
logit bit-identical. This is what proved the forward math was right
and pointed at the sampler.

### Coherence across the catalog (ZINC_BATCHED_PREFILL=1)

| model | supported? | output |
|---|---|---|
| qwen3-8b-q4k-m          | yes, batched path   | "The capital of France is **Paris**." |
| qwen35-35b-a3b-q4k-xl   | no (MoE+SSM gate)   | falls back to per-token, same output  |
| qwen36-35b-a3b-q4k-xl   | no (MoE+SSM gate)   | falls back to per-token, same output  |
| gpt-oss-20b-q4k-m       | no (architecture gate) | falls back, `<\|channel\|>analysis...`  |
| gemma4-31b-q4k-m        | no (architecture gate) | falls back, `Paris<turn\|>`            |
| gemma4-12b-q4k-m        | no (architecture gate) | falls back, correct                   |

All 6 still coherent, no regressions.

## How the debugging actually went, in rough order

1. "Make Qwen3.6-35B prefill faster." Obvious first step: check what
   `ZINC_BATCHED_PREFILL=1` does. Answer: nothing. The CLI calls
   `prefillBatch`; the server calls `prefillBatch`; `prefillBatched`
   exists but nobody reaches it.
2. Try to wire `prefillBatched` into the server path. Build breaks —
   `self.position` doesn't exist on Vulkan.
3. Fix the dead-code compile error. Build succeeds.
4. Enable the env gate. Runtime crash: `UnsupportedQuantType`.
5. Find the Q4_K-only switch in `recordBatchDispatchPush`. Two options:
   tighten the gate, or add Q6_K. Tighten for safety, then add Q6_K
   (Q4_K-only excluded every model in the catalog).
6. Open the gate to Q6_K. Output: `![](https://upload.wikimedia.org/...)`.
   The shader I just added must be wrong!
7. Dead-end pass through the Q6_K shader looking for a decode bug. No
   obvious issue — matches the reference `dmmv_q6k.comp` structurally.
8. Add `ZINC_BATCHED_PREFILL=validate`. Run it. Logits match
   per-token to 0.000000. The shader isn't wrong. The forward pass
   is correct.
9. Look *after* the forward pass. `sampleGreedy` reads
   `argmax_result_staging`. Find the call site in `runDecodeStep` that
   writes that buffer. Notice `prefillBatched` doesn't call it.
10. Add `argmax.record` + buffer copy to `prefillBatched`'s tail.
    Output: "Paris." Correctness restored.
11. Benchmark: still slower than per-token on the 105-token prompt.
    Shader is the wrong shape.
12. Read the existing `dmmv_q4k_batch.comp`. One thread per row,
    serial over K. Compare to `dmmv_q4k_moe_kpar.comp` — 64 threads
    per row, parallel over K, subgroupAdd at the end. That's the
    per-token winner.
13. Write `dmmv_q4k_batch_kpar.comp`: take the kpar thread/block
    layout, add the `sums[MAX_COLS]` register array, `subgroupAdd`
    per column at the end.
14. Plumb in a pipeline + env toggle. Measure: 143 tok/s. Done.

Commit trail on `main`:

```
328faa1 rdna: enable batched prefill on Q4_K_M models + kpar default-on
9c82ddd rdna: K-parallel Q4_K batched prefill shader (opt-in)
419e929 vulkan: run GPU argmax at the end of prefillBatched
4239529 vulkan: add validate mode to prefillBatched for correctness debugging
28ac9bc rdna: re-tighten batched-prefill gate to Q4_K projections only
24a837b rdna: add Q6_K batched DMMV shader — unblocks Q4_K_M prefill
e1428e2 vulkan: tighten canUseBatchedPrefillRdna — per-layer projs must be Q4_K
ed1e60d vulkan: route CLI prefill through prefillBatched as well
10c1737 vulkan: fix prefillBatched referring to nonexistent self.position
```

## Lessons worth keeping

**Zig's lazy compilation hides dead code indefinitely.** The Vulkan
`prefillBatched` body had a `self.position` compile error and we
didn't find it until something actually called the function from a
live code path five months after it landed. The shipped regression
tests all asserted string-level markers (`"must contain this call"`),
never numerical correctness. A structural test can't catch a function
that's never called.

**Validate modes are the right debugging primitive for numeric GPU
code.** The Metal path shipped `ZINC_BATCHED_PREFILL=validate` in
`5cc4f15` and I ported it to Vulkan in `4239529` as a targeted
response to "output is garbage". It pointed at the real bug in about
60 seconds. Without it I would have spent hours staring at KV layouts
and causal masks.

**Correctness bugs that *look* like math bugs are often plumbing
bugs.** The generated tokens looked like the model was doing real
autoregressive work on a corrupted hidden state — `!` leading a URL
template, plausible-looking tokens after. The actual cause was
`sampleGreedy` reading stale bytes from a host-visible buffer. The
GPU argmax shader that should have overwritten that buffer wasn't
being called from the batched path.

**The existing `dmmv_q*_batch` shaders fit a different mental model of
"batching".** They batched *input columns* but not *per-row work*.
llama.cpp's `mul_mm` path uses tiled matmul with warp-level
parallelism across both M and K. On gfx1201 wave64 the K-parallel
variant closes most of that gap without the complexity of a full
tiled matmul — you get the weight-read-once property from batching,
and the arithmetic-throughput property from `subgroupAdd`. The two
combine multiplicatively.

**`subgroupAdd` on wave64 is free.** The cost is a single shuffle-reduce
op the hardware implements in a handful of cycles; against 64 floats
of serial compute per thread that it replaces, the win is enormous.
The reason the per-token kpar shader beats the serial batched shader
in the first place is this instruction; the reason the kpar *batched*
shader compounds the win is the weight-read-once + subgroupAdd combo.

## What's next for the other 5 catalog models

Four of the six catalog entries still hit the gate's other guards
(MoE, SSM, Gemma architecture, gpt-oss architecture) and fall back to
per-token:

- Qwen3.5/3.6 35B-A3B: MoE+SSM hybrid. Needs batched MoE router +
  batched per-expert GEMM + batched SSM — distinct, substantial ports.
- Gemma-4 31B dense: gated out by `architecture == .gemma`. Probably
  a one-line gate relaxation plus a validate run; the per-token
  decode path already handles Gemma normalization quirks.
- Gemma-4 12B MoE: same MoE situation as Qwen.
- gpt-oss-20B: gated out by `architecture == .gpt_oss`. Has attention
  sinks that need special handling in the batched flash-attn path.

A Q6_K kpar variant would also be worth benchmarking — the current
implementation has Q6_K projections going through the serial shader,
and those still account for ~30% of Qwen3-8B's projections. Flipping
those to kpar could push past 143 tok/s.

The infrastructure that landed in this session — the argmax fix, the
validate mode, the K-parallel shader shape — is the foundation each
of those extensions will build on.
