# Effort 8 — RDNA4 batched prefill orchestration

## Status

**Foundation shipped** (previous session): `rope_batched.comp`, `flash_attn_batched.comp`, pipeline wrappers (`elementwise.pipeline_rope_batched`, `attention.pipeline_batched`), dispatchers (`recordRoPEBatched`, `recordFlashAttnBatched`), `pub fn prefillBatched` entry point in `forward.zig` that delegates to `prefillBatch` until this effort lands.

**Measured baseline** (Qwen3-8B Q4_K_M, 103-token prompt, AMD Radeon AI PRO R9700, RADV GFX1201):

| Path | Throughput | vs llama.cpp |
|---|---:|---:|
| llama.cpp (Vulkan)   | **662 tok/s** | 100% |
| ZINC per-token       |    59 tok/s   |   9% |

llama.cpp is **11x faster**. This document is the explicit plan to close that gap.

## Why we are 11x slower

### Dispatch count

Per token, decodeStep dispatches roughly 16 kernels per layer:
- 1× attn RMS norm
- 3× Q/K/V DMMV (weight read per token, not per batch)
- 2× Q/K per-head norm
- 2× Q/K RoPE
- 1× KV cache write
- 1× flash attention
- 1× O projection DMMV
- 1× residual + FFN RMS norm
- 2× gate/up DMMV
- 1× SwiGLU
- 1× down DMMV
- 1× residual

Qwen3-8B has 36 layers. Per token: **~576 dispatches**.
For a 103-token prompt: **~59 000 dispatches**.

Each vkCmdDispatch has a host-side recording cost (~1-3 µs) and a GPU-side launch overhead (~5-20 µs on RDNA depending on kernel size). Even at the optimistic end, 59 k × 6 µs = **354 ms** pure launch overhead. That is ~25% of the full 1358 ms prefill walltime, burned on nothing but kernel-launch bookkeeping.

### Weight bandwidth wasted on re-reads

The Q projection weight is ~18 MiB (Q4_K, 4096×4096, block size 144). Per-token prefill re-reads that 18 MiB once per token. For 103 tokens, that is **1.85 GiB** of Q-weight reads, 95% of which is redundant — the same bytes could serve every token in the batch.

Sum across all seven per-layer projections for a 103-token Qwen3-8B prefill:

| Projection | Weight per read | Per-token reads | Per-batch reads (N=103) | Redundant |
|---|---:|---:|---:|---:|
| Q  (4096×4096) | 18 MiB | 1854 MiB | 18 MiB | **1836 MiB** |
| K  (1024×4096) |  4.5 MiB | 464 MiB |  4.5 MiB | **459 MiB** |
| V  (1024×4096) |  4.5 MiB | 464 MiB |  4.5 MiB | **459 MiB** |
| O  (4096×4096) | 18 MiB | 1854 MiB | 18 MiB | **1836 MiB** |
| gate (12288×4096) | 54 MiB | 5562 MiB | 54 MiB | **5508 MiB** |
| up (12288×4096) | 54 MiB | 5562 MiB | 54 MiB | **5508 MiB** |
| down (4096×12288) | 54 MiB | 5562 MiB | 54 MiB | **5508 MiB** |
| **Per layer** | ~207 MiB | ~21 GiB | ~207 MiB | **~21 GiB** |
| **All 36 layers** | — | ~755 GiB | ~7.4 GiB | **~748 GiB** |

At 576 GB/s HBM bandwidth, 755 GiB of weight reads has a bandwidth floor of **1.3 seconds**. We measure 1.36 s total prefill — essentially all of it is bandwidth-bound on redundant weight reads.

A batched path reads each weight exactly once per prefill call, hitting a **7.4 GiB / 576 GB/s = 13 ms** floor just on weight traffic. That is a 100× reduction in weight bandwidth for the same prompt.

### What llama.cpp does differently

llama.cpp's Vulkan backend picks a different kernel at the `num_cols` threshold. For N=1 (decode) it uses `mul_mv` (DMMV). For N≥8 or so (prefill) it routes to `mul_mm` — a tiled matmul kernel that processes a 64×32 (or similar) output tile per workgroup, reading each weight block once and multiplying against all N input columns in registers. That is the structural change we need.

We already have `dmmv_q4k_batch.comp` which does the same invention with a different kernel shape (64 threads × 1 row with `MAX_COLS=32` column accumulators). It is currently only exercised for the LM-head fast path (line 6166 in `forward.zig`). Nothing else in the prefill path uses `num_cols > 1`.

## Plan

The implementation mirrors the already-shipped Metal `prefillBatched` in `src/compute/forward_metal.zig`. Every primitive it needs is already available on the Vulkan side:

| Metal primitive | Vulkan primitive | Status |
|---|---|---|
| `gemm_q4k` (simdgroup tiles) | `dmmv_q4k_batch` (column accumulators) | ✅ ready, `MAX_COLS=32`, `recordBatchDispatchPush` |
| `rope_batched` | `rope_batched.comp` + `recordRoPEBatched` | ✅ ready |
| `flash_attn_batched` | `flash_attn_batched.comp` + `recordFlashAttnBatched` | ✅ ready, paged KV layout |
| `kv_cache_write` batched | `kv_cache_write.comp` (elementwise N blocks) | ✅ works as-is via multi-block dispatch |
| Batched RMS norm | `rms_norm_mul.comp` (grid.x = token idx) | ✅ already batchable |
| Batched SwiGLU | `swiglu.comp` (elementwise) | ✅ already batchable |
| Batched residual add | `scale_accumulate.comp` (elementwise) | ✅ already batchable |
| Embedding pre-dequant | `prefill_embed_big` (existing) | ✅ reuse as-is |

### Body outline (~300 lines)

```zig
pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
    if (prompt_tokens.len == 0) return;
    const mode = std.posix.getenv("ZINC_BATCHED_PREFILL") orelse "";
    if (!std.mem.eql(u8, mode, "1") or !canUseBatchedPrefill(self)) {
        return self.prefillBatch(state, prompt_tokens);
    }
    if (state.position != 0 and state.position != self.position) return error.KvStateNotAvailable;

    const n_tokens: u32 = @intCast(prompt_tokens.len);
    const position_base: u32 = state.position;
    // … target_context_tokens / resetRequestState as in prefillBatch …

    // (1) Allocate N-token scratch buffers. Cache them in the engine
    //     across prefill calls so repeated requests do not re-allocate.
    var scratch = try self.acquireBatchedScratch(n_tokens);
    defer self.releaseBatchedScratch(&scratch);

    // (2) Pre-dequantize embeddings into scratch.hidden (CPU memcpy from
    //     prefill_embed_big which prefillBatch already populates).

    // (3) One command buffer for the whole prefill, pipelined with
    //     prefill_pipeline_mode to overlap record with GPU execution.
    try self.decode_cmd.reset();
    try self.decode_cmd.beginOneTime();

    for (0..cfg.n_layers) |layer_idx| {
        // attn path
        //   dispatchBatchedRmsNorm(hidden → norm, n_tokens)
        //   dispatchBatchedDmmv(Q_w, norm → q, num_cols=n_tokens chunked at 32)
        //   dispatchBatchedDmmv(K_w, norm → k, …)
        //   dispatchBatchedDmmv(V_w, norm → v, …)
        //   computeBarrier()
        //   optional Q-norm / K-norm dispatched with n_groups = n_heads * n_tokens
        //   recordRoPEBatched(q, …, position_base, n_tokens)
        //   recordRoPEBatched(k, …)
        //   dispatchBatchedKvCacheWrite(k, v → kv_cache at position_base * kv_dim, n_tokens * kv_dim elements)
        //   recordFlashAttnBatched(q, kv_cache, … n_queries=n_tokens, kv_pos_offset=position_base)
        //   dispatchBatchedDmmv(O_w, attn_out → down, num_cols=n_tokens)
        //   scale_accumulate(hidden += down, N * hidden_dim elements)

        // ffn path
        //   dispatchBatchedRmsNorm(hidden → norm, n_tokens)
        //   dispatchBatchedDmmv(gate_w, norm → gate)
        //   dispatchBatchedDmmv(up_w, norm → up)
        //   swiglu(gate, up → swiglu, N * inter_dim elements)
        //   dispatchBatchedDmmv(down_w, swiglu → down, num_cols=n_tokens)
        //   scale_accumulate(hidden += down, N * hidden_dim elements)
    }

    // (4) Final RMS norm over all N tokens (reuse batched path) and
    //     LM head on the last token only (use DMMV with x_offset =
    //     (n_tokens - 1) * hidden_dim * 4).

    self.decode_cmd.end();
    try self.decode_cmd.submit();
    try self.decode_cmd.waitForCompletion();

    self.position = position_base + n_tokens;
    state.position = self.position;
}
```

### Chunking for `MAX_COLS=32`

`dmmv_q4k_batch.comp`'s push constant `num_cols` is bounded by the `MAX_COLS=32` register allocation. For N > 32, issue `ceil(N / 32)` dispatches per projection, advancing `x_offset` by `chunk_start * K * 4` bytes and `y_offset` by `chunk_start * M * 4` bytes. For a 103-token prompt this is 4 dispatches per projection → **25× dispatch reduction** compared to per-token.

### Expected result

If the per-token path is 75% weight-bandwidth-bound (as the 755 GiB / 576 GB/s floor argues), replacing the 755 GiB of re-reads with 7.4 GiB yields roughly a **7-10× speedup** on weight traffic alone, ignoring dispatch-launch savings. Dispatch count drops from 59 000 to ~2 500 (projections ~1 000 at 4 chunks × 7 projections × 36 layers, elementwise ~1 500 still per-token for rope/flash/swiglu/residual). Dispatch overhead drops from ~354 ms to ~15 ms.

Combined, a realistic target on first commit is **350-500 tok/s**. Closing the remaining gap to llama.cpp's 662 tok/s will need either a tiled `mul_mm` kernel (matching their structure) or further dispatch-fusion. Both are orthogonal follow-ups.

## Work breakdown

1. **Scratch buffer lifecycle.** Add `BatchedPrefillScratch` to `InferenceEngine` with the 10 buffers Metal's version needs (`hidden, norm, q, k, v, attn_out, gate, up, swiglu, down`), grown on demand and kept across calls. Sizing: `max_n_tokens * dim * 4`. Use the same `Buffer.initDeviceLocal` the rest of `forward.zig` uses.
2. **Batched KV write helper.** Thin wrapper around `kv_cache_write.comp` that dispatches `N` blocks × `kv_dim / 16` per-page covers; the existing page-table plumbing already handles the offset arithmetic when pages are consecutive — which they are on a fresh prefill.
3. **Batched DMMV wrapper.** Add `dispatchDmmvBatched(tensor, x_buf, y_buf, M, K, x_offset, y_offset, num_cols)` that chunks at `MAX_COLS=32` and calls `recordBatchDispatchPush` for each chunk.
4. **Model gate.** Port `canUseBatchedPrefill` from `forward_metal.zig`: dense attention every layer, dense FFN, Q4_K weights, no biases / attn gate / post-norms / sliding window / sinks, not MoE / SSM / Gemma / gpt-oss. Q/K norms are OK.
5. **Body.** Translate the Metal `prefillBatched` body line-for-line to Vulkan idioms. Use `push_desc_fn` throughout (matches the rest of `forward.zig`'s hot path).
6. **Gate wiring.** `generateWithMetrics` on the Vulkan side should route through `prefillBatched` the same way Metal's does, so the env flag works end-to-end.
7. **Validate mode.** Port `ZINC_BATCHED_PREFILL=validate` from Metal — snapshot batched logits, rerun prefillBatch, diff max abs.

## Acceptance criteria

- `zig build test` stays at 362/362 passing (one pre-existing bun smoke unrelated to prefill).
- `ZINC_BATCHED_PREFILL=1 zinc -m Qwen3-8B-Q4_K_M.gguf --prompt <103-tok> -n 1` on RDNA4 reports ≥ 300 tok/s prefill on the three-run median.
- `ZINC_BATCHED_PREFILL=validate` on the same prompt logs `max_abs_diff ≤ 1e-2` against per-token (Q8 KV amplification gives ~1e-1 on Metal; RDNA with f32 KV should be tighter).
- Output token sampled by argmax after the prompt is coherent and deterministic across runs.

## What is not in scope

- A tiled `mul_mm_q4_K_f32` GEMM kernel (matches llama.cpp's structure) — tracked separately, required to fully close the gap to 662 tok/s.
- Q8 KV batched flash attention on Vulkan — the f32 KV path is sufficient to meet the acceptance target; Q8 is a follow-up.
- MoE / SSM / Gemma / gpt-oss batched paths — all fall back to `prefillBatch` via `canUseBatchedPrefill`.

## References

- Metal implementation: `src/compute/forward_metal.zig` → `pub fn prefillBatched` (~200 lines of orchestration plus `BatchedPrefillScratch` / `canUseBatchedPrefill` / helper dispatchers). 92% of llama.cpp on LLaMA 3.1 8B, 91% on Qwen3 8B, 40× over per-token.
- Blog post with the full measurement story: `site/src/content/posts/2026-04-20-metal-batched-prefill-38x-speedup-coherent-with-llama-cpp.md`.
- Vulkan foundation already shipped in commits `3da05c7` (GLSL rope_batched), `b6d973f` (flash_attn_batched + pipeline wrappers + prefillBatched entry), `7fc5700` (blog update).
