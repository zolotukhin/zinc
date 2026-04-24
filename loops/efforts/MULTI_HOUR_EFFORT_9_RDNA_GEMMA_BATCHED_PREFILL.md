# Effort 9 — RDNA4 batched prefill for Gemma 4

## Status

**Blocked.** `canUseBatchedPrefillRdna` rejects Gemma on `cfg.architecture == .gemma` and on `cfg.sliding_window_size != 0`. Opening those gates alone produces structurally wrong output because the `prefillBatched` body assumes LLaMA-shaped layers; Gemma has five architectural deltas from what the body handles today.

## Measured baseline (`main` @ `a41c185`, AMD R9700 RDNA4 gfx1201)

| model | path | prefill | decode |
|---|---|---:|---:|
| qwen3-8b-q4k-m          | batched (shipped)        | 187 tok/s | — |
| **gemma4-31b-q4k-m**    | **per-token (only path)** | **4.97 tok/s** | **7.65 tok/s** |

49-token prefill takes 9.85 s on Gemma. Profile shows attention at 62 ms/tok (22×60 ≈ 3.7 GB of attn compute per token) and roughly 140 ms/tok on the seven per-layer Q4_K projections against the 31 B weight matrix. The bottleneck is weight re-reads — ~27 GB of weight traffic per token, 49 tokens = ~1.3 TB, bandwidth-floor 2.3 s on 576 GB/s, measured 9.85 s (~5× over floor).

The only lever that moves this is batched prefill: read each weight row once per prompt instead of once per token. That's a ≥10× prefill speedup ceiling, equivalent to what Qwen3-8B already sees (72 → 187 tok/s, 2.6×; Qwen3-8B's ratio is smaller because its weights are smaller relative to attention cost).

## The five deltas

### 1. Per-layer `head_dim` variance

Gemma 4 uses different head dimensions for full-attention vs sliding-window-attention layers:

| layer type | head_dim | q_dim | kv_dim | interval |
|---|---:|---:|---:|---|
| full attention | 512 | 16384 | 8192 | every 6th |
| sliding window | 256 |  8192 | 4096 | the rest |

Shipped values for gemma4-31b-q4k-m: 10 full-attn layers (0-indexed at 5, 11, 17, 23, 29, 35, 41, 47, 53, 59) × `head_dim=512`; 50 SWA layers × `head_dim=256`.

`ensureBatchedScratchCapacity` currently sizes scratch buffers from `cfg.n_heads * cfg.head_dim`, a single value. For Gemma that number has to be `max(full_head_dim, swa_head_dim) * cfg.n_heads = 16384` so the scratch can hold either layer's output — we just use the per-layer dim when dispatching.

Impact: scratch alloc grows to the full-attn size. On a 105-token prompt that's 105 × 16384 × 4 bytes = 6.7 MiB per projection scratch, vs 3.4 MiB if it were all SWA. Fine.

### 2. `post_attention_norm` and `post_ffn_norm`

Every Gemma layer has:

```
x = x + post_attention_norm(attn_output)
x = x + post_ffn_norm(ffn_output)
```

The current `prefillBatched` body runs `scale_acc(residual)` instead of `post_*_norm(out) + residual`. Two extra `dispatchRmsNorm` calls per layer, reading from `lt.post_attention_norm.gpu_buffer` and `lt.post_ffn_norm.gpu_buffer`. The weights are always `f32` single-vector per layer.

The per-token path already handles this (see `apply_post_attn_norm` branch in `runDecodeStep` around line 4449). Porting the same branch to `prefillBatched` is ~40 lines of dispatches + one new scratch slot for the "normalized attn/ffn output before residual" intermediate.

### 3. Sliding window attention

50 of 60 Gemma layers use SWA with window=1024. On prompts ≤1024 tokens SWA degenerates to full causal attention — so for typical chat turns it's a no-op.

For prompts >1024 tokens the `flash_attn_batched` shader needs a `window_size` push constant that clamps the causal lower bound:

```glsl
uint lower = (query_abs_pos > window_size) ? (query_abs_pos - window_size) : 0;
for (uint block_start = lower; block_start < causal_len; ...
```

Trivial shader change. Plus the dispatch has to pass layer-specific window (`cfg.sliding_window_size` for SWA layers, `0` for full-attn).

### 4. Proportional RoPE (`rope_freqs.weight`)

Gemma ships a 256-entry `rope_freqs.weight` f32 tensor that scales each RoPE frequency per dimension. The `rope_batched.comp` shader already supports this via the `FreqBuf` binding when `freq_base_bits == 0`. Just need to populate `freq_buf_handle` from the right tensor per layer type.

Nuance: full-attn layers use the standard `rope_freq_base`, SWA layers use `rope_freq_base_swa`. Two freq buffers needed.

### 5. Embedding pre-scale

Gemma scales embeddings by `sqrt(hidden_dim)` before the first layer. The per-token path does this in `loadTokenEmbedding`; `prefillBatched`'s pre-dequant loop (around line 7903) needs the same:

```zig
if (cfg.architecture == .gemma) {
    const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(hidden_dim))));
    for (dst) |*v| v.* *= scale;
}
```

One line. The `gemma_scale` helper already exists in that block for the per-token path.

## Proposed order

Each of these is independently testable via `ZINC_BATCHED_PREFILL=validate` once the gate lets Gemma through. Suggested sequence:

1. **Embedding pre-scale** (1 commit, ~5 lines). Safe under any gate.
2. **Post-norms in prefillBatched** (1 commit, ~50 lines + 1 new scratch slot). Requires relaxing the gate's `post_attn_norm_present` / `post_ffn_norm_present` checks — currently neither Vulkan `canUseBatchedPrefillRdna` nor the body looks at them, so adding both guards is free.
3. **Per-layer head_dim and projection dims** (1 commit, ~30 lines). Touches every `dispatchProjectionBatched`/`dispatchRopeBatched`/`dispatchKvCacheWriteBatched`/`dispatchFlashAttnBatched` call site to thread `layer_head_dim` instead of `cfg.head_dim`.
4. **Dual RoPE freq buffers** (1 commit, ~20 lines). Load freq for full-attn and SWA layers separately, pick at dispatch time.
5. **SWA push constant on `flash_attn_batched`** (1 commit, ~10 lines of shader + 1 field in push). Only needed for prompts >1024 tokens; full-prompt correctness on shorter prompts holds without it.
6. **Relax the gate** (1 commit). Accept `.gemma`, accept `sliding_window_size != 0`, accept `layer_output_scales[i] != 1.0` (Gemma has a per-layer scalar that's currently rejected — apply it after the post-norm).

After each step, `ZINC_BATCHED_PREFILL=validate` on gemma4-31b-q4k-m should land near `max_abs_diff=0` against per-token if the change is correct. That's the sole measurement gate.

## Not in scope

- **Gemma 4 12B MoE**: `n_experts > 0` first, then MoE routing. Same scope as Qwen3.6 MoE port (Effort 8 follow-on).
- **LM head speedup**: Gemma's 262144-vocab LM head eats 45 ms/decode-token. Tangentially useful but not the prefill story.
- **Row-major X layout** (the "cache thrashing per column loop" rewrite). Would push Qwen3-8B past 200 tok/s on short prompts but is independent of Gemma.

## Expected outcome

On a 105-token prefill of gemma4-31b-q4k-m on R9700:

- Baseline: 4.97 tok/s (21.1 s for 105 tokens).
- Target: ~50 tok/s (weight-read-once gives ~10× bandwidth cut; compute-per-layer similar to Qwen's ratio of 2.6× gives a conservative 10× end-to-end).

Measure and adjust — the first pass landing in the 30–60 tok/s band would already be life-changing for Gemma users on RDNA.
