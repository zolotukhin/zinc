# Optimization 3: Batch Prefill (Multi-Token Forward Pass)

## Why

This is the **single highest-impact optimization** remaining. ZINC processes prompt tokens one at a time — each token reads ALL model weights from VRAM. For a 16-token prompt with a 10.5GB model:

- **Current (sequential)**: 16 × 10.5GB = **168GB of weight reads** → 168GB / 576 GB/s = 292ms minimum → actual ~2.6 seconds (16 × 165ms)
- **Batch (read weights once)**: 10.5GB + 16 × ~50MB compute = **11.3GB of reads** → 11.3GB / 576 = 20ms minimum → actual ~200ms with overhead
- **llama.cpp batch**: 23 tokens in 249ms = **10.8ms/token** = 85 tok/s prefill

Expected improvement: **10-14x faster prefill** → from ~6 tok/s to ~60-80 tok/s prompt processing. This eliminates the 25-40 second wait before first token in the chat UI.

## What

Replace `prefillBatch`'s sequential `for` loop with a **layer-by-layer batch processing** approach:

1. **Batch embed**: CPU-dequantize all N token embeddings at once
2. **Per layer**: Process all N tokens through the layer simultaneously
   - **Batch DMMV**: `Y[M,N] = W[M,K] × X[K,N]` — weight matrix read ONCE for N tokens
   - **Batch elementwise**: RMS norm, RoPE, activation for N tokens
   - **Sequential attention**: Each token's attention query still runs individually (causal mask prevents full batching without a batch attention shader)
3. **LM head**: Only compute logits for the LAST token (save N-1 LM head DMMVs)

### What stays sequential
Flash attention must process queries one at a time because token i can only attend to tokens 0..i (causal). A batch flash attention shader exists in llama.cpp (`flash_attn_cm2.comp`) but is complex. The sequential attention is a small fraction of total time for short-to-medium prompts.

### Already built infrastructure
- `dmmv_q4k_batch.comp` shader (committed): reads weights once, multiplies N input columns
- `DmmvDispatch.recordBatchDispatch()` method (committed): dispatches batch DMMV with fallback
- `BatchDmmvPushConstants` struct (committed): M, K, offsets, num_cols

## How

### Step 1: Allocate batch intermediate buffers

**File: `src/compute/forward.zig` init section (~line 614)**

Add N-wide buffers for all intermediates. Size them for `max_prefill_tokens` (e.g., 512):

```zig
const max_prefill: u32 = 512; // max tokens per batch prefill
const batch_hidden_size = @as(VkDeviceSize, max_prefill) * hidden_size;

// Batch versions of single-token buffers
var batch_hidden_buf = try Buffer.initDeviceLocal(instance, batch_hidden_size, buf_usage);
var batch_norm_buf = try Buffer.initDeviceLocal(instance, batch_hidden_size, buf_usage);
var batch_q_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * q_size, buf_usage);
var batch_k_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * kv_size, buf_usage);
var batch_v_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * kv_size, buf_usage);
var batch_attn_out_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * q_size, buf_usage);
var batch_o_proj_buf = try Buffer.initDeviceLocal(instance, batch_hidden_size, buf_usage);
var batch_gate_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * inter_size, buf_usage);
var batch_up_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * inter_size, buf_usage);
var batch_swiglu_buf = try Buffer.initDeviceLocal(instance, @as(VkDeviceSize, max_prefill) * inter_size, buf_usage);
var batch_down_buf = try Buffer.initDeviceLocal(instance, batch_hidden_size, buf_usage);
var batch_ffn_norm_buf = try Buffer.initDeviceLocal(instance, batch_hidden_size, buf_usage);
```

**Memory cost for Gemma 3 12B** (hidden=3072, inter=12288, q=8192, kv=2048, max_prefill=512):
- batch_hidden/norm/o_proj/down/ffn_norm: 5 × 512 × 3072 × 4 = 31.5MB
- batch_q/attn_out: 2 × 512 × 8192 × 4 = 33.6MB
- batch_k/v: 2 × 512 × 2048 × 4 = 8.4MB
- batch_gate/up/swiglu: 3 × 512 × 12288 × 4 = 75.5MB
- **Total: ~149MB** (on a 32GB GPU with 17.5GB model → plenty of room)

Also need a batch embedding staging buffer:
```zig
var batch_embed_staging = try Buffer.initStaging(instance,
    @as(VkDeviceSize, max_prefill) * hidden_size);
```

### Step 2: Batch CPU embedding

**File: `src/compute/forward.zig` in `prefillBatch`**

```zig
pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
    const n: u32 = @intCast(prompt_tokens.len);
    if (n == 0) return;
    if (n > max_prefill) {
        // Fallback: process in chunks of max_prefill
        // ...
    }

    // Phase 1: CPU batch embed all tokens
    const staging_f32: [*]f32 = @ptrCast(@alignCast(self.batch_embed_staging.mapped.?));
    for (prompt_tokens, 0..) |token_id, i| {
        dequantRow(mmap[data_start..], token_id, hidden_dim, embd.info.type_,
            staging_f32[i * hidden_dim ..][0..hidden_dim]);
        if (scale != 1.0) { /* scale embedding */ }
    }

    // Phase 2: GPU upload all embeddings
    try self.decode_cmd.begin();
    // Copy batch_embed_staging → batch_hidden_buf
    vk.c.vkCmdCopyBuffer(cmd, self.batch_embed_staging.handle, self.batch_hidden_buf.handle,
        1, &region{.size = n * hidden_size});
    self.decode_cmd.transferToComputeBarrier();

    // Phase 3: Layer-by-layer batch processing
    for (0..config.n_layers) |layer_idx| {
        try self.processLayerBatch(state, @intCast(layer_idx), n);
    }

    // Phase 4: Final norm + LM head for LAST token only
    // Copy last token's hidden state to single-token hidden_buf
    // ... then normal final_norm + LM head dispatch ...

    try self.decode_cmd.end();
    try self.decode_cmd.submitAndWait(self.instance.compute_queue);
    state.position = n;
}
```

### Step 3: Implement `processLayerBatch`

This is the core: processes all N tokens through one transformer layer.

```zig
fn processLayerBatch(self: *InferenceEngine, state: *DecodeState, layer: u32, n: u32) !void {
    const hidden_dim = self.model.config.hidden_dim;
    const hidden_size = @as(VkDeviceSize, hidden_dim) * @sizeOf(f32);

    // --- Batch RMS norm: batch_hidden[N] → batch_norm[N] ---
    // Dispatch N workgroups (one per token) instead of 1
    {
        const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds,
            self.batch_hidden_buf.handle, n * hidden_size,    // input: N hidden states
            attn_norm.gpu_buffer.handle, attn_norm.gpu_buffer.size,  // weights (shared)
            self.batch_norm_buf.handle, n * hidden_size);     // output: N normed states
        try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, n, rms_norm_eps);
        // NOTE: n workgroups instead of 1 — the RMS norm shader already supports this
        // via its `token = gl_WorkGroupID.x` indexing
    }
    self.decode_cmd.computeBarrier();

    // --- Batch Q/K/V DMMV: weight × batch_norm[K,N] → batch_q[M,N] ---
    // Uses dmmv_q4k_batch.comp: reads weights once, N input columns
    {
        try self.dmmv.recordBatchDispatch(&self.decode_cmd, q_tensor.info.type_,
            ds, layer_q_dim, hidden_dim, 0, 0, 0, n);
        // Similar for K, V...
    }
    self.decode_cmd.computeBarrier();

    // --- Batch Q/K norms: batch_q[N] → batch_q[N] ---
    // Same as single-token but with n * n_heads workgroups
    // ...

    // --- Batch RoPE: different position per token ---
    // Need a modified RoPE shader that takes a position ARRAY instead of scalar
    // Or dispatch N separate RoPE calls (each with position = token_index)
    for (0..n) |t| {
        // RoPE for token t at position t
        // ... record RoPE dispatch for batch_q[t*q_dim..(t+1)*q_dim] with position=t ...
    }
    self.decode_cmd.computeBarrier();

    // --- Batch KV cache write: copy all N K/V vectors at once ---
    for (0..n) |t| {
        const kv_offset = @as(VkDeviceSize, t) * layer_kv_vec_size;
        // Copy batch_k[t] → kv_k_cache[t]
        // Copy batch_v[t] → kv_v_cache[t]
    }
    self.decode_cmd.transferToComputeBarrier();

    // --- Sequential attention: each token attends to 0..t ---
    for (0..n) |t| {
        // Flash attention: Q=batch_q[t], KV_cache[0..t], out=batch_attn_out[t]
        // seq_len = t + 1
        // Need to set up descriptor sets pointing to the right offsets in batch buffers
    }
    self.decode_cmd.computeBarrier();

    // --- Batch O projection: batch_attn_out[N] → batch_o_proj[N] ---
    // Same batch DMMV approach
    // ...

    // --- Batch residual: batch_hidden[N] += batch_o_proj[N] ---
    // Use scale_accumulate with N*hidden_dim elements instead of hidden_dim
    // ...

    // --- Batch FFN (same pattern as attention projections) ---
    // batch_ffn_norm, batch_gate, batch_up, batch_activation, batch_down, batch_residual
    // ...

    // --- Batch output scale (Gemma 4) ---
    // ...
}
```

### Step 4: Modify batch DMMV shader for N columns

The `dmmv_q4k_batch.comp` shader (already committed) handles this:
```glsl
// For each weight row, compute N dot products (one per input column)
for (uint col = 0; col < num_cols; col++) {
    sums[col] += w_lo * x_data[col_base + elem_lo + e] + ...;
}
```

**IMPORTANT**: The batch DMMV uses 1-thread-per-row (not K-parallel). For batch prefill this is BETTER than K-parallel because:
- Each thread reads its weight row ONCE and computes N dot products
- The N input vectors fit in L2 cache (N × K × 4 bytes = 512 × 3072 × 4 = 6MB < 96MB L2)
- Weight bandwidth dominates: read once for all N instead of N times

### Step 5: Batch RoPE shader modification

**File: `src/shaders/rope_fused.comp`**

Current RoPE takes `position` as a scalar push constant. For batch prefill, each token has a different position. Options:

**Option A (simple)**: Dispatch N separate RoPE calls, each with different `position` value. This adds N RoPE dispatches per layer but each is tiny (<0.1ms).

**Option B (optimized)**: Add `position_offset` push constant. For token t in the batch, `actual_position = position + gl_WorkGroupID.x / n_heads`. This requires restructuring workgroup dispatch to encode both head and token indices.

**Option C (most flexible)**: Add a position buffer binding that contains N position values. Each workgroup reads its position from the buffer.

**Recommendation**: Start with Option A (N separate dispatches) for correctness. Optimize to Option B/C later if RoPE becomes a bottleneck.

### Step 6: Handle batch-incompatible operations

Some operations can't trivially batch:
- **Flash attention**: Must be sequential (causal mask). N separate dispatches.
- **Output scaling** (Gemma 4 `layer_output_scale`): Reads a per-layer scalar. Apply to all N tokens with a single scale_accumulate dispatch using N*hidden_dim elements.
- **Debug readback blocks**: Skip during batch prefill (only needed at position 0 for diagnostics).

### Step 7: Fallback path

If batch buffers can't be allocated (OOM) or N exceeds max_prefill:
```zig
if (n > max_prefill or !batch_buffers_available) {
    // Fallback: sequential prefill (current behavior)
    for (prompt_tokens, 0..) |token_id, i| {
        try self.decodeStep(state, token_id, i + 1 == n);
    }
    return;
}
```

## Performance model

For Gemma 3 12B with 16-token prompt:

| Component | Sequential (current) | Batch | Savings |
|-----------|---------------------|-------|---------|
| Weight reads | 16 × 10.5GB = 168GB | 10.5GB × 1 = 10.5GB | 16× |
| Attention | 16 × 34.7ms = 555ms | 16 × 34.7ms = 555ms | 0× (still sequential) |
| FFN DMMVs | 16 × 114ms = 1824ms | ~120ms (weights once) | 15× |
| LM head | 16 × 15.7ms = 251ms | 15.7ms (last token only) | 16× |
| Barriers | 16 × ~10ms = 160ms | ~15ms (40 layers × 1) | 10× |
| **Total** | **~2790ms** | **~706ms** | **~4×** |
| **tok/s** | **5.7 tok/s** | **~22.7 tok/s** | |

Note: attention stays sequential, limiting the speedup. For longer prompts where attention dominates, the speedup would be less. For very short prompts (< 8 tokens), the overhead of batch buffer setup might not be worth it.

llama.cpp achieves 85 tok/s because they also batch the flash attention (their shader handles N query rows with causal masking). Adding batch flash attention would push ZINC to ~60-80 tok/s.

## Testing

1. **Correctness (critical)**:
   - Run batch prefill with N=1 → must match sequential decode exactly (same hidden state, same logits)
   - Run with N=16 → compare final logits with sequential 16-token prefill
   - Test with both Gemma 3 12B and Gemma 4 31B (different dimensions, V=K sharing)

2. **Performance**:
   - Measure prefill time for N=8, 16, 32, 64 tokens
   - Compare with llama.cpp's prefill time for same prompts
   - Profile with `--profile` to verify weight reads dropped by N×

3. **Edge cases**:
   - N=1 (should work identically to sequential)
   - N > max_prefill (must fallback gracefully)
   - N=0 (no-op)
   - Mixed architectures: Gemma 4 with varying head_dim per layer

## Files to modify

| File | Changes |
|------|---------|
| `src/compute/forward.zig` | Add batch buffers to init, `processLayerBatch`, modified `prefillBatch` |
| `src/shaders/dmmv_q4k_batch.comp` | Already committed, may need tweaks |
| `src/shaders/rope_fused.comp` | Add position_offset or position buffer for batch RoPE |
| `src/compute/elementwise.zig` | May need batch-aware norm/activation dispatch |
| `src/compute/dmmv.zig` | `recordBatchDispatch` already exists |

## Risk

- **High complexity**: ~500 lines of new code in `processLayerBatch`, mirroring the entire `decodeStep` layer loop
- **Correctness risk**: Many subtle indexing bugs possible (batch offsets, KV cache positions, descriptor bindings for batch buffers)
- **Memory cost**: ~150MB extra VRAM for batch buffers (acceptable on 32GB GPU, may need to reduce max_prefill on 16GB)
- **Mitigation**: Implement incrementally — start with just batch DMMV for Q/K/V, keep everything else sequential. Verify correctness. Then batch FFN. Then batch norms. Each step independently testable.
