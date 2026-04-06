# Optimization 3: Batch Prefill (Multi-Token Forward Pass)

## Current State (2026-04-05)

- **Decode**: 62.4 tok/s (16.0 ms/tok) — Qwen3.5-35B on RX 9070 RDNA4
- **Prefill**: ~62 tok/s (sequential, one token at a time, reads ALL weights per token)
- **llama.cpp prefill**: ~500+ tok/s (batched, reads weights once for all tokens)
- **Target**: 200-400 tok/s prefill for Qwen3.5-35B

## Why

ZINC processes prompt tokens one at a time — each token reads ALL model weights from VRAM. For a 64-token prompt:

- **Current (sequential)**: 64 × ~1.28 GB = **82 GB of weight reads** → 82/576 = 142ms minimum → actual ~1.0s
- **Batch (read weights once)**: ~1.28 GB × 1 = **1.28 GB of reads** + 64 × compute → ~25ms minimum
- **Speedup**: ~4-8× faster prefill depending on prompt length

This eliminates the multi-second wait before first token for long prompts.

### Model-specific details (Qwen3.5-35B-A3B)

Per-token weight reads: ~1.28 GB (see RDNA4_PERFORMANCE_PLAN.md).
Bytes breakdown:
- SSM projections (qkv+gate+alpha+beta+out, 30 layers, Q8_0): 420 MB
- Shared expert (gate+up+down, 40 layers, Q8_0): 240 MB
- LM head (Q4_K): 289 MB
- MoE experts (8/256 active, gate+up+down, 40 layers, Q4_K/Q5_K): 192 MB
- Attention Q/K/V/O (10 layers, Q4_K): 120 MB
- MoE router (40 layers, F32): 12 MB
- Norms, SSM constants: ~5 MB

With batch prefill, ALL weights are read once and multiplied against N input vectors.

## How

### Step 1: Allocate batch intermediate buffers

**File: `src/compute/forward.zig` init section (~line 614)**

Add an `InferenceEngine` field for max prefill batch size:

```zig
const max_prefill_batch: u32 = 256; // max tokens per batch prefill
batch_hidden_buf: ?Buffer = null,
batch_norm_buf: ?Buffer = null,
batch_ffn_norm_buf: ?Buffer = null,
batch_q_buf: ?Buffer = null,
batch_k_buf: ?Buffer = null,
batch_v_buf: ?Buffer = null,
batch_attn_out_buf: ?Buffer = null,
batch_o_proj_buf: ?Buffer = null,
batch_gate_buf: ?Buffer = null,
batch_up_buf: ?Buffer = null,
batch_swiglu_buf: ?Buffer = null,
batch_down_buf: ?Buffer = null,
batch_moe_out_buf: ?Buffer = null,
batch_embed_staging: ?Buffer = null,
```

Allocate during init (after single-token buffers). Wrap in `catch null` so OOM doesn't crash:

```zig
const n = max_prefill_batch;
self.batch_hidden_buf = Buffer.initDeviceLocal(instance, @as(VkDeviceSize, n) * hidden_size, storage_xfer) catch null;
self.batch_norm_buf = Buffer.initDeviceLocal(instance, @as(VkDeviceSize, n) * hidden_size, storage_xfer) catch null;
// ... etc for all batch buffers ...
self.batch_embed_staging = Buffer.initStaging(instance, @as(VkDeviceSize, n) * hidden_size) catch null;
```

**Memory cost for Qwen3.5-35B** (hidden=2048, inter=512 MoE, q=256, kv=256, N=256):
- hidden/norm/ffn_norm/o_proj/down/moe_out: 6 × 256 × 2048 × 4 = 12.6 MB
- gate/up/swiglu: 3 × 256 × 512 × 8_experts × 4 = 12.6 MB (batched MoE)
- q/attn_out: 2 × 256 × 256 × 4 = 0.5 MB
- k/v: 2 × 256 × 256 × 4 = 0.5 MB
- embed staging: 256 × 2048 × 4 = 2 MB
- **Total: ~28 MB** — negligible on 16+ GB VRAM

**Build check**: `zig build test` — buffers allocated but unused.

### Step 2: Batch DMMV dispatch support (already partially built)

**File: `src/compute/dmmv.zig`** — `recordBatchDispatch` already exists.

Verify it works for all quant types used:
- Q8_0 (SSM projections) — needs batch Q8_0 shader
- Q4_K (MoE gate/up, attention) — `dmmv_q4k_batch.comp` already committed
- Q5_K (MoE down) — needs batch Q5_K shader
- F32 (MoE router) — needs batch F32 shader

**If batch shaders are missing**, create them. The batch shader pattern is:

```glsl
// Weight matrix read ONCE, multiplied against num_cols input columns
layout(push_constant) uniform PC {
    uint M;          // output rows
    uint K;          // input columns
    uint a_offset;   // weight byte offset
    uint x_offset;   // input byte offset (column-major: X[K, num_cols])
    uint y_offset;   // output byte offset (column-major: Y[M, num_cols])
    uint num_cols;   // number of input columns (batch size)
};

void main() {
    uint row = gl_WorkGroupID.x * 64 + tid;
    if (row >= M) return;

    // Read weight row ONCE
    // ... dequantize weight[row][0..K-1] ...

    // Compute dot product for each input column
    for (uint col = 0; col < num_cols; col++) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += weight_val[k] * x_data[col * K + k];
        }
        y[col * M + row] = sum;
    }
}
```

### Step 3: Batch elementwise operations

**RMS norm already supports batching**: The `gl_WorkGroupID.x` indexes into the token dimension. Dispatching with `N` workgroups processes N tokens.

```zig
// Single token:
try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, eps);

// Batch of N tokens:
try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, n, eps);
```

The shader processes `token = gl_WorkGroupID.x`, reading from `x[token * N .. (token+1)*N]`.

**SwiGLU/GEGLU**: Same — just dispatch with `n * inter_dim` elements instead of `inter_dim`.

**scale_acc (residual add)**: Same — `n * hidden_dim` elements.

### Step 4: Batch RoPE handling

**Current**: RoPE uses `position` as a scalar push constant. For batch prefill, each token has a different position.

**Option A (simplest, start here)**: Dispatch N separate RoPE calls in a loop:

```zig
for (0..n) |t| {
    state.position = @intCast(t);
    // Record RoPE dispatch for batch_q_buf[t * q_dim..(t+1) * q_dim]
    // with x_offset = t * q_dim * sizeof(f32)
    try self.elementwise.recordRoPE(&self.decode_cmd, ds_t, ...);
}
self.decode_cmd.computeBarrier();
```

N RoPE dispatches is negligible overhead — each is ~2 µs (tiny elementwise kernel).

**Option B (later optimization)**: Add position buffer binding to RoPE shader so all tokens are processed in one dispatch.

### Step 5: Sequential attention within batch

Flash attention must be sequential (causal: token i attends to 0..i only):

```zig
for (0..n) |t| {
    const seq_len = t + 1; // token t can see tokens 0..t
    // Flash attention dispatch for batch_q[t], KV cache [0..t], out = batch_attn_out[t]
    // ...
}
self.decode_cmd.computeBarrier();
```

This is the same cost as sequential prefill for attention, but the DMMV weight reads are batched.

### Step 6: MoE batching (Qwen3.5 specific)

For MoE layers, each token in the batch may select DIFFERENT experts. This complicates batching:

**Option A (simplest)**: Process MoE sequentially per token within the batch. Only batch the non-MoE parts (SSM projections, shared expert, norms, residuals).

**Option B (advanced)**: Group tokens by expert selection. After batch router DMMV + batch topk, gather tokens per expert and dispatch each expert's gate/up/down as a batch. This requires:
- Batch router DMMV: weight × X[K, N] → logits[256, N]
- Per-token topk: select top-8 experts per token
- Expert grouping: build per-expert token lists
- Per-expert batch DMMV: expert_weight × gathered_inputs → outputs
- Scatter outputs back to per-token positions

**Recommendation**: Start with Option A. MoE is 40 of 40 FFN layers for Qwen3.5-35B, so sequential MoE limits the batch speedup. But SSM projections (30 layers, 420 MB/tok) still benefit hugely from batching.

### Step 7: Implement `prefillBatch` function

**File: `src/compute/forward.zig`**

Replace the existing sequential loop in `prefillBatch`:

```zig
pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
    try self.resetRequestState();
    const n: u32 = @intCast(prompt_tokens.len);
    if (n == 0) return;

    // Check if batch buffers are available
    const batch_available = self.batch_hidden_buf != null and n <= max_prefill_batch;
    if (!batch_available) {
        // Fallback: sequential prefill
        for (prompt_tokens, 0..) |token_id, i| {
            try self.decodeStep(state, token_id, i + 1 == n);
        }
        return;
    }

    // Phase 1: CPU batch embed all tokens
    const staging: [*]f32 = @ptrCast(@alignCast(self.batch_embed_staging.?.mapped.?));
    for (prompt_tokens, 0..) |token_id, i| {
        self.loadTokenEmbedding(token_id, staging[i * hidden_dim ..][0..hidden_dim]);
    }

    // Phase 2: Upload all embeddings
    try self.decode_cmd.begin();
    const upload_size = @as(VkDeviceSize, n) * hidden_size;
    const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = upload_size };
    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle,
        self.batch_embed_staging.?.handle, self.batch_hidden_buf.?.handle, 1, &region);
    self.decode_cmd.transferToComputeBarrier();

    // Phase 3: Process each layer
    for (0..config.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const is_attn = (config.full_attn_interval > 0 and layer % config.full_attn_interval == 0);

        // Batch pre-attention/SSM norm
        try self.batchRmsNorm(self.batch_hidden_buf.?, self.batch_norm_buf.?, attn_norm_tensor, n);
        self.decode_cmd.computeBarrier();

        if (is_attn) {
            // Batch Q/K/V projection
            try self.batchDmmv(q_tensor, self.batch_norm_buf.?, self.batch_q_buf.?, q_dim, hidden_dim, n);
            try self.batchDmmv(k_tensor, self.batch_norm_buf.?, self.batch_k_buf.?, kv_dim, hidden_dim, n);
            try self.batchDmmv(v_tensor, self.batch_norm_buf.?, self.batch_v_buf.?, kv_dim, hidden_dim, n);
            self.decode_cmd.computeBarrier();

            // Per-token: Q/K norms, RoPE, KV cache write, attention, O projection
            for (0..n) |t| {
                // ... sequential attention per token ...
                state.position = @intCast(t);
            }
            self.decode_cmd.computeBarrier();

            // Batch O projection
            try self.batchDmmv(o_tensor, self.batch_attn_out_buf.?, self.batch_o_proj_buf.?, hidden_dim, q_dim, n);
            self.decode_cmd.computeBarrier();

            // Batch residual: batch_hidden += batch_o_proj
            try self.batchScaleAcc(self.batch_hidden_buf.?, self.batch_o_proj_buf.?, n * hidden_dim, 1.0);
            self.decode_cmd.computeBarrier();
        } else {
            // SSM: batch projections, sequential state update
            try self.batchDmmv(ssm_qkv, self.batch_norm_buf.?, self.batch_attn_out_buf.?, ssm_dim, hidden_dim, n);
            // ... per-token SSM state update (conv1d, delta-net, gated_norm) ...
            // ... batch ssm_out projection + residual ...
        }

        // Batch FFN norm
        try self.batchRmsNorm(self.batch_hidden_buf.?, self.batch_ffn_norm_buf.?, ffn_norm_tensor, n);
        self.decode_cmd.computeBarrier();

        // MoE: sequential per token (different expert selections)
        for (0..n) |t| {
            // ... MoE router + topk + expert dispatch for token t ...
        }

        // Batch FFN residual
        try self.batchScaleAcc(self.batch_hidden_buf.?, self.batch_moe_out_buf.?, n * hidden_dim, 1.0);
        self.decode_cmd.computeBarrier();
    }

    // Phase 4: Final norm + LM head for LAST token only
    // Copy last token's hidden state to single-token hidden_buf
    const last_offset = @as(VkDeviceSize, n - 1) * hidden_size;
    const copy_region = VkBufferCopy{ .srcOffset = last_offset, .dstOffset = 0, .size = hidden_size };
    self.decode_cmd.computeToTransferBarrier();
    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle,
        self.batch_hidden_buf.?.handle, self.hidden_buf.handle, 1, &copy_region);
    self.decode_cmd.transferToComputeBarrier();

    // Normal final norm + LM head + argmax
    try self.dispatchFinalTail(state, true);

    try self.decode_cmd.end();
    try self.decode_cmd.submitAndWait(self.instance.compute_queue);
    state.position = n;
}
```

### Step 8: Helper functions for batch operations

```zig
fn batchRmsNorm(self: *InferenceEngine, input: Buffer, output: Buffer, weight_tensor: anytype, n: u32) !void {
    const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
    const ds = try self.allocDescSet(pip.descriptor_set_layout);
    const batch_size = @as(VkDeviceSize, n) * @as(VkDeviceSize, self.model.config.hidden_dim) * @sizeOf(f32);
    self.writeDescSet3(ds, input.handle, batch_size, weight_tensor.gpu_buffer.handle, weight_tensor.gpu_buffer.size, output.handle, batch_size);
    try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, self.model.config.hidden_dim, n, self.model.config.rms_norm_eps);
}

fn batchDmmv(self: *InferenceEngine, tensor: anytype, input: Buffer, output: Buffer, M: u32, K: u32, n: u32) !void {
    try self.dmmv.recordBatchDispatch(&self.decode_cmd, tensor.info.type_, ds, M, K, 0, 0, 0, n);
}

fn batchScaleAcc(self: *InferenceEngine, accum: Buffer, src: Buffer, n_elements: u32, scale: f32) !void {
    const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
    const ds = try self.allocDescSet(pip.descriptor_set_layout);
    const size = @as(VkDeviceSize, n_elements) * @sizeOf(f32);
    self.writeDescSet2(ds, accum.handle, size, src.handle, size);
    try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, n_elements, scale);
}
```

## Implementation Order (incremental)

1. **Allocate batch buffers** → build, no behavior change
2. **Batch embedding upload** → replaces sequential embed loop
3. **Batch RMS norm** → replaces sequential norm dispatches
4. **Batch DMMV for SSM projections** → biggest data savings (420 MB/tok × N reduction)
5. **Batch residual add** → simple element count change
6. **Sequential attention within batch** → correctness-critical
7. **Batch O projection** → saves K weight reads for attention layers
8. **Batch FFN norm** → small savings
9. **Sequential MoE within batch** → maintains correctness, no batching yet
10. **Batch LM head skip** → only compute for last token (saves 289 MB × (N-1))
11. **Test with N=1** → must match sequential decode exactly
12. **Test with N=16, 64** → measure prefill speedup

## Expected Performance

For Qwen3.5-35B with 64-token prompt:

| Component | Sequential | Batch | Savings |
|-----------|-----------|-------|---------|
| SSM proj (30 layers, Q8_0) | 64 × 420MB = 26.9 GB | 420 MB | 64× |
| Shared expert (40 layers, Q8_0) | 64 × 240MB = 15.4 GB | 240 MB | 64× |
| MoE (40 layers, top-8) | 64 × 192MB = 12.3 GB | sequential | 1× |
| Attention (10 layers) | sequential | sequential | 1× |
| LM head | 64 × 289MB = 18.5 GB | 289 MB | 64× |
| **Total weight reads** | **~82 GB** | **~2.4 GB** | **~34×** |

With attention+MoE sequential bottleneck: **~4-8× overall prefill speedup**.

## Models to Test

| Model | Architecture | Test point |
|-------|-------------|------------|
| Qwen3.5-35B | MoE + SSM | Primary target |
| Qwen3.5-2B | SSM (no MoE) | Full batch benefit |
| Gemma3-12B | Dense attention | Full batch benefit |

## CRITICAL Lessons from Phase 1

1. **Do NOT add barrier() to Q8_0 shaders.** The batch Q8_0 DMMV must NOT use shared memory + barrier(). Use the barrier-free subgroupAdd pattern.

2. **Buffer-specific barriers don't help on RADV.** Use global `computeBarrier()` — it's just as fast.

3. **Atomic cross-workgroup sync is SLOWER than separate dispatch+barrier on RDNA4.** Don't try to fuse router+topk with atomics.

4. **Test output text after EVERY change.** Gibberish output is a sign of barrier ordering bugs.

## Risk

- **High complexity**: ~300-500 lines of new code in `prefillBatch`, mirroring the decode loop
- **Correctness**: Batch buffer indexing must be exact. KV cache position tracking across batch tokens is error-prone.
- **Memory**: ~28 MB extra VRAM for Qwen3.5-35B (negligible). For larger models, may need to reduce max_prefill_batch.
- **Mitigation**: Implement incrementally. Start with batch embedding + batch norms + sequential everything else. Each step independently testable.

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/compute/forward.zig` | ADD batch buffers, batch helpers, new `prefillBatch` |
| `src/shaders/dmmv_q8_0_batch.comp` | CREATE — batch Q8_0 DMMV (if not exists) |
| `src/shaders/dmmv_q5k_batch.comp` | CREATE — batch Q5_K DMMV (for MoE down) |
| `src/shaders/dmmv_f32_batch.comp` | CREATE — batch F32 DMMV (for MoE router) |
| `src/compute/dmmv.zig` | ADD batch pipeline loading for new quant types |
| `build.zig` | ADD new batch shaders to compilation list |
| `src/compute/elementwise.zig` | No changes needed (RMS norm already supports N workgroups) |
