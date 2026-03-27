# Initial Challenges: Building a GPU Inference Engine from Scratch

These are the engineering challenges we encountered building ZINC's forward pass for the Qwen3.5-35B-A3B model on AMD RDNA4 hardware, and how we solved each one.

## 1. The Empty Forward Pass

**Problem**: The initial `decodeStep` only ran three operations — token embedding, final RMS norm, and the LM head projection. It completely skipped all 40 transformer layers. Every token got embedded and immediately projected to logits, producing the same garbage output regardless of input.

**Solution**: Implemented the full 40-layer transformer loop with per-layer dispatch of RMS norm, QKV projections, RoPE, KV cache management, flash attention, output projection, MoE expert routing, and residual connections. The layer loop required careful Vulkan command buffer management — each layer needs its own descriptor set allocations and memory barriers between compute stages.

```zig
for (0..config.n_layers) |layer_idx| {
    // Reset descriptor pool for this layer's work
    _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
    try self.decode_cmd.reset();
    try self.decode_cmd.begin();

    // Input norm → QKV → RoPE → KV cache → attention → residual → FFN norm → MoE → residual
    ...
}
```

## 2. Discovering a Hybrid Architecture

**Problem**: We initially assumed Qwen3.5-35B-A3B was a standard MoE transformer. When we examined the GGUF tensor names, we found `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out` — this model is actually a **hybrid attention + SSM (delta-net) + MoE** architecture. 30 of 40 layers use SSM instead of standard attention, with the pattern `(i+1) % 4 != 0` determining recurrent vs full attention layers.

**Solution**: Implemented both execution paths. Full attention layers (every 4th) run Q/K/V projections, RoPE, flash attention with KV cache, and gated output. SSM layers use GPU DMMV for the large projections (wqkv, gate, alpha, beta, ssm_out) and CPU for the small but stateful operations (conv1d with persistent state, delta-net recurrent update, gated normalization).

```zig
const is_full_attn = ((layer + 1) % full_attn_interval == 0);

if (is_full_attn) {
    // Standard attention: Q/K/V → RoPE → KV cache → flash attention → gated output
} else {
    // SSM: GPU projections → CPU conv1d → delta-net state update → GPU output
    try self.runSsmLayerCpu(state, layer, layer_idx);
}
```

## 3. The 97% Zero Logits Bug

**Problem**: After getting all 40 layers running, the model generated diverse-looking tokens but the output was incoherent. Investigation revealed that **97% of the LM head's logits were exactly zero**. Only 7,760 of 248,320 vocabulary entries had non-zero values.

**Root cause**: The Q8_0 DMMV shader processes 2 rows per workgroup, but the dispatch code calculated workgroup count as `(M + 63) / 64` (assuming 64 rows per workgroup). For the LM head with M=248320: `(248320+63)/64 = 3880` workgroups × 2 rows = 7,760 rows — exactly 3% of the vocabulary.

**Solution**: Shader-specific dispatch count:

```zig
const workgroups_x = switch (quant_type) {
    .q8_0, .f16 => (M + 1) / 2,   // 2 rows per workgroup
    else => (M + 63) / 64,          // 1 row per thread, 64 threads per WG
};
```

This bug affected not just the LM head but every Q8_0 tensor in the model — all attention projections, embedding lookups, and shared expert weights were only computing 3% of their output rows across all 40 layers.

## 4. The GPU Hang: Flash Attention Page Table

**Problem**: On every decode step that hit a full attention layer, the GPU would hang (timeout exit 124). The engine loaded the model, ran prefill, then froze on the first decode token.

**Root cause**: The flash attention shader expects a page table buffer at binding 3 (uint array mapping logical positions to physical page IDs). The forward pass was binding the KV K-cache buffer (float data) as the page table. The shader read float values as uint page IDs, computed garbage memory offsets, and triggered an out-of-bounds GPU memory access.

**Solution**: Created an identity page table buffer where `page_ids[i] = i`, and used `page_size=1` to make the paged addressing equivalent to flat linear indexing:

```zig
// Identity page table: page_ids[i] = i (flat KV layout)
const page_table_size = @as(vk.c.VkDeviceSize, max_ctx) * @sizeOf(u32);
var page_table_buf = try Buffer.init(instance, page_table_size, ...);
const pt_u32: [*]u32 = @ptrCast(@alignCast(map_ptr));
for (0..max_ctx) |i| pt_u32[i] = @intCast(i);
```

## 5. GPT-2 Byte-Level Tokenization

**Problem**: The model generated topically relevant but incoherent text. Debugging the tokenizer revealed that `"The capital of France is"` produced 9 tokens `[760, 32, 62865, 32, 1020, 32, 47358, 32, 284]` instead of the correct 5 tokens `[760, 6511, 314, 9338, 369]`. Every space was being encoded as token 32 (`"A"`) instead of being merged into the following word with a Ġ prefix.

**Root cause**: Our BPE tokenizer split input text into raw UTF-8 characters. But GPT-2/Qwen tokenizers use a byte-to-unicode mapping where space (0x20) maps to Ġ (U+0120), and other non-printable bytes map to the U+0100+ range. Without this mapping, the BPE merge table couldn't find the right pairs to merge.

**Solution**: Implemented the full GPT-2 byte-to-unicode mapping before BPE merging:

```zig
fn gpt2ByteToUnicode(byte: u8) [4]u8 {
    const cp: u21 = switch (byte) {
        '!'...'~', 0xA1...0xAC, 0xAE...0xFF => byte,
        else => @as(u21, 256) + @as(u21, switch (byte) {
            0...0x20 => byte,      // 0-32 → U+0100..U+0120
            0x7F...0xA0 => byte - 0x7F + 33,
            0xAD => 33 + 34,
            else => byte,
        }),
    };
    // Encode codepoint as UTF-8...
}
```

## 6. Partial Rotation: IMRoPE

**Problem**: The model uses Interleaved Multi-section RoPE (IMRoPE) where only 64 of 256 head dimensions get rotary position encoding. Our RoPE shader was rotating all 256 dimensions, corrupting 75% of the Q/K vectors.

**Solution**: Extended the RoPE shader with `stride` (full head dimension) and `rope_dim` (dimensions to rotate), copying non-rotated dimensions unchanged:

```glsl
// Rotate the first rope_dim elements in pairs
for (uint i = tid; i < half_rot; i += 64) {
    // ... apply rotation
}
// Copy non-rotated dimensions unchanged
for (uint i = rope_dim + tid; i < stride; i += 64) {
    y[base_idx + i] = x[base_idx + i];
}
```

## 7. Head Dimension Mismatch

**Problem**: The model has `n_heads=16` and `hidden_dim=2048`, so our loader computed `head_dim = 2048/16 = 128`. But the actual Q/K head dimension is 256 (from `attention.key_length` in the GGUF metadata). This caused incorrect buffer sizes, wrong RoPE parameters, and misaligned attention computations.

**Solution**: Read `head_dim` from the GGUF metadata instead of computing it:

```zig
const head_dim = blk: {
    const key = std.fmt.bufPrint(&key_buf, "{s}.attention.key_length", .{prefix}) catch break :blk hidden_dim / n_heads;
    break :blk gf.getU32(key) orelse (hidden_dim / n_heads);
};
```

## 8. Missing Shared Expert

**Problem**: Qwen3.5's MoE architecture has both routed experts (256 experts, top-8 selected per token) and a **shared expert** that processes every token. The shared expert provides a stable baseline signal alongside the dynamic expert selection. We were only dispatching the routed experts.

**Solution**: Added the shared expert FFN path (gate + up → SwiGLU → down) and combined its output with the routed MoE output:

```zig
// Shared expert FFN: processes every token
if (gate_shexp != null and up_shexp != null and down_shexp != null) {
    try self.dispatchDmmv(gate_shexp.?, self.ffn_norm_buf, ...);
    try self.dispatchDmmv(up_shexp.?, self.ffn_norm_buf, ...);
    try self.elementwise.recordSwiglu(&self.decode_cmd, ...);
    try self.dispatchDmmv(down_shexp.?, self.swiglu_buf, ...);
    // Add shared expert output to MoE accumulator
    try self.elementwise.recordScaleAcc(&self.decode_cmd, ds2, hidden_dim, 1.0);
}
```

## 9. Empty KV Cache on First Decode

**Problem**: The prefill function was only processing the last prompt token through the final norm + LM head. It completely skipped all transformer layers for the prompt, leaving the KV cache empty and SSM state at zero. The first decode token had no context to attend to.

**Solution**: Replaced the shortcut prefill with a proper token-by-token forward pass through all 40 layers, populating the KV cache and SSM state:

```zig
fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
    for (prompt_tokens) |token_id| {
        try self.decodeStep(state, token_id);
    }
}
```

## 10. Mesa Driver Regression

**Problem**: The llama.cpp baseline dropped from 110 tok/s to 89 tok/s between benchmark runs. We initially suspected llama.cpp version changes, GPU throttling, or configuration drift.

**Root cause**: Ubuntu's package manager auto-updated Mesa from 25.0.7 to 25.2.8. The new RADV driver introduced a ~14% performance regression on RDNA4 cooperative matrix operations.

**Solution**: Downgraded Mesa to 25.0.7, pinned the package to prevent auto-updates, and documented the exact driver version in AGENTS.md as part of the benchmark reproducibility setup:

```bash
# /etc/apt/preferences.d/mesa-pin
Package: mesa-vulkan-drivers mesa-libgallium ...
Pin: version 25.0.7*
Pin-Priority: 1001
```

## Current State

After resolving all these challenges, ZINC processes 40 transformer layers (10 attention + 30 SSM) with MoE expert routing, native BPE tokenization, and correct Vulkan dispatch. The tokenizer matches llama.cpp's output exactly. The forward pass produces topically relevant output that responds to different prompts, though coherent sentence generation requires further work on the SSM delta-net computation and attention scaling.

The performance is 0.8 tok/s (limited by per-layer GPU submit/wait overhead and CPU-side SSM state updates), against a llama.cpp baseline of 107 tok/s. The optimization path is clear: move SSM projections to GPU, batch command buffer submissions, and reduce the 40+ per-token Vulkan submissions to a handful.
