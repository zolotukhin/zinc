# Optimization 2: Fused Gate+Up for MoE and Dense FFN

## Current State (2026-04-05)

- **Qwen3.5-35B-A3B**: 62.4 tok/s (16.0 ms/tok) on RX 9070 RDNA4
- **MoE gate+up**: 1.56 ms (40 layers, Q4_K, M=512, K=2048, 8 experts)
- **Shared expert proj**: 0.93 ms (40 layers, Q8_0, gate+up+gate_scalar)
- **Dense FFN gate+up**: used by Gemma3 (Q4_K, M=12288, K=3072)
- **Target**: Eliminate 40-80 barrier+dispatch pairs per token

## Why

Gate and up projections read the **same input vector** (`ffn_norm_buf`) and produce separate outputs (`gate_buf`, `up_buf`). Currently dispatched as two separate DMMVs with separate descriptor sets. By fusing them into a single dispatch:

1. **Input vector read once** instead of twice (saves ~8KB × 40 layers of L2 traffic)
2. **One dispatch instead of two** — halves CP overhead for this phase
3. **No barrier between gate and up** (currently they share one barrier AFTER both, so this is already optimized — but the two dispatch commands still have CP overhead)

The bigger win: paves the way for **fused gate+up+SwiGLU** which eliminates the SwiGLU dispatch AND its barrier entirely.

### Profiled savings estimate

For MoE (40 layers):
- Current: 2 dispatches per layer = 80 dispatch commands, 1 barrier after both
- Fused: 1 dispatch per layer = 40 dispatch commands, 1 barrier after
- Savings: 40 fewer vkCmdDispatch calls in the command buffer → ~0.2-0.5 ms CP overhead

For shared expert (40 layers):
- Current: 2 dispatches (gate + up) + 1 dispatch (gate_scalar) = 3 per layer
- Fused: 1 dispatch (fused gate+up) + 1 dispatch (gate_scalar) = 2 per layer
- Savings: 40 fewer dispatch calls → ~0.1-0.3 ms

**Total estimated savings**: ~0.3-0.8 ms → **63-64 tok/s (+1-2%)**

The real value is enabling the fully fused gate+up+SwiGLU which saves 40 barriers (~1 ms) on top.

## How

### Step 1: Create fused Q4_K gate+up shader

**File: `src/shaders/dmmv_q4k_fused_gate_up.comp`** (new)

This shader reads TWO weight matrices and ONE input vector, producing TWO outputs:

```glsl
#version 460
// ZINC — Fused gate+up Q4_K DMMV for MoE expert dispatch.
// Each workgroup computes one output row through BOTH gate and up weight matrices,
// reading the input vector once. Uses packed uint32 reads + vec4 dot products.
//
// 5 bindings: gate_weights, up_weights, input, gate_output, up_output
// Dispatch: ((M+63)/64, n_experts_used, 1) for MoE
//           ((M+63)/64, 1, 1) for dense FFN

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64) in;

layout(constant_id = 1) const uint SPEC_K = 4096;

layout(set = 0, binding = 0) readonly buffer GateWeights { uint gate_u32[]; };
layout(set = 0, binding = 1) readonly buffer UpWeights   { uint up_u32[];   };
layout(set = 0, binding = 2) readonly buffer VectorX     { float x_data[];  };
layout(set = 0, binding = 3) writeonly buffer OutputGate  { float y_gate[]; };
layout(set = 0, binding = 4) writeonly buffer OutputUp    { float y_up[];   };

layout(push_constant) uniform PC {
    uint M;               // output rows per expert
    uint K;               // input columns
    uint gate_stride;     // bytes per expert in gate weight tensor (MoE) or 0 (dense)
    uint up_stride;       // bytes per expert in up weight tensor (MoE) or 0 (dense)
    uint x_offset;        // input byte offset
    uint y_gate_offset;   // gate output byte offset
    uint y_up_offset;     // up output byte offset
};

shared float s_x[SPEC_K];

// ... (Q4K block decode logic: same as dmmv_q4k_moe.comp)
// ... unpack_nibbles_lo, unpack_nibbles_hi, scale/min decode ...

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint expert_slot = gl_WorkGroupID.y;

    // Load input vector into shared memory (same for both gate and up)
    uint x_base = x_offset / 4u;
    for (uint i = tid; i < K; i += 64u) {
        s_x[i] = x_data[x_base + i];
    }
    barrier();

    uint row = gl_WorkGroupID.x * 64u + tid;
    if (row >= M) return;

    // Compute gate and up dot products in a single pass over the input
    float gate_sum = 0.0;
    float up_sum = 0.0;

    uint blocks_per_row = K / 256u;
    uint gate_base = (expert_slot * gate_stride) / 4u + row * blocks_per_row * 36u;
    uint up_base = (expert_slot * up_stride) / 4u + row * blocks_per_row * 36u;

    for (uint blk = 0u; blk < blocks_per_row; blk++) {
        uint x_local = blk * 256u;

        // --- Gate weight block ---
        uint g_blk = gate_base + blk * 36u;
        // decode gate block: d, dmin, scales, qs → gate_sum += ...
        // (same Q4K decode logic as dmmv_q4k_moe.comp)

        // --- Up weight block ---
        uint u_blk = up_base + blk * 36u;
        // decode up block: d, dmin, scales, qs → up_sum += ...
        // (same Q4K decode logic, input from s_x is already in shared memory)
    }

    // Write both outputs
    uint y_g = y_gate_offset / 4u + expert_slot * M + row;
    uint y_u = y_up_offset / 4u + expert_slot * M + row;
    y_gate[y_g] = gate_sum;
    y_up[y_u] = up_sum;
}
```

**Implementation note**: The inner loop processes one Q4K block from gate weights, then one from up weights. The input vector elements for this block are already in `s_x` (shared memory), so the up pass costs zero extra input reads.

### Step 2: Add fused pipeline to dmmv.zig

**File: `src/compute/dmmv.zig`**

```zig
// Add field:
pipeline_q4k_fused_gate_up: ?Pipeline,

// Add loading (after MoE pipelines):
const FusedGateUpPush = extern struct {
    M: u32, K: u32,
    gate_stride: u32, up_stride: u32,
    x_offset: u32, y_gate_offset: u32, y_up_offset: u32,
};
const fused_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up.spv", .{shader_dir}) catch unreachable;
const pipeline_q4k_fused_gate_up = pipeline_mod.createFromSpirv(
    instance, fused_path, 5, @sizeOf(FusedGateUpPush), &spec_k, allocator,
) catch |err| blk: {
    log.warn("fused gate+up shader not loaded: {s}", .{@errorName(err)});
    break :blk null;
};

// Add dispatch method:
pub fn recordFusedGateUpMoe(
    self: *const DmmvDispatch,
    cmd: *const CommandBuffer,
    descriptor_set: vk.c.VkDescriptorSet,
    M: u32, K: u32,
    gate_stride: u32, up_stride: u32,
    n_experts_y: u32,
    x_offset: u32, y_gate_offset: u32, y_up_offset: u32,
) !void {
    const pip = self.pipeline_q4k_fused_gate_up orelse return error.UnsupportedQuantType;
    const push = FusedGateUpPush{
        .M = M, .K = K,
        .gate_stride = gate_stride, .up_stride = up_stride,
        .x_offset = x_offset, .y_gate_offset = y_gate_offset, .y_up_offset = y_up_offset,
    };
    cmd.dispatchWithPush(&pip, descriptor_set, std.mem.asBytes(&push), (M + 63) / 64, n_experts_y, 1);
}

// Add to deinit:
if (self.pipeline_q4k_fused_gate_up) |*p| p.deinit();
```

### Step 3: Add `writeDescSet5` helper if not already present

**File: `src/compute/forward.zig`**

Check if `writeDescSet5` exists. If not, add it (mirrors `writeDescSet4` with 5 bindings).

### Step 4: Wire up MoE gate+up dispatch

**File: `src/compute/forward.zig`** — in the GPU MoE path (~line 2092)

```zig
// BEFORE (two separate dispatches):
const moe_gate_up_phase = self.beginProfilePhase();
{
    // gate DMMV
    const qt = gate_exps.info.type_;
    const pip = self.dmmv.moePipelineForType(qt) orelse unreachable;
    const ds = try self.allocDescSet(pip.descriptor_set_layout);
    self.writeDescSet4(ds, gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size,
        self.ffn_norm_buf.handle, hidden_size,
        self.gate_buf.handle, self.gate_buf.size,
        self.router_output_buf.handle, self.router_output_buf.size);
    try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim,
        expert_gate_row_bytes, n_used, 0, 0, 0);
}
{
    // up DMMV (same structure)
    // ...
}
self.decode_cmd.computeBarrier();
self.endProfilePhase(.moe_gate_up, moe_gate_up_phase);

// AFTER (fused when available):
const moe_gate_up_phase = self.beginProfilePhase();
const use_fused = self.dmmv.pipeline_q4k_fused_gate_up != null and gate_quant == .q4_k;
if (use_fused) {
    const pip = &(self.dmmv.pipeline_q4k_fused_gate_up orelse unreachable);
    const ds = try self.allocDescSet(pip.descriptor_set_layout);
    self.writeDescSet5(ds,
        gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size,    // binding 0: gate weights
        up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size,        // binding 1: up weights
        self.ffn_norm_buf.handle, hidden_size,                      // binding 2: input
        self.gate_buf.handle, self.gate_buf.size,                   // binding 3: gate output
        self.up_buf.handle, self.up_buf.size,                       // binding 4: up output
    );
    try self.dmmv.recordFusedGateUpMoe(&self.decode_cmd, ds, inter_dim, hidden_dim,
        expert_gate_row_bytes, expert_gate_row_bytes, n_used, 0, 0, 0);
} else {
    // Fallback: separate dispatches (unchanged)
    // ... gate DMMV + up DMMV ...
}
self.decode_cmd.computeBarrier();
self.endProfilePhase(.moe_gate_up, moe_gate_up_phase);
```

### Step 5: Wire up shared expert gate+up

**File: `src/compute/forward.zig`** — in the shared expert section (~line 2249)

Same pattern: replace two `dispatchDmmv` calls with one fused dispatch.

### Step 6: Wire up dense FFN gate+up

**File: `src/compute/forward.zig`** — in the dense FFN section (~line 2339)

For non-MoE layers (Gemma3, Qwen3-8B), replace the two dense dispatches.

### Step 7: (Optional) Fused gate+up+SwiGLU

Add a `fuse_activation` push constant. When set:
- Compute `silu(gate_sum) * up_sum` in-place
- Write to a single `swiglu_buf` instead of separate gate/up
- Skip the separate SwiGLU dispatch + barrier

This eliminates 40 more dispatches + 40 barriers per token.

```glsl
// At the end of main():
if (fuse_activation == 2u) {
    // SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    float silu_gate = gate_sum / (1.0 + exp(-gate_sum));
    y_gate[y_g] = silu_gate * up_sum;  // write to swiglu_buf via gate output binding
} else {
    y_gate[y_g] = gate_sum;
    y_up[y_u] = up_sum;
}
```

## Testing

After each step, run:

```bash
# On RDNA node:
./zig-out/bin/zinc -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --prompt 'The capital of France is' -n 20
# → Must output "Paris."

./zig-out/bin/zinc -m /root/models/Qwen3-8B-Q4_K_M.gguf --prompt 'The capital of France is' -n 20
# → Must output "Paris."

./zig-out/bin/zinc -m /root/models/gemma-3-12b-it-Q4_K_M.gguf --prompt 'The capital of France is' -n 20
# → Must output "Paris."
```

Profile to verify gate_up time decreased:
```bash
./zig-out/bin/zinc -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --profile -n 200 --prompt 'Write an essay.'
# Check: avg MoE subphases gate_up should decrease
```

## CRITICAL Lessons from Phase 1

1. **Do NOT add barrier() to Q8_0-style shaders.** The Q8_0 DMMV uses barrier-free subgroupAdd. Adding shared memory + barrier() causes 2x regression. The fused shader should use shared memory for x (like Q4_K MoE does) since Q4_K already has barriers.

2. **Test correctness before benchmarking.** The fused residual+norm attempt produced gibberish because of SSM barrier ordering. Always verify output text first.

3. **Build incrementally.** Don't change all call sites at once. Add the fused pipeline, wire up ONE call site (e.g., MoE gate+up), test, then expand.

## Risk

- **Low risk**: The Q4K inner loop logic is identical to the existing `dmmv_q4k_moe.comp`. Just duplicated for two weight matrices sharing one input.
- **Medium risk**: 5-binding descriptor set is new. Need `writeDescSet5` helper. Flash attention already uses 5 bindings, so the infrastructure exists.
- **Register pressure**: Processing two weight blocks per iteration doubles register usage. May reduce occupancy. Monitor with profiling — if gate_up time increases, the register pressure outweighs the dispatch savings.

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/shaders/dmmv_q4k_fused_gate_up.comp` | CREATE — new fused shader |
| `build.zig` | ADD shader to compilation list |
| `src/compute/dmmv.zig` | ADD pipeline, dispatch method, deinit |
| `src/compute/forward.zig` | MODIFY MoE, shared expert, and dense FFN dispatch sites |
| `src/regression_tests.zig` | ADD test for fused shader structure |
