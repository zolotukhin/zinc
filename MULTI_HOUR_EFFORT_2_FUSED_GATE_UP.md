# Optimization 2: Fused Gate+Up DMMV

## Why

The dense FFN path dispatches gate and up projections as **two separate DMMV calls** that read from the **same input buffer** (`ffn_norm_buf`). Each dispatch:
- Reads the entire weight tensor from VRAM (21.2MB for Gemma 3, K=3072, M=12288)
- Reads the input vector (12KB) from global memory / L2 cache
- Has Vulkan dispatch overhead (~50us per dispatch including pipeline bind + descriptor bind)

Together they account for **40 barrier+dispatch pairs per token** (one per layer). The input vector read is duplicated — both gate and up read the same 12KB `ffn_norm_buf`.

More importantly: each dispatch pair includes a **barrier** between activation and the gate+up pair, plus the barrier AFTER gate+up. By fusing gate+up into a single dispatch, we eliminate **40 dispatch overheads** and **40 input vector re-reads** per token.

**Expected savings**: 40 dispatches × ~50us = **~2ms per token** → from 168ms to ~166ms → **~1.2% decode speedup**. Small but cumulative with other optimizations.

The bigger value: this lays the groundwork for **fused gate+up+activation** (a single dispatch that does `output = gelu(W_gate @ x) * (W_up @ x)`), which eliminates the activation dispatch and barrier too — saving another 40 dispatches.

## What

Create a new GLSL shader `dmmv_q4k_fused_gate_up.comp` that:
1. Reads ONE input vector from global memory
2. Reads from TWO weight tensors (gate and up)
3. Writes TWO output vectors (gate_buf and up_buf)
4. Optionally fuses the activation (GEGLU/SwiGLU) to write ONE output (swiglu_buf)

### Architecture constraint
Both gate and up projections have identical dimensions:
- Same M (= inter_dim = 12288 for Gemma 3, 21504 for Gemma 4)
- Same K (= hidden_dim = 3072 for Gemma 3, 5376 for Gemma 4)
- Same quantization (Q4_K for both in Q4_K_M models)

This makes fusion straightforward — same workgroup count, same inner loop, just double the weight reads and outputs.

## How

### Step 1: Create fused shader

**File: `src/shaders/dmmv_q4k_fused_gate_up.comp`** (new file)

```glsl
#version 460
// Fused Gate+Up DMMV: reads one input, two weight matrices, writes two outputs.
// Saves one dispatch + one input vector read vs separate gate and up dispatches.
// Optionally fuses GEGLU activation to produce swiglu output directly.

layout(local_size_x = 64) in;

layout(push_constant) uniform PushConstants {
    uint M;           // output rows (inter_dim)
    uint K;           // input columns (hidden_dim)
    uint a_gate_offset; // byte offset for gate weight tensor
    uint a_up_offset;   // byte offset for up weight tensor
    uint x_offset;      // input vector byte offset
    uint y_gate_offset; // gate output byte offset
    uint y_up_offset;   // up output byte offset
    uint fuse_activation; // 0=none, 1=GEGLU, 2=SwiGLU
};

layout(set = 0, binding = 0) readonly buffer GateWeights { uint8_t gate_data[]; };
layout(set = 0, binding = 1) readonly buffer UpWeights   { uint8_t up_data[];   };
layout(set = 0, binding = 2) readonly buffer VectorX     { float x_data[];      };
layout(set = 0, binding = 3) writeonly buffer OutputGate  { float y_gate[];      };
layout(set = 0, binding = 4) writeonly buffer OutputUp    { float y_up[];        };

// Inner loop: same Q4K dequantize logic as dmmv_q4k.comp, but done twice
// (once for gate weights, once for up weights) sharing the same input vector reads.
void main() {
    uint row = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x;
    if (row >= M) return;

    // Read input vector ONCE (shared between gate and up)
    // ... Q4K dequant loop for gate_data → gate_sum ...
    // ... Q4K dequant loop for up_data → up_sum (reusing cached x_data) ...

    // Write both outputs
    y_gate[row] = gate_sum;
    y_up[row] = up_sum;

    // Optional: fuse activation
    // if (fuse_activation == 1) y_out[row] = up_sum * gelu(gate_sum);  // GEGLU
    // if (fuse_activation == 2) y_out[row] = up_sum * silu(gate_sum);  // SwiGLU
}
```

**Key design choices:**
- **5 buffer bindings** (gate_weights, up_weights, input, gate_output, up_output). With fused activation: 6 bindings (add swiglu_output).
- **1-thread-per-row** (not K-parallel): each thread processes one output row through BOTH weight matrices. This keeps the input vector in registers/L1 across both dequant loops.
- The inner loop reads each Q4K block from gate_data and up_data sequentially. The x_data reads hit L1 cache on the second pass (up) since they were just read for gate.

### Step 2: Add pipeline and dispatch

**File: `src/compute/dmmv.zig`**

Add fields:
```zig
pipeline_q4k_fused_gate_up: ?Pipeline,
```

Add pipeline loading (after batch pipeline):
```zig
const FusedGateUpPush = extern struct {
    M: u32, K: u32, a_gate_offset: u32, a_up_offset: u32,
    x_offset: u32, y_gate_offset: u32, y_up_offset: u32, fuse_activation: u32,
};
const fused_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up.spv", .{shader_dir}) catch unreachable;
const pipeline_fused = pipeline_mod.createFromSpirv(instance, fused_path, 5, @sizeOf(FusedGateUpPush), &.{}, allocator) catch null;
```

Add dispatch method:
```zig
pub fn recordFusedGateUp(self, cmd, gate_tensor, up_tensor, input_ds, M, K, fuse_activation) !void {
    // Single dispatch: weight reads for both gate and up, shared input
}
```

### Step 3: Integrate into forward.zig FFN path

**File: `src/compute/forward.zig:2477-2500`** (dense FFN section)

**BEFORE:**
```zig
// Line 2481-2482: Two separate dispatches
try self.dispatchDmmv(gate_tensor, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim);
try self.dispatchDmmv(up_tensor, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim);
self.decode_cmd.computeBarrier();

// Line 2485-2495: Activation dispatch
if (use_geglu) {
    ... recordGeglu(gate_buf, up_buf → swiglu_buf) ...
} else {
    ... recordSwiglu(gate_buf, up_buf → swiglu_buf) ...
}
self.decode_cmd.computeBarrier();
```

**AFTER (unfused — just gate+up merged):**
```zig
if (gate_tensor.info.type_ == .q4_k and self.dmmv.pipeline_q4k_fused_gate_up != null) {
    // Fused gate+up: single dispatch, shared input vector read
    try self.dispatchFusedGateUp(gate_tensor, up_tensor, self.ffn_norm_buf, hidden_size,
        self.gate_buf, self.up_buf, inter_dim, hidden_dim, 0);
} else {
    try self.dispatchDmmv(gate_tensor, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim);
    try self.dispatchDmmv(up_tensor, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim);
}
self.decode_cmd.computeBarrier();
// Activation unchanged
```

**AFTER (fully fused — gate+up+activation merged):**
```zig
if (gate_tensor.info.type_ == .q4_k and self.dmmv.pipeline_q4k_fused_gate_up != null) {
    // Fused gate+up+activation: single dispatch, no separate activation dispatch
    const fuse_mode: u32 = if (use_geglu) 1 else 2;
    try self.dispatchFusedGateUp(gate_tensor, up_tensor, self.ffn_norm_buf, hidden_size,
        self.swiglu_buf, self.swiglu_buf, inter_dim, hidden_dim, fuse_mode);
    self.decode_cmd.computeBarrier();
    // Skip separate activation — already fused
} else {
    // Fallback to separate dispatches
    try self.dispatchDmmv(gate_tensor, ...);
    try self.dispatchDmmv(up_tensor, ...);
    self.decode_cmd.computeBarrier();
    ... activation ...
    self.decode_cmd.computeBarrier();
}
```

### Step 4: Handle descriptor set with 5 bindings

The fused shader needs 5 bindings (gate_weights, up_weights, input, output_gate, output_up). This is a new layout. Either:
- Use the same `writeDescSet5` helper (already exists for flash attention)
- Create a new `writeDescSet5` variant if the existing one has incompatible binding types

### Shader implementation detail

The inner loop processes one Q4K block from BOTH weight tensors:
```glsl
for (uint blk = 0; blk < blocks_per_row; blk++) {
    // Read shared input elements for this block
    // (these stay in registers/L0 cache)

    // Gate weight dequant + accumulate
    uint gate_off = a_gate_offset + row * bpr * 144 + blk * 144;
    // ... read gate block, dequant, multiply with input ...
    gate_sum += ...;

    // Up weight dequant + accumulate (input ALREADY in cache)
    uint up_off = a_up_offset + row * bpr * 144 + blk * 144;
    // ... read up block, dequant, multiply with input ...
    up_sum += ...;
}
```

The key: **input vector elements are loaded once and reused across both weight matrices**. On RDNA4, L0 cache is 16KB per CU — the 12KB input vector (K=3072) fits entirely and stays hot between the two dequant passes.

## Testing

1. **Correctness**: Must match separate gate+up to within f32 precision
   - Run with fused path → output text must be identical to non-fused
   - Add a one-time verification: dispatch BOTH fused and separate, compare outputs

2. **Performance**: Profile with `--profile`
   - Check that the FFN untracked time drops by ~2ms (from ~115ms to ~113ms)
   - Total tok/s should improve by ~1-2%

3. **Models**: Test on both Gemma 3 12B and Gemma 4 31B (different hidden_dim/inter_dim)

## Risk

- **Low risk for unfused (gate+up only)**: Same dequant math, just two passes per thread instead of two dispatches. Easy to verify.
- **Medium risk for fully fused (gate+up+activation)**: The activation (GELU tanh approximation) adds register pressure. May hurt occupancy on RDNA4 if VGPRs exceed ~48 per thread.
- **Binding count**: 5 bindings is fine (flash attention already uses 5). Fully fused with separate swiglu_output would need 6 — still within Vulkan limits.
