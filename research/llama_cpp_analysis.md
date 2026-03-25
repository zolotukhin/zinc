# llama.cpp Vulkan Backend Analysis

Analysis of llama.cpp's Vulkan backend to understand current RDNA4 inference performance
and identify opportunities for a purpose-built engine.

## Architecture Detection

RDNA4 (gfx1201) is classified as `AMD_RDNA3` — no RDNA4-specific enum exists.
RDNA3 has **no entry** in `gpu_pipeline_configs`, so `get_subgroup_size()` returns 0,
meaning the pipeline creation falls through to using the driver's default subgroup size (wave64).

## Matmul Path Selection

For single-token decode (n=1):
- `ggml_vk_should_use_mmvq()` checks if MMVQ (integer dot product quantized path) should be used
- For AMD with Q4_K and k >= 2048, MMVQ returns true IF `integer_dot_product` is enabled
- Without it, the FP16 DMMV path is used with `rm_kq=2` (2 rows per workgroup)

The MMVQ path requires `GL_EXT_integer_dot_product` GLSL extension which the default
Ubuntu glslc (shaderc 2023.8) doesn't support. Enabling it requires a newer glslc, but
newer glslc versions produce SPIR-V that RADV handles badly (5x slower).

## Existing Op Fusion

The Vulkan backend already fuses many op sequences:

| Fusion | Pattern | Dispatches Saved |
|--------|---------|-----------------|
| MULTI_ADD | N consecutive ADDs → 1 | ~280 |
| RMS_NORM_MUL | RMS_NORM + MUL → 1 | ~131 |
| TOPK_MOE | SOFTMAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV → 1 | ~360 |
| MUL_MAT_ID_MUL | MUL_MAT_ID + MUL → 1 | ~39 |
| MUL_MAT_ADD | MUL_MAT + ADD → 1 | ~9 |
| GLU | SILU/GELU + MUL (built-in op) | ~80 |

## Compute Graph (Qwen3.5-35B-A3B, single token decode)

- Total nodes: 3728
- Dispatchable ops: 2356
- After fusions: ~1500 dispatches

### Top consecutive pairs (fusion candidates)
```
ADD → ADD:                280x (handled by MULTI_ADD)
RMS_NORM → MUL:           131x (handled by RMS_NORM_MUL)
MUL → MUL_MAT:            121x
MUL_MAT → MUL_MAT:        100x
ADD → RMS_NORM:             80x
GET_ROWS → GET_ROWS:        60x
SCALE → GET_ROWS:           59x
MUL → UNARY:                58x (potential sigmoid_mul fusion)
UNARY → MUL:                51x (potential silu_mul fusion)
```

### Remaining unfused ops
```
MUL_MAT:    343 dispatches (can't fuse matmuls with each other)
UNARY:      170 dispatches (SILU, SIGMOID, SOFTPLUS)
GET_ROWS:   122 dispatches (MoE expert selection)
CPY:        121 dispatches (memory copies)
MUL:        111 dispatches (element-wise multiply)
ADD:        101 dispatches (residual connections)
GLU:         80 dispatches (already fused op)
SCALE:       60 dispatches
L2_NORM:     60 dispatches
```

## Command Buffer Submission

`nodes_per_submit = 100` — submits a command buffer every 100 nodes.

Testing showed:
- `nodes_per_submit=10000` (single submit): same performance as default
- `nodes_per_submit=10` (frequent): -8% regression

The default of 100 is already optimal.

## Why a New Engine

1. **15K+ lines of C++** in ggml-vulkan.cpp — hard to modify and extend
2. **Generic design** supporting 20+ backends — no RDNA4-specific optimization path
3. **No continuous batching** — server layer bolts it on top
4. **No paged attention** — KV cache is contiguous, limits concurrent requests
5. **SPIR-V toolchain fragility** — tightly coupled to specific glslc version
6. **Struct layout sensitivity** — adding a pipeline member can cause 20% regression due to cache effects
