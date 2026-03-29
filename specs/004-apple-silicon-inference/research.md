# Research: Apple Silicon Inference for ZINC

**Date**: 2026-03-28
**Hardware**: Mac Studio M4 Max, 64 GB unified memory, 546 GB/s bandwidth, 40-core GPU

## 1. Competitive Landscape

### Current Best Performance (M4 Max, Qwen3.5-35B-A3B, 4-bit)

| Engine | Single-request tok/s | Parallel (5 req) | Architecture |
|--------|---------------------|-------------------|--------------|
| **MetalRT** (RunAnywhere) | ~75-80 | N/A | C++ → Metal direct |
| **vllm-mlx** | 60-75 | ~233 aggregate | Python → MLX → Metal |
| **mlx-lm** | 60-75 | N/A (sequential) | Python → MLX → Metal |
| **llama.cpp Metal** | 30-38 | N/A (sequential) | C++ → ggml → Metal |
| **Ollama** | 30-38 | N/A (sequential) | Go → llama.cpp → Metal |

### Broader Benchmarks (M4 Max 128 GB, vllm-mlx paper arXiv 2601.19139)

| Model | vllm-mlx | mlx-lm | llama.cpp | vllm-mlx vs llama.cpp |
|-------|----------|--------|-----------|----------------------|
| Qwen3-0.6B | 525 tok/s | 356 | 281 | 1.87x |
| Qwen3-4B | 159 | 129 | 118 | 1.35x |
| Qwen3-8B | 93 | 80 | 77 | 1.21x |
| Qwen3-30B (MoE, 4-bit) | 110 | 107 | 90 | 1.22x |
| Nemotron-30B | 122 | 102 | 85 | 1.43x |

### MetalRT (RunAnywhere) — Current Speed King

On M4 Max 64 GB:
- Qwen3-0.6B: **658 tok/s**
- Qwen3-4B: **186 tok/s**
- 1.35-2.14x faster than llama.cpp, 1.10-1.19x faster than mlx-lm
- Approach: "throw away every abstraction and go straight to Metal" — native C++ with direct Metal API calls
- Proprietary, closed-source

### Why MLX is Faster than llama.cpp on Apple Silicon

1. **Zero-copy unified memory**: MLX is built natively for UMA; llama.cpp's Metal backend was bolted onto a cross-platform abstraction layer
2. **Lazy evaluation + operation fusion**: Graph-based execution fuses operations → fewer Metal dispatches
3. **Specialized quantized kernels**: Template-generated Metal kernels (QMV, QVM, QMM, GatherQMM) purpose-built for quantized inference
4. **Lower framework overhead**: Less abstraction between API and Metal dispatch

### When llama.cpp Wins

- Faster prefill for short-output workloads
- K-quant quality (Q4_K_M > MLX 4-bit affine at same bits-per-weight)
- Cross-platform GGUF compatibility
- Ecosystem maturity

---

## 2. Memory Bandwidth Analysis

### Theoretical Limits

Token generation is **memory-bandwidth bound** at batch size 1. Each token requires reading active model weights once.

```
theoretical_tok/s = bandwidth / active_model_bytes
```

For Qwen3.5-35B-A3B (MoE: 35B total, ~3B active per token):
- Active params at Q4: ~1.5 GB reads per token
- M4 Max (546 GB/s): **theoretical ~364 tok/s** for active params only
- But router weights, embeddings, norms, KV cache reads add overhead
- Realistic ceiling: **~150-200 tok/s** accounting for all memory traffic

### Bandwidth Utilization (measured, llama.cpp on M4 Max)

| Quant | Model Size | Theoretical | Actual | Utilization |
|-------|-----------|-------------|--------|-------------|
| Q4_0 (7B) | 3.5 GB | 156 tok/s | 83 tok/s | **53%** |
| Q8_0 (7B) | 7 GB | 78 tok/s | 54 tok/s | **69%** |
| F16 (7B) | 14 GB | 39 tok/s | 32 tok/s | **82%** |

Key insight: larger sequential reads → higher utilization. Q4 suffers from dequant overhead + scattered access patterns.

### Apple Silicon Bandwidth by Chip

| Chip | Max RAM | Bandwidth | Est. MoE Q4 tok/s |
|------|---------|-----------|-------------------|
| M4 | 32 GB | 120 GB/s | ~25-35 |
| M4 Pro | 48 GB | 273 GB/s | ~55-70 |
| **M4 Max (40c)** | **128 GB** | **546 GB/s** | **~75-120** |
| M3 Ultra | 192 GB | 819 GB/s | ~110-170 |
| M5 Max (upcoming) | TBD | 614 GB/s | ~85-135 |

Note: M4 Ultra was cancelled (no UltraFusion on M4 Max die). Next Ultra = M5 Ultra (late 2026).

---

## 3. Zig → Metal Integration Options

### Approach A: ObjC Runtime via `objc_msgSend` (Pure Zig)

Zig can't `@cImport` Objective-C headers, but can call the ObjC runtime directly:
- `objc_getClass`, `sel_registerName`, `objc_msgSend` — all plain C functions
- Library: **mitchellh/zig-objc** (by Ghostty creator, works with Zig 0.14+)
- **Pros**: No external compilers, pure Zig, minimal overhead
- **Cons**: Verbose — every Metal API call is a `msgSend` with string selectors

### Approach B: Objective-C Shim (Recommended for production)

Write a thin `metal_shim.m` exposing a C API that Zig `@cImport`s:
```c
// metal_shim.h
MetalContext* metal_init(void);
MetalBuffer* metal_create_buffer(MetalContext*, size_t, void** cpu_ptr);
void metal_dispatch(MetalContext*, MetalPipeline*, uint3 grid, uint3 block);
```

Build in `build.zig`:
```zig
const shim = b.addObject(.{ .name = "metal_shim" });
shim.addCSourceFile(.{ .file = b.path("src/metal/shim.m"), .flags = &.{"-fobjc-arc"} });
shim.linkFramework("Metal");
shim.linkFramework("Foundation");
exe.addObject(shim);
```

**This is exactly what llama.cpp does** (`ggml-metal.m`). Proven pattern.

### Approach C: Auto-generated Bindings

- `colbyhall/objective-zig` — covers Metal + 40 Apple frameworks, WIP
- `dmbfm/zig-metal` — archived Feb 2024, alpha
- Not recommended for production use

### Recommendation: Approach B (ObjC Shim)

Mirrors ZINC's current architecture: Zig host code with a thin C-API abstraction over the GPU backend. The `vulkan/` directory becomes paralleled by a `metal/` directory with identical buffer/pipeline/command abstractions backed by Metal instead of Vulkan.

---

## 4. Metal Compute vs Vulkan Compute

### Shader Translation

| Aspect | Vulkan (current) | Metal |
|--------|-----------------|-------|
| Language | GLSL 4.60 → SPIR-V | MSL (C++14 dialect) |
| Workgroup | `shared`, `barrier()` | `threadgroup`, `threadgroup_barrier()` |
| Thread ID | `gl_GlobalInvocationID` | `thread_position_in_grid` |
| Subgroup | `subgroupAdd()` | `simd_sum()` (always 32 threads) |
| Coop matrix | `VK_KHR_cooperative_matrix` | `simdgroup_matrix` (8x8) |
| WG size | Declared in shader | Specified at dispatch time |
| Barriers | Manual pipeline barriers | Automatic hazard tracking |

### SPIR-V → MSL Cross-compilation

**SPIRV-Cross** can convert existing SPIR-V to MSL:
```bash
spirv-cross --msl shader.spv --output shader.metal
```

Works for: all non-cooperative-matrix shaders (dmmv, flash_attn, rms_norm, swiglu, rope, etc.)

**Does NOT work for**: `VK_KHR_cooperative_matrix` → `simdgroup_matrix`. These have fundamentally different semantics and need manual MSL rewrites.

### Metal Advantages for ZINC

1. **No staging buffers**: `MTLResourceStorageModeShared` gives CPU+GPU the same pointer. Model loading = `mmap()` + `newBufferWithBytesNoCopy:` — zero copy
2. **Simpler pipeline**: No descriptor sets, no pipeline layouts, no descriptor pools
3. **Automatic hazard tracking**: No manual memory barriers (can opt out for perf)
4. **Dispatch-time workgroup size**: More flexibility for per-kernel tuning

---

## 5. Unified Memory Architecture — Impact on ZINC

### Current Vulkan Path (RDNA4)
```
mmap GGUF → CPU staging buffer → vkCmdCopyBuffer → GPU device-local buffer
```

### Metal Path (Apple Silicon)
```
mmap GGUF → newBufferWithBytesNoCopy → done (GPU reads mmap'd pages directly)
```

Key implications:
- **No DMA uploads**: Entire model loading pipeline simplifies dramatically
- **No staging buffers for readback**: Router logits, final logits can be read directly from GPU buffer pointers
- **KV cache**: CPU and GPU share the same memory — SSM state updates (currently CPU) can write directly to buffers the GPU reads
- **Memory budget = system RAM**: 64 GB M4 Max can use all 64 GB for model + KV + activations

---

## 6. Quantization Format Considerations

### MLX Native vs GGUF K-Quants

| Feature | MLX Affine 4-bit | GGUF Q4_K |
|---------|-----------------|-----------|
| Quality | ~Q4_0 equivalent | Better (mixed precision per layer) |
| Group size | 32, 64, 128 | 256 (super-blocks) |
| Format | safetensors | GGUF binary |
| Metal kernels | Template-generated (QMV/QMM) | Hand-written in ggml-metal.metal |

### Strategy for ZINC

ZINC already parses GGUF and has Q4_K/Q5_K/Q6_K/Q8_0 dequant in both shaders and CPU. **Keep GGUF as the model format** — same file works on Vulkan and Metal, and K-quants provide better quality than MLX's affine quants.

Write custom MSL dequant kernels that match the existing GLSL dequant logic. This is the same approach llama.cpp takes.

---

## 7. Parallel Requests / Continuous Batching

### State of the Art on Apple Silicon

| Engine | Approach | Scaling at 5 req |
|--------|----------|-----------------|
| vllm-mlx | Paged KV + continuous batching | 2.4x aggregate |
| oMLX | Paged prefix sharing + CoW KV | ~2x |
| llama.cpp | Sequential (no batching) | 1x |
| LM Studio 0.4.2+ | MLX continuous batching | ~2x |

### Why ZINC Can Win

1. **Zig + Metal direct** = no Python overhead, no framework abstraction tax
2. **Static compute graph**: Pre-record command buffers per batch config, replay without per-token CPU work
3. **Zero-copy KV cache management**: UMA means KV page tables are just pointer arithmetic
4. **MoE expert dispatch**: Can batch expert calls across requests (if same expert activated for multiple sequences)
5. **Continuous batching with pre-recorded graphs**: Different from MLX's dynamic approach — potentially lower dispatch latency

### Architecture Sketch

```
Request queue → Scheduler (assigns KV pages)
                    ↓
              Batch assembly (group sequences at same decode position)
                    ↓
              Pre-recorded Metal command buffer (per batch-size config)
                    ↓
              GPU dispatch (all sequences in one submit)
                    ↓
              Logits readback → per-sequence sampling → token emission
```

---

## 8. Performance Targets

### Baseline to Beat

| Metric | llama.cpp Metal | MLX/vllm-mlx | ZINC Target |
|--------|----------------|--------------|-------------|
| Single decode (M4 Max 64GB) | ~35 tok/s | ~65 tok/s | **80+ tok/s** |
| Prefill (512 tokens) | ~300 tok/s | ~250 tok/s | **350+ tok/s** |
| 5-request aggregate | ~35 tok/s | ~160 tok/s | **200+ tok/s** |
| Bandwidth utilization | ~53% | ~65% | **75%+** |
| Model load time | ~5s | ~3s | **<1s** (mmap, no copy) |

### Why These Targets Are Achievable

- MetalRT already hits 658 tok/s on small models (72% bandwidth utilization) with the "straight to Metal" approach — ZINC takes the same approach but in Zig
- Zero-copy mmap model loading is nearly free
- Static graph pre-recording eliminates per-token dispatch overhead that MLX pays
- Custom MSL kernels tuned for Apple GPU microarchitecture (32-thread simdgroups, 8 KB L1, simdgroup_matrix)

---

## 9. Model Availability (Qwen3.5-35B-A3B)

### GGUF (reuse existing file)
- `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` — **20.7 GB**, fits in 64 GB with ~40 GB headroom
- Same file works on Vulkan (RDNA4) and Metal (Apple Silicon)

### MLX (for baseline comparison)
- `mlx-community/Qwen3.5-35B-A3B-4bit` — 19.0 GB, 51K downloads (most popular)
- `mlx-community/Qwen3.5-35B-A3B-8bit` — 35.2 GB, fits in 64 GB

### Baseline Test Commands

```bash
# Install mlx-lm
pip install -U mlx-lm

# MLX baseline (4-bit)
mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "The capital of France is" \
  --max-tokens 256 --verbose

# llama.cpp baseline (same GGUF as Vulkan)
# Build (Metal enabled by default on macOS)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build --config Release -j16

# Run with the GGUF you already have
./build/bin/llama-bench \
  -m /path/to/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -p 512 -n 128

# Interactive
./build/bin/llama-cli \
  -m /path/to/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 4096 --conversation
```

---

## Sources

- [vllm-mlx paper (arXiv 2601.19139)](https://arxiv.org/html/2601.19139v1) — M4 Max benchmarks
- [MetalRT benchmarks](https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon)
- [llama.cpp Apple Silicon benchmarks (Discussion #4167)](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [MLX + M5 Neural Accelerators (Apple ML Research)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Comparative study (arXiv 2511.05502)](https://arxiv.org/abs/2511.05502)
- [mitchellh/zig-objc](https://github.com/mitchellh/zig-objc)
- [Lulzx/zeno](https://github.com/Lulzx/zeno) — Zig + Metal compute reference
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) — SPIR-V to MSL
- [Apple GPU Microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [llama.cpp ggml-metal.m](https://github.com/ggml-org/llama.cpp) — ObjC shim pattern
- [Qwen3.5-35B-A3B GGUF (unsloth)](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [Qwen3.5-35B-A3B MLX (mlx-community)](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- [Exo distributed inference](https://github.com/exo-explore/exo)
- [Multi-node expert parallelism (arXiv 2506.23635)](https://arxiv.org/html/2506.23635v1)
