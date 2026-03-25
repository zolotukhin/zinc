# ZINC — Zig INferenCe Engine for AMD GPUs

> Making AMD consumer GPUs actually usable for LLM inference.

## The Problem

AMD's RDNA3/RDNA4 GPUs (RX 9070, Radeon AI PRO R9700, etc.) have excellent memory bandwidth (576+ GB/s) and hardware features (cooperative matrix, integer dot product), but:

1. **ROCm doesn't support them** — only MI-series datacenter GPUs
2. **vLLM requires ROCm** — so it can't use these GPUs at all
3. **llama.cpp Vulkan works** but treats RDNA4 as an afterthought — no RDNA4-specific tuning, SPIR-V toolchain incompatibilities, no tensor parallelism
4. **No solution handles parallel requests well** on these GPUs for production use (agentic workloads, multi-user serving, etc.)

These cards cost $500-1000 (vs $15,000+ for MI300X) and sit in millions of desktops doing nothing during inference.

## The Solution

A purpose-built LLM inference engine written in **Zig** + **Vulkan compute**, targeting AMD RDNA3/RDNA4 consumer GPUs. OpenAI-compatible API, optimized for throughput on parallel requests.

## Architecture

```
┌─────────────────────────────────────────────┐
│                ZINC Server                   │
│  OpenAI-compatible HTTP API (Zig HTTP)       │
├─────────────────────────────────────────────┤
│           Request Scheduler                  │
│  Continuous batching, KV cache management    │
├─────────────────────────────────────────────┤
│         Compute Graph Engine                 │
│  Model graph → fused op sequences            │
│  Static graph recording + replay             │
├─────────────────────────────────────────────┤
│      Vulkan Compute Backend (Zig)            │
│  Hand-tuned RDNA4 shaders (GLSL → SPIR-V)   │
│  Cooperative matrix matmul                   │
│  Fused element-wise kernels                  │
│  Pre-recorded command buffer replay          │
├─────────────────────────────────────────────┤
│         GPU Memory Manager                   │
│  GGUF model loader                           │
│  Paged KV cache (like vLLM's PagedAttention) │
│  Quantized weight storage (Q4_K, Q8_0, etc.) │
├─────────────────────────────────────────────┤
│      Vulkan API (via Zig vulkan bindings)     │
│           AMD RADV / AMDVLK driver           │
│            RDNA3 / RDNA4 Hardware             │
└─────────────────────────────────────────────┘
```

## Why Zig

- **No hidden allocations** — critical for GPU memory management
- **C ABI compatible** — direct Vulkan API calls, zero bindings overhead
- **Comptime** — generate kernel dispatch tables at compile time
- **Cross-compilation** — single binary for any Linux target
- **No undefined behavior** — safety without runtime overhead

## Why Vulkan (not HIP/ROCm)

- **Actually works on RDNA4** — ROCm doesn't
- **Cooperative matrix extension** — hardware matmul acceleration
- **Compute dispatch is nearly free** — measured 0.016µs per dispatch on RDNA4
- **Command buffer replay** — pre-record the decode graph, replay per token
- **RADV driver is excellent** — open source, actively maintained

## Performance (measured on Radeon AI PRO R9700)

Based on extensive profiling and optimization of RDNA4 Vulkan inference:

| Metric | Result |
|--------|--------|
| DMMV bandwidth utilization | 67-93% of theoretical (576 GB/s) |
| Vulkan dispatch overhead | 0.016 µs per dispatch (negligible) |
| Single-request generation | 110 tok/s (Qwen3.5-35B-A3B Q4_K) |
| 4 concurrent requests | 108 tok/s each = 432 tok/s aggregate |
| Prefill throughput | 2800+ tok/s (pp512) |

The GPU handles 4x parallelism with zero per-slot degradation — ideal for multi-request serving.

## Performance Targets

### Single RX 9070 XT (16GB, 672 GB/s)
| Model | Quant | Single-req | 4-concurrent | Aggregate |
|-------|-------|------------|--------------|-----------|
| Qwen3-8B | Q4_K | 130+ tok/s | 120+ each | 480+ tok/s |
| Llama-3.1-70B | Q4_K | 20+ tok/s | 18+ each | 72+ tok/s |

### Single Radeon AI PRO R9700 (32GB, 576 GB/s)
| Model | Quant | Single-req | 4-concurrent | Aggregate |
|-------|-------|------------|--------------|-----------|
| Qwen3-8B | Q4_K | 120+ tok/s | 110+ each | 440+ tok/s |
| Llama-3.1-70B | Q4_K | 35+ tok/s | 32+ each | 128+ tok/s |

## Key Technical Features

- **Paged KV cache** — like vLLM's PagedAttention, enables efficient concurrent requests
- **Continuous batching** — process multiple requests simultaneously
- **RDNA4-tuned shaders** — hand-optimized GLSL compute shaders, not generic
- **Fused kernels** — RMS_NORM+MUL, SwiGLU, ROPE fusions reduce memory traffic
- **Command buffer replay** — static decode graph pre-recorded, replayed per token
- **GGUF model loading** — direct memory-map to GPU VRAM
- **MoE support** — Qwen, Mixtral mixture-of-experts architectures
- **SSM/Mamba support** — hybrid attention+SSM models

## Development Phases

### Phase 0: Scaffold
- Zig project setup with Vulkan bindings
- GPU detection + capability query
- Basic compute shader dispatch
- GGUF file parser

### Phase 1: Single-Request Inference
- Q8_0 and Q4_K dequant+matmul shaders
- RMS norm, ROPE, softmax, flash attention
- LLaMA architecture forward pass
- Correctness validation against llama.cpp

### Phase 2: Server + Batching
- HTTP server with OpenAI-compatible API
- SSE streaming
- Paged KV cache + continuous batching
- Chat template support

### Phase 3: Performance
- Cooperative matrix matmul for prefill
- Fused element-wise kernels
- Command buffer pre-recording + replay
- MoE and SSM/Mamba support

### Phase 4: Production
- Speculative decoding
- Prompt caching
- Multi-GPU tensor parallelism
- Docker image + benchmarks

## RDNA4 Tuning Notes

Key findings from profiling (see `research/` for details):

- **Disable GPU ECC**: `amdgpu.ras_enable=0` — ECC silently consumes ~10% memory bandwidth
- **Enable cooperative matrix**: `RADV_PERFTEST=coop_matrix`
- **Wave64 is optimal** for DMMV (wave32 measured slower)
- **System glslc (shaderc 2023.8)** produces RADV-compatible SPIR-V; newer versions cause 5x regression
- **SMU version mismatch** on kernel 6.17 limits RDNA4 to 2200 MHz (should be 2350)

## Getting Started

### Prerequisites

**macOS:**
```bash
brew install zig           # 0.15.2+
brew install vulkan-loader vulkan-headers
brew install oven-sh/bun/bun
```

**Linux (Ubuntu/Debian):**
```bash
# Zig 0.15.2+
# See https://ziglang.org/download/ for latest install instructions

# Vulkan
sudo apt install libvulkan-dev vulkan-tools

# Bun (for TypeScript loop tests)
curl -fsSL https://bun.sh/install | bash
```

**Linux (Fedora):**
```bash
sudo dnf install vulkan-loader-devel vulkan-headers
curl -fsSL https://bun.sh/install | bash
```

### Build

```bash
zig build
```

The binary is placed in `zig-out/bin/zinc`.

### Run

```bash
zig build run
# or directly:
./zig-out/bin/zinc -m model.gguf -p 8080
```

### Test

Run all tests (Zig unit tests + TypeScript bun tests):

```bash
zig build test
```

Run only TypeScript tests:

```bash
bun test
```

## Status

Early development. Star the repo and watch for updates.

## License

MIT
