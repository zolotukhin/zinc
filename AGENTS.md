# AGENTS.md — ZINC Inference Engine

Instructions for AI coding agents working on this repository.

## Commands

```bash
# Build (shaders compile on Linux only; macOS skips GPU inference)
zig build

# Run inference
./zig-out/bin/zinc -m model.gguf --prompt "Hello" [-d device_id] [--kv-quant 3]

# Run unit tests
zig build test

# Compile shaders manually (requires glslc / shaderc)
glslc --target-env=vulkan1.3 -O -o out.spv src/shaders/name.comp
```

## Tech Stack

- **Zig 0.15.2+** — host code, build system
- **GLSL 460** — compute shaders compiled to SPIR-V via `glslc`
- **Vulkan 1.3** — GPU API (direct C ABI calls via `@cImport`)
- **GGUF** — model format (parsed natively in Zig)

## Project Structure

```
src/
├── main.zig                    # CLI entry point, arg parsing, main loop
├── vulkan/                     # GPU abstraction layer
│   ├── vk.zig                  #   @cImport of vulkan/vulkan.h
│   ├── instance.zig            #   VkInstance, VkDevice, queue init
│   ├── buffer.zig              #   GPU buffer alloc, staging, DMA
│   ├── pipeline.zig            #   Compute pipeline from SPIR-V
│   ├── command.zig             #   Command pool/buffer, dispatch recording
│   └── gpu_detect.zig          #   GPU vendor detect + auto-tuning (RDNA3/4, NVIDIA, Intel)
├── model/                      # Model loading
│   ├── gguf.zig                #   GGUF format parser (header, metadata, tensors)
│   ├── loader.zig              #   Load GGUF → GPU buffers (mmap + DMA)
│   ├── architecture.zig        #   Compute graph builders per arch (Llama, Qwen, Mamba)
│   └── tokenizer.zig           #   Text ↔ tokens (shells to Python, native Zig planned)
├── compute/                    # Inference dispatch
│   ├── graph.zig               #   Static compute graph IR (nodes, deps, topo sort)
│   ├── dmmv.zig                #   Decode matmul-vec dispatch (Q4_K, Q8_0, F16)
│   ├── elementwise.zig         #   RMS norm, SwiGLU, RoPE dispatch
│   ├── attention.zig           #   Flash attention dispatch (paged, GQA)
│   └── forward.zig             #   Inference engine: decode loop, token generation
└── shaders/                    # GLSL compute kernels → compiled to .spv
    ├── dmmv_q4k.comp           #   Q4_K dequant + matvec
    ├── dmmv_q8_0.comp          #   Q8_0 dequant + matvec
    ├── dmmv_f16.comp           #   FP16 matvec (no dequant)
    ├── coop_matmul.comp        #   Cooperative matrix 16x16x16 for prefill
    ├── flash_attn.comp         #   Paged flash attention with GQA
    ├── rms_norm_mul.comp       #   RMS norm + scale multiply (fused)
    ├── swiglu.comp             #   SiLU-gated linear unit (fused)
    ├── rope_fused.comp         #   Rotary position embedding (fused)
    ├── sigmoid_mul.comp        #   Sigmoid * element-wise (SSM gating)
    ├── softmax_topk.comp       #   Softmax + top-k (MoE routing)
    ├── tq_quantize_keys.comp   #   TurboQuant: quantize K cache
    ├── tq_quantize_values.comp #   TurboQuant: quantize V cache
    ├── tq_attention_scores.comp#   TurboQuant: attention on quantized KV
    └── tq_decompress_values.comp#  TurboQuant: decompress V values

benchmarks/
├── bandwidth.zig               # DMMV bandwidth utilization benchmark
└── dispatch.zig                # Vulkan dispatch overhead benchmark

loops/                          # Self-improving optimization loops
├── optimize_zinc.ts            #   ZINC loop: rsync → build → run → agent → keep/revert
├── optimize_zinc.test.ts       #   Tests for ZINC loop (bun test)
├── optimize_llm_tps.ts         #   llama.cpp TPS optimization loop
└── optimize_llm_tps.test.ts    #   Tests for llama.cpp loop (bun test)

docs/                           # Technical specifications
├── SPEC.md                     #   Architecture overview
├── API.md                      #   OpenAI-compatible API spec
├── RDNA4_TUNING.md             #   RDNA4-specific optimizations
└── TURBOQUANT_SPEC.md          #   TurboQuant KV cache compression spec

site/                           # Astro website (zolotukhin.ai)
specs/                          # Feature specs and planning artifacts
```

## Module Dependency Graph

```
main.zig
├── vulkan/instance.zig
├── vulkan/gpu_detect.zig
├── model/loader.zig
│   ├── model/gguf.zig
│   ├── vulkan/buffer.zig
│   └── vulkan/command.zig
├── model/tokenizer.zig
└── compute/forward.zig
    ├── compute/graph.zig
    ├── compute/dmmv.zig        → vulkan/pipeline.zig → shaders/dmmv_*.spv
    ├── compute/elementwise.zig → vulkan/pipeline.zig → shaders/{rms_norm,swiglu,rope}.spv
    ├── compute/attention.zig   → vulkan/pipeline.zig → shaders/flash_attn.spv
    └── model/architecture.zig  → compute/graph.zig
```

## Key Architecture Decisions

- **Static graph pre-recording**: decode graph built once per model arch, command buffer recorded once, replayed per token with updated push constants/descriptors
- **Quantization isolated in shaders**: graph nodes use `dmmv` ops; dispatcher selects Q4_K/Q8_0/F16 pipeline at runtime
- **GPU auto-tuning**: `gpu_detect.zig` classifies hardware and derives wave size, tile sizes, cache parameters — no manual config needed
- **Paged KV cache**: 16-token pages (vLLM-style) via page table in flash attention shader
- **Fused kernels**: RMS_NORM_MUL, SWIGLU, ROPE_FUSED eliminate intermediate memory traffic

## Code Style

- **Zig**: follow standard Zig conventions, `zig fmt` for formatting
- **GLSL**: `#version 460`, `layout(local_size_x = 64)` default (wave64 for RDNA), push constants for per-dispatch params, storage buffers for data
- Keep shader workgroup size at 64 unless there's a measured reason to change it

## Boundaries

### Always
- Run `zig build test` before considering work complete
- Validate GPU kernels against llama.cpp reference outputs
- Use push constants (not UBOs) for per-dispatch parameters in shaders
- Keep shader local_size_x = 64 (RDNA4 wave64)

### Ask first
- Changing the compute graph IR (`graph.zig` OpType enum)
- Adding new model architectures to `architecture.zig`
- Modifying Vulkan initialization or device selection
- Changes to GGUF parsing that could break existing model loading

### Never
- Commit `.env`, credentials, or private IPs/ports
- Modify `.spv` binaries directly — always recompile from `.comp` source
- Add runtime dependencies beyond Vulkan and system libc
- Use wave32 without benchmarking against wave64 first

## Remote Test Node

An RDNA4 test node (AMD Radeon AI PRO R9700, 32GB, 576 GB/s) is available via SSH. Credentials are in `.env` (gitignored) as `ZINC_HOST`, `ZINC_USER`, `ZINC_PORT`.
