# Zinc codebase map

## Project Structure

```
src/
├── main.zig                     # CLI entry, arg parsing, server startup, chat subcommand
├── bench_hot_decode.zig         # Hot decode microbenchmark binary
├── bench_support.zig            # Shared helpers for Metal benchmarks
├── regression_tests.zig         # Source-level regression guards
├── zig-struct-analyzer.zig      # Zig API/doc structure extraction helper
├── dump_struct_layouts.zig      # Public loader/forward memory layout tool
├── compute/
│   ├── forward.zig              # Vulkan inference engine — prefill + decode loop
│   ├── forward_metal.zig        # Metal inference engine — prefill + decode loop
│   ├── forward_cuda*.zig        # CUDA inference engines
│   ├── forward_zinc_rt.zig      # ZINC_RT inference engine
│   ├── dmmv.zig                 # DMMV dispatch (quantized matmul-vec)
│   ├── elementwise.zig          # Fused elementwise ops (RMS norm, SwiGLU, etc.)
│   ├── attention.zig            # Flash attention dispatch
│   ├── argmax.zig               # Argmax / sampling dispatch
│   └── graph.zig                # Decode graph builder and exporter
├── model/
│   ├── tokenizer.zig            # BPE tokenizer, chat templates, thinking toggle
│   ├── catalog.zig              # Managed model catalog with thinking_stable flag
│   ├── gguf.zig                 # GGUF file parser and tensor metadata
│   ├── loader.zig               # Model loader (Vulkan — mmap + DMA to VRAM)
│   ├── loader_metal.zig         # Model loader (Metal — zero-copy mmap)
│   ├── loader_cuda.zig          # CUDA model loader
│   ├── architecture.zig         # Architecture detection (Qwen, MoE, SSM, etc.)
│   ├── config.zig               # Model configuration from GGUF metadata
│   └── managed.zig              # Managed model download, install, activation
├── server/
│   ├── routes.zig               # OpenAI-compatible API, streaming, stop detection
│   ├── chat.html                # Built-in chat UI (embedded at compile time)
│   ├── http.zig                 # HTTP server and connection handling
│   ├── model_manager.zig        # Hot model switching and catalog view
│   ├── model_discovery.zig      # Minimal CUDA/ROCm loaded-model list serializer
│   ├── runtime.zig              # Backend runtime dispatch
│   └── session.zig              # Chat session state
├── vulkan/
│   ├── instance.zig             # Vulkan instance and device init
│   ├── pipeline.zig             # Compute pipeline and shader loading
│   ├── buffer.zig               # GPU buffer allocation and transfers
│   ├── command.zig              # Command buffer recording and submission
│   ├── gpu_detect.zig           # GPU vendor/capability detection
│   └── vk.zig                   # Vulkan C API bindings
├── metal/
│   ├── c.zig                    # Shared C shim import for Metal bindings
│   ├── device.zig               # Metal device init and capability query
│   ├── pipeline.zig             # MSL compute pipeline compilation
│   ├── buffer.zig               # Metal buffer management
│   ├── command.zig              # Command buffer and encoder
│   ├── shim.h                   # C header exposed to Zig
│   └── shim.m                   # Objective-C shim (Metal.framework bridge)
├── cuda/                        # CUDA device, buffers, pipeline, and kernels
├── zinc_rt/                     # Native GPU runtime, IR, rings, and ISA
├── gpu/
│   ├── interface.zig            # Backend abstraction (Vulkan vs Metal)
│   ├── memory_plan.zig          # Cross-backend memory planning
│   └── process_lock.zig         # Cross-process GPU reservation lock
├── scheduler/
│   ├── request.zig              # Request/response scheduler types
│   ├── scheduler.zig            # Request scheduling
│   └── kv_cache.zig             # KV cache management
├── diagnostics.zig              # --check system diagnostics (Vulkan)
├── diagnostics_metal.zig        # --check system diagnostics (Metal)
├── shaders/
│   ├── *.comp                   # GLSL compute shaders (Vulkan/SPIR-V)
│   └── metal/*.metal            # MSL compute shaders (Apple Silicon)

benchmarks/
├── bandwidth.zig                # DMMV bandwidth utilization benchmark
├── dispatch.zig                 # Vulkan dispatch overhead benchmark
├── dispatch_overhead.c          # Vulkan dispatch overhead C helper
├── metal_inference.zig          # Metal inference benchmark
└── metal_q8_shapes.zig          # Exact-shape Metal q8 benchmark suite

loops/                           # Self-improving optimization loops
├── guided_change.ts             # Guided single-change optimization loop
├── guided_change.test.ts        # Tests for guided_change
├── implement_metal.ts           # Metal implementation loop
├── implement_metal.test.ts      # Tests for implement_metal
├── optimize_llm_tps.ts          # LLM-throughput optimization loop
├── optimize_llm_tps.test.ts     # Tests for optimize_llm_tps
├── optimize_perf.ts             # Performance optimization loop
├── optimize_perf.test.ts        # Tests for optimize_perf
├── optimize_zinc.ts             # ZINC loop: rsync → build → run → agent → keep/revert
├── optimize_zinc.test.ts        # Tests for optimize_zinc
└── zinc_rt_autopilot.ts         # Overnight A/B vs legacy_vulkan; drives docs/ZINC_RT_DESIGN.md

docs/                            # Technical documentation (published to site)
├── API.md                       # OpenAI-compatible API spec
├── APPLE_METAL_REFERENCE.md     # Metal/MSL kernel reference
├── APPLE_SILICON_METAL_ENABLEMENT.md # Metal port implementation notes
├── APPLE_SILICON_REFERENCE.md   # Apple Silicon M1–M5 reference
├── DEVELOPMENT.md               # Development guide (canonical dev reference)
├── GETTING_STARTED.md           # First run guide
├── AMD_GPU_REFERENCE.md         # RDNA3/RDNA4 hardware reference
├── HARDWARE_REQUIREMENTS.md     # GPU and host sizing guidance
├── METAL_PERFORMANCE_PLAN.md    # Metal performance work plan
├── RDNA4_TUNING.md              # RDNA4-specific optimizations
├── ROADMAP.md                   # Project roadmap
├── RUNNING_ZINC.md              # CLI usage and server mode
├── SPEC.md                      # Architecture overview
├── TURBOQUANT_SPEC.md           # TurboQuant KV cache compression spec
└── ZINC_RT_DESIGN.md            # ZINC's own GPU runtime — replaces Vulkan; design + milestone plan

site/                            # Astro website + docs frontend (zolotukhin.ai)
├── src/components/              # Shared Astro UI components
├── src/content/posts/           # Blog post markdown sources
├── src/layouts/                 # Page layouts
├── src/lib/                     # Search/docs/data loaders and tests
├── src/pages/                   # Site routes, docs, and API-ish endpoints
├── src/styles/                  # Global site styling
└── public/                      # Static assets served by Astro

specs/                           # Historical feature design artifacts
├── 001-zinc-inference-engine/
├── 002-microblog-zolotukhin-ai/
├── 003-decode-performance/
├── 004-openai-api-server/
└── 005-apple-silicon-inference/

tools/                           # API benchmark, HTML tooling, graph/report helpers
├── benchmark_api.mjs
├── chat.html
├── inspect_qwen_meta.zig
├── print_test_summary.ts
└── render_graph_report.ts

tests/                           # TypeScript integration/smoke tests
├── chat_ui_markdown.test.ts
├── test_openai_sdk.ts
├── test_openai_sdk.test.ts
├── test_qwen_smoke.ts
└── test_qwen_smoke.test.ts

assets/                          # Shared repo/media assets
research/                        # Analysis notes and external comparisons
scripts/                         # Deployment scripts
writing/                         # Draft writing and publishing notes
```

## Code Navigation

The source changes frequently, so this section intentionally avoids line numbers,
buffer inventories, kernel counts, and tuning thresholds. Search for the named
types or functions before editing.

### Entrypoints and API

| Area                                        | Start here                                                |
| ------------------------------------------- | --------------------------------------------------------- |
| CLI, argument parsing, server startup       | `src/main.zig`                                            |
| HTTP transport and OpenAI-compatible routes | `src/server/http.zig`, `src/server/routes.zig`            |
| ROCm/CUDA HTTP serving                      | `src/main.zig`: `runCudaServe`, `handleServeConn`         |
| Backend selection and model switching       | `src/server/runtime.zig`, `src/server/model_manager*.zig` |
| Built-in chat UI                            | `src/server/chat.html`                                    |
| Tokenization and chat templates             | `src/model/tokenizer.zig`                                 |

### Models and inference

| Area                         | Start here                                                           |
| ---------------------------- | -------------------------------------------------------------------- |
| GGUF metadata and tensors    | `src/model/gguf.zig`, `src/model/config.zig`                         |
| Architecture detection       | `src/model/architecture.zig`                                         |
| Managed model catalog/cache  | `src/model/catalog.zig`, `src/model/managed.zig`                     |
| Vulkan model loading         | `src/model/loader.zig`                                               |
| Metal and CUDA model loading | `src/model/loader_metal.zig`, `src/model/loader_cuda.zig`            |
| Vulkan inference             | `src/compute/forward.zig`                                            |
| Metal inference              | `src/compute/forward_metal.zig`                                      |
| CUDA inference               | `src/compute/forward_cuda.zig`, `src/compute/forward_cuda_gemma.zig` |
| ZINC_RT inference            | `src/compute/forward_zinc_rt.zig`, `src/zinc_rt/`                    |

The conceptual request path is: tokenize → load/activate model → prefill prompt →
decode tokens → sample → detokenize/stream. Architecture-specific attention, SSM,
dense FFN, and MoE branches live in the active backend's forward implementation.

### GPU operations

| Area                                    | Start here                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------ |
| Compute graph                           | `src/compute/graph.zig`                                                              |
| Quantized matrix-vector dispatch        | `src/compute/dmmv.zig`                                                               |
| Attention, elementwise ops, sampling    | `src/compute/attention.zig`, `src/compute/elementwise.zig`, `src/compute/argmax.zig` |
| Vulkan infrastructure                   | `src/vulkan/`, `src/shaders/*.comp`                                                  |
| Metal infrastructure                    | `src/metal/`, `src/shaders/metal/*.metal`                                            |
| CUDA infrastructure                     | `src/cuda/`                                                                          |
| ROCm C ABI and execution errors         | `src/rocm/rocm_shim.c`, `src/cuda/command.zig`                                       |
| Qwen chunked prefill and prompt reuse   | `src/compute/forward_cuda.zig`                                                       |
| ROCm Qwen KV storage accounting         | `src/compute/kv_storage.zig`                                                         |
| CUDA/ROCm scheduler and slot lifecycle  | `src/server/cuda_serve.zig`                                                          |
| Backend abstraction and memory planning | `src/gpu/`                                                                           |

### Tests and performance tooling

| Area                                                              | Start here                                                              |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Zig regression tests                                              | `src/regression_tests.zig`                                              |
| Integration and API tests                                         | `tests/`                                                                |
| Microbenchmarks                                                   | `benchmarks/`, `src/bench_hot_decode.zig`                               |
| Fair cross-backend suite                                          | `tools/performance_suite.mjs`                                           |
| ROCm attention CPU/GPU reference checks                           | `tools/validate_rocm_attention.c`                                       |
| ROCm packed matrix CPU/GPU and verifier row checks                | `tools/validate_rocm_q8_matvec.c`                                       |
| Exact-prefix cache parity                                         | `tools/validate_rocm_prompt_cache.mjs`                                  |
| OpenCode setup, bounded tool hook, verified rounds                | `tools/setup_opencode.mjs`                                              |
| CPU OpenCode protocol and lifecycle                               | `tools/validate_opencode_harness.mjs`                                   |
| OpenCode preview CLI regressions                                  | `tools/setup_opencode_preview.test.ts`                                  |
| Optimization harnesses                                            | `loops/`                                                                |
| Binary release policy, packaging, GPU certification and promotion | `config/release/targets.json`, `tools/release*.py`, `docs/RELEASING.md` |
| Verified installation and rollback                                | `scripts/install.sh`, `tools/release_test.py`                           |

Treat implementation source and command `--help` output as authoritative for
current environment flags, thresholds, supported formats, and benchmark cases.
Update this map only when ownership or major entrypoints move.
