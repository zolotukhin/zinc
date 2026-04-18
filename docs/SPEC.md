# ZINC Technical Specification

Last updated: 2026-04-17

ZINC is a local-first LLM inference engine written primarily in Zig. It reads GGUF directly, runs single-model inference through a CLI and OpenAI-compatible HTTP API, and targets two GPU backends:

- **Vulkan** on Linux, primarily for AMD RDNA3/RDNA4
- **Metal** on macOS for Apple Silicon

This page is a living architecture document for the **current implementation**. Where the repository contains prototype code or forward-looking work, that is called out explicitly instead of being presented as already shipped behavior.

## 1. Current State At A Glance

| Area | Current state |
|------|---------------|
| Backend selection | Compile-time: Linux builds the Vulkan path, macOS builds the Metal path |
| Model format | Native GGUF parsing and loading |
| Active model policy | One model loaded into runtime memory at a time |
| Supported serving | CLI, built-in chat UI, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/models/pull`, `/v1/models/activate`, `/v1/models/remove`, `/health` |
| Concurrency model | HTTP is concurrent, but generation is still serialized behind one engine lock |
| Scheduler status | `src/scheduler/*` contains groundwork for continuous batching and page allocation, but it is not the main serving hot path today |
| KV cache | Vulkan uses a paged 16-token layout; Metal currently favors a contiguous fast path with the same conceptual interface |
| Context planning | Runtime memory is budgeted through a shared planner; today's engine is still effectively tuned around a `4096` token context cap |
| KV compression | Metal has a `q8_0` KV option in the runtime; TurboQuant is specified elsewhere but is not integrated into the main runtime yet |
| Graph tooling | Decode graphs can be exported as JSON and DOT with per-op cost annotations |

## 2. Design Goals

ZINC is optimized around a small number of strong constraints:

- keep the runtime mostly Zig
- own GGUF parsing and model configuration directly
- keep the decode path explicit enough to profile and tune at kernel level
- keep the higher-level UX stable across backends
- make performance work observable through graph export, diagnostics, and microbenchmarks

That leads to a split architecture:

- shared upper layers for tokenization, GGUF parsing, model catalog, HTTP routes, diagnostics, and memory planning
- backend-specific substrate for device discovery, buffers, pipeline compilation, command submission, and kernel code
- model-family-specific decode planning driven from normalized GGUF metadata instead of hardcoded per-model scripts

## 3. System Layout

```text
CLI / HTTP server / chat UI
  -> tokenizer + chat templates
  -> managed model catalog + model manager
  -> GGUF parser + model config
  -> decode graph builder + graph export
  -> backend runtime
       -> Vulkan: SPIR-V compute shaders, device-local VRAM, command buffers
       -> Metal:  MSL kernels, unified memory, Objective-C shim
  -> sampling + streaming + health / memory reporting
```

### 3.1 Shared modules

The shared layers are centered in these files:

- `src/model/gguf.zig` parses metadata and tensor layouts from GGUF
- `src/model/config.zig` normalizes architecture-specific metadata into a runtime config
- `src/model/tokenizer.zig` owns vocab, merges, chat templates, and thinking-toggle handling
- `src/model/catalog.zig` defines the curated managed-model catalog
- `src/model/managed.zig` handles downloads, cache layout, active selection, and fit checks
- `src/server/routes.zig` serves the HTTP API and built-in chat UI
- `src/gpu/memory_plan.zig` computes fixed and per-token runtime memory costs for both backends

### 3.2 Backend-specific substrate

The backend switch happens in `src/gpu/interface.zig` at compile time:

- Linux => Vulkan backend
- macOS => Metal backend

This keeps the inactive backend out of the build and lets the upper layers call one runtime interface without paying for a large runtime abstraction.

## 4. Backend Architecture

### 4.1 Vulkan Path

The Vulkan runtime lives primarily in:

- `src/vulkan/instance.zig`
- `src/vulkan/buffer.zig`
- `src/vulkan/pipeline.zig`
- `src/vulkan/command.zig`
- `src/compute/forward.zig`
- `src/shaders/*.comp`

Key properties of the Vulkan path:

- weights are uploaded into **device-local VRAM**
- compute kernels are handwritten **GLSL 460** shaders compiled to **SPIR-V**
- decode work is recorded against explicit command buffers
- per-dispatch variability is carried through push constants and descriptor bindings
- AMD-specific tuning is built around wave64 and bandwidth-sensitive decode kernels

The Vulkan engine is where the most explicit static-graph and paged-KV machinery exists today.

### 4.2 Metal Path

The Metal runtime lives primarily in:

- `src/metal/device.zig`
- `src/metal/buffer.zig`
- `src/metal/pipeline.zig`
- `src/metal/command.zig`
- `src/metal/shim.m`
- `src/model/loader_metal.zig`
- `src/compute/forward_metal.zig`
- `src/shaders/metal/*.metal`

Key properties of the Metal path:

- model files are wrapped into `MTLBuffer`s through **zero-copy `mmap`**
- the Objective-C boundary is isolated to **one shim file**
- MSL sources are compiled into compute pipelines at startup
- unified memory changes the buffer strategy and removes the Vulkan-style upload path
- Metal keeps the same high-level inference model while using backend-specific kernels and buffer policy

On Apple Silicon, logits are CPU-visible through UMA, which simplifies sampling and debug readback compared with Vulkan.

### 4.3 Shared Runtime Behavior

Despite the backend split, both paths share the same user-facing model:

- same CLI entrypoint
- same HTTP routes
- same managed-model catalog and active-model selection
- same tokenizer and chat template logic
- same high-level concepts: prompt prefill, token-by-token decode, KV cache tracking, sampling, health reporting

## 5. Model Loading And Managed Models

### 5.1 GGUF Loading

ZINC reads GGUF directly instead of depending on an external runtime. The loader stack is responsible for:

- parsing tensor metadata and offsets
- inferring model architecture and normalized dimensions
- resolving tokenizer assets from GGUF metadata
- mapping tensor names to runtime operators

At a high level:

```text
GGUF file
  -> parse header + metadata + tensor table
  -> derive ModelConfig
  -> expose TensorInfo for runtime lookup
  -> construct backend-specific weight storage
```

### 5.2 Vulkan Loader

The Vulkan loader in `src/model/loader.zig` memory-maps the GGUF file and uploads tensors into device-local buffers. The runtime then reads weights from VRAM during decode.

This path is optimized for discrete GPUs where explicit upload cost is worth paying in exchange for device-local execution.

### 5.3 Metal Loader

The Metal loader in `src/model/loader_metal.zig` maps the GGUF file and wraps those regions directly with `newBufferWithBytesNoCopy`. That fits Apple Silicon's unified-memory model better than mirroring the Vulkan upload flow.

### 5.4 Managed Model Catalog

ZINC now ships a curated managed-model catalog in `src/model/catalog.zig`. Each entry includes:

- stable short id
- display name
- release date
- download URL and optional sha256 pin
- estimated VRAM requirement
- tested GPU profiles
- whether the model's thinking toggle is stable enough to expose in the UI

The managed-model flow is backed by `src/model/managed.zig` and is used by both CLI and server model management. It provides:

- `zinc model list`
- `zinc model pull <id>`
- `zinc model use <id>`
- `zinc model active`
- `zinc model rm <id>`

The HTTP server exposes the same lifecycle through `/v1/models`, `/v1/models/pull`, `/v1/models/activate`, and `/v1/models/remove`.

### 5.5 One-Model-At-A-Time Policy

The runtime model manager (`src/server/model_manager.zig` and `src/server/model_manager_metal.zig`) keeps **one active model bundle** loaded at a time:

- model
- tokenizer
- inference engine
- memory-usage accounting

Hot-swapping models is supported, but swaps are serialized with generation because the engine is not yet multi-tenant in the serving hot path.

## 6. Supported Model Families

The current runtime is designed around the model families ZINC is actively validating:

- **Qwen3 / Qwen3.5 / Qwen3.6**
- **Gemma 4**
- **OpenAI GPT-OSS**

At the execution-model level, that means ZINC handles:

- dense transformer layers
- MoE feed-forward blocks
- SSM-hybrid paths used by Qwen3.5-style models and related experimental families
- model-specific routing rules such as GPT-OSS selected-only softmax weighting

The architecture-normalization layer is in `src/model/architecture.zig` and `src/model/config.zig`.

## 7. Decode Planning And Graph IR

ZINC does not treat decode as an opaque loop. It has an explicit graph/planning layer:

- `src/compute/graph.zig`
- `src/model/architecture.zig`

The graph builder emits a logical decode graph with per-node metadata such as:

- operation type
- layer index
- execution domain
- workgroup counts
- estimated read/write/weight bytes
- approximate FLOPs
- host synchronization requirements

This graph is used for:

- understanding what a model family will execute
- exporting JSON and DOT reports
- bottleneck analysis and hotspot ranking
- validating whether a change actually alters the planned decode structure

The graph is **not** just a visualization artifact. It is the design-level representation of the decode pipeline.

### 7.1 Graph Families

The graph builder currently emits different logical plans for:

- standard transformer decode
- MoE decode
- SSM-hybrid decode
- Gemma-specific dense and MoE variants

### 7.2 Exported Artifacts

The CLI can export:

- JSON graph reports
- Graphviz DOT

The JSON report is rich enough for downstream tooling such as `tools/render_graph_report.ts`, which groups nodes into hotspots, op mix, bottleneck mix, and critical-path summaries.

## 8. Inference Runtime

The runtime is centered around `InferenceEngine`:

- `src/compute/forward.zig` for Vulkan
- `src/compute/forward_metal.zig` for Metal

The engine owns:

- decode buffers and scratch space
- KV cache storage
- per-layer pipelines and dispatch helpers
- model-family-specific fast paths
- sampling and logits readback support

### 8.1 Decode State

Per-request decode state tracks:

- current token position
- generated tokens
- stop/completion progress
- backend-specific runtime state needed to advance the sequence

### 8.2 High-Level Token Flow

For a single generated token, the logical flow is:

```text
token id
  -> embedding lookup / dequant
  -> per-layer input normalization
  -> attention or SSM body
  -> FFN or MoE body
  -> residual accumulation
  -> final normalization
  -> LM head projection
  -> logits readback or greedy argmax
  -> next token
```

### 8.3 Attention Layers

Attention layers perform the usual decode-time sequence:

- project Q, K, and V
- apply RoPE to the rotated dimensions
- write the new K/V vectors into cache storage
- run flash attention over the cached sequence
- apply output projection
- add residual

Grouped-query attention is supported in the flash-attention path.

### 8.4 SSM / Hybrid Layers

The hybrid runtime includes SSM-oriented operators such as:

- `ssm_conv1d`
- `ssm_delta_net`
- `ssm_gated_norm`

These are used in the Qwen3.5-style hybrid path and related architecture variants parsed by the loader.

### 8.5 FFN And MoE Execution

For MoE layers, the runtime supports a split between routing and expert execution:

- gate projection produces router logits
- the runtime selects top experts and normalized weights
- gate/up projections run for selected experts
- SwiGLU or model-specific activation is applied
- down projections run
- outputs are accumulated back into the hidden state

Important detail: ZINC currently supports **both** a fast GPU-routed path and CPU-assisted fallbacks depending on backend and tensor format availability.

Examples:

- Vulkan can keep more of the MoE route fully on GPU when quant-specific kernels plus `softmax_topk` and weighted accumulation are available
- otherwise it falls back to router readback plus CPU `topKSoftmax`
- Metal has a batched MoE path and quant-specific expert kernels, but still falls back where coverage is incomplete or validation requires it

### 8.6 Sampling

The runtime supports:

- greedy decoding
- temperature
- top-p
- top-k
- repetition penalty

On Vulkan, logits readback is explicitly controlled. On Metal, UMA makes logits CPU-visible by default.

## 9. GPU Kernel Library

### 9.1 DMMV Is The Decode Workhorse

The most important kernel family is decode matmul-vector:

- `Q4_K`
- `Q5_K`
- `Q6_K`
- `Q8_0`
- `F16`
- `F32`

These kernels back:

- attention projections
- FFN projections
- LM head
- MoE expert projections

The kernel library also includes MoE-specialized variants and batch-oriented variants for paths where multiple expert outputs are accumulated together.

### 9.2 Fused Elementwise Operators

The elementwise library exists to avoid turning decode into a long chain of tiny memory-bound kernels. Current fused or dedicated operators include:

- RMS norm
- SwiGLU
- RoPE
- deinterleave
- sigmoid-mul
- scale-accumulate
- sigmoid-scale-accumulate
- softmax-topk
- argmax
- KV cache write
- SSM conv / delta / gated norm

On Metal, additional batched operators exist for MoE accumulation and SwiGLU. On Vulkan, the equivalent strategy is expressed through GLSL compute shaders and explicit command-buffer recording.

### 9.3 Flash Attention

Flash attention is implemented as a dedicated kernel family rather than a naïve attention loop. The runtime handles:

- single-token decode attention
- paged or contiguous KV traversal depending on backend path
- grouped-query attention
- online softmax-style accumulation

### 9.4 Execution Strategy Differences By Backend

### Vulkan

The Vulkan runtime tries to keep a token step inside as few submissions as possible. Important properties:

- explicit command-buffer recording
- push constants for per-dispatch parameters
- stable decode structure from a preplanned graph
- host sync only at unavoidable boundaries such as CPU-assisted MoE routing or sampling

### Metal

The Metal runtime uses:

- runtime MSL compilation
- unified-memory buffers
- batched MoE fast paths for supported quant formats
- backend-specific pipeline capability queries for threadgroup tuning

The abstraction is similar, but the memory model and optimization priorities differ sharply from Vulkan.

## 10. KV Cache And Memory Planning

### 10.1 Vulkan KV Cache

The Vulkan runtime uses a paged KV layout with **16-token pages**. The core pieces are:

- per-layer K/V storage
- a page table buffer used by flash attention
- a page-pool allocator

This is the most explicit realization of ZINC's paged-KV design today.

### 10.2 Metal KV Cache

Metal keeps the same conceptual contract but currently favors a **contiguous fast path**. The flash-attention interface still understands page-table inputs, but the current engine can select a contiguous traversal mode for better practicality on Apple Silicon.

Metal also includes a `q8_0` KV-cache option for model families where the quality/performance tradeoff is acceptable.

### 10.3 Shared Memory Planner

`src/gpu/memory_plan.zig` is the shared source of truth for runtime memory budgeting. It splits memory into:

- fixed runtime bytes
- bytes per context token
- device-local / private bytes
- host-visible bytes

That planner feeds:

- diagnostics
- managed-model fit estimation
- active-model load policy
- `/health`
- `/v1/models`

This matters because the numbers shown by the CLI and server are intended to come from the same accounting model, not a collection of ad hoc estimates.

### 10.4 Current Context Policy

Even though some GGUFs advertise much larger theoretical contexts, the current public runtime is still effectively planned around a `4096` token operating cap. That affects:

- memory reservation
- health reporting
- managed-model fit estimates
- practical serving behavior

## 11. Serving Layer

The HTTP server is implemented in:

- `src/server/http.zig`
- `src/server/routes.zig`
- `src/server/runtime.zig`
- `src/server/model_manager.zig`
- `src/server/model_manager_metal.zig`
- `src/server/session.zig`

### 11.1 Current Endpoints

| Method | Path | Status |
|--------|------|--------|
| `GET` | `/` and `/chat` | Built-in chat UI |
| `GET` | `/health` | Implemented |
| `GET` | `/v1/models` | Implemented |
| `POST` | `/v1/models/pull` | Implemented |
| `POST` | `/v1/models/activate` | Implemented |
| `POST` | `/v1/models/remove` | Implemented |
| `POST` | `/v1/chat/completions` | Implemented |
| `POST` | `/v1/completions` | Implemented |

ZINC does **not** currently expose `/v1/embeddings`, so older drafts of this spec that listed it were ahead of the implementation.

### 11.2 Generation Concurrency

The server accepts overlapping requests and reports queue depth, but generation is still serialized behind `ServerState.generation_mutex`.

That means:

- multiple clients can connect concurrently
- queued work is tracked and reported
- `/health` remains useful during load
- decode itself is still one-active-generation-at-a-time

This is the most important serving limitation to understand today.

### 11.3 Chat Template And Thinking Control

The chat route is not a thin transport wrapper. It also owns:

- system-prompt insertion when needed
- model-specific chat templating
- thinking-toggle handling
- normalization and sanitization of assistant output
- streaming stop detection

Managed catalog entries additionally carry a `thinking_stable` bit so the UI and API can avoid exposing a toggle that produces poor results on a given model.

### 11.4 Session Reuse Cache

The chat server now includes a small session reuse cache keyed by session id and model path. Its goal is to avoid redoing the full transcript prefill when a new request extends an already-known conversation prefix.

This is **not** continuous batching, but it does give the server a concrete form of incremental prompt reuse.

### 11.5 Health And Model Introspection

`/health` and `/v1/models` expose real runtime counters such as:

- active requests
- queued requests
- active context tokens
- memory used vs budget
- weights bytes
- runtime bytes
- reserved context bytes
- active model information
- managed-model download progress

That makes the HTTP surface useful for both user UX and performance/debug workflows.

## 12. Scheduler Status

The repository contains:

- `src/scheduler/scheduler.zig`
- `src/scheduler/kv_cache.zig`
- `src/scheduler/request.zig`

These modules describe the intended shape of:

- request-slot management
- prefilling vs decoding phases
- KV-page allocation

But today they should be understood as **groundwork**, not as the main runtime scheduling mechanism used by the production server path.

In other words:

- ZINC already has serving
- ZINC does not yet have full production continuous batching
- the scheduler code exists to support that direction

## 13. Deliberate Non-Claims And Near-Term Work

The repository now has more real implementation detail than it did when this page was first written, but some areas are still explicitly incomplete:

- no `/v1/embeddings` endpoint yet
- no production continuous batching in the serving hot path yet
- no multi-GPU execution path
- no fully integrated TurboQuant runtime yet, despite CLI parsing and graph/placeholders for it
- Metal prefill is functional but still not the same thing as a separately optimized large-batch prefill kernel family

Those are active engineering directions, not hidden features.

## 14. Related Docs

- [Running ZINC](/zinc/docs/running-zinc) for CLI, server mode, and managed-model usage
- [API Reference](/zinc/docs/api) for HTTP request/response details
- [Development Guide](/zinc/docs/development) for build, test, graph export, and profiling workflow
- [Apple Silicon Metal Enablement](/zinc/docs/apple-silicon-metal-enablement) for the full Metal port narrative
- [TurboQuant Spec](/zinc/docs/turboquant-spec) for the forward-looking KV compression design
- [RDNA4 Tuning Guide](/zinc/docs/rdna4-tuning) for AMD-specific performance work
