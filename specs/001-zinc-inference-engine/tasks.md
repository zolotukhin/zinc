# Tasks: ZINC LLM Inference Engine

**Input**: Design documents from `specs/001-zinc-inference-engine/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/openai-api.md, quickstart.md

**Tests**: Tests are included as validation tasks within each phase.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, Vulkan setup, build system, GPU detection

- [ ] T001 Configure build.zig to compile GLSL shaders to SPIR-V via system glslc and link Vulkan loader
- [ ] T002 [P] Implement Vulkan instance creation, physical device selection, and compute queue setup in src/vulkan/instance.zig
- [ ] T003 [P] Implement GPU buffer allocation, memory type selection, and staging buffer management in src/vulkan/buffer.zig
- [ ] T004 [P] Implement GPU capability detection and auto-tuning parameter derivation in src/vulkan/gpu_detect.zig (vendor, VRAM, bandwidth, CU count, wave size, coopmat support → derived workgroup sizes, tile dims)
- [ ] T005 Implement compute pipeline creation from SPIR-V modules with specialization constants in src/vulkan/pipeline.zig (depends on T001)
- [ ] T006 Implement command buffer recording, submission, and pre-recorded replay in src/vulkan/command.zig (depends on T002)
- [ ] T007 [P] Implement CLI argument parsing in src/main.zig (model path, port, kv-quant, prompt)

**Checkpoint**: Vulkan infrastructure ready — can create device, allocate buffers, compile and dispatch shaders, detect GPU capabilities.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: GGUF loading and compute graph infrastructure that all user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T008 Implement GGUF file parser in src/model/gguf.zig — parse header, tensor descriptors, metadata key-value pairs, support split files
- [ ] T009 Implement model loader in src/model/loader.zig — memory-map GGUF file, DMA weight tensors to GPU VRAM buffers, parse architecture metadata (depends on T003, T008)
- [ ] T010 Implement compute graph definition in src/compute/graph.zig — represent ops as nodes, support topological ordering, static graph for decode path
- [ ] T011 [P] Implement architecture-specific graph builders for LLaMA/Mistral/Qwen transformer in src/model/architecture.zig (depends on T009, T010)
- [ ] T012 [P] Implement tokenizer interface in src/model/tokenizer.zig — shell out to external sentencepiece/tiktoken process, handle BPE encoding/decoding

**Checkpoint**: Can load a GGUF model into GPU VRAM, construct the compute graph, and tokenize input. Ready for kernel implementation.

---

## Phase 3: User Story 1 — Single-Request Inference (Priority: P1) MVP

**Goal**: Load a GGUF model and generate text from a single prompt with high bandwidth utilization.

**Independent Test**: Load Qwen3-8B Q4_K, generate 256 tokens, compare logits against llama.cpp (>99.5% cosine similarity), verify 90%+ bandwidth on large matmuls.

### GPU Kernel Implementation

- [ ] T013 [P] [US1] Write DMMV Q4_K shader in src/shaders/dmmv_q4k.comp — wave64, specialization constants for M/K, 2 rows per workgroup
- [ ] T014 [P] [US1] Write DMMV Q8_0 shader in src/shaders/dmmv_q8_0.comp — optimized for attention weight matmul
- [ ] T015 [P] [US1] Write DMMV F16 shader in src/shaders/dmmv_f16.comp — for KV cache and small tensors
- [ ] T016 [P] [US1] Write fused RMS_NORM_MUL shader in src/shaders/rms_norm_mul.comp — RMS norm + scale multiply in single dispatch
- [ ] T017 [P] [US1] Write fused SwiGLU shader in src/shaders/swiglu.comp — SILU(x) * y
- [ ] T018 [P] [US1] Write fused ROPE shader in src/shaders/rope_fused.comp — RoPE + reshape + cache write
- [ ] T019 [P] [US1] Write flash attention shader in src/shaders/flash_attn.comp — paged, 256-token blocks, GQA support
- [ ] T020 [P] [US1] Write cooperative matrix matmul shader in src/shaders/coop_matmul.comp — 16x16x16 tiles for prefill

### Host-Side Dispatch

- [ ] T021 [US1] Implement DMMV dispatch logic in src/compute/dmmv.zig — select shader by quant type, set specialization constants, manage push constants (depends on T005, T013-T015)
- [ ] T022 [US1] Implement fused element-wise dispatch in src/compute/elementwise.zig — dispatch RMS_NORM_MUL, SwiGLU, ROPE_FUSED with correct buffer bindings (depends on T005, T016-T018)
- [ ] T023 [US1] Implement flash attention dispatch in src/compute/attention.zig — page table lookup, block iteration, GQA head mapping (depends on T005, T019)

### Forward Pass Integration

- [ ] T024 [US1] Implement single-request decode loop in src/main.zig — tokenize prompt, prefill, decode token-by-token, detokenize output (depends on T011, T021-T023)
- [ ] T025 [US1] Implement command buffer pre-recording for decode graph in src/vulkan/command.zig — record once, replay per token via vkQueueSubmit (depends on T006, T024)

### Validation

- [ ] T026 [US1] Create bandwidth utilization benchmark in benchmarks/bandwidth.zig — measure effective GB/s for each DMMV quant type at various matrix sizes
- [ ] T027 [US1] Create dispatch overhead benchmark in benchmarks/dispatch.zig — measure single dispatch, 1500 dispatches, and pre-recorded replay
- [ ] T028 [US1] Create logit comparison test — generate tokens with ZINC and llama.cpp on same model/prompt/seed, compute cosine similarity (target >99.5%)

**Checkpoint**: Single-request inference works end-to-end. Can generate coherent text at target speed with measured bandwidth utilization.

---

## Phase 4: User Story 2 — Multi-Request Server with OpenAI API (Priority: P2)

**Goal**: Serve concurrent clients via OpenAI-compatible HTTP API with continuous batching and paged KV cache.

**Independent Test**: Start server, send 4 concurrent streaming chat completions, verify 100+ tok/s each, 400+ aggregate, no cross-contamination.

### KV Cache and Scheduler

- [ ] T029 [US2] Implement paged KV cache manager in src/scheduler/kv_cache.zig — page allocation/deallocation pool, page table mapping (seq_id, position) → GPU page, copy-on-write, LRU eviction (depends on T003)
- [ ] T030 [US2] Implement request state machine in src/scheduler/request.zig — Request struct with state transitions (pending → prefilling → decoding → completed/cancelled), generation params, SSE handle
- [ ] T031 [US2] Implement continuous batching scheduler in src/scheduler/scheduler.zig — collect pending requests, sort by priority, form batch up to max_batch_size, dispatch prefill/decode, check stopping conditions, manage KV pages (depends on T029, T030)

### HTTP Server

- [ ] T032 [P] [US2] Implement HTTP server in src/server/http.zig — Zig std.http listener, zero-copy request parsing, connection pooling
- [ ] T033 [P] [US2] Implement SSE streaming in src/server/sse.zig — chunked transfer encoding, `data: {...}\n\n` format, `[DONE]` terminator
- [ ] T034 [US2] Implement API route handlers in src/server/routes.zig — POST /v1/chat/completions, POST /v1/completions, POST /v1/embeddings, GET /v1/models, GET /health per contracts/openai-api.md (depends on T032, T033)

### Integration

- [ ] T035 [US2] Integrate scheduler with HTTP server — incoming requests → scheduler queue, scheduler token output → SSE stream, request cancellation on disconnect (depends on T031, T034)
- [ ] T036 [US2] Modify decode loop to support batched execution — batch multiple sequences in single command buffer submission, interleave prefill and decode (depends on T025, T031)

### Validation

- [ ] T037 [US2] Create API integration test in loops/test_api.ts — send chat completion request, verify OpenAI-compatible response format
- [ ] T038 [US2] Create streaming test in loops/test_streaming.ts — send 4 concurrent streaming requests, verify all complete correctly
- [ ] T039 [US2] Create KV cache leak test — send 1000 requests, verify VRAM usage returns to baseline after all complete

**Checkpoint**: Server handles 4 concurrent requests at target throughput. OpenAI API is drop-in compatible. No memory leaks.

---

## Phase 5: User Story 3 — TurboQuant KV Cache Compression (Priority: P3)

**Goal**: Compress KV cache to 2-4 bits per coordinate using TurboQuant, achieving 5x memory reduction at 3-bit with >99.5% attention accuracy.

**Independent Test**: Enable `--kv-quant 3`, verify VRAM savings and attention cosine similarity >99.5% against FP16 baseline.

### CPU-Side Initialization

- [ ] T040 [P] [US3] Implement Lloyd-Max codebook solver in src/turboquant/lloyd_max.zig — trapezoidal numerical integration for E[X | partition], Gaussian approximation for d≥64, produces n_levels centroids. Validate against PyTorch reference.
- [ ] T041 [P] [US3] Implement random orthogonal matrix generation in src/turboquant/rotation.zig — Gaussian RNG, Householder QR decomposition, sign correction. Validate Pi @ Pi^T ≈ I.
- [ ] T042 [P] [US3] Implement QJL projection matrix generation in src/turboquant/qjl.zig — d×m matrix of N(0,1) from seeded PRNG, upload to GPU buffer.
- [ ] T043 [US3] Implement TurboQuant configuration and CLI parsing in src/turboquant/config.zig — TurboQuantOptions struct, `--kv-quant` flag (depends on T007)

### Compression Shaders

- [ ] T044 [P] [US3] Write key compression shader in src/shaders/tq_quantize_keys.comp — load key, normalize, rotate (Pi^T), quantize to nearest centroid, compute residual, project through S, store sign bits, pack indices (depends on T040-T042)
- [ ] T045 [P] [US3] Write value compression shader in src/shaders/tq_quantize_values.comp — same as keys but MSE-only (no QJL stage)
- [ ] T046 [P] [US3] Write asymmetric attention shader in src/shaders/tq_attention_scores.comp — rotate query once, dot with centroid lookups (term1), project query through S + dot with signs (term2), output scores
- [ ] T047 [P] [US3] Write value decompression shader in src/shaders/tq_decompress_values.comp — dequantize, unrotate, scale by vec_norm, fuse with weighted accumulation

### Host-Side Integration

- [ ] T048 [US3] Implement GPU compression dispatch in src/turboquant/compress.zig — buffer management for compressed pages, push constants for bit-width and dimensions, dispatch key/value compression after K/V projection (depends on T044-T045)
- [ ] T049 [US3] Extend KVPage to support compressed format in src/scheduler/kv_cache.zig — CompressedKeyPage and CompressedValuePage variants, format-aware page allocation (depends on T029, T043)
- [ ] T050 [US3] Integrate asymmetric attention into attention dispatch in src/compute/attention.zig — detect compressed pages, dispatch tq_attention_scores instead of standard flash_attn, dispatch tq_decompress_values for weighted sum (depends on T023, T046-T047)
- [ ] T051 [US3] Hook compression into decode loop — compress KV after projection during prefill (batch) and decode (per-token), configurable via --kv-quant flag (depends on T036, T048)

### Validation

- [ ] T052 [US3] Create attention cosine similarity test — compress random vectors on GPU, compute attention scores, compare against FP16 baseline (target >99.5%)
- [ ] T053 [US3] Create inner product bias test — compute mean error of <q, k_compressed> vs <q, k_fp16> over 10K random vectors (target bias <0.001)
- [ ] T054 [US3] Create VRAM savings test — measure actual compressed page sizes and compare against theoretical (5x at 3-bit)
- [ ] T055 [US3] Create concurrent request test — verify 8 concurrent 8K-context requests on 16GB GPU with TQ-3bit (should OOM without TQ)

**Checkpoint**: TurboQuant compression works end-to-end. 5x KV cache reduction. Attention accuracy preserved. More concurrent requests fit in VRAM.

---

## Phase 6: User Story 4 — MoE and SSM/Mamba Support (Priority: P4)

**Goal**: Extend architecture support to Qwen MoE and Mamba/Jamba hybrid models.

**Independent Test**: Load Qwen3.5-35B-A3B, generate text, validate output against llama.cpp reference.

- [ ] T056 [P] [US4] Write fused SOFTMAX_TOPK shader in src/shaders/softmax_topk.comp — softmax + top-k selection for MoE expert routing
- [ ] T057 [P] [US4] Write fused SIGMOID_MUL shader in src/shaders/sigmoid_mul.comp — sigmoid(x) * y for SSM/Mamba gating
- [ ] T058 [US4] Implement MoE expert routing dispatch in src/compute/moe.zig — expert selection via SOFTMAX_TOPK, sparse expert matmul via MUL_MAT_ID, result combination (depends on T021, T056)
- [ ] T059 [US4] Extend architecture graph builder for Qwen MoE in src/model/architecture.zig — shared attention + per-expert FFN, expert routing nodes, sparse activation (depends on T011, T058)
- [ ] T060 [US4] Extend architecture graph builder for Mamba/Jamba in src/model/architecture.zig — SSM conv, gated delta net, sigmoid_mul, interleaved attention+SSM layers (depends on T011, T057)
- [ ] T061 [US4] Create MoE validation test — load Qwen MoE model, verify expert routing selects correct top-k, compare output against llama.cpp reference
- [ ] T062 [US4] Create Mamba validation test — load Jamba model, verify SSM operations, compare output against llama.cpp reference

**Checkpoint**: MoE and SSM/Mamba architectures work correctly. Output matches reference implementation.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Optimization, hardening, and cross-story improvements

- [ ] T063 [P] Fuse rotation + quantize into single shader pass for TurboQuant (eliminate intermediate buffer)
- [ ] T064 [P] Implement selective compression — keep recent N tokens in FP16 (sliding window), compress older pages
- [ ] T065 Profile and optimize workgroup sizes for all shaders on RDNA4 with real model data
- [ ] T066 [P] Add GPU ECC detection and warning in src/vulkan/gpu_detect.zig
- [ ] T067 [P] Add VRAM insufficiency detection — calculate required VRAM before loading, report shortfall
- [ ] T068 Implement graceful client disconnect handling — detect closed SSE connection, free request slot and KV pages within one scheduler tick
- [ ] T069 Run quickstart.md validation scenarios end-to-end on RDNA4 hardware

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T007)
- **US1 (Phase 3)**: Depends on Foundational (T008-T012) — MVP
- **US2 (Phase 4)**: Depends on US1 completion (needs working decode loop)
- **US3 (Phase 5)**: Depends on US2 completion (needs paged KV cache)
- **US4 (Phase 6)**: Depends on Foundational (T011 architecture builder) — can start after Phase 2, parallel with US2/US3
- **Polish (Phase 7)**: Depends on US1-US3 completion

### User Story Dependencies

- **US1 (P1)**: Blocked by Foundational. No other story dependencies.
- **US2 (P2)**: Depends on US1's decode loop (T024-T025) for batched execution.
- **US3 (P3)**: Depends on US2's paged KV cache (T029) for compressed page integration.
- **US4 (P4)**: Only depends on Foundational (T011). Can be developed in parallel with US2/US3 after Phase 2.

### Within Each User Story

- Shaders can be written in parallel (all marked [P])
- Host-side dispatch depends on corresponding shaders
- Integration depends on all dispatch modules
- Validation depends on integration

### Parallel Opportunities

- All shader tasks within a phase marked [P] can be written simultaneously
- US4 (MoE/Mamba) can start after Phase 2, independent of US2/US3
- TurboQuant CPU initialization (T040-T042) can run in parallel with each other
- TurboQuant GPU shaders (T044-T047) can run in parallel with each other

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (Vulkan infrastructure)
2. Complete Phase 2: Foundational (GGUF loading, compute graph)
3. Complete Phase 3: User Story 1 (single-request inference)
4. **STOP and VALIDATE**: Run bandwidth benchmark, logit comparison, generate sample text
5. This alone delivers value — a fast RDNA4-tuned inference CLI

### Incremental Delivery

1. Setup + Foundational → Infrastructure ready
2. US1 → Single-request inference → Benchmark → MVP!
3. US2 → Server + batching → API integration tests → Production-capable
4. US3 → TurboQuant KV → Memory tests → Extended concurrency
5. US4 → MoE/Mamba → Architecture coverage
6. Polish → Optimization, hardening

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All shader development can happen in parallel within a phase
- Total tasks: 69
