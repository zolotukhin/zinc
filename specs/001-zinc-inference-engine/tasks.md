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

- [X] T001 Configure build.zig to compile GLSL shaders to SPIR-V via system glslc and link Vulkan loader
- [X] T002 [P] Implement Vulkan instance creation, physical device selection, and compute queue setup in src/vulkan/instance.zig
- [X] T003 [P] Implement GPU buffer allocation, memory type selection, and staging buffer management in src/vulkan/buffer.zig
- [X] T004 [P] Implement GPU capability detection and auto-tuning parameter derivation in src/vulkan/gpu_detect.zig (vendor, VRAM, bandwidth, CU count, wave size, coopmat support → derived workgroup sizes, tile dims)
- [X] T005 Implement compute pipeline creation from SPIR-V modules with specialization constants in src/vulkan/pipeline.zig (depends on T001)
- [X] T006 Implement command buffer recording, submission, and pre-recorded replay in src/vulkan/command.zig (depends on T002)
- [X] T007 [P] Implement CLI argument parsing in src/main.zig (model path, port, kv-quant, prompt)

**Checkpoint**: Vulkan infrastructure ready — can create device, allocate buffers, compile and dispatch shaders, detect GPU capabilities.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: GGUF loading and compute graph infrastructure that all user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T008 Implement GGUF file parser in src/model/gguf.zig — parse header, tensor descriptors, metadata key-value pairs, support split files
- [X] T009 Implement model loader in src/model/loader.zig — memory-map GGUF file, DMA weight tensors to GPU VRAM buffers, parse architecture metadata (depends on T003, T008)
- [X] T010 Implement compute graph definition in src/compute/graph.zig — represent ops as nodes, support topological ordering, static graph for decode path
- [X] T011 [P] Implement architecture-specific graph builders for LLaMA/Mistral/Qwen transformer in src/model/architecture.zig (depends on T009, T010)
- [X] T012 [P] Implement tokenizer interface in src/model/tokenizer.zig — shell out to external sentencepiece/tiktoken process, handle BPE encoding/decoding

**Checkpoint**: Can load a GGUF model into GPU VRAM, construct the compute graph, and tokenize input. Ready for kernel implementation.

---

## Phase 3: User Story 1 — Single-Request Inference (Priority: P1) MVP

**Goal**: Load a GGUF model and generate text from a single prompt with high bandwidth utilization.

**Independent Test**: Load Qwen3-8B Q4_K, generate 256 tokens, compare logits against llama.cpp (>99.5% cosine similarity), verify 90%+ bandwidth on large matmuls.

### GPU Kernel Implementation

- [X] T013 [P] [US1] Write DMMV Q4_K shader in src/shaders/dmmv_q4k.comp — wave64, specialization constants for M/K, 2 rows per workgroup
- [X] T014 [P] [US1] Write DMMV Q8_0 shader in src/shaders/dmmv_q8_0.comp — optimized for attention weight matmul
- [X] T015 [P] [US1] Write DMMV F16 shader in src/shaders/dmmv_f16.comp — for KV cache and small tensors
- [X] T016 [P] [US1] Write fused RMS_NORM_MUL shader in src/shaders/rms_norm_mul.comp — RMS norm + scale multiply in single dispatch
- [X] T017 [P] [US1] Write fused SwiGLU shader in src/shaders/swiglu.comp — SILU(x) * y
- [X] T018 [P] [US1] Write fused ROPE shader in src/shaders/rope_fused.comp — RoPE + reshape + cache write
- [X] T019 [P] [US1] Write flash attention shader in src/shaders/flash_attn.comp — paged, 256-token blocks, GQA support
- [X] T020 [P] [US1] Write cooperative matrix matmul shader in src/shaders/coop_matmul.comp — 16x16x16 tiles for prefill

### Host-Side Dispatch

- [X] T021 [US1] Implement DMMV dispatch logic in src/compute/dmmv.zig — select shader by quant type, set specialization constants, manage push constants (depends on T005, T013-T015)
- [X] T022 [US1] Implement fused element-wise dispatch in src/compute/elementwise.zig — dispatch RMS_NORM_MUL, SwiGLU, ROPE_FUSED with correct buffer bindings (depends on T005, T016-T018)
- [X] T023 [US1] Implement flash attention dispatch in src/compute/attention.zig — page table lookup, block iteration, GQA head mapping (depends on T005, T019)

### Forward Pass Integration

- [X] T024 [US1] Implement single-request decode loop in src/main.zig — tokenize prompt, prefill, decode token-by-token, detokenize output (depends on T011, T021-T023)
- [X] T025 [US1] Implement command buffer pre-recording for decode graph in src/vulkan/command.zig — record once, replay per token via vkQueueSubmit (depends on T006, T024)

### Validation

- [X] T026 [US1] Create bandwidth utilization benchmark in benchmarks/bandwidth.zig — measure effective GB/s for each DMMV quant type at various matrix sizes
- [X] T027 [US1] Create dispatch overhead benchmark in benchmarks/dispatch.zig — measure single dispatch, 1500 dispatches, and pre-recorded replay
- [X] T028 [US1] Create logit comparison test — generate tokens with ZINC and llama.cpp on same model/prompt/seed, compute cosine similarity (target >99.5%)

**Checkpoint**: Infrastructure complete — shaders compile, dispatch wrappers work, forward pass executes all 40 layers with MoE routing and SSM state. Output not yet coherent.

---

## Phase 3b: Forward Pass Correctness ✅ COMPLETE

**Status**: All correctness bugs FIXED. Output: "Paris. The capital of Germany is Berlin. The capital of Italy is Rome..."

Bugs found and fixed (13 total, by self-improving loop + manual debugging):
- [X] T028a Q4_K DMMV sub-block pairing (consecutive pairs, not stride-4)
- [X] T028b Q5_K DMMV element ordering (contiguous, not interleaved)
- [X] T028c Q8_0/F16 wave32 cross-subgroup reduction
- [X] T028d Q4_K SPEC_K vs push-constant K for variable dimensions
- [X] T028e Decode loop: sample first token from prefill logits (was duplicating last prompt token)
- [X] T028f SSM conv1d: convolve before state update (was double-counting)
- [X] T028g SSM norm weight indexing (per-head vs shared)
- [X] T028h Shared expert sigmoid gating implementation
- [X] T028i Shared expert intermediate dim (was using per-expert 512 instead of actual)
- [X] T028j attn_out_buf buffer overflow (q_dim*4 → q_dim*2*4 for Q+gate)
- [X] T028k Flash attention page table (identity mapping)
- [X] T028l IMRoPE partial rotation (64/256 dims)
- [X] T028m head_dim from GGUF (256, not hidden_dim/n_heads=128)

**Checkpoint**: ✅ Output is coherent English. First token "Paris" matches llama.cpp. 26 build tests pass.

---

## Phase 3c: Decode Performance (Priority: P0 — CURRENT)

**Goal**: Achieve 107+ tok/s decode on RDNA4 (matching llama.cpp baseline). Currently 4 tok/s at 0.4% bandwidth utilization.

**Root Cause**: ~1600 vkQueueSubmit+vkWaitForFences per token. Each submit has ~0.1-0.5ms kernel overhead = ~200ms wasted. GPU is idle 99.6% of the time. Theoretical throughput at 576 GB/s is ~27 tok/s for this 21GB model.

**Independent Test**: Run `zinc --prompt "The capital of France is" --max-tokens 256` and measure decode tok/s. Target: ≥107 tok/s.

### Command Buffer Batching (highest impact)

- [ ] T029a [US1] Batch command buffers across attention layers — keep one cmd buffer open for norm + Q/K/V DMMV + RoPE + KV cache write + flash attention + gate + O-proj + residual + post-norm + MoE. Only submit for router readback. Target: 1 submit per attention layer (was ~15). File: src/compute/forward.zig
- [ ] T029b [US1] Batch command buffers across MoE expert dispatch — record all 8 experts (gate+up+SwiGLU+down+accumulate) in one cmd buffer before submitting. Currently submits per-expert. File: src/compute/forward.zig
- [ ] T029c [US1] Pre-allocate descriptor pool per layer — eliminate per-operation vkResetDescriptorPool + allocation. File: src/compute/forward.zig

### GPU-Side Router (eliminates 40 readbacks)

- [ ] T029d [US1] Write softmax+top-k compute shader — takes router logits buffer, outputs expert_ids[k] + expert_weights[k] to GPU buffer. CPU never sees router logits. File: src/shaders/softmax_topk.comp
- [ ] T029e [US1] Integrate GPU router into forward pass — dispatch softmax_topk shader, read expert IDs from GPU buffer for expert dispatch. File: src/compute/forward.zig

### GPU-Side SSM (eliminates 30 roundtrips)

- [ ] T029f [US1] Write conv1d + SiLU compute shader — fused convolution with state buffer on GPU. State is persistent GPU buffer, no CPU readback. File: src/shaders/ssm_conv1d.comp
- [ ] T029g [US1] Write delta-net state update shader — decay + outer product + readout as GPU compute. State buffer stays on GPU. File: src/shaders/ssm_delta_net.comp
- [ ] T029h [US1] Write SSM gated norm shader — fused RMS_norm(output) * SiLU(gate). File: src/shaders/ssm_gated_norm.comp
- [ ] T029i [US1] Integrate GPU SSM into forward pass — replace runSsmLayerCpu with GPU dispatch chain. Eliminate logits_staging readback for SSM. File: src/compute/forward.zig

### Shader Performance

- [ ] T029j [US1] Profile per-dispatch timing — add Vulkan timestamp queries around each shader dispatch to identify which shaders dominate runtime. File: src/compute/forward.zig
- [ ] T029k [US1] Optimize Q4_K DMMV occupancy — with batched submits, GPU utilization should approach 90%+. Tune workgroup sizes if needed. File: src/shaders/dmmv_q4k.comp
- [ ] T029l [US1] Increase max_tokens to 256 — once performance allows <60s generation. File: src/main.zig

**Checkpoint**: Decode throughput ≥27 tok/s (bandwidth-limited). Ready for further optimization toward 107 tok/s.

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

## Phase 6: User Story 4 — MoE and SSM/Mamba Support ✅ COMPLETE

**Status**: Qwen3.5-35B-A3B (hybrid attention+SSM+MoE) generates correct output.

- [X] T056 [US4] sigmoid_mul shader — sigmoid(x) * y for attention gating
- [X] T057 [US4] vadd, scale_accumulate, deinterleave shaders — MoE accumulation primitives
- [X] T058 [US4] MoE expert routing — CPU softmax+top-k, stacked 3D tensor offset dispatch, 256 experts top-8
- [X] T059 [US4] Shared expert path — gate+up+SwiGLU+down with sigmoid gating
- [X] T060 [US4] SSM delta-net — CPU-side conv1d + state decay + outer product update + gated norm
- [X] T061 [US4] MoE validation — expert offsets verified via build-time tests (expertSliceBytes, topKSoftmax)
- [X] T062 [US4] SSM validation — delta-net zero-state test, conv1d output verified against CPU reference

**Checkpoint**: ✅ MoE + SSM work correctly. Output matches llama.cpp quality.

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
