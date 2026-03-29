# Tasks: Apple Silicon Inference (Metal Backend)

**Input**: Design documents from `/specs/005-apple-silicon-inference/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks grouped by user story. US3 (Cross-Platform GGUF) and US1 (Fast Decode) are both P1 but US3 is a prerequisite for US1. US2 (Parallel) depends on US1.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3)

---

## Phase 1: Setup

**Purpose**: Create directory structure and configure build system for Metal backend on macOS.

- [x] T001 Create Metal backend directory structure: `src/metal/`, `src/gpu/`, `src/shaders/metal/`
- [x] T002 Update `build.zig` to compile `src/metal/shim.m` with `-fobjc-arc`, link Metal.framework and Foundation.framework on macOS, skip Vulkan and glslc on macOS
- [x] T003 [P] Add SPIRV-Cross shader cross-compilation step to `build.zig`: for each `src/shaders/*.comp`, run `spirv-cross --msl` to produce `src/shaders/metal/*.metal` (build-time, macOS only)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Metal shim, Zig wrappers, and GPU abstraction that ALL user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

### Metal Shim (C API boundary)

- [x] T004 Implement Metal shim C header and ObjC implementation per contract in `src/metal/shim.h` + `src/metal/shim.m` — device lifecycle (`mtl_init`, `mtl_destroy`, `mtl_chip_family`, `mtl_max_buffer_size`, `mtl_total_memory`), buffer management (`mtl_create_buffer`, `mtl_wrap_mmap`, `mtl_buffer_contents`, `mtl_free_buffer`), pipeline management (`mtl_create_pipeline`, `mtl_create_pipeline_from_lib`, `mtl_pipeline_max_threads`, `mtl_free_pipeline`), command buffer and dispatch (`mtl_begin_command`, `mtl_dispatch`, `mtl_barrier`, `mtl_commit_and_wait`, `mtl_commit_async`, `mtl_wait`)

### Metal Zig Wrappers

- [x] T005 [P] Implement `MetalDevice` in `src/metal/device.zig` — @cImport shim.h, wrap `mtl_init`/`mtl_destroy`, expose `ChipFamily` enum (m1/m2/m3/m4/unknown) from `mtl_chip_family`, implement `maxBufferSize()` and `totalMemory()` methods
- [x] T006 [P] Implement `MetalBuffer` in `src/metal/buffer.zig` — wrap `mtl_create_buffer`/`mtl_wrap_mmap`/`mtl_free_buffer`, expose `cpu_ptr` and `size` fields, track `is_mmap_wrapped` flag per data-model.md
- [x] T007 [P] Implement `MetalPipeline` in `src/metal/pipeline.zig` — wrap `mtl_create_pipeline`/`mtl_create_pipeline_from_lib`/`mtl_free_pipeline`, expose `max_threads_per_threadgroup`
- [x] T008 [P] Implement Metal command buffer in `src/metal/command.zig` — wrap `mtl_begin_command`/`mtl_dispatch`/`mtl_barrier`/`mtl_commit_and_wait`/`mtl_commit_async`/`mtl_wait`

### GPU Abstraction Layer

- [x] T009 Create comptime GPU abstraction in `src/gpu/interface.zig` per gpu-abstraction contract — `Backend`, `Buffer`, `Pipeline` type aliases resolved via `builtin.os.tag`, zero runtime overhead
- [ ] T010 [P] Refactor `src/compute/dmmv.zig` to use `gpu.Backend`, `gpu.Buffer`, `gpu.Pipeline` instead of direct `vulkan.*` imports
- [ ] T011 [P] Refactor `src/compute/elementwise.zig` to use `gpu.Backend` types and dispatch methods
- [ ] T012 [P] Refactor `src/compute/attention.zig` to use `gpu.Backend` types and dispatch methods
- [ ] T013 Refactor `src/compute/forward.zig` to use `gpu.Backend` — replace all `vulkan.Instance`, `vulkan.Buffer`, `vulkan.Pipeline` references with `gpu.Backend`, `gpu.Buffer`, `gpu.Pipeline` (largest refactor, ~2000 lines)
- [x] T014 Refactor `src/model/loader.zig` to use `gpu.Backend` for buffer creation — Metal path uses `wrapMmap()`, Vulkan path uses existing staging upload

### Shader Cross-Compilation

- [x] T015 [P] Cross-compile critical-path GLSL shaders to MSL via SPIRV-Cross: `dmmv_q4k`, `dmmv_q8_0`, `dmmv_q5k`, `dmmv_q6k`, `dmmv_f16`, `dmmv_f32` in `src/shaders/metal/`
- [x] T016 [P] Cross-compile elementwise shaders to MSL: `rms_norm_mul`, `swiglu`, `rope_fused`, `deinterleave`, `sigmoid_mul`, `vadd`, `scale_accumulate` in `src/shaders/metal/`
- [x] T017 [P] Cross-compile attention and special shaders to MSL: `flash_attn`, `embed_dequant_q4k`, `ssm_conv1d`, `ssm_delta_net`, `ssm_gated_norm` in `src/shaders/metal/`
- [x] T018 Verify all cross-compiled MSL shaders compile with `xcrun metal` — fix any SPIRV-Cross translation issues

### Validation

- [x] T019 Verify `zig build` succeeds on macOS with Metal backend (compiles shim, links frameworks, creates zinc binary)
- [x] T020 Verify `zig build test` passes on macOS (unit tests that don't require GPU still pass)
- [ ] T021 Verify Vulkan inference on Linux test node after GPU abstraction refactor — rsync refactored code, `zig build && zig build test`, run full decode with `--prompt "The capital of France is" --max-tokens 256`, verify token output is unchanged and tok/s is within 2% of pre-refactor baseline (captures regression from T010-T014 refactor)

**Checkpoint**: Metal backend compiles, GPU abstraction in place, all shaders translated. Vulkan path verified unbroken.

---

## Phase 3: User Story 3 — Cross-Platform GGUF (Priority: P1)

**Goal**: Same GGUF model file loads correctly on Metal, all tensors parse and dequantize to same values as Vulkan path.

**Independent Test**: Load `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` on Metal, verify tensor count, dimensions, and dequantized values match a reference snapshot from the Vulkan path.

### Implementation

- [x] T022 [US3] Implement zero-copy mmap buffer wrapping in `src/metal/buffer.zig` — page-align mmap pointer, call `mtl_wrap_mmap`, validate buffer creation succeeds for model-sized regions (~20 GB)
- [x] T023 [US3] Update `src/model/loader.zig` Metal path — when backend is Metal, use `gpu.Backend.wrapMmap()` for tensor data instead of staging buffer upload. Preserve Vulkan path unchanged.
- [x] T024 [US3] Add startup validation in `src/metal/device.zig` — check Apple Silicon (`mtl_init` non-null), check macOS ≥14, check available memory vs model size, print actionable error messages per spec Error Handling section
- [x] T025 [US3] Add `--context-length` CLI flag in `src/main.zig` (default 4096, max 32768, reject values >32768 with error) for configurable KV cache allocation
- [x] T026 [US3] End-to-end model load test — load `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` on Metal, verify all tensor count matches, print model config, confirm <2s load time (mmap, no copy)

**Checkpoint**: GGUF loads on Metal with zero-copy. Model config prints correctly. Same file works on both backends.

---

## Phase 4: User Story 1 — Fast Single-Request Decode (Priority: P1) MVP

**Goal**: Generate tokens on Metal at ≥80 tok/s with output matching Vulkan backend.

**Independent Test**: `./zig-out/bin/zinc -m Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --prompt "The capital of France is" --max-tokens 256` produces coherent text at ≥80 tok/s on M4 Max 64 GB.

### Reference Data

- [ ] T027 [US1] Generate reference token output and logits snapshot — run llama.cpp Metal (`llama-cli -m Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf -ngl 99 -p "The capital of France is" -n 256`) on this Mac, capture greedy-decoded token IDs and first-token logits (top-10 values) to `specs/005-apple-silicon-inference/reference-tokens.txt`. Also capture Vulkan ZINC output from Linux node via SSH for cross-backend comparison.

### Implementation — Get It Working

- [x] T028 [US1] Wire `InferenceEngine.init()` in `src/compute/forward_metal.zig` to Metal backend — allocate intermediate buffers (hidden, norm, q, k, v, attn_out, gate, up, swiglu, down, moe_out, logits) as Metal shared buffers, create all compute pipelines from cross-compiled MSL
- [ ] T029 [US1] Wire `InferenceEngine.decodeStep()` to Metal dispatch (partial: LM head CPU matmul works, 40-layer GPU dispatch pending) — embed lookup, per-layer attention/SSM, MoE FFN, final norm, LM head projection, logits readback via direct pointer (no staging needed on Metal)
- [ ] T030 [US1] Wire `InferenceEngine.prefillBatch()` to Metal — batch all prompt tokens in single GPU submission
- [ ] T031 [US1] End-to-end single-token decode validation — generate 1 token, compare top-10 logits against reference snapshot from T027
- [ ] T032 [US1] Multi-token generation test — greedy-decode 256 tokens with prompt "The capital of France is", verify token IDs match reference from T027

### Correctness Validation (Constitution VI)

- [ ] T033 [US1] Compute cosine similarity of Metal flash attention output vs FP16 reference — for a test prompt, dump attention output tensor from Metal `flash_attn.metal` and from a known-correct FP16 path (llama.cpp or Vulkan), compute cosine similarity, verify >99.5% per constitution requirement. Log results to `specs/005-apple-silicon-inference/baselines.md`.

### Implementation — Make It Fast

- [ ] T034 [US1] Benchmark cross-compiled shader performance — measure tok/s, record in `specs/005-apple-silicon-inference/baselines.md`
- [ ] T035 [US1] Hand-optimize `src/shaders/metal/dmmv_q4k.metal` — 32-thread simdgroups, `simdgroup_matrix` 8x8 multiply-accumulate, shared memory tiling tuned for Apple GPU 8KB L1. Record before/after tok/s and bandwidth utilization.
- [ ] T036 [P] [US1] Hand-optimize `src/shaders/metal/flash_attn.metal` — `simdgroup_async_copy` for overlapped K/V loads (M2+ codepath with M1 fallback), 32-thread tiles, online softmax. Record before/after profiling data.
- [ ] T037 [P] [US1] Tune `src/shaders/metal/rms_norm_mul.metal`, `swiglu.metal`, `rope_fused.metal` for 32-thread simdgroups and Metal dispatch-time workgroup sizes. Record before/after profiling data.
- [ ] T038 [US1] Add build-time `.metallib` compilation in `build.zig` — compile all `src/shaders/metal/*.metal` into single `zinc.metallib`, load via `mtl_create_pipeline_from_lib`
- [ ] T039 [US1] Profile with Metal System Trace (Instruments) — identify GPU idle time, bandwidth utilization, shader occupancy. Iterate on hot kernels until ≥75% bandwidth utilization
- [ ] T040 [US1] Final single-request benchmark — validate ≥80 tok/s on M4 Max 64 GB, record in baselines.md

**Checkpoint**: Single-request inference works correctly at ≥80 tok/s. Attention cosine similarity >99.5%. Tokens match reference. Cross-compiled and hand-optimized kernels.

---

## Phase 5: User Story 2 — Parallel Request Serving (Priority: P2)

**Goal**: Serve multiple concurrent requests via OpenAI-compatible API with ≥200 tok/s aggregate throughput.

**Independent Test**: Launch ZINC server, send 5 concurrent `curl` requests, verify each gets ≥40 tok/s and aggregate is ≥200 tok/s. Send request 17+ and verify HTTP 503.

### Implementation

- [ ] T041 [US2] Extend `src/scheduler/scheduler.zig` for concurrent Metal KV page allocation — dynamic page assignment per request, page recycling on completion/failure
- [ ] T042 [US2] Implement batched dispatch in `src/compute/forward.zig` — group sequences at same decode step, single Metal GPU submit for all sequences in batch, batched DMMV with batch dimension in push constants
- [ ] T043 [US2] Extend `src/server/http.zig` for concurrent Metal connections — SSE streaming per request, connection lifecycle management
- [ ] T044 [US2] Add max concurrent request enforcement (16) in `src/server/http.zig` — HTTP 503 `{"error": "max concurrent requests (16) reached"}` when limit exceeded, do not crash
- [ ] T045 [US2] Add GPU command timeout handling in `src/metal/command.zig` — detect `MTLCommandBuffer.status == .error`, abort current request, log error, keep server available
- [ ] T046 [US2] Benchmark 5 concurrent requests — validate ≥200 tok/s aggregate, ≥40 tok/s per request, no starvation. Record in baselines.md

**Checkpoint**: ZINC serves parallel requests on Metal. 5 concurrent requests at ≥200 tok/s aggregate. Server handles overload gracefully.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: CI, compatibility, documentation.

- [ ] T047 [P] Add macOS CI workflow in `.github/workflows/macos-build.yml` — build + unit test on Apple Silicon runner
- [ ] T048 [P] Test with additional models on Metal: Llama-3, Mistral, DeepSeek — verify loading and generation
- [ ] T049 [P] Validate all quant types on Metal: Q4_K, Q5_K, Q6_K, Q8_0, F16 — verify dequant correctness against Vulkan reference
- [ ] T050 Update `AGENTS.md` with Metal build instructions, architecture diagram, and `src/metal/` + `src/gpu/` module descriptions
- [ ] T051 Update `docs/SPEC.md` with dual-backend architecture overview

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US3 — GGUF)**: Depends on Phase 2 — model must load before decode
- **Phase 4 (US1 — Decode)**: Depends on Phase 3 — decode needs loaded model
- **Phase 5 (US2 — Parallel)**: Depends on Phase 4 — batching needs working single decode
- **Phase 6 (Polish)**: Depends on Phase 4 minimum (Phase 5 preferred)

### User Story Dependencies

- **US3 (P1)**: First — model loading is prerequisite for everything
- **US1 (P1)**: Second — single-request decode, depends on US3
- **US2 (P2)**: Third — parallel serving, depends on US1

### Within-Phase Parallel Opportunities

**Phase 2**:
```
T004 (shim) → T005, T006, T007, T008 (all [P] — different files)
T009 (gpu abstraction) → T010, T011, T012 (all [P] — different compute files)
T013 (forward.zig refactor) — sequential, depends on T009-T012
T015, T016, T017 (shader cross-compile) — all [P], independent of Zig refactors
T021 (Vulkan regression) — after T013-T014, requires Linux test node
```

**Phase 4**:
```
T027 (reference snapshots) — first, before any validation
T028-T032 (get it working) — sequential
T033 (cosine similarity) — after T031, requires attention output dump
T035, T036, T037 (shader optimization) — T036 and T037 are [P]
```

**Phase 6**:
```
T047, T048, T049 — all [P]
```

---

## Parallel Example: Phase 2 Foundational

```
# After T004 (shim) completes, launch Metal wrapper tasks in parallel:
Task T005: "Implement MetalDevice in src/metal/device.zig"
Task T006: "Implement MetalBuffer in src/metal/buffer.zig"
Task T007: "Implement MetalPipeline in src/metal/pipeline.zig"
Task T008: "Implement Metal command buffer in src/metal/command.zig"

# Simultaneously, launch shader cross-compilation in parallel:
Task T015: "Cross-compile critical-path GLSL shaders to MSL (dmmv_*)"
Task T016: "Cross-compile elementwise shaders to MSL"
Task T017: "Cross-compile attention and special shaders to MSL"
```

## Parallel Example: Phase 4 Performance

```
# After T035 (dmmv_q4k optimization), launch remaining shader optimization in parallel:
Task T036: "Hand-optimize flash_attn.metal"
Task T037: "Tune fused ops (rms_norm, swiglu, rope) for 32-thread simdgroups"
```

---

## Implementation Strategy

### MVP First (US3 + US1)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks everything)
3. Complete Phase 3: US3 — GGUF loads on Metal
4. Complete Phase 4: US1 — Single-request decode at ≥80 tok/s
5. **STOP and VALIDATE**: Run full benchmark suite, compare against baselines
6. Ship CLI-only Metal inference

### Full Delivery

7. Complete Phase 5: US2 — Parallel request serving
8. Complete Phase 6: Polish — CI, model compat, docs
9. Ship production Metal backend with server mode

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [US*] label maps task to spec user story for traceability
- Shader cross-compilation (T015-T017) is a fast path — hand optimization (T035-T037) follows in Phase 4
- All Metal code is macOS-only via comptime — Vulkan path must remain untouched
- Commit after each task. Run `zig build` after each Phase 2 task to catch compile errors early
- Stop at Phase 4 checkpoint for MVP validation before proceeding to server work
- Constitution VI requires >99.5% attention cosine similarity — T033 validates this explicitly
- Constitution I requires before/after profiling for kernel changes — T035-T037 each record profiling data
