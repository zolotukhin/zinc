# Tasks: Decode Performance Optimization

**Input**: Design documents from `/specs/003-decode-performance/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1 = Fast Token Generation, US2 = Profiling, US3 = Correctness Validation

---

## Phase 1: Setup

**Purpose**: Allocate persistent GPU buffers and prepare infrastructure for GPU-side computation.

- [X] T001 Record pre-optimization reference output — run `zinc --prompt "The capital of France is" --max-tokens 32` and save the 32 generated tokens to specs/003-decode-performance/reference-tokens.txt for correctness validation. File: specs/003-decode-performance/reference-tokens.txt
- [X] T002 Allocate persistent GPU SSM conv state buffers at engine init — 40 device-local buffers of (d_conv−1) × conv_channels × sizeof(f32) = 96 KB each, initialized to zero. Replace CPU-side ssm_conv_states arrays. File: src/compute/forward.zig
- [X] T003 [P] Allocate persistent GPU SSM recurrent state buffers at engine init — 40 device-local buffers of num_heads × head_v_dim × head_v_dim × sizeof(f32) = 2 MB each, initialized to zero. Replace CPU-side ssm_states arrays. File: src/compute/forward.zig
- [X] T004 [P] Allocate GPU router output buffer — single host-visible buffer of 8 × sizeof(u32) + 8 × sizeof(f32) = 64 bytes for expert_ids + expert_weights output from softmax_topk shader. File: src/compute/forward.zig

**Checkpoint**: Engine initializes with GPU-side SSM state and router output buffers. Existing decode still works (buffers allocated but not yet used by new shaders).

---

## Phase 2: User Story 2 — Profiling Infrastructure (Priority: P2) 🎯 DO FIRST

**Goal**: Add per-dispatch GPU timestamps and per-phase CPU timings to identify the actual bottleneck split between CPU SSM, submit overhead, and GPU underutilization.

**Independent Test**: Run `zinc --prompt "The capital of France is" --max-tokens 8 --profile` and verify timing breakdown printed per layer, with totals accounting for ≥95% of wall-clock time.

- [X] T005 [US2] Add --profile CLI flag to src/main.zig — when set, enable profiling mode. Default off. Pass flag through to InferenceEngine init. File: src/main.zig
- [X] T006 [US2] Create Vulkan timestamp query pool at device init — allocate VkQueryPool with 2 × max_dispatches_per_token queries (estimate: 2 × 800 = 1600). Read timestampPeriod from device properties for ns conversion. File: src/vulkan/instance.zig
- [X] T007 [US2] Add vkCmdWriteTimestamp before and after each shader dispatch in decodeStep — wrap each dispatchDmmv, recordRmsNorm, recordSwiglu, recordRoPE, recordFlashAttn, recordScaleAcc, etc. with timestamp writes. Only record when profiling enabled. File: src/compute/forward.zig
- [X] T008 [US2] Add std.time.Timer around CPU SSM phases — time conv1d, delta-net state update, gated norm, and alloc/free separately within runSsmLayerCpu. Print per-layer breakdown. File: src/compute/forward.zig
- [X] T009 [US2] Print per-token timing summary after each decode step — aggregate GPU dispatch times by shader type, CPU SSM time, submit overhead (total wall clock − GPU − CPU compute), and report percentages. File: src/compute/forward.zig

**Checkpoint**: Profiling output shows exactly where time is spent. Use this data to confirm implementation order for Phases 3-6.

---

## Phase 3: User Story 1+3 — GPU-Side SSM (Priority: P1, highest impact)

**Goal**: Move all CPU-side SSM computation to GPU shaders. Eliminates 30 submits/token and removes the likely-dominant CPU bottleneck. Output must remain token-identical to pre-optimization reference.

**Independent Test**: Run decode with GPU SSM, compare first 32 tokens against reference-tokens.txt. Must match exactly.

### New Shaders (parallel — different files)

- [X] T010 [P] [US1] Write ssm_conv1d.comp — 1D convolution over conv_channels=8192 with d_conv=4 taps, persistent GPU state buffer. Reads conv kernel weights from GPU tensor buffer (handle f16 storage). Fused SiLU: out[ch] = sum × sigmoid(sum). State shift: slide window left, write current input as newest. Dispatch: 128 workgroups × 64 threads (8192 channels / 64). File: src/shaders/ssm_conv1d.comp
- [X] T011 [P] [US1] Write ssm_delta_net.comp — per-head delta-net state update. 32 workgroups (one per head), 64 threads each. Per head: L2 normalize Q and K vectors (128 elems), read ssm_a for exponential decay, apply decay to 128×128 state, compute outer product K⊗V and accumulate, readout = state × Q. State buffer is persistent device-local. Bindings: conv_out (input), gate (z), alpha (dt), beta, ssm_a weights, state buffer (read+write), output buffer. File: src/shaders/ssm_delta_net.comp
- [X] T012 [P] [US1] Write ssm_gated_norm.comp — per-head RMS norm of delta-net output × SiLU(gate). Reads ssm_norm.weight from GPU tensor buffer (per-head if d_inner elements, else shared d_state). Fused: out[i] = (o[i]/rms) × w[i] × (z[i] × sigmoid(z[i])). Dispatch: 32 workgroups × 64 threads. File: src/shaders/ssm_gated_norm.comp

### Integration

- [X] T013 [US1] Create Vulkan compute pipelines for the 3 new SSM shaders — compile SPIR-V via system glslc, create pipeline layouts with appropriate descriptor set layouts and push constants. Add to InferenceEngine init alongside existing shader pipelines. File: src/compute/forward.zig
- [X] T014 [US1] Replace runSsmLayerCpu with GPU dispatch chain — within the existing command buffer (no new submit): dispatch ssm_conv1d → pipeline barrier → ssm_delta_net → pipeline barrier → ssm_gated_norm → pipeline barrier → ssm_out DMMV → residual. Remove the submitAndWait at L1490. Remove per-call alloc/free of conv_kernel_buf, ssm_output, conv_out, norm_w_buf. File: src/compute/forward.zig
- [ ] T015 [US3] Validate GPU SSM correctness — run decode with reference prompt, compare first 32 tokens against reference-tokens.txt. If tokens differ, add diagnostic readback after GPU ssm_gated_norm to compare per-layer output vs CPU reference values logged in SSM_DBG messages. File: src/compute/forward.zig

**Checkpoint**: GPU SSM replaces CPU SSM. 30 fewer submits. First 32 tokens match reference. Profiling shows CPU SSM time eliminated.

---

## Phase 4: User Story 1+3 — GPU-Side Router (Priority: P1)

**Goal**: Move MoE expert routing from CPU to GPU. Eliminates 40 submits/token (one per MoE layer).

**Independent Test**: Same reference prompt, first 32 tokens must match. Profiling shows no router readback time.

- [X] T016 [US1] Write softmax_topk.comp — replace placeholder. Input: router_logits_buf (256 floats). Output: expert_ids[8] (u32) + expert_weights[8] (f32) in router_output_buf. Single workgroup (64 threads). Algorithm: load 256 logits to shared memory, find top-8 via repeated parallel max+mask, softmax-normalize the 8 selected weights. File: src/shaders/softmax_topk.comp
- [X] T017 [US1] Create Vulkan compute pipeline for softmax_topk — compile SPIR-V, create pipeline layout with descriptor set layout (input: router_logits_buf, output: router_output_buf) and push constants (n_experts, n_experts_used). File: src/compute/forward.zig
- [X] T018 [US1] Integrate GPU router into forward pass — after router DMMV, dispatch softmax_topk shader instead of readback+CPU topKSoftmax. Read expert_ids and weights from host-visible router_output_buf (single vkCmdCopyBuffer to staging + submit, or use host-visible buffer directly). Remove submitAndWait at L1056 and CPU topKSoftmax call. Expert dispatch loop reads from mapped router output buffer. File: src/compute/forward.zig
- [ ] T019 [US3] Validate GPU router correctness — compare expert_ids and expert_weights from GPU softmax_topk vs CPU topKSoftmax for first 5 tokens. Same experts must be selected. Weights within ±1e-5. First 32 tokens match reference. File: src/compute/forward.zig

**Checkpoint**: GPU router replaces CPU router. ~40 fewer submits. Tokens match reference.

---

## Phase 5: User Story 1+3 — GPU-Side Shared Expert Gate (Priority: P1)

**Goal**: Eliminate shared expert gate sigmoid readback. Eliminates ~40 submits/token.

**Independent Test**: Same reference prompt, first 32 tokens must match.

- [X] T020 [US1] Move shared expert gate to GPU — replace readback+CPU sigmoid at L1158-1183. After shared expert down projection, dispatch sigmoid_mul (or sigmoid_scale_acc) to apply sigmoid(gate_scalar) × down_buf → accumulate into moe_out_buf. Gate scalar is in router_logits_buf[0] after the DMMV at L1141. Remove submitAndWait at L1175. Reuse existing sigmoid_scale_acc pipeline. File: src/compute/forward.zig
- [ ] T021 [US3] Validate shared expert gate correctness — compare gate weight from GPU sigmoid vs CPU sigmoid for first 5 tokens. Values within ±1e-6. First 32 tokens match reference. File: src/compute/forward.zig

**Checkpoint**: All CPU readbacks eliminated except expert_ids (Phase 4 staging). Submit count down to ~42/token. Profiling should show major speedup.

---

## Phase 6: User Story 1 — Command Buffer Batching (Priority: P1)

**Goal**: Batch command buffers across layers to minimize remaining submits. Target: ≤10 submits/token.

**Independent Test**: Same reference prompt, ≥27 tok/s, first 32 tokens match reference.

- [X] T022 [US1] Remove per-layer descriptor pool reset — eliminate vkResetDescriptorPool at L863. The pool has 4096 sets; 40 layers × ~15 sets = 600, well within budget. Only reset once before the full decode step. File: src/compute/forward.zig
- [X] T023 [US1] Batch command buffer across multiple layers — keep one command buffer open across consecutive layers. Only submit+wait when expert_ids must be read (MoE layers with GPU router). For attention-only layers and SSM layers (now fully GPU), no submit needed between layers. Target: 1 submit per MoE-layer cluster + 1 final. File: src/compute/forward.zig
- [X] T024 [US1] Add descriptor pool overflow detection — before allocDescSet, check if pool is near capacity. If so, submit current command buffer, reset pool, begin new command buffer. Prevents VK_ERROR_OUT_OF_POOL_MEMORY crash. File: src/compute/forward.zig
- [ ] T025 [US3] Validate batched decode correctness — first 32 tokens match reference. Run with --profile to verify submit count ≤10. File: src/compute/forward.zig

**Checkpoint**: Submit count ≤10/token. Throughput ≥27 tok/s (memory-bandwidth floor). Tokens match reference.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Shader tuning, cleanup, and final validation.

- [ ] T026 [P] [US1] Optimize Q4_K DMMV occupancy — with batched submits, profile actual GPU utilization via T029j timestamps. Tune workgroup sizes and shared memory usage if DMMV is bottleneck. File: src/shaders/dmmv_q4k.comp
- [X] T027 [P] [US1] Increase max_tokens to 256 — update default and CLI parsing once performance allows <60s generation. File: src/main.zig
- [X] T028 [P] [US1] Gate BOS diagnostic submits behind --debug flag — the per-layer hidden_buf readback at L1244-1316 adds 40 extra submits on first token. Only execute when --debug is passed. File: src/compute/forward.zig
- [ ] T029 [US1] Full 256-token validation — run `zinc --prompt "The capital of France is" --max-tokens 256`, verify coherent English output, report final tok/s. Target: ≥27 tok/s (milestone 1), ≥107 tok/s (stretch). File: specs/003-decode-performance/quickstart.md
- [ ] T030 Remove dead CPU SSM code — delete runSsmLayerCpu, topKSoftmax, CPU conv state alloc/free, ssm_hidden_staging buffer, and related CPU-side SSM helpers once GPU path is validated. File: src/compute/forward.zig

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Profiling)**: Depends on Phase 1 — DO FIRST to confirm bottleneck
- **Phase 3 (GPU SSM)**: Depends on Phase 1 (GPU buffers) — highest impact optimization
- **Phase 4 (GPU Router)**: Depends on Phase 1 (router output buffer) — independent of Phase 3
- **Phase 5 (Shared Expert Gate)**: Independent of Phases 3-4 — can run in parallel
- **Phase 6 (Batching)**: Depends on Phases 3-5 (all CPU readbacks must be eliminated first)
- **Phase 7 (Polish)**: Depends on Phase 6

### Parallel Opportunities

- T002, T003, T004 can run in parallel (different buffer allocations)
- T010, T011, T012 can run in parallel (different shader files)
- Phases 3, 4, 5 can run in parallel after Phase 1 (different code paths)
- T026, T027, T028 can run in parallel (different files)

### Within Each Phase

- Shaders before integration (compile shaders → create pipelines → integrate into forward pass)
- Integration before validation (implement → verify correctness)
- Validation before next phase (ensure no regression)

---

## Implementation Strategy

### MVP First (Phases 1-3 Only)

1. Complete Phase 1: Setup (GPU buffers)
2. Complete Phase 2: Profiling (confirm bottleneck)
3. Complete Phase 3: GPU SSM (biggest impact)
4. **STOP and VALIDATE**: Measure tok/s improvement, verify correctness
5. If significant speedup confirmed, proceed to Phases 4-6

### Incremental Delivery

1. Phase 1+2 → Profiling data available (understand bottleneck)
2. Phase 3 → GPU SSM (expect 2-3x speedup, ~10 tok/s)
3. Phase 4 → GPU Router (further ~1.5x, ~15 tok/s)
4. Phase 5 → Shared Expert Gate (further reduction in submits)
5. Phase 6 → Batching (approach 27 tok/s bandwidth floor)
6. Phase 7 → Tuning (approach 107 tok/s target)

---

## Notes

- All new shaders must be compiled with system glslc (shaderc 2023.8) — newer versions cause 5x regression on RADV
- All shaders target wave64 (local_size_x = 64)
- Correctness validated at each phase via reference-tokens.txt comparison
- Performance measured via --profile flag and wall-clock tok/s
