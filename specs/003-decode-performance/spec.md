# Feature Specification: Decode Performance Optimization

**Feature Branch**: `003-decode-performance`
**Created**: 2026-03-28
**Status**: Draft
**Input**: User description: "Optimize decode throughput from 4 tok/s to 107+ tok/s by moving CPU-side SSM, MoE routing, and shared expert gating to GPU compute shaders, and batching Vulkan command buffers across layers to eliminate per-layer CPU-GPU synchronization overhead."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fast Token Generation (Priority: P1)

A user runs ZINC with a prompt and receives generated text at a speed comparable to existing inference engines. The decode phase produces tokens fast enough that output feels responsive and real-time, rather than the current ~4 tokens per second which creates a noticeable multi-second delay for even short completions.

**Why this priority**: Decode throughput is the single metric that determines whether ZINC is usable as a real inference engine. At 4 tok/s, a 256-token response takes over a minute. At 107+ tok/s, it takes ~2.4 seconds. This is the difference between a demo and a product.

**Independent Test**: Run `zinc --prompt "The capital of France is" --max-tokens 256` on the target hardware (AMD Radeon AI PRO R9700) with Qwen3.5-35B-A3B Q4_K_XL and measure reported decode tok/s. Target: ≥107 tok/s.

**Acceptance Scenarios**:

1. **Given** ZINC is loaded with Qwen3.5-35B-A3B Q4_K_XL on RDNA4 hardware, **When** a user runs a 256-token generation, **Then** decode throughput is ≥107 tok/s as reported by the built-in timing output.
2. **Given** the same model and hardware, **When** a user generates text, **Then** the output is coherent English identical in quality to the current 4 tok/s output (no correctness regression).
3. **Given** the same model and hardware, **When** a user runs a 256-token generation, **Then** GPU memory bandwidth utilization is ≥80% during the decode phase.

---

### User Story 2 - Profiled and Measurable Performance (Priority: P2)

A developer working on ZINC can enable profiling to see exactly where time is spent during each decode token — per-shader GPU time, CPU-side computation time, and synchronization overhead — to identify bottlenecks and validate that optimizations are working.

**Why this priority**: Without measurement, optimization is guesswork. The current 4 tok/s performance has an unclear bottleneck split between CPU SSM, submit overhead, and GPU underutilization. Profiling enables data-driven optimization and prevents wasted effort.

**Independent Test**: Run ZINC with a profiling flag and verify that per-dispatch GPU timestamps and per-phase CPU timings are printed, with totals that account for the full token generation time.

**Acceptance Scenarios**:

1. **Given** ZINC is run with profiling enabled, **When** a token is generated, **Then** per-shader dispatch times (via GPU timestamps) and per-phase CPU times are reported for each layer.
2. **Given** profiling output is available, **When** a developer sums all reported times, **Then** the sum accounts for ≥95% of the wall-clock time per token (no unexplained gaps).
3. **Given** profiling is disabled (default), **When** a user runs inference, **Then** there is zero performance overhead from the profiling infrastructure.

---

### User Story 3 - Correct Output Across All Layer Types (Priority: P1)

After moving SSM computation, MoE routing, and shared expert gating from CPU to GPU, the generated text remains correct and coherent. The GPU implementations produce numerically equivalent results to the current CPU implementations.

**Why this priority**: Correctness is non-negotiable. Moving computation to GPU introduces numerical differences (floating-point ordering, precision). If the output degrades, the performance improvement is worthless.

**Independent Test**: Run the same prompt with the optimized code and compare the first 32 generated tokens against the current (pre-optimization) output. Tokens must match exactly (greedy sampling is deterministic).

**Acceptance Scenarios**:

1. **Given** the GPU SSM implementation replaces CPU SSM, **When** generating with the reference prompt, **Then** the first 32 tokens match the pre-optimization output exactly.
2. **Given** the GPU MoE router replaces CPU top-k softmax, **When** routing decisions are compared, **Then** the same experts are selected with the same weights (within floating-point tolerance of ±1e-5).
3. **Given** the GPU shared expert gate replaces CPU sigmoid, **When** gate weights are compared, **Then** values match within ±1e-6 of the CPU computation.

---

### Edge Cases

- What happens when the descriptor pool (4096 sets) fills during a multi-layer command buffer batch? The system must detect this and submit/reset gracefully rather than crashing.
- How does the system handle models with no SSM layers (pure transformer)? The GPU SSM path must be skipped cleanly.
- How does the system handle models with no MoE (dense FFN)? The GPU router path must be skipped cleanly.
- What happens if the GPU lacks sufficient VRAM for persistent SSM state buffers (conv state + recurrent state for all layers)? The system should report a clear error at init time.
- How does numerical precision in the GPU delta-net state update affect output quality over long sequences (512+ tokens)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST move SSM conv1d + SiLU computation from CPU to a GPU compute shader, maintaining persistent conv state on GPU across tokens.
- **FR-002**: System MUST move SSM delta-net state update (decay, outer product, readout) from CPU to a GPU compute shader, maintaining persistent recurrent state on GPU across tokens.
- **FR-003**: System MUST move SSM gated normalization (RMS norm × SiLU gate) from CPU to a GPU compute shader.
- **FR-004**: System MUST move MoE expert routing (softmax + top-k selection over 256 experts) from CPU to a GPU compute shader, eliminating the per-layer GPU-to-CPU readback of router logits.
- **FR-005**: System MUST move shared expert gate sigmoid from CPU to GPU, eliminating the per-layer GPU-to-CPU readback of the gate scalar.
- **FR-006**: System MUST batch GPU command buffer recording across multiple operations within a layer, reducing the number of GPU-CPU synchronization points from ~151 to ≤10 per decode token.
- **FR-007**: System MUST provide a profiling mode that reports per-dispatch GPU timestamps and per-phase CPU timings without affecting normal operation when disabled.
- **FR-008**: System MUST produce identical output tokens (greedy sampling) before and after optimization for the reference test prompt.
- **FR-009**: System MUST allocate persistent GPU buffers for SSM state (conv state and recurrent state) at initialization, not per-token.
- **FR-010**: System MUST handle hybrid models with mixed attention/SSM layers, routing each layer type to the correct GPU dispatch path.

### Key Entities

- **SSM Conv State**: Per-layer persistent buffer holding the previous d_conv−1 input vectors for 1D convolution. Lives on GPU across tokens. Shape: n_layers × (d_conv−1) × conv_channels.
- **SSM Recurrent State**: Per-layer persistent buffer holding the delta-net state matrices. Lives on GPU across tokens. Shape: n_layers × num_heads × head_v_dim × head_v_dim.
- **Router Output Buffer**: Per-layer GPU buffer holding expert_ids (top-k indices) and expert_weights (softmax-normalized) output by the GPU router shader. Consumed by the expert dispatch loop without CPU readback.
- **Command Buffer Batch**: A single Vulkan command buffer recording that spans multiple dispatches across operations within a layer (or across layers), submitted once rather than per-operation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Decode throughput ≥107 tok/s on the reference hardware and model (Qwen3.5-35B-A3B Q4_K_XL on AMD Radeon AI PRO R9700), matching the llama.cpp baseline.
- **SC-002**: GPU memory bandwidth utilization ≥80% during decode, up from the current 0.4%.
- **SC-003**: Number of GPU-CPU synchronization points per decode token ≤10, down from ~151.
- **SC-004**: Generated text is identical (token-for-token with greedy sampling) to pre-optimization output for the reference prompt over 256 tokens.
- **SC-005**: Profiling mode produces per-dispatch timing data covering ≥95% of wall-clock time per token.
- **SC-006**: A 256-token generation completes in under 3 seconds end-to-end (including prefill), making output feel responsive to users.

## Assumptions

- Target hardware is AMD Radeon AI PRO R9700 (RDNA4, 32 GB VRAM, 576 GB/s bandwidth). Performance targets are specific to this GPU.
- Target model is Qwen3.5-35B-A3B-UD-Q4_K_XL (21 GB, hybrid attention+SSM+MoE, 40 layers, 256 experts top-8, full_attn_interval=4).
- Greedy sampling (argmax) is the only sampling mode that needs correctness validation. Other sampling modes (temperature, top-p) are non-deterministic and don't require token-exact matching.
- The existing Vulkan 1.3 infrastructure (device init, buffer allocation, pipeline creation, SPIR-V compilation) is stable and does not need changes.
- VRAM is sufficient to hold persistent SSM state buffers alongside the model weights and KV cache. Estimated additional VRAM: ~80 MB for conv state + ~320 MB for recurrent state across 40 layers.
- The 107 tok/s llama.cpp baseline was measured with cooperative matrix extensions, flash attention, and mlock enabled. ZINC should match this using equivalent GPU features.
- The memory-bandwidth floor for this model at 576 GB/s is ~27 tok/s. Reaching 107 tok/s implies llama.cpp uses techniques beyond simple memory-bandwidth-bound DMMV (e.g., weight caching in L2, persistent threads, or multi-queue overlap). These techniques may need to be adopted.
