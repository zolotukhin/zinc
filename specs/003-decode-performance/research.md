# Research: Decode Performance Optimization

**Feature**: 003-decode-performance
**Date**: 2026-03-28

## R1: Bottleneck Analysis — CPU SSM vs Submit Overhead

**Decision**: CPU-side SSM is likely the dominant bottleneck, not submit overhead alone.

**Rationale**: Code analysis of `src/compute/forward.zig` reveals:
- 151 submits/token: 10 attention layers × 3 + 30 SSM layers × 4 + 1 final
- Each SSM layer (30 per token) does: alloc 128KB conv kernel from mmap, CPU conv1d over 8192 channels × 4 taps + SiLU, CPU delta-net update (32 heads × 128×128 state matrices), CPU gated norm, dealloc temp buffers
- Per-token CPU SSM work: ~45M FLOPs + 30 × (alloc + 128KB mmap read + dealloc)
- Submit overhead estimate: 151 × 0.3ms = ~45ms
- CPU SSM estimate: unknown without profiling, but likely 50-150ms based on FLOPs and memory access

**Alternatives considered**:
- Assumed submit overhead was 100% of bottleneck (earlier "1600 submits" estimate was wrong)
- Pure shader tuning without reducing submits (won't help if GPU is idle 95%+ of time)

**Action**: Profiling (T029j) must be done first to confirm split before committing to implementation order.

## R2: GPU Softmax + Top-K Shader Design

**Decision**: Single-workgroup (64 threads) shader with parallel top-k via bitonic partial sort.

**Rationale**:
- Input: 256 floats (expert logits). Output: 8 expert_ids (u32) + 8 expert_weights (f32).
- 256 elements fits comfortably in shared memory (1 KB).
- Algorithm: (1) load to shared memory, (2) parallel max-scan to find top-8, (3) softmax-normalize only the 8 selected weights.
- Top-k via repeated parallel max: find max, mask it, repeat 8 times. Simple and correct for k=8 out of 256.
- Alternative: full sort (overkill for k=8/256), radix select (complex for small N).

**Alternatives considered**:
- CPU-side top-k with async readback: still requires a submit+wait
- Host-visible router buffer with vkCmdCopyBuffer: same overhead as current approach

## R3: GPU Conv1d + SiLU Shader Design

**Decision**: One workgroup per channel group, persistent state buffer on GPU.

**Rationale**:
- Conv1d is over conv_channels=8192 with d_conv=4 taps. Each channel is independent.
- State: (d_conv-1) × conv_channels = 3 × 8192 = 24576 floats per layer (96 KB).
- Total for 40 layers: 40 × 96 KB = 3.75 MB (trivial VRAM cost).
- Shader reads conv kernel weights from GPU tensor buffer (not mmap), avoiding the per-token alloc+read.
- Fused output: out[ch] = sum * sigmoid(sum) where sum = Σ(kernel[ki] × state_or_input[ki]).
- Must handle f16 conv kernel storage (read as f16, convert to f32 in shader).

**Alternatives considered**:
- Keep CPU conv1d, only move delta-net to GPU: still 30 readbacks for conv output
- Use a separate upload buffer per SSM layer: wasteful, breaks batching

## R4: GPU Delta-Net State Update Shader Design

**Decision**: One workgroup per head, shared memory for state matrix, persistent GPU state buffer.

**Rationale**:
- Per-head: state matrix is 128×128 = 16384 floats (64 KB). Fits in shared memory on RDNA4 (32 KB L1/CU + 128 KB LDS).
- Actually 64 KB exceeds 32 KB LDS per workgroup. Use device-local buffer with careful access patterns instead.
- 32 heads per layer, 30 SSM layers = 960 head updates per token.
- Operations per head: L2 normalize Q and K (128 elems each), compute decay from ssm_a, outer product K×V (128×128), state = decay * state + outer, readout = Q × state.
- Total state: 40 layers × 32 heads × 128 × 128 × 4 bytes = 80 MB. Acceptable for 32 GB VRAM.
- Dispatch: 32 workgroups (one per head), 64 threads each. Each thread handles 2 rows of the 128×128 state.

**Alternatives considered**:
- Shared memory approach: 64 KB state exceeds LDS limits. Would need tiled approach.
- Keep delta-net on CPU, only move conv1d: still 30 readbacks for conv + delta-net inputs

## R5: Shared Expert Gate — Existing Shader Reuse

**Decision**: Reuse existing `sigmoid_scale_acc.comp` shader for shared expert gating.

**Rationale**:
- Current CPU path: readback 1 scalar gate value, compute sigmoid on CPU, use as weight for scale_acc.
- `sigmoid_scale_acc.comp` already implements: `out[i] += sigmoid(gate) * src[i]` where gate is a push constant.
- Problem: gate value is on GPU (in router_logits_buf after DMMV), but sigmoid_scale_acc takes it as push constant (CPU-side).
- Solution: modify sigmoid_scale_acc to read gate from a buffer binding instead of push constant, OR create a thin wrapper that reads the scalar from the buffer. Simpler: just record the sigmoid_scale_acc dispatch with gate=1.0 and add a separate sigmoid multiply via the existing sigmoid_mul shader.
- Actually simplest: the gate scalar (1 float) is already the output of a DMMV. Record a sigmoid_mul dispatch to apply it in-place, then use scale_acc with weight=1.0. Two dispatches, zero readback.

**Alternatives considered**:
- New dedicated shared_expert_gate shader: unnecessary, existing shaders compose to solve this
- Keep CPU sigmoid (only 1 float): still requires submitAndWait for the readback

## R6: Command Buffer Batching Strategy

**Decision**: After GPU-side SSM + router + gate (Steps 1-3), batch remaining submits across layers.

**Rationale**:
- After Steps 1-3: ~41 submits remain (1 per layer end + 1 final). No CPU readback needed.
- With no CPU readback, nothing prevents recording all 40 layers into a single command buffer.
- Constraint: descriptor pool has 4096 sets. Each layer uses ~15 sets. 40 layers × 15 = 600 sets. Well within budget.
- Strategy: record all 40 layers + final norm + LM head in one command buffer. Submit once. Single vkQueueSubmit per decode token.
- The logits readback at the end (for argmax) is the only mandatory readback point.
- GPU argmax shader exists (argmax.comp) — could eliminate even that, but CPU argmax on a vocab_size readback is fine.

**Alternatives considered**:
- Submit every N layers: adds complexity for no benefit if descriptor pool isn't exhausted
- Pre-recorded command buffer with push constants only: ideal but requires all dispatch parameters to be push constants (complex refactor, future work)

## R7: Phased Performance Target

**Decision**: Phased — ≥27 tok/s (milestone 1), ≥107 tok/s (target).

**Rationale**:
- Memory-bandwidth floor: 21 GB model at 576 GB/s = ~27 tok/s. This is achievable purely by eliminating CPU-GPU sync overhead.
- llama.cpp achieves 107 tok/s, implying techniques beyond memory-bound DMMV: L2 weight caching between layers, persistent threads, or overlapped compute+transfer.
- Reaching 107 tok/s may require additional optimization beyond the planned work (shader tuning, multi-queue overlap).
- Declaring 27 tok/s (7x improvement) as "failure" would be counterproductive.

**Alternatives considered**:
- Hard 107 tok/s target: risks blocking feature on unproven techniques
- Relative 25x target (100 tok/s): essentially the same as hard target
