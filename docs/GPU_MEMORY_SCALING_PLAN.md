# GPU Memory Scaling Plan

Date: 2026-04-06

## Why this exists

ZINC currently leaves performance and usable context on the table because memory policy is mostly static:

- Vulkan uploads all weights into device-local memory and then preallocates a flat F32 KV cache up front.
- Metal wraps GGUF tensors with zero-copy `mmap`, but still sizes decode buffers and KV cache with a fixed `4096`-token runtime cap.
- Fit checks use total reported VRAM or Metal working-set guidance, not a live post-load memory profile.
- `-c/--context` is parsed at the CLI level, but the runtime still contains backend caps and does not yet use a dynamic allocator policy end-to-end.

The goal is to replace this with a backend-aware memory planner that:

- aggressively fills safe GPU memory with useful work
- preserves low-latency decode for single-stream inference
- degrades gracefully when the model is larger than ideal GPU residency
- keeps Metal honest about unified-memory pressure instead of pretending it is discrete VRAM

## What vLLM does today

This analysis is based on current official vLLM docs/source pages:

- engine args: `gpu_memory_utilization`, `kv_cache_memory_bytes`, `swap_space`, `block_size`, `kv_cache_dtype`
- GPU worker memory profiling
- block pool / hybrid KV cache manager
- CPU offload config

Key ideas from vLLM:

1. It reserves only a fraction of GPU memory for itself.
   `gpu_memory_utilization` is a per-instance target, default `0.9`.

2. It profiles non-KV memory before deciding KV capacity.
   The worker computes available KV-cache bytes as requested GPU memory minus profiled non-KV usage and optional CUDA-graph overhead.

3. It uses a fixed-size block pool for KV cache.
   KV memory is managed in token blocks rather than as one contiguous flat cache per request.

4. It keeps eviction/caching metadata explicitly.
   The block pool keeps a free queue in eviction order and a hash map for cached-prefix lookup.

5. It generalizes KV management for hybrid models.
   The hybrid KV manager groups layers and shares physical buffers across KV-cache groups instead of forcing every layer type into a separate full allocation strategy.

6. It can extend effective GPU capacity with CPU memory.
   `cpu_offload_gb` is explicitly described as a virtual GPU-memory extension, with the tradeoff that parameters are fetched over the CPU-GPU interconnect during forward passes.

What is good about vLLM:

- robust capacity planning
- production-ready fragmentation handling
- prefix caching
- hybrid-model KV abstractions
- explicit operator-facing controls

What is not ideal for ZINC’s target:

- it is designed for generality and throughput-first serving, not just lowest-latency single-GPU decode
- block indirection and generic policies can leave backend-specific performance on the table
- it does not assume we can specialize around one Vulkan kernel stack and one Metal stack

ZINC should borrow the control-plane ideas, not copy the whole runtime policy.

## Current ZINC state

### Vulkan

- Weight loading is all-in VRAM: every GGUF tensor gets its own device-local buffer and is uploaded through staging.
- KV cache is preallocated as flat per-layer K/V buffers in F32.
- The current runtime path hard-caps context planning/allocation at `4096`.
- The page table is currently an identity mapping over that flat layout.
- `scheduler/kv_cache.zig` already contains a page-pool allocator, but it is not wired into the inference engine.
- VRAM “budget” currently means total device-local heap size, not live free budget.

### Metal

- Weights are zero-copy `mmap`-wrapped into Metal buffers instead of copied into a separate VRAM heap.
- Working-set guidance comes from `recommendedMaxWorkingSetSize()`, with fallback to total unified memory.
- Decode buffers/KV cache still use a fixed `4096` cap today.
- There are more hard-coded `4096` assumptions in the Metal path than in Vulkan, including at least one fixed `[4096]f32` CPU-side attention buffer.

## Design goals

1. Maximize useful GPU residency.
   After weights and required scratch are accounted for, convert the rest into decode value:
   - more context
   - more concurrent KV residency
   - faster kernels via better cache formats

2. Separate capacity planning from allocation policy.
   We need one shared planner and multiple backend/runtime policies.

3. Keep decode hot-path simple.
   Metadata-heavy schedulers are acceptable in the control plane, but not inside the inner decode loop unless they pay for themselves.

4. Make “fit” multi-dimensional.
   A model can:
   - fully fit with target context
   - fit with reduced context
   - fit with selective offload
   - partially fit with tiered KV

5. Prefer backend-specific strategies.
   Vulkan and Metal should share accounting, not necessarily the same allocator decisions.

## Proposed architecture

### Phase 0: shared planner

Introduce a shared memory-planning layer that computes:

- fixed runtime bytes independent of context
- bytes per additional context token
- device-local vs host-visible / unified contributions
- theoretical max context for a given memory budget

This phase is groundwork only. It should remove duplicated accounting across diagnostics and runtime managers.

Status in this change:

- implemented as `src/gpu/memory_plan.zig`
- used by Vulkan/Metal diagnostics and Metal model-manager accounting

### Phase 1: dynamic context sizing on Vulkan

Replace static `4096` context reservation with:

1. choose a target budget
2. subtract weights
3. subtract fixed runtime scratch
4. convert remaining bytes into context tokens
5. allocate KV/page-table buffers to that size

Initial policy:

- ceiling = min(user requested context, model context length)
- target budget = conservative fraction of available device-local memory
- allocation = contiguous flat KV as today

Why Vulkan first:

- the flash-attention shader already takes dynamic `seq_len`
- the current hard cap is mostly host-side
- the discrete-GPU case is easier to reason about than UMA pressure

### Phase 2: live budget instead of total heap

Current Vulkan fit logic uses total device-local heap size. That is too optimistic if the GPU is dirty and too pessimistic for shared heaps with budget extensions.

Target:

- query live memory budget / usage from the driver
- reserve explicit headroom for descriptors, transient allocations, and driver overhead
- track post-load headroom continuously

This is vLLM’s strongest control-plane idea: profile or measure the non-KV footprint, then size KV from the remainder.

### Phase 3: paged KV in the runtime, not just on paper

ZINC already has:

- a paged attention shader
- a scheduler KV page pool

But the active runtime still uses an identity page table over a flat allocation.

Next step:

- wire request/page allocation into the inference engine
- allocate KV in pages, not one fixed flat segment per request lifetime
- support reclaim and reuse across requests

Unlike vLLM, keep the physical layout specialized for ZINC’s decode kernels:

- tune page size per backend/model family
- avoid unnecessary indirection in the single-request hot path
- make the common case contiguous even if the logical abstraction is paged

### Phase 4: tiered KV residency

When the model or target context outgrows premium GPU residency:

- keep the hot decode window in the fastest residency tier
- move colder KV pages to a slower tier
- prefetch back before reuse

Vulkan tiers:

- device-local VRAM
- host-visible pinned memory / BAR-visible memory where practical
- optional NVMe-backed spill only for survival mode, not performance mode

Metal tiers:

- preferred working-set-resident pages
- colder pages still in unified memory but deprioritized
- optional compression / quantized KV before any spill-like behavior

This is where ZINC should aim to be better than vLLM for single-GPU inference: keep the hot working set aggressively resident and contiguous instead of treating all cached tokens equally.

### Phase 5: KV compression / quantization as a first-class policy

Today:

- Vulkan KV is F32
- Metal has some KV quantization support already

To fully use the GPU regardless of model size, KV bytes per token must become a policy knob:

- F16 KV for safer default discrete-GPU expansion
- Q8 / FP8 KV where quality holds
- selective precision:
  - recent tokens higher precision
  - cold tokens lower precision
  - per-layer or per-head policies if measurements justify it

This is a better long-term lever than only block swapping because it raises effective context capacity without forcing every decode step to touch slower memory.

### Phase 6: selective weight residency / offload

When the model is the limiting factor rather than KV:

- keep the hottest tensors resident
- offload colder or less frequently reused tensors
- for MoE, consider keeping shared experts and router hot while treating sparse experts differently

This must be explicitly model-aware. A blanket vLLM-style virtual-memory extension is not enough for ZINC’s latency target.

Good candidates:

- MoE expert weights with low reuse per token
- giant LM head on constrained GPUs
- rarely used tensors in hybrid architectures

Bad candidates:

- small tensors whose transfer overhead dominates
- tensors touched every token in the hottest path

### Phase 7: backend-specific policy engines

#### Vulkan policy

Favor:

- aggressive device-local fill
- dynamic context based on measured free budget
- paged KV with contiguous fast-path placement
- optional BAR / pinned host fallback only when it wins in benchmarks

#### Metal policy

Favor:

- working-set-aware residency, not “use all unified memory”
- zero-copy weights as baseline
- explicit protection of CPU-side responsiveness and OS pressure
- quantized or tiered KV before trying to occupy the whole machine

Metal should optimize for:

- not tripping memory compression / jetsam-like pressure
- keeping the decoder’s active working set inside Apple’s recommended range
- minimizing page-fault-like surprises from giant shared mappings

## Concrete implementation plan

### Step A

Done in this change:

- add shared `memory_plan.zig`
- dedupe accounting logic
- surface “budget-fit context” in diagnostics

### Step B

Implement Vulkan runtime context planning:

- add a runtime context-reservation policy struct
- thread requested context through model load / engine init
- replace static `4096` allocation with planner result
- keep flat KV layout initially

### Step C

Wire paged KV allocation into live runtime:

- use `scheduler/kv_cache.zig` in inference/server path
- turn page table from identity mapping into real allocation metadata
- preserve single-request contiguous fast path

### Step D

Add live budget sensing and utilization target:

- budget target flag or config
- default conservative headroom
- telemetry for:
  - weights
  - fixed scratch
  - KV reserved
  - active KV
  - budget-fit max context

### Step E

Add compressed / quantized KV policy:

- Vulkan F16 first
- then Q8 / FP8 where quality permits
- measure throughput, latency, and quality deltas

### Step F

Add tiered residency for model-specific hot/cold assets.

## Performance principles for “better than vLLM”

1. Prefer static specialization in kernels, dynamic policy in control plane.
2. Make the common case contiguous.
3. Spend indirection only when it buys reclaimed residency.
4. Keep hot recent tokens premium.
5. Treat MoE expert residency differently from dense layers.
6. On Metal, optimize for working set, not total bytes addressable.

## Risks

- lifting the `4096` cap exposes hidden assumptions in debug paths, tests, and CPU fallbacks
- Vulkan live-budget reporting can vary by driver and extension support
- Metal “it fits in total memory” can still be a bad plan if the working set is too large
- offloading and tiering can easily lose to a smaller but fully resident hot path

## Recommended next implementation slices

1. Vulkan-only dynamic context allocation using the shared planner.
2. Diagnostics/API telemetry for budget-fit context and reserved-vs-active KV.
3. Real paged KV integration using the existing scheduler page-pool.
4. Vulkan F16 KV as the first effective-capacity multiplier.

## External references

- vLLM engine args: https://docs.vllm.ai/configuration/engine_args.html
- vLLM GPU worker memory profiling: https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/
- vLLM block pool: https://docs.vllm.ai/en/stable/api/vllm/v1/core/block_pool/
- vLLM hybrid KV cache manager: https://docs.vllm.ai/en/v0.13.0/design/hybrid_kv_cache_manager/
- vLLM CPU offload config: https://docs.vllm.ai/en/latest/api/vllm/config/offload/
