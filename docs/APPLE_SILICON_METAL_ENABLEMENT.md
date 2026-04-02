# Apple Silicon Metal Enablement Notes

Last updated: 2026-03-31

This document is the internal "full-detail" version of how Apple Silicon support was enabled in ZINC. It is intentionally written as source material for a future blog post, but right now the goal is accuracy and completeness rather than polish.

It starts with the simple pieces first and moves toward the harder parts at the end:

1. what "Metal support" means in this repo
2. how the build and backend selection work
3. how the Metal runtime is structured
4. how model loading and execution work
5. how the fast MoE path and server support were added
6. how profiling, testing, and benchmarking currently work
7. what is still missing

Companion references:

- [`docs/APPLE_SILICON_REFERENCE.md`](./APPLE_SILICON_REFERENCE.md) covers Apple GPU family mapping, MLX/TensorOps context, and public capability boundaries.
- [`docs/APPLE_METAL_REFERENCE.md`](./APPLE_METAL_REFERENCE.md) covers the backend-facing Metal runtime contract and tuning guidance.

## Part I. What We Actually Enabled

## 1. Scope

When we say "Apple Silicon support" in ZINC, we do **not** mean "we made Vulkan work on macOS".

We mean:

- on macOS, ZINC compiles against a **Metal backend**
- it can **load GGUF models** without staging all weights through a Vulkan-style upload path
- it can **run local inference** through a Metal-specific forward runtime
- it can **serve the built-in HTTP API and chat UI**
- it can **profile and benchmark** the Metal runtime enough to iterate on it
- it has **kernel correctness tests** for the critical numeric paths

The high-level architecture stays the same:

- `main.zig` still owns CLI and process startup
- tokenizer, GGUF parsing, HTTP routes, scheduler concepts, and model catalog logic remain mostly shared
- backend-specific code is selected at **compile time**

The important difference is that Metal changed the low-level execution model:

- Vulkan-specific device/buffer/pipeline/command code is replaced with Metal equivalents
- macOS uses **shared/unified memory**
- on Metal, shader code lives in `src/shaders/metal/*.metal`
- one small Objective-C shim is used to cross the Zig/Metal boundary

## 2. Design Constraints

These constraints shaped the implementation:

- Keep the project mostly Zig.
- Keep Objective-C isolated to one file.
- Avoid adding non-system runtime dependencies on macOS.
- Lean into Apple Silicon unified memory instead of pretending it is a discrete GPU.
- Preserve the current CLI and HTTP UX as much as possible.
- Keep Linux/Vulkan and macOS/Metal sharing as much upper-level code as is practical.

That leads to a specific implementation style:

- the **backend switch** happens at comptime
- the **Metal shim** is intentionally thin
- the **loader** wraps `mmap` regions directly as `MTLBuffer`s
- the **forward runtime** is Metal-specific, but the **server routes** stay shared

## 3. Implementation Map

The Apple Silicon implementation is spread across a relatively small set of files:

### Build and backend selection

- `build.zig`
- `src/gpu/interface.zig`
- `src/main.zig`

### Metal substrate

- `src/metal/shim.h`
- `src/metal/shim.m`
- `src/metal/device.zig`
- `src/metal/buffer.zig`
- `src/metal/pipeline.zig`
- `src/metal/command.zig`

### Metal model loading and execution

- `src/model/loader_metal.zig`
- `src/compute/forward_metal.zig`
- `src/shaders/metal/*.metal`

### Metal HTTP/runtime integration

- `src/server/runtime.zig`
- `src/server/model_manager_metal.zig`
- `src/server/routes.zig`

### Validation and diagnostics

- `src/diagnostics_metal.zig`
- `benchmarks/metal_inference.zig`
- `tools/benchmark_api.mjs`

## Part II. Build, Platform Selection, and Process Startup

## 4. Compile-Time Backend Selection

Backend selection is handled in [`src/gpu/interface.zig`](../src/gpu/interface.zig).

The rule is simple:

- macOS => `is_metal = true`
- Linux => `is_vulkan = true`

There is no runtime "choose Vulkan vs Metal" switch on macOS. The inactive backend is not compiled into the binary in the same way. This keeps the backend abstraction cheap and avoids dragging Vulkan dependencies into the macOS build.

That file also exposes:

- `backend`
- Vulkan-only helper modules on Linux
- backend identity tests

The practical consequence is:

- the macOS build compiles Metal-specific code paths only
- the Linux build compiles Vulkan-specific code paths only

## 5. Build Integration on macOS

The macOS-specific behavior in [`build.zig`](../build.zig) does four main things:

1. It compiles `src/metal/shim.m`.
2. It adds `src/metal` to the include path.
3. It links the `Metal` framework.
4. It links the `Foundation` framework.

The same pattern is applied to:

- the main executable
- the Metal benchmark executable
- the Zig unit tests

Important build details:

- the Objective-C file is compiled with `-fobjc-arc -fmodules`
- no separate Xcode project is needed
- Metal shader sources are **not** precompiled during the standard macOS build
- instead, the runtime loads `src/shaders/metal/*.metal` source files and compiles them into `MTLComputePipelineState` objects at startup

This is different from the Vulkan path, where Linux typically compiles GLSL `.comp` files into SPIR-V.

### Why runtime MSL compilation was acceptable

This is not the final form one would pick for shipping startup latency, but it made the initial bring-up much simpler:

- no metallib packaging pipeline was required
- the source of truth stays in the repo
- shader edits are easy to iterate on
- diagnostics can verify the raw `.metal` sources directly

The cost is startup time and slightly more moving parts at runtime.

## 6. `main.zig` Startup Flow on Metal

The Metal startup path in [`src/main.zig`](../src/main.zig) does the following:

1. initialize `MetalDevice`
2. log device family and memory capabilities
3. load the GGUF model through `loader_metal`
4. either:
   - tokenize a prompt and run local inference, or
   - initialize the Metal model manager and start the HTTP server

At startup, the Metal path logs:

- Apple GPU family (`apple7` through `apple10`)
- `mac2` support
- unified-memory flag
- raytracing flag
- max threadgroup memory
- recommended max working set size

This log is not just cosmetic. Those values inform tuning decisions later:

- Apple9 means M3/M4-class
- Apple10 means M5-class
- unified memory means we should minimize fake staging-copy patterns
- threadgroup memory limits matter for wide DMMV kernels

## Part III. The Metal Runtime Substrate

## 7. The One-File Objective-C Shim

The bridge to Metal lives in [`src/metal/shim.m`](../src/metal/shim.m). This is intentionally the **only** Objective-C file in the repo.

That file wraps a small C ABI defined in [`src/metal/shim.h`](../src/metal/shim.h).

The shim owns:

- `MTLDevice`
- `MTLCommandQueue`
- `MTLBuffer`
- `MTLComputePipelineState`
- `MTLCommandBuffer`
- `MTLComputeCommandEncoder`

The shim exports just enough functionality for Zig to stay in control:

- device creation/destruction
- capability queries
- shared buffer allocation
- `mmap` wrapping via `newBufferWithBytesNoCopy`
- pipeline creation from MSL source or metallib data
- pipeline capability queries
- command-buffer creation
- dispatch submission
- memory barriers
- synchronous and asynchronous completion

### Why this split matters

This isolates all of the following into one place:

- ARC
- Objective-C method calls
- `id<MTL...>` types
- framework imports
- Apple-specific API surface

Everywhere else in the codebase, the Metal backend looks like ordinary Zig code calling a C-like interface.

## 8. Device and Capability Model

[`src/metal/device.zig`](../src/metal/device.zig) wraps the shim with a Zig-native `MetalDevice`.

Important fields:

- `ctx`
- `chip`
- `caps`

Important capability queries:

- `supports_apple7`
- `supports_apple8`
- `supports_apple9`
- `supports_apple10`
- `supports_mac2`
- `has_unified_memory`
- `supports_raytracing`
- `recommended_max_working_set_size`
- `max_threadgroup_memory_length`

The `GpuFamily` helper currently treats:

- `apple9` as "Apple9-or-newer"
- `apple10` as "M5-class"

That is used for tuning decisions such as:

- whether to use the 1024-thread LM-head specialization
- whether to reserve certain wide kernel shapes for Apple10-class devices

## 9. Shared Buffers and Unified Memory

[`src/metal/buffer.zig`](../src/metal/buffer.zig) is deliberately simple because the hardware model is different from discrete Vulkan devices.

Each `MetalBuffer` stores:

- the shim handle
- size
- a CPU pointer
- whether it wraps external `mmap` memory

The key implementation choice is:

- use `MTLResourceStorageModeShared`

That means CPU and GPU see the same underlying storage. In practice, this changes a lot of design decisions:

- logits do not need a dedicated "readback staging" buffer
- many runtime buffers can be directly inspected from the CPU
- model weights can be wrapped instead of copied
- explicit upload steps are reduced or eliminated

This is one of the main reasons the Metal port did not look like a literal translation of the Vulkan loader/runtime.

## 10. Command Recording and Push Constants

[`src/metal/command.zig`](../src/metal/command.zig) wraps a command buffer + compute encoder pair.

There are two important dispatch entry points:

- `dispatch(...)`
- `dispatchV2(...)`

`dispatchV2(...)` exists because many shaders come from SPIRV-Cross-style layouts where push constants must appear at a specific buffer index.

The shim emulates push constants with:

- `setBytes:length:atIndex:`

That is not identical to Vulkan push constants, but it gives a close-enough call shape for this project:

- data buffers are bound as `MTLBuffer`s
- small parameter structs are injected as inline constant data

The shim also exposes an in-encoder memory barrier:

- `memoryBarrierWithScope:MTLBarrierScopeBuffers`

That matters because the Metal runtime often records multiple dependent dispatches in one command encoder and needs buffer writes to become visible between them without ending the encoder.

## 11. Pipeline Creation Model

[`src/metal/pipeline.zig`](../src/metal/pipeline.zig) creates `MetalPipeline` values that cache:

- the pipeline handle
- `max_threads_per_threadgroup`
- `thread_execution_width`
- `static_threadgroup_memory_length`

These values are not just debug info. The forward runtime uses them to choose workgroup sizes and to decide whether a specialized path is legal on the active device.

This is especially important for Apple Silicon because:

- actual `threadExecutionWidth` is a better tuning input than guessing from marketing names
- threadgroup memory ceilings are real constraints on staged-input kernels

## Part IV. Model Loading on Apple Silicon

## 12. Why the Vulkan Loader Model Was Wrong for Metal

The Vulkan loader path is built around:

- device-local buffers
- staging buffers
- explicit DMA/upload flows

That is not the natural model on Apple Silicon.

The Metal loader in [`src/model/loader_metal.zig`](../src/model/loader_metal.zig) instead:

1. `mmap`s the GGUF file
2. parses GGUF metadata
3. extracts the model config
4. wraps tensor regions directly as Metal buffers using `newBufferWithBytesNoCopy`

This is the single biggest architectural shift in the Apple Silicon bring-up.

## 13. Zero-Copy GGUF Wrapping

For each tensor:

- the tensor's GGUF data offset is computed
- the containing region is page-aligned
- the aligned region is wrapped as a shared `MTLBuffer`
- the tensor descriptor and wrapped buffer are stored together as `LoadedTensor`

This is why `forward_metal.zig` needs `tensorPageOffset(...)`:

- the Metal buffer may begin at the aligned page start
- the actual tensor contents may begin later inside that page

So most shader dispatches receive:

- a wrapped buffer handle
- plus a byte offset inside that wrapped region

### Benefits

- avoids up-front copying of 20+ GB model weights
- keeps model load code simple
- matches unified memory well

### Tradeoffs

- requires page alignment care
- depends on the lifetime of the mapped file
- means some low-level offset logic is more subtle than the Vulkan loader

## 14. Config Extraction and Diagnostics

`loader_metal.zig` also preserves the useful higher-level behavior from the Vulkan loader:

- parse architecture and dimensions from GGUF metadata
- log architecture summary
- support lightweight inspect-only operations

That information is used by:

- inference engine initialization
- diagnostics
- model-manager memory accounting

## Part V. The Metal Inference Engine

## 15. High-Level Shape

The heart of the backend is [`src/compute/forward_metal.zig`](../src/compute/forward_metal.zig).

`InferenceEngine` owns:

- the loaded model
- the Metal device
- all intermediate/shared buffers
- the KV cache
- all compute pipelines
- SSM state and constants
- per-layer cached tensor pointers
- profiling state

Unlike the Linux Vulkan path, the Metal engine leans heavily on:

- shared memory
- direct CPU visibility of GPU buffers
- runtime MSL compilation

## 16. Engine Initialization

`InferenceEngine.init(...)` does a large amount of one-time setup.

It:

1. derives model-dependent dimensions
2. allocates intermediate buffers
3. allocates KV cache buffers
4. loads all Metal shader pipelines
5. preloads norm weights into f32 buffers
6. initializes SSM state/constant buffers
7. caches per-layer tensor pointers
8. resolves token embedding and LM-head tensors

Some notable implementation choices:

- intermediate buffers are sized for the maximum use across multiple phases
- expert scratch is allocated both in batched form and per-expert fallback form
- KV cache is capped to 4096 tokens on the Metal path
- page table is identity-filled up front

## 17. Pipeline Inventory

The Metal runtime currently loads shader families for:

### DMMV / matvec

- `dmmv_q4k`
- `dmmv_q4k_k2048`
- `dmmv_q4k_lmhead`
- `dmmv_q4k_lmhead_1024`
- `dmmv_q5k`
- `dmmv_q6k`
- `dmmv_q8_0`
- `dmmv_f16`
- `dmmv_f32`

### MoE-specific DMMV

- `dmmv_q4k_moe`
- `dmmv_q5k_moe`
- `dmmv_q6k_moe`
- `dmmv_q4k_moe_k2048`
- `dmmv_q4k_moe_k2048_1024`

### Elementwise / routing / accumulation

- `deinterleave`
- `flash_attn`
- `kv_cache_write`
- `rope_fused`
- `sigmoid_mul`
- `swiglu`
- `swiglu_batched`
- `scale_accumulate`
- `rms_norm_mul`
- `moe_accumulate`
- `moe_accumulate_batched`
- `softmax_topk`
- `sigmoid_scale_acc`
- `moe_weighted_acc`

### SSM

- `ssm_conv1d`
- `ssm_delta_net`
- `ssm_gated_norm`

## 18. Runtime State Model

The Metal engine keeps request-scoped state very small:

- current token position
- generated-token list
- profiling counters/timers

Actual persistent model-side state lives in engine-owned buffers:

- hidden/residual/norm/q/k/v/attn outputs
- logits
- router buffers
- expert scratch
- KV cache
- SSM conv state
- SSM recurrent state

The current sampling path on Metal is:

- greedy only
- CPU-side argmax over the shared `logits_buf`

This is important when comparing server behavior:

- Metal supports inference and server mode
- Metal does **not** currently expose the same sampling-control surface as Vulkan

`src/server/runtime.zig` makes that explicit:

- `supports_sampling_controls = gpu.is_vulkan`

## Part VI. Execution Model

## 19. Prefill and Decode

The public API in `forward_metal.zig` is intentionally close to the Vulkan version:

- `prefillBatch(...)`
- `decodeStep(...)`
- `generate(...)`

Current prefill behavior is simple:

- it resets request state
- it loops through prompt tokens
- for each token, it loads the token embedding and runs the decode step

So "prefill" here is still a repeated per-token execution of the forward path, not a special large-batch prefill kernel family.

`generate(...)` reports:

- prefill tokens / time / TPS
- generated tokens / decode time / TPS
- ms/token

## 20. Token Embedding and CPU Visibility

Each decode iteration begins by loading the token embedding into a shared runtime buffer.

The engine then:

- runs the model layers
- writes final logits into a shared buffer
- samples the next token on the CPU

This is viable because:

- the Apple Silicon memory model makes the CPU read path cheap enough for current implementation purposes
- the main bottlenecks ended up elsewhere first

## 21. Decode Step Structure

`runDecodeStep(...)` is the real core of the Metal backend.

At a high level, per token it performs:

1. full-attention or SSM block logic per layer
2. FFN/MoE logic per layer
3. final norm
4. LM head projection

For attention layers it performs:

- RMS norm
- Q/K/V preparation
- RoPE
- KV-cache writes
- flash attention
- attention output projection

For SSM layers it performs:

- SSM conv1d
- delta-net update
- gated norm
- output projection

Then every layer performs either:

- dense FFN, or
- MoE FFN

## 22. Shared-Command Decode vs Fragmented Decode

One of the big Metal performance lessons was that command-buffer fragmentation mattered much more than it first appeared.

The engine now checks whether every MoE layer can stay on the GPU-routed batched path:

- if yes, it can run the token on a **shared command buffer**
- if not, it falls back to more fragmented command recording and more waits

This check is driven by `canUseGpuRoutedBatchedMoe(...)`.

That distinction ended up being performance-critical.

## Part VII. Kernel Strategy

## 23. DMMV Strategy by Quant Type

Not all quant types are treated equally.

The current strategy is:

### Q4_K

This is the most tuned family on Metal.

It has:

- a general path
- a `K <= 2048` specialization
- LM-head specializations
- MoE-specific wide variants

These kernels stage input vectors in threadgroup memory and reuse them across multiple rows. That pattern maps well to Apple GPUs when the matrix shape is right.

### Q5_K and Q6_K

These originally relied on simpler SPIRV-Cross-style kernels where:

- one thread effectively handled one row
- there was much less reuse of the staged input vector

That was enough to get correctness and baseline execution working, but it was not enough to hit the target decode throughput on mixed-quant MoE models.

### Q8_0 / F16

These use smaller, simpler cooperative shapes where 64-thread workgroups handle a small number of rows.

## 24. Pipeline Selection Logic

`dmmvPipelineForType(...)` selects a kernel shape based on:

- quant type
- matrix dimensions
- whether the tensor is the LM head
- whether the device is Apple10-class
- pipeline threadgroup limits

This is where architectural heuristics are encoded, for example:

- reserve the 1024-thread LM-head shape for Apple10-class parts
- use wide Q4_K specializations for larger/hotter decode-side projections

## 25. Why Q4_K Got the First Deep Tuning

Q4_K was the first place where Apple Silicon-specific kernel tuning paid off because:

- it appears on hot decode-side paths
- it benefits visibly from staged-input reuse
- it is a common quantization in the target models

That is why the Q4_K family has more specialized kernels than the other quant families.

## Part VIII. The Hard Part: MoE on Metal

## 26. Baseline MoE Path

The straightforward MoE path looks like this:

1. compute router logits
2. read logits to CPU
3. do top-k softmax on CPU
4. for each selected expert:
   - gate projection
   - up projection
   - SwiGLU
   - down projection
   - accumulation
5. optionally apply the shared expert

This works, but it creates a lot of command-buffer boundaries and synchronization.

On the 35B/3B-active Qwen model, that path was too expensive.

## 27. GPU-Routed Batched MoE Path

The more advanced path is `recordGpuRoutedBatchedMoeOnCmd(...)`.

That path does:

1. `softmax_topk` on GPU
2. store selected expert ids/weights in a compact routing buffer
3. run batched expert gate/up projections
4. run `swiglu_batched`
5. run batched expert down projection
6. fold the weighted residual add into `moe_weighted_acc`
7. optionally add the shared expert

This path exists to keep MoE inside a single recorded GPU sequence as much as possible.

### Why it mattered

Before mixed-quant support was completed, a profiled 256-token decode run showed:

- `21141` command buffers / commits
- roughly `81` submits per decode step
- `submit/wait` dominating total traced time

After the mixed-quant GPU-routed path was completed, the same class of run dropped to:

- `261` command buffers / commits
- one shared token command per step

That was the architectural unlock that moved local decode from about `20 tok/s` to roughly `30 tok/s`.

## 28. Mixed-Quant MoE Was the Real Blocker

The model being optimized did not use a single expert quantization everywhere.

The blocking cases were combinations like:

- `q4_k / q4_k / q5_k`
- `q5_k / q5_k / q6_k`

So having only a Q4_K batched MoE path was not enough. Any unsupported expert tensor type forced a fallback for that layer, which in turn disabled the shared-command fast path.

## 29. Q5_K and Q6_K MoE Kernels

To fix that, the Metal backend gained:

- `dmmv_q5k_moe.metal`
- `dmmv_q6k_moe.metal`

The important implementation detail is that the current versions are not just correctness kernels. They were upgraded into **wide staged-input kernels** so they reuse the expert input vector across multiple rows, similar in spirit to the stronger Q4_K path.

That change matters because the first "correct but simple" q5_k/q6_k MoE kernels still left measurable performance on the table.

## 30. Current MoE Quant Support

`canUseGpuRoutedBatchedMoe(...)` now effectively allows the GPU-routed path when:

- the model uses the expected active-expert count
- gate/up/down expert tensors have supported quantizations
- the required routing and accumulation shaders are present

The supported down/expert quant types currently include:

- `q4_k`
- `q5_k`
- `q6_k`

Once those mixed-quant cases were covered, the shared-command path finally became legal across all MoE layers in the target model.

## Part IX. SSM, Attention, and Other Non-MoE Paths

## 31. SSM Support

The Metal backend also carries dedicated shader paths for the SSM parts of the hybrid architecture:

- `ssm_conv1d`
- `ssm_delta_net`
- `ssm_gated_norm`

The engine allocates persistent SSM state buffers and preloads needed constants into shared f32 buffers. The backend is therefore not "transformer only"; it handles the hybrid Qwen-like architecture in full.

## 32. Attention Support

Attention support on Metal includes:

- RoPE
- deinterleave
- KV-cache writes
- flash attention

This is not yet presented as a separate "Metal graph compiler"; it is a direct dispatch-driven implementation.

## Part X. HTTP Server Support on Metal

## 33. Shared Routes, Backend-Specific Runtime

One of the better architectural decisions in the codebase is that the route layer was kept shared.

[`src/server/runtime.zig`](../src/server/runtime.zig) acts as the backend adapter for the shared HTTP layer by aliasing:

- the loader
- the forward runtime
- the model manager
- the engine type
- the decode API
- the sampling API

This means the higher-level server code does not need to fork into "routes_metal" and "routes_vulkan".

## 34. What the Metal Model Manager Does

[`src/server/model_manager_metal.zig`](../src/server/model_manager_metal.zig) owns the active server-side resources:

- loaded model
- tokenizer
- inference engine
- memory accounting
- managed-model activation/removal

It also estimates memory usage in a Metal-aware way:

- weights bytes
- runtime bytes
- context reservation
- per-token context bytes
- budget based on `recommendedMaxWorkingSetSize`

## 35. Current Server Feature Shape on Metal

The Metal server can currently do the important operational things:

- run the built-in chat UI at `/`
- serve `/v1/chat/completions`
- serve `/v1/completions`
- expose `/health`
- manage the active model through the shared model-manager layer

But it still differs from Vulkan in one meaningful way:

- sampling controls are not yet exposed on Metal in the same way

`src/server/runtime.zig` makes this explicit:

- `supports_sampling_controls = gpu.is_vulkan`

So current Metal server behavior is effectively:

- greedy sampling path
- no Vulkan-style advanced sampling controls

## Part XI. Profiling and Observability

## 36. Metal Profiling Model

The Metal path now has request-scoped runtime profiling through:

- `InitOptions`
- `RuntimeProfile`

It currently tracks:

- decode-step counts
- shared-command-step counts
- command-buffer count
- commit/wait count
- sample calls
- layer mix counts
- embedding time
- command recording time
- router CPU time
- GPU-routed MoE recording time
- fallback MoE recording time
- dense FFN recording time
- final projection recording time
- submit/wait time
- sampling time
- total step time
- debug validation time

This is not the same thing as true per-dispatch GPU timestamps, but it is enough to identify whether we are dominated by:

- recording
- host waits
- routing
- sampling
- specific layer families

## 37. Why This Profiling Was Needed

Without it, a lot of early optimization work would have been guesswork.

The Metal profile is what showed that:

- `submit/wait` dominated traced time
- the runtime was spending too many submits per token
- the remaining large gain was in MoE command fragmentation, not in tokenizer code or output decoding

That directly informed the mixed-quant MoE work.

## 38. Benchmarks We Have Today

There are two useful benchmark paths in-tree:

### Local CLI-style benchmark

- [`benchmarks/metal_inference.zig`](../benchmarks/metal_inference.zig)

### HTTP/API benchmark

- [`tools/benchmark_api.mjs`](../tools/benchmark_api.mjs)

The HTTP benchmark is especially useful because it separates:

- raw `/v1/completions`
- templated `/v1/chat/completions`
- concurrency effects
- streaming TTFT

## 39. Current Local Numbers

On 2026-03-31, local Apple Silicon measurements for `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` were:

### CLI/plain decode

- profiled raw prompt run: about `30.04 tok/s`
- clean raw prompt run: about `30.08 tok/s`

### HTTP benchmark

From `/tmp/zinc_api_local_benchmark_20260331.json`:

- raw `/v1/completions`, `concurrency=1`, `max_tokens=256`: `29.46 tok/s`
- raw `/v1/completions`, `concurrency=4`, `max_tokens=256`: `29.59 tok/s` aggregate
- short chat, `concurrency=1`, `max_tokens=256`: `21.42 tok/s`
- short chat, `concurrency=4`, `max_tokens=256`: `21.44 tok/s` aggregate
- short streaming chat, `max_tokens=64`: `3.13s` average TTFT
- long chat, `~1007` prompt tokens, `max_tokens=64`: about `1.72 tok/s` completion-side throughput

This split is important:

- raw decode is now close to the kernel ceiling we wanted
- templated chat still pays substantial prompt and server-path overhead

## Part XII. Testing Strategy

## 40. What Is Already Covered

The Metal backend has unit coverage for:

### Substrate

- device init / capability queries
- buffer allocation and `mmap` wrapping
- pipeline compilation
- command recording and dispatch

### Runtime/kernels

- SSM kernels
- RMS norm
- deinterleave
- softmax top-k
- KV-cache write
- flash attention
- weighted MoE accumulation
- DMMV correctness for multiple quant paths

### Shader-compile sanity

- the batched MoE shader set is compile-tested directly

## 41. How Kernel Tests Work

The most valuable Metal tests use this pattern:

1. synthesize a small packed quantized tensor
2. synthesize an input vector
3. run the Metal shader
4. dequantize the same rows on CPU with `dequantRow(...)`
5. compute the CPU dot-product reference
6. compare GPU and CPU output with a small tolerance

This is exactly the kind of testing that catches optimization regressions without requiring brittle full-text generation snapshots.

## 42. Recent Test Additions

Recent Apple Silicon work added/fixed tests for:

- plain `dmmv_q6k` with nonzero `a_offset`
- `q4_k` batched MoE numerical parity
- `q5_k` batched MoE numerical parity
- `q6_k` batched MoE numerical parity
- Metal-side `q6_k` CPU-reference dequant support so those tests are actually meaningful

## 43. Why Test Noise Shows Up in `zig build test`

The test suite intentionally includes negative pipeline tests, for example:

- invalid MSL should fail
- wrong function name should fail

Those can print Metal compiler errors to stderr during `zig build test`, even when the test suite as a whole passes. That stderr noise is expected and does not necessarily indicate a failing build.

## Part XIII. Practical Commands

## 44. Build, Test, Run, Benchmark

Useful Apple Silicon commands:

```bash
zig build -Doptimize=ReleaseFast
zig build test
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
./zig-out/bin/zinc -m /path/to/model.gguf --chat --prompt "Explain why the sky appears blue"
./zig-out/bin/zinc -m /path/to/model.gguf --port 9090
```

HTTP benchmark:

```bash
bun tools/benchmark_api.mjs \
  --base http://127.0.0.1:9090/v1 \
  --mode both \
  --output /tmp/zinc_api_local_benchmark.json
```

Metal benchmark binary:

```bash
zig build bench-metal -- \
  -m /path/to/model.gguf \
  --prompt "The capital of France is"
```

## Part XIV. Current Limitations

## 45. Things That Are Still Rough

The Apple Silicon path is real and usable now, but it is not "finished".

Important current limitations:

- Metal shaders are compiled from source at runtime instead of shipping as precompiled metallibs.
- The Metal server does not currently expose the full Vulkan sampling-control feature set.
- Graph export / analysis paths are still Vulkan-centric.
- Prefill is still implemented as repeated decode-step execution rather than a separately optimized large-batch path.
- Generation is still serialized in the current server implementation.
- Sampling is still CPU greedy argmax on shared logits.
- Unified-memory simplicity is great for bring-up, but some buffers may eventually want more careful residency strategy if we chase higher absolute throughput.

## 46. Where The Next Performance Work Likely Is

Now that raw decode is around `30 tok/s` locally, the next likely gains are not from the original "enable Metal at all" work. They are from second-order optimization work:

- close the chat-path gap versus raw decode
- reduce TTFT on streaming chat
- consider more tuned non-Q4 quant kernels outside MoE
- improve prefill
- revisit LM-head and other wide projections
- decide whether Apple10/M5-class parts deserve a separate TensorOps-oriented path

## Part XV. Summary

## 47. The Core Architectural Decisions

If this entire document were collapsed into the handful of decisions that made Apple Silicon support possible, they would be:

1. Choose the backend at **compile time**, not through a large runtime abstraction layer.
2. Isolate Metal/Objective-C in **one shim file**.
3. Treat Apple Silicon as a **unified-memory system**, not as a fake discrete-GPU Vulkan clone.
4. Use **zero-copy GGUF `mmap` wrapping** for model weights.
5. Keep the **HTTP/routes layer shared** and swap only the backend runtime aliases.
6. Add enough **profiling** to see command-buffer fragmentation clearly.
7. Optimize the real bottleneck, which turned out to be **mixed-quant MoE command fragmentation**.
8. Back the fast paths with **CPU-reference kernel tests** so optimization work stays safe.

That is the implementation story in one page.

The rest of this document exists so future-you does not have to rediscover all of the details from the codebase again.
