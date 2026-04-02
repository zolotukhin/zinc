# Apple Metal Reference

Last updated: 2026-03-31

This document is the backend-facing Metal reference for ZINC on Apple Silicon. It focuses on the supported public API surface we should optimize against in shipping code, the runtime queries that matter, and the practical kernel and submission rules that follow from Apple GPU behavior.

Related docs:

- [`docs/APPLE_SILICON_REFERENCE.md`](./APPLE_SILICON_REFERENCE.md) covers chip generations, MLX/TensorOps context, public capability families, and opcode-surface boundaries.
- [`docs/APPLE_SILICON_METAL_ENABLEMENT.md`](./APPLE_SILICON_METAL_ENABLEMENT.md) covers the implementation history and file-by-file bring-up details of the current Metal backend.

## Scope

This document is about the public Metal contract that matters for ZINC:

1. the runtime objects we build around
2. the capability and tuning queries we should trust
3. the memory and submission model on Apple Silicon
4. the kernel-level features worth using for inference
5. the profiling and validation loop that keeps optimization work safe

It is not a guide to AGX machine code, ANE internals, or MLX APIs.

## Mental Model

The simplest durable model is:

- **Metal** is the low-level GPU runtime and shader interface.
- **MSL** is the shader language we write kernels in.
- **MPS / TensorOps / Metal 4 tensor APIs** are higher-level public acceleration surfaces that may matter on newer families, especially Apple10 / M5.
- **MLX** is a higher-level array/runtime layer that can target Metal, but it is not the backend contract ZINC is built on.

For ZINC, shipping backend work should target:

- `MTLDevice`
- `MTLCommandQueue`
- `MTLCommandBuffer`
- `MTLComputeCommandEncoder`
- `MTLBuffer`
- `MTLComputePipelineState`
- Metal capability tables and pipeline/device properties

## Runtime Objects That Matter

These are the runtime objects worth reasoning about directly in backend code:

| Metal object | Why ZINC cares |
|---|---|
| `MTLDevice` | device identity, family support, unified-memory behavior, threadgroup limits, working-set hints |
| `MTLCommandQueue` | submission path for decode and prefill work |
| `MTLCommandBuffer` | batching boundary; too many of these can materially hurt throughput |
| `MTLComputeCommandEncoder` | where dependent compute dispatches are recorded and ordered |
| `MTLBuffer` | all important model/runtime data paths are buffer-centric |
| `MTLComputePipelineState` | thread execution width, max threads, compiled kernel validity, specialization surface |

In this repo, these concepts are wrapped through:

- `src/metal/shim.h`
- `src/metal/shim.m`
- `src/metal/device.zig`
- `src/metal/buffer.zig`
- `src/metal/pipeline.zig`
- `src/metal/command.zig`

## Runtime Queries We Should Trust

These are the minimum runtime signals we should use for backend decisions:

- `MTLDevice.supportsFamily(.apple7/.apple8/.apple9/.apple10)`
- `MTLDevice.supportsFamily(.mac2)`
- `MTLDevice.hasUnifiedMemory`
- `MTLDevice.maxThreadgroupMemoryLength`
- `MTLDevice.recommendedMaxWorkingSetSize`
- `MTLDevice.supportsRaytracing`
- `MTLComputePipelineState.threadExecutionWidth`
- `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
- `MTLComputePipelineState.staticThreadgroupMemoryLength`

Practical interpretation:

- **Apple9** means M3/M4-class behavior for public GPU-family gating.
- **Apple10** means M5-class behavior and justifies TensorOps / cooperative-tensor investigation.
- **Unified memory** means we should not cargo-cult discrete-GPU staging patterns.
- **Pipeline properties** are more trustworthy than chip-name guesses for threadgroup sizing.

## Family-Level Guidance

### Apple7 / Apple8

- Treat these as earlier Apple Silicon Metal families with the same broad compute model.
- Prefer straightforward compute kernels, shared buffers, and conservative threadgroup sizing.
- Do not assume newer tensor-specific public acceleration paths exist.

### Apple9

- Treat this as the main M3/M4-class inference target.
- Favor tuned Metal compute kernels, 32-lane simdgroup reductions, and careful `threadgroup` staging.
- Do not try to distinguish M3 from M4 by public Metal family alone.

### Apple10

- Treat this as the first family where TensorOps / Metal 4 tensor resources deserve dedicated attention.
- Large GEMMs, batched expert matmuls, and prefill-heavy paths are the first candidates for a second fast path.
- Decode remains heavily bandwidth-sensitive; TensorOps are not a blanket replacement for every kernel.

## Memory Model on Apple Silicon

The Apple Silicon Metal path should assume:

- unified CPU/GPU memory
- cheap CPU visibility of shared buffers
- buffer-centric compute rather than texture-centric tricks

That leads to different implementation defaults than Vulkan on Linux:

- prefer shared buffers for runtime state and readback-visible data
- avoid inventing separate staging buffers unless they are measurably needed
- use `newBufferWithBytesNoCopy` style wrapping where it materially reduces unnecessary copies
- treat `recommendedMaxWorkingSetSize` as a budget hint for model and scratch residency

Practical consequences for ZINC:

- logits and router buffers can often be read directly without a dedicated readback staging path
- GGUF loading should lean into wrapped or shared memory rather than discrete upload choreography
- KV cache and expert scratch planning should be working-set-aware, not copied from a discrete-GPU design

## Submission Model

For inference throughput, command-submission shape matters a lot on Metal.

The rule of thumb is:

- batch dependent work into as few command buffers as correctness allows
- keep related dependent dispatches inside one compute encoder when possible
- use in-encoder barriers when buffer visibility is the real dependency
- avoid per-expert or per-small-op fragmentation if a fused or batched path is possible

The failure mode to watch for is not just slow kernels. It is:

- too many command buffers
- too many commits
- too many synchronization points between otherwise small compute steps

For ZINC, this matters especially for:

- MoE routing and expert execution
- decode-step pipelines with many small dependent kernels
- server mode, where request scheduling can amplify submission overhead

## Kernel Design Guidance

The public MSL features most worth using for inference are:

- `threadgroup` memory for staging vectors, tiles, and accumulators
- `threadgroup_barrier(...)` for ordered shared-memory phases
- simdgroup collectives such as `simd_sum`, `simd_max`, shuffle, and broadcast operations
- `[[thread_position_in_grid]]`
- `[[thread_position_in_threadgroup]]`
- `[[thread_index_in_simdgroup]]`
- `[[simdgroup_index_in_threadgroup]]`
- `half` and bf16-capable paths when precision and hardware support make them worthwhile
- `simdgroup_matrix` / TensorOps-style paths on the families where the public stack supports them

Useful kernel heuristics:

- start from workgroup sizes that are multiples of `threadExecutionWidth`
- on Apple9/Apple10, one-simdgroup-per-row is a strong first try for decode-style DMMV
- use `maxTotalThreadsPerThreadgroup` as a hard limit, not a target
- account for `staticThreadgroupMemoryLength` before adding more dynamic threadgroup storage
- if threadgroup memory is tight, favor cache-resident or simdgroup-centric designs over large shared-memory tiles

## Features To De-Prioritize

These are public features, but they are not where inference work should go first:

- hardware ray tracing
- mesh shading
- render-pipeline-specific tuning
- texture-centric strategies unless a kernel is actually texture-backed

They can be useful as generation clues, but not as first-order LLM acceleration paths.

## ZINC-Specific Guidance

For this codebase, the Metal backend should keep the following priorities:

1. keep backend selection at comptime
2. keep Objective-C isolated to the shim layer
3. exploit unified memory instead of mimicking Vulkan staging
4. tune submission count as aggressively as kernel math
5. validate fast paths against CPU-reference outputs
6. only branch on Apple10-specific tensor paths when they actually win

The code areas most relevant to this reference are:

- `src/metal/shim.h`
- `src/metal/shim.m`
- `src/metal/device.zig`
- `src/metal/buffer.zig`
- `src/metal/pipeline.zig`
- `src/metal/command.zig`
- `src/model/loader_metal.zig`
- `src/compute/forward_metal.zig`
- `src/shaders/metal/*.metal`
- `src/diagnostics_metal.zig`

## Profiling and Validation

Metal optimization work is only useful if it stays measurable and correct.

Minimum loop:

1. inspect capability logs and pipeline properties
2. profile command-buffer count, commits, and kernel timings
3. compare output against CPU-reference or known-good paths
4. keep changes only if they improve the intended metric without breaking correctness

Useful repo touchpoints:

- `src/diagnostics_metal.zig`
- `benchmarks/metal_inference.zig`
- `tools/benchmark_api.mjs`
- the Metal-oriented tests run under `zig build test`

Note:

- negative pipeline tests may print Metal compiler errors to stderr during `zig build test` even when the suite passes

## Boundaries

Always:

- optimize against public Metal/MSL behavior first
- rely on measured pipeline/device properties rather than chip marketing names
- keep CPU-reference kernel tests close to performance-sensitive changes

Ask first:

- backend model changes that break parity with shared higher-level runtime paths
- new Apple10-only tensor paths that materially complicate maintenance or portability

Do not:

- treat AGX shader ISA as a stable product contract
- assume MLX implies a specific low-level execution path
- assume M3 and M4 are distinct public Metal families

## Sources

- [Metal resources](https://developer.apple.com/metal/resources/)
- [Metal capability tables](https://developer.apple.com/metal/capabilities/)
- [MTLGPUFamily](https://developer.apple.com/documentation/metal/mtlgpufamily?language=objc)
- [Bring your Metal app to Apple silicon Macs](https://developer.apple.com/videos/play/wwdc2020/10631/)
- [Improving your game's graphics performance and settings](https://developer.apple.com/documentation/metal/improving-your-games-graphics-performance-and-settings)
- [storageModeShared](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared)
- [Metal Performance Primitives Programming Guide (2026)](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
