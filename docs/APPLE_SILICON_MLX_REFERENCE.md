# Apple Silicon + MLX Architecture Reference

Comprehensive reference for Apple Silicon generations, Metal/MLX-visible capabilities, and the current state of low-level instruction documentation. This is intended as a durable internal guide for ZINC work on Apple platforms.

Last updated: 2026-03-29

## Scope

This document separates four different things that are often conflated:

1. **Apple M-chip product families**: M1, M2, M3, M4, M5 and their Pro/Max/Ultra variants.
2. **Public programming surfaces**: Arm A64 on CPU, Metal/MSL on GPU, MPS/MPSGraph/Metal 4 tensor APIs, and MLX.
3. **Public capability families**: `MTLGPUFamilyApple7` through `Apple10`, plus `Mac2`.
4. **Undocumented instruction surfaces**: Apple AMX, AGX shader ISA internals, and ANE internals.

The practical rule for ZINC is simple:

- **Target public APIs for shipping code**: Metal + MSL + MLX/MPS as needed.
- **Use reverse-engineered opcode material only for research, profiling, and hypothesis generation.**

## What MLX Is and Is Not

**MLX is not an ISA.** It is an array framework for Apple silicon with lazy execution, unified-memory semantics, CPU/GPU backends, and optional custom Metal kernels.

From the public MLX docs and Apple research material:

- MLX is an open-source array framework for Apple silicon with Python, C++, C, and Swift APIs.
- MLX uses **lazy computation** and **dynamic graph construction**.
- MLX uses **unified memory**, so CPU and GPU operations can act on the same arrays without explicit host/device copies.
- MLX can integrate **custom Metal kernels**.
- On **M5**, MLX can use the new **Neural Accelerators in the GPU** through **Metal 4 TensorOps / Metal Performance Primitives**.

For future design work, treat MLX as a **high-level execution framework over public Apple compute APIs**, not as a low-level hardware contract.

## Generation Map

The most useful stable mapping is the one Apple publishes in the Metal capability tables:

| M-chip generation | Public Metal GPU family | Public Metal version | Key public notes |
|---|---|---|---|
| **M1-series** | **Apple7** | Metal 3 and 4 | First Apple GPU family on Mac |
| **M2-series** | **Apple8** | Metal 3 and 4 | Same broad programming model, higher-end scaling |
| **M3-series** | **Apple9** | Metal 3 and 4 | First Mac generation with hardware ray tracing and mesh shading |
| **M4-series** | **Apple9** | Metal 3 and 4 | Same public GPU family as M3; higher bandwidth / product scaling |
| **M5-series** | **Apple10** | Metal 3 and 4 | Adds the M5-era TensorOps / Neural Accelerator path |

Important consequences:

- **M3 and M4 are both Apple9.** A plain GPU-family check does **not** distinguish them.
- **M5 is Apple10.**
- On Apple Silicon Macs, Apple documents that the Apple GPU also reports **`Mac2`** support because the device exposes the union of Apple-family and Mac-family capabilities.

## Product-Family Ceilings

These are the headline public ceilings surfaced by Apple support pages and Apple newsroom material. They are the useful planning limits, not a complete SKU matrix.

| Family | Variants in market | Max CPU cores | Max GPU cores | Max public memory bandwidth | Notes |
|---|---|---:|---:|---:|---|
| **M1** | M1, M1 Pro, M1 Max, M1 Ultra | 20 | 64 | 800 GB/s | Ultra uses UltraFusion; M1 Pro/Max introduced 200 / 400 GB/s tiers |
| **M2** | M2, M2 Pro, M2 Max, M2 Ultra | 24 | 76 | 800 GB/s | Base M2 is 100 GB/s; Pro/Max/Ultra scale to 200 / 400 / 800 GB/s |
| **M3** | M3, M3 Pro, M3 Max, M3 Ultra | 32 | 80 | 819 GB/s | First Mac family with hardware RT + mesh shading + Dynamic Caching |
| **M4** | M4, M4 Pro, M4 Max | 16 | 40 | 546 GB/s | No public M4 Ultra shipping as of 2026-03-29 |
| **M5** | M5, M5 Pro, M5 Max | 18 | 40 | 614 GB/s | First M family with public M5 GPU Neural Accelerator / TensorOps story |

Notes for ZINC:

- If you care about **decode throughput**, memory bandwidth is often the most useful first-order chip-family discriminator.
- If you care about **TTFT / large GEMMs**, M5 deserves a separate path because Apple now publicly positions it around GPU neural accelerators and TensorOps.

## Public Capability Progression

These are the most relevant public feature deltas for inference and kernel planning.

### Apple7 / M1

- First Apple GPU family on Mac.
- Metal capability checks should use `supportsFamily(.apple7)` and not device-name matching.
- Apple Silicon defaults to **shared-memory resource storage** for buffers, and textures on Apple GPUs.

### Apple8 / M2

- Same broad Metal programming model as Apple7.
- Better family-level scaling, but still pre-Apple9 for major graphics/RT changes.

### Apple9 / M3 and M4

- Apple explicitly calls out **hardware-accelerated ray tracing** on `MTLGPUFamilyApple9`.
- Apple also documents **mesh shading** support on Apple9.
- The M3 newsroom material adds **Dynamic Caching** and says ray tracing + mesh shading come to Mac for the first time.
- Capability tables show larger mesh-shader grid limits and larger texture-view-pool limits than earlier families.

### Apple10 / M5

- Apple capability tables map **M5-series** to **Apple10**.
- Capability tables show a further jump in **maximum mesh shader grid size** over Apple9.
- Apple research and the 2026 Metal Performance Primitives guide describe **Metal 4 tensor resources / cooperative tensors / TensorOps** and a new **GPU Neural Accelerator** path on **M5**.

## What To Query At Runtime

Do not hard-code on `"M3"` or `"M4"` strings when capability detection is available.

The minimum runtime checks we should rely on:

- `MTLDevice.supportsFamily(.apple7/.apple8/.apple9/.apple10)`
- `MTLDevice.supportsFamily(.mac2)` on Apple Silicon Macs
- `MTLComputePipelineState.threadExecutionWidth`
- `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
- CPU topology via `sysctlbyname`:
  - `hw.perflevels`
  - `hw.perflevel0.logicalcpu`
  - `hw.perflevel1.logicalcpu`

Practical interpretation:

- Use **GPU family** to gate broad feature classes.
- Use **actual pipeline properties** for threadgroup sizing.
- Use **CPU sysctls** for P-core / E-core aware scheduling.
- Use **real measured bandwidth / clocks** for generation-specific tuning when M3 and M4 both report Apple9.

## MLX-Relevant Architecture Notes

### MLX baseline model

Public MLX design points:

- NumPy-like core API
- Lazy execution
- Dynamic graph construction
- Unified memory
- CPU and GPU execution targets
- Custom Metal kernels when the stock op set is insufficient

That means MLX is best thought of as:

- a graph/scheduler/runtime layer
- over Apple Silicon unified memory
- lowering onto public CPU and Metal execution paths

### M5-specific MLX path

Apple's late-2025 MLX research note is the clearest public signal of a new architectural split:

- **TTFT / large matrix multiplies** are compute-bound and benefit strongly from M5's GPU Neural Accelerators.
- **Subsequent token generation** remains largely **memory-bandwidth-bound**.
- Apple specifically ties the M5 fast path to **TensorOps** and **Metal Performance Primitives**.
- Apple also notes an OS dependency: **macOS 26.2 or later** is required to take advantage of the M5 Neural Accelerator path.

For ZINC planning:

- M1 through M4: assume Metal compute is the primary public fast path.
- M5: consider a second path built around Metal 4 tensor resources / MPP-style kernels if it materially beats plain compute kernels.

## Loop-Usable Optimization Checklist

This section is specifically for the ZINC implementation loop. These are the runtime signals it should look for and the corresponding optimization directions.

| Runtime signal | Where it comes from | What the loop should do |
|---|---|---|
| `apple10=true` | `MTLDevice.supportsFamily(.apple10)` | Treat the machine as **M5-class**. Investigate Metal 4 TensorOps / cooperative tensor / neural-accelerator paths for large GEMMs, MoE batched expert matmuls, and prefill-heavy kernels. |
| `apple9=true` and `apple10=false` | `MTLDevice.supportsFamily(.apple9)` | Treat the machine as **M3/M4-class**. Favor tuned Metal compute kernels, 32-lane simdgroup reductions, and shared-memory staging. Do not waste time looking for the M5 tensor path. |
| `threadExecutionWidth=32` | `MTLComputePipelineState.threadExecutionWidth` | Structure reductions and tile layouts around **32-lane simdgroups**. Prefer workgroup sizes that are multiples of 32. One-simdgroup-per-row is a good first try for decode DMMV. |
| `max_threads_per_threadgroup` | `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` | Upper bound for threadgroup size. Use this to decide whether to scale one-row kernels to 128/256 threads or keep them narrower. |
| `stgmem` / static threadgroup memory | `MTLComputePipelineState.staticThreadgroupMemoryLength` | If already nonzero, account for it before adding more `threadgroup` storage. Static + dynamic threadgroup memory must fit within the device limit. |
| `tgmem=... KiB` | `MTLDevice.maxThreadgroupMemoryLength` | Hard ceiling for staging activations, partial sums, or dequant tiles in `threadgroup` memory. If the ceiling is tight, bias toward cache-resident / simdgroup strategies instead of large threadgroup tiles. |
| `unified=true` | `MTLDevice.hasUnifiedMemory` | Avoid host/device staging copies. Keep KV cache, expert scratch, and zero-copy GGUF buffers in shared memory whenever practical. |
| `working-set=... GiB` | `MTLDevice.recommendedMaxWorkingSetSize` | Use this as a budget hint for preloading hot tensors, persistent scratch, and batched expert buffers. If close to the limit, avoid aggressive expert predecode caching. |
| `raytracing=true` | `MTLDevice.supportsRaytracing` | Ignore it for inference. It is useful as a generation clue, not as an LLM acceleration path. |

### MSL / Metal Functions Worth Trying

These are the public features most likely to matter for inference speed:

- `threadgroup` memory for staging activation vectors, dequant tiles, and expert accumulators
- `threadgroup_barrier(...)` for ordered shared-memory phases inside a workgroup
- simdgroup collectives such as `simd_sum`, `simd_max`, and broadcast/shuffle operations for reductions and softmax-like steps
- `[[thread_position_in_grid]]`, `[[thread_position_in_threadgroup]]`, `[[thread_index_in_simdgroup]]`, `[[simdgroup_index_in_threadgroup]]` for precise lane mapping
- `half` arithmetic for bandwidth-sensitive kernels when precision is acceptable
- `bfloat` / bf16-capable paths when the hardware and pipeline support make it worthwhile
- `simdgroup_matrix` / TensorOps style paths on the newer families where the public stack supports them

### Features To Ignore For Inference

The loop should explicitly de-prioritize these:

- hardware ray tracing
- mesh shading
- render-only features
- texture-centric optimizations unless a kernel is actually texture-backed

## Opcode / ISA Reality Check

This is the section we should rely on when someone asks for "the opcodes."

### 1. CPU ISA: public and stable

**Public source of truth**: Arm A-profile / Armv8-A A64 ISA documentation.

This is the stable instruction surface for Apple CPU-side kernels:

- scalar A64 integer / control-flow ops
- NEON / Advanced SIMD
- atomics
- crypto and other Arm architectural extensions, where present

If we ever write hand-tuned CPU microkernels, this is the official opcode set to target.

### 2. Apple AMX: real, useful, undocumented

Apple has never published an official AMX ISA manual. The best current references are reverse-engineered:

- `corsix/amx`
- Dougall Johnson's `aarch64_amx.py` notes

What matters for future work:

- AMX is distinct from the Apple Neural Engine.
- It is issued from CPU code but executed on a special matrix unit.
- Reverse-engineered work says:
  - **M2 adds bf16 support**
  - **M3 adds new modes to `ldx`, `ldy`, and `matint`**
  - **M4 changes low-bit handling for some `extrh`, `extrv`, `vecfp`, and `vecint` modes**

Examples of AMX instruction families called out in current reverse-engineered docs:

- load/store style ops: `ldx`, `ldy`
- matrix / vector style ops: `matint`, `vecfp`, `vecint`
- extraction style ops: `extrh`, `extrv`

Recommendation:

- **Do not make AMX a product dependency for ZINC today.**
- Keep it as a research lane for CPU-side matmul experiments only.

### 3. GPU ISA: public API is Metal, not AGX machine code

Apple does **not** publish a public shader-ISA manual comparable to AMD's RDNA ISA manuals.

For shipping work, the stable GPU programming surface is:

- Metal API
- Metal Shading Language
- Metal capability tables
- MPS / MPSGraph / Metal 4 tensor APIs

If you need actual machine-level reverse engineering, current references include:

- `dougallj/applegpu`
- Asahi Linux AGX docs
- `philipturner/metal-benchmarks`

The reverse-engineered `applegpu` tooling disassembles AGX shader instructions with mnemonics such as:

- `device_load`
- `wait`
- `convert`
- `fmul`
- `rcp`

But this is **not** a stable public contract. Treat it like a debugging / research aid.

### 4. ANE ISA: not public

Apple exposes the **Apple Neural Engine** through high-level ML frameworks and Apple ML tooling. There is no public opcode-level ANE ISA reference comparable to Arm or AMD documentation.

What Apple does publish:

- throughput-level marketing and research information
- Core ML / ML framework guidance
- Apple ML research showing how transformer workloads can be adapted to ANE-backed execution paths

If someone asks for ANE opcodes, the correct answer is:

- **there is no public, supported opcode guide**
- use public framework surfaces instead

### 5. Metal 4 tensor / MPP path: public, but not "raw opcodes"

The new M5-era machine-learning path is public, but it is still an API-level contract rather than a raw ISA contract.

Apple's 2026 Metal Performance Primitives guide describes:

- tensor resources
- cooperative tensors
- tensor operations exposed through MPP / TensorOps
- simdgroup-scoped and threadgroup-scoped tensor primitives

This is likely the right long-term "low-level but supported" interface for M5-specific acceleration, not AGX or AMX opcode hacking.

## Recommended Mental Model For ZINC

For future Apple-platform work, we should reason in the following order:

1. **Public family detection**
   - Apple7 / 8 / 9 / 10

2. **Public kernel contract**
   - Metal / MSL / MPS / TensorOps

3. **Measured hardware envelope**
   - memory bandwidth
   - thread execution width
   - max threadgroup size
   - product-specific thermal / clock behavior

4. **Only then reverse-engineered internals**
   - AMX
   - AGX shader ISA
   - AGX firmware / queue internals

This keeps us aligned with what is supportable while still giving us a place to stash harder-core opcode material.

## Immediate Guidance For Future Work

### Safe assumptions

- Apple Silicon Mac GPU code should assume **shared memory** and **low-overhead CPU/GPU data sharing**.
- Feature detection should be based on **`supportsFamily`**, not device name.
- **M3 and M4 must not be treated as separate Metal families**; they are both Apple9.
- **M5 is the first generation that deserves a separate TensorOps / Neural Accelerator investigation path.**

### Unsafe assumptions

- Assuming M4 has a distinct public GPU family from M3.
- Assuming MLX implies a specific hardware ISA.
- Assuming AGX shader opcodes are a stable optimization target.
- Assuming AMX behavior is identical across M1, M2, M3, and M4.

## Sources

### Apple official

- [Metal resources](https://developer.apple.com/metal/resources/)
- [Metal capability tables](https://developer.apple.com/metal/capabilities/)
- [MTLGPUFamily](https://developer.apple.com/documentation/metal/mtlgpufamily?language=objc)
- [Bring your Metal app to Apple silicon Macs](https://developer.apple.com/videos/play/wwdc2020/10631/)
- [Improving your game's graphics performance and settings](https://developer.apple.com/documentation/metal/improving-your-games-graphics-performance-and-settings)
- [storageModeShared](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared)
- [Apple Silicon CPU Optimization Guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide)
- [Apple unleashes M1](https://www.apple.com/gw/newsroom/2020/11/apple-unleashes-m1/)
- [Introducing M1 Pro and M1 Max](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)
- [Apple unveils M1 Ultra](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/)
- [Apple unveils M2](https://www.apple.com/au/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/)
- [Apple unveils M2 Pro and M2 Max](https://www.apple.com/gq/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)
- [Apple unveils M3, M3 Pro, and M3 Max](https://www.apple.com/si/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/)
- [Apple reveals M3 Ultra](https://www.apple.com/us-es/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [Mac Studio (2025) tech specs](https://support.apple.com/en-la/122211)
- [MacBook Pro (14-inch, M4, 2024) tech specs](https://support.apple.com/en-us/121552)
- [MacBook Pro (14-inch, M4 Pro or M4 Max, 2024) tech specs](https://support.apple.com/en-lamr/121553)
- [MacBook Pro (14-inch, M5) tech specs](https://support.apple.com/en-euro/125405)
- [MacBook Pro (16-inch, M5 Pro or M5 Max, 2026) tech specs](https://support.apple.com/es-la/126319)
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Metal Performance Primitives Programming Guide (2026)](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

### MLX

- [MLX GitHub repository / README](https://github.com/ml-explore/mlx)
- [Get started with MLX for Apple silicon (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/315/)

### Reverse-engineered / research material

- [Apple AMX instruction set](https://github.com/corsix/amx)
- [Dougall Johnson's AMX notes](https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f)
- [Apple GPU reverse-engineering project](https://dougallj.github.io/applegpu/)
- [Apple GPU / AGX docs from Asahi Linux](https://asahilinux.org/docs/hw/soc/agx/)
- [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

### CPU ISA

- [Armv8-A Instruction Set Architecture guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20Instruction%20Set%20Architecture.pdf)
