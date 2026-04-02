# Apple Silicon Architecture Reference

Comprehensive reference for Apple Silicon generations from M1 through M5, the public compute surfaces we can actually ship against, and the current state of opcode-level / ISA-level material. This is intended to be the "single source of truth" doc for ZINC work on Apple platforms.

Last updated: 2026-03-31

Related docs:

- [`docs/APPLE_METAL_REFERENCE.md`](./APPLE_METAL_REFERENCE.md) covers the backend-facing Metal runtime contract, tuning surface, and implementation guidance for ZINC on macOS.
- [`docs/APPLE_SILICON_METAL_ENABLEMENT.md`](./APPLE_SILICON_METAL_ENABLEMENT.md) covers the full-detail bring-up and implementation history of the current Metal backend.

## How To Use This Doc

This document intentionally separates four different layers that are often conflated:

1. **Apple M-chip product families**: M1, M2, M3, M4, and M5, plus their Pro/Max/Ultra variants.
2. **Public programming surfaces**: Arm A64 on CPU, Metal/MSL on GPU, MPS/MPSGraph/Metal 4 tensor APIs, and MLX.
3. **Public capability families**: `MTLGPUFamilyApple7` through `Apple10`, plus `Mac2`.
4. **Undocumented or reverse-engineered opcode surfaces**: Apple AMX, AGX shader ISA internals, and ANE internals.

Practical rule for ZINC:

- **Target public APIs for shipping code**: Metal + MSL + MPS / Metal 4 tensor APIs as needed.
- **Use MLX as a reference framework, not as the low-level contract.**
- **Use reverse-engineered opcode material only for research, profiling, and hypothesis generation.**

## Executive Summary

- **M1 maps to Apple7.**
- **M2 maps to Apple8.**
- **M3 and M4 both map to Apple9.**
- **M5 maps to Apple10.**
- **Metal 4 is supported on M1 and later**, but the M5 generation is the first one where Apple publicly ties the stack to **Neural Accelerators in the GPU**, **TensorOps**, and **Metal Performance Primitives** for ML acceleration.
- **M3 and M4 must not be treated as separate Metal families.** They share the same public GPU family even though their product ceilings differ.
- The only broadly public opcode surface is **Arm A64 + NEON / standard Arm architectural extensions**.
- **AMX is real but undocumented.** Good reverse-engineered material exists for M1 through M4; as of 2026-03-31, I did not find a similarly mature public M5 AMX delta reference.
- **AGX shader opcodes are not a supported product contract.** Current reverse-engineered shader ISA material is strongest for the M1-era G13 GPU.

## Source Confidence

When reading this doc, interpret statements in this order of confidence:

- **Official public API / official product specs**: safest basis for shipping code.
- **Official Apple research / WWDC sessions**: strong guidance for intended usage and performance direction.
- **Reverse-engineered documentation**: useful, often excellent, but not a support contract.
- **Inference**: useful for planning, but should be clearly labeled and never treated as a hard contract.

## M1 Through M5 At A Glance

| Family | Public Metal GPU family | Public process story | Public inflection points | Practical meaning for ZINC |
|---|---|---|---|---|
| **M1** | **Apple7** | First Mac-targeted Apple Silicon on 5nm | Unified memory, first Apple GPU family on Mac | Treat as the baseline Apple Silicon compute model |
| **M2** | **Apple8** | Second-generation 5nm | Higher ceilings, same broad compute model | More scale than M1, but still pre-Apple9 feature jump |
| **M3** | **Apple9** | First 3nm Mac family | Dynamic Caching, hardware ray tracing, mesh shading, AV1 decode | First major public graphics-feature inflection on Mac |
| **M4** | **Apple9** | Second-generation 3nm | Higher bandwidth and stronger CPU / NE while staying Apple9 | Same public Metal family as M3, different tuning envelope |
| **M5** | **Apple10** | Third-generation 3nm | Neural Accelerators in GPU, Apple10, TensorOps / Metal 4 ML story | First family that clearly deserves a dedicated tensor-path investigation |

## Family Ceilings

These are the headline public ceilings across the entire family, not the minimum or default SKU.

| Family | Variants publicly shipped | Max CPU cores | Max GPU cores | Max unified memory | Max public memory bandwidth | Notes |
|---|---|---:|---:|---:|---:|---|
| **M1** | M1, M1 Pro, M1 Max, M1 Ultra | 20 | 64 | 128GB | 800GB/s | First full Mac family; Ultra arrives in 2022 |
| **M2** | M2, M2 Pro, M2 Max, M2 Ultra | 24 | 76 | 192GB | 800GB/s | Matured UltraFusion and memory capacity |
| **M3** | M3, M3 Pro, M3 Max, M3 Ultra | 32 | 80 | 512GB | over 800GB/s | First 3nm Mac family and first with Apple9 graphics features |
| **M4** | M4, M4 Pro, M4 Max | 16 | 40 | 128GB | 546GB/s | No public M4 Ultra shipping as of 2026-03-31 |
| **M5** | M5, M5 Pro, M5 Max | 18 | 40 | 128GB | 614GB/s | No public M5 Ultra shipping as of 2026-03-31 |

## Detailed Family Notes

### M1 Family

Metal mapping:

- **Apple7**
- First Apple GPU family on Mac
- Pre-Apple9, so no public hardware ray tracing or mesh shading story

Variant details:

| Variant | Public process / transistor story | CPU | GPU | Neural Engine | Unified memory | Public bandwidth | Notes |
|---|---|---|---|---|---|---|---|
| **M1** | 5nm, **16 billion transistors** | 8-core CPU: 4 performance + 4 efficiency | 7-core or 8-core GPU | 16-core | up to 16GB | not foregrounded in the current Apple support pages I reviewed | First Apple Silicon Mac SoC |
| **M1 Pro** | 5nm, **33.7 billion transistors** | 10-core CPU: 8 performance + 2 efficiency | up to 16-core GPU | 16-core | up to 32GB | **200GB/s** | First Mac "Pro" SoC tier |
| **M1 Max** | 5nm, **57 billion transistors** | 10-core CPU: 8 performance + 2 efficiency | up to 32-core GPU | 16-core | up to 64GB | **400GB/s** | Large memory / media jump over Pro |
| **M1 Ultra** | UltraFusion package, **114 billion transistors** | 20-core CPU: 16 performance + 4 efficiency | 64-core GPU | 32-core | up to 128GB | **800GB/s** | Two M1 Max dies linked over more than 10,000 signals |

What matters for ZINC:

- This is the baseline Apple Silicon Mac model we should keep working on.
- Unified memory and coherent CPU/GPU access are already central here.
- Any AGX opcode material we discuss later is strongest on the M1 generation.

### M2 Family

Metal mapping:

- **Apple8**
- Same broad compute model as Apple7, but with larger ceilings and more memory

Variant details:

| Variant | Public process / transistor story | CPU | GPU | Neural Engine | Unified memory | Public bandwidth | Notes |
|---|---|---|---|---|---|---|---|
| **M2** | Second-generation 5nm, **20 billion transistors** | 8-core CPU: 4 performance + 4 efficiency | 8-core or 10-core GPU | 16-core | up to 24GB | **100GB/s** | Base family transition beyond M1 |
| **M2 Pro** | Second-generation 5nm, **40 billion transistors** | 10-core CPU: 6 performance + 4 efficiency, or 12-core CPU: 8 performance + 4 efficiency | 16-core or 19-core GPU | 16-core | up to 32GB | **200GB/s** | Pro tier scales up M2 cleanly |
| **M2 Max** | **67 billion transistors** | 12-core CPU: 8 performance + 4 efficiency | 30-core or 38-core GPU | 16-core | up to 96GB | **400GB/s** | Double-bandwidth class versus M2 Pro |
| **M2 Ultra** | UltraFusion package, **134 billion transistors** | 24-core CPU: 16 performance + 8 efficiency | 60-core or 76-core GPU | 32-core | up to 192GB | **800GB/s** | UltraFusion interposer exceeds 2.5TB/s inter-die bandwidth |

What matters for ZINC:

- M2 is still fundamentally a "plain Metal compute" target.
- It gives more room for larger models and bigger scratch / KV budgets than M1.
- There is still no public reason to branch into a special tensor-API path here.

### M3 Family

Metal mapping:

- **Apple9**
- First Mac family where Apple publicly ties the GPU to:
  - **Dynamic Caching**
  - **hardware-accelerated ray tracing**
  - **hardware-accelerated mesh shading**

Variant details:

| Variant | Public process / transistor story | CPU | GPU | Neural Engine | Unified memory | Public bandwidth | Notes |
|---|---|---|---|---|---|---|---|
| **M3** | 3nm, **25 billion transistors** | 8-core CPU: 4 performance + 4 efficiency | 8-core or 10-core GPU | 16-core | up to 24GB | **100GB/s** | Base Apple9 chip |
| **M3 Pro** | **37 billion transistors** | 11-core CPU: 5 performance + 6 efficiency, or 12-core CPU: 6 performance + 6 efficiency | 14-core or 18-core GPU | 16-core | up to 36GB | **150GB/s** | Apple9 with mid-tier bandwidth |
| **M3 Max** | Publicly described as the top non-Ultra M3 tier; current Apple support pages show the shipping ceilings below | 14-core CPU: 10 performance + 4 efficiency, or 16-core CPU: 12 performance + 4 efficiency | 30-core or 40-core GPU | 16-core | up to 128GB | **300GB/s** or **400GB/s** | Apple9 high-end memory / GPU tier |
| **M3 Ultra** | UltraFusion package, **184 billion transistors** | 32-core CPU: 24 performance + 8 efficiency | 80-core GPU | 32-core | **96GB to 512GB** | **over 800GB/s** | First Apple Silicon Mac tier publicly pitched for very large on-device LLMs |

What matters for ZINC:

- Apple9 is the main feature boundary for modern Apple GPU behavior.
- Ray tracing and mesh shading are useful generation clues, but not inference acceleration paths.
- Dynamic Caching is public background context, but for compute work we should still optimize from observed pipeline/device properties.

### M4 Family

Metal mapping:

- **Apple9**
- Same public Metal family as M3
- Higher-end CPU, GPU, and memory behavior inside the same public feature family

Variant details:

| Variant | Public process / transistor story | CPU | GPU | Neural Engine | Unified memory | Public bandwidth | Notes |
|---|---|---|---|---|---|---|---|
| **M4** | Second-generation 3nm, **28 billion transistors** | up to 10-core CPU: 4 performance + 6 efficiency | up to 10-core GPU | 16-core | up to 32GB | **120GB/s** | Base family; iMac also shipped lower 8-core CPU / 8-core GPU SKU |
| **M4 Pro** | Public M4 Pro Mac support pages show 12-core and 14-core CPU options | 12-core CPU: 8 performance + 4 efficiency, or 14-core CPU: 10 performance + 4 efficiency | 16-core or 20-core GPU | 16-core | up to 64GB | **273GB/s** | Thunderbolt 5 arrives on Mac here |
| **M4 Max** | Public M4 Max Mac support pages show two GPU / CPU bands | 14-core CPU: 10 performance + 4 efficiency, or 16-core CPU: 12 performance + 4 efficiency | 32-core or 40-core GPU | 16-core | up to 128GB | **410GB/s** or **546GB/s** | No public Ultra part as of 2026-03-31 |

What matters for ZINC:

- This is still Apple9, so do not invent an M4-only Metal family.
- The real changes for inference are more bandwidth and stronger CPU / Neural Engine ceilings, not a new public shader contract.
- Apple publicly quotes **120GB/s** for base M4, which matters when comparing M4 against base M5 generation speedups.

### M5 Family

Metal mapping:

- **Apple10**
- First public Apple Silicon family where Apple explicitly talks about:
  - **Neural Accelerators in the GPU**
  - **TensorOps**
  - **Metal Performance Primitives / Metal 4 ML integration**

Variant details:

| Variant | Public process / transistor story | CPU | GPU | Neural Engine | Unified memory | Public bandwidth | Notes |
|---|---|---|---|---|---|---|---|
| **M5** | Third-generation 3nm | up to 10-core CPU: 4 performance + 6 efficiency | 10-core GPU with **Neural Accelerator in each core** | 16-core | up to 32GB | **153GB/s** | Third-generation ray tracing; base Apple10 tier |
| **M5 Pro** | Apple support pages use new CPU terminology | 15-core CPU: 5 **super** + 10 performance, or 18-core CPU: 6 **super** + 12 performance | 16-core or 20-core GPU | 16-core | up to 64GB | **307GB/s** | Apple also presents this tier as part of a new Fusion Architecture story |
| **M5 Max** | High-end Apple10 tier | 18-core CPU: 6 **super** + 12 performance | 32-core or 40-core GPU | 16-core | up to 128GB | **460GB/s** or **614GB/s** | Neural Accelerators in GPU, no public Ultra part yet |

Important M5 notes:

- Apple changed CPU terminology in current public M5 support pages from the familiar performance / efficiency wording to **super cores** and **performance cores** for the Pro / Max tiers.
- Apple’s 2025 M5 press release says the GPU has a **Neural Accelerator in each core**.
- Apple’s 2025 M5 press release says base M5 has **153GB/s** of bandwidth, about **28 percent** above base M4’s **120GB/s**.
- As of **2026-03-31**, I did not find a public M5 Ultra announcement.

What matters for ZINC:

- This is the first family that clearly warrants a second optimization lane around TensorOps / cooperative tensor / MPP-style paths.
- Apple’s own MLX research says:
  - **TTFT / large matrix multiplies are compute-bound**
  - **subsequent token generation is memory-bandwidth-bound**
- So M5 does not automatically mean every kernel should move to a tensor path. The biggest wins should come from large GEMMs, prefill, and possibly batched expert matmuls.

## Public Metal Capability Progression

### Apple7 / M1

- First Apple GPU family on Mac.
- Metal capability checks should use `supportsFamily(.apple7)` instead of device-name matching.
- Shared-memory resource usage is already the natural default on Apple Silicon.

### Apple8 / M2

- Same broad programming model as Apple7.
- Higher ceilings and better scaling, but still pre-Apple9 for major graphics / geometry / RT feature jumps.

### Apple9 / M3 and M4

- Apple explicitly documents **hardware-accelerated ray tracing** and **mesh shading** on Apple9.
- Apple also publicly markets **Dynamic Caching** with the M3 generation.
- Apple9 is the family boundary that first differentiates "older Apple GPU compute" from the modern RT / mesh-shading-era Apple GPU.

### Apple10 / M5

- `MTLGPUFamilyApple10` is the new family constant in Apple’s Metal docs.
- Apple’s public M5 story adds:
  - GPU **Neural Accelerators**
  - **Metal 4** machine-learning integration
  - **TensorOps** / MPP acceleration
- This is the family where the public supported ML path gets materially more interesting.

### `Mac2` On Apple Silicon Macs

- Apple documents `MTLGPUFamilyMac2` support on Apple Silicon Macs.
- In practice, Apple Silicon Macs expose the union of their Apple-family and Mac-family capability sets.
- For ZINC, the Apple-family check remains the first coarse classifier.

## Metal 4 / MPP / TensorOps / Shader ML

Metal 4 is not a new chip family. It is a new API generation layered on top of the same Metal framework.

Official public highlights from the 2025 Apple materials:

- Metal 4 is supported on **M1 and later**.
- Metal 4 introduces:
  - `MTLTensor`
  - `MTL4MachineLearningCommandEncoder`
  - Shader ML
  - `MTL4ArgumentTable`
  - residency sets
  - placement sparse resources
  - new compiler APIs
  - lower-overhead barrier / synchronization models
- The 2026 Metal Performance Primitives guide describes:
  - tensor resources
  - cooperative tensors
  - tensor operations through MPP / TensorOps
  - simdgroup-scoped and threadgroup-scoped tensor primitives

Practical meaning:

- **Metal 4 is the supported low-level ML path.**
- **It is still an API contract, not a raw ISA contract.**
- On **M5**, this starts to look like a real performance lane for large ML kernels.

## What MLX Is And Is Not

**MLX is not an ISA.** It is an array framework for Apple Silicon with:

- lazy execution
- dynamic graph construction
- unified memory semantics
- CPU and GPU backends
- custom Metal kernel support
- Python, C++, C, and Swift APIs

Important MLX details from current Apple docs and MLX docs:

- MLX arrays live in shared memory.
- Operations can execute on CPU or GPU without explicit host / device copies.
- `mx.compile` can fuse work to reduce kernel-launch overhead.
- `mx.fast` provides tuned implementations for common ML operations.
- MLX exposes **custom Metal kernels**, and Apple’s public MLX docs show `mx.fast.metal_kernel(...)` and generated MSL signatures.

M5-specific MLX path:

- Apple’s **2025-11-19** research note says MLX can use the M5 GPU’s Neural Accelerators through **TensorOps** and **Metal Performance Primitives**.
- Apple says this requires **macOS 26.2 or later**.
- Apple reports that:
  - **TTFT** improves much more than decode speed
  - **subsequent generation** is still mostly bandwidth-bound
  - on the models they tested, subsequent-token speedup over M4 was roughly **19 percent to 27 percent**
  - TTFT speedups were much larger, including up to **4x** in the presented MLX results

For ZINC:

- Treat MLX as a **reference framework and signal source** for what Apple intends to accelerate.
- Do not treat MLX’s internals as the low-level backend contract we should code to.

## Opcode / ISA Reality

This is the section to consult when someone asks for "the opcodes."

### 1. CPU ISA: public and stable

The stable, supported opcode surface is:

- **Arm A64**
- **NEON / Advanced SIMD**
- standard Arm atomics and architectural extensions, where present

This is the only broadly public and official opcode set in the Apple Silicon stack that we should consider a hard product contract.

If we ever write hand-tuned CPU microkernels, this is the official ISA to target.

### 2. Apple AMX: real, useful, undocumented

Apple has never published an official AMX ISA manual. The best current public references are reverse-engineered:

- `corsix/amx`
- Dougall Johnson’s `aarch64_amx.py`

What those sources say:

- AMX is **not** the Apple Neural Engine.
- It is a CPU-side matrix coprocessor / matrix unit.
- The architectural state is described as:
  - `amx0` / `x`: **0x200 bytes**
  - `amx1` / `y`: **0x200 bytes**
  - `amx2` / `z`: **0x1000 bytes**
- Dougall’s notes describe an instruction form:
  - `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
- AMX enable / disable in those notes:
  - enable: `op=17`, `operand=0`
  - disable: `op=17`, `operand=1`

Instruction-family examples called out in the current reverse-engineered material:

- `ldx`
- `ldy`
- `matint`
- `vecfp`
- `vecint`
- `extrh`
- `extrv`

Data types currently described in the AMX reverse-engineered docs include:

- `f16`
- `f32`
- `f64`
- mixed `f16` multiplicands accumulating to `f32`
- integer 8-bit / 16-bit multiplicands accumulating to wider integer results
- on **M2 hardware**, `bf16` multiplicands accumulating to `bf16` or `f32`

Current public reverse-engineered generation deltas:

- **M1**: baseline reverse-engineered AMX behavior
- **M2**: adds `bf16` support and a few other tweaks
- **M3**: adds one extra mode to each of `ldx`, `ldy`, and `matint`
- **M4**: changes low-bit handling for some modes of `extrh`, `extrv`, `vecfp`, and `vecint`
- **M5**: as of **2026-03-31**, I did **not** find a comparably mature public reverse-engineered AMX delta note

Important caveat:

- reverse-engineered AMX docs are good enough for research
- they are **not** an Apple support contract
- ZINC should **not** depend on AMX today for core product functionality

### 3. GPU ISA: public API is Metal / MSL, not AGX machine code

Apple does **not** publish a public shader ISA manual comparable to AMD’s RDNA ISA manuals.

For shipping work, the stable public GPU contract is:

- Metal API
- Metal Shading Language
- Metal capability tables
- MPS / MPSGraph / Metal 4 tensor APIs

Current reverse-engineered GPU ISA material:

- `dougallj/applegpu`
- Asahi Linux AGX docs

Important scope note:

- `applegpu` currently documents the **G13** GPU architecture as used by **M1**
- it is the best public shader-ISA-style material I found, but it is **not** a full M1-to-M5 generation map

Useful reverse-engineered AGX facts from `applegpu`:

- a SIMD-group has **32 threads**
- each SIMD-group has:
  - program counter
  - stack pointer
  - 32-bit execution mask
  - up to **128 general-purpose registers**
- there are **256** 32-bit uniform registers

Sample reverse-engineered instruction mnemonics from `applegpu`:

- `device_load`
- `wait`
- `convert`
- `fmul`
- `rcp`
- `uniform_store`

What this means for ZINC:

- `applegpu` is useful for experimentation and debugging
- it is not a supported optimization target
- do **not** treat M1 reverse-engineered shader details as automatically portable to M2, M3, M4, or M5

### 4. ANE ISA: not public

Apple exposes the Apple Neural Engine through high-level frameworks and ML tooling.

There is **no public, supported opcode guide** for ANE comparable to:

- Arm’s A64 docs
- AMD’s GPU ISA manuals

If someone asks for ANE opcodes, the correct answer is:

- there is no public supported opcode reference
- use public framework surfaces instead

### 5. Metal 4 tensor / MPP path: public, low-level, but still not "raw opcodes"

The new M5-era ML path is public and important, but it is still an API-level contract.

It sits in a useful middle ground:

- lower-level than MLX
- fully supported by Apple
- still **not** raw AGX or ANE opcodes

For future Apple Silicon optimization work, this is likely the right "low-level but supported" lane.

## What To Query At Runtime

Do not hard-code on `"M3"` or `"M4"` strings when runtime capability detection is available.

Minimum runtime checks:

- `MTLDevice.supportsFamily(.apple7/.apple8/.apple9/.apple10)`
- `MTLDevice.supportsFamily(.mac2)` on Apple Silicon Macs
- `MTLDevice.hasUnifiedMemory`
- `MTLDevice.maxThreadgroupMemoryLength`
- `MTLDevice.recommendedMaxWorkingSetSize`
- `MTLDevice.supportsRaytracing`
- `MTLComputePipelineState.threadExecutionWidth`
- `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
- `MTLComputePipelineState.staticThreadgroupMemoryLength`
- CPU topology via `sysctlbyname`:
  - `hw.perflevels`
  - `hw.perflevel0.logicalcpu`
  - `hw.perflevel1.logicalcpu`

Practical interpretation:

- use **GPU family** to gate broad feature classes
- use **pipeline properties** for threadgroup sizing
- use **working-set and memory limits** for residency planning
- use **real measured bandwidth / clocks** to distinguish chips that share a public Metal family

## Immediate Guidance For ZINC

### Safe assumptions

- Apple Silicon Mac GPU code should assume **unified CPU/GPU memory**.
- Feature detection should be based on **`supportsFamily`**, not device name.
- **M3 and M4 are both Apple9**.
- **M5 is the first family that clearly justifies a TensorOps / MPP / Metal 4 ML investigation path**.
- The supported low-level contract is **Metal / MSL / MPP / Tensor APIs**, not AGX shader mnemonics.

### Unsafe assumptions

- Assuming M4 has a distinct public GPU family from M3.
- Assuming MLX implies a specific hardware ISA.
- Assuming AGX shader opcodes are a stable optimization target.
- Assuming AMX behavior is identical across M1, M2, M3, and M4.
- Assuming M5’s Neural Accelerator story automatically speeds up memory-bandwidth-bound decode kernels.

### Recommended optimization order

1. **Public family detection**
   - Apple7 / 8 / 9 / 10

2. **Public kernel contract**
   - Metal / MSL / MPP / TensorOps

3. **Measured hardware envelope**
   - memory bandwidth
   - thread execution width
   - max threadgroup size
   - threadgroup memory limits
   - working-set limits

4. **Only then reverse-engineered internals**
   - AMX
   - AGX shader ISA
   - AGX firmware / queue internals

## Sources

### Apple official: chip launches and product specs

- [Apple unleashes M1](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/)
- [Introducing M1 Pro and M1 Max](https://www.apple.com/ne/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)
- [Apple unveils M1 Ultra](https://www.apple.com/mu/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/)
- [Apple unveils M2](https://www.apple.com/sn/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/)
- [Apple unveils M2 Pro and M2 Max](https://www.apple.com/li/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)
- [Apple introduces M2 Ultra](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/)
- [Apple unveils M3, M3 Pro, and M3 Max](https://www.apple.com/li/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/)
- [Apple reveals M3 Ultra](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- [Apple introduces M4 chip](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)
- [Apple introduces M4 Pro and M4 Max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
- [Apple unleashes M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [Apple introduces MacBook Pro with all-new M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/)
- [MacBook Air (M2, 2022) tech specs](https://support.apple.com/en-us/111867)
- [MacBook Pro (16-inch, 2023) tech specs](https://support.apple.com/en-gb/111838)
- [Mac Studio (2023) tech specs](https://support.apple.com/en-us/111835)
- [MacBook Pro (14-inch, M3, Nov 2023) tech specs](https://support.apple.com/en-lamr/117735)
- [MacBook Pro (14-inch, M3 Pro or M3 Max, Nov 2023) tech specs](https://support.apple.com/en-gb/117736)
- [MacBook Pro (16-inch, Nov 2023) tech specs](https://support.apple.com/en-lamr/117737)
- [MacBook Pro (14-inch, M4, 2024) tech specs](https://support.apple.com/en-lamr/121552)
- [MacBook Pro (14-inch, M4 Pro or M4 Max, 2024) tech specs](https://support.apple.com/en-in/121553)
- [MacBook Pro (16-inch, 2024) tech specs](https://support.apple.com/en-euro/121554)
- [Mac mini (2024) tech specs](https://support.apple.com/en-afri/121555)
- [MacBook Pro (14-inch, M5) tech specs](https://support.apple.com/en-us/125405)
- [MacBook Pro (14-inch, M5 Pro or M5 Max, 2026) tech specs](https://support.apple.com/en-la/126318)
- [MacBook Pro (16-inch, M5 Pro or M5 Max, 2026) tech specs](https://support.apple.com/en-afri/126319)

### Apple official: Metal / ML framework guidance

- [MTLGPUFamily](https://developer.apple.com/documentation/metal/mtlgpufamily?language=objc)
- [Metal capability tables](https://developer.apple.com/metal/capabilities/)
- [Metal Feature Set Tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Improving your game’s graphics performance and settings](https://developer.apple.com/documentation/metal/improving-your-games-graphics-performance-and-settings)
- [Discover Metal 4 (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/205/)
- [Combine Metal 4 machine learning and graphics (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/262)
- [Metal Performance Primitives Programming Guide (2026)](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

### Apple official: MLX and Apple ML research

- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Get started with MLX for Apple silicon (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/315)
- [MLX GitHub repository / README](https://github.com/ml-explore/mlx)
- [MLX custom Metal kernels documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

### Reverse-engineered / research material

- [corsix/amx](https://github.com/corsix/amx)
- [Dougall Johnson’s `aarch64_amx.py`](https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f)
- [applegpu project](https://dougallj.github.io/applegpu/)
- [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html)
- [Asahi Linux AGX documentation](https://asahilinux.org/docs/hw/soc/agx/)

### CPU ISA

- [Armv8-A Instruction Set Architecture guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20Instruction%20Set%20Architecture.pdf)
