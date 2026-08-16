# Intel Arc B-Series GPU Reference for Inference

Hardware specifications, memory-bandwidth analysis, compiler-visible opcode surface, and ZINC tuning notes for Intel Arc B-series GPUs. This page is the Intel counterpart to the AMD RDNA reference and focuses on Battlemage / Xe2 discrete cards.

Scope note: this reference was checked on 2026-05-17 and updated after Intel Arc entered ZINC's official support matrix. It covers the currently public B-series desktop and workstation line: Arc B580, Arc B570, Arc Pro B70, Arc Pro B65, Arc Pro B60, and Arc Pro B50. ZINC's Intel Vulkan path is supported for the managed catalog models validated in the public benchmark suite, but the tuning depth is still younger than the RDNA4 path. Use this page as an engineering reference for Arc-specific performance work, not a claim that every RDNA optimization has an Intel equivalent yet.

## Reading The Tables

Intel product pages publish the card-level facts: Xe cores, XMX engines, clocks, VRAM, bus width, memory bandwidth, board power, PCIe link, Vulkan version, and PCI device ID. Some derived values below are marked with `*`:

- `Vector engines` = `Xe cores * 8` for Xe2-HPG B-series products.
- Desktop B580/B570 FP32 TFLOPS are derived as `vector_engines * graphics_clock_GHz * 32 FP32 ops/clock`. Intel publishes FP32 values directly for the Arc Pro cards.
- B70/B65 memory speed is derived from Intel's published `608 GB/s` over a `256 bit` bus: `608 * 8 / 256 = 19 Gbps`.
- Bandwidth-per-watt and bandwidth-per-core ratios are planning numbers. They do not include compression, cache hit rate, driver overhead, or quantization unpack cost.

## Current B-Series Line

### Product Specifications

| SKU | Segment | Launch | Xe cores | Render slices | Vector engines | XMX engines | Graphics clock | FP32 TFLOPS | INT8 TOPS | TBP | PCIe | Vulkan | Device ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Arc Pro B70 | Pro AI / workstation | Q1'26 | 32 | 8 | 256 | 256 | 2280 MHz, 2800 MHz max dynamic | 22.94 | 367 | 230 W | PCIe 5.0 x16 | 1.3 | 0xE223 |
| Arc Pro B65 | Pro AI / workstation | Q1'26 | 20 | 5 | 160 | 160 | 2400 MHz | 12.28 | 197 | 200 W | PCIe 5.0 x16 | 1.3 | 0xE222 |
| Arc Pro B60 | Pro AI / workstation | Q2'25 | 20 | 5 | 160 | 160 | 2400 MHz, 2000 MHz LP | 12.28 | 197 | 200 W, 120-200 W LP | PCIe 5.0 x8 | 1.3 | 0xE211 |
| Arc Pro B50 | Pro SFF workstation | Q3'25 | 16 | 4 | 128 | 128 | 1700 MHz, 2600 MHz max dynamic | 10.65 | 170 | 70 W | PCIe 5.0 x8 | 1.4 | 0xE212 |
| Arc B580 | Desktop gaming / creator | Q4'24 | 20 | 5 | 160 | 160 | 2670 MHz | 13.67* | 233 | 190 W | PCIe 4.0 x8 | 1.3 | 0xE20B |
| Arc B570 | Desktop gaming / creator | Q4'24 | 18 | 5 | 144 | 144 | 2500 MHz | 11.52* | 203 | 150 W | PCIe 4.0 x8 | 1.3 | 0xE20C |

### Memory System

| SKU | VRAM | Bus | Memory speed | Bandwidth | GB/Xe core | GB/s/Xe core | GB/s/W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arc Pro B70 | 32 GB GDDR6, ECC | 256 bit | 19 Gbps* | 608 GB/s | 1.00 | 19.0 | 2.64 |
| Arc Pro B65 | 32 GB GDDR6 | 256 bit | 19 Gbps* | 608 GB/s | 1.60 | 30.4 | 3.04 |
| Arc Pro B60 | 24 GB GDDR6 | 192 bit | 19 Gbps | 456 GB/s | 1.20 | 22.8 | 2.28 |
| Arc Pro B50 | 16 GB GDDR6, ECC | 128 bit | 14 Gbps | 224 GB/s | 1.00 | 14.0 | 3.20 |
| Arc B580 | 12 GB GDDR6 | 192 bit | 19 Gbps | 456 GB/s | 0.60 | 22.8 | 2.40 |
| Arc B570 | 10 GB GDDR6 | 160 bit | 19 Gbps | 380 GB/s | 0.56 | 21.1 | 2.53 |

The important inference split is not "gaming card versus pro card"; it is memory capacity and sustained bandwidth:

- B70 and B65 are the only public B-series cards with both 32 GB VRAM and 608 GB/s. They are the natural targets for 27B dense and 35B MoE GGUFs.
- B60 has the same high-level compute shape as B580 but doubles the memory capacity to 24 GB. It is useful when the model fits only on the pro card, but it does not increase bandwidth over B580.
- B50 is a low-power 16 GB card. Its 224 GB/s bandwidth is the limiting factor for single-stream decode, not the 16 GB capacity.
- B580 is the strongest consumer B-series small-model card: 12 GB, 456 GB/s, high clock, and 233 INT8 TOPS.
- B570 is capacity-limited first. The 10 GB VRAM pool is tight for 8B-class models once KV cache and temporary buffers are included.

## ZINC Support Status

Intel Arc is a supported Linux Vulkan target in ZINC. The current public Intel benchmark artifact validates four managed catalog rows:

- Qwen 3.5 9B Q4_K_M
- Qwen3.6 35B-A3B UD Q4_K_XL
- Gemma 4 31B Q4_K_M
- Gemma 4 26B-A4B MoE Q4_K_M

The benchmark target is an Intel Arc BMG G31-class Vulkan node. The headline rows are ahead of the same-machine llama.cpp baseline on prefill and decode, but this should still be read as "officially supported and measurable" rather than "fully optimized." The next Arc work is deeper prefill/attention tuning, cleaner server latency, and more coverage across consumer B-series SKUs.

## LLM Inference Analysis

### Decode Roofline

Single-token LLM decode is usually memory-bandwidth bound. The rough upper bound is:

```text
decode_tokens_per_second <= sustained_bandwidth_bytes_per_second / active_weight_bytes_per_token
```

That makes B65 unusual: it has the same 608 GB/s memory bandwidth as B70 but only 20 Xe cores. For single-stream decode on quantized weights, B65 can be much closer to B70 than its FP32 or INT8 TOPS suggest, assuming the kernels keep memory coalesced and maintain enough resident hardware threads.

The ranking for raw decode bandwidth is:

```text
B70 = B65 > B580 = B60 > B570 > B50
608      608     456    456     380    224 GB/s
```

For ZINC's current DMMV-heavy decode path, memory bandwidth matters more than peak matrix TOPS until the model or kernel becomes arithmetic-heavy. Q4_K/Q5_K/Q6_K unpacking can move part of the cost back to ALU, but the large weight stream is still the dominant term.

### Prefill And Batched Work

Prompt prefill is different. Prefill exposes matrix-matrix work and larger attention tiles, so XMX/DPAS can matter once the Vulkan driver exposes cooperative matrix properties that match the shader's data types and tile shapes.

Expected prefill ordering:

| SKU | Prefill expectation | Why |
| --- | --- | --- |
| B70 | Best B-series target | 32 Xe cores, 256 XMX engines, 608 GB/s |
| B580 | Strong for 8B-class prompts | 20 Xe cores, 160 XMX, high clock, 456 GB/s |
| B65 | Good but compute-capped versus B70 | Same 608 GB/s as B70, but 20 Xe cores |
| B60 | Similar compute to B580, more VRAM | 20 Xe cores, 456 GB/s, 24 GB capacity |
| B570 | Lower memory and compute than B580 | 18 Xe cores, 380 GB/s |
| B50 | Capacity useful, bandwidth low | 16 GB, but only 224 GB/s |

For ZINC, the right sequence is:

1. Get DMMV and attention coherent with subgroup-specialized scalar/vector kernels.
2. Benchmark B-series decode against the same GGUFs on the same node.
3. Only then wire XMX/DPAS into batched prefill through `VK_KHR_cooperative_matrix` if the driver reports usable matrix properties.

### Model Fit Guide

Use `./zig-out/bin/zinc --check --model-id <id>` for exact fit. The table below is a planning guide for Q4_K-ish GGUFs plus ZINC temporary buffers:

| VRAM class | B-series cards | Practical target |
| --- | --- | --- |
| 10 GB | B570 | 7B/8B only, short to moderate context |
| 12 GB | B580 | 8B comfortably, 12B only if buffers and context are small |
| 16 GB | B50 | 8B with more context; 12B possible depending on architecture |
| 24 GB | B60 | 20B class and some 27B/35B MoE experiments with tight KV budgeting |
| 32 GB | B65, B70 | 27B dense and 35B MoE targets; best fit for ZINC's larger catalog models |

KV cache can dominate long-context serving. A backend-independent estimate is:

```text
kv_bytes = layers * kv_heads * head_dim * 2 * bytes_per_scalar * context_tokens
```

The `2` is for K and V. GQA/MLA/MoE details change `kv_heads` and `head_dim`; KV quantization changes `bytes_per_scalar`. For B570/B580, the KV budget usually decides the maximum useful context before raw compute does.

## Compute Unit Architecture

### Xe2 Xe-core Layout (B70)

![Xe2 Xe-core Layout (B70 Battlemage)](/xe-core-b70.svg)

The Xe core is the Battlemage scaling unit and the Intel counterpart to the AMD RDNA Compute Unit. On B70, 32 Xe cores are grouped into 8 render slices of 4 Xe cores each.

Each Xe2 Xe core contains:

- **8 vector engines** (XVE) — SIMD execution units, the Intel analogue of an RDNA SIMD
- **8 XMX engines** — systolic matrix arrays, paired one-to-one with the XVEs (DPAS dispatch)
- **256 KB L1 cache** — shared across all vector engines in the Xe core
- **128 KB SLM** — workgroup-managed scratchpad, shared by the workgroup currently scheduled
- **1 load/store unit** — message-routed (`send` family); coalescing and cache routing happen here
- **Instruction cache** — shared across the Xe core's 8 XVEs

Each vector engine has:

- **8 hardware threads** (regular GRF mode) or **4 hardware threads** (large GRF mode) — latency-hiding scheduling slots, the closest analogue of an RDNA wave slot
- **128 GRF registers per thread** (regular) or **256 GRFs** (large) — one GRF is 64 bytes (512 bits)
- **FPU pipe + Extended-Math pipe** — independent issue ports; FPU runs FMA32 / FMA16 / integer math, EM runs transcendentals (`exp`, `log`, `rcp`, `rsqrt`, sin / cos)
- **XMX coprocessor** — fed through the XVE's instruction stream but issues DPAS to the paired matrix array
- **Native SIMD16 and SIMD32 execution** at the same compute width — the compiler picks the SIMD size; the lane count is not a hardware-defined wave64

### RDNA CU ↔ Xe core Mapping

For a reader already familiar with the [RDNA reference](/zinc/docs/amd-gpu-reference/), this table maps the core concepts:

| RDNA term | Intel Xe2 term | Notes |
| --- | --- | --- |
| Shader Engine | Render Slice | Coarse scheduling group; Navi 48 has 2 SEs, B70 has 8 render slices |
| Compute Unit (CU) | Xe core | 1 CU ↔ 1 Xe core for shared L1/scratchpad scope |
| WGP (2 CUs) | — | No equivalent on Battlemage; the Xe core is the indivisible unit |
| SIMD (2 per CU) | XVE (8 per Xe core) | Intel has 4× more issue ports per core, each issuing a narrower vector |
| Wave32 / Wave64 | SIMD16 / SIMD32 | Wave size is fixed in hardware on RDNA; subgroup size is compiler-chosen on Intel |
| 16 wave slots per SIMD | 8 hw threads per XVE | RDNA's deeper slot count compensates for fewer issue lanes |
| Matrix Core (WMMA) | XMX engine (DPAS) | Tile shapes differ — RDNA4 is 16×16×16; Intel tiles are driver-reported, commonly 8×16×16 |
| VGPR file (192 KB per SIMD) | GRF (≈ 512 KB per Xe core) | Intel GRF is shared by all threads on a vector engine; per-thread 8 or 16 KB |
| SGPR (32 KB per CU) | — | Intel has no separate scalar register file; uniform values live in the GRF |
| LDS (64 KB per CU / 128 KB per WGP) | SLM (128 KB per Xe core) | Same scope and barrier model; banking differs |
| L0 vector cache (32 KB per CU) | L1 (256 KB per Xe core) | Different naming; same role |
| L2 (8 MB per SE on Navi 48) | L2 (GPU-wide, driver-reported) | Intel does not publish per-card L2 sizes as consistently |
| Infinity Cache (64 MB on Navi 48) | — | No direct analogue on B70 |

Two practical takeaways from the mapping:

1. **An Intel subgroup is a compiler-vectorized SIMD thread**, commonly 16 or 32 work-items wide. A 64-thread workgroup on Intel is two or four subgroups, not one wave64. Ported shaders should treat subgroup size as a specialization input, not a compile-time constant inherited from RDNA.
2. **The Xe core has 4× more issue ports than an RDNA CU (8 vs 2) but fewer thread slots per port (8 vs 16).** Total parallelism is similar; the latency-hiding strategy differs. Kernels that hit register pressure on RDNA may hit thread-count pressure on Intel and vice versa.

### B70 Render Slice and Card Totals

At the 2280 MHz graphics clock B70 ships at:

| Property | B70 value | Inference consequence |
| --- | ---: | --- |
| Render slices | 8 | Coarse scheduling group |
| Xe cores | 32 | 4 per render slice |
| Vector engines | 256 | 32 Xe cores × 8 XVE per core |
| XMX engines | 256 | 32 Xe cores × 8 XMX per core, one per XVE |
| Hardware threads | 2048 | 256 XVE × 8 threads — drives latency hiding |
| Supported subgroup sizes | 16, 32 | Do not assume RDNA-style wave64 |
| GRF per thread | 128 / 256 (regular / large) | Register pressure drives spills and occupancy |
| Register width | 512 bits | One GRF is 64 bytes |
| L1 per Xe core | 256 KB | Per-Xe-core, shared by vector engines |
| L1 total | 8 MB | 32 × 256 KB across the GPU |
| SLM per Xe core | 128 KB | Workgroup-scoped scratchpad |
| SLM total | 4 MB | 32 × 128 KB across the GPU |
| Max SLM per workgroup | 128 KB | Cap reduces residency to one workgroup per Xe core |
| Max workgroup size | 1024 | API ceiling, not the optimal local size |
| FP32 vector throughput | 22.94 TFLOPS | XVE FPU pipe path, no XMX |
| FP16 / BF16 XMX throughput | ~184 TFLOPS | Half of INT8, derives from DPAS rate |
| INT8 XMX throughput | 367 TOPS | DPAS engine peak |
| VRAM | 32 GB GDDR6, 256-bit, 608 GB/s | Decode-bound workloads scale with bandwidth |
| PCIe | 5.0 x16, ~64 GB/s/direction | Resizable BAR required for stable benchmarks |

### Subgroup Execution Model

A subgroup is the fundamental Intel execution unit. Lanes within a subgroup execute the same instruction simultaneously, predicated by an execution mask.

| Property | SIMD16 | SIMD32 |
|---|---|---|
| Work-items per subgroup | 16 | 32 |
| Lane width per scalar | 32 bits | 32 bits |
| Register footprint per SIMD value | 1 GRF (64 B) | 2 GRFs (128 B) |
| Threads/XVE at regular GRF | up to 8 | up to 8 (compiler dependent) |
| Best for | Small kernels, high occupancy | DMMV, reductions, XMX/DPAS |
| RDNA analogue | half of wave32 | wave32, single-cycle issue |

**For ZINC's DMMV and dequant on B70:** SIMD32 is the default starting point. Memory bandwidth saturates more easily with wider subgroups, and SIMD32 lines up with most public Xe2 cooperative-matrix tile shapes.

**For XMX/DPAS:** subgroup size is dictated by the cooperative-matrix tile reported by the driver. Query at runtime; do not assume a fixed value.

**Divergence:** branching within a subgroup is predicated. Divergent control flow does not change the instruction stream but does increase dynamic work — minimize branches in the hot path.

### Register File (GRF)

Each vector engine on Xe2-HPG has a banked GRF shared across all 8 hardware threads:

- Per thread (regular mode): 128 GRFs × 64 bytes = **8 KB**
- Per thread (large GRF mode): 256 GRFs × 64 bytes = **16 KB**, with hardware threads per XVE halved to 4
- Per Xe core (regular mode): 8 XVE × 8 threads × 8 KB ≈ **512 KB** of GRF
- Per Xe core (large GRF): 8 XVE × 4 threads × 16 KB ≈ **512 KB**, with half the latency-hiding headroom

| GRF mode | Threads/XVE | GRFs/thread | Use case |
|---|---:|---:|---|
| Regular | 8 | 128 | DMMV, dequant, decode-path reductions |
| Large | 4 | 256 | Cooperative-matrix tile loops, fused attention with big tiles |

GRF pressure is the primary Intel occupancy lever:

- Exceeding the per-thread GRF budget spills to scratch memory (private memory in VRAM). Spills are slow.
- IGC reports spill counts ahead of time. `unitrace` reports post-JIT private-memory bytes for live shaders.
- Large GRF helps register-heavy kernels but worsens latency hiding because there are half as many resident threads per XVE.

For B70, prefer regular GRF for DMMV and quant unpack. Reach for large GRF only when XMX tile sizes or fused attention pipelines force more than 128 GRFs per thread.

There is no separate scalar register file. Constants, descriptors, and uniform values live in the same GRF and are kept uniform across the subgroup by the compiler.

### Memory Hierarchy

For compute kernels on B70:

```text
Thread → GRF registers (8 KB regular / 16 KB large per thread)
  ↓
Subgroup → SLM (128 KB/Xe core, workgroup-managed scratchpad)
  ↓
Xe core → L1 cache (256 KB)
  ↓
GPU → L2 cache (shared last-level GPU cache, driver-reported size)
  ↓
GPU → GDDR6 VRAM (32 GB, 256-bit, 608 GB/s)
  ↓
Host → PCIe 5.0 x16 (~64 GB/s per direction)
```

**Notes for ZINC kernels:**

- SLM is not coherent with global memory. Treat it as a load–barrier–compute–store scratchpad scoped to the workgroup.
- L1 is per Xe core. Two workgroups on different Xe cores do not share L1 state even when accessing the same weights.
- L2 is shared GPU-wide. Working sets that fit in L2 amplify effective bandwidth above the 608 GB/s VRAM spec.
- Memory operations are message-based (`send` / `sendc` / `sendsc` / `sends`). A load is a routed request with coalescing, cache, and response behavior; latency is hidden by hardware-thread occupancy, not ILP alone.
- Block messages (`load_block2d`, `store_block2d`) carry more bytes per send and reduce instruction pressure on tiled kernels.
- Resizable BAR is effectively mandatory for B-series benchmark nodes; without it, host-visible VRAM uploads stall.

**For DMMV and attention:**

- Use contiguous subgroup-lane access. Memory access shape across the subgroup controls SIMD lane and memory efficiency.
- Prefer block-like loads/stores for tiled work; the compiler may lower structured access into more efficient send messages.
- Avoid atomics, fences, and cross-workgroup coordination in the token hot path.
- Watch register spills. IGC reports ahead-of-time spill warnings; `unitrace` covers JIT paths.

### SLM (Shared Local Memory)

- **128 KB per Xe core**, scoped to the workgroup scheduled on that Xe core
- **Max SLM per workgroup: 128 KB** — at the maximum, only one workgroup can reside per Xe core
- Bank conflicts apply on power-of-two strided access; broadcast (multiple lanes reading the same address) is free
- Latency is lower than L1 but higher than GRF; cross-subgroup reductions through SLM cost a barrier
- Used for tiled attention, multi-stage reductions across subgroups, and dequant scale/min metadata staging

For decode-path reductions, prefer subgroup reductions first; spill into SLM only when reducing across more than one subgroup.

### XMX Matrix Engines (B70)

B70 exposes **256 XMX engines** — 8 per Xe core × 32 Xe cores — paired one-to-one with vector engines. They execute DPAS (Dot Product Accumulate Systolic) at subgroup granularity.

| Property | B70 |
|---|---|
| XMX engines | 256 (32 Xe cores × 8) |
| Subgroup width for DPAS | SIMD16 or SIMD32 (tile-dependent) |
| Native low-precision paths | INT8, FP16, BF16 |
| INT8 peak | **367 TOPS** |
| FP32 (vector path, no XMX) | 22.94 TFLOPS |
| Vulkan entry | `VK_KHR_cooperative_matrix` when properties are usable |
| SYCL / Level Zero entry | `joint_matrix_mad` |

For ZINC, XMX is a prefill and batched-matmul lever, not a decode lever. Query `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` at runtime for tile shapes — do not hard-code RDNA4's 16×16×16. Plan for an unpack step between Q4_K/Q5_K weights and the matrix tile inputs; the systolic path does not consume packed GGUF blocks directly.

## Opcode And ISA Surface

Intel does not publish a current Xe2 native ISA manual in the same direct style as AMD's RDNA ISA PDFs. The public surface we can rely on is:

- Intel oneAPI documentation for Xe architecture, subgroups, SLM, and XMX.
- Intel Graphics Compiler source, especially the G4/vISA opcode tables and send-op tables.
- Older public Intel processor graphics architecture material and Alchemist PRM files for the general EU instruction model.

This means the table below is best read as the compiler-visible opcode/mnemonic surface relevant to B-series shader work, not a guaranteed complete Xe2 binary encoding manual.

### Execution Model Primitives

| Primitive | What it means for kernels |
| --- | --- |
| SIMD execution size | Instructions operate over SIMD lanes, typically 16 or 32 work-items for Xe2-HPG compute. |
| Predication | Per-lane flag predicates disable lanes without changing the instruction stream. Useful but branch divergence still increases dynamic work. |
| Execution mask | Implicit lane mask tracks active lanes through control flow. |
| Flag/condition modifiers | `cmp` and arithmetic can set flags such as equal, greater, less, overflow, unordered. |
| Source modifiers | Negate, absolute value, and related modifiers can fold simple transforms into operand fetch. |
| Saturation | Clamp arithmetic to destination range when the instruction supports it. |
| Regioning | GRF operands can be addressed with strides and subregister regions. This is powerful but easy to turn into scattered access. |

### Opcode Families

| Family | Representative mnemonics | Inference relevance |
| --- | --- | --- |
| Move/select | `mov`, `sel`, `csel`, `movi`, `smov`, `fcvt` | Copies, predicated select, conversions, dequant plumbing |
| Integer/bit logic | `and`, `or`, `xor`, `not`, `bfe`, `bfi1`, `bfi2`, `bfrev`, `fbh`, `fbl`, `cbit`, `bfn` | Quantized unpack, masks, bitfield extraction, packed GGUF block decode |
| Scalar/vector arithmetic | `add`, `mul`, `avg`, `frc`, `rndu`, `rndd`, `rnde`, `rndz`, `mac`, `mach`, `mad`, `madm`, `add3`, `addc`, `subb`, `shr`, `shl`, `asr`, `ror`, `rol`, `lzd` | DMMV inner loops, address math, reductions, quant scale application |
| Dot/matrix | `dp4a`, `dpas`, `dpasw` | INT8 dot products and XMX systolic matrix operations |
| Compare | `cmp`, `cmpn` | Masks, stop conditions, bounds checks, reductions |
| Control flow | `if`, `else`, `endif`, `while`, `brd`, `brc`, `break`, `cont`, `goto`, `jmpi`, `call`, `return`, `halt`, `join` | Branching and loops; divergence should be minimized in SIMD kernels |
| Math unit | `math` functions such as reciprocal, log, exp, sqrt, rsqrt, pow, sin, cos, integer divide | Softmax, normalization, activation approximations, though compilers may lower or approximate |
| Send/message | `send`, `sendc`, `sends`, `sendsc` | Loads, stores, scatter/gather, atomics, sampler, fences, barriers |
| Sync/misc | `wait`, `nop`, `sync_nop`, `sync_allrd`, `sync_allwr`, `sync_fence` | Ordering and scoreboard control; keep out of hot paths unless required |

### Data Types

IGC's type table includes:

| Syntax | Meaning | Notes |
| --- | --- | --- |
| `ub`, `b` | 8-bit unsigned/signed integer | Quant blocks, packed metadata |
| `uw`, `w` | 16-bit unsigned/signed integer | Halfword unpack, offsets |
| `ud`, `d` | 32-bit unsigned/signed integer | Most address and loop math |
| `uq`, `q` | 64-bit unsigned/signed integer | Pointers and large offsets |
| `hf` | IEEE FP16 | Common XMX/prefill input type |
| `bf` | BF16 | Useful if driver/compiler exposes BF16 matrix support |
| `f` | FP32 | Accumulators, normalization, logits |
| `df` | FP64 | Present in architecture tables, not useful for LLM hot paths |
| `tf32` | TensorFloat-32 container | Matrix path only when exposed by the compiler/device |
| `hf8`, `bf8`, `e2m1` | FP8 / low-bit compiler IR types | Do not assume usable B-series Vulkan exposure without probing features |

For Vulkan, always query device features and cooperative-matrix properties. Do not infer FP8, BF16, or TF32 shader support from a compiler enum alone.

### DPAS And XMX

`dpas` stands for Dot Product Accumulate Systolic. Intel's XMX documentation describes XMX as systolic hardware executing DPAS-style operations for low-precision matrix work. In SYCL, the lower-level route is `joint_matrix_mad`; in Vulkan, the portable route is `VK_KHR_cooperative_matrix` / `SPV_KHR_cooperative_matrix` when the driver advertises matching properties.

For ZINC:

- DPAS/XMX is a prefill and batched-matmul opportunity first.
- Single-token DMMV should not be forced onto XMX until profiling proves the packing and tile overhead are worth it.
- Q4_K and related GGUF formats need unpacking. If the driver exposes only FP16/BF16/INT8 cooperative matrices, a native 4-bit path still needs a conversion strategy.
- The B70 is the obvious XMX target. The B65/B60/B580 all expose 160 XMX engines; the real difference is memory bandwidth and clock.

### Send Messages

Intel memory operations are message based. The public compiler send-op table includes:

| Send group | Representative operations | ZINC interpretation |
| --- | --- | --- |
| Loads | `load`, `load_strided`, `load_quad`, `load_block2d`, status variants | Use for weight/KV reads; contiguous/block forms are the goal |
| Stores | `store`, `store_strided`, `store_quad`, `store_block2d`, uncompressed variants | Use for hidden buffers, KV writes, logits, staging |
| Atomics | integer, floating, and BF16 add/sub/min/max/CAS variants | Avoid in decode unless a reduction cannot be expressed within a subgroup/workgroup |
| Fences/barriers | `fence`, `signal_barrier`, named/system barriers, `wait` | Correctness tools; latency hazards in the token loop |
| Sampler/render | sample, gather4, render read/write | Mostly irrelevant for ZINC compute kernels |

The practical lesson is that memory latency is not a simple load instruction latency. A load is a message with routing, coalescing, cache, and response behavior. Occupancy and access shape are the performance levers.

## ZINC Bring-Up Notes

### Device Detection

Use both PCI device ID and device name. The public B-series IDs are:

| SKU | Device ID |
| --- | --- |
| Arc Pro B70 | 0xE223 |
| Arc Pro B65 | 0xE222 |
| Arc Pro B60 | 0xE211 |
| Arc Pro B50 | 0xE212 |
| Arc B580 | 0xE20B |
| Arc B570 | 0xE20C |

Recommended ZINC defaults by SKU:

| SKU | Bandwidth default | Xe cores | Subgroup starting point | Notes |
| --- | ---: | ---: | ---: | --- |
| B70 | 608 GB/s | 32 | 32 | Best 32 GB target |
| B65 | 608 GB/s | 20 | 32 | Decode-friendly memory/core ratio |
| B60 | 456 GB/s | 20 | 32 | 24 GB capacity, B580-class bandwidth |
| B50 | 224 GB/s | 16 | 32 | Low-power 16 GB, bandwidth limited |
| B580 | 456 GB/s | 20 | 32 | Best consumer 8B target |
| B570 | 380 GB/s | 18 | 32 | Capacity-limited |

Avoid a single "B-series = 640 GB/s" heuristic. The actual public line spans 224 to 608 GB/s.

### Vulkan Capability Probe

Before judging performance, log these fields from `vulkaninfo` or ZINC diagnostics:

- `subgroupSize`, `minSubgroupSize`, `maxSubgroupSize`, and `VK_EXT_subgroup_size_control`
- `VK_KHR_shader_float16_int8`
- `VK_KHR_8bit_storage`
- `VK_KHR_shader_integer_dot_product`
- `VK_KHR_cooperative_matrix`
- `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` output if cooperative matrix is present
- maximum workgroup size and maximum shared memory / workgroup
- PCIe generation/link width and whether Resizable BAR is enabled in firmware

Resizable BAR is worth treating as mandatory for benchmark nodes. Intel's support guidance describes it as required for optimal Arc performance; without it, host-visible VRAM access and upload behavior can be misleading.

### Shader Tuning Starting Points

| Kernel family | Initial Intel strategy |
| --- | --- |
| DMMV Q4_K/Q5_K/Q6_K/Q8_0 | Specialize for subgroup 32 first. Compare local sizes 32, 64, and 128. Keep one output row per subgroup until profiling says otherwise. |
| Dequant/unpack | Use vectorized packed loads and bitfield ops. Keep scale/min metadata contiguous. Avoid scattered per-lane byte reads. |
| RMS norm / reductions | Use subgroup reductions for the first reduction stage; spill to SLM only when reducing across multiple subgroups. |
| Flash attention | Use 16/32 subgroup reductions and SLM tiles. Tune tile width against SLM residency and L2 hit rate. |
| MoE routing | Router logits are small; avoid global atomics. CPU top-k may be acceptable until GPU routing is measured. |
| Batched prefill | Prototype cooperative matrix only after scalar/subgroup path is coherent. Query matrix tile shapes instead of assuming RDNA's 16x16x16 path. |
| KV cache | Align pages and rows to cache-friendly boundaries. For B570/B580, cap context before temp buffers push the card into memory pressure. |

### Benchmark Interpretation

Use the same clean-node discipline as RDNA:

- stop stale `zinc`, `llama-server`, and other GPU users
- warm once before measuring
- measure CLI decode separately from HTTP latency
- collect at least three runs
- record driver, kernel, Mesa/intel driver package, Vulkan ICD, and firmware/BIOS ReBAR state

For B65/B70 specifically, compare:

1. 8B decode, to validate DMMV against B580/B60-class cards.
2. 27B/35B fit and decode, to prove the 32 GB cards are buying usable capacity.
3. long-context attention, to see whether L2/SLM tuning or KV bandwidth dominates.
4. batched prefill with and without cooperative matrix, if the driver exposes it.

## Card-By-Card Engineering Summary

### Arc Pro B70

B70 is the flagship B-series inference card: 32 GB, 608 GB/s, 32 Xe cores, 256 XMX engines, 367 INT8 TOPS, and PCIe 5.0 x16. It is the first target for ZINC's large-model Intel work because it has both enough memory and enough compute to make prefill tuning meaningful. Expect decode to scale primarily with 608 GB/s bandwidth, while prefill should benefit from the extra XMX engines if cooperative matrix is usable.

### Arc Pro B65

B65 keeps the 32 GB / 608 GB/s memory system but drops to 20 Xe cores. That makes it a strong decode candidate: for bandwidth-bound single-token inference, it may be near B70 while using less silicon. It is weaker for large prompt prefill and batched serving because matrix compute and scheduler occupancy have less headroom.

### Arc Pro B60

B60 is best understood as a 24 GB, pro-oriented version of the 20 Xe-core Battlemage shape. It has the same 456 GB/s bandwidth class as B580 but twice the memory capacity. It is useful for models that do not fit on B580, but it will not fix a bandwidth-bound decode bottleneck by itself.

### Arc Pro B50

B50 is the power-efficient and small-form-factor option: 16 GB, 70 W, no auxiliary power connector, ECC support on Intel's product page, and 224 GB/s memory bandwidth. It is attractive for compact 8B inference boxes, but the bandwidth is roughly half of B580/B60 and about 37% of B65/B70. Do not expect high single-stream decode throughput.

### Arc B580

B580 is the consumer card to bring up first for small models. It has 20 Xe cores, 160 XMX engines, a high 2670 MHz graphics clock, 12 GB VRAM, and 456 GB/s bandwidth. Its practical ceiling is memory capacity, not compute. It should be a good 8B ZINC target once subgroup and memory-message behavior are tuned.

### Arc B570

B570 is a trimmed 18 Xe-core, 10 GB, 380 GB/s card. The bandwidth is still respectable, but the 10 GB memory pool leaves little room for larger GGUFs, long context, and temporary buffers. It is a useful correctness and lower-end coverage target, not the main optimization target.

## Open Questions For ZINC

- Do B-series Vulkan drivers expose cooperative matrix shapes that map cleanly to GGUF prefill data types?
- Is subgroup 32 always the best decode width, or do some B-series drivers default to subgroup 16 for specific shaders?
- Does B65 match B70 on single-stream decode once kernels are memory-bound?
- How much practical bandwidth does each card sustain on ZINC's quantized DMMV, not synthetic copy tests?
- Can Q4_K unpack be restructured to use INT8 dot-product or XMX paths without losing the bandwidth advantage of compact weights?
- Does Arc Pro ECC materially reduce bandwidth on B70/B50, and can it be toggled or queried reliably on Linux?
- What Linux kernel, Mesa/ANV, and firmware versions are required for stable B70/B65 operation on the benchmark node?

## References

### Intel Product Pages

- [Intel Arc B580 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)
- [Intel Arc B570 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html)
- [Intel Arc Pro B70 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)
- [Intel Arc Pro B65 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html)
- [Intel Arc Pro B60 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html)
- [Intel Arc Pro B50 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html)
- [Intel Arc B-Series desktop overview](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html)
- [Intel Arc Pro B-Series workstation overview](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)
- [Intel Arc B-Series Graphics Quick Reference Guide](https://cdrdv2-public.intel.com/839907/Intel%20Arc%20B-Series%20Graphics%20Quick%20Reference%20Guide%20V1.1.pdf)

### Architecture And Programming

- [Intel oneAPI GPU Optimization Guide: Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
- [Intel oneAPI GPU Optimization Guide: Sub-Groups and SIMD Vectorization](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/sub-groups-and-simd-vectorization.html)
- [Intel oneAPI GPU Optimization Guide: Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html)
- [Intel oneAPI GPU Optimization Guide: Programming Intel XMX Using SYCL Joint Matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html)
- [Intel oneAPI GPU Optimization Guide: Boost Matrix Multiplication Performance with Intel Xe Matrix Extensions](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/boost-matrix-multiplication-performance-with-intel.html)
- [Intel Processor Graphics: Architecture, ISA and Microarchitecture](https://www.intel.cn/content/dam/develop/external/us/en/documents/intel-graphics-architecture-isa-and-microarchitecture-698638.pdf)
- [Vulkan VK_KHR_cooperative_matrix reference](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [Intel support: What Is Resizable BAR and How Do I Enable It?](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html)

### Compiler Source

- [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
- [IGC G4 instruction list](https://github.com/intel/intel-graphics-compiler/blob/master/visa/G4_Instruction.h)
- [IGC G4 opcode and type definitions](https://github.com/intel/intel-graphics-compiler/blob/master/visa/G4_Opcode.h)
- [IGC send operation table](https://github.com/intel/intel-graphics-compiler/blob/master/visa/iga/IGALibrary/IR/EnumSendOpInfo.hpp)
