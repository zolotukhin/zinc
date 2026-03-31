# AMD RDNA3/RDNA4 GPU Reference for Inference

Hardware specifications, ISA details, and compute architecture reference for LLM inference on AMD consumer GPUs. Consolidated from AMD product pages, ROCm hardware tables, AMD ISA manuals, GPUOpen documentation, and profiling data.

## Hardware Specifications

> Note: Board power, clocks, VRAM, bus width, and memory bandwidth below were re-checked against AMD product pages on 2026-03-30. Wave size, LDS, Infinity Cache, and L2 cache rows are cross-checked against AMD's ROCm hardware tables. `Radeon AI PRO R9700` memory speed is derived from AMD's published 640 GB/s bandwidth over a 256-bit bus because the product page lists bandwidth but not Gbps directly. `Radeon RX 9070 GRE` ships as a 12 GB board on AMD's product page even though current ROCm tables still list 16 GiB; this reference follows the product page for shipping board specs. Blank clock cells for `Radeon RX 9060` are intentional because AMD's product page does not currently publish game / boost clocks there.

### RDNA4 — Navi 48 (gfx1201)

| | RX 9070 XT | RX 9070 | RX 9070 GRE | Radeon AI PRO R9700 |
|---|---|---|---|---|
| **Compute Units** | 64 | 56 | 48 | 64 |
| **Stream Processors** | 4096 | 3584 | 3072 | 4096 |
| **AI Accelerators (WMMA)** | 128 | 112 | 96 | 128 |
| **Wave Size** | 32 or 64 | 32 or 64 | 32 or 64 | 32 or 64 |
| **VRAM** | 16 GB GDDR6 | 16 GB GDDR6 | 12 GB GDDR6 | 32 GB GDDR6 |
| **Memory Bus** | 256-bit | 256-bit | 192-bit | 256-bit |
| **Memory Speed** | 20 Gbps | 20 Gbps | 18 Gbps | 20 Gbps* |
| **Memory Bandwidth** | 640 GB/s | 640 GB/s | 432 GB/s | 640 GB/s |
| **Infinity Cache** | 64 MB | 64 MB | 48 MB | 64 MB |
| **L2 Cache** | 8 MB | 8 MB | 6 MB | 8 MB |
| **Game Clock** | 2400 MHz | 2070 MHz | 2220 MHz | 2350 MHz |
| **Boost Clock** | 2970 MHz | 2520 MHz | 2790 MHz | 2920 MHz |
| **Board Power** | 304 W | 220 W | 220 W | 300 W |
| **Process** | TSMC N4P | TSMC N4P | TSMC N4P | TSMC N4P |
| **Transistors** | 53.9B | 53.9B | 53.9B | 53.9B |
| **Die Size** | 356.5 mm² | 356.5 mm² | 356.5 mm² | 356.5 mm² |

### RDNA3 — Navi 31 (gfx1100) and Navi 32 (gfx1101)

| | RX 7900 XTX | RX 7900 XT | RX 7800 XT | RX 7700 XT |
|---|---|---|---|---|
| **Die** | Navi 31 (gfx1100) | Navi 31 (gfx1100) | Navi 32 (gfx1101) | Navi 32 (gfx1101) |
| **Compute Units** | 96 | 84 | 60 | 54 |
| **Stream Processors** | 6144 | 5376 | 3840 | 3456 |
| **VRAM** | 24 GB GDDR6 | 20 GB GDDR6 | 16 GB GDDR6 | 12 GB GDDR6 |
| **Memory Bus** | 384-bit | 320-bit | 256-bit | 192-bit |
| **Memory Bandwidth** | 960 GB/s | 800 GB/s | 624 GB/s | 432 GB/s |
| **Infinity Cache** | 96 MB | 80 MB | 64 MB | 48 MB |
| **L2 Cache** | 6 MB | 6 MB | 4 MB | 4 MB |
| **Game Clock** | 2300 MHz | 2000 MHz | 2124 MHz | 2171 MHz |
| **Boost Clock** | 2500 MHz | 2400 MHz | 2430 MHz | 2544 MHz |
| **Board Power** | 355 W | 315 W | 263 W | 245 W |
| **Process** | TSMC N5 (GCD) + N6 (MCD) | TSMC N5 + N6 | TSMC N5 + N6 | TSMC N5 + N6 |

### RDNA4 — Navi 44 (gfx1200)

| | RX 9060 XT (16 GB) | RX 9060 XT (8 GB) | RX 9060 |
|---|---|---|---|
| **Die** | Navi 44 | Navi 44 | Navi 44 |
| **Compute Units** | 32 | 32 | 28 |
| **Stream Processors** | 2048 | 2048 | 1792 |
| **VRAM** | 16 GB GDDR6 | 8 GB GDDR6 | 8 GB GDDR6 |
| **Memory Bus** | 128-bit | 128-bit | 128-bit |
| **Memory Speed** | 20 Gbps | 20 Gbps | 18 Gbps |
| **Memory Bandwidth** | 320 GB/s | 320 GB/s | 288 GB/s |
| **L2 Cache** | 4 MB | 4 MB | 4 MB |
| **Infinity Cache** | 32 MB | 32 MB | 32 MB |
| **Game Clock** | 2530 MHz | 2530 MHz | — |
| **Boost Clock** | 3130 MHz | 3130 MHz | — |
| **Board Power** | 160 W | 150 W | 132 W |
| **Transistors** | 29.7B | 29.7B | 29.7B |
| **Die Size** | 199 mm² | 199 mm² | 199 mm² |

## Compute Unit Architecture

### RDNA4 CU Layout

![RDNA4 WGP Architecture](/compute-unit.svg)

Each RDNA4 Compute Unit contains:

- **2 SIMDs** (Shader Instruction, Multiple Data units)
- **1 Scalar Unit** shared between SIMDs
- **1 LDS** (Local Data Share) — 128 KB per WGP (Work Group Processor = 2 CUs), 64 KB per CU in CU mode
- **L0 Vector Cache** — 32 KB (replaces RDNA3's L1)
- **L0 Scalar Cache** — 16 KB
- **L0 Instruction Cache** — 32 KB

Each SIMD has:
- **192 KB register file** (VGPR)
- **16 thread slots** (waves)
- **256 addressable VGPRs** per thread (1024 bits wide in wave32)
- **Dynamic VGPR allocation** — registers allocated in blocks of 16 or 32, up to 8 blocks per thread

### Wave Execution Model

A **wave** (wavefront) is the fundamental execution unit. All threads in a wave execute the same instruction simultaneously.

| Property | Wave32 | Wave64 |
|---|---|---|
| Threads per wave | 32 | 64 |
| VGPR width per lane | 32 bits | 32 bits |
| VGPR register count | up to 256 | up to 256 |
| Occupancy (max waves/SIMD) | 16 | 16 |
| Execution | 1 cycle per instruction | 2 cycles per instruction (on RDNA4 SIMDs) |
| Best for | Latency-sensitive, low-register | Bandwidth-bound, high-register |

**For LLM inference decode (DMMV):** Wave64 is optimal on RDNA4. Measured faster than wave32 for memory-bandwidth-bound matmul-vec operations. Wave32 showed no improvement despite halving execution width.

**For WMMA (cooperative matrix):** Wave32 only on RDNA4. The WMMA instructions operate on 16x16x16 tiles and require wave32 execution.

### Register File

**VGPRs (Vector General Purpose Registers):**
- 192 KB per SIMD (RDNA4), shared across all active waves
- Each VGPR is 32 bits × wave_size lanes (32 lanes for wave32, 64 lanes for wave64)
- Max 256 VGPRs per thread
- RDNA4 dynamic allocation: 16 or 32 register blocks, allocated on demand
- **Occupancy threshold**: ≤96 VGPRs = max occupancy (16 waves/SIMD). Above 96, occupancy drops.

| VGPRs Used | Waves/SIMD (Wave32) | Waves/SIMD (Wave64) |
|---|---|---|
| ≤96 | 16 | 16 |
| 97–128 | 12 | 12 |
| 129–192 | 8 | 8 |
| 193–256 | 6 | 6 |

Note: Wave64 waves consume 2x the physical register space of wave32 waves (64 lanes vs 32). The VGPR counts above are per-thread. The dynamic allocation granularity (16 or 32 register blocks, up to 8 blocks) means a shader using 97 VGPRs may be rounded up to 112 (7×16) or 128 (4×32), depending on the block size the allocator chooses.

**LDS-limited occupancy:** If a workgroup uses too much LDS, occupancy drops independently of VGPRs. With 64 KB per CU: a workgroup using 32 KB of LDS limits the CU to 2 concurrent workgroups. A workgroup using 16 KB allows 4 concurrent workgroups. For inference kernels, keep LDS usage ≤16 KB per workgroup when occupancy matters.

**SGPRs (Scalar General Purpose Registers):**
- 32 KB per CU, shared among waves
- Fixed allocation — always enough for max occupancy
- Used for uniform values (constants, buffer descriptors, loop counters)

### Memory Hierarchy

![RDNA4 Memory Hierarchy](/memory-hierarchy.svg)

```
Thread → VGPRs (192 KB/SIMD)
  ↓
Wave → LDS (64 KB/CU, 128 KB/WGP) — 32 banks, 4 bytes/bank
  ↓
CU → L0 Vector Cache (32 KB, ~30 cycles latency)
  ↓
Shader Engine → L2 Cache (8 MB on Navi 48, ~100 cycles)
  ↓
GPU → Infinity Cache (64 MB on Navi 48, ~150 cycles)
  ↓
GPU → GDDR6 VRAM (8–32 GB, ~300+ cycles)
```

**Cache line sizes:**
- L0 Vector Cache: 64 bytes per line
- L2 Cache: 128 bytes per line
- Infinity Cache: 64 bytes per line

**Key bandwidths for inference:**
- LDS: ~64 bytes/clock/CU (dual-issue capable)
- L0 Vector Cache: ~64 bytes/clock read
- L2: doubled bandwidth per slice vs RDNA3
- VRAM: 288–640 GB/s theoretical across current RDNA4 boards; on Navi 48 we typically see 67–93% utilization on large DMMV
- PCIe 5.0 x16: ~64 GB/s per direction (128 GB/s bidirectional) — relevant for model loading

**Memory coalescing:** Contiguous 4-byte accesses across all threads in a wave coalesce into full cache line requests. A wave32 reading 32 consecutive 4-byte values (128 bytes) generates 2 cache line requests (64 bytes each). Non-coalesced (scattered) access patterns cause replay penalties — each unique cache line touched generates a separate request. For DMMV, weight data is accessed sequentially so coalescing is natural. For KV cache with paged attention, page alignment to cache lines matters.

**Infinity Cache** acts as a victim cache / last-level cache between L2 and VRAM. It amplifies effective bandwidth: workloads with working sets that fit in Infinity Cache see much higher effective bandwidth than the VRAM spec. On Navi 48 with 64 MB, this covers the KV cache of a few concurrent 4K-context sessions.

### LDS (Local Data Share)

- 128 KB per WGP (2 CUs share), 64 KB per CU in CU mode
- 32 banks, each 4 bytes (32 bits) wide
- Bank conflicts serialize access — conflict degree determines stall cycles
- Broadcast: multiple threads reading the same address is free (no conflict)
- Padding strategy: add 1 element per row to avoid bank conflicts on column access patterns
- Used for: shared memory in workgroups, cross-lane communication, reduction operations
- Access latency: ~2 cycles (no bank conflict)

## WMMA (Wave Matrix Multiply Accumulate)

### RDNA4 WMMA

RDNA4 matrix cores operate via WMMA instructions at **wave32** granularity.

| Input Types | Output Types | Tile Size | FLOPS/clock/CU |
|---|---|---|---|
| FP16 × FP16 | FP32 or FP16 | 16 × 16 × 16 | 1024 |
| BF16 × BF16 | FP32 or BF16 | 16 × 16 × 16 | 1024 |
| I8 × I8 | I32 | 16 × 16 × 16 | 2048 |

**VGPR cost per matrix fragment (wave32):**

| Fragment | FP16/BF16 | I8 | I4 |
|---|---|---|---|
| A (input) | 8 VGPRs | 4 VGPRs | 2 VGPRs |
| B (input) | 8 VGPRs | 4 VGPRs | 2 VGPRs |
| C/D (accumulator) | 8 VGPRs | 8 VGPRs | 8 VGPRs |
| **Total per MulAdd** | **24 VGPRs** | **16 VGPRs** | **12 VGPRs** |

At 24 VGPRs for FP16 WMMA, a shader doing only WMMA can still achieve high occupancy (16 waves/SIMD with ≤96 VGPRs means 4 concurrent WMMA tiles per SIMD).

**Matrix layout (RDNA4, simplified vs RDNA3):**
- Matrix A: column-major, 8 elements per lane across 32 lanes
- Matrix B: row-major, 8 elements per lane across 32 lanes
- Matrix C/D: row-major, 8 elements per lane
- No inter-lane data shuffling required (RDNA3 required lanes 0-15 to replicate into 16-31)
- Each lane holds 8 elements → 32 lanes × 8 = 256 elements = 16×16 matrix

**GLSL intrinsic (via VK_KHR_cooperative_matrix):**
```glsl
// Declare cooperative matrix types
coopmat<float16_t, gl_ScopeSubgroup, 16, 16> A, B;
coopmat<float, gl_ScopeSubgroup, 16, 16> C;

// Load from buffer
coopMatLoad(A, buf_a, offset, stride, gl_CooperativeMatrixLayoutColumnMajor);
coopMatLoad(B, buf_b, offset, stride, gl_CooperativeMatrixLayoutRowMajor);

// Multiply-accumulate: D = A × B + C
coopmat<float, gl_ScopeSubgroup, 16, 16> D = coopMatMulAdd(A, B, C);

// Store result
coopMatStore(D, buf_d, offset, stride, gl_CooperativeMatrixLayoutRowMajor);
```

**Cooperative matrix also supports element-wise operations** on loaded tiles: `+`, `-`, `*`, `/`, type conversions, and scalar multiplication. This enables fusing post-WMMA operations (bias add, activation) without storing the tile back to memory.

**SPIR-V operand flags for `OpCooperativeMatrixMulAddKHR`:**
- `MatrixASignedComponentsKHR` (0x1) — treat A elements as signed (for integer types)
- `MatrixBSignedComponentsKHR` (0x2) — treat B elements as signed
- `MatrixCSignedComponentsKHR` (0x4) — treat C elements as signed
- `MatrixResultSignedComponentsKHR` (0x8) — treat result as signed
- `SaturatingAccumulationKHR` (0x10) — clamp accumulation to representable range

**Querying supported configurations:** Use `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` to enumerate what the driver supports. RADV on RDNA4 reports the 16×16×16 tiles for FP16, BF16, and I8. The SPIR-V shader must declare `OpCapability CooperativeMatrixKHR`.

### RDNA3 WMMA

| Input Types | Output Types | Tile Size | FLOPS/clock/CU |
|---|---|---|---|
| FP16 × FP16 | FP32 or FP16 | 16 × 16 × 16 | 512 |
| BF16 × BF16 | FP32 or BF16 | 16 × 16 × 16 | 512 |
| I8 × I8 | I32 | 16 × 16 × 16 | 512 |
| I4 × I4 | I32 | 16 × 16 × 16 | 1024 |

- Both wave32 and wave64 supported
- More complex register layout — lanes 0-15 must replicate data into lanes 16-31 (wave32) or 0-15 into 32-47 and 48-63 (wave64) before WMMA
- **OPSEL parameter**: boolean flag controlling whether 16-bit output elements are stored in the lower or upper half of 32-bit VGPRs. Affects how chained WMMA operations pass data.
- Enable BF16 on RDNA3 with: `RADV_PERFTEST=bfloat16`

### When to Use WMMA for Inference

| Operation | Use WMMA? | Why |
|---|---|---|
| Prefill attention (large batch) | Yes | Compute-bound, large matrices, high FLOPS utilization |
| Decode DMMV (single token) | No | Memory-bandwidth-bound, n=1, can't fill tiles efficiently |
| Batched decode (4+ tokens) | Maybe | Depends on batch size vs. tile size |
| KV cache operations | No | Element-wise, not matrix multiply |

## Instruction Set Highlights for Inference

### Relevant RDNA4 Instructions

**Vector ALU (VALU):**
- `V_FMA_F32` / `V_FMA_F16` — fused multiply-add (core of all matmul)
- `V_DOT2_F32_F16` — 2-element FP16 dot product to FP32 (useful for dequant+dot in DMMV)
- `V_DOT4_I32_IU8` — 4-element int8 dot product to int32 (integer quantized path)
- `V_CVT_F32_F16` / `V_CVT_F16_F32` — float conversion (KV cache, dequantization)
- `V_PACK_B32_F16` — pack two FP16 values into one 32-bit register
- `V_LSHRREV_B32` / `V_AND_B32` / `V_BFE_U32` — bit extraction (quantized weight unpacking)
- `V_CNDMASK_B32` — conditional select (used in top-k, sampling)
- `V_MAX_F32` / `V_MIN_F32` — reduction operations (softmax max, RMS norm)
- `V_EXP_F32` — base-2 exponential (softmax exp, approximate via `V_EXP_F16`)
- `V_RCP_F32` — reciprocal (softmax normalization, RMS norm)
- `V_SQRT_F32` — square root (RMS norm)

**Transcendental (approximate):**
- `V_EXP_F32` — 2^x (1 ULP accuracy)
- `V_LOG_F32` — log2(x) (1 ULP accuracy)
- `V_RCP_F32` — 1/x (1 ULP accuracy)
- `V_RSQ_F32` — 1/sqrt(x) (1 ULP accuracy)
- `V_SIN_F32` / `V_COS_F32` — trig (not used in inference)

**Additional dot product instructions:**
- `V_DOT2_F32_BF16` — 2-element BF16 dot product to FP32
- `V_DOT4_I32_IU4` — 4-element 4-bit integer dot product (for INT4 quantized paths)
- `V_DOT8_I32_IU4` — 8-element 4-bit integer dot product

**Cross-lane operations (critical for reductions in softmax, normalization):**
- `V_READLANE_B32` / `V_WRITELANE_B32` — read/write specific lane (scalar broadcast)
- `DS_PERMUTE_B32` / `DS_BPERMUTE_B32` — arbitrary cross-lane permutation via LDS hardware
- `V_PERM_B32` — byte permutation within a register (data layout transforms)
- DPP (Data Parallel Primitives) modifiers on VALU instructions:
  - `row_shr:N` — shift right by N lanes within a row
  - `row_shl:N` — shift left by N lanes
  - `row_ror:N` — rotate right within a row
  - `row_bcast:15/31` — broadcast lane 15 or 31 to the row
  - `wave_shr:1` / `wave_rol:1` — cross-row shifts
  - Used for efficient warp-level reductions: parallel prefix sum in log2(N) steps

**Scalar ALU (SALU) — new FP on RDNA4:**
- `S_CVT_F32_I32` / `S_CVT_F32_U32` — scalar float conversion (new in RDNA4)
- `S_FMAAK_F32` / `S_FMAMK_F32` — scalar FMA with inline constant
- Used for: uniform scaling factors, RoPE frequency computation, quantization scale factors

**Memory:**
- `GLOBAL_LOAD_DWORD` / `GLOBAL_LOAD_DWORDX4` — global memory load (weight/KV reads)
- `GLOBAL_STORE_DWORD` — global memory store
- `DS_READ_B32` / `DS_READ_B128` — LDS read (shared memory)
- `DS_WRITE_B32` / `DS_WRITE_B128` — LDS write
- `BUFFER_LOAD_FORMAT_XY` — structured buffer load (descriptor-based, for weight buffers)

**RDNA4 memory model changes:**
- Split memory counters: `vmcnt` is now split into `loadcnt` (global loads), `samplecnt` (texture), `bvhcnt` (ray tracing)
- `lgkmcnt` split into `kmcnt` (scalar memory) and `dscnt` (LDS/GDS)
- Split barriers: `S_BARRIER_SIGNAL` / `S_BARRIER_WAIT` replace `S_BARRIER` — enables overlapping barrier wait with independent work
- Out-of-order memory: RDNA4 removes false cross-wave memory ordering dependencies

### Quantized Weight Unpacking (Q4_K example)

Q4_K stores weights as 4-bit indices with per-block scales. The DMMV inner loop:

```
// Pseudocode for one wave64 thread processing Q4_K
// Each thread handles 2 rows, iterating over K dimension

1. GLOBAL_LOAD_DWORDX4 — load 16 bytes of packed 4-bit weights (32 values)
2. V_BFE_U32 × 8 — extract 4-bit indices from packed word
3. V_LSHRREV_B32 — shift for upper nibble extraction
4. V_AND_B32 (mask 0xF) — isolate 4-bit index
5. GLOBAL_LOAD_DWORD — load scale + min for this block
6. V_FMA_F32 — dequant: value = scale × index + min
7. V_FMA_F32 — accumulate: dot += dequant_value × input_activation
8. Repeat for K/32 iterations
9. DS_WRITE_B32 — write partial sum to LDS
10. S_BARRIER — sync workgroup
11. DS_READ_B32 + V_ADD_F32 — reduce partial sums across threads
12. GLOBAL_STORE_DWORD — write final output element
```

## Cycle Costs — Complete Reference

### Instruction Throughput (per SIMD, per clock)

| Instruction | Throughput (wave32) | Throughput (wave64) | Notes |
|---|---|---|---|
| `V_FMA_F32` | 1/clk (32 FMAs) | 1/2 clk (64 FMAs in 2 clk) | Core of all matmul |
| `V_FMA_F16` | 1/clk (32 FMAs) | 1/2 clk | Same rate as F32 |
| `V_DOT2_F32_F16` | 1/clk (64 FMAs) | 1/2 clk | 2× throughput vs FMA — use for DMMV |
| `V_DOT4_I32_IU8` | 1/clk (128 ops) | 1/2 clk | 4× throughput — INT8 quantized path |
| `V_ADD_F32` / `V_MUL_F32` | 1/clk | 1/2 clk | Simple ALU |
| `V_MAX_F32` / `V_MIN_F32` | 1/clk | 1/2 clk | Softmax max, reductions |
| `V_CVT_F32_F16` | 1/clk | 1/2 clk | Type conversion |
| `V_EXP_F32` | 1/4 clk | 1/8 clk | Transcendental unit, shared |
| `V_LOG_F32` | 1/4 clk | 1/8 clk | Transcendental unit, shared |
| `V_RCP_F32` | 1/4 clk | 1/8 clk | Reciprocal |
| `V_RSQ_F32` | 1/4 clk | 1/8 clk | Inverse square root |
| `V_SQRT_F32` | 1/4 clk | 1/8 clk | Square root |
| `V_LSHRREV_B32` / `V_AND_B32` | 1/clk | 1/2 clk | Bit ops for quant unpacking |
| `V_BFE_U32` | 1/clk | 1/2 clk | Bit field extract — 4-bit weight unpacking |
| `V_CNDMASK_B32` | 1/clk | 1/2 clk | Conditional select — top-k, sampling |
| `WMMA FP16 16×16×16` | 1/32 clk (1024 FMAs) | N/A (wave32 only on RDNA4) | Matrix core — prefill |
| `WMMA I8 16×16×16` | 1/32 clk (2048 ops) | N/A | 2× the integer throughput |
| `S_FMA_F32` (RDNA4) | 1/clk (scalar) | 1/clk (scalar) | New — uniform scale factors |

### Memory Latency

| Access | Latency | Bandwidth | Cache Line | Notes |
|---|---|---|---|---|
| **VGPR read** | 0 cycles | — | — | Register file, no latency |
| **SGPR read** | ~2 cycles | — | — | Scalar register |
| **LDS read** | ~2 cycles | 64 B/clk/CU | — | No bank conflict. N-way conflict = N× latency |
| **LDS read (broadcast)** | ~2 cycles | 64 B/clk/CU | — | Multiple threads same address = free |
| **L0 Vector Cache hit** | ~30 cycles | 64 B/clk | 64 B | Per-CU private |
| **L0 Scalar Cache hit** | ~30 cycles | — | — | Constants, descriptors |
| **L2 Cache hit** | ~100 cycles | 2× vs RDNA3 per slice | 128 B | Shared per Shader Engine |
| **Infinity Cache hit** | ~150 cycles | — | 64 B | Victim cache, amplifies BW |
| **VRAM (GDDR6)** | ~300+ cycles | 288–640 GB/s | 64 B | Full miss path, depends on SKU |
| **PCIe 5.0 x16** | ~1000+ cycles | 64 GB/s per direction | — | CPU↔GPU DMA (model loading) |

### Synchronization Costs

| Operation | Cost | Notes |
|---|---|---|
| `S_BARRIER` (RDNA3) | ~20 cycles | Full workgroup barrier |
| `S_BARRIER_SIGNAL` + `S_BARRIER_WAIT` (RDNA4) | ~20 cycles total | Can overlap wait with independent work |
| `S_WAITCNT` (memory fence) | 0 if ready, up to memory latency if not | Waits for outstanding loads |
| Vulkan dispatch (GPU-side) | 0.016 µs (~40 cycles at 2.5 GHz) | Negligible |
| `vkQueueSubmit` (CPU-side) | ~33 µs | One-time per submission |
| Command buffer replay | ~54 µs for 1500 dispatches | Pre-recorded, no re-recording |

### Worked Example: Single-Token DMMV (Q4_K, one row)

Computing one output element of a decode matmul-vec for Q4_K quantized weights on wave64.

Parameters: K=4096 (input dimension), 4-bit weights packed as Q4_K (32 values per 18-byte super-block).

```
Per super-block (32 weights, 18 bytes):
  1× GLOBAL_LOAD_DWORDX4          load 16B of packed weights     [~300 clk, pipelined]
  1× GLOBAL_LOAD_USHORT           load 2B scale+min              [~300 clk, pipelined]
  8× V_BFE_U32                    extract 4-bit indices           [8 clk on wave64]
  8× V_LSHRREV_B32 + V_AND_B32   upper nibble extraction         [8 clk]
  1× V_FMA_F32                    scale dequantization            [2 clk on wave64]
  16× V_FMA_F32                    dequant × activation + accum   [32 clk on wave64]
                                                                  ─────────
  Total ALU per super-block:                                      ~50 clk
  Total memory per super-block:                                   18 bytes

For K=4096: 4096/32 = 128 super-blocks
  Total memory: 128 × 18B = 2304 bytes per row
  Total ALU: 128 × 50 clk = 6400 clk ≈ 2.6 µs at 2.5 GHz

At 640 GB/s bandwidth (R9700 / RX 9070-class Navi 48): 2304 bytes / 640 GB/s = 0.0036 µs (memory)
The operation is ALU-bound for a single row — but with wave-level parallelism
across many rows, the GPU hides ALU latency behind memory latency of other waves.

With 16 waves per SIMD doing different rows simultaneously:
  Total = max(memory_latency, ALU_per_row) / occupancy_factor
  Effective rate ≈ bandwidth-limited for large matrices (m ≥ 8192)
```

### Worked Example: Softmax (1024 elements)

```
Phase 1: Find max (reduction)
  Load 1024 floats: 1024 × 4B = 4096 bytes
  Each wave64 thread loads 16 values: 16 × GLOBAL_LOAD_DWORD
  Per-thread max: 15 × V_MAX_F32 = 30 clk
  Cross-lane reduction (DPP):
    6 steps of row_shr + V_MAX_F32 = 12 clk (within wave64)
  Write max to LDS: DS_WRITE_B32 = 2 clk
  S_BARRIER = 20 clk
  Read global max from LDS: DS_READ_B32 = 2 clk

Phase 2: exp(x - max) and sum
  16 × V_SUB_F32 = 32 clk (subtract max)
  16 × V_MUL_F32 = 32 clk (scale for V_EXP_F32 base-2 input: x × log2(e))
  16 × V_EXP_F32 = 128 clk (transcendental, 1/4 throughput on wave64)
  Cross-lane sum reduction (DPP): 12 clk
  LDS write + barrier + read: 24 clk

Phase 3: divide by sum
  16 × V_RCP_F32 (once for 1/sum) = 8 clk
  16 × V_MUL_F32 = 32 clk
  16 × GLOBAL_STORE_DWORD (write output)

Total: ~300 cycles per wave64 ≈ 0.12 µs at 2.5 GHz
The bottleneck is V_EXP_F32 (128 clk for 16 values on the shared transcendental unit).
Fusing softmax with the preceding matmul avoids the extra global load/store round-trip.
```

### Worked Example: RMS Norm (hidden_dim=4096)

```
Input: 4096 floats
Each wave64 thread handles 64 values.

Phase 1: compute sum of squares
  64 × GLOBAL_LOAD_DWORD (pipelined)
  64 × V_FMA_F32 (x² accumulation) = 128 clk on wave64
  Cross-lane reduction: 12 clk
  LDS reduce across workgroup: 24 clk

Phase 2: normalize
  1 × V_RCP_F32 (1/N) = 8 clk
  1 × V_RSQ_F32 (1/sqrt(mean_sq)) = 8 clk
  64 × V_MUL_F32 (normalize) = 128 clk
  64 × V_MUL_F32 (multiply by learned scale — this is the "MUL" in RMS_NORM_MUL) = 128 clk
  64 × GLOBAL_STORE_DWORD

Total: ~450 cycles ≈ 0.18 µs
Fusing RMS_NORM + MUL into a single shader saves one global read + write pass (32 KB for 4096 floats).
Measured savings: ~500 µs per token (131 fused dispatches on Qwen3.5-35B-A3B).
```

## Dispatch and Command Buffer

### Vulkan Dispatch Overhead on RDNA4

Measured on Radeon AI PRO R9700:

| Test | Result |
|---|---|
| Single dispatch (record + submit + wait) | 33 µs |
| 1500 empty dispatches (GPU-side) | 24 µs = **0.016 µs/dispatch** |
| 1500 dispatches (wall clock) | 85 µs = 0.057 µs/dispatch |
| Pre-recorded command buffer replay (1500 dispatches) | 54 µs |

**Implication:** Dispatch overhead is negligible. The 2–5 µs per "dispatch" seen in inference profiling is real kernel execution time on small memory-bound tensors, not submission overhead.

### Command Buffer Strategy for Inference

The decode phase of LLM inference has a **static compute graph** — the same sequence of dispatches runs for every generated token, only the data (activations, KV cache pointers) changes.

**Pre-recording:**
1. Record the full decode graph into a command buffer once
2. Use push constants and descriptor set updates for per-token data
3. Replay via `vkQueueSubmit` each token — only 54 µs for 1500 dispatches
4. Double-buffer command buffers for pipeline overlap

This eliminates all CPU-side recording overhead from the hot path.

## RDNA4 vs RDNA3 Key Differences for Inference

| Feature | RDNA3 (gfx1100) | RDNA4 (gfx1201) | Impact on Inference |
|---|---|---|---|
| WMMA FLOPS/CU (FP16) | 512 | **1024 (2x)** | Prefill 2x faster per CU |
| WMMA register layout | Complex (inter-lane shuffle) | **Simplified (no shuffle)** | Easier shader code, fewer instructions |
| VGPR allocation | Static | **Dynamic (16/32 blocks)** | Better occupancy for high-register shaders |
| Memory ordering | In-order across waves | **Out-of-order** | Fewer false stalls on unrelated memory ops |
| Memory counters | vmcnt + lgkmcnt | **Split (loadcnt, kmcnt, dscnt)** | Finer-grained synchronization |
| Barriers | S_BARRIER (full stall) | **Split signal/wait** | Overlap barrier wait with independent work |
| Scalar FP | None | **S_FMA, S_CVT, etc.** | Uniform computations without VALU pressure |
| L0 Cache | 256 KB L1 shared | **32 KB vector + 16 KB scalar + 32 KB instruction** | Dedicated per-type caches, less contention |
| L2 Cache | 4–6 MB | **6–8 MB, 2x BW/slice** | Better hit rates for KV cache pages |
| CU count (top SKU) | 96 | **64** | Fewer CUs but each is more capable |

## Driver Configuration for Inference

### RADV (Mesa open-source Vulkan driver)

**RADV_PERFTEST flags (performance experiments):**
```bash
export RADV_PERFTEST=coop_matrix    # Enable cooperative matrix / WMMA (required on RDNA4)
export RADV_PERFTEST=cswave32       # Force wave32 for compute shaders (useful for WMMA-only shaders)
export RADV_PERFTEST=bfloat16       # Enable BF16 cooperative matrix on RDNA3/3.5
export RADV_PERFTEST=dmashaders     # Upload shader code to invisible VRAM (perf gain without resizable BAR)
export RADV_PERFTEST=sam            # Move driver objects to VRAM
```

**RADV_DEBUG flags (shader development):**
```bash
export RADV_DEBUG=shaders           # Dump compiled shaders (ACO output)
export RADV_DEBUG=shaderstats       # Print shader statistics (VGPR count, occupancy, spills)
export RADV_DEBUG=cs                # Dump compute shaders specifically
export RADV_DEBUG=spirv             # Dump SPIR-V input before compilation
export RADV_DEBUG=preoptir          # Dump ACO IR before optimization passes
export RADV_DEBUG=info              # Print GPU info at startup
export RADV_DEBUG=syncshaders       # Synchronize after every dispatch (debugging correctness)
export RADV_DEBUG=nocache           # Disable shader cache (force recompilation)
```

**RADV_PROFILE_PSTATE (power states):**
```bash
export RADV_PROFILE_PSTATE=standard   # Normal power management
export RADV_PROFILE_PSTATE=peak       # Force max clocks (WARNING: causes -23% regression on memory-bound RDNA4 work due to power throttling)
export RADV_PROFILE_PSTATE=min_sclk   # Force minimum shader clock
export RADV_PROFILE_PSTATE=min_mclk   # Force minimum memory clock
```

**Forcing wave size in Vulkan:** Use `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` to request wave32 (for WMMA shaders) or wave64 (for DMMV shaders) per pipeline.

### GPU ECC (GECC)

RDNA4 enables GPU Error Correction Codes by default, consuming ~10% memory bandwidth.

```bash
# Disable for inference (bit flips are acceptable, bandwidth is not):
# Add to /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.ras_enable=0"
# Then: update-grub && reboot
```

Measured impact: 101 tok/s → 110 tok/s (+9%) on Qwen3.5-35B-A3B Q4_K.

### SPIR-V Toolchain

| glslc Version | RADV Compatibility | Performance |
|---|---|---|
| shaderc 2023.8 (Ubuntu 24.04 system package) | Excellent | 110 tok/s |
| shaderc v2026.2-dev (custom build) | **Broken** | 19–25 tok/s (5x regression) |

Newer glslc adds `NonWritable`/`NonReadable` SPIR-V decorations and different control flow that RADV's ACO compiler cannot optimize efficiently. **Use the system package only.**

### SMU Firmware

Kernel 6.17 has SMU driver interface v0x2e, but RDNA4 firmware expects v0x32. On Radeon AI PRO R9700 this can pin sustained clocks around ~2200 MHz under compute load, below the card's advertised 2350 MHz game clock and 2920 MHz boost clock. Kernel 6.14 or earlier may work correctly.

## Official Documentation Links

### ISA Manuals (opcodes, encoding, registers)

- [RDNA4 ISA Reference (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
- [RDNA3 ISA Reference (PDF)](https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [RDNA3.5 ISA Reference (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf)
- [Machine-Readable ISA (XML)](https://gpuopen.com/machine-readable-isa/) — [GitHub](https://github.com/GPUOpen-Tools/isa_spec_manager)

### Architecture Guides

- [RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/)
- [Using Matrix Cores of RDNA4](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)
- [WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/)
- [Occupancy Explained](https://gpuopen.com/learn/occupancy-explained/)
- [AMD GPU Architecture Documentation Hub](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)

### Vulkan Extensions

- [VK_KHR_cooperative_matrix](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html)
- [GLSL_KHR_cooperative_matrix](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GLSL_KHR_cooperative_matrix.txt)
- [SPV_KHR_cooperative_matrix](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html)
- [VK_EXT_subgroup_size_control](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html) (wave32/64 control, core in Vulkan 1.3)

### Hardware Deep Dives

- [Dynamic Register Allocation on RDNA4 — Chips and Cheese](https://chipsandcheese.com/p/dynamic-register-allocation-on-amds)
- [RDNA4 Out-of-Order Memory — Chips and Cheese](https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses)
- [RDNA4 at Hot Chips 2025 — Chips and Cheese](https://old.chipsandcheese.com/2025/09/13/amds-rdna4-gpu-architecture-at-hot-chips-2025/)
- [RDNA4 Quick Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon/amd-rdna-4-quick-reference-guide.pdf)

### Driver Sources

- [AMDVLK (AMD open-source Vulkan driver)](https://github.com/GPUOpen-Drivers/AMDVLK)
- [RADV documentation (Mesa)](https://docs.mesa3d.org/drivers/radv.html)
- [Mesa feature matrix](https://mesamatrix.net/#Vulkan)
- [ROCm GPU Hardware Specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
