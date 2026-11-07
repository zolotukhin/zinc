# CUDA backend for ZINC — design & implementation plan

Status: **M1 kernel library COMPLETE** — all 18 decode kernels validated on the 5090 (≤7.6e-5 vs CPU ref). Next: integration (`forward_cuda.zig`). Branch: `feat/cuda-backend`.
Target hardware: NVIDIA RTX 5090 (Blackwell, sm_120) + RTX 4090 (Ada, sm_89), CUDA 13.2.

## 1. Why a CUDA backend

ZINC today has three backends: `metal` (Apple), `vulkan` (Linux/RDNA), and `zinc_rt`
(from-scratch AMD direct-submission runtime). On NVIDIA, the natural path looked like
Vulkan — but on the deployment box (Windows + WSL2) **NVIDIA exposes only CUDA to WSL2,
not Vulkan**. Proven empirically:

- The only Vulkan ICD reachable is `llvmpipe` (CPU software). Device enumeration returns
  exactly one device, `vendor=0x10005 type=CPU name=llvmpipe`.
- No NVIDIA Vulkan ICD exists in the WSL passthrough (`/usr/lib/wsl/lib` has `libcuda.so`,
  `libnvidia-ml`, `libnvidia-gpucomp`, NVENC — **no** `libGLX_nvidia`, no `nvidia_icd.json`),
  and none is installable without operator/sudo. Dozen (Vulkan-on-D3D12) is absent and
  would not run ZINC's compute shaders anyway.

So the only way ZINC's kernels touch these GPUs is a native CUDA backend.

### M0 result (validated)

A standalone `nvcc` smoke test on the box confirms the foundation:

```
device[0]: NVIDIA GeForce RTX 4090  cc=8.9  SMs=128  vram=25.8GB
vadd:  c[0]=3.0  c[N-1]=3.0  status=no error          # launch + H2D/D2H OK
dp4a:  1*5+2*6+3*7+4*8 = 70 (expect 70)                # __dp4a OK
libnvrtc.so.13  libcublas.so.13  libcudart.so.13       # runtime-compile + BLAS present
```

`__dp4a` working is the key result: ZINC's GEMMs are **not** cooperative-matrix / tensor-core
based — they use `GL_EXT_integer_dot_product` (`dotPacked4x8AccSatEXT`, = AMD `v_dot4_i32_i8`),
which maps 1:1 to CUDA `__dp4a`. The matmul port is therefore mechanical, not a rewrite.

## 2. How ZINC selects a backend (the seam we change)

Backend selection is **compile-time**, but today the discriminant is the target OS, not the
`-Dbackend` flag:

- `build.zig` — `const Backend = enum { auto, vulkan, metal, zinc_rt }` (`build.zig:3-8`);
  `selected_backend = auto ? (macos?metal:vulkan) : requested` (`build.zig:85-88`); compiled
  into `build_options.backend` as a string (`build.zig:110`) but that string is consumed only
  for `--version` (`src/build_info.zig:13`).
- `src/gpu/interface.zig` — the real dispatcher, keyed on `builtin.os.tag`:
  `is_metal = (os==.macos)`, `is_vulkan = (os==.linux)`, and
  `pub const backend = is_metal ? @import("../metal/device.zig") : @import("../vulkan/instance.zig")`
  (`interface.zig:7-28`).

Because Linux currently means Vulkan unconditionally, **CUDA cannot be OS-selected** — it
shares Linux with Vulkan. The required change: thread the `-Dbackend` choice into
`gpu/interface.zig` (a `build_options`-driven `is_cuda`) so `cuda` and `vulkan` can both
target Linux. Every `if (gpu.is_vulkan) … else …` site assumes exactly two backends and
becomes three-way: `src/main.zig:11-14, 39-49, 86-130` and
`src/server/model_manager_runtime.zig:9-12`.

## 3. The backend contract (what `src/cuda/*` must provide)

There is no vtable — the contract is a duck-typed module surface the compute layer calls
directly. **Metal is the reference to mirror** (raw-pointer binds like CUDA, and its
`commitAsync`/`wait`/`releaseCompleted` maps 1:1 onto CUDA streams+events). Vulkan is the
reference for one thing Metal doesn't need: explicit H2D/D2H staging (CUDA has no unified
memory like Apple). Mirror `src/metal/` → `src/cuda/`:

| Metal file            | CUDA equivalent     | Responsibility                                                                                                                              |
| --------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `metal/shim.h`        | `cuda/cuda_shim.h`  | C ABI contract (the backend boundary)                                                                                                       |
| `metal/shim.m` (ObjC) | `cuda/cuda_shim.c`  | Driver/Runtime API impl (`cuCtx*`, `cudaMalloc`, `cuMemcpy*`, NVRTC, `cuLaunchKernel`, streams/events)                                      |
| `metal/c.zig`         | `cuda/c.zig`        | shared `@cImport("cuda_shim.h")`                                                                                                            |
| `metal/device.zig`    | `cuda/device.zig`   | `CudaDevice.init/deinit` + caps (SM count, cc, vram)                                                                                        |
| `metal/buffer.zig`    | `cuda/buffer.zig`   | `cudaMalloc` device buffers + pinned-host staging + `upload`/`download`                                                                     |
| `metal/pipeline.zig`  | `cuda/pipeline.zig` | NVRTC compile source→PTX→`CUfunction` (or cubin load)                                                                                       |
| `metal/command.zig`   | `cuda/command.zig`  | `CudaCommand` over a `CUstream`: `dispatch`=`cuLaunchKernel`; `commitAndWait`; async `commitAsync`/`wait`/`releaseCompleted` over `CUevent` |

Required method surface (from `metal/command.zig`, `buffer.zig`, `pipeline.zig`, `device.zig`):

- **Device:** `init(allocator, device_index)`, `deinit`, caps getters (`totalMemory`,
  `maxThreadgroupMemoryLength`→`sharedMemPerBlock`, …).
- **Buffer:** `createBuffer(size)` (device), `createPrivateBuffer`, `wrapMmap` (weights — CUDA:
  `cudaHostRegister` or a staged device copy), `aliasBuffer(base, off, size)`, `freeBuffer`,
  `upload(data)`/`download(dst)` (explicit, via pinned staging).
- **Pipeline:** `createPipeline(src, fn_name)` (NVRTC), `createPipelineFromLib(cubin,…)`,
  `freePipeline`, introspection (`maxThreadsPerBlock`, `sharedMem`).
- **Command:** `beginCommand(ctx)`, `dispatch(pipe, grid, block, bufs, push_data, push_size)`
  (→ `cuLaunchKernel` with a packed-args/`__constant__` push block), `barrier*` (CUDA: same
  stream is implicitly ordered; cross-stream → events), `commitAndWait`, and the async trio
  `commitAsync`/`wait`/`releaseCompleted`.

The async ring that overlaps GPU exec with CPU command-building lives in the compute layer
(`forward_metal.zig:19816-19882`, `[256]MetalCommand` pending ring at `:20104`) and must be
mirrored in `forward_cuda.zig` using `CUstream` + `CUevent`.

## 4. Kernel port strategy

The Vulkan backend is **110 `.comp` shaders** — the authoritative spec for the CUDA kernel
set. Categories: GEMM/dequant ~58, MoE routing 10, SSM 9, Norm/act 13, Attention 4, Quantize
4, Elementwise 6, RoPE 2, KV 2. Mapping cheat-sheet (the whole port is variations of this):

| Vulkan / GLSL                                                | CUDA                                                                                                      |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `dotPacked4x8AccSatEXT` (int8 dot)                           | `__dp4a`                                                                                                  |
| wave64 workgroup (`local_size_x=64`)                         | warp32 block; use the **existing wave32 fallback path** (cross-subgroup shared-mem merge) as the template |
| `subgroupAdd` / `subgroupClusteredAdd`                       | `__shfl_down_sync` reductions / `__reduce_add_sync`                                                       |
| `subgroupMax` / `subgroupShuffle` / `subgroupBroadcastFirst` | `__reduce_max_sync` / `__shfl_sync`                                                                       |
| `subgroupBallot` / `subgroupElect`                           | `__ballot_sync` / `__activemask`                                                                          |
| `vec4`/`uvec4` SSBO (128-bit loads)                          | `float4`/`int4`                                                                                           |
| push constants                                               | kernel params or `__constant__` block                                                                     |
| specialization constants (K=2048/4096/12288, NUM_ROWS)       | template params or runtime-tuned variants                                                                 |
| `vkCmdDispatchIndirect`                                      | `cudaLaunchKernel` host indirection (or device launch)                                                    |

GGML quant block layouts (Q4_K/Q5_K/Q6_K 256-elem super-blocks; Q8_0/Q5_1/Q5_0 32-elem;
MXFP4 32-elem) are identical bit-for-bit, so the in-shader dequant unpack ports verbatim.

NVRTC vs. offline `nvcc`: start with **NVRTC runtime compilation** (mirrors Metal's
`createPipeline(msl_source)`; kernels live as `.cu`/string sources, compiled to PTX for the
running GPU's arch on load — handles sm_120 vs sm_89 transparently). An offline `nvcc → cubin`
build step (parallel to the glslc shader step in `build.zig`) is a later optimization.

Hardest kernels (port last / carefully): tiled `mul_mm_*` + `*_full_dp4a` GEMMs (register
blocking, LDS bank padding); `ssm_delta_net[_cols8]` (autoregressive selective scan with
register-resident per-row state + clustered subgroup reductions); `flash_attn` (paged KV,
split-K, attention sinks); `softmax_topk` + route-pack/indirect-dispatch chain.

## 5. The target model & milestone ordering

Target: `qwen36-35b-a3b-q4k-xl` ("Qwen3.6 35B-A3B", ~18 GiB) — internal arch `.qwen2_moe` **with
SSM**. Layer pattern: `is_full_attn = ((layer+1) % full_attention_interval==0)`, interval=4 →
**3 of every 4 layers are delta-net SSM, every 4th is attention**; every layer has an MoE FFN
(top-k routed experts + one sigmoid-gated shared expert). Decode driven by
`forward.zig:decodeStep (:5585)`, SSM core `runSsmLayerGpu (:12566)`, MoE `:7366-9024`,
tail `:9501-9645`.

Quant mix (UD-Q4_K_XL): experts gate/up **Q4_K**, experts down **Q5_K**, shared-expert + SSM
in/out **Q8_0**, norms/α/β/router **F32**, conv1d + A_log **F16**, LM head **Q6_K**.

Milestones:

- **M0 — toolchain (DONE).** `nvcc` + `__dp4a` + NVRTC/cuBLAS validated on the box.
- **M0.5 — primitive layer (DONE, validated on the 5090 sm_120).** Both the C core
  (`cuda_shim.h/.c`) and the **Zig wrappers** (`src/cuda/{c,device,buffer,pipeline,command}.zig`,
  mirroring `src/metal/*`) are written and pass via `smoke.c` _and_ `smoke.zig`: device select
  (picks the 5090), staged buffers + H2D/D2H, NVRTC runtime compile, the buffers+push dispatch
  ABI, sync + async (`commitAsync`/`wait`) commit, and `__dp4a`. Use the pinned
  Zig 0.17.1 wrapper from the repository root:
  `./scripts/zig.sh build cuda-smoke -Dbackend=cuda -Dshaders=false`.
  This target builds and runs the smoke test on the designated NVIDIA host.
  For a host-only compiler check without a GPU, use
  `./scripts/zig.sh build check-cuda`. Minor follow-up:
  add `cuda_device_count()` to the shim to drop the noisy enumeration probe.
- **M1 — single correct token.** `forward_cuda.zig` minimal decode for one qwen36-35b token.
  ~14 kernels (unfused reference variants), in forward order:
  1. embedding dequant (host gather) → 2. `rms_norm_mul` → 3. one templated **DMMV** covering
     Q4_K/Q5_K/Q6_K/Q8_0/F32 + accumulate mode (Q/K/V/O, SSM in/out, router, LM head) →
  2. attention: qk per-head rmsnorm + RoPE (partial/IMRoPE) + `kv_cache_write` (paged) +
     `flash_attn` (or naive softmax(QKᵀ)V for 1 query) → 5. SSM: `ssm_conv1d` + `ssm_delta_net`
     (unfused scalar) + `ssm_gated_norm` → 6. MoE: `softmax_topk` + `dmmv_q4k_moe` (gate/up) +
     `swiglu` + `dmmv_q5k_moe` (down) + `moe_weighted_acc` + shared expert (Q8_0 + `sigmoid_scale_acc`)
     → 7. `scale_accumulate` residuals → 8. `argmax`.
     Validate token-for-token against the Vulkan/Metal reference on the same prompt.
  - **Kernel library in progress** — `src/shaders/cuda/kernels.cu`, each validated numerically
    against an independent CPU reference on the 5090 via `src/cuda/kernels_test.c` (run from
    `~/cuda_proto`). Done (all `max_rel_err ≤ ~1e-5`): `rms_norm`, `dmmv_q4k` (canonical GGML
    Q4_K dequant + matvec, cross-checked vs llama.cpp `get_scale_min_k4`), `dmmv_f32`, `dmmv_q8_0`,
    `dmmv_q5k`, `dmmv_q6k`, `swiglu`, `scale_accumulate`, `sigmoid_scale_acc` (9 done — the full
    dequant-matvec set f32/Q4_K/Q5_K/Q6_K/Q8_0 + norm + FFN/residual/gating elementwise).
    Plus `softmax_topk` (router), `rope`, `argmax` (exact), `moe_weighted_acc` (4.4e-5) —
    **13 done**; `qk_norm` reuses `rms_norm` per-head (no new kernel). The entire
    non-attention/non-SSM decode path is complete. Remaining: `kv_cache_write`, naive attention
    (`softmax(QKᵀ)V` for the 1-query decode), and `ssm_delta_net` (the one hard SSM kernel left). **Now 15 done** (added `ssm_conv1d`
    exact-state + `ssm_gated_norm`) — **ALL 18 DONE** (added `kv_cache_write`, `naive_attention` GQA+sinks, and
    `ssm_delta_net` — the gated delta-net selective scan, validated multi-token incl. final
    recurrent state at 7.6e-5). **The M1 kernel library is COMPLETE.** (MoE-indexed dmmv variants
    - flash/paged attention are M2/M3 perf items.)
- **M2 — full model, coherent generation.** Paged KV pool, SSM conv/recurrent state ring
  buffers (conv `(d_conv-1)*8192` f32/layer; recurrent `dt_rank*128*128` f32/layer), all quant
  formats, prompt prefill, chat template, server path (`model_manager_cuda.zig`). Correctness
  parity with Metal/Vulkan output.
- **M3 — performance.** Fused kernels (`dmmv_*_fused_gate_up*`, `*_moe_fused_down_acc`,
  `qk_norm_rope_kv_write`, `rms_norm_dmmv_*`), the async stream/event pending-command ring,
  DP4a tiled GEMMs (and optionally tensor-core `mma.sync` GEMMs as a 5090-specific win),
  Blackwell tuning (warp32, occupancy, sm_120). Benchmark decode/prefill tok/s vs. llama.cpp
  CUDA on the same model.

## 6. Build integration

1. `build.zig`: add `cuda` to `Backend` (`:3-8`); add `configureCudaModule(b, target, module)`
   that adds `src/cuda/cuda_shim.c`, includes `/usr/local/cuda/include`, links
   `cudart`/`cuda`/`nvrtc` (and `cublas` later) with `-L/usr/local/cuda/lib64`; in selection,
   allow `cuda` only on Linux (coexists with vulkan via the explicit `-Dbackend=cuda`).
2. Add a standalone `cuda-smoke` exe/step (`src/cuda/smoke.zig` + `cuda_shim.c`) gated to Linux
   — builds & runs independently of the main `zinc` exe and the `gpu/interface.zig` dispatch,
   so the primitive layer can be validated before `forward_cuda` exists.
3. `src/gpu/interface.zig`: introduce `is_cuda` from `build_options.backend` and route
   `backend`/`buffer_mod`/`pipeline_mod`/`command_mod` to `../cuda/*` when set. Make the
   `main.zig` / `model_manager_runtime.zig` branches three-way.
4. `src/runtime_assets.zig`: add a `cuda`/`ptx` `ShaderKind` + candidate dirs so compiled
   kernels (if offline-built) resolve; the NVRTC path reads `.cu` sources from `src/shaders/cuda/`.

## 7. Prerequisites & open questions

- **Repo on the box:** ZINC is public (`https://github.com/zolotukhin/zinc`) — clone it
  read-only over HTTPS into `~/workspace`, no GitHub auth or token needed. (A token is only
  needed to push changes back from the box.)
- **Zig 0.17.1:** use `./scripts/zig.sh` from the repository root. It downloads
  and verifies the pinned release when needed, without replacing the system compiler.
- **Device order:** set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (cuda:0=5090, cuda:1=4090) or select by
  cc/SM count at runtime; non-login SSH does not inherit it.
- **24 GB cgroup cap** applies to host RAM, not VRAM — the 18 GiB model loads fine into the
  5090's 32 GB, but host-side staging/mmap must stay under 24 GB (stream weights to device).
- **wave64→warp32 numerics:** reductions change width; validate each ported kernel against the
  reference, not just compile.
- **zinc_rt `t_cuda` Tier** (`zinc_rt/engine.zig:14-21`) is a _different_ integration (direct
  submission, no CUDA driver) — out of scope; this plan uses the cudart/Driver API path.

## 8. Reference index

- Backend dispatch: `src/gpu/interface.zig`; build: `build.zig:3-117`.
- Metal reference backend: `src/metal/{shim.h,shim.m,c.zig,device.zig,buffer.zig,pipeline.zig,command.zig}`;
  async ring `src/compute/forward_metal.zig:19816-20104`.
- Vulkan kernel spec: `src/shaders/*.comp` (110), hosts `src/compute/{dmmv,attention,elementwise,argmax}.zig`.
- Model forward pass: `src/compute/forward.zig:5585` (decode), `:12566` (SSM), `:7366` (MoE), `:9501` (tail);
  config `src/model/config.zig:55-84`, loader `src/model/loader.zig:180-440`, catalog `src/model/catalog.zig:64-88`.
