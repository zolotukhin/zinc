# ROCm backend

ZINC has a supported Linux ROCm backend for AMD GPUs. It reuses the CUDA-side
model loader and scheduler behind the same C ABI, implemented with HIP, hipRTC,
and hipBLAS. Vulkan and ROCm are independently supported AMD paths; select ROCm
explicitly at build time.

## Validated stack

The current reference configuration is:

- Linux kernel `7.2.2-070202-generic`
- ROCm userspace `7.2.4`
- Radeon AI PRO R9700 (`gfx1201`, wave32)
- Zig `0.15.2` or newer
- Qwen 3.5 9B, Qwen 3.6 35B-A3B, and Qwen 3.8 27B
- Gemma 4 26B-A4B MoE and Gemma 4 31B

Kernel `7.2.2` is the validated reference, not a build-time requirement. The
running `amdgpu` driver, ROCr runtime, and ROCm userspace must agree well enough
for `rocminfo` and a HIP device query to expose the target GPU.

## Build and verify

Install the ROCm HIP development files that provide `amdhip64`, `hiprtc`, and
`hipblas`, then build with the toolkit root:

```bash
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast

ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
./zig-out/bin/zinc --version
```

If the ROCm libraries are not registered with the system dynamic linker, add
`/opt/rocm/lib` to `LD_LIBRARY_PATH` for the build and run commands.

Run a GGUF directly:

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --chat --prompt "Explain why fair LLM benchmarks need a warmup." -n 96
```

Start the OpenAI-compatible server:

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --parallel 1 --context 2048 --port 9090
```

Gemma 4 26B-A4B currently serves through its single-sequence MoE path, so ZINC
uses one request slot for that model even if a larger `--parallel` value is
given. The other validated models keep the requested slot count.

The ROCm-tuned prefill and decode fusions are default-on. These switches remain
available as correctness/performance A/B opt-outs:

- `ZINC_BATCHED_PREFILL=0`
- `ZINC_PREFILL_Q8=0`
- `ZINC_PREFILL_Q8_REUSE=0`
- `ZINC_PREFILL_WMMA_TILES=0`
- `ZINC_PREFILL_WMMA_T80=0`
- `ZINC_PREFILL_WMMA_TAIL=0`
- `ZINC_PREFILL_WMMA_Q4_DIRECT=0`
- `ZINC_QWEN_MOE_BATCHED=0`
- `ZINC_ATTN_V2=0`
- `ZINC_SSM_PREPARED=0`
- `ZINC_SSM_COL_WARP=0`
- `ZINC_SSM_COL_WARP_FAST=0`
- `ZINC_ROCM_FUSED_FFN=0`
- `ZINC_ROCM_FUSED_SSM_F32=0`
- `ZINC_ROCM_FUSED_Q4_PAIRS=0`
- `ZINC_ROCM_FUSED_ATTN_FRONTEND=0`
- `ZINC_ROCM_DECODE_PAIR_REDUCE=0`
- `ZINC_ROCM_Q4_PAIR_REDUCE=0`
- `ZINC_ROCM_DECODE_Q8_FFN=0`
- `ZINC_ROCM_DECODE_Q8_Q6=0`
- `ZINC_ROCM_DECODE_Q8_LM=0`
- `ZINC_ROCM_DECODE_Q8_Q4=0`
- `ZINC_ROCM_DECODE_Q8_Q6_PROJ=0`
- `ZINC_ROCM_DECODE_Q8_Q5=0`
- `ZINC_ROCM_DECODE_Q8_Q4_PAIR=0`
- `ZINC_ROCM_RMS_Q8=0`
- `ZINC_ROCM_ARGMAX_V2=0`
- `ZINC_ROCM_DECODE_SSM_COL_WARP=0`
- `ZINC_ROCM_DECODE_SSM_FAST=0`
- `ZINC_BATCH_B1_MATVEC=0`

The Qwen 3.8 decode path uses packed Q8 activations for the hot Q4_K, Q5_K,
and Q6_K projections, paired projection kernels where the shapes permit it, a
prepared wave32 column scan for the recurrent SSM state, an ILP-unrolled
single-query attention kernel, and a two-stage GPU argmax. RMS normalization
and Q8 activation packing share one wide per-token kernel in prefill and decode,
avoiding a second activation read and launch. Shape-specific WMMA tiles cover
short prompts and long-prompt tails without padding them to a substantially
larger tile. All are ROCm defaults only; setting the corresponding variable to
`0` restores the preceding path for comparison.

Qwen 3.6 MoE prefill uses the native token-batched Q4_K/Q5_K/Q6_K expert
kernels. CUDA's grouped tensor-core expert path is not dispatched by the ROCm
build; its compatibility symbols are intentionally inert.

## Fair llama.cpp comparison

Use the reusable-server performance suite for claims. It runs one engine at a
time, uses the same GGUF and scenario matrix, discards a warmup, and reports the
median of measured runs:

The RDNA runner applies the same stable host policy before either engine: PCIe
ASPM is set to performance and the discrete GPU's memory DPM policy is locked
high. This prevents an idle server from changing clocks between phases.

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --models qwen38-27b-q4k-m \
  --runs 3 \
  --warmup 1 \
  --rdna-backend rocm \
  --rdna-build \
  --rdna-start-llama \
  --rdna-llama-server /path/to/llama-server \
  --rdna-llama-device ROCm0 \
  --no-site-write \
  --output /tmp/zinc-rocm-qwen38.json
```

On the validated R9700 stack, the 2026-08-31 Qwen 3.8 27B four-scenario server
matrix produced these medians. `Overall` is the comparable
prompt-plus-generation phase-wall-time score; higher is better. Each server
receives the same textual scenario prompt and applies its native chat template;
prefill and decode rates below are the server-reported phase rates.

| Scenario | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
|---|---:|---:|---:|---:|---:|
| Quick Chat | **377.54 tok/s** | 279.74 tok/s | **27.87 tok/s** | 24.48 tok/s | **118.49%** |
| Coding Review | **591.11 tok/s** | 464.26 tok/s | **27.48 tok/s** | 24.32 tok/s | **114.43%** |
| Incident Context | **655.74 tok/s** | 600.81 tok/s | **27.34 tok/s** | 24.38 tok/s | **112.90%** |
| Long Coding Draft | **417.17 tok/s** | 331.18 tok/s | **27.56 tok/s** | 24.29 tok/s | **114.92%** |

ZINC won prefill, decode, and end-to-end phase time in all four scenarios and
reached **114.89%** of llama.cpp on the summed full-matrix phase-wall-time score
(24.385 s versus 28.016 s), or **12.96% less phase wall time**. Across the four
scenarios, the prefill lead ranged from 9.14% to 34.96% and the decode lead from
12.15% to 13.84%. All four captured ZINC previews passed the suite's output
quality checks.

## Troubleshooting

- `hipErrorNoBinaryForGpu`: verify hipRTC receives the detected `gfx*` target and
  that the installed ROCm release supports it.
- No device or the wrong device: set `ROCR_VISIBLE_DEVICES` and pass `-d 0` after
  visibility filtering.
- Missing shared library: register `/opt/rocm/lib` with the dynamic linker or set
  `LD_LIBRARY_PATH`.
- Performance drift: stop other GPU processes, discard at least one warmup, and
  compare persistent server processes rather than a cold CLI launch.
