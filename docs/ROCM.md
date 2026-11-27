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

## Inkling 2 Mini

Start with **Inkling 2 Mini Q4_0** on the Radeon RX 9070 XT or Radeon AI PRO
R9700. The development build runs text generation at a 4,096-token context
with one request slot. The decoder gives each pass of the loop its own
attention and convolution cache. The managed download is 14.9 GB.
[Pinned checkpoint](https://huggingface.co/codegood/Inkling-2-Mini-GGUF/blob/7d95146e8e432955d714c17db9a3d9703cd53263/Inkling-2-Mini-Q4_0.gguf)

```bash
./zig-out/bin/zinc model pull inkling-2-mini-q4-0
./zig-out/bin/zinc --model-id inkling-2-mini-q4-0 --context 4096 \
  --chat --prompt "Explain how a write-ahead log works." -n 256
./zig-out/bin/zinc --model-id inkling-2-mini-q4-0 \
  --context 4096 --parallel 1 --port 9090
```

The ROCm server uses packed Q8 activations for Q4_0 projections and replays
the decode graph across tokens. Set `ZINC_INKLING_Q8=0` to use floating-point
activations for reference checks. The benchmark suite registers Inkling under
`inkling-2-mini-q4-0`. Point the suite at the directory containing the GGUF so
both engines load the same checkpoint.

```bash
bun tools/performance_suite.mjs \
  --target rdna --rdna-backend rocm \
  --models inkling-2-mini-q4-0 --runs 3 --warmup 1 \
  --rdna-workdir "$ZINC_RDNA_WORKDIR" --rdna-model-root "$ZINC_RDNA_MODEL_ROOT" \
  --rdna-vk-device "${ZINC_RDNA_VK_DEVICE:-0}" --rdna-rocm-device "${ZINC_RDNA_ROCM_DEVICE:-0}" \
  --rdna-llama-server "$ZINC_RDNA_LLAMA_SERVER_REMOTE" --rdna-llama-device "$ZINC_RDNA_LLAMA_DEVICE" \
  --rdna-keep-dpm --no-site-write --output /tmp/zinc-inkling-rdna.json
```

In the four-scenario server suite, each engine used the same GGUF and token
budgets, with one warmup and three measured runs per scenario. ZINC ROCm won
prefill and decode in every scenario on both cards. The decode ranges below
use the scenario medians. [llama.cpp v1.1.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0)

| GPU | ZINC ROCm decode | llama.cpp Vulkan decode | Reduction in total prefill and decode time |
| --- | ---: | ---: | ---: |
| Radeon RX 9070 XT | 59.7–62.4 tok/s | 50.3–50.9 tok/s | 20.7% |
| Radeon AI PRO R9700 | 63.8–66.9 tok/s | 46.0–46.2 tok/s | 28.9% |

## Qwen 3.8 NextN

Run the Qwen 3.8 GGUF directly to test its built-in drafter.

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --chat --prompt "Explain why fair LLM benchmarks need a warmup." -n 96
```

With capacity for the appended NextN block, Qwen 3.8 can draft tokens in CLI
mode and on a single-slot server. ZINC verifies the seed and drafts together
with the full 64-layer model and restores the recurrent state at the accepted
boundary. `ZINC_MTP=0` selects ordinary greedy decode. `ZINC_MTP_DRAFTS=2` is
the measured R9700 default, with `1` and `3` available for other workloads.

Start the OpenAI-compatible server:

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --parallel 1 --context 2048 --port 9090
```

The reduced-memory configurations below also use one request slot.

### Radeon RX 9070 XT with 16 GB

ROCm runs the original Qwen 3.8 27B and Muse Glimmer 30B Q4_K_M checkpoints with
a 2,048-token context and one request slot. Input embeddings stay in the
memory-mapped GGUF. ZINC packs the active decoder weights into a single VRAM
allocation and processes long prompts in batches of up to 224 tokens. In this
configuration Qwen 3.8 uses ordinary decode with the optional NextN block left
in the file mapping. Muse Glimmer reuses its existing KV cache for serving.

```bash
./zig-out/bin/zinc model list
./zig-out/bin/zinc model pull qwen38-27b-q4k-m
./zig-out/bin/zinc model pull muse-glimmer-30b-q4k-m
./zig-out/bin/zinc --model-id qwen38-27b-q4k-m --port 9090
```

On the RX 9070 XT, the benchmark harness used the same GGUF, a 2,048-token
context, and one request slot in both servers. Rates are medians from three
runs after one warmup per scenario. Each server applied its own chat template.
[llama.cpp v1.1.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0)

| Model and scenario | ZINC ROCm prefill | llama.cpp Vulkan prefill | ZINC ROCm decode | llama.cpp Vulkan decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.8 27B, Quick Chat | 209.8 tok/s | 136.9 tok/s | 32.4 tok/s | 30.5 tok/s |
| Qwen 3.8 27B, Coding Review | 519.3 tok/s | 222.7 tok/s | 32.1 tok/s | 30.3 tok/s |
| Muse Glimmer 30B, Quick Chat | 808.2 tok/s | 208.1 tok/s | 30.7 tok/s | 32.7 tok/s |
| Muse Glimmer 30B, Coding Review | 1,006.1 tok/s | 277.7 tok/s | 28.5 tok/s | 32.6 tok/s |

A separate recall check after live model switching used roughly 1,500 input
tokens. Both engines returned the exact requested code for both checkpoints.
ZINC prefills reached 658 tok/s on Qwen and 691 tok/s on Muse, compared with 356
and 350 tok/s on llama.cpp. Muse decoded at 18.3 tok/s on that long prompt,
compared with 32.2 tok/s on llama.cpp.

Pass `--rdna-context 2048` to `tools/performance_suite.mjs` to set the same
context capacity in both benchmark servers.

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
- `ZINC_MTP=0` (Qwen 3.8 CLI: disable model-native NextN drafting)
- `ZINC_MTP_Q8=0` (Qwen 3.8 CLI: use the float multi-token verifier)

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
  --runs 5 \
  --warmup 1 \
  --rdna-backend rocm \
  --rdna-build \
  --rdna-start-llama \
  --rdna-llama-server /path/to/llama-server \
  --rdna-llama-device Vulkan0 \
  --no-site-write \
  --output /tmp/zinc-rocm-qwen38.json
```

On the validated R9700 stack, the 2026-09-02 Qwen 3.8 27B four-scenario server
matrix produced these five-run medians. The comparison used llama.cpp commit
`9400c8946`, which was the tip of upstream `master` when the run started; it was
not an older historical pin. `Overall` is the comparable
prompt-plus-generation phase-wall-time score; higher is better. Each server
receives the same textual scenario prompt and applies its native chat template;
prefill and decode rates below are the server-reported phase rates.

| Scenario | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
|---|---:|---:|---:|---:|---:|
| Quick Chat | **391.76 tok/s** | 150.51 tok/s | **32.27 tok/s** | 29.83 tok/s | **122.55%** |
| Coding Review | **612.68 tok/s** | 253.09 tok/s | **31.97 tok/s** | 29.83 tok/s | **116.59%** |
| Incident Context | **680.43 tok/s** | 308.56 tok/s | **31.88 tok/s** | 29.83 tok/s | **121.97%** |
| Long Coding Draft | **426.03 tok/s** | 186.73 tok/s | **31.97 tok/s** | 29.84 tok/s | **112.11%** |

ZINC won prefill, decode, and end-to-end phase time in all four scenarios and
reached **116.88%** of llama.cpp on the summed full-matrix phase-wall-time score
(21.111 s versus 24.675 s), or **14.45% less phase wall time**. Across the four
scenarios, the prefill lead ranged from 120.52% to 160.29% and the decode lead
from 6.86% to 8.16%. All four captured ZINC previews passed the suite's output
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
