# ROCm backend

ZINC has a supported Linux ROCm backend for AMD GPUs. It reuses the CUDA-side
model loader and scheduler behind the same C ABI, implemented with HIP, hipRTC,
and hipBLAS. Vulkan and ROCm are independently supported AMD paths; select ROCm
explicitly at build time.

For prebuilt packages, see the [binary release guide](https://zolotukhin.ai/zinc/docs/binary-releases/).
They remove the Zig/source-build requirement. HIP still compiles embedded kernels
at runtime, so install matching HIP/hipRTC/hipBLAS development files and rocWMMA
headers alongside the supported ROCm userspace. The standalone ROCm server
provides CLI/HTTP chat and completion routes; it does not serve the Metal/Vulkan
browser UI or hot model switching.

## Validated stack

The current reference configuration is:

- Linux kernel `7.2.2-070202-generic`
- ROCm userspace `7.2.4`
- Radeon AI PRO R9700 (`gfx1201`, wave32)
- Zig `0.17.0`, pinned by `.zig-version` and selected by `scripts/zig.sh`
- Qwen 3.5 9B, Qwen 3.6 35B-A3B, and Qwen 3.8 27B
- Qwen 3.6 dense 27B at 4K in the [packaged prefill follow-up](ROCM_QWEN27_PREFILL.md)
- Gemma 4 26B-A4B MoE and Gemma 4 31B

Kernel `7.2.2` is the validated reference, not a build-time requirement. The
running `amdgpu` driver, ROCr runtime, and ROCm userspace must agree well enough
for `rocminfo` and a HIP device query to expose the target GPU.

## Build and verify

Install the ROCm HIP development files that provide `amdhip64`, `hiprtc`, and
`hipblas`, then build with the toolkit root:

```bash
ROCM_PATH=/opt/rocm ./scripts/zig.sh build -Dbackend=rocm -Doptimize=ReleaseFast

ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
./zig-out/bin/zinc --version
```

The executable records the toolkit's `lib` directory as a runtime search path.
The standard `/opt/rocm` installation needs no `LD_LIBRARY_PATH` setup. If you
move a custom toolkit after building, rebuild against its new location or add
that toolkit's `lib` directory to `LD_LIBRARY_PATH`.

Run a GGUF directly:

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --chat --prompt "Explain why fair LLM benchmarks need a warmup." -n 96
```

Qwen 3.8 GGUFs that include the model's appended NextN block use it
automatically. ZINC drafts three tokens, verifies the seed and drafts together
with the full 64-layer model, and restores the recurrent state at the accepted
boundary when a draft is rejected. Only tokens the full model agreed with are
emitted, so output is identical to ordinary greedy decode — on the R9700 it
measures byte-for-byte equal with speculation on and off. `ZINC_MTP=0` returns
to ordinary greedy decode. `ZINC_MTP_DRAFTS=3` is the measured R9700 default
(a four-row verify costs about 4% more than a three-row one);
`1` and `2` remain available for GPUs or workloads with a different acceptance
profile, and `ZINC_MTP_ADAPT_C=<c>` picks two or three per cycle from the
running acceptance.

On ROCm, the draft head scores 98,304 vocabulary rows by default; the target
still verifies against the complete vocabulary. `ZINC_MTP_DRAFT_VOCAB=65536`
restores the previous draft-head size. MTP verification also fuses attention
setup across its token rows; `ZINC_MTP_FUSED_ATTN=0` restores the separate
operations. The [Zig 0.17 MTP follow-up](https://github.com/zolotukhin/zinc/blob/main/docs/ROCM_MTP_0_17.md) records the matched
three-round R9700 results: 71.17 to 77.04 tok/s on the long coding workload,
with roughly 1–1.4% gains on the other three workloads.

The server uses it too, but only when it owns a single request slot
(`--parallel 1`): drafting writes the single-sequence KV cache and rewinds it on
a rejected draft, which needs the whole sequence, while multi-slot serving keeps
the batched loop that amortizes weights across concurrent requests. On the
R9700 with Qwen 3.8 27B Q4_K_M, single-slot serving measures 64-73 tok/s decode
on the suite prompts against 32 tok/s with `ZINC_MTP=0` (llama.cpp HIP: 29 tok/s),
at 50-65% acceptance of three drafts.

Start the OpenAI-compatible server:

```bash
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --parallel 1 --context 2048 --port 9090
```

Gemma 4 26B-A4B currently serves through its single-sequence MoE path, so ZINC
uses one request slot for that model even if a larger `--parallel` value is
given. The other validated models keep the requested slot count.

Start with `--parallel 1` on a memory-limited GPU. Single-slot Qwen and Gemma
servers reuse the existing KV cache; multiple slots require separate caches.
Gemma 31B Q4_K_M passed the packaged 4K/F32 checks on the 32GB R9700 after
removing a duplicate single-slot cache. Larger contexts and additional slots
still need their own memory budget and validation.

Short dense hybrid Qwen CLI prompts use per-token prefill below 32 tokens;
long prompts remain batched. Prepared MTP retains the batched hidden rows needed
for priming. The debug harness shares this routing policy. See the
[short-prompt agreement checks](ROCM_QWEN27_PREFILL.md) for the regression tests
and packaged evidence.

Single-slot F32 serving also aligns short MoE prefill with CLI. Cached scratch
from an earlier request cannot enable packed decode on a later serial-prefilled
request. See the [single-slot consistency checks](ROCM_SINGLE_SLOT_CORRECTNESS.md)
for the controls, recovery regressions and complete ROCm model sweep.

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
- `ZINC_MTP=0` (Qwen 3.8 CLI and single-slot server: disable model-native NextN drafting)
- `ZINC_MTP_Q8=0` (Qwen 3.8: use the float multi-token verifier)
- `ZINC_MTP_FUSED_ATTN=0` (Qwen 3.8: use separate attention setup operations during verification)

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

## API client notes

The ROCm server exposes generation, `/health`, `/stats`, and a minimal
`GET /v1/models` list for its loaded model. Discovery includes only the model's
basename, fixed ownership/creation fields, and per-slot context capacity; it
does not expose parent directories. It does not provide the full Vulkan/Metal
model-manager or browser chat interface. Use the default Vulkan or Metal build
for the browser-chat walkthrough. Direct GGUF and managed-ID startup, discovery,
generation, and all 23 API checks passed after the
[test GPU recovered](ROCM_AGENT_CONTEXT.md#remaining-hardware-validation).

Send `stream: false` explicitly for a JSON response; this handler defaults to
streaming. SSE replies now include finish reasons and usage. Qwen chat supports
tool schemas, streamed tool calls, and assistant/tool-result history, including
null assistant content. The `stop` parameter remains unimplemented. The OpenAI JavaScript SDK 7.43.0 passes matched
JSON and streaming generation checks with the explicit stream setting. See the
[MTP validation report](https://github.com/zolotukhin/zinc/blob/main/docs/ROCM_MTP_0_17.md)
for the earlier parity tests and generic SDK-suite limitation.

The [OpenCode harness guide](OPENCODE.md) provides a portable setup script and
headless task/API checks. For repository tasks, start the server with
`--context 32768 --parallel 1` and pass the same context to the setup script.
The recurrent prefill scan stages its
gate/beta values in bounded tiles, allowing agent prompts above the previous
8K shared-memory limit. Streaming clients can cancel a request without keeping
its slot occupied for the rest of the output budget.

### Long coding sessions

The initial fixture checks used a 16K server context and did not validate
requests near its full capacity. A full Zinc worktree experiment exposed limits
beyond those fixtures: the 27B model ran out of
VRAM during prefill with a 32K allocation, and the 9B model's decode attention
kernel exceeded the GPU's shared-memory limit once its prompt crossed 16K.
That kernel failure produced corrupted text even though the HTTP request and
OpenCode process completed successfully. A later 27B run at 16K also failed:
`attention_causal_batched_v2` requested a 65,612-byte group segment at 16,080
prompt tokens, followed by illegal-memory and rocBLASLt/Tensile errors.
The [follow-up fixes and validation](ROCM_AGENT_CONTEXT.md) replace the full-score
attention allocation above 8K with bounded tiles, chunk prefill scratch, and
remove duplicate single-slot state. The 27B model now passes exact-key probes
at 8,548, 17,048, and 30,048 prompt tokens with a 32K context and NextN enabled.
GPU execution failures also fail generation and health checks explicitly.
These execution checks remain separate from repository coding acceptance.

Prefill defaults to chunks of 4,096 tokens. `ZINC_PREFILL_CHUNK=<positive count>`
changes the scratch bound; larger chunks consume more VRAM. Optional
`ZINC_PROMPT_CACHE=1` reuses exact prefixes for single-slot Qwen requests by
restoring recurrent checkpoints. It defaults off and requires additional VRAM;
see the follow-up for parity checks and the portable cache validator.

The 32K coding configuration is not Qwen 3.8 27B's model limit: its native
window is 262,144 tokens. The [native-context guide](ROCM_NATIVE_CONTEXT.md)
documents opt-in Q8 KV storage, bounded BLAS prefill and the separate FP16-input
attention option for fitting larger capacities on a 32GB card. F32 KV remains
the default. Q8 currently requires ROCm Qwen and a single request slot.

Keep repository instructions concise and read source in small line ranges.
OpenCode 1.19.4's [built-in file reader](https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/tool/read.ts) has an independent 50KB cap, which can
overflow a local model's context when reading large files. Its general
`tool_output` setting does not constrain that reader, as the
[tool wrapper](https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/tool/tool.ts) skips results already marked by the reader. The setup helper now installs
a local hook that caps reads at 80 lines and shortens large tool outputs.
Rerun setup with `--force` to update an existing configuration. Keep checking the worktree
diff, server log, and project tests independently. The setup script's
`run --verify '<project check>'` mode enforces checks between short editing
rounds and returns failure when they do not pass.

## Fair llama.cpp comparison

Use the reusable-server performance suite for claims. It runs one engine at a
time, uses the same GGUF and scenario matrix, discards a warmup, and reports the
median of measured runs:

The RDNA runner applies the same host policy before either engine: PCIe ASPM
is set to performance and the discrete GPU uses its automatic clock policy.
Warmups and repeated runs account for clock ramping.

ROCm preflight uses `rocminfo` with ROCr visibility. The legacy
`--rdna-vk-device` option selects a ROCr ordinal for the ROCm backend; Vulkan
may enumerate the same card at a different index. The selected GPU appears as
logical device zero to ZINC and the HIP baseline. Both servers use the same
context capacity: the model setting or 4,096 by default, with
`--rdna-context` to override it.

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --models qwen38-27b-q4k-m \
  --runs 5 \
  --warmup 1 \
  --rdna-backend rocm \
  --rdna-context 32768 \
  --rdna-build \
  --rdna-llama-server /path/to/llama.cpp/build-hip/bin/llama-server \
  --rdna-llama-device ROCm0 \
  --no-site-write \
  --output /tmp/zinc-rocm-qwen38.json
```

Build that llama.cpp with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201` at the same
commit as the Vulkan baseline, so the ROCm tab compares ROCm with ROCm. The
current numbers, with both engines reading identical prompt tokens, are on the
[benchmark dashboard](https://zolotukhin.ai/zinc/benchmarks/#rdna-rocm).

The table below is a historical record. It was measured before the harness
sent both engines the same system turn and thinking switch, so each server
applied its own chat template and the prompt token counts differed. On the
validated R9700 stack, the 2026-09-02 Qwen 3.8 27B four-scenario server
matrix produced these five-run medians. The comparison used llama.cpp commit
`9400c8946`, which was the tip of upstream `master` when the run started; it was
not an older historical pin. `Overall` is the comparable
prompt-plus-generation phase-wall-time score; higher is better. Each server
receives the same textual scenario prompt and applies its native chat template;
prefill and decode rates below are the server-reported phase rates.

| Scenario          |     ZINC prefill | llama.cpp prefill |     ZINC decode | llama.cpp decode |     Overall |
| ----------------- | ---------------: | ----------------: | --------------: | ---------------: | ----------: |
| Quick Chat        | **391.76 tok/s** |      150.51 tok/s | **32.27 tok/s** |      29.83 tok/s | **122.55%** |
| Coding Review     | **612.68 tok/s** |      253.09 tok/s | **31.97 tok/s** |      29.83 tok/s | **116.59%** |
| Incident Context  | **680.43 tok/s** |      308.56 tok/s | **31.88 tok/s** |      29.83 tok/s | **121.97%** |
| Long Coding Draft | **426.03 tok/s** |      186.73 tok/s | **31.97 tok/s** |      29.84 tok/s | **112.11%** |

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
