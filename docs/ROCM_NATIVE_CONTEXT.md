# Native context on the ROCm Qwen path

Qwen 3.8 27B has a native **262,144-token** window in its
[official configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
The 32K capacity in the earlier [coding trial](ROCM_AGENT_CONTEXT.md) was an
explicit server/test setting. It was not the model's architectural limit or
the largest capacity the engine could allocate.

The opt-in ROCm Qwen cache format `ZINC_KV_CACHE=q8` stores 32 values with one
FP32 scale in each 36-byte block. It changes the runtime KV representation;
the GGUF file and weight quantization stay the same. F32 remains the default.
This first implementation supports a single request slot and head dimensions
divisible by 32. CUDA hardware, other model families, and multi-slot Q8 serving
are outside this validation.

Build the [ROCm backend](ROCM.md#build-and-verify), then run on the GPU host:

```bash
ZINC_KV_CACHE=q8 ZINC_Q8_ATTN_F16=1 ZINC_MTP=1 ZINC_PROMPT_CACHE=0 \
  ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set your visible GPU selector}" \
  ./zig-out/bin/zinc -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --context 262144 --parallel 1 --port 9090
```

The FP16 switch explicitly selects the faster precision mode described below.
Omit it for FP32-input attention with Q8 cache storage. Match the coding client's
capacity with `bun tools/setup_opencode.mjs --context 262144 --project ../my-project`,
or use setup-only `--context auto` to read the live server's advertised capacity.
The [repository end-to-end report](OPENCODE_E2E_REPORT.md) records actual OpenCode
coding requests above 32K, separately from the near-native retrieval below.
The window includes input and generated output, so leave space for the answer.
OpenCode may compact history earlier to reserve room for output and tool results.
At this capacity, the current helper previews a 4,096-token output limit and
69,632-token compaction reserve. That reserve is a client budget within the
server's total window.

## Memory accounting

The base 27B model has 16 full-attention layers, four KV heads per layer and
256 dimensions per head. There are two caches, K and V:

```text
F32 bytes = tokens × 16 layers × 2 × 4 heads × 256 dimensions × 4
Q8 bytes  = tokens × 16 layers × 2 × (4 × 256 / 32) × 36
```

| Allocated context | Base F32 KV | Base Q8 KV |
| ----------------: | ----------: | ---------: |
|            32,768 |       4 GiB |  1.125 GiB |
|            65,536 |       8 GiB |   2.25 GiB |
|           131,072 |      16 GiB |    4.5 GiB |
|           262,144 |      32 GiB |      9 GiB |

The appended NextN attention block adds another 1/16 of the base KV allocation:
at native capacity, **34 GiB total F32 KV versus 9.5625 GiB total Q8 KV**.
Weights, recurrent state, speculative checkpoints, optional prompt checkpoints
and prefill scratch are additional allocations. A model's supported window,
configured server capacity and actual prompt length are different quantities.

## Bounded prefill and decoding

Cache writes happen after K normalization and RoPE. Each block uses a symmetric
scale based on its maximum magnitude and rounds to signed 8-bit values.
Attention reconstructs values while reading the cache; it never allocates a
second full F32 copy. Slot aliases use the active row size, including scales.
NextN catch-up and speculative verification use the same format.

Long-context decoding retains tiled online softmax. Prompt attention defaults
to BLAS for batches of at least 256 queries. It packs queries by head, expands
one 1,024-key Q8 tile at a time, computes QKᵀ, updates the running softmax and
accumulates PV. The score buffer is bounded by query chunk size × heads ×
1,024, rather than query chunk size × total context. GQA expansion is temporary
and bounded by that key tile. No second full F32 cache is allocated.

Smaller batches use scalar tiled kernels. For 256-dimensional prompt attention,
32 queries share a compressed 128-key tile in GPU shared memory. Four queries
per warp reuse reconstructed K/V values. An eight-query path and the general
tiled kernel cover smaller batches, tails and other supported head dimensions.
Prompt scratch is bounded by `ZINC_PREFILL_CHUNK`, defaulting to 4,096.

| Setting              | Default         | Effect                                                                                                      |
| -------------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `ZINC_KV_CACHE`      | `f32`           | `q8` enables the compressed ROCm Qwen cache                                                                 |
| `ZINC_Q8_ATTN_BLAS`  | `1`             | Set `0` to compare scalar Q8 prefill                                                                        |
| `ZINC_Q8_ATTN_F16`   | `0`             | Set `1` for FP16 matrix inputs and probabilities; scores, online-softmax state and accumulation remain FP32 |
| `ZINC_PREFILL_CHUNK` | `4096`          | Bounds prompt scratch; larger chunks use more VRAM                                                          |
| `ZINC_PROMPT_CACHE`  | backend default | The native-capacity validation explicitly disables prefix checkpoints with `0`                              |

## Precision is an explicit choice

Q8 is lossy. The validator downloads the production GPU writer's blocks and
checks guards, scales and rounding bounds before using their reconstructed
values in an independent double-precision attention reference. The FP32-input
BLAS path and scalar paths must produce finite results with error at most
`0.0003 * (1 + abs(reference))`.

FP16 matrix inputs introduce another precision change. The first mixed-precision
trial failed the original FP32-reference guard. That guard was not relaxed.
Instead, the mode remains off by default and is checked against a separately
declared reference: IEEE FP16 Q/K/V and unnormalized probabilities, with FP32
softmax state and accumulation. It passes the same tolerance against that
reference. Against FP32 matrix inputs over the same reconstructed Q8 cache,
24,064 sampled outputs had
**0.000183 RMSE** and **0.00296 maximum absolute error**. These are measurements
of the sampled kernels, not a bound on model quality or all possible inputs.

A Q8 response need not match F32 byte for byte. Tool calls, streaming, retrieval
and repository coding require separate model-level checks.

### Ordinary and speculative decoding must agree on their inputs

The first cross-mode model check found a JSON-formatting difference between
NextN and ordinary greedy decoding. The F32 control passed all 48 responses.
In Q8, ordinary slot prefill forced a different recurrent reduction from the
single-sequence MTP path. Aligning prefill made its hidden state bit-identical,
but did not eliminate later output differences.

Layer traces then exposed two projection differences. The fast verifier used a
different matrix reduction from single-token decode, and small K/V projections
used packed Q8 activations in verification but float activations in ordinary
decode. Q8 now shares the verifier's one-wave projection templates through
one-row entry points and uses the same activation precision for these pairs.
The small Q5_K/Q6_K projection eligibility follows that policy too. On the
traced coding prompt, all 129 hidden-row checksums agree across prefill and
both block boundaries of the first decode step's 64 layers.
The F32 dispatch keeps its existing selection. A dedicated packed-matrix
validator checks 64 CPU-reference, one-row/verifier-parity and guard cases.

## Validation

The unchanged F32 attention path passes **26 GPU/reference cases**. Q8 passes
**68 cases**, including both writers, zero/tiny/outlier blocks, write guards
and offsets, 32K/64K/128K/262K key lengths, GQA, sinks, scales, head-dimension
stripes, graph updates, scalar query tails, both BLAS precisions and sticky
execution errors. The Q8 count includes the declared mixed-precision checks.

The clean release build also passes **48 MTP/ordinary JSON/SSE responses** at
4K capacity and **24 prefix-cache requests** at 32K capacity, split between MTP
off and on. Each cache mode matches its uncached control in output bytes and
usage, and the two MTP modes match each other on these fixtures. Shared-prefix
requests reuse 8,192 tokens; changed prefixes and resets correctly miss.

A frozen F32 64K server returned the exact key from 62,048 prompt tokens and
then passed a fresh short request. The following single-request diagnostics
used the same 27B GGUF, greedy prompt, NextN and disabled prefix reuse:

| Engine mode                | Allocated capacity | Actual prompt | Wall seconds | Exact key |
| -------------------------- | -----------------: | ------------: | -----------: | --------- |
| Frozen F32 KV              |             65,536 |        62,048 |       816.52 | pass      |
| Q8, scalar 32-query tile   |            262,144 |        62,048 |       454.90 | pass      |
| Q8, FP32-input BLAS        |            262,144 |        62,048 |       354.68 | pass      |
| Q8, opt-in FP16-input BLAS |            262,144 |        62,048 |       142.50 | pass      |

These are diagnostics, each with one long request, rather than repeated
headline benchmarks. The 142.50-second result is 5.73× faster than this frozen
F32 control. It combines cache format and prefill-kernel changes; it does not
isolate either change's contribution.
The Q8 candidates ran API checks before retrieval; the frozen control ran a
short warmup. Their capacity and warmup histories differ.

All **23 actual-model API checks** pass at native allocated capacity with Q8,
NextN and opt-in FP16-input attention. That covers JSON/SSE parity, Unicode,
structured tool arguments and history, truncation safety, validation failures,
context rejection and recovery.

The native-capacity server returned **`CEDAR_9482` exactly from 260,048 prompt
tokens**, with eight generated tokens, in **1,016.34 seconds** (16.94 minutes).
A fresh short request then returned a different exact key in 0.175 seconds.
The server stopped and all temporary power settings were restored. This proves
execution and one synthetic retrieval near the native limit; it is not a
long-document accuracy study or a 260K repository-task acceptance.

Earlier scalar and FP32-BLAS native probes were deliberately stopped for tuning
after their 62K checks; they are not counted as native-length passes. The initial eight-query prototype
was also stopped during its 62K prefill and is not a long-context pass.

## Matched server measurements

The canonical runner used one reusable server per engine, one discarded warmup
and five measured requests per workload. Both servers allocated **262,144 tokens**,
used the same GGUF and prompts, and disabled prefix reuse. ZINC used Q8 KV with
opt-in FP16-input attention and NextN; llama.cpp used `q8_0` KV and ordinary
decoding. Their quantized cache representations differ. The prompts below are
**58–349 tokens**, so these numbers describe short requests at native allocated
capacity, separately from the 260K retrieval above.

| Workload          | ZINC decode tok/s | llama.cpp decode tok/s |
| ----------------- | ----------------: | ---------------------: |
| Quick Chat        |             63.52 |                  29.43 |
| Coding Review     |             76.20 |                  29.33 |
| Incident Context  |             73.93 |                  29.37 |
| Long Coding Draft |             73.11 |                  29.39 |

All output budgets and prompt counts match. The captured previews pass the
output check after correcting a false positive for a valid Markdown table
delimiter in the long coding plan. The original flag and corrected classifier
review are retained in the portable evidence; timing samples were not changed.

A separate 32K F32 control checks the default path: captured text and token
counts match on all four workloads. Decode changes range from **−0.09% to
+0.08%**, and prefill from **−0.60% to −0.32%**. These small changes do not establish
a default-path speedup. Shared CPU work was active during the campaign; each
engine's measured host-load samples are retained in the evidence.

[Portable evidence](https://zolotukhin.ai/zinc/validation/rocm-native-context-2026-10-26.json)
pins runtime commit `59e91681`, binary/source/GGUF hashes, precision choices,
individual samples, failed trials and cleanup results.

## Reproduce

Run model and kernel jobs only on the GPU host, after checking existing GPU
users. The kernel validator loads no model:

```bash
cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -D__HIP_PLATFORM_AMD__=1 \
  -Isrc/cuda -I/opt/rocm/include tools/validate_rocm_attention.c \
  src/rocm/rocm_shim.c -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
  -lamdhip64 -lhiprtc -lhipblas -lm -o /tmp/validate-rocm-attention
ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set your visible GPU selector}" \
  /tmp/validate-rocm-attention --q8
```

For a matched native-capacity comparison, the canonical runner can explicitly
select quantized KV for the baseline too:

```bash
ZINC_KV_CACHE=q8 ZINC_Q8_ATTN_F16=1 ZINC_MTP=1 ZINC_PROMPT_CACHE=0 \
  bun tools/performance_suite.mjs \
  --target rdna --phase all --models qwen38-27b-q4k-m \
  --rdna-backend rocm --rdna-context 262144 --rdna-llama-kv q8_0 \
  --runs 5 --warmup 1 --no-site-write --output /tmp/native-context-suite.json
```

Connection settings come from the environment or the ignored `.env` file;
set the baseline binary and source checkout as described in [ROCm](ROCM.md).

The suite records both KV selections, precision switches and matched capacity.
Its fixed prompts remain short: allocating a 262K cache is not a benchmark of
a 262K-token prompt. The baseline stays at its F16 default unless Q8 is
explicitly requested. ZINC Q8 uses an FP32 block scale; llama.cpp `q8_0` uses
its own representation, so quantized cache formats are not bit-identical.

Compile `tools/validate_rocm_q8_matvec.c` with the same shim and library flags
as the attention validator, then run it on the GPU host to check the packed
projections. The model-level parity fixture is public too:

```bash
ZINC_KV_CACHE=q8 ZINC_Q8_ATTN_F16=1 ZINC_PROMPT_CACHE=0 \
  ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set your visible GPU selector}" \
  python3 tools/validate_mtp_api.py \
  --baseline "$PWD" --candidate "$PWD" \
  --model /path/to/Qwen3.8-27B-Q4_K_M.gguf --device 0 \
  --cases tools/fixtures/rocm_q8_mtp_cases.json \
  --output /tmp/q8-mtp-parity.json
```

This checks one build across three serial servers: MTP on, MTP on again, and
MTP off. The output file must be fresh. The eight cases exercise 48 JSON/SSE
responses at 4,096-token capacity; native-length retrieval is a separate check.

For cache parity, use a fresh output directory for each mode:

```bash
for mtp in 0 1; do
  ZINC_Q8_ATTN_F16=1 bun tools/validate_rocm_prompt_cache.mjs \
    --model /path/to/Qwen3.8-27B-Q4_K_M.gguf \
    --gpu "${ZINC_GPU:?Set your visible GPU selector}" \
    --context 32768 --mtp "$mtp" --kv-cache q8 \
    --output "/tmp/q8-cache-mtp-$mtp"
done
```
