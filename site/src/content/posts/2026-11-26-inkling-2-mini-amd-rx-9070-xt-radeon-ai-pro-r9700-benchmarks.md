---
title: "Inkling 2 Mini on AMD Radeon RX 9070 XT and Radeon AI PRO R9700"
seoTitle: "Inkling 2 Mini AMD Benchmarks | RX 9070 XT and R9700"
date: "2026-11-26"
tags:
  - zinc
  - inkling-2-mini
  - amd
  - rdna4
  - rx-9070-xt
  - radeon-ai-pro-r9700
  - rocm
  - vulkan
  - llama-cpp
  - speculative-decoding
  - benchmarks
keywords:
  - Inkling 2 Mini AMD benchmark
  - Inkling 2 Mini RX 9070 XT
  - Inkling 2 Mini Radeon AI PRO R9700
  - Thinking Machines Inkling 2 Mini GGUF
  - Inkling 2 Mini Q4_0 16 GB VRAM
  - Inkling 2 Mini Vulkan ROCm
  - Inkling 2 Mini llama.cpp
  - Inkling 2 Mini DSpark AMD
excerpt: "Inkling 2 Mini runs on the 16 GB RX 9070 XT and 32 GB Radeon AI PRO R9700. See the same-checkpoint results against llama.cpp ROCm and Vulkan and what a verified DSpark prototype changed."
seoDescription: "Inkling 2 Mini Q4_0 benchmarks on AMD RX 9070 XT and Radeon AI PRO R9700, with ZINC ROCm, Vulkan, llama.cpp, VRAM and DSpark results."
faqs:
  - question: "Can an AMD Radeon RX 9070 XT run Inkling 2 Mini?"
    answer: "Yes. A 16 GB RX 9070 XT ran the 14.9 GB Inkling 2 Mini Q4_0 GGUF in ZINC on both ROCm and Vulkan with a 4,096-token context."
  - question: "How fast is Inkling 2 Mini on the Radeon AI PRO R9700?"
    answer: "ZINC ROCm decoded at 63.8 to 66.9 tokens per second across four single-sequence workloads. An experimental DSpark run reached 143.5 tokens per second on the Coding Review prompt."
  - question: "Which Inkling 2 Mini download works with ZINC?"
    answer: "The managed model ID is inkling-2-mini-q4-0. The Inkling development build uses it to install the checksum-pinned Inkling 2 Mini Q4_0 GGUF for text inference on AMD RDNA4."
---

Inkling 2 Mini has 24 stored layers, but it visits 48 layers to generate each token. The middle 12 layers run three times. That means the 16 GB Radeon RX 9070 XT can hold the Q4_0 checkpoint while the GPU still does substantially more work than a single pass through the weights. Getting the model to fit was the easy part. Getting those repeated visits to keep their own history was where the port got interesting.

Across four single-sequence workloads, ZINC ROCm decoded at **59.7–62.4 tok/s** on the RX 9070 XT and **63.8–66.9 tok/s** on the Radeon AI PRO R9700. Against the llama.cpp v1.1.0 Vulkan build, ZINC ROCm led both prefill and decode in all eight comparisons. ZINC Vulkan ran at **33.7–35.9 tok/s** and **34.7–36.0 tok/s** respectively. The experimental DSpark verifier reached **142.1 tok/s** on the RX 9070 XT and **143.5 tok/s** on the R9700 on the Coding Review workload.

## Why Inkling 2 Mini is interesting locally

Thinking Machines Lab released Inkling 2 Mini on November 20, 2026. The model stores 26.4 billion parameters and activates roughly 3.4 billion unique parameters per token. Counting the repeated layers brings the number of weight activations to 5.4 billion per token. The release reported 48.2% on DeepSWE v1.1 and 77.9% on the hard subset of SWE-bench Pro v2, compared with 42.2% and 73.8% for Qwen 3.8 27B. Both models scored 20.1 on the Artificial Analysis Intelligence Index. Those are model-level results reported by Thinking Machines Lab. The measurements here compare inference engines using the same Inkling GGUF. [Release and capability results](https://thinkingmachines.ai/news/inkling-2-mini/)

The loop saves weight capacity because the middle layers reside in memory once. KV state and short-convolution history depend on the pass, so each chronological visit needs its own cache. Each pass can also read residual information from the previous pass. Reusing the same cache across passes would let a later visit read keys, values, or convolution inputs from a different calculation.

Inkling mixes sliding attention with full attention. For this checkpoint, the sliding layers use a 1,024-token window. ZINC gives those visits a ring buffer with room for the oldest position needed by the first token of a prefill chunk. The full-attention visits keep all positions within the configured context. Once the ring wraps, a test has to check both ends of the window. A quick question that fits inside one chunk can give a good answer even when that boundary is wrong.

## RX 9070 XT and R9700 benchmark results

Each cell shows **prefill / decode** in tokens per second. Prefill processes the submitted prompt. Decode emits new tokens. Each value is a median of three runs after one warmup. Both engines ran a reusable server with one sequence on the same card, with an identical Q4_0 checkpoint, a 4,096-token context, and the same chat workloads. The tables use llama.cpp Vulkan as the common reference for both ZINC backends. The measurement file retains each independently paired Vulkan reference run. The ROCm column therefore measures ZINC ROCm against llama.cpp Vulkan. The baseline is the Vulkan build from the llama.cpp v1.1.0 release tag. The downloadable measurements include response latency, end-to-end throughput, token counts, and run ranges. [llama.cpp v1.1.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0) · [Benchmark data](/blog/inkling-2-mini-amd/benchmark-results.json)

### Radeon RX 9070 XT, 16 GB

| Workload | Prompt → output tokens | ZINC ROCm | ZINC Vulkan | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| Quick Chat | 53 → 96 | 698.8 / 62.4 | 323.9 / 33.7 | 144.3 / 50.8 |
| Coding Review | 213 → 160 | 742.4 / 59.7 | 325.4 / 35.8 | 402.4 / 50.3 |
| Incident Context | 336 → 128 | 748.5 / 59.7 | 312.4 / 34.2 | 534.0 / 50.7 |
| Long Coding Draft | 73 → 256 | 581.1 / 60.1 | 295.6 / 35.9 | 180.3 / 51.0 |

### Radeon AI PRO R9700, 32 GB

| Workload | Prompt → output tokens | ZINC ROCm | ZINC Vulkan | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| Quick Chat | 53 → 96 | 707.2 / 66.9 | 294.9 / 36.0 | 205.2 / 45.7 |
| Coding Review | 213 → 160 | 738.2 / 64.0 | 301.8 / 34.7 | 545.2 / 45.5 |
| Incident Context | 336 → 128 | 742.5 / 63.8 | 298.0 / 34.7 | 712.3 / 45.5 |
| Long Coding Draft | 73 → 256 | 582.9 / 64.3 | 280.6 / 34.9 | 262.4 / 45.7 |

<figure class="diagram-card diagram-wide">
  <img src="/blog/inkling-2-mini-amd/decode-throughput.png" width="1890" height="750" loading="lazy" decoding="async" alt="Grouped chart comparing median quick-chat and extended decode rates for Inkling 2 Mini on the RX 9070 XT and Radeon AI PRO R9700" />
</figure>

Quick Chat makes the difference between prefill and decode easy to see. On the RX 9070 XT, ZINC ROCm processed the 53-token prompt at 698.8 tok/s and decoded at 62.4 tok/s. The baseline managed 153.0 and 50.9 tok/s respectively. Median request latency went from 2.39 s to 1.79 s. On the R9700, the largest prompt in the speed matrix was already processed at 713.7 tok/s by llama.cpp, so ZINC's 742.5 tok/s gave it a much smaller prefill lead. Decode still came in at 63.8 against 46.0 tok/s.

On ROCm, ZINC packs the activations for the Q4_0 projections and groups expert routes during prefill, giving the matrix hardware multiple tokens to work on for a given expert. The Vulkan path also uses packed activations when integer dot products are available and replays the recorded single-token commands. ZINC Vulkan led prefill in four of eight comparisons. llama.cpp decoded faster in all eight. ROCm delivered the highest sustained generation rate on both cards.

## ROCm against ROCm

The first matrix used a common Vulkan reference. For this follow-up, both
engines used ROCm on the same card, with llama.cpp built from the v1.1.0
release tag for `gfx1201`. The RX 9070 XT used ROCm 7.2.1 and the R9700 used
7.2.4. Both servers used the same Q4_0 weights, one slot, a 4,096-token
context and the same four chat workloads. Drafting was disabled. Each
cell reports the median of three measured requests after one warmup.
[Measurements and build hashes](/blog/inkling-2-mini-amd/benchmark-results.json)

| GPU | Workload | ZINC ROCm prefill / decode | llama.cpp ROCm prefill / decode | ZINC decode lead |
| --- | --- | ---: | ---: | ---: |
| RX 9070 XT | Quick Chat | 606.3 / 61.3 | 13.6 / 33.9 | +80.7% |
| RX 9070 XT | Coding Review | 704.1 / 58.5 | 54.1 / 34.6 | +69.2% |
| RX 9070 XT | Incident Context | 718.9 / 58.6 | 87.1 / 35.0 | +67.6% |
| RX 9070 XT | Long Coding Draft | 526.8 / 59.0 | 19.3 / 34.5 | +70.9% |
| R9700 | Quick Chat | 711.4 / 67.1 | 30.5 / 40.5 | +65.7% |
| R9700 | Coding Review | 742.4 / 64.1 | 119.6 / 41.2 | +55.5% |
| R9700 | Incident Context | 746.2 / 64.0 | 226.0 / 41.5 | +54.3% |
| R9700 | Long Coding Draft | 585.0 / 64.4 | 50.5 / 41.2 | +56.4% |

## DSpark reached 143.5 tok/s with the same target weights

Inkling's repeated layers are part of generating one token. DSpark adds a separate Q8_0 drafter that proposes several future tokens. The target verifies them in a batch, commits the matching prefix, and continues from the first disagreement. The prototype can propose up to seven tokens. It reduces the draft depth when a workload starts rejecting proposals and raises it again when acceptance recovers.

There are two pieces of state to recover after a rejected draft. The target can logically drop uncommitted KV positions, but its short convolutions have already advanced through the proposed text. The verifier snapshots the convolution history and restores it past the last accepted position. That recovery matters on a looped model because it must restore the state of each visit independently.

In the separate DSpark experiment, both servers loaded the same Q4_0 target and Q8_0 drafter. ZINC used adaptive depth up to seven. The llama.cpp Vulkan server used a fixed depth of seven and a physical batch size of 256. That build came from upstream commit 074f1f3. Each cell shows median prefill / decode in tok/s from three runs and one warmup.

| GPU | Workload | ZINC ROCm DSpark | llama.cpp Vulkan DSpark |
| --- | --- | ---: | ---: |
| RX 9070 XT | Quick Chat | 695.1 / **98.7** | 128.0 / 43.0 |
| RX 9070 XT | Coding Review | 739.2 / **142.1** | 370.9 / 64.8 |
| RX 9070 XT | Incident Context | 744.9 / **92.3** | 363.1 / 50.6 |
| RX 9070 XT | Long Coding Draft | 578.1 / **81.2** | 168.8 / 30.4 |
| AI PRO R9700 | Quick Chat | 704.4 / **98.6** | 204.3 / 78.6 |
| AI PRO R9700 | Coding Review | 733.0 / **143.5** | 539.2 / 114.7 |
| AI PRO R9700 | Incident Context | 738.2 / **91.1** | 549.4 / 64.8 |
| AI PRO R9700 | Long Coding Draft | 579.7 / **80.7** | 255.1 / 53.1 |

The coding-review workload benefited the most on both cards. The 256-token Long Coding Draft workload came in around 81 tok/s. In the earlier fixed-depth sweep on the RX 9070 XT, increasing the reference server's depth from zero to seven moved Coding Review from 50.0 to 64.0 tok/s and moved Long Coding Draft from 50.5 to 30.0 tok/s. That spread is why the headline spec rate needs a workload name. One accepted token still takes target work, and work done for rejected proposals comes out of the same budget.

The native prototype peaked at **16,033 MiB** of device memory on the RX 9070 XT and **15,795 MiB** on the R9700 in a separate memory probe. The drafter weighs about 504 MB on disk. The 16 GB card had about 271 MiB between the observed peak and the driver's reported capacity. The depth sweep in the measurement file shows how the reference server's memory use and decode speed changed at depths 0, 1, 2, 3, and 7. [Benchmark data](/blog/inkling-2-mini-amd/benchmark-results.json)

## Did the answers stay coherent?

In the experimental ROCm server, we issued the same prompts with greedy decoding and DSpark enabled and disabled. The visible text and token counts matched for both complete answers on both cards. The draft acceptance went from 140 of 236 proposals on the database question to 220 of 285 on the incident ledger. The database prompt asked why the system flushes a write-ahead log before it writes a data page.

> The log is the durable record of the change, so the database can replay it even if the dirty page never reached disk. On recovery, committed log records are redone to rebuild the page, restoring the committed update.

The second question hid a cause and a mitigation at different positions in a 1,600-token synthetic incident ledger. The model associated both with the right ticket and returned valid JSON.

```json
{"ticket":"INC-417","root_cause_time_utc":"09:17 UTC","mitigation":"rotate shard lease keys"}
```

Both Vulkan runs answered the short Paris question correctly and returned identical streaming and buffered text. With reasoning hidden and a 768-token completion budget, both runs reached the limit before returning an answer on the 1,600-token ledger. Raising the output budget to 1,600 tokens with packed Q8 still reached the limit on both cards. In a separate 1,608-token raw test with the assistant answer already started, Vulkan extracted the correct ticket and mitigation on both cards. It returned the time as 09:17, while llama.cpp preserved the UTC suffix. The short and long raw continuations also differed between engines.

## How far 32 GB goes on the R9700

The single-slot ROCm DSpark prototype loaded a **401,408-token** window with
about 248 MiB of VRAM free after startup. At **409,600 tokens**, the target
model loaded and the DSpark draft allocation ran out of VRAM. At 401,408,
a short prompt retrieved all three planted codes and DSpark accepted all 12
of its draft tokens. The largest window test used 3,167 prompt tokens.

| Configured window | Prompt tokens | Peak VRAM | Prefill | Decode | Codes retrieved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 401,408 | 3,167 | 32,394.78 MiB | 612.75 tok/s | 107.95 tok/s | 3/3 |
| 65,536 | 62,591 | 18,336.74 MiB | 245.87 tok/s | 50.65 tok/s | 3/3 |
| 65,536 | 64,870 | 18,336.75 MiB | 249.50 tok/s | 74.56 tok/s | 0/3 |

The 62,591-token prompt spread three audit codes through synthetic prose.
Prefill took **254.6 seconds** and the 19-token reply contained all three codes.
The 64,870-token prompt used repeated filler. It finished prefill in
260.0 seconds, then repeated filler through its 256-token reply. Both requests
used `/v1/completions` with a preformatted prompt because the prototype's chat
formatter hit its 64 KiB buffer. The long-window runs measured ZINC only.
The matched llama.cpp comparison above used a 4,096-token window.

## Run Inkling 2 Mini on AMD

The managed model ID is `inkling-2-mini-q4-0`. The tested file is 14,922,755,456 bytes, about 13.9 GiB. Both tested cards can run a 4,096-token context with it. Use the development checkout below for fixed-loop text generation with Zig 0.15.2. The DSpark prototype was measured separately. [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) · [Q4_0 checkpoint](https://huggingface.co/codegood/Inkling-2-Mini-GGUF/blob/7d95146e8e432955d714c17db9a3d9703cd53263/Inkling-2-Mini-Q4_0.gguf)

```bash
git clone --branch fix/qwen-smoke-token-anchor https://github.com/zolotukhin/zinc.git
cd zinc
git checkout 88360b9fda3c23cd32b799c36dfe347b0d36ac67
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
./zig-out/bin/zinc model pull inkling-2-mini-q4-0
./zig-out/bin/zinc --model-id inkling-2-mini-q4-0 --context 4096 \
  --chat --prompt "Explain why a database uses a write-ahead log." -n 256
```

For Vulkan, build with `zig build -Dbackend=vulkan -Doptimize=ReleaseFast` and set `RADV_PERFTEST=coop_matrix`. The server recipe uses one request slot and lists the ID in `GET /v1/models`. [Running ZINC](/zinc/docs/running-zinc/#inkling-2-mini-on-amd-rdna4) · [ROCm build guide](/zinc/docs/rocm/)

The performance suite can reproduce the four-server-workload comparison with a prepared RDNA node and a llama-server build from the release tag. Point the model root at the directory containing the same GGUF for both engines.

```bash
bun tools/performance_suite.mjs --target rdna --rdna-backend vulkan \
  --models inkling-2-mini-q4-0 --runs 3 --warmup 1 --rdna-keep-dpm \
  --rdna-workdir "$ZINC_RDNA_WORKDIR" --rdna-model-root "$ZINC_RDNA_MODEL_ROOT" \
  --rdna-vk-device "$ZINC_RDNA_VK_DEVICE" --rdna-llama-device "$ZINC_RDNA_LLAMA_DEVICE" \
  --rdna-llama-server "$ZINC_RDNA_LLAMA_SERVER_REMOTE" \
  --no-site-write --output /tmp/inkling-2-mini-amd-vulkan.json
```

For the ROCm run, select `--rdna-backend rocm` while keeping the model file and llama.cpp Vulkan baseline the same. [ZINC benchmarks](/zinc/benchmarks/) · [Measurement data](/blog/inkling-2-mini-amd/benchmark-results.json)
