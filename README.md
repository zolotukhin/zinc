<p align="center">
  <img src="assets/zinc_trademark_new.png" alt="ZINC" width="360">
</p>

# ZINC

Run open-weight models on the GPU you already own. ZINC is a single Zig binary —
no Python, no CUDA-only assumptions — that loads GGUF files and gives you a
command line, a browser chat, a model manager, and an OpenAI-compatible API.

<p>
  <a href="https://github.com/zolotukhin/zinc/actions/workflows/test.yml"><img src="https://github.com/zolotukhin/zinc/actions/workflows/test.yml/badge.svg" alt="CI status"></a>
  <a href="https://zolotukhin.ai/zinc"><img src="https://img.shields.io/badge/website-zolotukhin.ai%2Fzinc-d35400" alt="ZINC website"></a>
  <a href="https://discord.gg/QRUgWH2aGV"><img src="https://img.shields.io/badge/Discord-Join%20ZINC-5865F2?logo=discord&logoColor=white" alt="ZINC Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

[Get started](https://zolotukhin.ai/zinc/docs/getting-started/) ·
[See every benchmark](https://zolotukhin.ai/zinc/benchmarks/#rdna-rocm) ·
[Read the docs](https://zolotukhin.ai/zinc/docs/) ·
[Join Discord](https://discord.gg/QRUgWH2aGV)

## Speed, measured honestly

Each model below was run on the same Radeon AI PRO R9700 through ZINC's ROCm
backend and through llama.cpp's ROCm (HIP) backend, built from the same
llama.cpp commit as our Vulkan comparison. Both engines load the same GGUF file,
read the same prompt tokens (the harness checks the counts match), run in a
reusable server, and get the same warmups and repeat counts. The bars show how
ZINC compares on the two things you feel while using a model: how fast it reads
your prompt, and how fast it writes the answer.

<a href="https://zolotukhin.ai/zinc/benchmarks/#rdna-rocm">
  <img src="assets/rocm-r9700-benchmark.svg" alt="ZINC compared with llama.cpp across six models on a Radeon AI PRO R9700, both using ROCm" width="100%">
</a>

Reading that honestly:

- **Prompt processing** is level to 2.0x llama.cpp on the short chat prompt shown
  here. The lead narrows as prompts grow: across all four workloads llama.cpp is
  ahead in three of the 24 prompt-processing rows (Qwen 3.6 35B-A3B on 150- and
  300-token prompts, Gemma 4 31B on the longest).
- **Token generation** is ahead in all 24 rows, from 1.01x (Gemma 4 31B on the
  longest prompt) to 1.27x (Gemma 4 26B-A4B and Qwen 3.6 35B-A3B).
- **Qwen 3.8 can speculate, and the chart does not count it.** That model ships
  an extra "NextN" block, and ZINC uses it to draft tokens the full model then
  verifies in one batched pass: identical output, fewer passes, 1.9x the
  generation speed. llama.cpp can speculate with the same block (its converter
  exports it as a separate draft model), and our runs do not give it one, so
  that number would not be a fair comparison. The chart therefore shows both
  engines **without** speculation; the benchmark page has a switch for the
  speculative figure.
- **On AMD, pick the backend for the job.** ZINC's ROCm backend reads prompts
  much faster than its Vulkan backend; for generation the two are close, and
  Vulkan is ahead on some models (Qwen 3.6 35B-A3B: 105 tok/s on Vulkan, 97 on
  ROCm). The Vulkan build is compared against llama.cpp's Vulkan backend and
  published separately on the
  [benchmark page](https://zolotukhin.ai/zinc/benchmarks/).

This is one GPU and six models, not a universal claim. Other cards are measured
separately, and rows where llama.cpp is ahead stay on the page.

Every number comes from `tools/performance_suite.mjs` and lands in
[`site/src/data/zinc-performance.json`](site/src/data/zinc-performance.json),
recording prompts, raw samples, build revisions and the llama.cpp commit used.
The chart above is regenerated from that file, never hand-edited.

## Get running

ZINC needs Zig 0.15.2 or newer. Linux Vulkan builds also need `glslc` and a
Vulkan loader; ROCm builds need a working ROCm installation.

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build -Doptimize=ReleaseFast

./zig-out/bin/zinc --check                      # what GPU did it find?
./zig-out/bin/zinc model pull qwen35-9b-q4k-m   # fetch a model
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m --prompt "Hello" --chat
```

On AMD you can pick either backend. Vulkan is the portable default; ROCm uses
HIP with kernels tuned for RDNA4 and leads on prompt processing:

```bash
ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast
ROCR_VISIBLE_DEVICES=0 ./zig-out/bin/zinc --check
```

See [Getting started](https://zolotukhin.ai/zinc/docs/getting-started/) for
package prerequisites and the first-run walkthrough, or the dedicated
[ROCm setup guide](https://zolotukhin.ai/zinc/docs/rocm/).

## Chat and API

```bash
./zig-out/bin/zinc chat --model-id qwen35-9b-q4k-m
```

That one command starts the browser chat and an OpenAI-compatible API on the
same port, so existing clients and SDKs work unchanged: `/health` for liveness,
`/v1/models` and `/v1/chat/completions` for the API. The
[API guide](https://zolotukhin.ai/zinc/docs/api/) has curl and SDK examples.

## GPU support

| Hardware | Backend | Status |
| --- | --- | --- |
| AMD Radeon (RDNA3/RDNA4) | ROCm/HIP | Supported · fastest prompt processing on Radeon |
| AMD Radeon (RDNA3/RDNA4) | Vulkan | Supported · portable, no ROCm install needed |
| Intel Arc (Xe2/Battlemage) | Vulkan | Supported |
| Apple Silicon | Metal | Supported |
| NVIDIA RTX | CUDA | Experimental |

Each backend has its own kernels and is benchmarked separately — no backend
inherits another's results. The
[hardware guide](https://zolotukhin.ai/zinc/docs/hardware-requirements/) lists
validated cards, drivers, memory requirements and current limitations.

## Models

ZINC reads local GGUF files and ships a managed catalog you can pull from.
Current tuning work covers Qwen 3.5, Qwen 3.6, Qwen 3.8, Gemma 4 and Muse
Glimmer, including mixture-of-experts and hybrid-attention architectures.

```bash
./zig-out/bin/zinc -m /path/to/model.gguf --prompt "The capital of France is"
./zig-out/bin/zinc -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --prompt "Hello" --chat
```

The Muse checkpoint used in ZINC measurements is the exact
[Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/blob/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf)
file published by Meta.

## Build, test, contribute

```bash
zig build -Doptimize=ReleaseFast
zig build test
```

Start with the [development guide](docs/DEVELOPMENT.md) and the
[contributing guide](CONTRIBUTING.md). To reproduce or extend the benchmarks,
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) documents the suite, including how
to publish a speculative-decoding on/off pair.

ZINC is active engineering work. When a model or GPU path is incomplete, the
benchmark page leaves that result visible instead of quietly dropping it.

MIT licensed.
