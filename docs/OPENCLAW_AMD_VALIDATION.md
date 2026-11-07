# OpenClaw on AMD: model acceptance

The first OpenClaw validation uses the **published ZINC 0.1.0 Linux archives**
on a **Radeon AI PRO R9700 32 GB**. It tests the actual **OpenClaw 2026.11.1**
client with Node **24.22.2**, rather than substituting a synthetic agent for the
model runs. See the [setup guide](OPENCLAW.md) for repeatable commands.

## Scope and identity

The binaries report source `c3d4c0a23defa92b12394af09750554615b3cf41`.
Their release archive SHA-256 values are:

| Backend | Archive SHA-256                                                    |
| ------- | ------------------------------------------------------------------ |
| Vulkan  | `b203e3d408a374de56e114a16853af6fef294b75f064f75a8e41c0969ec27844` |
| ROCm    | `3e3b78ca71fc1859f39d71e5a97260486190bd4e4a35dafe1320efeb0caa0801` |

Each GGUF's bytes were checked against the release model pins. The node ran
Ubuntu 24.04.3, kernel 7.2.2, Mesa 25.0.7 and ROCm 7.2.4. The runs selected the
discrete R9700 explicitly and used one server slot. Vulkan retained its default
F16 KV cache; ROCm retained F32. No workstation inference was used.

## What each model must do

Each available backend/model combination gets a fresh project and client state:

1. Return an exact text sentinel through the real OpenClaw client.
2. Read a random receipt from a file, implement a small JavaScript function,
   write the exact receipt, and run the project's tests through tools.
3. Return a second exact sentinel, then pass a server health/discovery check.

The evaluator independently reruns the project tests, compares receipt bytes,
and hashes the protected input and test files. It rejects an assistant that
reports success without making the edits, or changes the tests to hide a bad
implementation. The [evaluator source](https://github.com/zolotukhin/zinc/blob/main/tools/openclaw_eval.mjs)
and [portable evidence](/zinc/validation/openclaw-amd-2026-11-05.json) record the checks.

These are bounded acceptance fixtures. They do not establish broad coding
quality, a long-running agent soak result, or performance superiority over
another engine. Task wall times include the client and tools.

## Model matrix

| Model                   | Vulkan      | ROCm                     | Tested context   |
| ----------------------- | ----------- | ------------------------ | ---------------- |
| Qwen3.5 9B Q4_K_M       | Pass        | Pass                     | 16,384           |
| Qwen3.6 27B Q4_K_M      | Pass        | Pass                     | 16,384           |
| Qwen3.6 35B-A3B Q4_K_XL | Pass        | Pass                     | 16,384           |
| Qwen3.8 27B Q4_K_M      | Pass        | Pass                     | 16,384           |
| Gemma 4 26B-A4B Q4_K_M  | Pass        | Pass                     | 16,384           |
| Gemma 4 31B Q4_K_M      | Pass        | Pass, bounded scratch    | 8,192            |
| Qwen3 0.6B Q8_0 starter | Task failed | Unsupported architecture | 16,384 attempted |

Qwen3.5 9B is the smaller recommended starting point. OpenClaw's lean coding
preset removes unused bundled skills and the tool-search catalog; its first
Gemma prompt was about 3,900 tokens, compared with about 6,300 with those extras.
Use the server's real allocated capacity rather than the model's native maximum.

## Limits found during validation

Gemma MoE initially answered text requests at 8K with the broader tool preset,
but its coding task overflowed the context. A subsequent text request and health
check passed. The retained 16K row uses the lean coding preset and completes the
actual file/tool task. Setup cannot allocate a larger server context by writing
a larger number in a client config.

Gemma 31B at 8K/F32 on ROCm completed the task and recovered, but scratch growth
caused allocation retries and smaller prefill chunks. The
[bounded-prefill guide](ROCM_GEMMA_BOUNDED_PREFILL.md) explains this memory path.
The repeat with `ZINC_PREFILL_CHUNK=128` passed all three tasks without any
allocation failures. For that 32 GB ROCm configuration, start the server with:

```bash
ZINC_PREFILL_CHUNK=128 zinc -m /path/to/gemma-4-31B-it-Q4_K_M.gguf \
  -c 8192 --parallel 1 --port 9090
```

This bounds prompt scratch while preserving F32 KV and the 8K context. It does
not change the model weights or add a new kernel. The repeat is recorded
separately from the initial allocation-pressure row in the evidence.

The 0.6B starter's Vulkan requests completed and the server stayed healthy, but
the model appended punctuation to exact sentinels and failed the file task.
It attempted tool actions that did not produce the requested receipt or passing
code. A successful client JSON report did not make those checks pass. ROCm
rejects this plain-Qwen3 architecture before starting a server; the
[release profile](https://github.com/zolotukhin/zinc/blob/main/config/release/targets.json)
and [starter qualification](STARTER_QUALIFICATION.md) already exclude it from
ROCm and from coding recommendations.

The same OpenClaw fixture also failed with the same starter GGUF on a fully
GPU-offloaded llama.cpp Vulkan server at source
`5f436dddb440a288ee5611d7d1eca564a6aca9f4`. Both engines left the required receipt
and function incorrect. The native control passed the two text sentinels;
ZINC's starter responses added periods. This control tests the task outcome,
not token-for-token equivalence or a latest-version throughput comparison.

## Client errors and cleanup

The separate CPU fixture runs the pinned, actual OpenClaw client against a
scripted HTTP server. It verifies split streamed tool arguments, correlated
assistant/tool history, real file writes and command execution, rejection of an
HTTP API error, forced cancellation, connection closure, and a new successful
task afterward. Scripted responses establish client behavior; they are not
included as GPU/model successes.

The GPU campaign held the shared worker reservation, checked exact inference
process names before each server, and retired each owned process group before
loading the next model. The evidence records the final idle check. Intel and
Metal are outside this AMD report.
