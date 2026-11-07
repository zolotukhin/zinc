---
title: 'OpenCode and OpenClaw on AMD: R9700 and RX 9070 XT'
seoTitle: 'OpenCode & OpenClaw on AMD R9700 and RX 9070 XT'
date: '2026-11-07'
tags:
  - zinc
  - amd
  - rdna4
  - rocm
  - vulkan
  - opencode
  - openclaw
keywords:
  - OpenClaw R9700
  - OpenCode R9700
  - OpenCode RX 9070 XT
  - OpenClaw RX 9070 XT
  - AMD Radeon AI PRO R9700 local LLM
  - AMD Radeon RX 9070 XT coding agent
  - OpenCode AMD GPU
  - OpenClaw AMD GPU
  - ZINC ROCm Vulkan
excerpt: 'Connect OpenCode or OpenClaw to a local AMD GPU with ZINC. A practical Linux guide for R9700 owners, an RX 9070 XT starting point, and the model tests behind it.'
seoDescription: 'Run OpenCode and OpenClaw with ZINC on AMD GPUs. R9700 tested results, RX 9070 XT memory guidance, Linux setup, Vulkan vs ROCm, and working client commands.'
faqs:
  - question: 'Can I run OpenClaw on a Radeon AI PRO R9700?'
    answer: 'Yes. Published ZINC 0.1.0 Vulkan and ROCm binaries passed bounded OpenClaw 2026.11.1 tasks on a 32 GB Radeon AI PRO R9700 with six larger models. The tasks required real file edits, independently passing tests, intact protected files and recovery.'
  - question: 'Can I use OpenCode or OpenClaw with an RX 9070 XT?'
    answer: 'The clients connect to an HTTP model server, so the integration is independent of the GPU vendor. The RX 9070 XT has 16 GB of VRAM, and Qwen 3.5 9B Q4_K_M is a smaller model to try with ZINC Vulkan. This guide does not certify that card: ZINC’s published AMD qualification is on the R9700.'
  - question: 'Do OpenCode and OpenClaw need CUDA for AMD GPUs?'
    answer: 'No. ZINC runs the model through Vulkan or its separate ROCm backend. OpenCode and OpenClaw send API requests and execute tools; they do not need CUDA to connect to that server.'
  - question: 'How much context should I allocate for a local coding agent?'
    answer: 'The Qwen examples use a 16,384-token server context and one slot. Tool schemas, source files, outputs and generated tokens share that capacity. Check that the model and cache fit your GPU, then use automatic context discovery in the client helper.'
draft: false
---

A local coding agent has to do something more convincing than say “done.” In our AMD tests, OpenClaw had to read a random receipt from disk, fix a JavaScript function, write that receipt back and run the tests. We then ran the tests ourselves and checked that the agent had left the protected files intact.

Six larger models passed that sequence on the **AMD Radeon AI PRO R9700**, through both ZINC Vulkan and ZINC ROCm. The small starter model failed the coding task. Keeping that failure in the results makes the setup advice much more useful.

If you are looking for **OpenClaw on an R9700**, or **OpenCode on an RX 9070 XT**, the connection is straightforward: ZINC runs the local model, and the coding client talks to its HTTP API. Here is how to connect them, choose a sensible starting model and verify that the agent actually works.

## The GPU runs the model; the harness runs the tools

OpenCode and OpenClaw handle the agent loop. They send messages and tool definitions, receive the model’s proposed actions, execute those actions and send the results back. ZINC supplies the inference engine behind an OpenAI-compatible API.

<figure class="diagram-card diagram-compact">
  <div class="flow-diagram" role="img" aria-label="OpenCode or OpenClaw sends chat and tool requests to ZINC. ZINC runs GGUF inference on the AMD GPU through Vulkan or ROCm. File edits and tests run on the coding client's machine.">
    <div class="flow-main">
      <div class="flow-node">OpenCode or OpenClaw<br />Files, edits and test commands</div>
      <div class="flow-arrow" aria-hidden="true">↕</div>
      <div class="flow-node flow-node-accent">ZINC HTTP API<br /><code>/v1/chat/completions</code></div>
      <div class="flow-arrow" aria-hidden="true">↕</div>
      <div class="flow-node flow-node-output">GGUF model on the AMD GPU<br />Vulkan or ROCm</div>
    </div>
  </div>
  <figcaption>The inference server and coding client have separate jobs. Tools execute where the client runs.</figcaption>
</figure>

The examples below put both on one Linux machine and use `127.0.0.1`. That keeps the first setup simple. A remote GPU server is also possible; the [OpenCode](/zinc/docs/opencode/) and [OpenClaw](/zinc/docs/openclaw/) guides explain custom endpoints and environment-based API keys.

## R9700 versus RX 9070 XT: start with the memory budget

AMD lists **32 GB** of video memory for the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) and **16 GB** for the [Radeon RX 9070 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html). Both use RDNA 4, but their available model and context budgets differ substantially.

<table tabindex="0" aria-label="AMD GPU memory and ZINC qualification comparison">
  <thead>
    <tr><th scope="col">Card</th><th scope="col">VRAM</th><th scope="col">ZINC evidence</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Radeon AI PRO R9700</th><td>32 GB</td><td>Published Vulkan and ROCm binaries tested</td></tr>
    <tr><th scope="row">Radeon RX 9070 XT</th><td>16 GB</td><td>Bring-up guidance; no board-specific qualification here</td></tr>
  </tbody>
</table>

The R9700 results do not transfer automatically to the RX 9070 XT. Leave room for the key/value cache, which stores attention history, and temporary prompt-processing buffers as well as weights. A model file fitting in VRAM does not prove that a long coding session will fit.

For OpenCode or OpenClaw on an RX 9070 XT, start with the smaller Qwen model rather than copying a 27B or 35B R9700 configuration. Inspect the server’s allocation and device logs. The [hardware guide](/zinc/docs/hardware-requirements/) explains the remaining memory costs.

## Install ZINC and choose Vulkan or ROCm

These instructions target **Linux x86_64**. ZINC 0.1.0 ships separate Vulkan and ROCm executables; each binary contains one backend. The published AMD qualification covers the R9700. There is no native Windows ZINC 0.1.0 binary.

Download the versioned installer, inspect it, then install the Vulkan archive:

```bash
curl -fL https://github.com/zolotukhin/zinc/releases/download/v0.1.0/install.sh \
  -o install.sh

# Inspect install.sh before running it.
ZINC_VERSION=v0.1.0 bash install.sh
export PATH="$HOME/.local/bin:$PATH"
zinc --version
zinc --check
```

Vulkan needs a compatible Vulkan 1.3 driver. On AMD’s Mesa RADV path, use `export RADV_PERFTEST=coop_matrix` before starting the server. The archive already contains compiled shaders; installing it does not require Zig or a shader compiler.

For the R9700’s separately tested ROCm path, install with `ZINC_VERSION=v0.1.0 ZINC_BACKEND=rocm bash install.sh` instead. The [ROCm guide](/zinc/docs/rocm/) lists the required HIP, hipRTC, hipBLAS and header packages. Our recorded R9700 userspace is ROCm **7.2.4**; this is a tested configuration, not a claim about every newer driver release. `RADV_PERFTEST` does not affect ROCm.

AMD’s [ROCm 7.2.1 Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) lists both card names. That establishes AMD’s documented stack support for that version; ZINC’s own binary and model qualification remains separate. Check the matching OS, kernel and runtime instructions before changing an existing installation.

## Start one model with room for tools

Pull the managed model and start a single-slot server:

```bash
zinc model pull qwen35-9b-q4k-m
zinc --model-id qwen35-9b-q4k-m \
  --context 16384 --parallel 1 --port 9090
```

Keep this terminal open. On a machine with integrated and discrete graphics, check that ZINC selected the intended GPU. Vulkan accepts `-d <device-index>`; ROCm uses its own visibility controls, described in the backend guide. Those device lists are separate.

The 16K context includes instructions, tool schemas, source code, tool results and generated output. OpenClaw’s lean coding prompt already consumes several thousand tokens. A 4K chat demo is therefore a poor starting configuration for an agent.

If allocation fails on a 16 GB card, reduce the model or context within the client’s documented limits. OpenCode needs at least 8,192 tokens. Increasing the client’s context number cannot allocate more memory on the server; use automatic discovery below to keep the two consistent.

## Connect OpenCode to your AMD GPU

The tested client versions include OpenCode **2.0.21**. Install its CLI with **Bun 1.4.4 or newer**, create a disposable project and download ZINC’s standalone helper:

```bash
bun add --global --trust @opencode/cli@2.0.21
mkdir zinc-agent-demo
cd zinc-agent-demo
curl -fsS https://zolotukhin.ai/zinc/harnesses/setup-opencode.mjs \
  -o setup-opencode.mjs

bun setup-opencode.mjs --project . --model qwen35-9b-q4k-m \
  --context auto --output 2048 --steps 16
bun setup-opencode.mjs check --project .
```

Setup writes `opencode.zinc.json` and a small context hook under `.zinc/`. It preserves the ordinary OpenCode config and keeps both primary and small-model requests on ZINC. The hook limits individual reads and trims oversized tool output so a single command is less likely to consume the entire context.

`check` probes health, ordinary chat, JSON tool calls and streamed tool calls. It does not execute the probe’s requested tool. Once it passes, give the real client a bounded task:

```bash
bun setup-opencode.mjs run --project . --auto -- \
  "Create normalizeTag.mjs and a Node test file. The function must trim whitespace, lowercase text, and replace runs of whitespace with a hyphen. Run node --test and fix failures."
```

`--auto` approves tool requests for this headless run, subject to explicit deny rules. Use a project you intend the agent to edit. After it finishes, inspect the diff and run `node --test` yourself. The [full OpenCode guide](/zinc/docs/opencode/) also covers verification commands, retries and larger repository tasks.

## Connect OpenClaw to your R9700 or RX 9070 XT server

The AMD acceptance campaign used [OpenClaw 2026.11.1](https://github.com/openclaw/openclaw/releases/tag/v2026.11.1) with **Node 24.22.2**. That client requires Node 24.16+ in the 24.x line, or Node 26.1+. With a compatible Node installation, use the same running server and project:

```bash
npm install --global openclaw@2026.11.1
curl -fsS https://zolotukhin.ai/zinc/harnesses/setup-openclaw.mjs \
  -o setup-openclaw.mjs

node setup-openclaw.mjs setup --project . --output 1536
node setup-openclaw.mjs check --project .
node setup-openclaw.mjs run --project . --timeout 600 -- \
  "Review normalizeTag.mjs and its tests. Add a test for tabs and newlines, fix the implementation if needed, and run node --test."
```

If you skipped the OpenCode example, create an empty project directory first; OpenClaw can create its own implementation and tests from a similarly bounded task.

The helper discovers the loaded model ID and actual server context, then writes **`openclaw.zinc.json`**. ROCm may expose the GGUF filename as the model ID, so discovery avoids guessing. The preset uses direct coding tools, thinking off, a lean local-model prompt and no bundled skill or tool-search catalog. It starts a one-shot agent task without a gateway or messaging channel.

OpenClaw’s config check verifies the connection and client configuration. It cannot establish that a model will make correct edits. The [OpenClaw guide](/zinc/docs/openclaw/) includes the independent file-and-test evaluator used for our AMD results.

## What actually passed on the R9700

The [November 5 AMD OpenClaw report](/zinc/docs/openclaw-amd-validation/) tests the **published 0.1.0 archives**, rather than an unnamed development build. Qwen 3.5 9B, Qwen 3.6 27B, Qwen 3.6 35B-A3B, Qwen 3.8 27B and both Gemma 4 models passed on Vulkan and ROCm. Most rows used 16K context; Gemma 31B used 8K. Its bounded ROCm repeat used `ZINC_PREFILL_CHUNK=128` and completed without allocation failures.

That is **12 accepted backend/model combinations**. Each had to pass exact text, actual workspace changes, independent tests, protected-file checks and recovery. The [raw portable report](/zinc/validation/openclaw-amd-2026-11-05.json) retains the contexts, archive hashes and outcomes. The [OpenCode validation](/zinc/docs/opencode-release-validation/) separately records its coding fixtures and source-specific limits.

The Qwen 3 0.6B starter failed OpenClaw’s coding acceptance, and the same file task also failed on the native llama.cpp GPU control. It is useful for a small Vulkan installation check, but it is a poor choice for this workflow. ROCm does not support that plain-Qwen3 starter in this release.

These bounded checks do not establish a long-running soak result or a universal speed advantage. They establish something concrete: the selected model and engine completed real tool actions and produced files that passed an independent test.

## When the first task fails

**The client cannot connect.** Confirm that the server is still running, that the backend and GPU are correct, and that the endpoint ends in `/v1`. For an initial connection check, `curl -fsS http://127.0.0.1:9090/health` should return healthy JSON. Then run the client’s stronger `check` command.

**The tool history overflows.** Read the server’s actual context allocation. Rerun setup with `--force` after changing it. Shorten file reads and command output. Our earlier Gemma OpenClaw attempt overflowed an 8K configuration with the broader prompt; the lean 16K configuration passed. The failed row remains in the report.

**The model talks about edits but makes none.** Inspect the files and run their tests. A fluent answer or zero exit status is insufficient. Start with Qwen 3.5 9B and a small task before moving to a larger model or repository.

**The agent is slow between tool calls.** Watch prefill as well as decode. Every turn brings instructions, code and prior tool results back to the model. A fast tokens-per-second decode number says little about the wait to process a long prompt. Compare backends on the same card, model, context and task, and keep client/tool wall time separate from engine timings.

The useful milestone is the first clean diff followed by a test command you ran yourself. From there, the R9700’s larger memory budget gives room to explore bigger models and histories. On an RX 9070 XT, keep that same verification loop while establishing your own card’s memory and model limits.
