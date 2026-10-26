---
title: 'From fast tokens to a verified coding task on RDNA4'
seoTitle: 'ZINC, OpenCode and a Local 27B Model: Debugging a Real Coding Task'
date: '2026-10-26'
tags:
  - zinc
  - rdna4
  - rocm
  - opencode
  - coding-agents
  - qwen3.8
  - gpu-kernels
  - performance
keywords:
  - local 27B coding agent
  - OpenCode ZINC harness
  - ROCm long context attention
  - hybrid recurrent prefix cache
  - Radeon AI PRO R9700 inference
excerpt: 'A local 27B model was fast at generation and still failed in a real worktree. Fixing GPU memory, tool output and independent verification finally got one feature through review.'
seoDescription: 'Debugging ZINC and OpenCode on an R9700: bounded attention, recurrent prefix checkpoints, context-safe tools, independent checks and a measured 27B feature task.'
faqs:
  - question: 'Did the local model complete a real repository feature?'
    answer: 'Yes, one bounded feature with code-review feedback: a setup-only --dry-run configuration preview. The initial green check missed two contract violations. The same model session repaired them, and independent acceptance plus project checks passed. This does not establish general unsupervised coding reliability.'
  - question: 'Does the 64–75 tokens per second result use 32K-token prompts?'
    answer: 'No. Both servers had 32,768-token capacity, but the four benchmark prompts contained 58–349 tokens. ZINC used NextN; prompt reuse was disabled for both engines. Long-context execution and repository coding were tested separately.'
  - question: 'Why use 32K when Qwen 3.8 27B supports 262K?'
    answer: '32K was the coding trial configuration. Native capacity needs 34 GiB of FP32 KV including NextN, before weights and scratch. The opt-in Q8 cache reduces that to 9.5625 GiB. A separate mixed-precision attention mode completed a 260,048-token synthetic retrieval on the 32GB card; that does not establish native-length coding quality.'
draft: false
---

Our local Qwen 3.8 27B model generates **64–75 tokens per second** on a Radeon AI PRO R9700. In a real ZINC worktree, one coding attempt lasted 200 seconds, made five tool calls, and produced no edits. The next request failed with `prompt longer than context`.

The model had read three files. Their tool results contained **28,358, 16,630 and 36,181 characters**. A 32K context window disappeared into source inspection before the first patch.

That was already a follow-up to harder failures: out-of-memory prefill, attention kernels exceeding shared memory, and corrupted generation near the context limit. Getting one repository feature through review required changes from the GPU kernel to the acceptance test. This is the engineering behind that result, including the first green check we had to reject.

The stack was **Zig 0.17.0, Bun 1.4.4, ROCm 7.2.4 and OpenCode 1.19.4**, with Qwen 3.8 27B Q4_K_M on the R9700's 32GB of VRAM. The [validation report](/zinc/docs/rocm-agent-context/) and [portable measurements](/zinc/validation/rocm-agent-context-2026-10-25.json) retain the failed trials alongside the completed checks.

## A configured context was larger than the kernel could execute

The [original worktree trial](/zinc/docs/opencode-worktree-report/) tried to add loaded-model discovery to ZINC's minimal ROCm server. Small tool-call fixtures had worked. The repository prompt reached execution paths they had never exercised.

At 32K capacity, the 27B model exhausted VRAM during prefill. Reducing the capacity exposed another failure. At **16,080 prompt tokens**, `attention_causal_batched_v2` requested a **65,612-byte group segment**, followed by illegal-memory and rocBLASLt/Tensile errors. Another attention path rejected launches above 16K. Some requests still produced output after execution had failed.

There were two different memory budgets to fix:

| Budget            | What grew                                        | What changed                                             |
| ----------------- | ------------------------------------------------ | -------------------------------------------------------- |
| GPU shared memory | An attention score array spanning the context    | Process keys in fixed-size tiles                         |
| VRAM              | Prompt scratch and duplicate request-slot caches | Chunk prefill and share the existing single-slot buffers |

A larger `--context` value increased the allocation request without making either implementation safe.

For attention above **8,192 keys**, the new path processes **1,024-key tiles**. It retains a running maximum, softmax denominator and weighted value sum. For a tile of scores $s$ and values $v$:

$$
\begin{aligned}
m' &= \max(m, \max_j s_j) \\
a &= e^{m-m'}
\end{aligned}
$$

$$
\begin{aligned}
\ell' &= a\ell + \sum_j e^{s_j-m'} \\
u' &= au + \sum_j e^{s_j-m'}v_j
\end{aligned}
$$

Initialize $m=-\infty$, $\ell=0$ and $u=0$; the final output is $u/\ell$. Rescaling both accumulators keeps earlier tiles valid when a later tile contains a larger score. Attention sinks contribute to the final denominator. This uses the established [tiled exact-attention formulation](https://arxiv.org/abs/2205.14135); it changes the storage and reduction order while retaining the full attention calculation.

Each compiled long-context kernel now uses **4,364 bytes of static shared memory and zero dynamic shared memory**. The old kernels remain below the threshold. Decode, causal prefill, speculative verification and independent request slots all need the bounded path; fixing only ordinary single-token decode would have left the coding workload broken.

The validator passed **25 GPU cases** against an independent double-precision CPU reference. We tested tile tails, 8,191/8,192/8,193 keys, the failing 16,080-token shape, 32K, attention sinks, GQA, unusual head dimensions, permuted slots and graph updates. Outputs must be finite and satisfy `abs(error) <= 0.0003 * (1 + abs(reference))`. These are ROCm/R9700 results; the shared CUDA source still needs its own hardware validation.

## Chunking a hybrid model requires preserving more than positions

Prefill now processes at most **4,096 tokens per chunk** by default. That bounds prompt scratch. The single-slot server also reuses the existing KV and recurrent buffers instead of allocating a second complete cache. Partial allocation failures release what was already allocated.

Chunking has a correctness cost: every boundary must carry absolute RoPE positions, KV offsets, recurrent state, convolution history and its offsets. NextN also needs the preceding chunk's final hidden row when priming its prediction head.

One fallback bug made that state requirement concrete. Suppose chunk one advances the recurrent state, then allocation for a later chunk fails. Restarting at token zero through the ordinary-decode fallback applies the prompt to a state that has already consumed part of it. The fallback can finish and still be wrong.

We now clear that partial recurrent state before retrying. A private Debug build injected an allocation failure after the first chunk; two requests then matched an independent ordinary-decode control in response bytes and token usage.

Execution errors also become sticky. A failed launch, transfer, BLAS operation or synchronization invalidates the execution context. JSON generation returns an error, `/health` returns 503, and streaming emits an error without a successful finish. The server requires a restart. This prevents stale GPU results from becoming plausible-looking model output. Controlled failure checks exercised that behavior without deliberately injecting an illegal memory access.

Full-model retrieval probes subsequently returned the exact key from the start of **8,548-, 17,048- and 30,048-token prompts**. We also checked a short request afterward, rejection above capacity, and another successful request after that rejection. Multi-slot and graph tests included distinct concurrent keys and generation crossing the 8K kernel-dispatch boundary.

Those checks established usable long-context execution. The next repository runs still timed out.

## Prefix reuse needs a recurrent checkpoint

Agent turns repeatedly send nearly the same history. Avoiding repeated prefill helps, but this model combines attention with recurrent layers. Keeping only attention KV entries does not reconstruct the recurrent state at the end of the reused prefix.

The opt-in `ZINC_PROMPT_CACHE=1` path keeps attention KV in place and saves a checkpoint at a fixed prefill-chunk boundary. That checkpoint includes the recurrent state, convolution history and offsets, plus the NextN boundary state. Reuse requires **exact token IDs and the same execution mode**. Changed or shortened prefixes, failed prefill and incompatible slot use invalidate it.

Fixed boundaries matter: restoring the same checkpoint preserves the chunking and floating-point operation order used by the uncached path. This is a VRAM snapshot, roughly 144MB of additional recurrent state for the 27B model, and currently supports one Qwen request slot. It is off by default.

Cache-off/cache-on sequences matched greedy response hashes and full token usage on both the 9B and 27B models with NextN enabled. Only the expected shared prefixes produced hits. Usage continues to report the complete prompt; computed-prefill counters report the rows actually processed.

A captured **25,649-token coding request** illustrates the potential: two uncached replays took 71.87 and 71.69 seconds, the cold cached replay took 71.91, and the warm replay took **26.22 seconds**. Content, tool arguments, finish reason and usage matched exactly. This is a single-request diagnostic. It measures repeated work avoided, without demonstrating completion of the coding task.

## The next bottleneck was source coming back through tools

After the GPU fixes, earlier full-worktree attempts still spent 900–1,686 seconds inspecting, repeating calls or producing an uncompilable patch. In a supervised repair, the agent eventually ran passing project tests itself. But the successful command's verbose output expanded the next request to **28,200 prompt tokens**, and the wrapper timed out before finishing its independent check. We recorded a timeout.

The later five-call failure made the context problem easier to isolate: ordinary whole-file reads alone overflowed 32K. Instructions asking for small reads were insufficient. We needed bounds on what the tools actually returned.

The setup helper now installs a small local OpenCode plugin using its [before/after tool hooks](https://opencode.ai/docs/plugins/):

- Before a file read, cap it at **80 lines**, preserving the requested offset.
- After a tool call, shorten results above **6,000 JavaScript string units**, retaining 4,000 from the head and 1,500 from the tail with a truncation notice.
- Preserve metadata and avoid splitting Unicode surrogate pairs at the cut.
- Reserve context for incoming tool results as well as the next generated response when configuring compaction.

These are practical bounds per result. Accumulated history can still exhaust a context window, and a truncated result can omit information the model needs. The notice tells it to request a narrower range or command.

The plugin initially passed our direct tests and failed the actual headless client test. We were launching OpenCode with `--pure`. In the [tested 1.19.4 implementation](https://github.com/anomalyco/opencode/blob/v1.19.4/packages/opencode/src/plugin/index.ts), pure mode disables external plugins, including explicitly configured local ones. Removing that flag and setting `OPENCODE_PURE=false` in the child made the hook load.

That justified a protocol test using the real CLI: read a 600-line file, return a large Bash result, then inspect what reaches the scripted model endpoint. The same CPU-only validator also exercises timeout/repair and fatal API errors. Scripted responses verify transport and lifecycle behavior cheaply; actual model inference remains a separate test.

## Verification needs time outside the model conversation

The harness now runs full project checks between short editing rounds. Complete logs stream to stderr; a failed check returns bounded diagnostics to the model, including errors from the middle of a large log. A successful build no longer needs to become the next prompt's largest document.

The overall deadline reserves time for that independent check. When an editing round reaches its deadline, the wrapper stops OpenCode and tests the partial work. A failing check can resume the same session if time remains. A passing check can accept a complete patch even if the model never reaches its closing message.

There are separate failure paths. A fatal API error fails the run immediately and skips verification: otherwise an already-correct fixture could turn a broken model interaction into a false pass. User cancellation remains cancellation. A check returning zero after the overall deadline cannot retroactively rescue the run. Owned process groups, including background tools, are stopped.

The real-CLI fixtures caught these cases without occupying a GPU. They now run in CI alongside the unit tests.

## The first green feature still had two bugs

For the next actual-model trial, we chose a useful, bounded feature in a fresh ZINC worktree: add a setup-only **`--dry-run`** to the OpenCode helper.

The contract was stricter than “print some valid JSON.” Preview output had to match the normal config file byte for byte, including exactly one trailing newline. It had to honor options without creating directories, writing config or plugin files, reading existing config, contacting services, or capturing runtime endpoints and keys. Other commands had to reject the flag.

With the tool limits installed, the first editing round inspected source and failed its independent check. The second implemented the feature and documentation. After **465 seconds and 14 tool calls**, the wrapper returned zero.

Review found two mistakes:

1. `console.log(JSON.stringify(config, null, 2) + '\n')` emitted **two** newlines. The acceptance test parsed the JSON and checked that output ended with a newline, so it missed the byte mismatch.
2. `run --dry-run` without a task returned the task-required error before rejecting the setup-only option. The acceptance test had always supplied a task.

The implementation violated the contract, and our tests were too weak to detect it. We kept the original checks and added stricter byte-equality and command-scope checks before accepting anything.

The same model session received the review findings and repaired both issues with **one read, one edit and one Bash call**. The supervisor did not edit the implementation. The editing round reached its three-minute deadline; the wrapper stopped the client, then independently verified the patch. All **12 stricter acceptance categories**, existing harness tests, project CPU checks, formatting and whitespace checks passed. The repair run took 185 seconds including verification.

That is one accepted repository feature **with review feedback**, exercising the timeout-verification path with actual inference. Twenty traced passthrough requests contained no synthetic or repaired model responses. Six permanent CLI regression tests now cover preview bytes, option overrides, filesystem preservation, runtime-value privacy, command scope and invalid budgets.

The earlier failed trials remain failed trials. One reviewed feature does not establish a general autonomous-coding success rate.

## Measuring throughput exposed two harness bugs too

After a separate GPU resume failure, device-only recovery was unsuccessful. The node returned after a reboot, and ROCm diagnostics and the queued hardware checks passed. An older control binary had also reported no device before loading a model during the outage, so that interval could not supply a performance comparison.

The matched benchmark then exposed two mistakes in our runner.

First, the ROCm preflight was consulting **Vulkan device inventory**. On this machine, Vulkan and ROCr enumerated the integrated and discrete GPUs in opposite orders. The guard rejected a correct ROCm selection. The fix uses ROCr inventory under the selected visibility filter, then uses logical device zero for both engines. Device ordinals belong to the API that produced them.

Second, server defaults differed: ZINC used 2K capacity while the baseline launcher used 4K. We added explicit paired context arguments and reran both at **32,768 tokens per slot**. The earlier mismatched run stays outside the final headline.

The final comparison uses the same 27B GGUF, one persistent server per engine, one slot, one discarded warmup and five measured requests per scenario. ZINC uses NextN; llama.cpp uses HIP at `5f436dddb`. Prompt reuse is disabled for both. Both engines receive the same prompts and output budgets under the same ASPM-performance and automatic-clock policy.

<div style="max-width:100%; overflow-x:auto;" role="region" aria-label="Matched R9700 decode performance" tabindex="0">

| Scenario          | Prompt tokens | Output tokens | ZINC tok/s | llama.cpp tok/s |
| ----------------- | ------------: | ------------: | ---------: | --------------: |
| Quick Chat        |            58 |            96 |  **64.33** |           30.13 |
| Coding Review     |           183 |           160 |  **74.67** |           30.09 |
| Incident Context  |           349 |           128 |  **67.59** |           30.10 |
| Long Coding Draft |            80 |           256 |  **73.06** |           30.10 |

</div>

Values are median decode tokens per second. Summing prompt-plus-generation phase times across the four scenarios gives **10.033 seconds for ZINC versus 22.555 seconds for llama.cpp**: 2.248× the phase throughput, or 55.52% less phase time. This excludes other HTTP and client overhead.

The prompts contain **58–349 tokens**. The 32K value describes allocated server capacity; these rates do not describe a 30K coding history. Prompt counts match, both engines reach every output budget, and automatic preview checks pass. That checks the benchmark workload without proving semantic equivalence of every answer.

This comparison also does not attribute a new speedup to the context fixes. A separate short-context control matched response hashes and usage, with decode 0.19–0.29% lower: essentially steady. The [Zig migration post](/blog/2026-10-24-zig-0-17-faster-prompts-on-rdna4/) measures the separate comparison against Zig 0.15.2.

## Why this trial used 32K when the model supports 262K

Qwen 3.8 27B's [official configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) supports **262,144 tokens**. The 32K capacity in this coding trial was a server and validation setting. Increasing it safely required another change in the engine.

The base model has 16 full-attention layers, four KV heads and 256 dimensions per head. With separate FP32 K and V caches, that costs **128 KiB per token**. At native capacity, the base KV alone consumes 32 GiB. The appended NextN attention block brings it to **34 GiB**, before loading the roughly 15.93 GiB weight file or allocating recurrent state and prefill scratch.

The new opt-in ROCm cache stores 32 signed bytes with one FP32 scale per block. That reduces the native-capacity allocation, including NextN, to **9.5625 GiB**. The GGUF weights stay the same. KV writers, decode, prompt attention, slot aliases and speculative catch-up all have to agree on the representation; changing only the allocation would corrupt the first attention read.

Compressed storage made capacity possible, but the first long-prompt kernel still spent too much time reconstructing values for individual queries. Sharing a compressed key tile between 32 queries helped. The next improvement expands only one 1,024-key tile at a time for BLAS, then combines tiles with the same online-softmax recurrence. Its score buffer grows with the bounded query chunk, rather than the total context.

A 62,048-token exact-key request took **816.52 seconds** on the frozen FP32 control, **354.68 seconds** with Q8 and FP32-input BLAS, and **142.50 seconds** with opt-in FP16 matrix inputs. Those are single-request diagnostics, not the repeated short-prompt benchmark above. They combine a cache-format change with new attention kernels.

The complete model then returned the exact key from **260,048 prompt tokens** in **1,016 seconds**, about 16.9 minutes, and passed a fresh short request afterward. Its server advertised 262,144-token capacity and passed all 23 API checks. That is a completed near-native retrieval, not merely a successful cache allocation. It is still a synthetic key buried in repeated padding, so it does not measure long-document comprehension or repository coding quality.

Precision needed its own check. The initial FP16-input mode failed the original FP32 reference tolerance. We kept that guard and left the mode off by default. A separate reference now models FP16 Q/K/V and probabilities with FP32 softmax state and accumulation. The tested outputs pass that contract; measured against FP32 matrix inputs over the same reconstructed Q8 cache, their RMSE is 0.000183 and maximum absolute error is 0.00296. This measures the extra mixed-precision difference, rather than the total effect of quantizing KV. Kernel agreement does not establish unchanged coding quality.

API success also missed a cross-mode discrepancy: the first Q8 test produced different JSON formatting with MTP on and off. A control run with FP32 KV passed. Matching prefill reductions fixed one difference, but tracing the first generated token exposed more: the verifier and ordinary decoder used different matrix reduction paths and different activation precision for small K/V projections. Sharing the fast projection templates with a one-row entry point aligns those choices. A dedicated GPU validator now checks their packed-weight math, row parity and guards against a separate CPU implementation.

The clean build passes 48 MTP/ordinary JSON/SSE responses, 24 cache checks and all 23 API checks, alongside 158 GPU reference/error cases. In the repeated server comparison at 262K allocated capacity, ZINC measured **63.5–76.2 decode tokens/s** against **29.3–29.4** for llama.cpp with quantized KV. Those prompts are still only 58–349 tokens; these are not 260K decode rates. The default FP32 control retained matching captured output and token counts, with decode changes from −0.09% to +0.08%.

The benchmark guard also caught a defect in its own heuristic: a valid Markdown table delimiter looked like repeated tokens. The regression test fails under the original detector. The corrected check exempts a delimiter only after a matching textual header; malformed separators, repeated lines and control-token checks still fail. Reclassifying the captured output leaves every timing sample intact.

The [native-context guide](/zinc/docs/rocm-native-context/) and [portable evidence](/zinc/validation/rocm-native-context-2026-10-26.json) have the memory accounting, switches, hashes, individual measurements and validation results. This remains a single-slot ROCm Qwen feature. The preview-feature acceptance described earlier belongs to its original 32K FP32 run.

## Actual coding beyond 32K still needed review

We then assigned three more real Zinc tasks in fresh worktrees: reject partially parsed benchmark integers, keep model-discovery IDs valid strings for invalid filesystem bytes, and discover the server capacity automatically during OpenCode setup. We ran the same tasks at native Q8 capacity and then with the 32K FP32 configuration. The [end-to-end report](/zinc/docs/opencode-e2e-report/) retains both configurations, exact prompts, frozen acceptance and separate completion review.

The native runs passed their original functional checks in **6m05s, 4m00s and 14m10s**. The setup feature needed four automatic verification rounds. Its actual reported input reached **46,520 tokens**, beyond the earlier server capacity, with no API errors. That is a real coding history above 32K; it is still far short of a 262K coding-quality test.

Under the same limits, the 32K configuration passed the first two functional tasks but failed automatic setup after **19m05s and four verification rounds**. It made 26 tool calls and no feature edits, reaching 31,519 reported input tokens without an API error. This is one ordered trial on a shared host, with capacity, cache format and attention precision changed together; it cannot establish a general quality or speed advantage.

All three native patches omitted the requested regression tests. Two also omitted documentation. Our verifier had allowed the wrapper to stop as soon as the functional code passed, so the whole workflow accepted incomplete assignments. Existing project tests passing could not detect missing new tests.

Review found another defect in the larger feature. Model discovery searched for either an exact ID or its `.gguf` suffix in the same `find` call. If a suffixed entry appeared first, it won over a later exact match. An unmatched multi-model list could also be accepted as an alias. The contract required exact matching first and allowed aliasing only for a singleton. The original test had checked an exact match and a singleton separately, missing their interaction.

The 32K Zig run also added regression tests that did not compile: a Unicode scalar was placed in a byte array, and an allocator had the wrong type. Its project checks had still passed because those paths had not analyzed the module's embedded tests. Direct compilation failed both locally and on the node. The landing build now runs discovery as a dedicated CPU test target on every backend.

We kept those original outcomes. Reviewer-authored regressions and fixes completed the landing patches, including malformed/duplicate IDs, credential-safe failures and a timeout that covers the response body. The resulting helper offers **`--context auto`** without saving runtime endpoints or keys. That makes setup easier; it does not turn three functional passes into three fully delivered model tasks.

A separate fresh-build test exercised the revised setup itself: `--context auto` matched the live 262K server, preview bytes matched the written config, and all 23 API checks passed. OpenCode then fixed a protected Python NDJSON fixture, added tests and wrote usage docs in 70 seconds. Review caught Python's permissive handling of nonstandard JSON constants; the same session repaired it in another 115 seconds. Seven unit tests and six stricter external cases passed, including failure of the new tests against the original source. That is a bounded fixture with review feedback, separate from the six repository trials; numeric overflow and general schema validation were outside its checks.

## Try the same bounded workflow

The [OpenCode guide](/zinc/docs/opencode/) covers server setup, isolated worktrees and task evaluation. The [standalone helper](/zinc/harnesses/setup-opencode.mjs) requires Bun 1.4.4 or later. With a running server, setup can now discover its actual capacity:

```bash
bun tools/setup_opencode.mjs --project ../my-project \
  --model qwen38-27b-q4k-m --context auto --dry-run

bun tools/setup_opencode.mjs --project ../my-project \
  --model qwen38-27b-q4k-m --context auto

bun tools/setup_opencode.mjs check --project ../my-project

bun tools/setup_opencode.mjs run --project ../my-project --auto \
  --verify 'bun test' -- "Implement the requested feature and run its tests."
```

Replace the acceptance command with the project's own checks. Keep the generated config and its versioned `.zinc/` plugin together. Custom endpoints and keys are supplied through environment variables at runtime; generated files contain no machine-specific connection data.

The most revealing defect in this work was the extra newline. We had working GPU kernels, a responsive model, bounded tools and a green wrapper. Review still found a contract violation because the test measured valid JSON while the feature required exact bytes. A faster token loop only becomes useful coding throughput when the rest of the system can carry the task to a result that survives that kind of check.
