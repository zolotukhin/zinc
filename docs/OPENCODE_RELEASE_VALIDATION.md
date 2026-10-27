# OpenCode model validation for the first release

On 2026-10-26, release validation covered all seven managed catalog models on
a 32 GB Radeon AI PRO R9700 through Vulkan, plus the six ROCm-supported models.
It combined API probes with actual OpenCode 1.19.4 sessions where serving
capacity permitted them. The four larger Qwen models completed the
three core coding fixtures, with Qwen 3.5 9B on Vulkan requiring a server fix
and a separate retry. This establishes useful coding capability on this board,
with the limits below. It does not qualify every model for the first release.

**Gemma is not OpenCode-qualified.** Gemma 26B failed tool use on both backends.
Gemma 31B could not meet the helper's 8K context minimum on Vulkan. Its initial
ROCm allocation failure was fixed in the memory follow-up below; chat now works,
but its actual OpenCode attempt still made no tool calls and failed the fixture.
The experimental Qwen3 0.6B starter also failed the coding fixture and several
API checks.

The [portable result JSON](/zinc/validation/opencode-release-2026-10-26.json)
retains model pins, binary and harness hashes, every completed fixture outcome,
failed checks, interrupted sessions, and separate diagnostic probes. Different
source revisions and budgets remain separate; they do not form one success
rate. These small fixture checks also do not establish complete delivery on
larger repository assignments; see the [repository trial](OPENCODE_E2E_REPORT.md).

## Environment and method

The board ran Ubuntu 24.04.3 LTS on x86_64, RADV Mesa
25.0.7-0ubuntu0.24.04.2 and ROCm 7.2.4. Tools were Zig 0.17.0, Bun 1.4.4 and
OpenCode 1.19.4. Every GGUF matched its immutable release-policy SHA-256.
ROCm does not implement the plain-Qwen3 path used by the tiny starter.

Initial binaries were the packaged Vulkan and ROCm archives from
[packaging run 40430602151](https://github.com/zolotukhin/zinc/actions/runs/40430602151),
source `681931cd7567506978ce7a7115ec440df84d002d`. Follow-ups used development
binaries with the same packaged runtime assets. The JSON distinguishes those
builds and their hashes; development results do not certify final release bytes.

OpenCode connected directly to ZINC with bounded source reads and tool output.
No synthetic responses, tool repairs or supervisor solution edits were used.
Frozen tests failed before each task and ran independently after the client
exited. Protected tests, specifications, generated config and the context hook
remained unchanged in all completed attempts. One separate transparent capture
replayed identical requests as JSON to diagnose a streaming mismatch.

The initial nine-fixture ROCm runs used 16K context, 2,048 output tokens,
16 agent steps and 180 seconds per task. After a timed-out generation delayed
the next request, follow-ups restarted the server for every fixture and used
4,096 output tokens and 300 seconds. Temperature was zero, top-p one and
thinking disabled throughout. Gemma 26B used 8K context and 2,048 output tokens;
the setup helper requires output to be strictly less than half the context.

The three core fixtures were a TypeScript rate limiter, a multiple-file cart
fix and a Python NDJSON CLI. These test tool reads, edits, shell execution and
independent correctness across two languages. The nine-fixture suite also
covered CLI validation, refactoring, source-only changes, long policy input,
duration parsing and Unicode CSV handling.

## ROCm results

These Qwen results used the original packaged build. API checks covered
JSON/SSE parity, Unicode, typed tool arguments, history, malformed requests,
queue isolation, cancellation and a prompt beyond 8K.

| Model                    | API checks | Coding result                                          |
| ------------------------ | ---------: | ------------------------------------------------------ |
| Qwen 3.5 9B Q4_K_M       |      23/23 | 8/9 initial fixtures; includes all three core fixtures |
| Qwen 3.6 35B-A3B Q4_K_XL |      23/23 | 8/9 initial fixtures; includes all three core fixtures |
| Qwen 3.6 27B Q4_K_M      |      23/23 | 3/3 core fixtures with a fresh server per task         |
| Qwen 3.8 27B Q4_K_M      |      23/23 | 3/3 core fixtures with a fresh server per task         |

Both nine-fixture runs failed `zinc-cli-validation`. Fresh-server retries with
the larger output/time budgets still failed, leaving five of six tests passing.
The initial Qwen 3.6 dense 27B run was interrupted after two unsuccessful client
results and one started, unverified session; its clean core retries are separate
evidence. Qwen 3.6 dense 27B remains experimental in the catalog.

After the Unicode decoder fix, Gemma 26B passed **17/22 API checks** at 8K but
failed the rate-limiter fixture: zero tool calls and zero of three tests passing.
Remaining API failures involved tool arguments, typed schemas and tool history.
Its earlier 16K run also exhausted a GPU allocation near token 8,159 and left
subsequent requests failing. Basic chat success at 4K does not certify that path.

## Vulkan results

The complete core sweep used development source
`2bb5627eaa5829b3b39330a4cf80ffe555106e56`, 16K context and the fresh-server
4,096-token/300-second protocol.

| Model                    | API checks | Core fixtures                                                      |
| ------------------------ | ---------: | ------------------------------------------------------------------ |
| Qwen 3.5 9B Q4_K_M       |      23/23 | 2/3; rate limiter failed before the final payload fix              |
| Qwen 3.6 35B-A3B Q4_K_XL |      23/23 | 3/3                                                                |
| Qwen 3.6 27B Q4_K_M      |      23/23 | 3/3                                                                |
| Qwen 3.8 27B Q4_K_M      |      23/23 | 3/3; additional coverage outside the required Vulkan policy        |
| Qwen3 0.6B Q8_0          |      19/23 | 0/1 sentinel; broader run stopped after earlier transport failures |

The failed Qwen 3.5 rate-limiter case was retried separately at source
`8475094dd717c384c3cd79fa8d03684f5352bc4c` after preserving tool payloads from
prose cleanup. It passed with six tool calls and all three independent tests;
the API checks remained 23/23. The earlier 2/3 result remains unchanged.

At the same source, Gemma 26B passed **16/22 API checks** at 8K but again made
zero tool calls and failed all three sentinel tests. An earlier attempt never
launched OpenCode because its 4,096-token output budget equaled half the context;
that setup error is recorded separately. The API harness also previously tried
an oversized long-prompt probe on 8K servers; it now runs that probe only when
capacity permits a prompt above 8K.

Gemma 31B's requested 8,192-token context was reduced to **8,018** by the memory
planner. The helper rejected this below-minimum capacity before OpenCode ran.
Qwen3 0.6B's targeted retry remained 19/23 API checks and 0/1 coding fixtures.
Neither model gains an OpenCode support claim from these results.

## Bugs found and fixed

- Vulkan/Metal streaming now emits final token usage, which OpenCode uses for
  context accounting. The evaluation harness loads and protects its real
  bounded-context plugin instead of rejecting that plugin as an unexpected file.
- JSON and streaming completions end on complete UTF-8 boundaries. Gemma and
  SentencePiece vocabularies decode as raw UTF-8, retaining accented text, CJK
  and four-byte emoji instead of applying a byte-remapping decoder.
- Hidden leading reasoning no longer shrinks visible streaming prefixes.
  Streaming scratch capacity matches the full generated-text buffer, preserving
  tool arguments larger than 4 KB.
- Structured tool payloads bypass prose repetition/cleanup heuristics. Only
  leading reasoning blocks are removed, preserving literal `<think>` source
  strings inside tool arguments and repeated code lines.
- Buffered tool responses send SSE keepalives and detect disconnected clients.
  The release validator accepts the diagnostic's known GPU architecture suffix
  while retaining the exact ROCm target check.

A real buffered-tool cancellation probe failed to recover within ten seconds
on the original ROCm build. After the fix, recovery took 1,212 ms on ROCm and
326 ms on Vulkan. These are individual recovery checks, not throughput claims.
CPU negative controls also reject a server that leaves the recovery request
stalled.

CPU-only token-piece comparison against llama.cpp
`5f436dddb440a288ee5611d7d1eca564a6aca9f4` matched all **262,144 pieces in each
Gemma vocabulary**. All non-unused Qwen pieces matched; pre-existing unused
padding labels differ from llama.cpp's empty output and were left unchanged.
This decoder comparison does not substitute for numerical GPU kernel parity.

Before the memory fix, strict ASCII/Unicode JSON/SSE probes at the release-default
4K context and output budgets 1, 2, 3 and 64 produced:

| Backend | Gemma 26B | Gemma 31B                                       |
| ------- | --------: | ----------------------------------------------- |
| Vulkan  |       8/8 | 8/8                                             |
| ROCm    |       8/8 | 0/8; allocation failure on generation token one |

The failing ROCm server still answered health requests successfully. A healthy
endpoint is therefore insufficient qualification. Release GPU jobs now require
Unicode chat parity, complete JSON/SSE termination and usage, and full Unicode
echo at a sufficient budget; unit negative controls cover broken transport.

## Gemma ROCm memory fix

The Gemma 31B allocation failure is resolved for the policy's **4K, single-slot**
configuration. The same required model remains in the release policy.

Single-slot serving previously duplicated the engine's KV cache, consuming
another **6.875 GiB**. Its dense-weight scratch also multiplied independent
dimension maxima, reserving 672 MiB for a matrix that needs only 220.5 MiB.
The observed first request had 372 MiB free and failed that 672 MiB allocation.

The fix reuses single-sequence KV through non-owning views, sizes scratch to
the actual dense projection/FFN dimensions, and rolls back partial slot and
batch allocations. A larger 8K request exposed a second bug: failed batched
prefill freed temporary KV views twice. Those views now have one cleanup path.

On the R9700, cold single-slot 4K startup fell from about 31.6 GiB to 24.7 GiB.
The original binary still returned HTTP 500 in the controlled replay; the fixed
binary generated `OK`. At 4K, ten Unicode, scratch-growth, overflow/recovery and
isolation checks passed. A two-slot 1K check passed 7/7, and Gemma 26B passed
the same ten 4K checks.

The first memory fix still crashed during an 8K allocation failure; that failed
run remains in the evidence. After correcting cleanup, the 8K run passed 10/10,
including three temporary-buffer allocation failures that safely fell back to
token-wise prefill. This configuration has little spare memory and can prefill
more slowly. It does not expand the required 4K release context.

The actual ROCm archive from
[packaging run 40458670841](https://github.com/zolotukhin/zinc/actions/runs/40458670841)
passed extraction/identity checks and GPU diagnostics without a library-path
override, all twelve tokenizer corpus cases, three complete 16-token CLI
continuations against pinned llama.cpp, and the packaged HTTP gate. That gate
compares full reference continuations and checks Unicode streaming, cancellation,
near-capacity input, malformed/oversized requests and subsequent recovery.
Its source tree matches fix commit `ae7fea1f11af7d24725ae56a8f208a81f29e8d0b`;
the JSON binds the preview archive, executable and reference digests. This
qualifies one packaged model row on this board, not the complete release matrix.

With the server now running at 8K, actual OpenCode 1.19.4 completed a
rate-limiter attempt without a timeout or API error. It made zero tool calls
and failed all three independent tests. Gemma 31B therefore remains outside
the OpenCode support claim despite the working inference path.

## What still blocks 0.1.0

The original required ROCm Gemma 31B allocation blocker is fixed. Both Gemma
models still need a separate tool-calling fix before advertising OpenCode support.
The tiny starter remains experimental; stable publication already rejects an
unqualified starter.

These runs cover one RDNA4 board. The required Apple families, other Linux GPU
families, signed/notarized Mac delivery and protected release infrastructure
still need qualification. At this audit, the repository had no registered
self-hosted runners or configured environments. Rebuild and qualify the final
candidate's exact packaged bytes after these fixes; do not promote the older
archives using development-build results.

The final process audit found no remaining ZINC, llama.cpp or OpenCode
processes. All model inference ran on the remote RDNA node. No local workstation
GPU was used. [Issue 10](https://github.com/zolotukhin/zinc/issues/10) remains
open until the public release and automatic process work end to end.

For repeatable runs, use [the evaluation tools](OPENCODE.md#evaluate-coding-tasks): start
one model/backend at the intended context, run the API checks, then invoke
`opencode_eval.mjs` for one fixture at a time with the documented budgets.
Restart the server between fixtures, especially after failures or timeouts.
Keep the portable metadata; raw traces contain local paths and session data.
