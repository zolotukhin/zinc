# Intel OpenCode validation

The November 1 checks on an Intel Arc Pro B70 exercise all seven catalog models
with actual OpenCode 2.0.21. They exposed client compatibility, request recovery,
Gemma cache allocation, long-prompt attention and working-directory defects.
A model that loads or answers a short prompt has not yet passed a coding task.

The earlier client runs below inherited a stale `PWD`. Their individual
outcomes remain observations of that launch; they do not qualify the corrected
configuration. The [actual-client working-directory control](/zinc/validation/opencode-working-directory-protocol-2026-11-02.json)
fails with relative tools before the fix and passes on both pinned client
versions afterward. A new seven-model matrix uses one exact CI archive, the
corrected launcher and independent tests; its results must stand separately.

OpenCode 2.0.21 removed the CLI's `--dir` option and changed plugin registration.
It also sends `max_completion_tokens`, which the old server ignored. A request
for one token generated 256; both budget spellings now generate one. The
[client compatibility evidence](/zinc/validation/opencode-client-compatibility-2026-11-01.json)
retains the live negative control and the actual OpenCode 1.19.4/2.0.21 CPU
protocol checks. Scripted CPU responses test client behavior without a model
coding claim.

## Initial seven-model OpenCode 2 sweep

The [complete seven-model diagnostic](/zinc/validation/intel-opencode-seven-models-2026-11-01.json)
uses actual OpenCode 2.0.21 and Bun 1.4.4 on the Intel Arc Pro B70. All
GGUFs match their immutable catalog pins. Each coding fixture starts a fresh,
single-slot ZINC server and gets 16 steps, 2,048 output tokens and 900 seconds.
Temperature is zero, top-p is one and thinking is disabled. Tests fail before
the task and run independently after the client exits. No supervisor edits,
repaired tool calls or synthetic model responses are used.

| Model                            |         Context |                API checks | Completed coding fixtures | Same-server recovery |
| -------------------------------- | --------------: | ------------------------: | ------------------------: | -------------------- |
| Qwen 3.6 35B-A3B                 |          16,384 |                     29/29 |                       3/3 | 394–402 ms           |
| Qwen 3.6 27B                     |          16,384 |                     29/29 |                       3/3 | 394–409 ms           |
| Qwen 3.5 9B                      |          16,384 |                     29/29 |                       3/3 | 174–198 ms           |
| Qwen 3.8 27B                     |          16,384 |                     29/29 |                       3/3 | 351–386 ms           |
| Gemma 4 31B, initial attempt     | 8,192 requested | Server did not initialize |  0/3, clients not started | Not measured         |
| Gemma 4 26B-A4B, initial attempt |           8,192 |                     28/28 |                       0/3 | 300–311 ms           |
| Qwen 3 0.6B                      |          16,384 |                     25/29 |                       0/3 | 156–166 ms           |

Completed coding means a successful client exit, passing independent tests,
protected files unchanged and no reported client/API errors. A zero client
exit without fixing the task is a failure. The three core fixtures cover a
TypeScript rate limiter, a multi-file cart fix and a Python NDJSON CLI.
The server receives another bounded request after every fixture, before it
is stopped. The sweep finishes with no GPU process left behind.

The starter initially fails four API checks because of server defects in the
JSON-prefilled tool path. These are not evidence against its model weights;
the [parser investigation](JSON_PREFILL_TOOL_CALLS.md) separates that fix from
coding ability. Its rate-limiter
trial times out after 900 seconds with 17 tool calls and three failed tools;
the cart trial makes no tool call; the Python trial makes one failed edit.
No acceptance test improves from its baseline. Recovery is prompt in all
three cases. It remains an experimental qualification model, without an
OpenCode coding recommendation.

Gemma 26B's initial API checks and 48/48 short native reference tokens pass,
but every 8K coding client fails during compaction before a first tool call:
`Compaction summary reached the output token limit`. The context is exhausted
by the prompt and summary budget. These attempts also used the old long-prompt
attention path: the 16K allocation-only retry below still fails. More context
and a longer timeout alone do not establish a working configuration.

Gemma 31B's original uniform KV allocation combines metadata maxima from
different layer types. It reserves about 15 GiB at a trimmed 8K context and
fails initialization. Its actual projections require 901,120 F16 cache bytes
per token: 50 sliding layers with width 4,096 and 10 global layers with width
2,048. K-as-V layers still need separate cache storage for normalized values.
The correction uses those same widths in the planner and allocations, without
quantizing the cache. The 12K follow-up below keeps the default F16 cache.

These are native development diagnostics. Their source snapshots, binary,
shader and client-harness hashes remain in the JSON. The initial sweep and
Gemma follow-up use different source snapshots and must not be treated as a
single qualified release archive.

## Reproduce the Intel client configuration

On the Intel machine, pull a catalog model and start one server. Qwen 3.5 9B is
the smallest model to complete all three coding fixtures in this sweep:

```bash
zinc model pull qwen35-9b-q4k-m
zinc --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

Use a build containing the corrections above. The catalog's tested-profile
promotion is separate from these source-specific diagnostics; an untested
profile warning remains possible. An explicit `--model-id` loads the installed
model without changing its global activation.

In a second terminal, use Bun 1.4.4 or newer and actual OpenCode 2.0.21:

```bash
bun add --global --trust @opencode/cli@2.0.21
curl -fsS https://zolotukhin.ai/zinc/harnesses/setup-opencode.mjs -o setup-opencode.mjs
bun setup-opencode.mjs --project ../my-project --model qwen35-9b-q4k-m \
  --context auto --output 2048 --steps 16
bun setup-opencode.mjs check --project ../my-project
bun setup-opencode.mjs run --project ../my-project --auto --timeout-ms 900000 -- \
  "Fix the failing tests. Read the relevant files, edit the source, and run the tests."
```

Set `ZINC_API_BASE_URL` and `ZINC_API_KEY` in the client environment when the
endpoint differs from the helper's local default. OpenCode executes tools on
the client machine. The [OpenCode guide](OPENCODE.md) covers configuration,
permissions and independent verification. The 0.6B qualification model is not
a coding recommendation.

## Passing tests did not mean a completed client run

With the budget and client fixes, Qwen 3.6 35B-A3B read files, edited source and
ran tests through actual OpenCode 2. The three independent acceptance checks
passed, but the client timed out at 300 seconds while preparing a later turn.
Those remain failed coding runs.

| Task                    | Independent tests after exit | Client outcome | Same-server recovery within 30 seconds |
| ----------------------- | ---------------------------: | -------------- | -------------------------------------- |
| TypeScript rate limiter |                          3/3 | Timed out      | Failed                                 |
| Multiple-file cart fix  |                          3/3 | Timed out      | Passed in 29.7 seconds                 |
| Python NDJSON CLI       |                          4/4 | Timed out      | Failed                                 |

These runs used 16K context, 2,048 output tokens, 16 steps, temperature zero,
top-p one and thinking disabled. Tests and generated configuration stayed
protected. There were no repaired tool calls, synthetic model responses or
supervisor solution edits. All 28 API checks passed on the same diagnostic
build; the real task exposed a lifecycle gap that the short cancellation probe
had missed.

## Cancelling prefill must release the slot

The streaming route checked for disconnects during decode, but a long prefill
could continue processing thousands of tokens after the client exited. Vulkan
prefill now polls request liveness between tokens, chunks and layers, and sends
periodic SSE comments. A failed write or hard disconnect cancels the request.
It drains submitted GPU work before releasing request buffers, clears reuse
state and removes the callback when the handler exits. HTTP upload half-closes
remain valid while the client continues reading.
JSON requests use hard-disconnect detection without writing SSE bytes. A
graceful upload half-close alone cannot distinguish a disconnected JSON reader
from a client still waiting for its response.

The [portable cancellation evidence](/zinc/validation/intel-prefill-cancellation-2026-11-01.json)
records the original failures and a separate controlled comparison. The old
diagnostic missed a 10-second recovery limit after an abandoned 6K-word
prefill. The fixed diagnostic recovered in **418, 419 and 417 ms**, across
three cancellations on the same server. Each positive probe observed a prefill
keepalive before any generated content. Gemma 26B also retained **48/48 exact
reference tokens with its default settings** on that fixed build.

These are native development diagnostics with recorded source and binary
hashes, not qualified release archives. The longer-budget coding sweep is
separate evidence; a fast recovery probe does not establish coding success.
The Metal preview retains its existing decode-time cancellation behavior.

## The larger-context Gemma follow-up

Correcting the cache allocation lets Gemma 31B start at 12K and Gemma 26B at
16K with default F16 KV storage. Short continuations still match 48/48 native
reference tokens for each model. Actual coding tasks then expose repetitive
output and no tool calls once their prompts exceed Gemma's 1,024-token window.
The Vulkan shaders had never received that window and attended to the complete
prefix in every layer.

The [sliding-window correction](VULKAN_GEMMA_SLIDING_WINDOW.md) adds the missing
mask and numerical/page-boundary regressions. Extended model references also
expose a dense-prefill activation approximation; dense Gemma now keeps F32
activations unless explicitly requested. These checks are distinct from the
successful allocation fix. Coding qualification requires actual-client runs
on the corrected build; the failed attempts remain evidence.

## Corrected backend with direct tools

OpenCode 2's Code Mode was another independent failure: its restricted
JavaScript runtime rejected imports and undiscovered tool names. The retained
[Code Mode control](/zinc/validation/intel-gemma-window-codemode-control-2026-11-01.json)
contains one completed 900-second failure, followed by an intentionally stopped
sweep. The direct-tools preset exposes file and shell tools without `execute`;
actual OpenCode 1.19.4 and 2.0.21 each pass three scripted CPU protocol checks.

The [complete direct-tools follow-up](/zinc/validation/intel-opencode-direct-tools-2026-11-02.json)
uses the corrected Vulkan executable, unchanged model pins, default F16 KV and
the same 900-second, 16-step, 2,048-output-token limits. It retains every attempt:

| Model           | Context |                API checks | Successful coding fixtures | Same-server recovery |
| --------------- | ------: | ------------------------: | -------------------------: | -------------------- |
| Gemma 4 31B     |  12,288 |                     29/29 |                        2/3 | 688–705 ms           |
| Gemma 4 26B-A4B |  16,384 |                     29/29 |                        1/3 | 297–305 ms           |
| Qwen 3 0.6B     |  16,384 | 25/29 on the older parser |                        0/3 | 162–167 ms           |

Gemma 31B completes the limiter and cart tasks. Its Python acceptance tests
pass 4/4, but the client times out; the fixture remains failed. Gemma 26B
completes the limiter; the cart and Python trials time out without fixing the
source. The starter completes no fixture, and its cart attempt modifies a
protected file. The independent checks reject it. No tool calls are repaired
and no supervisor supplies implementation edits. Each of the nine fixtures
passes its three post-client native reference replays, and the coordinator
leaves no GPU process running.

The starter's four API failures led to a separate server fix: restore the
tool prefix already in the prompt, select root schema fields from parsed JSON,
and keep malformed/truncated continuations as text instead of manufacturing an
empty executable call. A [separate R9700 Vulkan control](/zinc/validation/json-prefill-api-reference-2026-11-02.json)
passes 29/29 API checks on both the starter and the 9B control, retaining each
model's 48 native choices and three complete HTTP reference replays. That is
an API diagnostic on a different executable and GPU; it does not replace the
Intel coding failures or qualify a release archive.

## The inherited directory was wrong

The separate [1,800-second Gemma 31B Python control](/zinc/validation/intel-gemma31-python-longer-2026-11-02.json)
finishes successfully in 1,752 seconds, with 4/4 independent tests and clean
recovery/reference checks. The original 900-second failure remains failed.
Reading the longer trace exposes a launch defect: after an absolute source
edit, relative shell commands search outside the fixture. They cannot find
`src` or `test`, and spend additional turns searching for files.

Setting a child process's `cwd` does not replace inherited `PWD`. OpenCode
uses that variable to choose its workspace. The helper, evaluator and
independent verification now align both values with the project at launch.
The CPU regression deliberately supplies a stale directory and uses relative
read/write/shell tools, rather than allowing absolute paths to hide the bug.
Both OpenCode 1.19.4 and 2.0.21 fail before the fix and pass all three protocol
fixtures afterward; the outside sentinel remains unchanged on the fixed run.

The original Intel reference checkout and frozen client harness files were
audited and remain unchanged. The corrected GPU rerun retains the old failures
with their launch context and does not attribute those timeouts to model
ability alone.

The [packaged session-reuse control](/zinc/validation/vulkan-session-reuse-packaged-control-2026-11-02.json)
checks the same Vulkan archive on the R9700. Both attention-only Gemmas reuse
1,211 tokens on a tool-history turn, match full prefill, and retain correct
output after an intervening request. Hybrid Qwen advertises no automatic
reuse and uses full prefill. Across these three models, 144/144 short native
choices and all nine post-control HTTP continuations pass. The earlier hybrid
checkpoint mismatch remains in a separate retained negative control.

## Regression checks

CPU tests cover keepalive timing, failed writes, latched cancellation and API
probes that reject decode-only evidence or a blocked recovery request. Both
Vulkan release roles must pass active-prefill cancellation on the Qwen 35B
flagship. Readiness and candidate checks record the measured recovery time;
candidate aggregation rejects absent, failed or over-limit evidence.

Against an already running Vulkan server, the expanded API suite also runs the
probe explicitly:

```bash
bun tools/validate_opencode_api.mjs --model qwen36-35b-a3b-q4k-xl \
  --context 16384 --buffered-tool-cancel --prefill-cancel --output result.json
```

Supply `ZINC_API_BASE_URL` and `ZINC_API_KEY` at runtime. The script starts no
server. General setup instructions are in the [OpenCode guide](OPENCODE.md).
