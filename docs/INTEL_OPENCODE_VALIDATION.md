# Intel OpenCode validation

The Intel Arc Pro B70 checks keep inference correctness, API behavior, client
exit and independent acceptance tests separate. Earlier failures retain their
source, binary and launch settings.

## November 9 Gemma cache and client qualification

The [two-Gemma exact-package replay](/zinc/validation/intel-gemma-chat-prefix-2026-11-09.json)
checks CI source `2d0e1369`, binary `3bea62b8`, with actual **OpenCode 2.2.6**
and **OpenClaw 2026.11.1**. Both models pass native continuations before and
after the clients, eight cache/invalidation controls and independent coding
acceptance:

| Model           | Context | Cache controls | OpenCode fixtures | OpenClaw text / workspace / recovery |
| --------------- | ------: | -------------: | ----------------: | ------------------------------------ |
| Gemma 4 26B-A4B |  16,384 |            8/8 |               3/3 | Pass / pass / pass                   |
| Gemma 4 31B     |  12,288 |            8/8 |               3/3 | Pass / pass / pass                   |

Both use F16 KV, MTP/thinking off and one slot. OpenCode runs the same
rate-limiter, multi-file cart and Python CLI fixtures with 16 steps, 2,048
output tokens and 900 seconds per fixture. OpenClaw retains 300 seconds per
task; its workspace checks the protected input/test hashes, exact random
receipt and an independent test execution. All twelve real-client task checks
pass. Thirty complete native HTTP continuations include the before/after
replays and the separate Gemma MoE disable control. Package and hardware
identities stay unchanged through cleanup.

The development server enables canonical prefix rewind by default only for
Intel Xe2 Gemma MoE. Gemma 31B keeps its previous default path. Edited system
messages, early edits before the old sliding window, shortened prompts/history,
identical retries, context growth, intervening requests and disconnects all
match full-prefill output. The late-system-edit control takes 646.9 ms with
reuse versus 27,276.0 ms with full prefill; an early edit replays almost the
whole prompt. These are same-server cache controls. Uncached engine throughput
uses its separate matched protocol.

Earlier failures remain in the record: an OpenClaw preparation failure, a stop
at the global client-version gate, and an interrupted nine-fixture attempt.
That attempt's additional repository-maintenance task exits successfully but
still fails four independent tests. Three passing fixtures establish bounded
acceptance, with other coding tasks retaining their actual outcomes.
The [seven-model inference replay](/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json)
covers the catalog separately. These development-package checks do not
recertify the published 0.1.0 archives.

## November 7 candidate: OpenCode and OpenClaw

The [seven-model client follow-up](/zinc/validation/intel-client-acceptance-2026-11-07.json)
checks the Q8 decode candidate's exact CI Vulkan archive with actual
**OpenCode 2.0.21** and **OpenClaw 2026.11.1**. All fourteen model/client rows
pass frozen native HTTP continuations before and after the client, with a
healthy server afterward. Independent coding acceptance remains separate:

| Model            | OpenCode fixtures | OpenClaw text / workspace / recovery |
| ---------------- | ----------------: | ------------------------------------ |
| Qwen 3.5 9B      |               3/3 | Pass / pass / pass                   |
| Gemma 4 26B-A4B  |               3/3 | Pass / timeout / pass                |
| Gemma 4 31B      |               3/3 | Pass / timeout / pass                |
| Qwen 3.6 27B     |               3/3 | Pass / pass / pass                   |
| Qwen 3.6 35B-A3B |               3/3 | Pass / pass / pass                   |
| Qwen 3.8 27B     |               3/3 | Pass / pass / pass                   |
| Qwen 3 0.6B      |               0/3 | Fail / fail / fail                   |

OpenCode uses three independent fixtures, 16 steps, 2,048 output tokens and
900 seconds per fixture. OpenClaw uses exact text, a real file-edit/test/receipt
task and recovery, with 1,536 output tokens and 300 seconds per task. Both use
one slot, F16 KV, MTP off and thinking off. Gemma 31B uses 12,288 context;
other models use 16,384. No supervisor edits, response repair or deadline
extensions enter these results. The starter's zero OpenCode exit does not
count as success when its independent tests fail.

The initial OpenClaw helper does not inject a session ID. OpenCode already
uses automatic session IDs on qualified attention-only Vulkan models, and
its observed reuse is retained per row. The original campaign's
`ZINC_PROMPT_CACHE=0` setting does **not** disable Vulkan chat-session reuse;
these client timings are not a matched cache-off engine comparison. The raw
performance suite uses separate uncached `/v1/completions` requests.

The capability-gated OpenClaw session-reuse follow-up changes Gemma 31B's
workspace outcome to **pass in 101.23 seconds**, within the same 300-second
limit. Its text and recovery checks also pass. Gemma 26B still times out on
the workspace task. Both Gemmas pass cached-versus-full-prefill tool-history
checks beyond their 1,024-token sliding window and after an intervening request;
all frozen native checks before and after the client remain correct. Actual
client reuse is observed on both models. The original failed rows remain in
the report; session reuse does not qualify Gemma 26B's workspace acceptance.

The [separate native Vulkan controls](https://zolotukhin.ai/zinc/validation/intel-native-client-controls-2026-11-07.json)
use the same pinned client versions, GGUFs, contexts and task limits, with
llama.cpp `9f0e5098` from the earlier recorded campaign. Both Gemmas pass
OpenClaw, including workspace tasks in 73.56 seconds (26B) and 140.33 seconds
(31B). The starter fails both clients' coding acceptance on the native GPU
server too. Every native before/after continuation remains identical and each
server stays healthy. These are bounded task controls, with different native
templates and cache behavior, not a matched engine-speed comparison.

The first OpenClaw control stopped before launching the client because native
`/v1/models` lacks ZINC's context-capacity field. Those launch failures remain
in the report and do not count as model failures. The corrected discovery reads
actual native `/props` capacity, slot count and alias, plus health and model
identity. The report includes the discovery adapter source and its metadata
tests; inference responses, the actual client and independent verifier remain
unchanged.

This is development candidate evidence; it does not recertify the published
0.1.0 archives or establish a broad coding benchmark or a performance win.

## November 2 package and corrected launcher

The [complete November 2 matrix](/zinc/validation/intel-opencode-packaged-default-2026-11-02.json)
uses one exact CI Vulkan archive and actual OpenCode 2.0.21. All seven
models pass the default CLI/HTTP reference gates; API probes pass
**203/203**, and coding fixtures pass **18/21**.
The table reports completed clients with passing independent tests, protected
files intact, a responsive server and correct post-client native replays.

Gemma 31B uses 12,288 context tokens; every other row uses 16,384. Coding
counts require all acceptance conditions above, including a successful client
exit.

| Model            |   API | Coding |
| ---------------- | ----: | -----: |
| Gemma 4 31B      | 29/29 |    3/3 |
| Gemma 4 26B-A4B  | 29/29 |    3/3 |
| Qwen 3.6 35B-A3B | 29/29 |    3/3 |
| Qwen 3.6 27B     | 29/29 |    3/3 |
| Qwen 3.5 9B      | 29/29 |    3/3 |
| Qwen 3.8 27B     | 29/29 |    3/3 |
| Qwen 3 0.6B      | 29/29 |    0/3 |

All 21 same-server recovery requests pass in **158–739 ms**, and all 63
post-client HTTP continuations pass their frozen native references. The JSON
retains individual contexts, timings and fixture outcomes.

The numerical shader suite passes **376/376** cases. The seven short native
checks and two extended Gemma checks compare **432 choices: 431 exact and
one policy-qualified near-tie**. The latter is Qwen 3.6 35B-A3B's second-ranked
choice with a 0.0155 log-probability gap; the native reference independently
qualifies its remaining suffix. No tolerance changes. Both Gemmas also pass
complete HTTP continuations beyond their 1,024-token window. Qwen 3.8 passes
all 48 native choices with the new Intel default and no MTP override.

Each task starts a fresh single-slot server, with 16 steps, 2,048 output
tokens and 900 seconds. Temperature is zero, top-p is one and thinking is
disabled. The client and independent verification use the fixture directory
for both `cwd` and `PWD`. No supervisor edits, repaired calls, synthetic model
responses or deadline extensions enter this matrix. Cache misses use full
prefill; automatic reuse is advertised only for attention-only models.

**Qwen 3 0.6B retained failures:**

- `rate-limiter-single`: acceptance conditions not met; independent tests 0 pass / 1 fail, protected-file violations 0.
- `cart-multi-file`: acceptance conditions not met; independent tests 1 pass / 2 fail, protected-file violations 0.
- `python-log-cli`: verification exits 5 with no tests run after a protected-file change; protected-file violations 1.

Qwen 3.5 9B is the smallest successful coding model in this matrix. Its three
tasks take 34–84 seconds. The starter exits zero without satisfying acceptance:
its limiter implementation removes the required export after an invalid edit
and an unavailable `report` call; its cart attempt reads repeatedly without a
fix; its Python attempt changes a protected file. None counts as coding success.
The [separate actual-client native control](/zinc/validation/intel-starter-native-opencode-control-2026-11-02.json)
also fails all three tasks on the same Intel GPU and pinned GGUF. It uses the
same frozen client, 16K context, 16 steps, 2,048 output tokens and 900-second
limit, with native Vulkan F32 KV and reasoning disabled. Its acceptance tests
do not improve from baseline; recovery and all nine post-client native
continuations pass. Both engines fail these bounded tasks. Their chat templates
are not claimed to be token-for-token equivalent, and the native control does
not erase ZINC's failures. The starter remains a qualification model without
a coding recommendation.

Actual OpenCode history reuse is observed on both Gemmas: 31B records 12 hits,
reusing up to 7,473 tokens, and 26B records three hits, up to 5,573 tokens. Misses
use full prefill. The hybrid Qwens advertise no automatic reuse; their completed
coding tasks exercise that conservative path.

The package is [CI run 41382805638](https://github.com/zolotukhin/zinc/actions/runs/41382805638),
version `0.1.0-dev.41382805638`, source
`83b22e7b708f80bdb93d8727535ca6bd62677f6c` and archive SHA256
`b83dd9823cc59cdd8eaf1dbb52d9fca91241cc17155655b8dec00b4645cd4629`. The JSON retains the
binary, validation-source, frozen-client, model and native-reference identities.
All attempts complete and cleanup leaves no GPU process. This SSH diagnostic
does not certify a final release candidate or establish general coding ability.

## Earlier client configurations

The November 1–2 checks on an Intel Arc Pro B70 exercise all seven catalog models
with actual OpenCode 2.0.21. They exposed client compatibility, request recovery,
Gemma cache allocation, long-prompt attention and working-directory defects.
A model that loads or answers a short prompt has not yet passed a coding task.

The earlier client runs below inherited a stale `PWD`. Their individual
outcomes remain observations of that launch; they do not qualify the corrected
configuration. The [actual-client working-directory control](/zinc/validation/opencode-working-directory-protocol-2026-11-02.json)
fails with relative tools before the fix and passes on both pinned client
versions afterward. The latest seven-model matrix above uses one exact CI archive, the
corrected launcher and independent tests; its results stand separately.

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

Use a build containing the corrections recorded for the selected result.
The published 0.1.0 archives and later development candidates have separate
qualification evidence. The catalog's tested-profile promotion is separate
from these source-specific diagnostics; an untested profile warning remains possible. An explicit `--model-id` loads the installed
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
reference tokens for each model. This allocation-only executable passes
**28/29 API checks on each Gemma**: the 9.2K long-prompt check fails. That check
was absent from the earlier 8K 28/28 run because the request could not fit its
context. The [allocation follow-up](/zinc/validation/intel-gemma-allocation-followup-2026-11-01.json)
retains the failure; the [long-context negative control](/zinc/validation/intel-gemma-long-context-control-2026-11-01.json)
compares the repetitive output with the native answer. Actual coding tasks then expose repetitive
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
uses the later window-corrected `1dee8061` Vulkan source snapshot, unchanged
model pins, default F16 KV and
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

The 29/29 results apply to this later executable. They do not clear the
allocation-only executable's 28/29 failure or combine different binaries into
one qualification.

## Pre-fix budget-only control and the inherited directory

The **pre-fix budget-only** [1,800-second Gemma 31B Python control](/zinc/validation/intel-gemma31-python-longer-2026-11-02.json)
uses the stale-`PWD` launcher. It finishes in 1,752 seconds, with 4/4 independent tests and clean
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

## The NextN model exposed another missing gate

The [first corrected-launch package preflight](/zinc/validation/intel-qwen38-mtp-controls-2026-11-02.json)
passes 376 numerical cases and five models, including both Gemmas beyond their
window. It then stops at Qwen 3.8's lighthouse continuation: the seventh generated token (index six) differs
from the native reference and is not an allowed near-tie. No coding fixtures
start in that failed run; cleanup completes.

Against the same frozen native references, ordinary decode with `ZINC_MTP=0`
and normal F16 KV matches all 48 choices. F32 KV, disabled column matmul and
disabled chained submissions retain the failing MTP continuations. A separate
[R9700 Vulkan control](/zinc/validation/rdna4-qwen38-mtp-controls-2026-11-02.json)
matches all 48 choices with MTP both on and off. The default correction is
therefore scoped to Intel, preserving the passing AMD path.

Qwen 3.8 was absent from the earlier six-model Vulkan qualification roles.
Both roles now require it. A positive overridden control is not a pass for the
old default package. The fresh package passes all seven default CLI/HTTP model
gates and both extended Gemma windows: 431 of 432 native choices match exactly,
and one meets the unchanged near-tie rule. Qwen 3.8 matches all 48 exactly.
The completed default-package matrix passes all 18 coding tasks across the
six larger models. The starter fails its three tasks while retaining 29/29 API
checks, recovery and native-reference replays; those failures remain explicit.

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
