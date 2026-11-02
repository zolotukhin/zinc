# Intel OpenCode validation

The November 1 checks on an Intel Arc Pro B70 found two separate compatibility
problems and a server recovery defect. They do not yet establish that every
catalog model completes coding tasks.

OpenCode 2.0.21 removed the CLI's `--dir` option and changed plugin registration.
It also sends `max_completion_tokens`, which the old server ignored. A request
for one token generated 256; both budget spellings now generate one. The
[client compatibility evidence](/zinc/validation/opencode-client-compatibility-2026-11-01.json)
retains the live negative control and the actual OpenCode 1.19.4/2.0.21 CPU
protocol checks. Scripted CPU responses test client behavior without a model
coding claim.

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
