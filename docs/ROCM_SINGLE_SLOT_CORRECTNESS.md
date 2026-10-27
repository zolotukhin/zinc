# ROCm single-slot continuation consistency

This follow-up fixes two request-path differences in F32, single-slot ROCm
serving: short MoE prefill differed from CLI, and dense-model continuations could
change after a previous request allocated batch scratch. Numerical kernels,
GGUF parsing and the release agreement policy are unchanged.

## Short MoE prefill

The merged #36 package matches Qwen 3.6 35B-A3B's reference in the CLI, but HTTP
says “box” where CLI and the canonical HIP reference say “fridge” on the same
19-token prompt. The HTTP scheduler uses per-token prefill below 32 tokens;
CLI batches these MoE prompts with the warp SSM scan.

Enabling slot batching alone still forces the block scan. The development
controls reproduce the mismatch with either serial prefill or the block scan;
batched prefill with the CLI's warp scan matches all three full continuations.

One shared predicate now selects both short slot batching and the SSM scan.
It applies only to ROCm MoE with one slot, F32 KV and the B1 decoder enabled,
for 2–31 prompt tokens. Long inputs retain their previous path. The shared
32-token threshold and tests prevent the scheduler and scan choices from
separating again.

## Decode arithmetic after scratch growth

The new recovery checks exposed a second issue in dense 27B and 9B. Their
initial continuations matched the reference, including after cancellation.
After a long request, a short request could choose a different continuation.
A 64-token warmup already reproduced it.

Batch scratch was allocated lazily. Decode dispatchers treated its presence as
permission to use packed-Q8 activations and fused normalization. Consequently,
a retained allocation could change the next request's arithmetic even though
its recurrent state was reset. Disabling those decode paths removed the drift
in both diagnostic controls.

The slot now selects decode eligibility per request. Reset starts the serial
path; batched prefill enables the existing packed fast path. All decode
consumers use that permission together with the allocation check. Scratch stays
cached, so this does not add allocation/free churn. CLI, CUDA, Q8 KV and
multi-slot serving retain their existing selection behavior. MTP still uses
batched prefill and its packed decode path.

## Regression coverage

The shared host policy in
[`prefill_policy.zig`](https://github.com/zolotukhin/zinc/blob/main/src/compute/prefill_policy.zig)
has seven CPU tests. They cover the frozen short-prompt lengths, 31/32/33-token
boundaries, the short–long–short scratch transition, MTP, explicit opt-out,
other backends/KV formats, and concurrent serving. The old short-slot policy
and the old scratch-only policy each fail their corresponding new test.

`./scripts/zig.sh build test` runs these tests without opening a GPU or model.
The Linux CUDA host check and 44 CPU release-contract tests also pass.

The packaged hardware harness keeps the frozen model hashes, three reference
prompts and full 16-token continuations. Its HTTP gate now checks those
continuations in three phases:

1. On the initial server.
2. After disconnecting an active stream and recovering the slot.
3. After the exact near-capacity prompt, malformed JSON, oversized input and
   recovery.

A nonempty response or healthy server cannot substitute for numerical parity.
The first MoE-fix preview passed its own gate but failed the new post-context
checks for both dense models; those failures remain separate in the evidence.

## Packaged results

The [packaging run](https://github.com/zolotukhin/zinc/actions/runs/40574592584)
built code commit `8f957952` as synthetic merge `3fa5014b`, with an identical source
tree. The extracted ROCm archive was tested without rebuilding its executable or
overriding the library path. All three platform packaging jobs passed.

| Model                | CLI reference choices | Complete 4K API gate |
| -------------------- | --------------------- | -------------------- |
| Qwen 3.5 9B          | 48/48                 | Pass                 |
| Qwen 3.6 dense 27B   | 48/48                 | Pass                 |
| Qwen 3.6 35B-A3B MoE | 48/48                 | Pass                 |
| Qwen 3.8 27B/MTP     | 48/48                 | Pass                 |
| Gemma 4 26B-A4B      | 48/48                 | Pass                 |
| Gemma 4 31B          | 48/48                 | Pass                 |

That is **288/288 exact CLI choices**, with no near-tie exceptions, and **54 full
HTTP reference continuations** across the three phases. Each model also passes
12 tokenizer corpus cases, Unicode/SSE, usage, cancellation, the exact 4K
boundary and error recovery. Qwen 3.8's MTP was observed active; disabling MTP
also matches all 48 CLI choices.

On the same final package, Qwen 3.6 MoE passes **24/24 OpenCode API checks** at
16K and **3/3 core coding fixtures**, with all ten independent acceptance tests
passing. The rate limiter, cart and Python NDJSON CLI use eight, seven and four
real tool calls. Tests failed before each task; protected files and generated
configuration stayed unchanged. Each fixture has a fresh server, a 4,096-token
output budget, 16 steps and a 300-second limit, with thinking off and temperature
zero.

The [portable evidence](/zinc/validation/rocm-single-slot-2026-10-27.json) keeps
the baseline, SSM controls, scratch controls, unsuccessful first package sweep,
and final sweep separate, with executable, archive, model and harness hashes.
All owned model/client processes were stopped after the runs.

## Qualification scope

All model and GPU work runs remotely on the 32 GB R9700. Qwen generation uses
the worker's canonical HIP reference; Gemma uses the pinned Vulkan reference
from its earlier validation. Each executable hash and reference backend is
recorded per model, at the unchanged policy commit and serial/F32 generation
settings. Tokenizer checks use the same commit's tokenizer executable.

These are unsigned preview/developer checks on one board. A final candidate
still needs the full required hardware matrix, signing, registered workers and
protected promotion. The 16K OpenCode trial is separate from the required 4K
model rows. No optional concurrency/KV claim or model catalog promotion follows
from these checks.
