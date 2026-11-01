# ROCm Qwen short-prompt prefill agreement

The subsequent [single-slot consistency follow-up](ROCM_SINGLE_SLOT_CORRECTNESS.md)
fixes the MoE HTTP mismatch and a request-reuse issue found by stronger checks.
The original report below retains its failed MoE row and exact package identity.

Dense hybrid Qwen CLI and debug runs on ROCm now use per-token prefill below
32 tokens unless MTP is prepared. Longer prompts retain batched prefill. This fixes a release reference mismatch
without changing attention or SSM kernels, disabling batching globally, or
relaxing the agreement policy.

## Failure and isolation

The package from [run 40540857233](https://github.com/zolotukhin/zinc/actions/runs/40540857233)
contains the same source tree as merged PR #35. Qwen 3.6 dense 27B matched
47/48 reference choices. On the 19-token refrigerator prompt, generated token
16 was 11 instead of 4424. The reference log-probability gap was **0.21713**,
above the existing **0.05** near-tie limit. The canonical HIP oracle also
rejects that choice, with a **0.08444** gap; changing reference backends does
not resolve the failure.

The three frozen release prompts contain 18–22 tokens. At these lengths,
dense prefill projections already use per-token matvecs. Batched attention and
SSM arithmetic still differ from serial prefill. Disabling batching restores
all 48 reference choices, but applying that setting to every prompt makes the
4K request exceed the harness timeout.

An attention-only change restored the failing 27B case but regressed 9B.
The fix therefore keeps the established numerical kernels and selects serial
prefill only for short dense hybrid Qwen CLI/debug prompts. The HTTP scheduler
already uses a 32-token batching threshold for its ordinary slot path.

## Routing contract and regression tests

Production CLI and the debug harness share
[`prefill_policy.zig`](https://github.com/zolotukhin/zinc/blob/main/src/compute/prefill_policy.zig).
Its CPU tests run through `./scripts/zig.sh build test`, without loading a model
or opening a GPU. They cover:

- The reported 18/19/22-token prompts and the 31/32/33-token boundary.
- Batched 4,032/4,096-token inputs, preventing a return to global serial prefill.
- Prepared MTP, which requires batched hidden rows for priming, and an explicit
  batching opt-out taking precedence.
- Existing MoE, Gemma and CUDA routing, plus empty and single-token inputs.

Comments at the policy and both call sites explain why short dense prompts
need serial arithmetic, why MTP is retained, and why long prompts must remain
batched. A negative control replacing the routing decision with the previous
always-batch behavior fails the short-prompt test.

The existing packaged release harness also acts as the hardware regression:
its fixed GGUF hashes, three prompts, full 16-token continuations, and 4K API
boundary check exercise both the numerical failure and the timeout constraint.
Do not replace those prompts or widen the near-tie limit to accept a mismatch.

## Packaged and OpenCode results

The [packaging run](https://github.com/zolotukhin/zinc/actions/runs/40558175678)
built fix commit `dc89acae` as synthetic merge `abd76520`, with an identical source
tree. Checks ran from its extracted ROCm archive, without rebuilding the
executable or overriding its library path. The
[portable result JSON](/zinc/validation/qwen27-prefill-2026-10-27.json) records all
binary/model/reference hashes, original failures, controls, and remaining limits.

| Model                | CLI reference choices | Complete 4K API gate               |
| -------------------- | --------------------- | ---------------------------------- |
| Qwen 3.6 dense 27B   | 48/48                 | Pass                               |
| Qwen 3.5 9B          | 48/48                 | Pass                               |
| Qwen 3.6 35B-A3B MoE | 48/48                 | Blocked: HTTP continuation differs |
| Qwen 3.8 27B/MTP     | 48/48                 | Pass                               |

Qwen 3.8 also matches all 48 choices with MTP disabled. The debug harness matches
all 16 refrigerator-prompt choices for dense 27B and 9B. Dense 27B's 4,032-token
API boundary prompt prefills in **6.33 seconds** and the full gate passes. This
is one latency diagnostic, not a server throughput comparison.

At 16K, the same packaged dense 27B executable passes **24/24 OpenCode API checks**
and **3/3 coding fixtures**, with all ten independent acceptance tests passing.
The rate limiter, cart and Python NDJSON CLI used six, seven and four tool calls,
respectively. Frozen tests failed before the tasks; protected files and generated
configuration stayed unchanged. Each fixture used a fresh server, 4,096 output
tokens, 16 steps and a 300-second limit, with thinking off and temperature zero.
This is separate developer evidence from the required 4K release row.

The MoE HTTP failure is **pre-existing**, reproduced with the unchanged #35
package and the same GGUF, prompt and reference. Its refrigerator continuation
says “box” where the HIP reference and CLI say “fridge.” The aggregate Qwen API
matrix remains failed in the JSON. That preview did not qualify the HTTP row.
The subsequent [single-slot follow-up](ROCM_SINGLE_SLOT_CORRECTNESS.md) fixes and rechecks it;
a correct CLI alone does not qualify HTTP.

## Reference scope

ROCm generation uses the release worker's canonical HIP llama.cpp build at
policy-pinned commit `5f436dddb440a288ee5611d7d1eca564a6aca9f4`, with batch 512,
micro-batch 1, one slot, F32 KV, and flash attention disabled. Tokenizer checks
use the same commit's tokenizer executable. Reference executable hashes are
recorded separately.

Separate Vulkan-oracle diagnostics show why the reference backend must be
identified: the MoE model's refrigerator choice differs between HIP and Vulkan
by more than the allowed near-tie gap even in the unchanged baseline package.
This follow-up preserves that diagnostic failure and uses the existing worker
reference consistently across the four Qwen model rows.

These are developer checks on one R9700 board. Full packaged candidate
qualification, the complete required hardware matrix, signing, and protected
promotion remain release gates. No local workstation GPU/model inference is
part of this follow-up.
