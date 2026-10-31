# Dense Gemma ROCm bounded prefill

Dense Gemma can load its 8K KV cache on the 32 GB R9700 while leaving too little
memory for whole-prompt prefill scratch. The previous allocation failure selected
serial prefill. A captured 5,037-token OpenCode request spent about 199 seconds in
that fallback before producing its first response.

The new ROCm dense path retries smaller scratch allocations **before executing
any prompt layer**. It halves capacities above 32 tokens until allocation succeeds, stopping at
32; a smaller requested capacity is tried once. Only
device-buffer allocation failures are retryable. Host allocation, execution and
shape errors retain their original cause, and an exhausted minimum allocation
still propagates to the existing caller recovery path.

## Keep history across chunks

Each chunk embeds its own input rows while retaining absolute RoPE positions.
Temporary non-owning views select the destination range for K/V writes. Attention
reads the complete original cache and receives each query's absolute position,
so causal limits and sliding windows include the preceding chunks. Projection
and FFN work remains batched. Only the final chunk runs the output head.

Position/slot metadata costs eight bytes per scratch row (about 2 KiB at the
observed 254-row capacity). Partial scratch allocations retain their existing
rollback path. Attention checks static plus dynamic shared memory against the
device limit before dispatch; its absolute query span includes the earlier KV.

The implementation reuses the existing batched-position attention kernel. A
first development version dispatched one attention query at a time and timed
out at the near-capacity request; that failed result remains separate.
No numerical GPU kernel source, GGUF parsing, model hash or reference agreement
policy changes in this follow-up.

The ROCm dense quantized matrix multiplication (MMQ) path also stops reserving
an unused **220.5 MiB** fp16 weight
workspace. CUDA and explicit BLAS thresholds below the MMQ boundary retain that
workspace. CUDA, MoE and Muse retain their existing prefill routing, and an
ordinary whole-prompt batch remains in use when its scratch fits.

`ZINC_PREFILL_CHUNK=<tokens>` optionally caps scratch capacity for dense ROCm
Gemma, as it already does for Qwen. The default first tries the whole prompt.
Zero, malformed and overflowing values keep that default; a positive value
cannot expand a short prompt. Smaller chunks trade more weight passes for lower
scratch usage. This setting does not increase the KV context capacity.

## Regression coverage

Five CPU tests cover successful allocation, the 5K coding-prompt retry sequence,
minimum-capacity exhaustion, invalid chunk settings and non-retryable errors.
Both an old single-allocation policy and an incorrect catch-all retry policy
fail these regressions. The full CPU unit suite, Linux CUDA host compilation and
46 release-contract tests pass.

The release GPU harness now forces **16-token chunks** for the Gemma 31B model
row and compares every frozen continuation token and its complete text with the
same reference. Short final chunks and nonzero absolute positions are exercised
even on a GPU with enough free scratch for the ordinary full prompt. CPU
negative controls reject skipped chunking, changed IDs, truncated text and a
failed process. The required release context remains **4K/F32/single-slot**.

## Packaged model checks

The [three-target packaging run](https://github.com/zolotukhin/zinc/actions/runs/41247994350)
built code commit `11519aaa`. The exact extracted ROCm archive passes all six
required model rows: **288/288 exact CLI choices**, **72 tokenizer cases** and
**54 complete HTTP reference continuations** through initial, cancellation and
context/error recovery. Qwen 3.8's default MTP is observed active; MTP-off also
matches all 48 choices. References remain HIP for Qwen and Vulkan for Gemma,
with the same pinned commit and serial/F32 generation settings recorded per row.

| Model                | Reference backend | CLI choices | Complete 4K API gate |
| -------------------- | ----------------- | ----------- | -------------------- |
| Qwen 3.5 9B          | HIP               | 48/48       | Pass                 |
| Qwen 3.6 dense 27B   | HIP               | 48/48       | Pass                 |
| Qwen 3.6 35B-A3B MoE | HIP               | 48/48       | Pass                 |
| Qwen 3.8 27B/MTP     | HIP               | 48/48       | Pass                 |
| Gemma 4 26B-A4B      | Vulkan            | 48/48       | Pass                 |
| Gemma 4 31B          | Vulkan            | 48/48       | Pass                 |

Gemma 31B additionally passes **48/48 forced-small-chunk choices** at 4K and the
complete API gate at 8K, including nine more full reference continuations. The
8,128-token boundary request uses 254-token chunks and completes prefill in
86.8 seconds in this packaged diagnostic. Recovery stays numerically correct.

A separate 4,861-token prompt stores a code at the beginning and another in the
middle. The exact same prompt IDs, 8K/F32 single-slot configuration and 32-token
budget are replayed through fresh servers:

| Package             | Request seconds | Both codes returned | Full reference text  |
| ------------------- | --------------: | ------------------- | -------------------- |
| Previous `8f957952` |         192.784 | Yes                 | Different formatting |
| Bounded `11519aaa`  |          26.418 | Yes                 | Exact match          |

These are single cold-request diagnostics, not a headline throughput benchmark.
The original aggregate replay remains failed because it includes the baseline's
text mismatch; the candidate's exact match is reported separately. The frozen
reference and an earlier tokenizer-output capture failure retain their identities.

## Actual OpenCode result

The same extracted archive passes **23/23 API checks** and **3/3 core coding
fixtures at 8K** with OpenCode 1.19.4. All ten independent acceptance tests pass.
Each fixture has a fresh server, thinking off, temperature zero, a 2,048-token
output cap, 16 steps and a **600-second limit**:

| Fixture           | Independent tests | Real tool calls | Seconds |
| ----------------- | ----------------- | --------------: | ------: |
| Rate limiter      | 3/3               |               7 | 327.095 |
| Multi-file cart   | 3/3               |              10 | 358.622 |
| Python NDJSON CLI | 4/4               |               5 | 270.286 |

No API errors, timeouts, protected-file changes, unexpected files, tool repairs
or supervisor edits occur in these runs. The shorter **300-second** development
trial remains failed: seven tool calls, 0/3 tests and a timeout. The ten-minute
result has its own budget and executable identity; it does not replace that
failure or establish larger repository delivery. Existing native tool-string
limits remain in the [OpenCode validation report](OPENCODE_RELEASE_VALIDATION.md).

The [portable evidence](/zinc/validation/rocm-gemma-bounded-prefill-2026-10-31.json)
keeps the prototype timeout, failed shorter coding trial, normal/forced/8K model
checks, long-history baseline and final coding runs separate. It records
executable, archive, GGUF, kernel and harness digests. Every owned GPU/model and
client process is stopped after validation.

CPU-only tests and packaging now run for stacked PR bases too. The actual
stacked PR schedules all four test optimization modes, macOS/Nix builds and the
three archive jobs. Protected candidate/signing/publication workflows retain
their existing main-branch restrictions.

## Validation scope

All GPU/model work runs remotely on one R9700. Preview packages are unsigned;
CPU packaging on Metal and Vulkan does not qualify their GPU hardware. The full
release hardware matrix, registered workers, signing and protected promotion
remain separate gates. An 8K inference pass does not by itself qualify OpenCode
coding or promote a model in the catalog.
