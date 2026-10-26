# OpenCode repository end-to-end trial

On 2026-10-26 we ran actual OpenCode 1.19.4 coding sessions against Qwen 3.8
27B Q4_K_M served by Zinc on a Radeon AI PRO R9700. Three tasks ran in fresh,
complete Zinc worktrees under each of two server configurations. This follows
the [original failed worktree trial](OPENCODE_WORKTREE_REPORT.md) and the
[reviewed preview feature](ROCM_AGENT_CONTEXT.md#repository-coding-result).

The native-capacity Q8 runs passed the frozen functional checks for all three
tasks. The largest reported coding input reached **46,520 tokens**, with no
API errors. Review still found incomplete delivery in all three: regression
tests were missing, documentation was missing for two, and automatic model
selection had an edge-case bug. These are functional passes, not three fully
completed assignments.

The [portable result JSON](/zinc/validation/opencode-e2e-2026-10-26.json)
records per-run outcomes, hashes, tool counts, load and cleanup. The original
acceptance remains frozen; manual completion review is recorded separately.

## What actually ran

- Source: `571b85182953c7af761cdd41e019252c5fee48c4`, before any of these fixes.
- Serving runtime: `59e91681e097e76015be7cb34223b93d28b60a7e`. Its frozen
  inference source matched the source checkout; later changes before the task
  base did not change the serving engine. The portable JSON records both.
- Model: Qwen 3.8 27B Q4_K_M, identical 17,106,775,008-byte GGUF in both modes;
  SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Zig 0.17.0, Bun 1.4.4 and OpenCode 1.19.4.
- Native mode: 262,144-token capacity, Q8 KV and explicit FP16-input attention.
- Control: 32,768-token capacity, FP32 KV and FP32 attention.
- Both: one request slot, MTP and exact-prefix prompt reuse enabled; 3,072 output
  tokens, 24 configured agent steps, up to four verified editing rounds, five
  minutes per editing round and twenty minutes per task.

The same frozen helper capped each verified editing round at eight model
steps. Source reads were bounded to 80 lines and tool output was shortened by
the public context hook. Checks ran outside the model conversation and returned
bounded failure diagnostics to the same session. Neither the supervisor nor
the trace proxy edited a candidate during the trial. The proxy forwarded actual
requests and responses without synthetic answers, tool calls or repairs.

Each task had its own worktree and isolated OpenCode session directories.
The model could edit only the specified source/docs and append tests; it could
not change the external acceptance, verifier, generated config or context hook.
All three external functional checks failed on the original source before
inference, while the existing CPU suite passed. Relevant unit tests and the
full CPU project suite ran after functional acceptance; the Zig task also
required CUDA host compilation.

## Tasks and results

The tasks were actual defects or useful changes in Zinc:

1. **Strict benchmark integers:** reject partial parses, fractions, exponents,
   signs, whitespace, hexadecimal and unsafe integers. Preserve leading zeroes,
   defaults and each flag's zero rule. Eighty external cases covered the CLI.
2. **Discovery UTF-8:** return `error.InvalidUtf8` before allocation for an
   invalid model basename, preserve valid Unicode/escaping and allocator errors,
   and ignore private parent-directory bytes. Independent Zig checks covered
   malformed sequences, valid names and allocation failure.
3. **Automatic setup context:** add setup-only `--context auto`, discover live
   server capacity with one authenticated request, preserve portable config and
   numeric offline setup, and fail before writes on invalid metadata or timeout.
   Twenty-two acceptance categories included actual Zinc discovery.

Every assignment also requested regression tests and documentation. The
[frozen manifest and acceptance files](validation/opencode-e2e-2026-10-26/README.md)
retain the complete prompts and original checks.

| Mode     | Task                    | Elapsed | Max reported input | Frozen functional check / review               |
| -------- | ----------------------- | ------: | -----------------: | ---------------------------------------------- |
| 262K Q8  | Benchmark integers      |   6m05s |             28,176 | Passed; tests/docs omitted                     |
| 262K Q8  | Discovery UTF-8         |   4m00s |             17,461 | Passed; tests/docs omitted                     |
| 262K Q8  | Automatic setup context |  14m10s |             46,520 | Passed; tests omitted; selection bug found     |
| 32K FP32 | Benchmark integers      |   9m55s |             25,734 | Passed; tests/docs omitted                     |
| 32K FP32 | Discovery UTF-8         |  10m50s |             22,923 | Passed; new tests do not compile; docs omitted |
| 32K FP32 | Automatic setup context |  19m05s |             31,519 | Failed; assignment incomplete                  |

Across the six runs, **5/6 passed frozen functional acceptance** and **0/6
satisfied the complete assigned contract on review**. These are different
scores. The original checks were too weak to enforce full delivery.

Reported times include editing, automatic repair and independent checks. Input
lengths come from OpenCode's `step_finish` usage events and describe actual
requests, not the configured maximum. Passthrough response previews independently
contain a 45,852-token prompt usage record for native coding. Previews are capped
at 64 KiB, so their observed maxima can omit the longest response's final usage.
Every retained trace has zero synthetic shortcuts, repairs or suppressed events. Native mode's 46.5K history demonstrates
coding transport/execution beyond 32K; it does not demonstrate 262K-token coding
quality. The [near-native retrieval](ROCM_NATIVE_CONTEXT.md) is a separate test.

## Why green checks were insufficient

The harness accepted a partial patch as soon as its functional verifier passed.
For the first two tasks, that verifier did not require the new regression tests
or documentation. Both native runs therefore ended with a correct, small
production edit and an incomplete assignment. The third required help/docs,
but still omitted tests and missed model-selection edge cases.

In automatic context discovery, the model searched for either `target` or
`target.gguf` in one `find` operation. Given a multi-model list containing the
suffixed entry first, it chose that capacity before a later exact match. With
only the suffixed entry and an unrelated second model, it accepted an ambiguous
alias. Both violated the explicit contract; neither appeared in the original
acceptance examples. New review cases reproduced both failures without model
inference. Malformed singleton IDs and duplicate IDs also needed validation.

The 32K Zig run exposed a second coverage gap. It added embedded regression
tests, but a Unicode scalar literal was placed in a `u8` array and an allocator
was constructed with the wrong type. Both the local compiler and direct node
compilation rejected them. Its original project checks had still passed:
those build paths had not analyzed the embedded tests in that configuration.
Compiling the serializer function through external checks or a host executable
was insufficient to compile its test declarations. The landing build now adds
a dedicated, backend-independent `model-discovery-tests` CPU target to both
`build test` and `test-build`.

The original acceptance results remain intact. Manual review is a separate
score, and later reviewer changes do not retroactively turn the assignments
into complete model deliveries. Requiring a passing check is useful only when
the check covers the whole contract; patch review still matters.

## What landed after review

The reviewer retained the two correct native bug fixes and added focused
regressions and documentation. The numeric tests fail on the original parser;
the UTF-8 tests expose the original array-valued ID and allocation-error order.

For automatic setup context, the reviewer adopted the native patch, removed
suffix inference from multi-model selection, validated model IDs/duplicates,
and sanitized lookup failures. New permanent tests cover the original checks,
exact-match precedence, ambiguity, malformed IDs, large output overrides,
portable files, setup-only command order and a stalled response body. Numeric
setup remains synchronous and offline. Those repairs and regression tests
were reviewer-authored, not produced by the evaluated model.

Use it with a server already running at the capacity you want:

```bash
bun tools/setup_opencode.mjs --project ../my-project --context auto --dry-run
bun tools/setup_opencode.mjs --project ../my-project --context auto
bun tools/setup_opencode.mjs check --project ../my-project
```

## Test setup and limits

Two preliminary setup attempts were stopped and preserved before the final
batch. First, a dependency-directory symlink was incorrectly treated as a new
file by the change guard. Fresh real dependency directories fixed that.
Second, GPU-server tuning variables leaked into the client/test environment;
a pre-existing unit fixture expected its own prompt-cache value and failed.
The final batch isolated client and GPU-server environments. Task prompts and
functional acceptance did not change. Neither setup failure is counted as a
completed coding run or hidden as a model success.

All native tasks ran before all control tasks, once per task/configuration.
This is a small, ordered capability trial, not a randomized coding benchmark
or a general success rate. Capacity, cache representation and attention
precision changed together. The matrix also predates later upstream BOS and
Vulkan shared-cache fixes merged while this trial was running; its runtime
hashes identify the tested engine. The shared host retained eight unrelated CPU
workers; load was recorded per run and agent/check work used lower priority.
A lower-priority, two-job compilation of the later adoption build overlapped
part of the 32K Zig trial. Wall times cannot isolate a KV-format speedup or
support a new tokens/s claim.

Existing CPU tests passing do not establish GPU correctness or task completeness.
Separate [GPU, cache, API and precision validation](ROCM_NATIVE_CONTEXT.md)
remains relevant. No local workstation model inference was used. Public
artifacts contain portable metadata and hashes; raw traces, session state,
endpoints and host-specific launchers remain private.

All six model/proxy/wrapper groups were stopped after the batch. Temporary
power settings were restored, with no inference process remaining. This
cleanup was recorded before the separate adoption smoke test started.

## Separate test of the revised setup

After the matrix, a clean ROCm build at
`cb7832df0311964c704633664e3fdaa9054bf739` included the reviewed production fixes
and newer upstream changes. Its binary SHA-256 was
`728294b8415b7633e8b4544435f5e8d64783fd6b1ce83ac753d043f8f5181bbf`.
The standalone helper's SHA-256 was
`ef526fbcdf4007c58ea77eb1c0043c15675c5e432f74d52456fd6598210fc6f0`.
Later test-target and documentation changes do not alter that serving code.

The helper discovered 262,144 tokens through actual `--context auto`, and its
preview exactly matched its written portable configuration. Live preflight
passed, followed by **23/23 API checks**, covering JSON/SSE parity, typed tool
arguments, tool history, invalid input, context overflow, queued requests,
cancellation and a prompt beyond the former 8K SSM limit.

We then extended the existing `python-log-cli` fixture with explicit regression
and README deliverables. The original four tests were protected. The stronger
verifier required two added tests, documentation, passing candidate checks and
failure of the new tests against the original source. This was an adaptive
follow-up with a stronger contract, not another run in the six-task matrix.

In **70.006 seconds**, one editing round made two reads, three writes and one
Bash call. The model produced the Python fix, two regressions and portable usage
docs. All six tests passed; both new tests failed against the original source.
The largest reported input was 7,804 tokens, so this checks the actual setup and
coding flow at native allocated capacity, rather than long-history coding.

Manual review still rejected the initial patch as complete: [Python's default JSON decoder](https://docs.python.org/3/library/json.html#infinite-and-nan-number-values) accepts `NaN`, `Infinity` and `-Infinity`. The CLI consequently
returned success and emitted those non-JSON values. Additional external cases
reproduced the violation. The same OpenCode session received that diagnostic
for one bounded repair round, with append-only tests and stricter checks.

The resumed session passed the repair in **115.008 seconds**, with three reads,
four edits and two Bash calls. Seven unit tests, six external nonstandard-number
cases and the original-source negative control passed. Existing tests and
generated configuration remained unchanged; the supervisor made no implementation
edits. The highest reported input was 11,715 tokens. Both launches cleaned up
all owned processes and restored temporary power settings.

This establishes a reviewed, bounded fixture result. Ordinary finite numeric
amounts were exercised; numeric overflow and general record-schema validation
were outside its acceptance. It is not a production-parser audit or a new
autonomous repository-coding success rate. The six original outcomes above
remain unchanged.
