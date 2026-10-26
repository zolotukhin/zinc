# OpenCode coding harness

Use OpenCode with a Qwen model served by ZINC. The
[setup script](https://github.com/zolotukhin/zinc/blob/main/tools/setup_opencode.mjs)
creates a project configuration, checks the API, and runs tasks headlessly.
It requires **Bun 1.4.4 or newer**. This workflow is tested with
**OpenCode 1.19.4**.

[Download the standalone script](/zinc/harnesses/setup-opencode.mjs), or use
`tools/setup_opencode.mjs` from a ZINC checkout. The downloaded script has no
package dependencies and accepts the same arguments.

## Start the model server

Build for [your GPU](GETTING_STARTED.md), then start a Qwen model with enough
context for source files, tool definitions, and tool results:

```bash
./zig-out/bin/zinc --model-id qwen38-27b-q4k-m -c 32768 -p 9090 --parallel 1
```

The model must fit your GPU. Qwen 3.5 9B is a smaller starting point; Qwen 3.8
27B Q4_K_M is the dense model tested on a Radeon AI PRO R9700 with ROCm. For
ROCm, follow the [backend setup](ROCM.md) and select your own device using
`ROCR_VISIBLE_DEVICES`. An existing GGUF can be supplied with
`-m /path/to/model.gguf` instead of a managed ID.

Use a server build with tool-call support. The setup check reports missing
support before you start a coding task. ZINC generates tool calls; OpenCode
reads files, edits code, and runs commands on the machine where OpenCode runs.

For ROCm repository tasks, the validated 32K configuration leaves more room for
source and tool
history. [Bounded prefill and attention](ROCM_AGENT_CONTEXT.md) have been checked
with 27B prompts up to 30,048 tokens. Optional `ZINC_PROMPT_CACHE=1` reuses exact
prefixes between single-slot Qwen requests. Neither a larger context nor a
faster prefill establishes that the model will finish a coding task.
For Qwen 3.8 27B's larger native window, see the [Q8 context guide](ROCM_NATIVE_CONTEXT.md)
and pass the server's actual capacity to setup. The 32K example is not a model limit.

## Configure OpenCode

From the ZINC checkout:

```bash
bun install --global opencode-ai@1.19.4

bun tools/setup_opencode.mjs \
  --project ../my-project \
  --model qwen38-27b-q4k-m \
  --context 32768

bun tools/setup_opencode.mjs check --project ../my-project
```

To configure a project without cloning ZINC, download the same script and run it
with Bun:

```bash
curl -fsS https://zolotukhin.ai/zinc/harnesses/setup-opencode.mjs -o setup-opencode.mjs
bun setup-opencode.mjs --project ../my-project --context 32768
bun setup-opencode.mjs check --project ../my-project
```

Use `bun setup-opencode.mjs run` for the headless commands below when using the
downloaded file.

Setup creates **`opencode.zinc.json`** and a small, versioned context hook in
**`.zinc/`** in the target project. Keep both when sharing the setup. It leaves
`opencode.json`, global settings, authentication files, and existing generated
configs alone. To update an existing setup, rerun setup with `--force`. The
helper replaces only its generated config and preserves unrelated hook files.

To preview the exact configuration without writing anything, use `--dry-run`
(setup only). It prints the same JSON, including the trailing newline, and
creates no directories, hook, or config files:

```bash
bun tools/setup_opencode.mjs --dry-run --context 32768
```

The default limits are 16,384 context tokens, 4,096 output tokens per turn,
and 24 agent steps. Match `--context` to the actual server context; it is not
a request to allocate more memory on the server. Use `--output` and `--steps`
to change the other limits. A coding session needs at least 8,192 tokens;
16,384 fits the fixture tasks; use 32,768 for substantial repository work when
the model and cache fit your GPU. At smaller
context limits, the default output budget is one quarter of the context.

Use **`--context auto`** during setup or preview to query the running server's
`/v1/models` once. An exact `--model` ID takes precedence; a single returned
model can supply capacity for a client alias. An unmatched multi-model list
is ambiguous and fails. The selected `context_length` must be a safe integer
of at least 8,192. Missing or malformed metadata, duplicate model IDs, HTTP
errors and a ten-second lookup deadline fail before project/config/hook writes.
The deadline covers the response body as well as the connection.

```bash
bun tools/setup_opencode.mjs --project ../my-project --context auto --dry-run
bun tools/setup_opencode.mjs --project ../my-project --context auto
```

Discovery uses the validated runtime endpoint and `ZINC_API_KEY`; their values
stay out of generated configuration and error messages. Numeric context remains
offline. `check` and `run` reject `--context auto` before project/task access:
rerun setup with `--force` when changing a generated config's capacity. Auto
requires discovery metadata from the server; it does not increase GPU capacity.

The configuration uses `@ai-sdk/openai-compatible`, keeps both the main model
and small-model requests on ZINC, and disables automatic sharing. The local
coding agent uses greedy decoding and excludes delegation and web tools.
It does not install a formatter or LSP server.

The local [OpenCode hook](https://opencode.ai/docs/plugins/) caps file reads at
80 lines and shortens tool output above 6,000 characters, retaining the start
and end with an explicit truncation notice. Reads can continue with `offset`
and `limit`. Compaction reserves space for incoming tool results as well as
generated output. Headless runs load this hook; OpenCode's `--pure` mode would
disable it. Large histories can still exceed the context, so API errors remain
failures rather than successful coding runs.

`check` verifies health, a JSON chat completion, a JSON tool call, and a
streaming tool call with a terminal finish reason. It never executes the
probe's requested tool. A successful health response alone does not establish
coding-agent compatibility.

## Run a headless task

```bash
bun tools/setup_opencode.mjs run \
  --project ../my-project \
  --auto -- "Fix the failing tests. Read the relevant files, edit the source, and run the tests."
```

The command requires a nonempty task after `--`, including when using
`--verify`. It streams OpenCode JSON events and propagates its exit status.
An OpenCode JSON error event stops the child and fails the run, even when the
CLI exits zero; independent verification cannot mask that error. The default
timeout is 15 minutes; `--timeout-ms` changes it, and a timeout exits with status 124.
Task and round timeouts must be integer milliseconds from 1 to 2,147,483,647.
To continue a previous session, pass `--session ID` using the `sessionID` from
its JSON events.
`--auto` lets OpenCode approve tool requests for this headless run; explicit
deny rules still apply. Use a disposable checkout for evaluation tasks.

To enforce a project check between editing rounds, pass `--verify`:

```bash
bun tools/setup_opencode.mjs run --project ../my-project --auto \
  --verify 'bun test' --attempts 3 --timeout-ms 1200000 -- \
  "Implement the agreed feature and pass the project tests."
```

The caller-supplied command runs in the selected project after each completed
or timed-out editing round. A failed check feeds real diagnostic excerpts back
into the same session. Feedback is bounded, retaining the start, selected error lines, and end
of a long log; the complete check output still streams to stderr.
Each editing round is capped at eight model steps; `--attempts` defaults to
three and allows at most ten. An editing round also has a five-minute deadline;
use `--round-timeout-ms` to change it. The wrapper reserves up to a minute
(one quarter of the remaining overall budget) for an independent check.
If editing times out, the wrapper stops that OpenCode process and checks the
partial patch. A passing check can finish the task; a failed check resumes the
same session when attempts and time remain. API errors and user cancellation
fail immediately. Round starts, deadlines, and check results appear on stderr,
while stdout remains the OpenCode JSON event stream.

The overall timeout covers editing and checks; a check finishing after the
deadline cannot turn the run into a success. Interrupted/timed-out checks
and exhausted attempts return a failure. Only a zero check status produces
success. On POSIX systems, the wrapper also stops remaining processes in each
owned child group, including background tools left after the main child exits.
Choose a check that actually covers the requested feature, and protect
acceptance tests when evaluating a model. The command is used at runtime and
is not written into the portable configuration.

For an interactive session, change into your project and select the generated
configuration:

```bash
OPENCODE_CONFIG=./opencode.zinc.json opencode
```

OpenCode merges configuration sources. The script's `run` command supplies
its provider/model configuration as a runtime override so an existing project
model selection does not switch the task to a hosted provider.

## Try a real task in a worktree

A Git worktree gives OpenCode a complete checkout and its own branch. It uses
the same coding loop as a regular project: read files, request edits, run
commands, inspect the results, and continue. The model proposes tool calls;
OpenCode executes them in the selected directory.

With a model server already running at the matching context limit:

```bash
git worktree add -b experiment/opencode ../zinc-opencode HEAD
bun tools/setup_opencode.mjs --project ../zinc-opencode \
  --context 32768 --output 3072 --steps 24
bun tools/setup_opencode.mjs check --project ../zinc-opencode
bun tools/setup_opencode.mjs run --project ../zinc-opencode \
  --auto --timeout-ms 1200000 --verify './scripts/zig.sh build test' -- \
  "Implement the agreed feature. Read source in ranges of at most 80 lines, edit the code, and run the CPU unit tests. Follow AGENTS.md. Do not start model servers or commit changes. Report actual test results."
git -C ../zinc-opencode diff --stat
git -C ../zinc-opencode diff
```

Keep JSON event logs outside the checkout. Monitor completed tool calls, the
diff, test output, and the model server log. Run acceptance checks independently
before accepting the patch. Short project instructions leave more context for
source code; ZINC's detailed code map and benchmark recipes live in linked
reference guides rather than the agent entrypoint.

The **2026-10-25 full-worktree trial failed** to implement its assigned
`GET /v1/models` feature. Across eight configurations, OpenCode made 57 tool
calls, produced no feature edits, and ran no project tests. The runs exposed
context overflow, repeated inspection after compaction, GPU allocation
failure, and attention-kernel failures with corrupted output. Two sessions
returned exit status zero without completing the task. The earlier fixture
results remain separate from this repository-level result. See the
[worktree report](OPENCODE_WORKTREE_REPORT.md) and
[ROCm context limits](ROCM.md#long-coding-sessions).

The [follow-up](ROCM_AGENT_CONTEXT.md#repository-coding-result) fixes the backend
memory/attention failures and records the later reviewed `--dry-run` feature.
GPU recovery and live acceptance completed. The
[repository end-to-end report](OPENCODE_E2E_REPORT.md) tests three further real
Zinc tasks with native-capacity Q8 and 32K FP32 serving. It distinguishes
functional checks from complete delivery: omitted regressions and review
findings still count against the assigned task. General repository coding
remains a supervised workflow.

## Use your own endpoint

The generated file always contains a loopback URL and an environment reference
for the key. For `run` and `check`, supply any endpoint override in your shell:

```bash
export ZINC_API_BASE_URL="http://127.0.0.1:9090/v1"
export ZINC_API_KEY="local"
```

A local ZINC server does not require an API key; `local` is a client placeholder.
Use your own key when connecting through an authenticated gateway. Neither
the endpoint override nor the key value is saved to the generated file.
`OPENCODE_BIN` can select an existing OpenCode executable at runtime.

If the GPU is remote, an SSH tunnel lets the configuration keep using
localhost. Fill these variables in your own shell:

```bash
ssh -N -L 9090:127.0.0.1:9090 \
  -p "${ZINC_PORT:?Set your SSH port}" \
  "${ZINC_USER:?Set your SSH user}@${ZINC_HOST:?Set your SSH host}"
```

## Check the harness without a model

In a ZINC checkout, run the actual OpenCode CLI against local scripted responses:

```bash
bun tools/validate_opencode_harness.mjs --output /tmp/zinc-harness-check
```

This checks timeout/repair in the same session, independent failing/passing
checks, fatal API errors that must skip verification, and the actual file-read
and command-output limits. The acceptance test
and portable configuration must remain unchanged. It needs Bun and an installed
OpenCode CLI; `--opencode FILE` or `OPENCODE_BIN` selects a different executable.
CI runs it with OpenCode 1.19.4.

Choose a fresh output directory. Its `result.json` contains portable outcomes;
raw event and check logs stay in that directory. The default fixture includes
an intentional 15-second stall. No model performs inference in these checks,
and the results do not establish coding quality or inference performance.
Use the live-server evaluation below for those questions.

## Evaluate coding tasks

Run the reproducible evaluation against an already running server:

```bash
bun tools/opencode_eval.mjs --list
bun tools/opencode_eval.mjs --manage-proxy --fixtures all
bun tools/validate_opencode_api.mjs
```

The suite creates disposable projects and drives actual `opencode run`
sessions. It covers a snapshot of Zinc's CLI validation code, a rate limiter,
multiple-file cart and formatter fixes, a duration parser, source-only edits,
a long policy specification, Unicode and spaces in file paths, CSV escaping,
and a Python NDJSON CLI. The CLI snapshot is frozen so fixing the real project
does not erase the reproducible test case.

Every task starts with failing tests. After OpenCode exits, the harness runs
the tests independently and checks that protected tests and specifications
were unchanged. A timeout, agent error, failed test, or synthetic proxy response
fails the task. Evaluation exits nonzero when any task fails.

`--context`, `--max-tokens-cap`, and `--timeout-ms` control the server-context
assumption, output budget, and per-task timeout. Use `--output DIR` to choose a
local artifact directory. These settings do not start or reconfigure ZINC.

The managed proxy is **transparent by default**: it forwards the model's
request/response bytes and records traces. Normal setup uses ZINC directly.
The old proxy's `--compat` mode can synthesize responses and source edits;
keep it out of model evaluations. Report results obtained with any repair or
synthetic behavior separately.

The API checks cover JSON/SSE parity, small token budgets, Unicode, tool
arguments, tool-result history, invalid requests, context overflow, queued
requests, streaming cancellation, and a prompt above the former 8K SSM limit.
Run them on a server with at least the default 16K context. Their report
contains check names, outcomes, and durations; it omits endpoints, local paths,
and response transcripts.

ZINC selects Qwen's JSON or XML function-call dialect from the loaded model's
template. Native XML calls retain multiline source strings, use the tool schema
for typed arguments, and round-trip through OpenCode's tool-result history.
Malformed or incomplete calls remain text; the server does not invent edits
or arguments to make a task pass.

## Troubleshooting

| Symptom                             | Check                                                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Health or chat check fails          | Confirm the server is running and `ZINC_API_BASE_URL` ends in `/v1`.                                                 |
| Tool check fails                    | Update ZINC, use a Qwen/ChatML model, and keep `ZINC_TOOL_CALLING` enabled.                                          |
| Context overflow                    | Match the configured context to the server and leave room for output and tool results.                               |
| Agent exits without fixing the task | Inspect JSON error/tool events and run the project's tests independently; exit status alone is not proof of success. |
| Task hits the step or output limit  | Increase `--steps` or `--output` when configuring the project, within the actual context budget.                     |
| OpenCode cannot be found            | Install the pinned version or set `OPENCODE_BIN` in your shell.                                                      |

Generated configs are portable. Runtime logs and traces can contain local
paths, source files, prompts, and tool output; keep them outside git and review
anything you intend to publish. No private endpoint, machine path, credential,
or raw session transcript belongs in a public setup example.

References: [OpenCode providers](https://opencode.ai/docs/providers/),
[configuration](https://opencode.ai/docs/config/),
[headless CLI](https://opencode.ai/docs/cli/), and
[permissions](https://opencode.ai/docs/permissions/).
