# OpenCode with ZINC

Connect OpenCode to a local ZINC server, configure an existing project, and run
a bounded coding task with an independent test command. Use **Bun 1.4.4+** and
**OpenCode 2.0.21**, the pin used by the
[0.1.1 final-archive AMD cohort](releases/0.1.1-plan.md#final-archive-client-checks).
The [Intel development-package replay](INTEL_OPENCODE_VALIDATION.md#november-9-full-catalog-opencode-replay)
used 2.2.6; its client, source and task limits remain separate. CPU protocol
checks cover both versions, but do not establish model task quality.

## Start the model server

First [install ZINC and download Qwen3.5 9B](GETTING_STARTED.md). Stop an earlier
chat server and follow the shared [16K server recipe](RUNNING_ZINC.md#start-the-api).
Keep it running while the client uses `http://127.0.0.1:9090/v1`.

Use Qwen3.5 9B as a smaller starting point. Check
[model memory](HARDWARE_REQUIREMENTS.md#model-memory) and the
[recorded client results](#tested-scope) before choosing a larger model. The
0.6B installation-test model is unsuitable for these coding tasks.

## Configure OpenCode

On the client machine, install the pinned client and download the standalone helper:

```bash
bun add --global --trust @opencode/cli@2.0.21
curl -fL https://zolotukhin.ai/zinc/harnesses/setup-opencode.mjs -o setup-opencode.mjs
```

Replace `../my-project` with an **existing project directory**. From the directory
where you downloaded the helper:

```bash
bun setup-opencode.mjs --project ../my-project \
  --model qwen35-9b-q4k-m --context auto --output 2048 --steps 16
bun setup-opencode.mjs check --project ../my-project
```

`--context auto` reads the actual capacity from `/v1/models`; it does not allocate
more GPU memory. An exact model ID wins; a single returned model can supply
capacity for an alias on native backends. For a different model, pass its loaded
discovery ID. Missing or ambiguous metadata and context below 8,192 tokens fail
setup. Numeric `--context <tokens>` allows offline setup when you already know
the server's capacity.

Setup writes `opencode.zinc.json` and a versioned context hook in `.zinc/`.
It preserves your existing OpenCode configuration and authentication. Use
`--dry-run` to preview without writes; use `--force` to regenerate the dedicated
config and hook after changing model, context or client version.

**`check` sends live model requests** for JSON chat, JSON tools and streamed tools
with a terminal finish reason. It does not execute the probe's requested tool.
Resolve failures before a task; passing health alone is insufficient.

## Run a headless task

```bash
bun setup-opencode.mjs run --project ../my-project \
  --auto --verify 'bun test' --attempts 3 --timeout-ms 1200000 -- \
  "Fix the failing tests and explain the changes."
```

Replace `bun test` with your project's acceptance command. `--auto` approves
headless tool requests within the preset's explicit denies. `--verify` runs
your check independently between bounded editing rounds and stops with failure
if the checks do not pass. A task is required; the helper does not invent one.

The default per-task timeout is 15 minutes; the example allows 20. Timeout exits 124. API errors and failure events remain failures even if the client process
exits zero. Review the resulting diff as well as the test result: passing a
check does not establish that every requested change was delivered.

Without `--verify`, the helper runs one client task. `--session <id>` continues
a session from a previous JSON event. For interactive use, run inside the project:

```bash
OPENCODE_CONFIG=./opencode.zinc.json opencode
```

## Context and tools

The preset keeps main and small-model requests on ZINC, disables automatic
sharing, and uses greedy generation with local file/shell tools. OpenCode 2
Code Mode is disabled in this preset so direct tools are available. It does not
install an LSP server or formatter.

The local hook caps file reads at 80 lines and shortens tool output over 6,000
characters. Read larger files in ranges with `offset` and `limit`. Do not use
`--pure`, which disables this hook. Large instructions and histories can still
exceed context. Use short, testable tasks before attempting a whole repository.

Optional prompt reuse is gated by the server's advertised capability and the
loaded model. See [API compatibility limits](API.md#status-and-compatibility-limits)
for attention-only Vulkan reuse and the Gemma behavior released in 0.1.1. It does not
turn a full-prefill benchmark into a comparable cached run.

## Use your own endpoint

Set runtime overrides before setup with automatic discovery, `check` or `run`:

```bash
export ZINC_API_BASE_URL="http://127.0.0.1:9090/v1"
export ZINC_API_KEY="local"
```

The helper keeps endpoint overrides and key values out of generated files.
`local` is a client placeholder; the local ZINC server does not require a key.
Use the appropriate key for an authenticated gateway. `OPENCODE_BIN` selects
an existing client executable. Interactive clients need their own matching
provider configuration if you bypass the helper's runtime overrides.

For a remote GPU, keep a separate SSH tunnel running. Set these variables for
your own host first:

```bash
ssh -N -L 9090:127.0.0.1:9090 \
  -p "${ZINC_PORT:?Set your SSH port}" \
  "${ZINC_USER:?Set your SSH user}@${ZINC_HOST:?Set your SSH host}"
```

## Troubleshooting

| Symptom                                    | Action                                                                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Missing model / automatic context fails    | Check the loaded ID and context at `/v1/models`; pass the correct `--model` during setup.                         |
| Config already exists                      | Review it, then use setup `--force` to replace only the helper's files.                                           |
| Tool or streaming check fails              | Inspect the server error and consult the model/backend validation; do not assume all models support useful tools. |
| Context overflow                           | Restart with a supported larger context or shorten instructions and tool output; regenerate the config.           |
| Run reports success but work is incomplete | Review the diff and use an independent `--verify` command.                                                        |
| Files changed in the wrong directory       | Use the intended `--project`; the helper sets the child process directory and `PWD` together.                     |

## Tested scope

- [Intel model matrix](INTEL_OPENCODE_VALIDATION.md): the November 9 OpenCode
  2.2.6 replay passes three core fixtures for six larger models on Arc Pro B70;
  the starter model fails coding acceptance. Context, package and recovery
  conditions are recorded per model.
- [AMD release validation](OPENCODE_RELEASE_VALIDATION.md): model/backend
  protocol and coding outcomes, with failures retained.
- [Repository tasks](OPENCODE_E2E_REPORT.md) and the
  [earlier worktree investigation](OPENCODE_WORKTREE_REPORT.md): longer tasks
  expose limits beyond bounded fixtures. General repository coding remains supervised.
- [ROCm context](ROCM_AGENT_CONTEXT.md) and [Q8 KV](ROCM_NATIVE_CONTEXT.md):
  optional larger contexts need their own memory and quality checks.

For CPU protocol fixtures and live acceptance tooling, see
[client harness development](DEVELOPMENT.md#client-harness-checks).
The [standalone helper source](https://github.com/zolotukhin/zinc/blob/main/tools/setup_opencode.mjs)
is the reference for options and failure handling. Upstream references cover
[providers](https://opencode.ai/docs/providers/),
[configuration](https://opencode.ai/docs/config/) and
[permissions](https://opencode.ai/docs/permissions/).
