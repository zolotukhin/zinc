# OpenClaw with ZINC

Use OpenClaw's one-shot coding agent with a model served by ZINC. The
[setup helper](https://github.com/zolotukhin/zinc/blob/main/tools/setup_openclaw.mjs)
discovers the loaded model and its actual context capacity, creates a dedicated
project config, validates it with the client, and runs a task.

The integration is tested with **OpenClaw 2026.11.1** and **Node.js 24.22.2**.
That OpenClaw version requires Node **24.16 or newer in the 24.x line**, or
**26.1 or newer**. See the [upstream release](https://github.com/openclaw/openclaw/releases/tag/v2026.11.1)
and [one-shot command reference](https://docs.openclaw.ai/cli/agent).
The helper itself uses only Node's standard library.

The [AMD model acceptance report](OPENCLAW_AMD_VALIDATION.md) records the
published binaries, actual file/tool outcomes, contexts and model-specific limits.
The [Intel client matrix](INTEL_OPENCODE_VALIDATION.md#november-7-candidate-opencode-and-openclaw)
records the separate Arc Pro B70 results, including Gemma workspace timeouts
and the starter model's failed coding acceptance.

For a complete Linux walkthrough, see
[OpenCode and OpenClaw on AMD: R9700 and RX 9070 XT](https://zolotukhin.ai/blog/2026-11-07-opencode-openclaw-amd-r9700-rx-9070-xt/).

## Start ZINC

Choose the [Linux release](BINARY_RELEASES.md) for Vulkan or ROCm. Start one
model with enough context for the agent's instructions, tool schemas, source
files and tool results. On a 32 GB GPU, Qwen3.5 9B is a practical starting point.
With the installed Vulkan binary:

```bash
zinc --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090
```

For a checkout build, use `./zig-out/bin/zinc`. For an existing GGUF, use
`-m /path/to/model.gguf`. Follow the
[ROCm guide](ROCM.md) for that backend's model paths and device selection.
Select your discrete GPU explicitly on systems with an integrated GPU too;
check the server's device log before accepting a result.

OpenClaw needs at least a 4,000-token context. Its coding prompt consumes
several thousand tokens before your task begins. A model's advertised native
window does not allocate GPU memory: configure the server first, then let setup
discover its capacity. A 16K context is the starting point for this Qwen example;
larger models need their own memory checks. Keep the backend's default KV
precision for a comparable validation.

## Configure a project

Install the pinned client with a compatible Node installation:

```bash
npm install --global openclaw@2026.11.1
node tools/setup_openclaw.mjs setup --project ../my-project
node tools/setup_openclaw.mjs check --project ../my-project
```

Or [download the standalone helper](https://zolotukhin.ai/zinc/harnesses/setup-openclaw.mjs) and use
`node setup-openclaw.mjs` with the same arguments. The project directory must
already exist. `check` runs the real OpenClaw config validator and verifies ZINC
health, model identity and available context; it does not generate tokens or
prove coding ability.

Setup writes `openclaw.zinc.json`. It preserves the user's global OpenClaw
settings and refuses a differing project config unless passed `--force`.
Symlink configs are rejected. The dedicated preset uses a single ZINC provider,
chat completions, streaming tool usage, thinking off, and a bounded output
budget. Filesystem tools stay within the project; shell tools run commands on
the machine where OpenClaw runs. Use it in a project you intend the agent to edit.
No gateway, messaging channel, background daemon, or delivery action is started.
The preset omits bundled skills and the tool-search catalog to leave more of
the context for code and tool results.

For a checkout with the exact dependency tree used in AMD qualification, use
`npm ci --prefix tools/openclaw-client` instead of a global install, then pass
`--openclaw-bin tools/openclaw-client/node_modules/.bin/openclaw` to `check` and
`run`. CI uses this locked client, including npm's integrity checks, in an
isolated prefix. The ZINC runtime does not depend on these client packages.

To connect to a different endpoint, supply its API base at setup time:

```bash
node tools/setup_openclaw.mjs setup --project ../my-project \
  --base-url "$ZINC_API_BASE_URL" --api-key-env ZINC_API_KEY
```

Export `ZINC_API_KEY` separately for a protected endpoint. The saved config
contains an environment reference, never the key's value. Local unauthenticated
servers use a dummy key. Keep configs containing private endpoint addresses out
of version control. `--model` selects an exact **loaded** discovery ID; setup
does not silently load or switch models. ROCm may report a GGUF filename as its
ID. Rediscover after changing the server's model or context.

## Run and verify a task

```bash
node tools/setup_openclaw.mjs run --project ../my-project --timeout 300 -- \
  "Fix the failing tests, run them, and explain the changes."
```

The helper invokes `openclaw agent exec` with the project directory, explicit
config and model, direct tools, the lean local-model prompt, and thinking off.
Each task gets temporary client state. Task failure, timeout, interruption,
oversized diagnostics or an unsuccessful JSON task report makes the helper
fail. It stops its process group and removes that state on completion.
An interrupted evaluation also writes a partial summary and stops before
starting another task.

OpenClaw 2026.11.1 can reject its prepared model runtime before admitting the
agent run. For a local HTTP ZINC connection, the helper retries that exact
preparation error once, within the original deadline. An empty report with no
selected model/provider, payloads, usage or tool activity is required, together
with an independent local forwarding check that no inference request was sent.
The forwarder preserves response bytes and streaming; any non-metadata request
blocks a retry. Remote or HTTPS connections retain their normal failure result. Each attempt owns and removes its client
state. The evaluator retains the first failure in a private diagnostics file
and records `preparationRetries` in the task summary. Ordinary task failures
keep their existing failure outcome.

The helper also queries ZINC's optional `chat_session_reuse` health capability
before a run. On qualified attention-only Vulkan models, successive tool turns
share a fresh per-task `session_id` so the server can reuse an exact prompt
prefix. The key is supplied through OpenClaw's native extra-body parameter in a
private temporary config beside the saved config. The original file and its
environment-based key references are preserved; the temporary copy is removed
when the client exits, including after a timeout or interruption. Separate tasks
get separate keys. Hybrid recurrent models, ROCm, older servers and failed
capability discovery retain full prefill.

Set `ZINC_OPENCLAW_SESSION_REUSE=0` for an uncached client control. An explicit
`session_id`, including an empty or null value, in the model's `extraBody` or
`extra_body` parameters takes precedence. A cached task result is a separate
measurement from a full-prefill benchmark; do not combine their timings into
an engine speed claim.

OpenClaw's successful task report means the client completed its run. Review
the actual diff and run the project's acceptance command independently before
accepting the change. In particular, a model can stop successfully without
making the requested edits.

## Repeatable validation

Run these commands on your authorized inference worker, with ZINC already
running. They do not start or stop the model server:

```bash
node tools/openclaw_eval.mjs --base-url http://127.0.0.1:9090/v1 \
  --openclaw-bin openclaw --output /tmp/zinc-openclaw-trial
```

Use a fresh output directory. The evaluator checks an exact text response,
then asks the real client to read a random receipt, fix a small JavaScript
module, write that receipt and run tests. It independently reruns the tests and
checks the protected input/test hashes and receipt bytes. A final text request
and health check test recovery. Raw task logs stay in files marked private;
`summary.json` contains the outcomes. These bounded tasks are acceptance checks,
not a broad coding benchmark or a throughput comparison.

The separate CPU protocol test uses the actual client with scripted streamed
HTTP responses:

```bash
node tools/validate_openclaw_harness.mjs --openclaw-bin openclaw \
  --output /tmp/zinc-openclaw-protocol
```

It exercises fragmented tool arguments, correlated tool-result history, real
file writes, command execution, API errors, cancellation and recovery without loading a model. CI runs this check
with the pinned client; it is deliberately separate from GPU/model evidence.
