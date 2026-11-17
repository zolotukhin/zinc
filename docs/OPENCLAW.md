# OpenClaw with ZINC

Connect OpenClaw to a model served by ZINC. This recipe configures one existing
project, checks the connection, and runs a task. ZINC runs inference; OpenClaw's
file and shell tools run on the client machine.

Use **OpenClaw 2026.11.1** with **Node.js 24.22.2** for the recorded setup.
The client's supported Node ranges are 24.16+ within 24.x, or 26.1+.
See the [pinned upstream release](https://github.com/openclaw/openclaw/releases/tag/v2026.11.1).

## Start ZINC

First [install ZINC and download Qwen3.5 9B](GETTING_STARTED.md). Stop any
previous chat server, then use the common [16K server recipe](RUNNING_ZINC.md#start-the-api).
Keep it running while you use OpenClaw. The default API address is
`http://127.0.0.1:9090/v1`.

Qwen3.5 9B is the smaller starting model in the recorded AMD agent checks.
Check [model memory](HARDWARE_REQUIREMENTS.md#model-memory) and
[tested scope](#tested-scope) before choosing another model. OpenClaw setup
requires at least 4,000 context tokens; this recipe uses 16,384 for tools and history.

## Configure a project

On the client machine, install the pinned client and download the standalone helper:

```bash
npm install --global openclaw@2026.11.1
curl -fL https://zolotukhin.ai/zinc/harnesses/setup-openclaw.mjs -o setup-openclaw.mjs
```

Replace `../my-project` with an **existing project directory**. From the directory
where you downloaded the helper, configure and check it:

```bash
node setup-openclaw.mjs setup --project ../my-project
node setup-openclaw.mjs check --project ../my-project
```

Setup discovers the loaded model ID and live context. It writes only the
project's `openclaw.zinc.json`, leaving global OpenClaw settings alone. Review
that file. If it already differs, rerun setup with `--force` to replace it.
Use `--model <exact-discovery-id>` if discovery has more than one candidate.

`check` validates the real client's configuration and the server's health/model
metadata. **It does not generate tokens or prove tool quality.** A failed check
must be resolved before running a task. No ZINC source checkout is needed.

## Run and verify a task

```bash
node setup-openclaw.mjs run --project ../my-project --timeout 300 -- \
  "Fix the failing tests, run them, and explain the changes."
```

The helper invokes `openclaw agent exec` with this project, bounded output,
thinking off and a lean local-model tool preset. It does not start a gateway,
install a daemon or deliver messages. Each task uses temporary client state,
which is removed when the run ends.

Task failure, timeout, interruption or an unsuccessful JSON task report returns
failure. A successful report only means the client finished: inspect the diff
and **run the project's acceptance command independently**. A model can finish
without making the requested change.

## Custom endpoints and settings

For another server, pass its base URL during setup. Keep a protected endpoint's
key in an environment variable:

```bash
node setup-openclaw.mjs setup --project ../my-project \
  --base-url "$ZINC_API_BASE_URL" --api-key-env ZINC_API_KEY
```

Set those variables in your shell first. The generated config stores the endpoint
and an environment reference for the key, not the key value. Keep private endpoint
configuration out of version control. For remote GPUs, an
[SSH tunnel](OPENCODE.md#use-your-own-endpoint) lets the default loopback URL work.

| Option                                      | Purpose                                                               |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `--output <tokens>`                         | Bound output per turn; default 1,024.                                 |
| `--timeout <seconds>`                       | Bound the task; default 600.                                          |
| `--openclaw-bin <path>`                     | Select an existing client executable.                                 |
| `--dry-run --model <id> --context <tokens>` | Preview setup without writing files; explicit model/context required. |
| `--force`                                   | Replace this helper's dedicated config after a model/context change.  |

## Troubleshooting

| Symptom                                 | Action                                                                                                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Setup cannot discover a model           | Check `/health` and `/v1/models`; start a model before setup.                                                                                              |
| Context too small or stale              | Restart ZINC with sufficient memory/context, then rerun setup with `--force`.                                                                              |
| Client configuration rejected           | Confirm the pinned client and Node versions; rerun `check`.                                                                                                |
| Client finishes but files are unchanged | Treat it as a failed task; inspect output, reduce task size and verify independently.                                                                      |
| Preparation fails before a task starts  | Local HTTP preparation errors can be retried once by the helper only when no inference or file change occurred. Other failures retain their normal result. |

Optional prompt reuse is enabled only when the server advertises the appropriate
capability. Hybrid recurrent models, ROCm and older servers retain full prefill.
`ZINC_OPENCLAW_SESSION_REUSE=0` disables the client optimization. Reuse behavior
and Gemma controls are documented in the [API reference](API.md#status-and-compatibility-limits).

## Tested scope

The [0.1.1 release checks](releases/0.1.1-plan.md#final-archive-client-checks)
use the published archives on the 32 GB R9700 with this client. Qwen3.5 9B
passes at 16K; the complete AMD cohort passes 35/36 tasks. One Qwen3.6 35B-A3B
Vulkan task wrote an incorrect receipt and remains failed. Larger Qwen and
Gemma rows have their own contexts and limits. The starter 0.6B does not
qualify for agent tasks, and this cohort does not qualify Intel or Metal.
The earlier [0.1.0 AMD report](OPENCLAW_AMD_VALIDATION.md) retains its own outcomes.
The later [Intel client matrix](INTEL_OPENCODE_VALIDATION.md#november-9-gemma-cache-and-client-qualification)
records separate development-package evidence.

For contributor checks using the actual client against scripted responses, see
[client harness development](DEVELOPMENT.md#client-harness-checks).
The [helper source](https://github.com/zolotukhin/zinc/blob/main/tools/setup_openclaw.mjs)
and [evaluator](https://github.com/zolotukhin/zinc/blob/main/tools/openclaw_eval.mjs)
contain retry, cleanup and acceptance contracts.
