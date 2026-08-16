# Configure OpenCode with ZINC and Qwen

OpenCode can use ZINC as a local OpenAI-compatible coding backend for Qwen-family models. This is the practical setup for running an OpenCode local LLM workflow with ZINC instead of a hosted coding model:

1. run a ZINC server with a Qwen-family coding-capable model,
2. point OpenCode at `http://127.0.0.1:<port>/v1`,
3. keep tool calling enabled,
4. set realistic context and output limits for the model and machine,
5. optionally put the ZINC OpenCode trace proxy between OpenCode and ZINC while you are testing.

This page intentionally uses only localhost URLs and placeholder environment variables. Do not commit private hosts, SSH ports, API keys, model paths, or `.env` files.

OpenCode reference pages:

- OpenCode provider config: https://opencode.ai/docs/providers/
- OpenCode config files, `model`, `small_model`, and environment substitution: https://opencode.ai/docs/config/

## Start ZINC

Build ZINC and start the server on a local port:

```bash
zig build -Doptimize=ReleaseFast

# Linux AMD/RDNA shells should enable cooperative matrix support.
export RADV_PERFTEST=coop_matrix

# Use a managed model id when possible.
./zig-out/bin/zinc \
  --model-id qwen36-35b-a3b-q4k-xl \
  -p 9090
```

Check that the server is alive:

```bash
curl -fsS http://127.0.0.1:9090/health
curl -fsS http://127.0.0.1:9090/v1/models
```

ZINC does not require an API key for local use. Some clients still expect one to exist; use a harmless local placeholder:

```bash
export ZINC_API_KEY=local
```

## Pick A Qwen Coding Model

For OpenCode, prefer a Qwen/ChatML-family model because ZINC's thinking and tool-call path is built around that template family.

Good starting points:

- `qwen36-35b-a3b-q4k-xl`: best fit for a large local coding assistant on a 32 GB-class GPU or enough unified memory.
- `qwen35-9b-q4k-m`: smaller, easier to run, and useful for validating OpenCode tool calls before moving to a larger model.
- `qwen38-27b-q4k-m`: supported dense Qwen target on validated AMD RDNA4 32 GB and Apple Silicon 32+ GB systems.

The model you put in OpenCode's `model` field should match an installed ZINC managed model id. Check available and installed models with:

```bash
./zig-out/bin/zinc model list
./zig-out/bin/zinc model active
```

## If OpenCode Runs On A Different Machine

Keep ZINC bound to localhost on the GPU machine, then forward it to your laptop:

```bash
# Run this on the machine where OpenCode runs.
# Fill these from your own shell or password manager. Do not commit them.
export ZINC_REMOTE_USER="<user>"
export ZINC_REMOTE_HOST="<host>"
export ZINC_REMOTE_SSH_PORT="<ssh-port>"

ssh -N \
  -L 9090:127.0.0.1:9090 \
  -p "$ZINC_REMOTE_SSH_PORT" \
  "$ZINC_REMOTE_USER@$ZINC_REMOTE_HOST"
```

OpenCode should still use `http://127.0.0.1:9090/v1`. The tunnel hides the private host and keeps the OpenCode config portable.

## Direct OpenCode Config

Create or update `opencode.json` in the project where you run OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "zinc/qwen36-35b-a3b-q4k-xl",
  "small_model": "zinc/qwen36-35b-a3b-q4k-xl",
  "provider": {
    "zinc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ZINC",
      "options": {
        "baseURL": "http://127.0.0.1:9090/v1",
        "apiKey": "{env:ZINC_API_KEY}"
      },
      "models": {
        "qwen36-35b-a3b-q4k-xl": {
          "name": "Qwen3.6 35B A3B on ZINC",
          "limit": {
            "context": 4096,
            "output": 2048
          }
        }
      }
    }
  }
}
```

Tune the limits to the context length you start ZINC with and the amount of VRAM or unified memory available. If the model was launched with a smaller context, keep `limit.context` at or below that server-side limit. Use a lower `output` value such as `1024` if coding turns are slow or you are testing tool behavior.

## Recommended Trace Proxy Setup

For real coding work, use the trace proxy while ZINC's OpenCode compatibility is still evolving. It keeps the upstream URL local, records request/response traces, can force thinking mode, caps output length, and repairs common local-tool path issues.

Start ZINC on `9090`, then start the proxy on `9091`:

```bash
bun tools/opencode_trace_proxy.mjs \
  --upstream http://127.0.0.1:9090/v1 \
  --listen 9091 \
  --force-enable-thinking true \
  --force-tool-choice auto \
  --max-tokens-cap 2048 \
  --inject-path-guard \
  --repair-tool-paths
```

Then point OpenCode at the proxy instead of the server:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "zinc/qwen36-35b-a3b-q4k-xl",
  "small_model": "zinc/qwen36-35b-a3b-q4k-xl",
  "provider": {
    "zinc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ZINC via local trace proxy",
      "options": {
        "baseURL": "http://127.0.0.1:9091/v1",
        "apiKey": "{env:ZINC_API_KEY}"
      },
      "models": {
        "qwen36-35b-a3b-q4k-xl": {
          "name": "Qwen3.6 35B A3B on ZINC",
          "limit": {
            "context": 4096,
            "output": 2048
          }
        }
      }
    }
  }
}
```

Trace files are written under `/tmp/zinc-opencode-traces` by default. They may contain prompts, source snippets, tool arguments, and generated output, so treat them like local development artifacts and do not publish them blindly.

## Thinking And Tools

ZINC exposes thinking and OpenAI-style tool calling through `/v1/chat/completions`.

- Thinking: request body field `enable_thinking: true`. The trace proxy can force this with `--force-enable-thinking true` when the client does not expose a setting.
- Tool calling: enabled by default for ChatML/Qwen-family models. ZINC renders tool definitions, parses model-emitted `<tool_call>` blocks, and returns structured `tool_calls`.
- Tool execution: ZINC does not execute tools itself. OpenCode executes local tools and sends the results back as `role: "tool"` messages.
- Disabling tools: start ZINC with `ZINC_TOOL_CALLING=0` only when you are debugging plain chat behavior. Do not disable it for OpenCode coding sessions.

## Quick Smoke Test

Start in a disposable project:

```bash
mkdir -p /tmp/zinc-opencode-smoke/src /tmp/zinc-opencode-smoke/test
cd /tmp/zinc-opencode-smoke

cat > src/math.mjs <<'JS'
export function add(a, b) {
  return a - b;
}
JS

cat > test/math.test.mjs <<'JS'
import { strict as assert } from "node:assert";
import { add } from "../src/math.mjs";

assert.equal(add(2, 3), 5);
console.log("1 pass");
JS

node test/math.test.mjs
```

Then run OpenCode in that directory and ask:

```text
Fix the failing test. Read the source and test files first. Do not stop until `node test/math.test.mjs` passes.
```

A healthy run should read files, edit `src/math.mjs`, run the test, and finish only after the test passes.

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| OpenCode cannot connect | `curl http://127.0.0.1:9090/health` from the OpenCode machine, or `9091` if using the proxy. |
| It connects but never calls tools | Make sure you are using `/v1/chat/completions`, tool calling is not disabled, and the model is a Qwen/ChatML-family model with tool support. |
| It runs out of context | Lower OpenCode `limit.context`, lower `limit.output`, or start ZINC with a larger server context if the machine has memory headroom. |
| It writes strange paths | Use the trace proxy with `--inject-path-guard --repair-tool-paths`. |
| The answer is too short | Raise `limit.output` and the proxy `--max-tokens-cap`, then retry. |
| The answer is slow | Test the same request with `curl` against ZINC, then compare the proxy traces to see whether time is in model generation, repeated tool calls, or a failing loop. |

## Security Notes

- Bind ZINC and the proxy to localhost unless you are deliberately building a trusted internal service.
- Use SSH tunnels instead of public GPU-node URLs.
- Keep real hosts, SSH ports, usernames, tokens, model paths, and traces out of git.
- Do not paste private source or trace files into public issues unless you have reviewed them.
