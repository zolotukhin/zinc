#!/usr/bin/env bun
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { lstat, mkdir, open, readFile } from 'node:fs/promises';
import path from 'node:path';

export const DEFAULT_BASE_URL = 'http://127.0.0.1:9090/v1';
export const CONFIG_NAME = 'opencode.zinc.json';

// Self-contained so the standalone helper can install this local OpenCode hook
// without fetching a package or embedding any machine-specific configuration.
export async function contextPlugin() {
  return {
    'tool.execute.before': async (input, output) => {
      if (input.tool === 'read') {
        const requested = output.args.limit;
        output.args.limit = Number.isSafeInteger(requested)
          ? Math.min(80, Math.max(0, requested))
          : 80;
      }
    },
    'tool.execute.after': async (_input, output) => {
      if (typeof output.output !== 'string' || output.output.length <= 6000) return;
      const text = output.output;
      output.output =
        text.slice(0, 4000).replace(/[\uD800-\uDBFF]$/, '') +
        '\n[ZINC context limit: output shortened. Read smaller ranges with offset/limit, or narrow the command.]\n' +
        text.slice(-1500).replace(/^[\uDC00-\uDFFF]/, '');
      output.metadata = { ...output.metadata, truncated: true, zinc_context_truncated: true };
    },
  };
}

const CONTEXT_PLUGIN_SOURCE = `export default ${contextPlugin.toString()};\n`;
const CONTEXT_PLUGIN_HASH = createHash('sha256')
  .update(CONTEXT_PLUGIN_SOURCE)
  .digest('hex')
  .slice(0, 16);
export const CONTEXT_PLUGIN_PATH = `.zinc/opencode-context-${CONTEXT_PLUGIN_HASH}.js`;

async function ensureContextPlugin(dir) {
  const pluginDir = path.join(dir, '.zinc');
  await mkdir(pluginDir, { mode: 0o700 }).catch((error) => {
    if (error.code !== 'EEXIST') throw error;
  });
  const stat = await lstat(pluginDir);
  if (!stat.isDirectory() || stat.isSymbolicLink())
    throw new Error('ZINC context hook directory must be a directory, not a symlink');
  const file = path.join(dir, CONTEXT_PLUGIN_PATH);
  let handle;
  try {
    handle = await open(
      file,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o600,
    );
    await handle.writeFile(CONTEXT_PLUGIN_SOURCE);
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const existing = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      if ((await existing.readFile('utf8')) !== CONTEXT_PLUGIN_SOURCE)
        throw new Error('ZINC context hook conflicts with an existing file; no file was replaced');
    } finally {
      await existing.close();
    }
  } finally {
    await handle?.close();
  }
}

function requireTask(opts) {
  if (!Array.isArray(opts.prompt) || !opts.prompt.join(' ').trim())
    throw new Error('Pass a task after --');
}

function timeoutValue(value, label = 'Timeout') {
  // Node-compatible timers overflow beyond this limit and can fire immediately.
  if (!Number.isSafeInteger(value) || value <= 0 || value > 2147483647)
    throw new Error(`${label} must be a positive integer (maximum: 2147483647 ms)`);
  return value;
}

export function parseArgs(argv = process.argv.slice(2)) {
  const opts = {
    command: 'setup',
    project: '.',
    model: 'qwen38-27b-q4k-m',
    context: 16384,
    output: undefined,
    steps: 24,
    force: false,
    dryRun: false,
    prompt: [],
    auto: false,
    timeoutMs: 900000,
  };
  const args = [...argv];
  if (['setup', 'check', 'run'].includes(args[0])) opts.command = args.shift();
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--') {
      opts.prompt = args.slice(i + 1);
      break;
    }
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      continue;
    }
    if (arg === '--force') {
      opts.force = true;
      continue;
    }
    if (arg === '--dry-run') {
      opts.dryRun = true;
      continue;
    }
    if (arg === '--auto') {
      opts.auto = true;
      continue;
    }
    const key = {
      '--project': 'project',
      '--model': 'model',
      '--context': 'context',
      '--output': 'output',
      '--steps': 'steps',
      '--timeout-ms': 'timeoutMs',
      '--round-timeout-ms': 'roundTimeoutMs',
      '--session': 'session',
      '--verify': 'verify',
      '--attempts': 'attempts',
    }[arg];
    if (!key) throw new Error(`Unknown option: ${arg}`);
    const value = args[++i];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${arg}`);
    if (key === 'context' && value === 'auto') {
      opts.context = 'auto';
      continue;
    }
    opts[key] = ['context', 'output', 'steps', 'timeoutMs', 'roundTimeoutMs', 'attempts'].includes(
      key,
    )
      ? Number(value)
      : value;
  }
  return opts;
}

export function buildConfig(opts = {}) {
  const { model = 'qwen38-27b-q4k-m', context = 16384, steps = 24 } = opts;
  const output = opts.output ?? Math.min(4096, Math.floor(context / 4));
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$/.test(model))
    throw new Error('Use a model ID, not a file path or URL');
  if (!Number.isSafeInteger(context) || context < 8192)
    throw new Error('Context must be an integer of at least 8192 tokens');
  if (!Number.isSafeInteger(output) || output < 1 || output >= context / 2)
    throw new Error('Output must be positive and less than half the context');
  if (!Number.isSafeInteger(steps) || steps < 1 || steps > 100)
    throw new Error('Steps must be between 1 and 100');
  return {
    $schema: 'https://opencode.ai/config.json',
    model: `zinc/${model}`,
    small_model: `zinc/${model}`,
    enabled_providers: ['zinc'],
    default_agent: 'zinc',
    autoupdate: false,
    share: 'disabled',
    formatter: false,
    lsp: false,
    plugin: [`./${CONTEXT_PLUGIN_PATH}`],
    // Leave room for new tool results as well as the next generated response.
    compaction: {
      auto: true,
      prune: true,
      reserved: Math.min(Math.floor(context / 2), output + Math.floor(context / 4)),
    },
    provider: {
      zinc: {
        npm: '@ai-sdk/openai-compatible',
        name: 'ZINC',
        options: { baseURL: DEFAULT_BASE_URL, apiKey: '{env:ZINC_API_KEY}' },
        models: { [model]: { name: model, tool_call: true, limit: { context, output } } },
      },
    },
    agent: {
      zinc: {
        description: 'Local coding with ZINC',
        mode: 'primary',
        temperature: 0,
        top_p: 1,
        steps,
        prompt:
          'Complete the coding task using the available tools. File reads return at most 80 lines: use offset and limit to read the relevant ranges, then make a focused edit and run the requested tests before exploring further. Use compiler and test output to correct failures. Tool output is bounded: save verbose build/test logs to a temporary file and show only failure diagnostics and a short summary. Never change tests just to make them pass. Use portable paths in source, tests, and examples; do not copy this workspace path or private endpoints into them. Finish with a short report of the changes and actual test results.',
        permission: { task: 'deny', question: 'deny', webfetch: 'deny', websearch: 'deny' },
      },
    },
  };
}

export async function writeConfig(project, config, { force = false } = {}) {
  const dir = path.resolve(project);
  await mkdir(dir, { recursive: true });
  const file = path.join(dir, CONFIG_NAME);
  // Check before installing the hook so a refused setup leaves existing
  // projects alone. The final open still enforces no-follow/exclusive writes.
  const existing = await lstat(file).catch((error) => {
    if (error.code === 'ENOENT') return undefined;
    throw error;
  });
  if (existing && !force) throw Object.assign(new Error('Config exists'), { code: 'EEXIST' });
  if (existing?.isSymbolicLink())
    throw new Error('ZINC context hook setup refuses config symlinks');
  await ensureContextPlugin(dir);
  // O_NOFOLLOW also protects an existing symlink when explicitly replacing a
  // generated config. Other OpenCode config/auth files are never written.
  const flags =
    constants.O_WRONLY |
    constants.O_CREAT |
    constants.O_NOFOLLOW |
    (force ? constants.O_TRUNC : constants.O_EXCL);
  const handle = await open(file, flags, 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(config, null, 2)}\n`);
  } finally {
    await handle.close();
  }
  return file;
}

export function runtimeConfig(config, env = process.env) {
  const copy = structuredClone(config);
  const baseURL = env.ZINC_API_BASE_URL || DEFAULT_BASE_URL;
  let url;
  try {
    url = new URL(baseURL);
  } catch {
    throw new Error('ZINC_API_BASE_URL must be an HTTP(S) URL ending in /v1');
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !/\/v1\/?$/.test(url.pathname)
  ) {
    throw new Error(
      'ZINC_API_BASE_URL must be an HTTP(S) URL ending in /v1 without credentials, query or fragment',
    );
  }
  copy.provider.zinc.options.baseURL = baseURL.replace(/\/$/, '');
  // Keep the actual key in the environment, including in inline config. No
  // runtime URL or secret is written to the portable configuration on disk.
  copy.provider.zinc.options.apiKey = '{env:ZINC_API_KEY}';
  return copy;
}

// Resolve the actual server context for `--context auto` with a single GET to
// the runtime /v1/models endpoint. The endpoint and key come from the same
// validated runtime configuration as check/run, and neither value is ever
// included in an error message.
export async function resolveAutoContext(
  { model = 'qwen38-27b-q4k-m' } = {},
  { env = process.env, fetchImpl = fetch } = {},
) {
  const live = runtimeConfig({ provider: { zinc: { options: {} } } }, env);
  const base = live.provider.zinc.options.baseURL;
  const signal = AbortSignal.timeout(10000);
  let response;
  try {
    response = await fetchImpl(`${base}/models`, {
      headers: { authorization: `Bearer ${env.ZINC_API_KEY || 'local'}` },
      signal,
    });
  } catch {
    throw new Error('ZINC model discovery request failed or timed out');
  }
  if (!response.ok) throw new Error(`ZINC model discovery returned HTTP ${response.status}`);
  let data;
  try {
    data = await response.json();
  } catch {
    if (signal.aborted) throw new Error('ZINC model discovery lookup timed out');
    throw new Error('ZINC model discovery returned malformed JSON');
  }
  const models = data?.data;
  if (!Array.isArray(models) || models.length === 0)
    throw new Error('ZINC model discovery returned no loaded models');
  const ids = new Set();
  for (const item of models) {
    if (
      !item ||
      typeof item !== 'object' ||
      Array.isArray(item) ||
      typeof item.id !== 'string' ||
      !item.id.length ||
      ids.has(item.id)
    )
      throw new Error('ZINC model discovery returned invalid or duplicate model IDs');
    ids.add(item.id);
  }
  const entry =
    models.find((item) => item.id === model) ?? (models.length === 1 ? models[0] : undefined);
  if (!entry)
    throw new Error(
      'ZINC model discovery is ambiguous: the requested model is not loaded and multiple models are available',
    );
  const context = entry?.context_length;
  if (!Number.isSafeInteger(context) || context < 8192)
    throw new Error('ZINC model discovery returned a missing or invalid context length');
  return context;
}

export async function checkServer(config, { env = process.env, fetchImpl = fetch } = {}) {
  const live = runtimeConfig(config, env);
  const base = live.provider.zinc.options.baseURL;
  const headers = { authorization: `Bearer ${env.ZINC_API_KEY || 'local'}` };
  const health = await fetchImpl(`${base.slice(0, -3)}/health`, {
    headers,
    signal: AbortSignal.timeout(10000),
  });
  if (!health.ok) throw new Error(`ZINC health check returned HTTP ${health.status}`);
  const healthData = await health.json();
  if (healthData.status !== 'ok') throw new Error('ZINC returned an unhealthy status');
  const limit = Object.values(live.provider.zinc.models)[0].limit.context;
  if (Number.isFinite(healthData.context_length) && limit > healthData.context_length) {
    throw new Error(
      'ZINC returned a smaller context than configured; rerun setup with the actual server context',
    );
  }
  const request = {
    model: live.model.slice('zinc/'.length),
    messages: [{ role: 'user', content: 'Reply with OK.' }],
    max_tokens: 8,
    temperature: 0,
    stream: false,
    enable_thinking: false,
  };
  const response = await fetchImpl(`${base}/chat/completions`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(120000),
  });
  if (!response.ok) throw new Error(`ZINC chat check returned HTTP ${response.status}`);
  const result = await response.json();
  if (!result.choices?.[0]?.message || !result.choices?.[0]?.finish_reason)
    throw new Error('ZINC returned an invalid chat completion');
  if (!String(result.choices[0].message.content || '').includes('OK'))
    throw new Error('ZINC returned an unexpected probe answer');
  const toolRequest = {
    ...request,
    max_tokens: 64,
    messages: [{ role: 'user', content: 'Call zinc_probe with message OK.' }],
    tools: [
      {
        type: 'function',
        function: {
          name: 'zinc_probe',
          description: 'Return a diagnostic message; this probe does not execute a tool.',
          parameters: {
            type: 'object',
            properties: { message: { type: 'string' } },
            required: ['message'],
          },
        },
      },
    ],
    tool_choice: 'required',
  };
  const toolResponse = await fetchImpl(`${base}/chat/completions`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify(toolRequest),
    signal: AbortSignal.timeout(120000),
  });
  if (!toolResponse.ok) throw new Error(`ZINC tool check returned HTTP ${toolResponse.status}`);
  const toolResult = await toolResponse.json();
  const call = toolResult.choices?.[0]?.message?.tool_calls?.[0];
  if (
    toolResult.choices?.[0]?.finish_reason !== 'tool_calls' ||
    call?.function?.name !== 'zinc_probe' ||
    JSON.parse(call.function.arguments).message !== 'OK'
  ) {
    throw new Error(
      'ZINC returned an invalid tool completion; use a Qwen model and a server with tool support',
    );
  }
  const streamResponse = await fetchImpl(`${base}/chat/completions`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify({ ...toolRequest, stream: true }),
    signal: AbortSignal.timeout(120000),
  });
  if (!streamResponse.ok)
    throw new Error(`ZINC streaming check returned HTTP ${streamResponse.status}`);
  const text = await streamResponse.text();
  const events = text
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data: ') && line !== 'data: [DONE]')
    .map((line) => JSON.parse(line.slice(6)));
  const toolEvents = events.flatMap((event) => event.choices?.[0]?.delta?.tool_calls || []);
  const args = toolEvents.map((event) => event.function?.arguments || '').join('');
  if (
    !text.includes('data: [DONE]') ||
    !events.some((event) => event.choices?.[0]?.finish_reason === 'tool_calls') ||
    !toolEvents.some((event) => event.function?.name === 'zinc_probe') ||
    JSON.parse(args).message !== 'OK'
  ) {
    throw new Error('ZINC returned an invalid streaming tool completion');
  }
  return {
    model: request.model,
    health: 'pass',
    chat: 'pass',
    tools: 'pass',
    streaming_tools: 'pass',
  };
}

export async function runOpenCode(
  config,
  opts,
  {
    env = process.env,
    spawnImpl = spawn,
    stdout = process.stdout,
    stderr = process.stderr,
    onEvent,
  } = {},
) {
  requireTask(opts);
  const timeoutMs = timeoutValue(opts.timeoutMs ?? 900000);
  const live = runtimeConfig(config, env);
  const args = [
    'run',
    '--format',
    'json',
    '--model',
    live.model,
    '--agent',
    'zinc',
    '--dir',
    path.resolve(opts.project),
  ];
  if (opts.auto) args.push('--auto');
  if (opts.session) args.push('--session', opts.session);
  args.push('--', opts.prompt.join(' '));
  const child = spawnImpl(env.OPENCODE_BIN || 'opencode', args, {
    cwd: path.resolve(opts.project),
    env: {
      ...env,
      ZINC_API_KEY: env.ZINC_API_KEY || 'local',
      OPENCODE_CONFIG_CONTENT: JSON.stringify(live),
      OPENCODE_DISABLE_AUTOUPDATE: 'true',
      // --pure disables the local context hook, including explicit plugins.
      OPENCODE_PURE: 'false',
    },
    stdio: ['inherit', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });
  let agentError = false;
  let stopping = false;
  let closed = false;
  let timedOut = false;
  let interruptedCode;
  let timer;
  let killTimer;
  const stop = (signal) => {
    if (child.pid && process.platform !== 'win32') {
      try {
        process.kill(-child.pid, signal);
      } catch (error) {
        if (error.code !== 'ESRCH') throw error;
      }
    } else child.kill(signal);
  };
  const terminate = (signal) => {
    stopping = true;
    clearTimeout(timer);
    clearTimeout(killTimer);
    killTimer = setTimeout(() => stop('SIGKILL'), 5000);
    stop(signal);
  };
  let pending = '';
  const decoder = new TextDecoder();
  child.stderr?.on('data', (chunk) => stderr.write(chunk));
  const inspectLine = (line) => {
    try {
      const event = JSON.parse(line);
      if (event.type === 'error') {
        agentError = true;
        if (!stopping && !closed) terminate('SIGTERM');
      }
      onEvent?.(event);
    } catch {
      /* Other CLI output is forwarded verbatim. */
    }
  };
  child.stdout?.on('data', (chunk) => {
    stdout.write(chunk);
    pending += decoder.decode(chunk, { stream: true });
    let newline;
    while ((newline = pending.indexOf('\n')) !== -1) {
      inspectLine(pending.slice(0, newline));
      pending = pending.slice(newline + 1);
    }
  });
  timer = setTimeout(() => {
    timedOut = true;
    terminate('SIGTERM');
  }, timeoutMs);
  const interrupt = (signal, code) => {
    interruptedCode = code;
    terminate(signal);
  };
  const onInt = () => interrupt('SIGINT', 130);
  const onTerm = () => interrupt('SIGTERM', 143);
  process.on('SIGINT', onInt);
  process.on('SIGTERM', onTerm);
  try {
    return await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) => {
        closed = true;
        inspectLine(pending + decoder.decode());
        resolve(
          interruptedCode ??
            (agentError
              ? 1
              : timedOut
                ? 124
                : (code ?? (signal === 'SIGINT' ? 130 : signal === 'SIGTERM' ? 143 : 1))),
        );
      });
    });
  } finally {
    closed = true;
    process.off('SIGINT', onInt);
    process.off('SIGTERM', onTerm);
    clearTimeout(timer);
    clearTimeout(killTimer);
    // A CLI can exit while an owned background tool ignores SIGTERM and has
    // already closed its pipes. Do not leave that detached group alive.
    if (child.pid && process.platform !== 'win32') stop('SIGKILL');
  }
}

export async function runVerification(
  command,
  opts,
  { env = process.env, spawnImpl = spawn, stderr = process.stderr } = {},
) {
  const timeoutMs = timeoutValue(opts.timeoutMs ?? 900000);
  const shell = process.platform === 'win32' ? env.ComSpec || 'cmd.exe' : '/bin/sh';
  const args = process.platform === 'win32' ? ['/d', '/s', '/c', command] : ['-c', command];
  const child = spawnImpl(shell, args, {
    cwd: path.resolve(opts.project),
    env,
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let head = '',
    tail = '',
    truncated = false,
    diagnostics = '';
  const decoders = [new TextDecoder(), new TextDecoder()];
  const pendingLines = ['', ''];
  const contextLines = [0, 0];
  const inspect = (line, stream) => {
    const plain = line.replace(/\x1b\[[0-9;]*m/g, '');
    if (/^\s*\((pass|skip)\)/.test(plain)) {
      contextLines[stream] = 0;
      return;
    }
    if (
      /\berror:|\b(AssertionError|SyntaxError|TypeError|ReferenceError|panic)\b|^\s*(\(fail\)|FAIL(ED)?\b)|expected:|received:/i.test(
        plain,
      )
    )
      contextLines[stream] = 5;
    if (contextLines[stream] > 0) {
      diagnostics = (diagnostics + line.slice(0, 500) + '\n').slice(-3000);
      contextLines[stream]--;
    }
  };
  const append = (text, stream) => {
    pendingLines[stream] += text;
    let newline;
    while ((newline = pendingLines[stream].indexOf('\n')) !== -1) {
      inspect(pendingLines[stream].slice(0, newline), stream);
      pendingLines[stream] = pendingLines[stream].slice(newline + 1);
    }
    pendingLines[stream] = pendingLines[stream].slice(-8000);
    if (!truncated) {
      head += text;
      if (head.length > 8000) {
        tail = head.slice(-2500);
        head = head.slice(0, 2000);
        truncated = true;
      }
    } else tail = (tail + text).slice(-2500);
  };
  for (const [i, stream] of [child.stdout, child.stderr].entries())
    stream?.on('data', (chunk) => {
      stderr.write(chunk);
      append(decoders[i].decode(chunk, { stream: true }), i);
    });
  const stop = (signal) => {
    if (child.pid && process.platform !== 'win32') {
      try {
        process.kill(-child.pid, signal);
      } catch (error) {
        if (error.code !== 'ESRCH') throw error;
      }
    } else child.kill(signal);
  };
  let timedOut = false,
    killTimer,
    interruptedCode;
  const timer = setTimeout(() => {
    timedOut = true;
    stop('SIGTERM');
    killTimer = setTimeout(() => stop('SIGKILL'), 5000);
  }, timeoutMs);
  const interrupt = (signal, code) => {
    interruptedCode = code;
    clearTimeout(timer);
    clearTimeout(killTimer);
    stop(signal);
    killTimer = setTimeout(() => stop('SIGKILL'), 5000);
  };
  const onInt = () => interrupt('SIGINT', 130);
  const onTerm = () => interrupt('SIGTERM', 143);
  process.on('SIGINT', onInt);
  process.on('SIGTERM', onTerm);
  try {
    const code = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) =>
        resolve(
          interruptedCode ??
            (timedOut
              ? 124
              : (code ?? (signal === 'SIGINT' ? 130 : signal === 'SIGTERM' ? 143 : 1))),
        ),
      );
    });
    for (const [i, decoder] of decoders.entries()) {
      append(decoder.decode(), i);
      inspect(pendingLines[i], i);
    }
    return {
      code,
      output: truncated
        ? `${head}\n[verification output truncated; selected diagnostics]\n${diagnostics}\n[final output]\n${tail}`
        : head,
    };
  } finally {
    process.off('SIGINT', onInt);
    process.off('SIGTERM', onTerm);
    clearTimeout(timer);
    clearTimeout(killTimer);
    if (child.pid && process.platform !== 'win32') stop('SIGKILL');
  }
}

export async function runVerifiedOpenCode(config, opts, deps = {}) {
  requireTask(opts);
  if (!opts.verify?.trim()) throw new Error('Verification requires a nonempty command');
  const attempts = opts.attempts ?? 3;
  if (!Number.isSafeInteger(attempts) || attempts < 1 || attempts > 10)
    throw new Error('Attempts must be between 1 and 10');
  const timeout = timeoutValue(opts.timeoutMs ?? 900000);
  const roundTimeout = timeoutValue(opts.roundTimeoutMs ?? 300000, 'Round timeout');
  const now = deps.now ?? (() => performance.now());
  const deadline = now() + timeout;
  const roundConfig = structuredClone(config);
  roundConfig.agent.zinc.steps = Math.min(roundConfig.agent.zinc.steps, 8);
  let session = opts.session;
  const roundInstructions =
    'The wrapper runs the full verification command outside the agent after this editing round. Make focused edits, then end the round promptly for that independent check. Do not run the full verification command yourself or paste its log into the conversation. Short targeted diagnostics are fine when needed. Report changes without claiming checks the wrapper has not run.';
  let prompt = [
    `${opts.prompt.join(' ')}\n\n${roundInstructions}\nVerification command:\n${opts.verify}`,
  ];
  let last = 1;
  const executeModel = deps.runModel ?? runOpenCode;
  const executeVerification = deps.runCheck ?? runVerification;
  const stderr = deps.stderr ?? process.stderr;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    const remaining = Math.floor(deadline - now());
    if (remaining <= 1) return 124;
    // Leave part of the overall budget for an independent check even when a
    // model never ends its turn. Five minutes also bounds slow/repeated tools.
    const checkReserve = Math.min(60000, Math.max(1, Math.floor(remaining / 4)));
    const modelBudget = Math.min(roundTimeout, remaining - checkReserve);
    stderr.write(`Editing round ${attempt}/${attempts}: budget ${modelBudget} ms\n`);
    const code = await executeModel(
      roundConfig,
      { ...opts, session, prompt, timeoutMs: modelBudget },
      {
        ...deps,
        onEvent: (event) => {
          if (event.sessionID) session = event.sessionID;
          deps.onEvent?.(event);
        },
      },
    );
    if (code !== 0 && code !== 124) return code;
    if (code === 124)
      stderr.write(`Editing round ${attempt}/${attempts}: timed out; checking partial work\n`);
    const checkBudget = Math.floor(deadline - now());
    if (checkBudget <= 0) return 124;
    stderr.write(`Verification round ${attempt}/${attempts}: started\n`);
    const checked = await executeVerification(
      opts.verify,
      { project: opts.project, timeoutMs: checkBudget },
      deps,
    );
    const checkCode = [130, 143].includes(checked.code)
      ? checked.code
      : now() >= deadline
        ? 124
        : checked.code;
    const status = checkCode === 0 ? 'passed' : checkCode === 124 ? 'timed out' : 'failed';
    stderr.write(`Verification round ${attempt}/${attempts}: ${status}\n`);
    if (checkCode === 0) return 0;
    last = code === 124 ? 124 : checkCode;
    if ([124, 130, 143].includes(checkCode)) return checkCode;
    if (!session) return last;
    prompt = [
      `Continue the coding task.${code === 124 ? ' The previous editing round reached its time limit.' : ''} Independent verification failed with exit ${checked.code}. Fix the implementation using these actual diagnostics; do not weaken tests or remove checks. ${roundInstructions}\n\nOriginal task:\n${opts.prompt.join(' ')}\n\nVerification command:\n${opts.verify}\n\nActual output:\n${checked.output}`,
    ];
  }
  return last;
}

async function main() {
  const version = globalThis.Bun?.version || '0.0.0';
  const [major, minor, patch] = version.split('.').map(Number.parseFloat);
  if (
    major < 1 ||
    (major === 1 &&
      (minor < 4 || (minor === 4 && (patch < 4 || (patch === 4 && version.includes('-'))))))
  ) {
    throw new Error('Bun 1.4.4 or newer is required');
  }
  const opts = parseArgs();
  if (opts.help) {
    console.log(`Usage: bun tools/setup_opencode.mjs [setup|check|run] [options] [-- task]

  --project DIR     Project directory (default: current directory)
  --model ID        Loaded model ID (default: qwen38-27b-q4k-m)
  --context N|auto  Actual server context (default: 16384; minimum: 8192).
                    auto queries the running server's /v1/models once (setup only)
  --output N        Output budget per turn (default: min(4096, context / 4))
  --steps N         Maximum agent steps (default: 24)
  --timeout-ms N    Task timeout (default: 900000; timeout exits 124)
  --round-timeout-ms N  Editing round timeout with --verify (default: 300000)
  --session ID      Continue the session ID from a previous JSON event
  --verify CMD      Run this shell command between short editing rounds (run only)
  --attempts N      Maximum verified editing rounds (default: 3; maximum: 10)
  --force           Replace only opencode.zinc.json
  --dry-run         Setup only: print the config JSON without writing anything
  --auto            Approve headless tool requests; explicit denies still apply

Setup writes opencode.zinc.json and a portable local hook in .zinc/.
The hook bounds file reads and tool output; existing configs are left alone.
Run/check read it. Override the endpoint with ZINC_API_BASE_URL and the key
with ZINC_API_KEY; their values are never saved by this script.
Install OpenCode first: bun install --global opencode-ai@1.19.4

  bun tools/setup_opencode.mjs --project ./my-project
  bun tools/setup_opencode.mjs --dry-run --context 32768
  bun tools/setup_opencode.mjs check --project ./my-project
  bun tools/setup_opencode.mjs run --project ./my-project --auto -- "Fix the failing tests"

Interactive: OPENCODE_CONFIG=./opencode.zinc.json opencode
`);
    return;
  }
  if (
    (opts.verify || opts.attempts !== undefined || opts.roundTimeoutMs !== undefined) &&
    opts.command !== 'run'
  )
    throw new Error('Verification options are only available for run');
  if (opts.attempts !== undefined && !opts.verify) throw new Error('Attempts requires --verify');
  if (opts.roundTimeoutMs !== undefined && !opts.verify)
    throw new Error('Round timeout requires --verify');
  if (opts.dryRun && opts.command !== 'setup')
    throw new Error('--dry-run is only available for setup');
  if (opts.context === 'auto' && opts.command !== 'setup')
    throw new Error('--context auto is setup-only; run setup first, then check or run');
  if (opts.command === 'run') requireTask(opts);
  if (opts.command === 'setup') {
    const config =
      opts.context === 'auto'
        ? buildConfig({ ...opts, context: await resolveAutoContext(opts) })
        : buildConfig(opts);
    if (opts.dryRun) {
      console.log(JSON.stringify(config, null, 2));
      return;
    }
    await writeConfig(opts.project, config, opts);
    console.log(
      `Created ${CONFIG_NAME}. Start ZINC with the same context limit, then use check or run.`,
    );
    return;
  }
  const config = JSON.parse(
    await readFile(path.join(path.resolve(opts.project), CONFIG_NAME), 'utf8'),
  );
  if (opts.command === 'check') {
    console.log(JSON.stringify(await checkServer(config)));
  } else {
    process.exitCode = await (opts.verify
      ? runVerifiedOpenCode(config, opts)
      : runOpenCode(config, opts));
  }
}

if (import.meta.main)
  main().catch((error) => {
    // Do not print fetch/child errors that could include a private URL or token.
    const known =
      /^(Bun 1\.4\.4|Unknown option:|Missing value for|Use a model ID|Context must|Output must|Steps must|Timeout must|Round timeout |Verification |Attempts |ZINC_API_BASE_URL|ZINC .*check|ZINC returned|ZINC context hook|ZINC model discovery|Pass a task|--dry-run is only available for setup|--context auto is setup-only)/.test(
        error.message,
      );
    console.error(
      known
        ? error.message
        : error.code === 'EEXIST'
          ? `${CONFIG_NAME} already exists; use --force to replace it.`
          : 'OpenCode setup/run failed. Check the project, runtime, and ZINC connection.',
    );
    process.exitCode = 1;
  });
