#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { link, lstat, mkdtemp, open, readFile, rename, rm, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const CONFIG_NAME = 'openclaw.zinc.json';
export const DEFAULT_BASE_URL = 'http://127.0.0.1:9090/v1';
export const INTERRUPTED_CODE = 'OPENCLAW_INTERRUPTED';

export function normalizeBaseUrl(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error('Use an HTTP(S) ZINC API base URL ending in /v1');
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error('Use an HTTP(S) API URL without credentials, query parameters or a fragment');
  url.pathname = url.pathname.replace(/\/+$/, '');
  if (!url.pathname.endsWith('/v1')) throw new Error('The ZINC API base URL must end in /v1');
  return url.href.replace(/\/$/, '');
}

function integer(value, name, minimum, maximum) {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < minimum || result > maximum)
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  return result;
}

export function parseArgs(argv = process.argv.slice(2)) {
  argv = [...argv];
  const opts = {
    command: 'setup',
    project: process.cwd(),
    model: 'auto',
    context: 'auto',
    output: 1024,
    timeout: 600,
  };
  if (argv[0] && !argv[0].startsWith('-')) opts.command = argv.shift();
  if (!['setup', 'check', 'run'].includes(opts.command)) throw new Error('Use setup, check or run');
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') {
      opts.prompt = argv.slice(i + 1).join(' ');
      break;
    }
    if (arg === '--force') {
      opts.force = true;
      continue;
    }
    if (arg === '--dry-run') {
      opts.dryRun = true;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      continue;
    }
    const field = {
      '--project': 'project',
      '--base-url': 'baseUrl',
      '--model': 'model',
      '--context': 'context',
      '--output': 'output',
      '--timeout': 'timeout',
      '--api-key-env': 'apiKeyEnv',
      '--openclaw-bin': 'binary',
    }[arg];
    if (!field) throw new Error('Unknown option; use --help');
    if (!argv[i + 1] || argv[i + 1].startsWith('--'))
      throw new Error('Missing option value; use --help');
    opts[field] = argv[++i];
  }
  opts.output = integer(opts.output, 'Output', 32, 65536);
  opts.timeout = integer(opts.timeout, 'Timeout', 1, 7200);
  if (opts.context !== 'auto') opts.context = integer(opts.context, 'Context', 4000, 1048576);
  if (opts.apiKeyEnv && !/^[A-Z][A-Z0-9_]*$/.test(opts.apiKeyEnv))
    throw new Error('Use an environment variable name for the API key');
  if (opts.dryRun && opts.command !== 'setup') throw new Error('--dry-run is setup-only');
  if (opts.command === 'run' && !opts.help && !opts.prompt?.trim())
    throw new Error('Pass a task after --');
  return opts;
}

export function buildConfig({
  baseUrl = DEFAULT_BASE_URL,
  model,
  context,
  output = 1024,
  timeout = 600,
  apiKeyEnv,
} = {}) {
  const base = normalizeBaseUrl(baseUrl);
  if (
    typeof model !== 'string' ||
    !model.trim() ||
    model === 'auto' ||
    !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(model)
  )
    throw new Error('Use an exact model ID from ZINC discovery');
  const capacity = integer(context, 'Context', 4000, 1048576);
  const cap = integer(output, 'Output', 32, capacity - 1024);
  const deadline = integer(timeout, 'Timeout', 1, 7200);
  if (apiKeyEnv && !/^[A-Z][A-Z0-9_]*$/.test(apiKeyEnv))
    throw new Error('Use an environment variable name for the API key');
  const ref = `zinc/${model}`;
  return {
    agents: {
      defaults: {
        skills: [],
        model: { primary: ref },
        timeoutSeconds: deadline,
        models: { [ref]: { alias: 'ZINC', params: { temperature: 0, maxTokens: cap } } },
      },
    },
    models: {
      mode: 'replace',
      providers: {
        zinc: {
          baseUrl: base,
          api: 'openai-completions',
          apiKey: apiKeyEnv ? `\${${apiKeyEnv}}` : 'zinc-local',
          timeoutSeconds: deadline,
          models: [
            {
              id: model,
              name: model,
              reasoning: false,
              input: ['text'],
              contextWindow: capacity,
              maxTokens: cap,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              compat: {
                supportsStore: false,
                supportsDeveloperRole: false,
                supportsReasoningEffort: false,
                supportsUsageInStreaming: true,
                supportsTools: true,
                supportsStrictMode: false,
                maxTokensField: 'max_tokens',
              },
            },
          ],
        },
      },
    },
    tools: {
      profile: 'coding',
      toolSearch: false,
      allow: ['read', 'write', 'edit', 'exec', 'process'],
      fs: { workspaceOnly: true },
    },
  };
}

function authValue(apiKeyEnv, env) {
  if (!apiKeyEnv) return 'zinc-local';
  if (!env[apiKeyEnv]?.trim())
    throw new Error('The configured ZINC API key environment variable is missing');
  return env[apiKeyEnv];
}

async function jsonRequest(url, { key, fetchImpl = fetch, timeout = 10000 } = {}) {
  let response;
  try {
    response = await fetchImpl(url, {
      headers: { authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(timeout),
    });
  } catch {
    throw new Error('ZINC discovery failed or timed out');
  }
  if (!response.ok) throw new Error(`ZINC discovery returned HTTP ${response.status}`);
  const limit = 1024 * 1024;
  const reader = response.body?.getReader();
  if (!reader) throw new Error('ZINC discovery returned an empty body');
  let text,
    tooLarge = false;
  const chunks = [];
  let bytes = 0;
  try {
    if (Number(response.headers.get('content-length')) > limit) {
      tooLarge = true;
      await reader.cancel();
      throw new Error();
    }
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > limit) {
        tooLarge = true;
        await reader.cancel();
        throw new Error();
      }
      chunks.push(value);
    }
    text = Buffer.concat(chunks, bytes).toString('utf8');
  } catch {
    if (tooLarge) throw new Error('ZINC discovery response is too large');
    throw new Error('ZINC discovery body failed or timed out');
  } finally {
    reader.releaseLock();
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error('ZINC discovery returned malformed JSON');
  }
}

export async function discoverServer(
  { baseUrl = DEFAULT_BASE_URL, model = 'auto', context = 'auto', apiKeyEnv } = {},
  { fetchImpl = fetch, env = process.env } = {},
) {
  const base = normalizeBaseUrl(baseUrl),
    key = authValue(apiKeyEnv, env);
  const [health, listing] = await Promise.all([
    jsonRequest(`${base.slice(0, -3)}/health`, { key, fetchImpl }),
    jsonRequest(`${base}/models`, { key, fetchImpl }),
  ]);
  if (health?.status !== 'ok') throw new Error('ZINC is not healthy');
  if (!Array.isArray(listing?.data) || !listing.data.length)
    throw new Error('ZINC discovery returned no models');
  const ids = new Set();
  for (const row of listing.data) {
    if (!row || typeof row.id !== 'string' || !row.id.trim() || ids.has(row.id))
      throw new Error('ZINC discovery returned invalid or duplicate model IDs');
    ids.add(row.id);
  }
  const matches =
    model === 'auto'
      ? listing.data.filter(
          (row) => row.active === true || (listing.data.length === 1 && row.active !== false),
        )
      : listing.data.filter((row) => row.id === model && row.active !== false);
  if (matches.length !== 1)
    throw new Error('Select exactly one loaded ZINC model; use --model with its discovery ID');
  const row = matches[0];
  const available = integer(row.context_length, 'Discovered context', 4000, 1048576);
  const healthCapacity = health.gpu_context_capacity_tokens ?? health.context_length;
  if (healthCapacity !== undefined && healthCapacity !== available)
    throw new Error('ZINC health and model discovery disagree about context capacity');
  const requested = context === 'auto' ? available : integer(context, 'Context', 4000, available);
  return { baseUrl: base, model: row.id, context: requested };
}

async function regularFile(filename) {
  const stat = await lstat(filename);
  if (!stat.isFile() || stat.isSymbolicLink())
    throw new Error('The OpenClaw config must be a regular file');
  return stat;
}

export async function writeConfig(project, config, { force = false } = {}) {
  const directory = path.resolve(project);
  const stat = await lstat(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Use a real project directory');
  const filename = path.join(directory, CONFIG_NAME);
  const text = JSON.stringify(config, null, 2) + '\n';
  let exists = false;
  try {
    await regularFile(filename);
    exists = true;
    if ((await readFile(filename, 'utf8')) === text) return { filename, changed: false };
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  if (exists && !force)
    throw new Error('OpenClaw config already differs; use --force to replace this dedicated file');
  const temporary = path.join(directory, `.${CONFIG_NAME}.${randomUUID()}.tmp`);
  try {
    const handle = await open(temporary, 'wx', 0o600);
    try {
      await handle.writeFile(text);
      await handle.sync();
    } finally {
      await handle.close();
    }
    // Existing global OpenClaw configuration is outside this helper's scope.
    if (force) await rename(temporary, filename);
    else {
      // A hard link publishes the complete file without replacing a config
      // created by another setup process between our check and this write.
      try {
        await link(temporary, filename);
      } catch (error) {
        if (error.code === 'EEXIST')
          throw new Error('OpenClaw config appeared during setup; retry without --force');
        throw error;
      }
    }
  } finally {
    await unlink(temporary).catch((error) => {
      if (error.code !== 'ENOENT') throw error;
    });
  }
  return { filename, changed: true };
}

export async function loadConfig(project) {
  const filename = path.join(path.resolve(project), CONFIG_NAME);
  await regularFile(filename);
  let config;
  try {
    config = JSON.parse(await readFile(filename, 'utf8'));
  } catch {
    throw new Error('The dedicated OpenClaw config is not valid JSON');
  }
  const provider = config.models?.providers?.zinc;
  const model = provider?.models?.find(
    (row) => `zinc/${row.id}` === config.agents?.defaults?.model?.primary,
  );
  if (!model || provider.api !== 'openai-completions')
    throw new Error('The config has no ZINC chat-completions model');
  normalizeBaseUrl(provider.baseUrl);
  integer(model.contextWindow, 'Configured context', 4000, 1048576);
  integer(model.maxTokens, 'Configured output', 32, model.contextWindow - 1024);
  return { filename, config, provider, model };
}

export function runArguments(filename, model, project, timeout) {
  return [
    'agent',
    'exec',
    '--config',
    filename,
    '--cwd',
    path.resolve(project),
    '--model',
    `zinc/${model}`,
    '--code-mode',
    'direct',
    '--local-model-lean',
    '--thinking',
    'off',
    '--timeout',
    String(timeout),
    '--json',
    '--message-file',
    '-',
  ];
}

async function clientCommand(
  { binary, args, project, input = '', timeout },
  { env = process.env, spawnImpl = spawn, deadlineMs = (timeout + 30) * 1000, graceMs = 2000 } = {},
) {
  // One-shot OpenClaw owns its temporary state and retires its resources. Own
  // the process group too, so timeout or Ctrl-C cannot orphan tool processes.
  const state = await mkdtemp(path.join(tmpdir(), 'zinc-openclaw-'));
  const childEnv = { ...env, OPENCLAW_STATE_DIR: state };
  let child;
  try {
    const executable =
      binary.includes('/') || binary.includes('\\') ? path.resolve(binary) : binary;
    child = spawnImpl(executable, args, {
      env: childEnv,
      cwd: path.resolve(project),
      detached: process.platform !== 'win32',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch {
    await rm(state, { recursive: true, force: true });
    throw new Error('OpenClaw could not start; check --openclaw-bin and its Node.js requirement');
  }
  const stop = (signal) => {
    if (!child.pid) return;
    try {
      process.platform === 'win32' ? child.kill(signal) : process.kill(-child.pid, signal);
    } catch (error) {
      if (error.code !== 'ESRCH') throw error;
    }
  };
  let stdout = '',
    stderr = '',
    failure,
    escalation;
  const cancel = (message, code = 'OPENCLAW_OUTPUT_LIMIT') => {
    if (failure) return;
    failure = { message, code };
    stop('SIGTERM');
    escalation = setTimeout(() => stop('SIGKILL'), graceMs);
  };
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (text) => {
    if (stdout.length <= 4 * 1024 * 1024) stdout += text;
    if (stdout.length > 4 * 1024 * 1024) cancel('OpenClaw output exceeded the diagnostic limit');
  });
  child.stderr.on('data', (text) => {
    if (stderr.length <= 4 * 1024 * 1024) stderr += text;
    if (stderr.length > 4 * 1024 * 1024) cancel('OpenClaw output exceeded the diagnostic limit');
  });
  const timer = setTimeout(
    () => cancel('OpenClaw exceeded its task timeout', 'OPENCLAW_TIMEOUT'),
    deadlineMs,
  );
  const interrupt = () => cancel('OpenClaw was interrupted', INTERRUPTED_CODE);
  process.once('SIGINT', interrupt);
  process.once('SIGTERM', interrupt);
  try {
    const completion = new Promise((resolve, reject) => {
      child.once('error', () =>
        reject(
          new Error('OpenClaw could not start; check --openclaw-bin and its Node.js requirement'),
        ),
      );
      child.once('close', (code, signal) => resolve({ code, signal }));
    });
    child.stdin.on('error', () => {});
    child.stdin.end(input);
    const result = await completion;
    if (failure) {
      const error = new Error(failure.message);
      error.code = failure.code;
      error.diagnostics = { code: result.code, signal: result.signal, stdout, stderr };
      throw error;
    }
    if (result.code !== 0) {
      const error = new Error('OpenClaw failed; inspect its task diagnostics');
      error.diagnostics = { code: result.code, signal: result.signal, stdout, stderr };
      throw error;
    }
    let report;
    try {
      report = JSON.parse(stdout);
    } catch {
      throw new Error('OpenClaw returned invalid JSON');
    }
    return { report, stderr };
  } finally {
    clearTimeout(timer);
    clearTimeout(escalation);
    process.removeListener('SIGINT', interrupt);
    process.removeListener('SIGTERM', interrupt);
    stop('SIGKILL');
    await rm(state, { recursive: true, force: true });
  }
}

export async function validateClient({ binary = 'openclaw', filename, project }, dependencies) {
  const result = await clientCommand(
    { binary, args: ['config', 'validate', '--json'], project, timeout: 30 },
    {
      ...dependencies,
      env: { ...(dependencies?.env || process.env), OPENCLAW_CONFIG_PATH: filename },
    },
  );
  if (result.report.valid !== true) throw new Error('OpenClaw rejected the dedicated config');
  return result;
}

export async function runOpenClaw(
  { binary = 'openclaw', filename, model, project, prompt, timeout = 600 },
  dependencies,
) {
  const result = await clientCommand(
    {
      binary,
      args: runArguments(filename, model, project, timeout),
      project,
      input: prompt,
      timeout,
    },
    dependencies,
  );
  if (result.report.ok !== true || result.report.status !== 'ok' || result.report.error)
    throw new Error('OpenClaw did not report a successful completed task');
  return result;
}

export async function main(argv = process.argv.slice(2)) {
  const opts = parseArgs([...argv]);
  if (opts.help) {
    console.log(
      `ZINC → OpenClaw setup (OpenClaw 2026.11.1 tested)\n\nnode setup-openclaw.mjs [setup|check|run] [options] [-- task]\n  --base-url URL      ZINC API base (default http://127.0.0.1:9090/v1)\n  --project DIR       Project containing ${CONFIG_NAME}\n  --model ID          Exact loaded discovery ID (default auto)\n  --context N|auto    Actual server capacity (default auto; minimum 4000)\n  --output N          Output budget (default 1024)\n  --timeout SECONDS   One-shot deadline (default 600)\n  --api-key-env NAME  Keep a protected endpoint's key in the environment\n  --openclaw-bin PATH OpenClaw executable\n  --dry-run           Print config; requires explicit model and context\n  --force             Replace only the dedicated project config\n\nStart ZINC before setup. Run invokes the real one-shot client; verify its file changes and project tests independently.`,
    );
    return;
  }
  if (opts.command === 'setup') {
    const discovered = opts.dryRun
      ? opts
      : await discoverServer({
          ...opts,
          baseUrl: opts.baseUrl || process.env.ZINC_API_BASE_URL || DEFAULT_BASE_URL,
        });
    const config = buildConfig({
      ...opts,
      ...discovered,
      baseUrl: discovered.baseUrl || process.env.ZINC_API_BASE_URL || DEFAULT_BASE_URL,
    });
    if (opts.dryRun) {
      console.log(JSON.stringify(config, null, 2));
      return;
    }
    const written = await writeConfig(opts.project, config, opts);
    console.log(
      written.changed
        ? 'Wrote the dedicated OpenClaw ZINC config.'
        : 'OpenClaw ZINC config is already current.',
    );
    return;
  }
  const loaded = await loadConfig(opts.project);
  const key = loaded.provider.apiKey;
  const match = typeof key === 'string' && key.match(/^\$\{([A-Z][A-Z0-9_]*)\}$/);
  await discoverServer({
    baseUrl: loaded.provider.baseUrl,
    model: loaded.model.id,
    context: loaded.model.contextWindow,
    ...(match ? { apiKeyEnv: match[1] } : {}),
  });
  if (opts.command === 'check') {
    await validateClient({
      binary: opts.binary || process.env.OPENCLAW_BIN || 'openclaw',
      filename: loaded.filename,
      project: opts.project,
    });
    console.log('OpenClaw accepts the config; ZINC health, model identity and context match.');
    return;
  }
  const result = await runOpenClaw({
    binary: opts.binary || process.env.OPENCLAW_BIN || 'openclaw',
    filename: loaded.filename,
    model: loaded.model.id,
    project: opts.project,
    prompt: opts.prompt,
    timeout: opts.timeout,
  });
  console.log(JSON.stringify(result.report, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`OpenClaw setup: ${error.message}`);
    process.exitCode = 1;
  });
}
