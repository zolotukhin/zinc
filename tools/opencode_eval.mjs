#!/usr/bin/env bun
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, readFile, realpath, rm, stat, writeFile } from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FIXTURES, fixtureById, readOnlyPathsForFixture } from './opencode_eval_fixtures.mjs';
import { buildConfig, CONTEXT_PLUGIN_PATH, runtimeConfig, writeConfig } from './setup_opencode.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(SCRIPT_DIR);
const DEFAULT_MODEL = 'zinc/qwen38-27b-q4k-m';
const DEFAULT_PROXY_MODEL = 'qwen38-27b-q4k-m';
const DEFAULT_ZINC_UPSTREAM = 'http://127.0.0.1:9090/v1';
const DEFAULT_LLAMA_UPSTREAM = 'http://127.0.0.1:9088/v1';
const DEFAULT_PROXY_PORT = 9091;
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const OUTPUT_LIMIT = 8 * 1024 * 1024;

export function parseArgs(argv = process.argv.slice(2)) {
  const opts = {
    provider: 'zinc',
    fixtureIds: ['all'],
    outputDir: null,
    opencodeBin: 'opencode',
    model: DEFAULT_MODEL,
    proxyModel: DEFAULT_PROXY_MODEL,
    proxyPort: DEFAULT_PROXY_PORT,
    zincUpstream: DEFAULT_ZINC_UPSTREAM,
    llamaUpstream: DEFAULT_LLAMA_UPSTREAM,
    traceDir: null,
    manageProxy: false,
    replaceProxy: false,
    keepProxy: false,
    dryRun: false,
    runBaseline: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    maxTokensCap: 4096,
    context: 16384,
    steps: 24,
    forceEnableThinking: false,
    temperature: 0,
    topP: 1,
    repoRoot: REPO_ROOT,
  };

  const args = [...argv];
  const readValue = (i) => {
    if (i + 1 >= args.length || args[i + 1].startsWith('--'))
      throw new Error(`Missing value for ${args[i]}`);
    return [args[i + 1], i + 1];
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--provider') {
      const [v, ni] = readValue(i);
      opts.provider = String(v);
      i = ni;
    } else if (arg === '--fixtures' || arg === '--fixture') {
      const [v, ni] = readValue(i);
      opts.fixtureIds = String(v)
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      i = ni;
    } else if (arg === '--output' || arg === '--output-dir') {
      const [v, ni] = readValue(i);
      opts.outputDir = String(v);
      i = ni;
    } else if (arg === '--opencode-bin') {
      const [v, ni] = readValue(i);
      opts.opencodeBin = String(v);
      i = ni;
    } else if (arg === '--model') {
      const [v, ni] = readValue(i);
      opts.model = String(v);
      i = ni;
    } else if (arg === '--proxy-model') {
      const [v, ni] = readValue(i);
      opts.proxyModel = String(v);
      i = ni;
    } else if (arg === '--proxy-port') {
      const [v, ni] = readValue(i);
      opts.proxyPort = Number(v);
      i = ni;
    } else if (arg === '--zinc-upstream') {
      const [v, ni] = readValue(i);
      opts.zincUpstream = String(v);
      i = ni;
    } else if (arg === '--llama-upstream') {
      const [v, ni] = readValue(i);
      opts.llamaUpstream = String(v);
      i = ni;
    } else if (arg === '--trace-dir') {
      const [v, ni] = readValue(i);
      opts.traceDir = String(v);
      i = ni;
    } else if (arg === '--manage-proxy') {
      opts.manageProxy = true;
    } else if (arg === '--replace-proxy') {
      opts.replaceProxy = true;
    } else if (arg === '--keep-proxy') {
      opts.keepProxy = true;
    } else if (arg === '--dry-run') {
      opts.dryRun = true;
    } else if (arg === '--no-baseline') {
      opts.runBaseline = false;
    } else if (arg === '--timeout-ms') {
      const [v, ni] = readValue(i);
      opts.timeoutMs = Number(v);
      i = ni;
    } else if (arg === '--max-tokens-cap') {
      const [v, ni] = readValue(i);
      opts.maxTokensCap = Number(v);
      i = ni;
    } else if (arg === '--context' || arg === '--steps') {
      const [v, ni] = readValue(i);
      opts[arg.slice(2)] = Number(v);
      i = ni;
    } else if (arg === '--force-enable-thinking') {
      const [v, ni] = readValue(i);
      opts.forceEnableThinking = parseBool(v);
      i = ni;
    } else if (arg === '--temperature') {
      const [v, ni] = readValue(i);
      opts.temperature = Number(v);
      i = ni;
    } else if (arg === '--top-p') {
      const [v, ni] = readValue(i);
      opts.topP = Number(v);
      i = ni;
    } else if (arg === '--list') {
      opts.list = true;
    } else if (arg === '--help' || arg === '-h') {
      opts.help = true;
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (!['zinc', 'llama', 'both'].includes(opts.provider)) throw new Error('Invalid provider');
  if (!opts.fixtureIds.length) throw new Error('Empty fixture list');
  for (const [name, value] of Object.entries({
    timeoutMs: opts.timeoutMs,
    maxTokensCap: opts.maxTokensCap,
    context: opts.context,
    steps: opts.steps,
  })) {
    if (!Number.isSafeInteger(value) || value <= 0)
      throw new Error(`${name} must be a positive integer`);
  }
  if (!Number.isSafeInteger(opts.proxyPort) || opts.proxyPort < 1 || opts.proxyPort > 65535)
    throw new Error('Invalid proxy port');
  if (opts.temperature !== 0 || opts.topP !== 1 || opts.forceEnableThinking)
    throw new Error('This evaluation uses greedy decoding with thinking disabled');

  return opts;
}

function parseBool(value) {
  if (typeof value === 'boolean') return value;
  return /^(1|true|yes|on)$/i.test(String(value));
}

function providersForOption(provider) {
  if (provider === 'both') return ['zinc', 'llama'];
  if (provider === 'zinc' || provider === 'llama') return [provider];
  throw new Error(`Unknown provider: ${provider}`);
}

export function fixturesForOption(ids) {
  if (ids.length === 0 || ids.includes('all')) return FIXTURES;
  return ids.map((id) => {
    const fixture = fixtureById(id);
    if (!fixture) throw new Error(`Unknown fixture: ${id}`);
    return fixture;
  });
}

function providerUpstream(provider, opts) {
  if (provider === 'zinc') return opts.zincUpstream;
  if (provider === 'llama') return opts.llamaUpstream;
  throw new Error(`Unknown provider: ${provider}`);
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
}

async function sha256File(file) {
  try {
    const data = await readFile(file);
    return createHash('sha256').update(data).digest('hex');
  } catch {
    return null;
  }
}

export async function writeFixtureProject(fixture, projectDir) {
  const dir =
    projectDir ?? (await mkdtemp(path.join(os.tmpdir(), `zinc-opencode-eval-${fixture.id}-`)));
  await rm(dir, { recursive: true, force: true });
  await mkdir(dir, { recursive: true });

  for (const [relativePath, content] of Object.entries(fixture.files)) {
    const fullPath = path.join(dir, relativePath);
    await mkdir(path.dirname(fullPath), { recursive: true });
    await writeFile(fullPath, content);
  }

  return dir;
}

function truncateForPrompt(value, limit = 6000) {
  const text = String(value ?? '').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}\n... [truncated ${text.length - limit} chars]`;
}

export function buildPrompt(fixture, projectDir, baselineOutput = '') {
  const files = fixture.requestedFiles.map((file) =>
    path.posix.join(projectDir, file).replace(/\\/g, '/'),
  );
  const lines = [
    fixture.prompt,
    '',
    `Working directory: ${projectDir}`,
    `Read these files before editing: ${files.join(', ')}`,
    'Change only source files. Do not edit package.json, docs, fixtures, or tests unless the task explicitly asks for it.',
    `Run ${fixture.testCommand} after edits. Continue until all tests pass, then stop.`,
  ];
  const trimmedBaseline = truncateForPrompt(baselineOutput);
  if (trimmedBaseline) {
    lines.push(
      '',
      'Initial failing test output from this exact project:',
      '```',
      trimmedBaseline,
      '```',
      'Use the failure lines above to decide the source edit; do not spend turns rereading files already shown.',
    );
  }
  return lines.join('\n');
}

export async function snapshotReadOnlyFiles(projectDir, fixture) {
  const snapshot = {};
  for (const relativePath of readOnlyPathsForFixture(fixture)) {
    snapshot[relativePath] = await sha256File(path.join(projectDir, relativePath));
  }
  return snapshot;
}

export async function readOnlyViolations(projectDir, before) {
  const violations = [];
  for (const [relativePath, hash] of Object.entries(before)) {
    const after = await sha256File(path.join(projectDir, relativePath));
    if (after !== hash) violations.push(relativePath);
  }
  return violations;
}

export async function unexpectedProjectFiles(projectDir, fixture) {
  const expected = new Set([
    ...Object.keys(fixture.files),
    'opencode.zinc.json',
    'opencode.jsonl',
    'baseline-test.log',
    'final-test.log',
    'result.json',
    CONTEXT_PLUGIN_PATH,
  ]);
  const violations = [];
  async function walk(dir, relative = '') {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const name = path.posix.join(relative, entry.name);
      if (entry.isDirectory()) await walk(path.join(dir, entry.name), name);
      else if (
        !entry.isSymbolicLink() &&
        name.startsWith('test/__pycache__/') &&
        name.endsWith('.pyc')
      )
        continue;
      else if (entry.isSymbolicLink() || (!expected.has(name) && !name.startsWith('src/')))
        violations.push(name);
    }
  }
  await walk(projectDir);
  return violations.sort();
}

export async function runCommand(
  command,
  { cwd, timeoutMs = DEFAULT_TIMEOUT_MS, env = process.env, outputFile = null } = {},
) {
  return runProcess('sh', ['-lc', command], { cwd, timeoutMs, env, outputFile });
}

export function buildOpenCodeArgs(provider, fixture, projectDir, opts, baselineOutput = '') {
  return [
    'run',
    '--model',
    opts.model,
    '--auto',
    '--agent',
    'zinc',
    '--format',
    'json',
    '--title',
    `${provider}-${fixture.id}`,
    '--dir',
    projectDir,
    buildPrompt(fixture, projectDir, baselineOutput),
  ];
}

export function parseTestSummary(output) {
  const unittest = String(output).match(/Ran (\d+) tests? in [\d.]+s/);
  if (unittest) {
    const failures =
      Number(String(output).match(/failures=(\d+)/)?.[1] ?? 0) +
      Number(String(output).match(/errors=(\d+)/)?.[1] ?? 0);
    return { pass: Number(unittest[1]) - failures, fail: failures };
  }
  const pass = [...String(output).matchAll(/(?:^|\n)\s*(\d+)\s+pass\b/gi)].at(-1)?.[1];
  const fail = [...String(output).matchAll(/(?:^|\n)\s*(\d+)\s+fail\b/gi)].at(-1)?.[1];
  return {
    pass: pass === undefined ? null : Number(pass),
    fail: fail === undefined ? null : Number(fail),
  };
}

export function parseOpenCodeJsonl(text) {
  const summary = {
    totalEvents: 0,
    malformedJsonLines: 0,
    toolCalls: 0,
    toolCounts: {},
    bashCommands: [],
    fileWrites: [],
    failedTools: 0,
    malformedCommands: [],
    finalTokens: null,
    errors: [],
    sessionId: null,
  };

  for (const line of String(text).split(/\r?\n/)) {
    if (!line.trim()) continue;
    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      summary.malformedJsonLines += 1;
      continue;
    }
    summary.totalEvents += 1;
    if (entry.sessionID) summary.sessionId = entry.sessionID;
    if (entry.type === 'error')
      summary.errors.push(entry.error?.name ?? entry.error?.data?.message ?? 'OpenCode error');
    const part = entry.part;
    if (entry.type === 'step_finish' && part?.tokens) summary.finalTokens = part.tokens;
    if (entry.type !== 'tool_use' || part?.type !== 'tool') continue;

    const tool = part.tool ?? 'unknown';
    const state = part.state ?? {};
    const input = state.input ?? {};
    const metadata = state.metadata ?? {};
    summary.toolCalls += 1;
    summary.toolCounts[tool] = (summary.toolCounts[tool] ?? 0) + 1;
    if (state.status && state.status !== 'completed') summary.failedTools += 1;

    if (tool === 'bash') {
      const command = String(input.command ?? '');
      const item = {
        command,
        exit: Number.isFinite(metadata.exit) ? metadata.exit : null,
        description: input.description ?? '',
      };
      summary.bashCommands.push(item);
      if (/\b(?:npm|bun|pnpm|yarn)\s+test[12]?>/.test(command)) {
        summary.malformedCommands.push(command);
      }
    }

    if (tool === 'write' || tool === 'edit') {
      summary.fileWrites.push({
        tool,
        filePath: input.filePath ?? input.file_path ?? input.path ?? metadata.filepath ?? null,
      });
    }
  }

  return summary;
}

export async function summarizeTraceFiles(traceFiles) {
  const summary = {
    files: traceFiles.length,
    repairedEvents: 0,
    suppressedContentEvents: 0,
    suppressedContentChars: 0,
    bytes: 0,
    shortcuts: {},
    upstreams: {},
  };

  for (const file of traceFiles) {
    let parsed;
    try {
      parsed = JSON.parse(await readFile(file, 'utf8'));
    } catch {
      continue;
    }
    const metrics = parsed.response_metrics ?? {};
    summary.repairedEvents += Number(metrics.repaired_events ?? 0);
    summary.suppressedContentEvents += Number(metrics.suppressed_content_events ?? 0);
    summary.suppressedContentChars += Number(metrics.suppressed_content_chars ?? 0);
    summary.bytes += Number(metrics.bytes ?? 0);
    if (parsed.shortcut)
      summary.shortcuts[parsed.shortcut] = (summary.shortcuts[parsed.shortcut] ?? 0) + 1;
    if (parsed.upstream)
      summary.upstreams[parsed.upstream] = (summary.upstreams[parsed.upstream] ?? 0) + 1;
  }

  return summary;
}

async function listTraceFiles(traceDir, sinceMs = 0) {
  if (!traceDir || !existsSync(traceDir)) return [];
  const names = await readdir(traceDir);
  const files = [];
  for (const name of names) {
    if (!name.endsWith('.json')) continue;
    const fullPath = path.join(traceDir, name);
    const s = await stat(fullPath);
    if (s.mtimeMs >= sinceMs) files.push(fullPath);
  }
  return files.sort();
}

async function runProcess(
  command,
  args,
  { cwd, timeoutMs, env = process.env, outputFile = null } = {},
) {
  const startedAt = Date.now();
  let stdout = '';
  let stderr = '';
  let timedOut = false;
  const outStream = outputFile ? createWriteStream(outputFile, { flags: 'w' }) : null;

  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
    });
    const stop = (signal) => {
      try {
        if (child.pid && process.platform !== 'win32') process.kill(-child.pid, signal);
        else child.kill(signal);
      } catch (error) {
        if (error.code !== 'ESRCH') throw error;
      }
    };
    let killTimer = null;
    const timer = setTimeout(() => {
      timedOut = true;
      stop('SIGTERM');
      killTimer = setTimeout(() => stop('SIGKILL'), 5000);
    }, timeoutMs);

    child.on('error', (err) => {
      clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);
      if (outStream) outStream.end();
      reject(err);
    });
    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      if (outStream) outStream.write(text);
      if (stdout.length < OUTPUT_LIMIT) stdout += text;
    });
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      if (outStream) outStream.write(text);
      if (stderr.length < OUTPUT_LIMIT) stderr += text;
    });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);
      if (outStream) outStream.end();
      resolve({
        command,
        args,
        cwd,
        exitCode: code,
        signal,
        stdout,
        stderr,
        output: `${stdout}${stderr}`,
        timedOut,
        durationMs: Date.now() - startedAt,
      });
    });
  });
}

async function httpOk(url, timeoutMs = 1500) {
  return await new Promise((resolve) => {
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      res.resume();
      res.on('end', () => resolve(res.statusCode >= 200 && res.statusCode < 500));
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

async function waitForHttp(url, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await httpOk(url)) return true;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

export function parsePortListenPids(output) {
  return String(output ?? '')
    .split(/\s+/)
    .map((value) => value.trim())
    .filter((value) => /^\d+$/.test(value));
}

async function portListenPids(port) {
  const result = await runCommand(`lsof -tiTCP:${Number(port)} -sTCP:LISTEN 2>/dev/null || true`, {
    cwd: REPO_ROOT,
    timeoutMs: 5000,
  });
  return parsePortListenPids(result.output);
}

async function killPort(port) {
  const numericPort = Number(port);
  const initialPids = await portListenPids(numericPort);
  if (initialPids.length === 0) return;

  await runCommand(
    `for pid in $(lsof -tiTCP:${numericPort} -sTCP:LISTEN 2>/dev/null); do kill "$pid" 2>/dev/null || true; done`,
    {
      cwd: REPO_ROOT,
      timeoutMs: 5000,
    },
  );

  for (let attempt = 0; attempt < 20; attempt += 1) {
    if ((await portListenPids(numericPort)).length === 0) return;
    await sleep(100);
  }

  await runCommand(
    `for pid in $(lsof -tiTCP:${numericPort} -sTCP:LISTEN 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done`,
    {
      cwd: REPO_ROOT,
      timeoutMs: 5000,
    },
  );

  for (let attempt = 0; attempt < 20; attempt += 1) {
    if ((await portListenPids(numericPort)).length === 0) return;
    await sleep(100);
  }

  throw new Error(`Port ${numericPort} is still in use after replace-proxy cleanup`);
}

async function startManagedProxy(provider, opts, traceDir, logFile) {
  const modelsUrl = `http://127.0.0.1:${opts.proxyPort}/v1/models`;
  if (await httpOk(modelsUrl, 300)) {
    if (!opts.replaceProxy) {
      throw new Error(
        `Port ${opts.proxyPort} is already serving. Choose another --proxy-port or omit --manage-proxy.`,
      );
    }
    await killPort(opts.proxyPort);
    if (await httpOk(modelsUrl, 300)) {
      throw new Error(`Port ${opts.proxyPort} is still serving after --replace-proxy cleanup.`);
    }
  }

  await ensureDir(path.dirname(logFile));
  const log = createWriteStream(logFile, { flags: 'a' });
  const args = [
    path.join('tools', 'opencode_trace_proxy.mjs'),
    'proxy',
    '--listen',
    String(opts.proxyPort),
    '--upstream',
    providerUpstream(provider, opts),
    '--trace-dir',
    traceDir,
    '--force-enable-thinking',
    String(opts.forceEnableThinking),
    '--temperature',
    String(opts.temperature),
    '--top-p',
    String(opts.topP),
    '--max-tokens-cap',
    String(opts.maxTokensCap),
    '--model',
    opts.proxyModel,
    '--inject-path-guard',
    '--repair-tool-paths',
  ];
  const child = spawn('bun', args, {
    cwd: opts.repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.pipe(log);
  child.stderr.pipe(log);

  const ready = await waitForHttp(modelsUrl, 10000);
  await sleep(50);
  if (!ready) {
    child.kill('SIGTERM');
    throw new Error(`Managed proxy for ${provider} did not become ready on ${modelsUrl}`);
  }
  if (child.exitCode !== null || child.signalCode !== null) {
    throw new Error(`Managed proxy for ${provider} exited during startup; see ${logFile}`);
  }

  return {
    child,
    async stop() {
      log.end();
      child.kill('SIGTERM');
      await new Promise((resolve) => setTimeout(resolve, 250));
    },
  };
}

export async function runFixture(provider, fixture, opts, providerDir) {
  const rawProjectDir = path.join(providerDir, 'projects', fixture.id);
  const traceDir = opts.traceDir ?? path.join(providerDir, 'traces');
  await ensureDir(path.dirname(rawProjectDir));
  await ensureDir(traceDir);
  await writeFixtureProject(fixture, rawProjectDir);
  const projectDir = await realpath(rawProjectDir);

  const beforeReadOnly = await snapshotReadOnlyFiles(projectDir, fixture);
  const config = buildConfig({
    model: opts.model.replace(/^zinc\//, ''),
    context: opts.context,
    output: opts.maxTokensCap,
    steps: opts.steps,
  });
  await writeConfig(projectDir, config);
  beforeReadOnly['opencode.zinc.json'] = await sha256File(
    path.join(projectDir, 'opencode.zinc.json'),
  );
  beforeReadOnly[CONTEXT_PLUGIN_PATH] = await sha256File(
    path.join(projectDir, CONTEXT_PLUGIN_PATH),
  );
  const live = runtimeConfig(config, {
    ZINC_API_BASE_URL: opts.manageProxy
      ? `http://127.0.0.1:${opts.proxyPort}/v1`
      : providerUpstream(provider, opts),
  });
  const stateDir = path.join(providerDir, 'state', fixture.id);
  const baseline = opts.runBaseline
    ? await runCommand(fixture.testCommand, {
        cwd: projectDir,
        timeoutMs: Math.min(opts.timeoutMs, 120000),
        outputFile: path.join(projectDir, 'baseline-test.log'),
      })
    : null;
  const prompt = buildPrompt(fixture, projectDir, baseline?.output ?? '');

  let opencode = null;
  let traceFiles = [];
  const traceStartMs = Date.now() - 1000;
  if (!opts.dryRun) {
    const outputFile = path.join(projectDir, 'opencode.jsonl');
    opencode = await runProcess(
      opts.opencodeBin,
      buildOpenCodeArgs(provider, fixture, projectDir, opts, baseline?.output ?? ''),
      {
        cwd: projectDir,
        timeoutMs: opts.timeoutMs,
        outputFile,
        env: {
          ...process.env,
          ZINC_API_KEY: process.env.ZINC_API_KEY || 'local',
          OPENCODE_CONFIG_CONTENT: JSON.stringify(live),
          OPENCODE_DISABLE_AUTOUPDATE: 'true',
          XDG_CONFIG_HOME: path.join(stateDir, 'config'),
          XDG_DATA_HOME: path.join(stateDir, 'data'),
          XDG_CACHE_HOME: path.join(stateDir, 'cache'),
          XDG_STATE_HOME: path.join(stateDir, 'state'),
        },
      },
    );
    traceFiles = await listTraceFiles(traceDir, traceStartMs);
  }

  const finalTest = await runCommand(fixture.testCommand, {
    cwd: projectDir,
    timeoutMs: Math.min(opts.timeoutMs, 120000),
    outputFile: path.join(projectDir, 'final-test.log'),
  });
  const violations = await readOnlyViolations(projectDir, beforeReadOnly);
  const unexpectedFiles = await unexpectedProjectFiles(projectDir, fixture);
  const openCodeSummary = parseOpenCodeJsonl(opencode?.output ?? '');
  const traceSummary = await summarizeTraceFiles(traceFiles);
  const finalSummary = parseTestSummary(finalTest.output);
  const baselineSummary = baseline ? parseTestSummary(baseline.output) : null;
  const success =
    finalTest.exitCode === 0 &&
    finalSummary.pass > 0 &&
    finalSummary.fail === 0 &&
    violations.length === 0 &&
    unexpectedFiles.length === 0 &&
    baseline?.exitCode !== 0 &&
    opencode?.exitCode === 0 &&
    !opencode?.timedOut &&
    openCodeSummary.toolCalls > 0 &&
    openCodeSummary.errors.length === 0 &&
    Object.keys(traceSummary.shortcuts).length === 0;

  const result = {
    provider,
    fixture: fixture.id,
    title: fixture.title,
    tags: fixture.tags,
    success,
    projectDir,
    prompt,
    baseline: baseline
      ? {
          exitCode: baseline.exitCode,
          durationMs: baseline.durationMs,
          tests: baselineSummary,
        }
      : null,
    opencode: opencode
      ? {
          exitCode: opencode.exitCode,
          signal: opencode.signal,
          timedOut: opencode.timedOut,
          durationMs: opencode.durationMs,
          summary: openCodeSummary,
        }
      : null,
    finalTest: {
      exitCode: finalTest.exitCode,
      durationMs: finalTest.durationMs,
      tests: finalSummary,
    },
    readOnlyViolations: violations,
    unexpectedFiles,
    traces: traceSummary,
  };
  await writeFile(path.join(projectDir, 'result.json'), `${JSON.stringify(result, null, 2)}\n`);
  return result;
}

function formatMs(ms) {
  if (ms == null) return '-';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function toolCountsLabel(summary) {
  const counts = summary?.toolCounts ?? {};
  const parts = Object.entries(counts).map(([tool, count]) => `${tool}:${count}`);
  return parts.length ? parts.join(' ') : '-';
}

export function renderSummary(results) {
  const lines = [
    '# OpenCode Coding Eval',
    '',
    '| Provider | Fixture | Result | Tests | Tools | Repairs | Suppressed | Duration |',
    '|---|---|---:|---:|---|---:|---:|---:|',
  ];
  for (const result of results) {
    const tests = result.finalTest.tests;
    const testLabel = tests.pass == null ? 'unknown' : `${tests.pass} pass / ${tests.fail} fail`;
    lines.push(
      `| ${result.provider} | ${result.fixture} | ${result.success ? 'pass' : 'fail'} | ${testLabel} | ${toolCountsLabel(result.opencode?.summary)} | ${result.traces.repairedEvents} | ${result.traces.suppressedContentEvents} | ${formatMs(result.opencode?.durationMs)} |`,
    );
  }
  lines.push('');
  lines.push('Artifacts:');
  for (const result of results) {
    lines.push(`- ${result.provider}/${result.fixture}: ${result.projectDir}`);
  }
  return `${lines.join('\n')}\n`;
}

export async function runEval(opts) {
  const providers = providersForOption(opts.provider);
  const fixtures = fixturesForOption(opts.fixtureIds);
  if (providers.length > 1 && !opts.manageProxy) {
    throw new Error(
      '--provider both requires --manage-proxy so the same OpenCode provider can be pointed at each upstream.',
    );
  }

  const outputDir =
    opts.outputDir ??
    path.join(os.tmpdir(), `zinc-opencode-eval-${new Date().toISOString().replace(/[:.]/g, '-')}`);
  if (existsSync(outputDir) && (await readdir(outputDir)).length > 0)
    throw new Error('Output directory must be empty; choose a new artifact directory');
  await ensureDir(outputDir);

  const results = [];
  for (const provider of providers) {
    const providerDir = path.join(outputDir, provider);
    await ensureDir(providerDir);
    let proxy = null;
    if (opts.manageProxy) {
      proxy = await startManagedProxy(
        provider,
        opts,
        opts.traceDir ?? path.join(providerDir, 'traces'),
        path.join(providerDir, 'proxy.log'),
      );
    }
    try {
      for (const fixture of fixtures) {
        const result = await runFixture(provider, fixture, opts, providerDir);
        results.push(result);
        process.stderr.write(
          `${provider}/${fixture.id}: ${result.success ? 'pass' : 'fail'} (${result.finalTest.tests.pass ?? '?'} pass / ${result.finalTest.tests.fail ?? '?'} fail)\n`,
        );
      }
    } finally {
      if (proxy && !opts.keepProxy) await proxy.stop();
    }
  }

  const summary = renderSummary(results);
  await writeFile(
    path.join(outputDir, 'results.json'),
    `${JSON.stringify({ outputDir, results }, null, 2)}\n`,
  );
  await writeFile(path.join(outputDir, 'summary.md'), summary);
  return { outputDir, results, summary };
}

function printHelp() {
  console.log(`Usage: bun tools/opencode_eval.mjs [options]

Options:
  --provider zinc|llama|both      Provider label to run (default: zinc)
  --fixtures all|id,id            Fixture list (default: all)
  --output DIR                    Artifact directory
  --opencode-bin PATH             OpenCode executable (default: opencode)
  --model MODEL                   OpenCode model id (default: ${DEFAULT_MODEL})
  --manage-proxy                  Start tools/opencode_trace_proxy.mjs for each provider
  --replace-proxy                 Kill an existing listener on --proxy-port before managed runs
  --keep-proxy                    Leave the managed proxy running
  --proxy-port PORT               Proxy port used by OpenCode config (default: ${DEFAULT_PROXY_PORT})
  --zinc-upstream URL             ZINC upstream for managed proxy
  --llama-upstream URL            llama.cpp upstream for managed proxy
  --trace-dir DIR                 Existing or managed proxy trace directory
  --dry-run                       Create projects and run tests without invoking OpenCode
  --timeout-ms MS                 Per-OpenCode task timeout (default: ${DEFAULT_TIMEOUT_MS})
  --list                          List fixtures

Examples:
  bun tools/opencode_eval.mjs --provider zinc --fixtures rate-limiter-single
  bun tools/opencode_eval.mjs --provider both --manage-proxy --replace-proxy
`);
}

async function main() {
  const opts = parseArgs();
  if (opts.help) {
    printHelp();
    return;
  }
  if (opts.list) {
    for (const fixture of FIXTURES) {
      console.log(`${fixture.id}\t${fixture.title}\t${fixture.tags.join(',')}`);
    }
    return;
  }
  const result = await runEval(opts);
  console.log(result.summary);
  console.log(`Wrote ${result.outputDir}`);
  if (result.results.some((entry) => !entry.success)) process.exitCode = 1;
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(err?.stack ?? err);
    process.exit(1);
  });
}
