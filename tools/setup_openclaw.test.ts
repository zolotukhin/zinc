import { afterEach, describe, expect, test } from 'bun:test';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  buildConfig,
  CONFIG_NAME,
  discoverServer,
  loadConfig,
  normalizeBaseUrl,
  parseArgs,
  runOpenClaw,
  validateClient,
  writeConfig,
} from './setup_openclaw.mjs';

const dirs: string[] = [];
async function temp() {
  const dir = await mkdtemp(path.join(tmpdir(), 'zinc-openclaw-test-'));
  dirs.push(dir);
  return dir;
}
afterEach(async () => {
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true });
});
const config = (extra = {}) => buildConfig({ model: 'qwen35-9b-q4k-m', context: 8192, ...extra });
const discovery = (data: any[], health: any = { status: 'ok' }) => ({
  env: { TEST_ZINC_KEY: 'private-test-key' },
  fetchImpl: async (url: string, options: any) => {
    expect(options.headers.authorization).toBe('Bearer private-test-key');
    return Response.json(url.endsWith('/health') ? health : { data });
  },
});

describe('OpenClaw project setup', () => {
  test('pins the actual chat API, model limits and compatible tool protocol', () => {
    const value = config({ apiKeyEnv: 'TEST_ZINC_KEY' });
    expect(value.agents.defaults.model.primary).toBe('zinc/qwen35-9b-q4k-m');
    expect(value.models.mode).toBe('replace');
    expect(value.models.providers.zinc.apiKey).toBe('${TEST_ZINC_KEY}');
    expect(value.models.providers.zinc.api).toBe('openai-completions');
    expect(value.models.providers.zinc.models[0]).toMatchObject({
      contextWindow: 8192,
      maxTokens: 1024,
      reasoning: false,
      compat: {
        supportsDeveloperRole: false,
        supportsTools: true,
        supportsUsageInStreaming: true,
        maxTokensField: 'max_tokens',
      },
    });
    expect(value.tools.fs.workspaceOnly).toBe(true);
    expect(value.tools.toolSearch).toBe(false);
    expect(value.agents.defaults.skills).toEqual([]);
    expect(value.agents.defaults).not.toHaveProperty('contextTokens');
    expect(JSON.stringify(value)).not.toContain(process.cwd());
  });
  test('parses literal task text without mutating caller arguments', () => {
    const args = ['run', '--project', 'with spaces', '--', 'explain --help $(date)'];
    const before = [...args];
    expect(parseArgs(args).prompt).toBe('explain --help $(date)');
    expect(args).toEqual(before);
    for (const args of [['--bad'], ['--model'], ['run'], ['check', '--dry-run']])
      expect(() => parseArgs(args)).toThrow();
  });
  test('rejects invalid limits, substitution IDs, credential URLs and API key values', () => {
    for (const model of ['', 'auto', '../model', 'a/b', '${SECRET}', 'a\nb', 'a b'])
      expect(() => config({ model })).toThrow();
    for (const context of [3999, NaN, 8000.5, Infinity])
      expect(() => config({ context })).toThrow();
    for (const output of [0, 8000, -1, NaN]) expect(() => config({ output })).toThrow();
    for (const base of [
      'file:///v1',
      'http://u:p@localhost/v1',
      'http://localhost/v1?q=key',
      'http://localhost/v1#secret',
      'http://localhost/api',
    ])
      expect(() => normalizeBaseUrl(base)).toThrow();
    expect(() => config({ apiKeyEnv: 'an actual key' })).toThrow();
    expect(normalizeBaseUrl('http://localhost:9090/v1/')).toBe('http://localhost:9090/v1');
  });
  test('uses Vulkan active identity and actual capacity, or ROCm single-row identity', async () => {
    const rows = [
      { id: 'inactive', active: false, context_length: 0 },
      { id: 'loaded', active: true, context_length: 16384 },
    ];
    expect(
      await discoverServer(
        { apiKeyEnv: 'TEST_ZINC_KEY' },
        discovery(rows, { status: 'ok', gpu_context_capacity_tokens: 16384 }),
      ),
    ).toMatchObject({ model: 'loaded', context: 16384 });
    expect(
      await discoverServer(
        { apiKeyEnv: 'TEST_ZINC_KEY', context: 8192 },
        discovery([{ id: 'loaded.gguf', context_length: 16384 }]),
      ),
    ).toMatchObject({ model: 'loaded.gguf', context: 8192 });
  });
  test('refuses inactive, ambiguous, duplicate, unhealthy and oversized discovery', async () => {
    const opts = { apiKeyEnv: 'TEST_ZINC_KEY' };
    for (const rows of [
      [],
      [{ id: 'a', active: false, context_length: 8192 }],
      [
        { id: 'a', active: true, context_length: 8192 },
        { id: 'b', active: true, context_length: 8192 },
      ],
      [{ id: 'a' }, { id: 'a' }],
      [{ id: 'a', context_length: 2048 }],
    ])
      await expect(discoverServer(opts, discovery(rows))).rejects.toThrow();
    const row = [{ id: 'loaded', active: true, context_length: 8192 }];
    await expect(discoverServer({ ...opts, model: 'other' }, discovery(row))).rejects.toThrow();
    await expect(discoverServer({ ...opts, context: 16384 }, discovery(row))).rejects.toThrow();
    await expect(discoverServer(opts, discovery(row, { status: 'loading' }))).rejects.toThrow();
    await expect(
      discoverServer(opts, discovery(row, { status: 'ok', gpu_context_capacity_tokens: 4096 })),
    ).rejects.toThrow('disagree');
    await expect(discoverServer(opts, { env: {} })).rejects.toThrow('missing');
    for (const fetchImpl of [
      async () => new Response('no', { status: 503 }),
      async () => new Response('broken'),
      async () => {
        throw new Error('secret endpoint');
      },
    ])
      await expect(discoverServer({}, { fetchImpl })).rejects.toThrow('ZINC discovery');
  });
  test('cancels oversized discovery during transfer, including multibyte bodies', async () => {
    for (const declared of [false, true]) {
      let pulls = 0,
        cancelled = false;
      const body = new ReadableStream({
        pull(controller) {
          pulls++;
          controller.enqueue(new TextEncoder().encode('é'.repeat(300000)));
          if (pulls === 20) controller.close();
        },
        cancel() {
          cancelled = true;
        },
      });
      const fetchImpl = async (url: string) =>
        url.endsWith('/health')
          ? Response.json({ status: 'ok' })
          : new Response(body, {
              headers: declared ? { 'content-length': '12000000' } : {},
            });
      await expect(discoverServer({}, { fetchImpl })).rejects.toThrow('too large');
      expect(cancelled).toBe(true);
      expect(pulls).toBeLessThanOrEqual(3);
    }
  });
  test('preserves global and unrelated files, is idempotent, and requires explicit replacement', async () => {
    const project = await temp();
    await writeFile(path.join(project, 'openclaw.json'), 'keep global settings');
    expect((await writeConfig(project, config())).changed).toBe(true);
    expect((await stat(path.join(project, CONFIG_NAME))).mode & 0o777).toBe(0o600);
    expect((await writeConfig(project, config())).changed).toBe(false);
    await expect(writeConfig(project, config({ context: 16384 }))).rejects.toThrow('differs');
    await writeConfig(project, config({ context: 16384 }), { force: true });
    expect((await loadConfig(project)).model.contextWindow).toBe(16384);
    expect(await readFile(path.join(project, 'openclaw.json'), 'utf8')).toBe(
      'keep global settings',
    );
    expect((await readdir(project)).sort()).toEqual(['openclaw.json', CONFIG_NAME].sort());
  });
  test('refuses file and project symlinks even with force', async () => {
    const project = await temp(),
      target = path.join(await temp(), 'keep');
    await writeFile(target, 'preserve');
    await symlink(target, path.join(project, CONFIG_NAME));
    await expect(writeConfig(project, config(), { force: true })).rejects.toThrow('regular file');
    await expect(loadConfig(project)).rejects.toThrow('regular file');
    expect(await readFile(target, 'utf8')).toBe('preserve');
    const alias = path.join(await temp(), 'project');
    await symlink(project, alias);
    await expect(writeConfig(alias, config())).rejects.toThrow('real project');
  });
  test('simultaneous setup cannot clobber a competing config', async () => {
    const project = await temp();
    const values = [config(), config({ context: 16384 })];
    const results = await Promise.allSettled(values.map((value) => writeConfig(project, value)));
    expect(results.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
    const winner = results.findIndex((result) => result.status === 'fulfilled');
    expect(JSON.parse(await readFile(path.join(project, CONFIG_NAME), 'utf8'))).toEqual(
      values[winner],
    );
  });
});

async function fakeClient(source: string) {
  const project = await temp(),
    script = path.join(project, 'client.mjs');
  await writeFile(script, source);
  return {
    project,
    dependencies: {
      spawnImpl: (_bin: string, args: string[], options: any) =>
        spawn(process.execPath, [script, ...args], options),
    },
    options: {
      project,
      filename: path.join(project, CONFIG_NAME),
      model: 'loaded',
      prompt: 'literal $(date) --flags',
      timeout: 1,
    },
  };
}

describe('OpenClaw owned process and task result', () => {
  test('passes stdin literally, uses one-shot flags, and removes isolated state', async () => {
    const f = await fakeClient(
      `let input='';for await(const text of process.stdin)input+=text;console.log(JSON.stringify({ok:true,status:'ok',input,args:process.argv.slice(2),state:process.env.OPENCLAW_STATE_DIR}));`,
    );
    const { report } = await runOpenClaw(f.options, f.dependencies);
    expect(report.input).toBe(f.options.prompt);
    expect(report.args.slice(0, 2)).toEqual(['agent', 'exec']);
    expect(report.args).toContain('--local-model-lean');
    expect(report.args).toContain('--message-file');
    expect(report.args).not.toContain('--deliver');
    expect(report.args).not.toContain('--isolated');
    await expect(stat(report.state)).rejects.toThrow();
  });
  test('does not mistake exit zero for a completed task', async () => {
    for (const source of [
      `console.log('broken')`,
      `console.log(JSON.stringify({ok:false,status:'error'}))`,
      `console.log(JSON.stringify({ok:true,status:'ok',error:'failed'}))`,
      `process.exit(7)`,
    ]) {
      const f = await fakeClient(source);
      await expect(runOpenClaw(f.options, f.dependencies)).rejects.toThrow('OpenClaw');
    }
  });
  test('escalates a deadline even if a child ignores TERM', async () => {
    const f = await fakeClient(`process.on('SIGTERM',()=>{});setInterval(()=>{},1000);`);
    const start = Date.now();
    await expect(
      runOpenClaw(f.options, { ...f.dependencies, deadlineMs: 200, graceMs: 100 }),
    ).rejects.toThrow('timeout');
    expect(Date.now() - start).toBeLessThan(3000);
  });
  test('handles missing executable without leaking state or masking the error', async () => {
    const project = await temp();
    await expect(
      runOpenClaw({
        binary: path.join(project, 'missing'),
        project,
        filename: 'config',
        model: 'loaded',
        prompt: 'task',
      }),
    ).rejects.toThrow('could not start');
  });
  test('resolves a relative client binary before changing to the task project', async () => {
    const directory = await temp(),
      project = await temp(),
      binary = path.join(directory, 'client with spaces.mjs');
    await writeFile(
      binary,
      '#!/usr/bin/env node\nconsole.log(JSON.stringify({ok:true,status:"ok"}));\n',
      { mode: 0o700 },
    );
    const result = await runOpenClaw({
      binary: path.relative(process.cwd(), binary),
      project,
      filename: 'config',
      model: 'loaded',
      prompt: 'task',
    });
    expect(result.report.ok).toBe(true);
  });
  test('runs the client schema validator with the dedicated config path', async () => {
    const f = await fakeClient(
      `console.log(JSON.stringify({valid:true,args:process.argv.slice(2),config:process.env.OPENCLAW_CONFIG_PATH}));`,
    );
    const { report } = await validateClient(f.options, f.dependencies);
    expect(report.args).toEqual(['config', 'validate', '--json']);
    expect(report.config).toBe(f.options.filename);
  });
});
