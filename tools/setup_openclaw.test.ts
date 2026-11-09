import { afterEach, describe, expect, test } from 'bun:test';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'node:http';
import path from 'node:path';
import {
  buildConfig,
  CONFIG_NAME,
  discoverServer,
  isRuntimePreparationFailure,
  loadConfig,
  normalizeBaseUrl,
  parseArgs,
  runOpenClaw,
  sessionRuntimeConfig,
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

describe('OpenClaw per-run session reuse', () => {
  const runtime = () => config({ model: 'loaded', apiKeyEnv: 'TEST_ZINC_KEY' });
  const capable = (extra = {}) => ({
    env: { TEST_ZINC_KEY: 'private-test-key' },
    sessionId: () => 'test-run',
    fetchImpl: async (url: string, options: any) => {
      expect(url).toBe('http://127.0.0.1:9090/health');
      expect(options.headers.authorization).toBe('Bearer private-test-key');
      expect(options.signal).toBeInstanceOf(AbortSignal);
      return Response.json({
        status: 'ok',
        capabilities: { chat_session_reuse: true },
      });
    },
    ...extra,
  });
  test('only injects into the qualified model, preserving configuration and environment references', async () => {
    const original = runtime();
    original.agents.defaults.models['zinc/loaded'].params.extraBody = {
      custom: 7,
    };
    const before = JSON.stringify(original);
    const value = await sessionRuntimeConfig(original, 'loaded', capable());
    expect(value.agents.defaults.models['zinc/loaded'].params).toEqual({
      temperature: 0,
      maxTokens: 1024,
      extraBody: { custom: 7, session_id: 'openclaw-test-run' },
    });
    expect(value.models.providers.zinc.apiKey).toBe('${TEST_ZINC_KEY}');
    expect(JSON.stringify(value)).not.toContain('private-test-key');
    expect(JSON.stringify(original)).toBe(before);
    expect(await sessionRuntimeConfig(original, 'different', capable())).toBe(original);
  });
  test('normalizes trailing slashes and preserves reverse-proxy path prefixes', async () => {
    for (const [baseUrl, healthUrl] of [
      ['http://localhost:9090/v1/', 'http://localhost:9090/health'],
      ['http://localhost:9090/api/v1', 'http://localhost:9090/api/health'],
      ['http://localhost:9090/api/v1///', 'http://localhost:9090/api/health'],
    ]) {
      const original = runtime();
      original.models.providers.zinc.baseUrl = baseUrl;
      let observed;
      const value = await sessionRuntimeConfig(
        original,
        'loaded',
        capable({
          fetchImpl: async (url: string) => {
            observed = url;
            return Response.json({
              status: 'ok',
              capabilities: { chat_session_reuse: true },
            });
          },
        }),
      );
      expect(observed).toBe(healthUrl);
      expect(value).not.toBe(original);
    }
  });
  test('rechecks capability after a model or backend changes at the same endpoint', async () => {
    let requests = 0;
    const deps = capable({
      fetchImpl: async () =>
        Response.json({
          status: 'ok',
          capabilities: { chat_session_reuse: ++requests === 1 },
        }),
    });
    const attention = runtime();
    expect(await sessionRuntimeConfig(attention, 'loaded', deps)).not.toBe(attention);
    const hybrid = config({ model: 'hybrid', apiKeyEnv: 'TEST_ZINC_KEY' });
    expect(await sessionRuntimeConfig(hybrid, 'hybrid', deps)).toBe(hybrid);
    expect(requests).toBe(2);
  });
  test('retains full prefill for absent, false, malformed or unhealthy capabilities and discovery errors', async () => {
    for (const health of [
      null,
      {},
      { status: 'ok' },
      { status: 'loading', capabilities: { chat_session_reuse: true } },
      { status: 'ok', capabilities: { chat_session_reuse: false } },
      { status: 'ok', capabilities: { chat_session_reuse: 'true' } },
    ]) {
      const original = runtime();
      expect(
        await sessionRuntimeConfig(
          original,
          'loaded',
          capable({ fetchImpl: async () => Response.json(health) }),
        ),
      ).toBe(original);
    }
    for (const fetchImpl of [
      async () => new Response('bad JSON'),
      async () => new Response('', { status: 503 }),
      async () => {
        throw new Error('private endpoint detail');
      },
    ]) {
      const original = runtime();
      expect(await sessionRuntimeConfig(original, 'loaded', capable({ fetchImpl }))).toBe(original);
    }
  });
  test('respects both body aliases and all explicit session opt-outs without querying health', async () => {
    const forbidden = async () => {
      throw new Error('health must not be queried');
    };
    for (const alias of ['extraBody', 'extra_body'])
      for (const key of ['', null, 'caller-session']) {
        const original = runtime();
        original.agents.defaults.models['zinc/loaded'].params[alias] = {
          session_id: key,
        };
        let called = false;
        expect(
          await sessionRuntimeConfig(
            original,
            'loaded',
            capable({
              fetchImpl: async () => {
                called = true;
                return forbidden();
              },
            }),
          ),
        ).toBe(original);
        expect(called).toBe(false);
      }
    const original = runtime();
    const params = original.agents.defaults.models['zinc/loaded'].params;
    params.extra_body = { custom: 7 };
    const value = await sessionRuntimeConfig(original, 'loaded', capable());
    expect(value.agents.defaults.models['zinc/loaded'].params.extra_body).toEqual({
      custom: 7,
      session_id: 'openclaw-test-run',
    });
    expect(value.agents.defaults.models['zinc/loaded'].params).not.toHaveProperty('extraBody');
  });
  test('does not repair malformed or ambiguous custom options', async () => {
    for (const params of [
      null,
      [],
      { extraBody: null },
      { extra_body: [] },
      { extraBody: {}, extra_body: {} },
    ]) {
      const original = runtime();
      original.agents.defaults.models['zinc/loaded'].params = params;
      expect(await sessionRuntimeConfig(original, 'loaded', capable())).toBe(original);
    }
    expect(await sessionRuntimeConfig(null, 'loaded', capable())).toBe(null);
  });
  test('has a control opt-out and creates fresh keys for separate runs', async () => {
    for (const flag of ['0', 'false', 'unknown']) {
      const original = runtime();
      let called = false;
      const deps = capable({
        env: { ZINC_OPENCLAW_SESSION_REUSE: flag },
        fetchImpl: async () => {
          called = true;
          throw new Error('unexpected health');
        },
      });
      expect(await sessionRuntimeConfig(original, 'loaded', deps)).toBe(original);
      expect(called).toBe(false);
    }
    const deps = capable({ sessionId: undefined });
    const a = await sessionRuntimeConfig(runtime(), 'loaded', deps);
    const b = await sessionRuntimeConfig(runtime(), 'loaded', deps);
    const key = (value: any) =>
      value.agents.defaults.models['zinc/loaded'].params.extraBody.session_id;
    expect(key(a)).toMatch(/^openclaw-[0-9a-f-]{36}$/);
    expect(key(a)).not.toBe(key(b));
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

const preparationReport = () => ({
  ok: false,
  status: 'error',
  final: '',
  payloads: [],
  model: null,
  provider: null,
  error: {
    kind: 'exception',
    message: 'Model runtime changed during both preparation attempts; retry the request',
  },
});
const preparationStderr =
  '[agents/prepared-model-runtime] auth-triggered model runtime refresh failed';
const preparationError = (report = preparationReport()) =>
  Object.assign(new Error('client failed'), {
    diagnostics: {
      code: 1,
      signal: null,
      stdout: JSON.stringify(report),
      stderr: preparationStderr,
    },
  });

async function httpFixture(handler: Parameters<typeof createServer>[0]) {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (typeof address !== 'object' || address === null)
    throw new Error('HTTP fixture did not listen');
  return {
    baseUrl: `http://127.0.0.1:${address.port}/proxy/v1`,
    close: async () => {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

describe('independent local inference admission guard', () => {
  test('metadata discovery can precede one preparation retry', async () => {
    const paths: string[] = [];
    const upstream = await httpFixture((request, response) => {
      paths.push(request.url!);
      response.end('{"data":[]}');
    });
    try {
      const marker = path.join(await temp(), 'attempts');
      const f = await fakeClient(`import {existsSync,readFileSync,writeFileSync} from 'node:fs';
const args=process.argv.slice(2),config=JSON.parse(readFileSync(args[args.indexOf('--config')+1],'utf8'));
await (await fetch(config.models.providers.zinc.baseUrl+'/models')).text();
if(!existsSync(${JSON.stringify(marker)})){writeFileSync(${JSON.stringify(marker)},'1');console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);}
console.log(JSON.stringify({ok:true,status:'ok'}));`);
      await writeConfig(f.project, config({ model: 'loaded', baseUrl: upstream.baseUrl }));
      const result = await runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
      });
      expect(result.preparationFailures).toHaveLength(1);
      expect(paths).toEqual(['/proxy/v1/models', '/proxy/v1/models']);
      expect((await readdir(f.project)).some((name) => name.endsWith('.run'))).toBe(false);
    } finally {
      await upstream.close();
    }
  });

  test('an inference POST prevents retry even when the client report omits all activity', async () => {
    const requests: string[] = [];
    const upstream = await httpFixture((request, response) => {
      requests.push(`${request.method} ${request.url}`);
      request.resume();
      response.end('{"choices":[]}');
    });
    try {
      const f = await fakeClient(`import {readFileSync} from 'node:fs';
const args=process.argv.slice(2),config=JSON.parse(readFileSync(args[args.indexOf('--config')+1],'utf8'));
await (await fetch(config.models.providers.zinc.baseUrl+'/chat/completions',{method:'POST',body:'{"stream":true}'})).text();
console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);`);
      await writeConfig(f.project, config({ model: 'loaded', baseUrl: upstream.baseUrl }));
      let starts = 0;
      await expect(
        runOpenClaw(f.options, {
          ...f.dependencies,
          fetchImpl: async () => Response.json({ status: 'ok' }),
          spawnImpl: (binary, args, options) => {
            starts++;
            return f.dependencies.spawnImpl(binary, args, options);
          },
        }),
      ).rejects.toThrow('OpenClaw failed');
      expect(starts).toBe(1);
      expect(requests).toEqual(['POST /proxy/v1/chat/completions']);
    } finally {
      await upstream.close();
    }
  });

  test('forwarding preserves Unicode SSE bytes, response status, headers and authorization', async () => {
    const sse = 'data: {"choices":[{"text":"café 東京"}]}\n\ndata: [DONE]\n\n';
    const received: any[] = [];
    const upstream = await httpFixture((request, response) => {
      const parts: Buffer[] = [];
      request.on('data', (part) => parts.push(part));
      request.on('end', () => {
        received.push({
          url: request.url,
          auth: request.headers.authorization,
          body: Buffer.concat(parts).toString(),
        });
        response.writeHead(201, { 'content-type': 'text/event-stream', 'x-fixture': 'unchanged' });
        const bytes = Buffer.from(sse),
          split = bytes.indexOf(Buffer.from('é')) + 1;
        response.write(bytes.subarray(0, split));
        setTimeout(() => response.end(bytes.subarray(split)), 10);
      });
    });
    try {
      const f = await fakeClient(`import {readFileSync} from 'node:fs';
const args=process.argv.slice(2),config=JSON.parse(readFileSync(args[args.indexOf('--config')+1],'utf8'));
const response=await fetch(config.models.providers.zinc.baseUrl+'/completions?fixture=1',{method:'POST',headers:{authorization:'Bearer fixture-key','content-type':'application/json'},body:'{"prompt":"東京","stream":true}'});
console.log(JSON.stringify({ok:true,status:'ok',http:response.status,header:response.headers.get('x-fixture'),type:response.headers.get('content-type'),text:await response.text()}));`);
      await writeConfig(f.project, config({ model: 'loaded', baseUrl: upstream.baseUrl }));
      const result = await runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
      });
      expect(result.report).toEqual({
        ok: true,
        status: 'ok',
        http: 201,
        header: 'unchanged',
        type: 'text/event-stream',
        text: sse,
      });
      expect(received).toEqual([
        {
          url: '/proxy/v1/completions?fixture=1',
          auth: 'Bearer fixture-key',
          body: '{"prompt":"東京","stream":true}',
        },
      ]);
      expect(
        JSON.parse(await readFile(f.options.filename, 'utf8')).models.providers.zinc.baseUrl,
      ).toBe(upstream.baseUrl);
    } finally {
      await upstream.close();
    }
  });

  test('client disconnect closes the owned upstream stream', async () => {
    let disconnected = false;
    const upstream = await httpFixture((request, response) => {
      request.resume();
      response.on('close', () => {
        disconnected = true;
      });
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write('data: first\n\n');
    });
    try {
      const f = await fakeClient(`import {readFileSync} from 'node:fs';
const args=process.argv.slice(2),config=JSON.parse(readFileSync(args[args.indexOf('--config')+1],'utf8'));
const response=await fetch(config.models.providers.zinc.baseUrl+'/completions',{method:'POST',body:'{}'});
const reader=response.body.getReader();await reader.read();await reader.cancel();console.log(JSON.stringify({ok:true,status:'ok'}));`);
      await writeConfig(f.project, config({ model: 'loaded', baseUrl: upstream.baseUrl }));
      await runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
      });
      for (let i = 0; i < 100 && !disconnected; i++)
        await new Promise((resolve) => setTimeout(resolve, 5));
      expect(disconnected).toBe(true);
    } finally {
      await upstream.close();
    }
  });

  test('a connection without an independent local guard cannot retry', async () => {
    const f = await fakeClient(
      `console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);`,
    );
    await writeConfig(
      f.project,
      config({ model: 'loaded', baseUrl: 'https://example.invalid/v1' }),
    );
    let starts = 0;
    await expect(
      runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
        spawnImpl: (binary, args, options) => {
          starts++;
          return f.dependencies.spawnImpl(binary, args, options);
        },
      }),
    ).rejects.toThrow('OpenClaw failed');
    expect(starts).toBe(1);
  });

  test('file changes and projects that cannot be fully checked prevent preparation retry', async () => {
    for (const mode of ['write', 'delete', 'symlink', 'oversize', 'entries']) {
      const f =
        await fakeClient(`${mode === 'write' ? "import {writeFileSync} from 'node:fs';writeFileSync('effect.txt','changed');" : mode === 'delete' ? "import {unlinkSync} from 'node:fs';unlinkSync('protected.txt');" : ''}
console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);`);
      await writeConfig(f.project, config({ model: 'loaded' }));
      if (mode === 'delete') await writeFile(path.join(f.project, 'protected.txt'), 'original');
      if (mode === 'symlink')
        await symlink(path.join(await temp(), 'absent'), path.join(f.project, 'linked'));
      if (mode === 'oversize')
        await writeFile(path.join(f.project, 'large'), Buffer.alloc(16 * 1024 * 1024 + 1));
      if (mode === 'entries')
        await Promise.all(
          Array.from({ length: 4096 }, (_, i) => writeFile(path.join(f.project, String(i)), '')),
        );
      let starts = 0;
      await expect(
        runOpenClaw(f.options, {
          ...f.dependencies,
          fetchImpl: async () => Response.json({ status: 'ok' }),
          spawnImpl: (binary, args, options) => {
            starts++;
            return f.dependencies.spawnImpl(binary, args, options);
          },
        }),
      ).rejects.toThrow('OpenClaw failed');
      expect(starts).toBe(1);
      if (mode === 'write')
        expect(await readFile(path.join(f.project, 'effect.txt'), 'utf8')).toBe('changed');
    }
  });
});

test('only empty pre-admission runtime failures qualify for a preparation retry', () => {
  expect(isRuntimePreparationFailure(preparationError())).toBe(true);
  const nested = preparationReport();
  nested.error.message +=
    ' | prepared model runtime publication was superseded for <runtime-owner>';
  expect(isRuntimePreparationFailure(preparationError(nested))).toBe(true);
  for (const changes of [
    { ok: true },
    { status: 'ok' },
    { final: 'partial answer' },
    { model: 'loaded' },
    { provider: 'zinc' },
    { payloads: [{ text: 'ran a task' }] },
    { usage: {} },
    { toolSummary: { calls: 0 } },
    { error: null },
    { error: { kind: 'exception', message: 'an unrelated task failure' } },
  ])
    expect(
      isRuntimePreparationFailure(preparationError({ ...preparationReport(), ...changes })),
    ).toBe(false);
  for (const changes of [
    { code: 0 },
    { code: 7 },
    { signal: 'SIGTERM' },
    { stderr: '' },
    { stdout: 'invalid' },
  ]) {
    const error = preparationError();
    Object.assign(error.diagnostics, changes);
    expect(isRuntimePreparationFailure(error)).toBe(false);
  }
  for (const code of ['OPENCLAW_TIMEOUT', 'OPENCLAW_INTERRUPTED', 'OPENCLAW_OUTPUT_LIMIT'])
    expect(isRuntimePreparationFailure(Object.assign(preparationError(), { code }))).toBe(false);
  expect(isRuntimePreparationFailure(null)).toBe(false);
});

describe('OpenClaw owned process and task result', () => {
  test('retries one preparation failure with fresh owned state and preserves the native report', async () => {
    const marker = path.join(await temp(), 'attempts');
    const f = await fakeClient(`import {existsSync,readFileSync,writeFileSync} from 'node:fs';
let input='';for await(const text of process.stdin)input+=text;
const count=existsSync(${JSON.stringify(marker)})?Number(readFileSync(${JSON.stringify(marker)},'utf8')):0;writeFileSync(${JSON.stringify(marker)},String(count+1));
if(count===0){console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);}
console.log(JSON.stringify({ok:true,status:'ok',final:'completed',input}));`);
    await writeConfig(f.project, config({ model: 'loaded' }));
    const states: string[] = [];
    const result = await runOpenClaw(f.options, {
      ...f.dependencies,
      fetchImpl: async () => Response.json({ status: 'ok' }),
      spawnImpl: (binary, args, options) => {
        states.push(options.env.OPENCLAW_STATE_DIR);
        return f.dependencies.spawnImpl(binary, args, options);
      },
    });
    expect(result.report).toEqual({
      ok: true,
      status: 'ok',
      final: 'completed',
      input: f.options.prompt,
    });
    expect(result.preparationFailures).toHaveLength(1);
    expect(result.preparationFailures[0].stdout).toContain('both preparation attempts');
    expect(states).toHaveLength(2);
    expect(states[0]).not.toBe(states[1]);
    for (const state of states) await expect(stat(state)).rejects.toThrow();
  });
  test('a repeated preparation failure stops after two processes and retains both diagnostics', async () => {
    const f = await fakeClient(
      `console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);`,
    );
    await writeConfig(f.project, config({ model: 'loaded' }));
    let starts = 0;
    const listeners = process.listenerCount('SIGINT');
    let failure: any;
    try {
      await runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
        spawnImpl: (binary, args, options) => {
          starts++;
          return f.dependencies.spawnImpl(binary, args, options);
        },
      });
    } catch (error) {
      failure = error;
    }
    expect(starts).toBe(2);
    expect(failure.preparationFailures).toHaveLength(1);
    expect(failure.diagnostics.stdout).toContain('both preparation attempts');
    expect(process.listenerCount('SIGINT')).toBe(listeners);
  });
  test('a preparation retry keeps the original outer deadline', async () => {
    const marker = path.join(await temp(), 'attempts');
    const f = await fakeClient(`import {existsSync,writeFileSync} from 'node:fs';
if(!existsSync(${JSON.stringify(marker)})){writeFileSync(${JSON.stringify(marker)},'1');await new Promise(r=>setTimeout(r,250));console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);}
await new Promise(r=>setTimeout(r,450));console.log(JSON.stringify({ok:true,status:'ok'}));`);
    await writeConfig(f.project, config({ model: 'loaded' }));
    let starts = 0;
    let failure: any;
    try {
      await runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () => Response.json({ status: 'ok' }),
        deadlineMs: 600,
        graceMs: 100,
        spawnImpl: (binary, args, options) => {
          starts++;
          return f.dependencies.spawnImpl(binary, args, options);
        },
      });
    } catch (error) {
      failure = error;
    }
    expect(starts).toBe(2);
    expect(failure?.code).toBe('OPENCLAW_TIMEOUT');
    expect(failure?.preparationFailures).toHaveLength(1);
  });
  test('an admitted failure cannot replay a task that may have changed files', async () => {
    const report = { ...preparationReport(), model: 'loaded', provider: 'zinc' };
    const f = await fakeClient(
      `import {writeFileSync} from 'node:fs';writeFileSync('effect.txt','one task');console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(report))});process.exit(1);`,
    );
    let starts = 0;
    await expect(
      runOpenClaw(f.options, {
        ...f.dependencies,
        spawnImpl: (binary, args, options) => {
          starts++;
          return f.dependencies.spawnImpl(binary, args, options);
        },
      }),
    ).rejects.toThrow('OpenClaw failed');
    expect(starts).toBe(1);
    expect(await readFile(path.join(f.project, 'effect.txt'), 'utf8')).toBe('one task');
  });
  test('interruption while preparation state is retiring cannot launch another attempt', async () => {
    const f = await fakeClient(
      `import {writeFileSync} from 'node:fs';import path from 'node:path';for(let i=0;i<120;i++)writeFileSync(path.join(process.env.OPENCLAW_STATE_DIR,String(i)),'state');console.error(${JSON.stringify(preparationStderr)});console.log(${JSON.stringify(JSON.stringify(preparationReport()))});process.exit(1);`,
    );
    let starts = 0;
    let failure: any;
    try {
      await runOpenClaw(f.options, {
        ...f.dependencies,
        spawnImpl: (binary, args, options) => {
          starts++;
          const child = f.dependencies.spawnImpl(binary, args, options);
          child.once('close', () => setImmediate(() => process.emit('SIGINT')));
          return child;
        },
      });
    } catch (error) {
      failure = error;
    }
    expect(starts).toBe(1);
    expect(failure?.code).toBe('OPENCLAW_INTERRUPTED');
  });
  test('uses a private temporary sibling config and removes it without changing the saved file', async () => {
    const f = await fakeClient(
      `import {readFileSync,statSync} from 'node:fs';let input='';for await(const s of process.stdin)input+=s;const args=process.argv.slice(2),filename=args[args.indexOf('--config')+1];console.log(JSON.stringify({ok:true,status:'ok',filename,config:JSON.parse(readFileSync(filename,'utf8')),mode:statSync(filename).mode&511}));`,
    );
    const original = config({ model: 'loaded', apiKeyEnv: 'TEST_ZINC_KEY' });
    await writeConfig(f.project, original);
    const result = await runOpenClaw(f.options, {
      ...f.dependencies,
      env: { ...process.env, TEST_ZINC_KEY: 'private-test-key' },
      fetchImpl: async () =>
        Response.json({
          status: 'ok',
          capabilities: { chat_session_reuse: true },
        }),
    });
    expect(path.dirname(result.report.filename)).toBe(f.project);
    expect(result.report.filename).not.toBe(f.options.filename);
    expect(result.report.mode).toBe(0o600);
    expect(result.report.config.models.providers.zinc.apiKey).toBe('${TEST_ZINC_KEY}');
    expect(
      result.report.config.agents.defaults.models['zinc/loaded'].params.extraBody.session_id,
    ).toMatch(/^openclaw-/);
    await expect(stat(result.report.filename)).rejects.toThrow();
    expect(JSON.parse(await readFile(f.options.filename, 'utf8'))).toEqual(original);
    expect((await readdir(f.project)).sort()).toEqual([CONFIG_NAME, 'client.mjs'].sort());
  });
  test('removes per-run configs after a client failure, timeout or interruption', async () => {
    for (const failure of ['exit', 'timeout', 'interrupt']) {
      const f = await fakeClient(
        failure === 'exit'
          ? `process.exit(7)`
          : `process.stdout.write('READY');setInterval(()=>{},1000);`,
      );
      await writeConfig(f.project, config({ model: 'loaded' }));
      const spawnImpl = f.dependencies.spawnImpl;
      const deps = {
        ...f.dependencies,
        deadlineMs: 300,
        graceMs: 100,
        fetchImpl: async () =>
          Response.json({
            status: 'ok',
            capabilities: { chat_session_reuse: true },
          }),
        spawnImpl: (binary: string, args: string[], options: any) => {
          const child = spawnImpl(binary, args, options);
          if (failure === 'interrupt') child.stdout.once('data', () => process.emit('SIGINT'));
          return child;
        },
      };
      await expect(runOpenClaw(f.options, deps)).rejects.toThrow();
      expect((await readdir(f.project)).sort()).toEqual([CONFIG_NAME, 'client.mjs'].sort());
    }
  });
  test('accepts an already-removed run config without changing the client outcome', async () => {
    const f = await fakeClient(`console.log(JSON.stringify({ok:true,status:'ok'}));`);
    const original = config({ model: 'loaded' });
    await writeConfig(f.project, original);
    const result = await runOpenClaw(f.options, {
      ...f.dependencies,
      fetchImpl: async () =>
        Response.json({
          status: 'ok',
          capabilities: { chat_session_reuse: true },
        }),
      unlinkImpl: async (filename: string) => {
        await rm(filename);
        throw Object.assign(new Error('already removed'), { code: 'ENOENT' });
      },
    });
    expect(result.report.ok).toBe(true);
    expect(JSON.parse(await readFile(f.options.filename, 'utf8'))).toEqual(original);
    expect((await readdir(f.project)).filter((name) => name.endsWith('.run'))).toEqual([]);
  });
  test('reports cleanup failure after client success without exposing filesystem diagnostics', async () => {
    const f = await fakeClient(`console.log(JSON.stringify({ok:true,status:'ok'}));`);
    await writeConfig(f.project, config({ model: 'loaded' }));
    await expect(
      runOpenClaw(f.options, {
        ...f.dependencies,
        fetchImpl: async () =>
          Response.json({
            status: 'ok',
            capabilities: { chat_session_reuse: true },
          }),
        unlinkImpl: async () => {
          throw Object.assign(new Error('EPERM private-project-diagnostic'), {
            code: 'EPERM',
          });
        },
      }),
    ).rejects.toThrow(/^OpenClaw temporary run config cleanup failed$/);
    expect((await readdir(f.project)).filter((name) => name.endsWith('.run'))).toHaveLength(1);
  });
  test('preserves client failure and its cause when temporary-config cleanup also fails', async () => {
    for (const source of [
      'process.exit(7)',
      `console.log(JSON.stringify({ok:false,status:'error'}));`,
      'setInterval(()=>{},1000);',
    ]) {
      const f = await fakeClient(source);
      await writeConfig(f.project, config({ model: 'loaded' }));
      let caught: any;
      try {
        await runOpenClaw(f.options, {
          ...f.dependencies,
          deadlineMs: 300,
          graceMs: 100,
          fetchImpl: async () =>
            Response.json({
              status: 'ok',
              capabilities: { chat_session_reuse: true },
            }),
          unlinkImpl: async () => {
            throw Object.assign(new Error('EBUSY private-project-diagnostic'), {
              code: 'EBUSY',
            });
          },
        });
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(AggregateError);
      expect(caught.errors).toHaveLength(2);
      expect(caught.cause).toBe(caught.errors[0]);
      expect(caught.diagnostics).toBe(caught.cause.diagnostics);
      expect(caught.code).toBe(caught.cause.code);
      expect(caught.message).toContain(caught.cause.message);
      expect(caught.message).toContain('temporary run config cleanup failed');
      expect(caught.message).not.toContain('private-project-diagnostic');
      if (source === 'process.exit(7)') expect(caught.cause.diagnostics.code).toBe(7);
      else if (source.startsWith('setInterval')) expect(caught.code).toBe('OPENCLAW_TIMEOUT');
      else expect(caught.cause.message).toContain('did not report a successful completed task');
    }
  });
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
      runOpenClaw(f.options, {
        ...f.dependencies,
        deadlineMs: 200,
        graceMs: 100,
      }),
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
    // Real Node startup can exceed Bun's default five-second test limit on
    // a loaded CI worker. Keep a shorter owned-process deadline than the test.
    const result = await runOpenClaw(
      {
        binary: path.relative(process.cwd(), binary),
        project,
        filename: 'config',
        model: 'loaded',
        prompt: 'task',
      },
      { deadlineMs: 10000, graceMs: 100 },
    );
    expect(result.report.ok).toBe(true);
  }, 15000);
  test('runs the client schema validator with the dedicated config path', async () => {
    const f = await fakeClient(
      `console.log(JSON.stringify({valid:true,args:process.argv.slice(2),config:process.env.OPENCLAW_CONFIG_PATH}));`,
    );
    const { report } = await validateClient(f.options, f.dependencies);
    expect(report.args).toEqual(['config', 'validate', '--json']);
    expect(report.config).toBe(f.options.filename);
  });
});
