import { afterEach, expect, test } from 'bun:test';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { buildConfig, CONFIG_NAME, resolveAutoContext } from './setup_opencode.mjs';

const exec = promisify(execFile);
const helper = fileURLToPath(new URL('./setup_opencode.mjs', import.meta.url));
const directories: string[] = [];
const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await server.stop(true);
  for (const dir of directories.splice(0)) await rm(dir, { recursive: true, force: true });
});
async function temp() {
  const dir = await mkdtemp(path.join(tmpdir(), 'zinc-auto-context-'));
  directories.push(dir);
  return dir;
}
function endpoint(fetch: (request: Request) => Response | Promise<Response>) {
  const server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch });
  servers.push(server);
  return `${server.url.origin}/v1`;
}
function discovery(data: unknown) {
  return { env: {}, fetchImpl: async () => Response.json(data) };
}
async function cli(args: string[], env: Record<string, string> = {}) {
  try {
    return {
      code: 0,
      ...(await exec(process.execPath, [helper, ...args], {
        env: { ...process.env, ZINC_API_BASE_URL: '', ZINC_API_KEY: '', ...env },
        timeout: 15000,
      })),
    };
  } catch (error: any) {
    return { code: error.code, stdout: error.stdout || '', stderr: error.stderr || '' };
  }
}

test('auto context uses one authenticated GET and keeps configuration portable', async () => {
  const requests: { method: string; path: string; authorization: string | null }[] = [];
  const base = endpoint((request) => {
    requests.push({
      method: request.method,
      path: new URL(request.url).pathname,
      authorization: request.headers.get('authorization'),
    });
    return Response.json({ data: [{ id: 'loaded.gguf', context_length: 262144 }] });
  });
  const root = await temp();
  const project = path.join(root, 'new project ✓');
  const result = await cli(
    [
      '--project',
      project,
      '--context',
      'auto',
      '--model',
      'client:alias',
      '--output',
      '10000',
      '--steps',
      '7',
    ],
    { ZINC_API_BASE_URL: `${base}/`, ZINC_API_KEY: 'fixture-key-do-not-save' },
  );
  expect(result.code).toBe(0);
  expect(requests).toEqual([
    { method: 'GET', path: '/v1/models', authorization: 'Bearer fixture-key-do-not-save' },
  ]);
  const text = await readFile(path.join(project, CONFIG_NAME), 'utf8');
  expect(JSON.parse(text)).toEqual(
    buildConfig({ model: 'client:alias', context: 262144, output: 10000, steps: 7 }),
  );
  expect(text).not.toContain(base);
  expect(text).not.toContain('fixture-key-do-not-save');
  expect((await readdir(path.join(project, '.zinc'))).length).toBe(1);
});

test('auto preview makes no writes and matches the synchronous numeric builder', async () => {
  const base = endpoint(() => Response.json({ data: [{ id: 'loaded', context_length: 8192 }] }));
  const root = await temp();
  const project = path.join(root, 'missing', 'project');
  const result = await cli(['--project', project, '--context', 'auto', '--dry-run'], {
    ZINC_API_BASE_URL: base,
  });
  expect(result.code).toBe(0);
  expect(result.stdout).toBe(JSON.stringify(buildConfig({ context: 8192 }), null, 2) + '\n');
  expect(await readdir(root)).toEqual([]);
  expect(() => buildConfig({ context: 'auto' })).toThrow('Context must');
});

test('exact model ID wins over a suffixed ID in either order', async () => {
  const models = [
    { id: 'target.gguf', context_length: 8192 },
    { id: 'target', context_length: 262144 },
  ];
  for (const data of [models, [...models].reverse()])
    expect(await resolveAutoContext({ model: 'target' }, discovery({ data }))).toBe(262144);
});

test('a singleton alias works but a suffix cannot resolve an unmatched multi-model list', async () => {
  const model = { id: 'target.gguf', context_length: 32768 };
  expect(await resolveAutoContext({ model: 'target' }, discovery({ data: [model] }))).toBe(32768);
  await expect(
    resolveAutoContext(
      { model: 'target' },
      discovery({ data: [model, { id: 'other', context_length: 8192 }] }),
    ),
  ).rejects.toThrow('ambiguous');
});

test('malformed discovery and duplicate model IDs fail closed', async () => {
  for (const data of [
    null,
    [],
    {},
    { data: [] },
    { models: [{ id: 'target', context_length: 32768 }] },
    { data: [null] },
    { data: ['target'] },
    { data: [{ context_length: 32768 }] },
    { data: [{ id: '', context_length: 32768 }] },
    {
      data: [
        { id: 'target', context_length: 32768 },
        { id: 'target', context_length: 8192 },
      ],
    },
  ])
    await expect(resolveAutoContext({ model: 'target' }, discovery(data))).rejects.toThrow(
      'ZINC model discovery',
    );
});

test('discovered capacity must be a safe integer of at least 8192', async () => {
  for (const context_length of [
    undefined,
    null,
    '32768',
    8191,
    8192.5,
    -1,
    Number.MAX_SAFE_INTEGER + 1,
  ])
    await expect(
      resolveAutoContext({}, discovery({ data: [{ id: 'loaded', context_length }] })),
    ).rejects.toThrow('context length');
  for (const context_length of [8192, 32768, 262144, Number.MAX_SAFE_INTEGER])
    expect(
      await resolveAutoContext({}, discovery({ data: [{ id: 'loaded', context_length }] })),
    ).toBe(context_length);
});

test('invalid runtime endpoints fail before the lookup', async () => {
  let requests = 0;
  for (const base of [
    'not a URL',
    'file:///v1',
    'https://user:secret@example.invalid/v1',
    'https://example.invalid/v1?key=secret',
    'https://example.invalid/other',
  ])
    await expect(
      resolveAutoContext(
        {},
        {
          env: { ZINC_API_BASE_URL: base },
          fetchImpl: async () => {
            requests++;
            return Response.json({});
          },
        },
      ),
    ).rejects.toThrow('ZINC_API_BASE_URL');
  expect(requests).toBe(0);
});

test('lookup failures are sanitized before files are created', async () => {
  const root = await temp();
  for (const [name, response] of [
    ['http', () => new Response('fixture-secret-body', { status: 401 })],
    ['json', () => new Response('fixture-secret-body')],
    [
      'capacity',
      () => Response.json({ data: [{ id: 'loaded', context_length: 'fixture-secret-body' }] }),
    ],
    ['ids', () => Response.json({ data: [{ context_length: 32768 }] })],
  ] as const) {
    const base = endpoint(response);
    const result = await cli(['--project', path.join(root, name), '--context', 'auto'], {
      ZINC_API_BASE_URL: base,
      ZINC_API_KEY: 'fixture-secret-key',
    });
    expect(result.code).toBe(1);
    expect(result.stdout).toBe('');
    expect(result.stderr).toContain('ZINC model discovery');
    expect(result.stderr).not.toContain(base);
    expect(result.stderr).not.toContain('fixture-secret');
  }
  expect(await readdir(root)).toEqual([]);
  await expect(
    resolveAutoContext(
      {},
      {
        env: {},
        fetchImpl: async () => {
          throw new Error('https://example.invalid/private fixture-secret');
        },
      },
    ),
  ).rejects.toThrow('ZINC model discovery request failed or timed out');
});

test('numeric setup stays offline and auto is rejected before check/run project access', async () => {
  const root = await temp();
  const blocked = path.join(root, 'file');
  await writeFile(blocked, 'keep');
  const project = path.join(blocked, 'child');
  const numeric = await cli(['--project', project, '--context', '32768', '--dry-run'], {
    ZINC_API_BASE_URL: 'not a URL',
  });
  expect(numeric.code).toBe(0);
  expect(numeric.stdout).toBe(JSON.stringify(buildConfig({ context: 32768 }), null, 2) + '\n');
  for (const command of ['check', 'run']) {
    const result = await cli([command, '--project', project, '--context', 'auto'], {
      ZINC_API_BASE_URL: 'not a URL',
    });
    expect(result.code).toBe(1);
    expect(result.stderr).toContain('--context auto is setup-only');
  }
  expect(await readFile(blocked, 'utf8')).toBe('keep');
});

test('the ten-second lookup deadline also covers a stalled response body', async () => {
  const base = endpoint(
    () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('{"data":'));
          },
        }),
        { headers: { 'content-type': 'application/json' } },
      ),
  );
  const start = performance.now();
  await expect(resolveAutoContext({}, { env: { ZINC_API_BASE_URL: base } })).rejects.toThrow(
    'ZINC model discovery',
  );
  expect(performance.now() - start).toBeGreaterThanOrEqual(9000);
  expect(performance.now() - start).toBeLessThan(14000);
}, 15000);
