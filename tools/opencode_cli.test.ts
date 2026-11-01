import { afterEach, expect, test } from 'bun:test';
import { chmod, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import {
  buildConfig,
  openCodeRunFlags,
  parseOpenCodeVersion,
  configForOpenCode,
  contextPluginPaths,
  contextPluginV2,
  runOpenCode,
  runtimeConfig,
  writeConfig,
} from './setup_opencode.mjs';

const dirs: string[] = [];
afterEach(async () => {
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true });
});

test('parses the actual legacy and OpenCode 2 version formats strictly', () => {
  for (const text of ['1.19.4', 'v1.19.4\n']) expect(parseOpenCodeVersion(text)).toBe('1.19.4');
  expect(parseOpenCodeVersion('opencode v2.0.21\n')).toBe('2.0.21');
  for (const text of ['2', 'garbage 2.0.21', 'opencode v2.0.21 extra'])
    expect(() => parseOpenCodeVersion(text)).toThrow('invalid version');
});

test('the downloaded helper stays a single portable file', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'zinc-opencode-download-'));
  dirs.push(dir);
  const helper = path.join(dir, 'setup-opencode.mjs');
  await writeFile(helper, await readFile(new URL('./setup_opencode.mjs', import.meta.url)));
  const child = Bun.spawn([process.execPath, helper, '--dry-run'], {
    cwd: dir,
    env: { ...process.env, OPENCODE_BIN: path.join(dir, 'missing-cli') },
    stdout: 'pipe',
    stderr: 'pipe',
  });
  const output = await new Response(child.stdout).text();
  expect(await child.exited).toBe(0);
  expect(JSON.parse(output)).toEqual(buildConfig());
});

test('client conversion preserves unrelated plugin packages and options', () => {
  const config: any = buildConfig();
  config.plugin.push(['unrelated-plugin', { enabled: true }]);
  const modern = configForOpenCode(config, ['--standalone']);
  expect(modern.plugins[0]).toEqual({ package: 'unrelated-plugin', options: { enabled: true } });
  const legacy = configForOpenCode(modern, []);
  expect(legacy.plugin[0]).toEqual(['unrelated-plugin', { enabled: true }]);
  expect(config.plugin[1]).toEqual(['unrelated-plugin', { enabled: true }]);
});

test('OpenCode 2 hooks bound actual event inputs and combined text output', async () => {
  const hooks: Record<string, Function> = {};
  await contextPluginV2({
    tool: {
      async hook(name: string, callback: Function) {
        hooks[name] = callback;
      },
    },
  });
  const read = { tool: 'read', input: { path: 'large.txt', limit: 2000 } };
  await hooks['execute.before'](read);
  expect(read.input.limit).toBe(80);
  const text = 'HEAD' + 'x'.repeat(8000) + 'TAIL';
  const result: any = {
    tool: 'shell',
    status: 'completed',
    result: {
      content: [
        { type: 'text', text: text.slice(0, 4500) },
        { type: 'image', data: 'fixture' },
        { type: 'text', text: text.slice(4500) },
      ],
      metadata: { existing: 'kept' },
    },
  };
  await hooks['execute.after'](result);
  expect(result.result.content.filter((item: any) => item.type === 'text')).toHaveLength(1);
  const bounded = result.result.content[0].text;
  expect(bounded.length).toBeLessThan(6000);
  expect(bounded.startsWith('HEAD')).toBe(true);
  expect(bounded.endsWith('TAIL')).toBe(true);
  expect(bounded).toContain('ZINC context limit');
  expect(result.result.content[1]).toEqual({ type: 'image', data: 'fixture' });
  expect(result.result.metadata).toMatchObject({ existing: 'kept', zinc_context_truncated: true });
  const failed = { status: 'error', error: { message: 'fixture failure' } };
  await hooks['execute.after'](failed);
  expect(failed.error.message).toBe('fixture failure');
});

test('writes a standalone OpenCode 2 plugin package and refuses changed files or symlinks', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'zinc-opencode-plugin2-'));
  dirs.push(dir);
  const config = configForOpenCode(buildConfig(), ['--standalone']);
  expect(config.plugin).toBeUndefined();
  expect(configForOpenCode(config, []).plugins).toBeUndefined();
  const paths = contextPluginPaths(config);
  const live = runtimeConfig(config, {}, { project: dir });
  expect(live.plugins[0]).toBe(path.dirname(path.join(dir, paths[0])));
  expect(config.plugins[0].startsWith('./')).toBe(true);
  expect(paths).toHaveLength(2);
  await writeConfig(dir, config);
  expect(JSON.parse(await readFile(path.join(dir, paths[1]), 'utf8')).type).toBe('module');
  const source = await readFile(path.join(dir, paths[0]), 'utf8');
  expect(source).not.toContain(dir);
  await writeFile(path.join(dir, paths[0]), 'changed');
  await expect(writeConfig(dir, config, { force: true })).rejects.toThrow('conflicts');
  expect(await readFile(path.join(dir, paths[0]), 'utf8')).toBe('changed');
  const alias = await mkdtemp(path.join(os.tmpdir(), 'zinc-opencode-alias-'));
  dirs.push(alias);
  await import('node:fs/promises').then(({ mkdir }) => mkdir(path.join(alias, '.zinc')));
  await symlink(path.dirname(path.join(dir, paths[0])), path.dirname(path.join(alias, paths[0])));
  await expect(writeConfig(alias, config)).rejects.toThrow('must be a directory');
  expect(await readFile(path.join(dir, paths[0]), 'utf8')).toBe('changed');
});

for (const duringTimeout of [false, true]) {
  test(`transport closure ${duringTimeout ? 'caused by our timeout keeps 124' : 'during work remains an error'}`, async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'zinc-opencode-transport-'));
    dirs.push(dir);
    const details: any[] = [];
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['task'], timeoutMs: 10 },
      {
        env: {},
        runFlags: [],
        stdout: { write() {} },
        onEvent(_event: any, metadata: any) {
          details.push(metadata);
        },
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          let stopped = false;
          const emitError = () =>
            child.stdout.emit(
              'data',
              Buffer.from(
                JSON.stringify({
                  type: 'error',
                  error: {
                    type: 'unknown',
                    message:
                      'Transport: The socket connection was closed unexpectedly. For more information, pass verbose.',
                  },
                }) + '\n',
              ),
            );
          child.kill = () => {
            if (stopped) return;
            stopped = true;
            if (duringTimeout) emitError();
            queueMicrotask(() => child.emit('close', 1, null));
          };
          if (!duringTimeout) queueMicrotask(emitError);
          return child;
        }) as any,
      },
    );
    expect(code).toBe(duringTimeout ? 124 : 1);
    expect(details.some((item) => item.expectedTimeoutShutdown)).toBe(duringTimeout);
  });
}

test('discovers a private OpenCode 2 run and bounds the help probe', () => {
  let captured: any;
  const env = { OPENCODE_BIN: 'custom-opencode' };
  expect(
    openCodeRunFlags({
      env,
      spawnSyncImpl: ((...args: any[]) => {
        captured = args;
        return { status: 0, stdout: '--model --format\n  --standalone  Private server' };
      }) as any,
    }),
  ).toEqual(['--standalone']);
  expect(captured[0]).toBe('custom-opencode');
  expect(captured[1]).toEqual(['run', '--help']);
  expect(captured[2]).toMatchObject({
    timeout: 10000,
    maxBuffer: 65536,
    killSignal: 'SIGKILL',
    env,
  });
  expect(captured[2].shell).toBeUndefined();
});

test('keeps OpenCode 1 compatible and ignores lookalike flags', () => {
  for (const help of ['--model --format --dir', '--model --format not--standalone'])
    expect(
      openCodeRunFlags({ spawnSyncImpl: (() => ({ status: 0, stderr: help })) as any }),
    ).toEqual([]);
});

test('setup can preview the pinned legacy config before OpenCode is installed', () => {
  const missing = { status: null, error: Object.assign(new Error('missing'), { code: 'ENOENT' }) };
  expect(openCodeRunFlags({ allowMissing: true, spawnSyncImpl: (() => missing) as any })).toEqual(
    [],
  );
  expect(() => openCodeRunFlags({ spawnSyncImpl: (() => missing) as any })).toThrow(
    'run help failed',
  );
});

test('refuses failed, missing, timed-out or oversized help without echoing output', () => {
  for (const result of [
    { status: 1, stdout: '--model --format --standalone private-test-output' },
    { status: 0, stdout: 'private-test-output' },
    { status: null, error: new Error('ETIMEDOUT') },
    { status: null, error: new Error('ENOBUFS') },
  ]) {
    expect(() => openCodeRunFlags({ spawnSyncImpl: (() => result) as any })).toThrow(
      'OpenCode run help failed',
    );
  }
});

for (const standalone of [false, true]) {
  test(`launches the ${standalone ? '2' : '1'} CLI in the project with literal arguments`, async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'zinc-opencode-cli-'));
    dirs.push(dir);
    const binary = path.join(dir, 'fake-opencode');
    const captured = path.join(dir, 'captured.json');
    await writeFile(
      binary,
      `#!/usr/bin/env bun
import { writeFileSync } from 'node:fs';
if (process.argv.includes('--help')) {
  console.log('--model --format ${standalone ? '--standalone' : '--dir'}');
} else {
  writeFileSync(process.env.CLI_CAPTURE, JSON.stringify({ args: process.argv.slice(2), cwd: process.cwd() }));
  console.log(JSON.stringify({ type: 'text', part: { text: 'OK' } }));
}
`,
    );
    await chmod(binary, 0o755);
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['Fix $(literal) `text`'], auto: true },
      {
        env: { ...process.env, OPENCODE_BIN: binary, CLI_CAPTURE: captured },
        stdout: { write() {} },
        stderr: { write() {} },
      },
    );
    expect(code).toBe(0);
    const result = JSON.parse(await readFile(captured, 'utf8'));
    const expected = await stat(dir),
      actual = await stat(result.cwd);
    expect([actual.dev, actual.ino]).toEqual([expected.dev, expected.ino]);
    expect(result.args).not.toContain('--dir');
    expect(result.args.includes('--standalone')).toBe(standalone);
    expect(result.args.slice(-2)).toEqual(['--', 'Fix $(literal) `text`']);
    const hook = contextPluginPaths(
      configForOpenCode(buildConfig(), standalone ? ['--standalone'] : []),
    )[0];
    expect((await stat(path.join(dir, hook))).isFile()).toBe(true);
  });
}
