import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import {
  buildConfig,
  checkServer,
  CONFIG_NAME,
  CONTEXT_PLUGIN_PATH,
  contextPlugin,
  parseArgs,
  runOpenCode,
  runVerification,
  runVerifiedOpenCode,
  runtimeConfig,
  sessionRuntimeConfig,
  writeConfig,
} from './setup_opencode.mjs';

const dirs: string[] = [];
const noSessionCacheFetch = (async () => Response.json({ status: 'ok' })) as any;
let testProject: string;
beforeEach(async () => {
  testProject = await temp();
});
afterEach(async () => {
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true });
});
async function temp() {
  const dir = await mkdtemp(path.join(tmpdir(), 'zinc-opencode-setup-'));
  dirs.push(dir);
  return dir;
}

describe('portable OpenCode setup', () => {
  test('has only the local provider, bounded context, and no captured environment', () => {
    const config = buildConfig();
    expect(config.enabled_providers).toEqual(['zinc']);
    expect(config.small_model).toBe(config.model);
    expect(config.provider.zinc.options).toEqual({
      baseURL: 'http://127.0.0.1:9090/v1',
      apiKey: '{env:ZINC_API_KEY}',
      body: { temperature: 0, top_p: 1, enable_thinking: false },
    });
    expect(config.agent.zinc.steps).toBe(24);
    expect(config.provider.zinc.models['qwen38-27b-q4k-m'].limit.output).toBe(4096);
    expect(
      buildConfig({ context: 8192 }).provider.zinc.models['qwen38-27b-q4k-m'].limit.output,
    ).toBe(2048);
    expect(config.compaction.reserved).toBe(8192);
    expect(buildConfig({ context: 8192 }).compaction.reserved).toBe(4096);
    expect(config.compaction.reserved).toBeLessThan(
      config.provider.zinc.models['qwen38-27b-q4k-m'].limit.context,
    );
    expect(JSON.stringify(config)).not.toContain(process.cwd());
  });
  test('bounds real reader arguments without changing paths or requested offsets', async () => {
    const hooks = await contextPlugin();
    for (const [requested, expected] of [
      [undefined, 80],
      [2000, 80],
      [12, 12],
      [0, 0],
    ]) {
      const output = { args: { filePath: 'source.ts', offset: 200, limit: requested } };
      await hooks['tool.execute.before']({ tool: 'read' }, output);
      expect(output.args).toEqual({ filePath: 'source.ts', offset: 200, limit: expected });
    }
    const bash = { args: { command: 'echo unchanged' } };
    await hooks['tool.execute.before']({ tool: 'bash' }, bash);
    expect(bash.args.command).toBe('echo unchanged');
  });
  test('bounds long tool output and preserves Unicode, diagnostics and metadata', async () => {
    const hooks = await contextPlugin();
    const output = {
      output: 'x'.repeat(3999) + '🚀' + 'm'.repeat(9000) + '🚀' + 'z'.repeat(1499),
      metadata: { exit: 7 },
    };
    await hooks['tool.execute.after']({ tool: 'bash' }, output);
    expect(output.output.length).toBeLessThanOrEqual(6000);
    expect(output.output).toContain('ZINC context limit');
    expect(output.output.endsWith('z'.repeat(1499))).toBe(true);
    expect(output.output.isWellFormed()).toBe(true);
    expect(output.metadata).toEqual({ exit: 7, truncated: true, zinc_context_truncated: true });
    const short = { output: 'complete diagnostics', metadata: { exit: 1 } };
    await hooks['tool.execute.after']({ tool: 'bash' }, short);
    expect(short).toEqual({ output: 'complete diagnostics', metadata: { exit: 1 } });
  });
  test('installs a portable standalone hook and refuses unrelated replacements', async () => {
    const dir = await temp();
    const config = buildConfig();
    await writeConfig(dir, config);
    expect(config.plugin).toEqual([`./${CONTEXT_PLUGIN_PATH}`]);
    const file = path.join(dir, CONTEXT_PLUGIN_PATH);
    const source = await readFile(file, 'utf8');
    expect(source).not.toContain(dir);
    const module = await import(file);
    expect(typeof (await module.default())['tool.execute.before']).toBe('function');
    await writeConfig(dir, config, { force: true });
    expect(await readFile(file, 'utf8')).toBe(source);
    await writeFile(file, 'unrelated user file\n');
    const before = await readFile(path.join(dir, CONFIG_NAME), 'utf8');
    await expect(writeConfig(dir, config, { force: true })).rejects.toThrow('conflicts');
    expect(await readFile(file, 'utf8')).toBe('unrelated user file\n');
    expect(await readFile(path.join(dir, CONFIG_NAME), 'utf8')).toBe(before);
  });
  test('refuses hook directory and file symlinks without changing their targets', async () => {
    const dir = await temp();
    const elsewhere = await temp();
    await symlink(elsewhere, path.join(dir, '.zinc'));
    await expect(writeConfig(dir, buildConfig())).rejects.toThrow('symlink');
    expect(await readdir(elsewhere)).toEqual([]);
    await rm(path.join(dir, '.zinc'));
    await mkdir(path.join(dir, '.zinc'));
    const target = path.join(elsewhere, 'keep');
    await writeFile(target, 'preserve');
    await symlink(target, path.join(dir, CONTEXT_PLUGIN_PATH));
    await expect(writeConfig(dir, buildConfig())).rejects.toThrow();
    expect(await readFile(target, 'utf8')).toBe('preserve');
    expect(await readdir(dir)).toEqual(['.zinc']);
  });
  test('parses task text without interpreting it as options', () => {
    const opts = parseArgs([
      'run',
      '--project',
      'project with spaces',
      '--auto',
      '--',
      'explain --help',
    ]);
    expect(opts.prompt).toEqual(['explain --help']);
    expect(opts.auto).toBe(true);
    expect(() => parseArgs(['--misspelled'])).toThrow('Unknown option');
    expect(() => parseArgs(['--model'])).toThrow('Missing value');
  });
  test('rejects invalid model IDs and memory budgets', () => {
    for (const model of [
      '../model',
      '/model.gguf',
      'https://example.com/model',
      '{env:SECRET}',
      'a\nb',
    ])
      expect(() => buildConfig({ model })).toThrow();
    for (const context of [NaN, -1, 4096, 8192.5]) expect(() => buildConfig({ context })).toThrow();
    for (const output of [0, NaN, -1, 16384]) expect(() => buildConfig({ output })).toThrow();
  });
  test('writes a standalone config, preserves existing files, and refuses symlinks', async () => {
    const dir = await temp();
    await writeFile(path.join(dir, 'opencode.json'), '{"model":"existing/model"}\n');
    await writeConfig(dir, buildConfig());
    expect(await readFile(path.join(dir, 'opencode.json'), 'utf8')).toBe(
      '{"model":"existing/model"}\n',
    );
    await expect(writeConfig(dir, buildConfig())).rejects.toThrow();
    await rm(path.join(dir, CONFIG_NAME));
    const target = path.join(dir, 'do-not-overwrite');
    await writeFile(target, 'keep');
    await symlink(target, path.join(dir, CONFIG_NAME));
    await expect(writeConfig(dir, buildConfig(), { force: true })).rejects.toThrow();
    expect(await readFile(target, 'utf8')).toBe('keep');
  });
  test('runtime endpoints and keys never enter the portable config', async () => {
    const config = buildConfig();
    const env = {
      ZINC_API_BASE_URL: 'https://example.invalid/api/v1/',
      ZINC_API_KEY: 'test-only-key',
    };
    const live = runtimeConfig(config, env);
    expect(live.provider.zinc.options.baseURL).toBe('https://example.invalid/api/v1');
    expect(JSON.stringify(live)).not.toContain(env.ZINC_API_KEY);
    const dir = await temp();
    await writeConfig(dir, config);
    const saved = await readFile(path.join(dir, CONFIG_NAME), 'utf8');
    expect(saved).not.toContain('example.invalid');
    expect(saved).not.toContain(env.ZINC_API_KEY);
    expect(saved).not.toContain(dir);
  });
  test('rejects credential-bearing URLs and malformed endpoints', () => {
    for (const url of [
      'file:///v1',
      'https://user:password@example.invalid/v1',
      'http://localhost/v1?token=x',
      'http://localhost/v1#x',
      'http://localhost/chat',
      'http://localhost/not-v1',
      'not a URL',
    ]) {
      expect(() => runtimeConfig(buildConfig(), { ZINC_API_BASE_URL: url })).toThrow(
        'ZINC_API_BASE_URL',
      );
    }
  });
  test('session reuse is discovered strictly and stays out of saved configuration', async () => {
    const config = buildConfig();
    const before = structuredClone(config);
    const seen: any[] = [];
    let next = 0;
    const deps = {
      sessionId: () => String(++next),
      fetchImpl: (async (url: string, options: any) => {
        seen.push({ url, options });
        return Response.json({ status: 'ok', capabilities: { chat_session_reuse: true } });
      }) as any,
    };
    const env = { ZINC_API_BASE_URL: 'https://example.invalid/api/v1', ZINC_API_KEY: 'test-key' };
    const first = await sessionRuntimeConfig(config, env, deps);
    const second = await sessionRuntimeConfig(config, env, deps);
    expect(first.provider.zinc.options.body.session_id).toBe('opencode-1');
    expect(first.agent.zinc.options.session_id).toBe('opencode-1');
    expect(second.provider.zinc.options.body.session_id).toBe('opencode-2');
    expect(seen[0].url).toBe('https://example.invalid/api/health');
    expect(seen[0].options.headers.authorization).toBe('Bearer test-key');
    expect(seen[0].options.signal).toBeInstanceOf(AbortSignal);
    expect(JSON.stringify(first)).not.toContain('test-key');
    expect(config).toEqual(before);
    await writeConfig(testProject, config);
    expect(await readFile(path.join(testProject, CONFIG_NAME), 'utf8')).not.toContain('session_id');
  });
  test('missing false malformed unhealthy and unavailable discovery preserve uncached execution', async () => {
    const config = buildConfig();
    for (const fetchImpl of [
      async () => Response.json({ status: 'ok' }),
      async () => Response.json({ status: 'ok', capabilities: { chat_session_reuse: false } }),
      async () => Response.json({ status: 'ok', capabilities: { chat_session_reuse: 'true' } }),
      async () => Response.json({ status: 'failed', capabilities: { chat_session_reuse: true } }),
      async () => new Response('invalid json'),
      async () => new Response(null, { status: 503 }),
      async () => {
        throw new Error('private endpoint failure');
      },
    ]) {
      const live = await sessionRuntimeConfig(config, {}, { fetchImpl: fetchImpl as any });
      expect(live).toEqual(runtimeConfig(config, {}));
    }
  });
  test('session discovery normalizes trailing slashes while preserving proxy path prefixes', async () => {
    for (const [base, health] of [
      ['https://example.invalid/v1/', 'https://example.invalid/health'],
      ['https://example.invalid/proxy/zinc/v1', 'https://example.invalid/proxy/zinc/health'],
      ['https://example.invalid/proxy/zinc/v1/', 'https://example.invalid/proxy/zinc/health'],
    ]) {
      let requested;
      await sessionRuntimeConfig(
        buildConfig(),
        { ZINC_API_BASE_URL: base },
        {
          fetchImpl: (async (url) => {
            requested = url;
            return Response.json({ status: 'ok' });
          }) as any,
        },
      );
      expect(requested).toBe(health);
    }
  });
  test('explicit provider model and agent session keys including opt-outs win without discovery', async () => {
    for (const value of ['chosen', '', null]) {
      for (const location of ['provider', 'model', 'agent']) {
        const config: any = buildConfig();
        const provider = config.provider.zinc;
        if (location === 'provider') provider.options.body.session_id = value;
        if (location === 'model')
          provider.models['qwen38-27b-q4k-m'].options = { body: { session_id: value } };
        if (location === 'agent') config.agent.zinc.options.session_id = value;
        let calls = 0;
        const live = await sessionRuntimeConfig(
          config,
          {},
          {
            fetchImpl: (async () => {
              calls++;
              throw new Error('unexpected lookup');
            }) as any,
          },
        );
        expect(calls).toBe(0);
        expect(live).toEqual(runtimeConfig(config, {}));
      }
    }
  });
  test('headless execution forwards literal task text and the child exit status', async () => {
    const dir = await temp();
    let captured: any;
    const status = await runOpenCode(
      buildConfig(),
      { project: dir, auto: true, prompt: ['Fix $(literal) `text`'] },
      {
        env: { OPENCODE_BIN: 'custom-opencode', ZINC_API_KEY: 'test-only-key' },
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: ((bin: string, args: string[], options: any) => {
          captured = { bin, args, options };
          const child = new EventEmitter();
          queueMicrotask(() => child.emit('close', 7, null));
          return child;
        }) as any,
      },
    );
    expect(status).toBe(7);
    expect(captured.bin).toBe('custom-opencode');
    expect(captured.args.at(-1)).toBe('Fix $(literal) `text`');
    expect(captured.args).toContain('--auto');
    expect(captured.args).not.toContain('--pure');
    expect(captured.args).not.toContain('--dir');
    expect(captured.options.cwd).toBe(dir);
    expect(captured.options.env.OPENCODE_PURE).toBe('false');
    expect(captured.options.shell).toBeUndefined();
    expect(captured.options.env.OPENCODE_CONFIG_CONTENT).not.toContain('test-only-key');
  });
  test('launch gives one ephemeral key to the provider and legacy agent paths', async () => {
    const config = buildConfig();
    let captured: any;
    const code = await runOpenCode(
      config,
      { project: testProject, prompt: ['task'] },
      {
        env: {},
        runFlags: [],
        fetchImpl: (async () =>
          Response.json({ status: 'ok', capabilities: { chat_session_reuse: true } })) as any,
        spawnImpl: ((_bin, _args, options) => {
          captured = JSON.parse(options.env.OPENCODE_CONFIG_CONTENT);
          const child = new EventEmitter();
          queueMicrotask(() => child.emit('close', 0, null));
          return child;
        }) as any,
      },
    );
    expect(code).toBe(0);
    const key = captured.provider.zinc.options.body.session_id;
    expect(key).toMatch(/^opencode-[a-f0-9-]{36}$/);
    expect(captured.agent.zinc.options.session_id).toBe(key);
    expect(config.provider.zinc.options.body).not.toHaveProperty('session_id');
    expect(config.agent.zinc.options).not.toHaveProperty('session_id');
  });
  test('a timed-out agent returns 124 and receives a termination signal', async () => {
    const dir = await temp();
    const signals: string[] = [];
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['task'], timeoutMs: 20 },
      {
        env: {},
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.kill = (signal: string) => {
            signals.push(signal);
            queueMicrotask(() => child.emit('close', null, signal));
          };
          return child;
        }) as any,
      },
    );
    expect(code).toBe(124);
    expect(signals).toEqual(['SIGTERM']);
  });
  test('JSON error events fail a task even when OpenCode exits zero', async () => {
    const dir = await temp();
    let output = '';
    const code = await runOpenCode(
      buildConfig(),
      { project: dir, prompt: ['task'], session: 'ses_example' },
      {
        env: {},
        stdout: {
          write: (chunk) => {
            output += chunk.toString();
          },
        } as any,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: ((_bin, args) => {
          expect(args).toContain('ses_example');
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          queueMicrotask(() => {
            child.stdout.emit(
              'data',
              Buffer.from('{"type":"error","error":{"name":"APIError"}}\n'),
            );
            child.emit('close', 0, null);
          });
          return child;
        }) as any,
      },
    );
    expect(code).toBe(1);
    expect(output).toContain('APIError');
  });

  test('captures CLI diagnostics separately from the JSON event stream', async () => {
    let stdout = '',
      stderr = '';
    const code = await runOpenCode(
      buildConfig(),
      { project: testProject, prompt: ['task'] },
      {
        env: {},
        stdout: {
          write: (chunk) => {
            stdout += chunk.toString();
          },
        } as any,
        stderr: {
          write: (chunk) => {
            stderr += chunk.toString();
          },
        } as any,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.stderr = new EventEmitter();
          queueMicrotask(() => {
            child.stderr.emit('data', Buffer.from('CLI diagnostic\n'));
            child.stdout.emit('data', Buffer.from('{"type":"step_finish"}\n'));
            child.emit('close', 0, null);
          });
          return child;
        }) as any,
      },
    );
    expect(code).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ type: 'step_finish' });
    expect(stderr).toBe('CLI diagnostic\n');
  });
});

describe('independent verification', () => {
  const quiet = { write: () => {} } as any;

  test('rejects empty tasks before any model or verification process starts', async () => {
    for (const prompt of [undefined, [], [' ', '\n']]) {
      const deps = {
        runModel: async () => {
          throw new Error('model started');
        },
        runCheck: async () => {
          throw new Error('check started');
        },
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          throw new Error('process started');
        }) as any,
      };
      const opts = { project: testProject, prompt, verify: 'check' };
      await expect(runOpenCode(buildConfig(), opts, deps)).rejects.toThrow('Pass a task');
      await expect(runVerifiedOpenCode(buildConfig(), opts, deps)).rejects.toThrow('Pass a task');
    }
    const dir = await temp();
    for (const tail of [[], ['--', ' ', '\n']]) {
      const child = Bun.spawnSync([
        process.execPath,
        path.join(import.meta.dir, 'setup_opencode.mjs'),
        'run',
        '--project',
        path.join(dir, 'uncreated'),
        '--verify',
        'must-not-run',
        ...tail,
      ]);
      expect(child.exitCode).toBe(1);
      expect(child.stderr.toString()).toContain('Pass a task after --');
      expect(await readdir(dir)).toEqual([]);
    }
  });

  test('rejects deadlines that overflow native timers before starting processes', async () => {
    const deps = {
      runFlags: [],
      fetchImpl: noSessionCacheFetch,
      spawnImpl: (() => {
        throw new Error('process started');
      }) as any,
      runModel: async () => {
        throw new Error('model started');
      },
    };
    for (const timeoutMs of [2147483648, Number.MAX_SAFE_INTEGER]) {
      const opts = { project: testProject, prompt: ['task'], verify: 'check', timeoutMs };
      await expect(runOpenCode(buildConfig(), opts, deps)).rejects.toThrow('Timeout must');
      await expect(runVerification('check', opts, deps)).rejects.toThrow('Timeout must');
      await expect(runVerifiedOpenCode(buildConfig(), opts, deps)).rejects.toThrow('Timeout must');
      await expect(
        runVerifiedOpenCode(
          buildConfig(),
          { ...opts, timeoutMs: 1000, roundTimeoutMs: timeoutMs },
          deps,
        ),
      ).rejects.toThrow('Round timeout must');
    }
  });

  test('stops a stalled CLI immediately after its JSON API error and never verifies it', async () => {
    const signals: string[] = [];
    let checked = false;
    let deliveringError = false;
    let stoppedDuringError = false;
    const code = await runVerifiedOpenCode(
      buildConfig(),
      { project: testProject, prompt: ['task'], verify: 'check', timeoutMs: 1000 },
      {
        env: {},
        stdout: quiet,
        stderr: quiet,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.kill = (signal) => {
            signals.push(signal);
            stoppedDuringError = deliveringError;
            queueMicrotask(() => child.emit('close', 0, null));
          };
          queueMicrotask(() => {
            child.stdout.emit('data', Buffer.from('{"type":"er'));
            deliveringError = true;
            child.stdout.emit('data', Buffer.from('ror","error":{"name":"APIError"}}\n'));
            deliveringError = false;
            // A broken CLI stays alive after reporting the fatal error.
          });
          return child;
        }) as any,
        runCheck: async () => {
          checked = true;
          return { code: 0, output: 'pass' };
        },
      },
    );
    expect(code).toBe(1);
    expect(signals).toEqual(['SIGTERM']);
    expect(checked).toBe(false);
    expect(stoppedDuringError).toBe(true);
  });

  test('does not accept verification finishing after the overall deadline', async () => {
    let now = 0;
    for (const checkCode of [0, 7, 130, 143]) {
      now = 0;
      const code = await runVerifiedOpenCode(
        buildConfig(),
        { project: testProject, prompt: ['task'], verify: 'check', timeoutMs: 1000 },
        {
          now: () => now,
          stderr: quiet,
          runModel: async () => {
            now = 500;
            return 0;
          },
          runCheck: async (_command, opts) => {
            expect(opts.timeoutMs).toBe(500);
            now = 1001;
            return { code: checkCode, output: 'check finished' };
          },
        },
      );
      expect(code).toBe([130, 143].includes(checkCode) ? checkCode : 124);
    }
  });

  test.skipIf(process.platform === 'win32')(
    'a real CLI reporting an API error cannot continue tool work',
    async () => {
      const dir = await temp();
      const fixture = path.join(dir, 'opencode-error-fixture');
      await writeFile(
        fixture,
        `#!${process.execPath}
import { writeFileSync } from 'node:fs';
if (process.argv.includes('--help')) { console.log('--model --format'); process.exit(0); }
process.on('SIGTERM', () => process.exit(0));
console.log(JSON.stringify({ type: 'error', error: { name: 'APIError' } }));
setTimeout(() => writeFileSync('continued-after-error', 'unexpected'), 1000);
setInterval(() => {}, 1000);
`,
        { mode: 0o700 },
      );
      let checked = false;
      const code = await runVerifiedOpenCode(
        buildConfig(),
        { project: dir, prompt: ['task'], verify: 'check', timeoutMs: 10000, roundTimeoutMs: 3000 },
        {
          env: { ...process.env, OPENCODE_BIN: fixture },
          stdout: quiet,
          stderr: quiet,
          runCheck: async () => {
            checked = true;
            return { code: 0, output: 'pass' };
          },
        },
      );
      expect(code).toBe(1);
      expect(checked).toBe(false);
      expect(await Bun.file(path.join(dir, 'continued-after-error')).exists()).toBe(false);
    },
  );

  test('feeds a failing check into the same session and accepts only a passing check', async () => {
    const config = buildConfig({ steps: 64 });
    const before = JSON.stringify(config);
    const rounds: any[] = [];
    let checks = 0;
    const code = await runVerifiedOpenCode(
      config,
      {
        project: testProject,
        prompt: ['Implement the feature'],
        verify: 'project-tests',
        attempts: 3,
      },
      {
        stderr: quiet,
        runModel: async (live, opts, deps) => {
          rounds.push({ live, opts });
          deps.onEvent({ sessionID: 'ses_verified' });
          return 0;
        },
        runCheck: async (command) => {
          expect(command).toBe('project-tests');
          checks++;
          return checks === 1
            ? { code: 1, output: 'compiler: missing route implementation' }
            : { code: 0, output: 'tests passed' };
        },
      },
    );
    expect(code).toBe(0);
    expect(checks).toBe(2);
    expect(rounds).toHaveLength(2);
    expect(rounds[0].live.agent.zinc.steps).toBe(8);
    expect(rounds[1].opts.session).toBe('ses_verified');
    expect(rounds[1].opts.prompt[0]).toContain('compiler: missing route implementation');
    expect(rounds[1].opts.prompt[0]).toContain('Implement the feature');
    expect(JSON.stringify(config)).toBe(before);
  });

  test('never turns failed checks or model errors into success', async () => {
    const opts = { project: testProject, prompt: ['task'], verify: 'check', attempts: 2 };
    let checks = 0;
    const failing = await runVerifiedOpenCode(buildConfig(), opts, {
      stderr: quiet,
      runModel: async (_config, _opts, deps) => {
        deps.onEvent({ sessionID: 'ses_failed' });
        return 0;
      },
      runCheck: async () => {
        checks++;
        return { code: 7, output: 'failed assertion' };
      },
    });
    expect(failing).toBe(7);
    expect(checks).toBe(2);
    const modelError = await runVerifiedOpenCode(buildConfig(), opts, {
      stderr: quiet,
      runModel: async () => 1,
      runCheck: async () => {
        throw new Error('verification must not mask an API error');
      },
    });
    expect(modelError).toBe(1);
  });

  test('reserves verification time and checks partial work after an editing timeout', async () => {
    const rounds: any[] = [];
    let checks = 0;
    let progress = '';
    const code = await runVerifiedOpenCode(
      buildConfig(),
      {
        project: testProject,
        prompt: ['Implement the feature'],
        verify: 'acceptance-check',
        timeoutMs: 10000,
        roundTimeoutMs: 10000,
      },
      {
        stderr: {
          write: (text) => {
            progress += text;
          },
        } as any,
        runModel: async (_config, opts, deps) => {
          rounds.push(opts);
          deps.onEvent({ sessionID: 'ses_timed_round' });
          return rounds.length === 1 ? 124 : 0;
        },
        runCheck: async (_command, opts) => {
          expect(opts.timeoutMs).toBeGreaterThan(0);
          return ++checks === 1
            ? { code: 7, output: 'error: missing implementation' }
            : { code: 0, output: 'accepted' };
        },
      },
    );
    expect(code).toBe(0);
    expect(checks).toBe(2);
    expect(rounds[0].timeoutMs).toBeLessThanOrEqual(7500);
    expect(rounds[0].timeoutMs).toBeGreaterThan(0);
    expect(rounds[1].session).toBe('ses_timed_round');
    expect(rounds[1].prompt[0]).toContain('reached its time limit');
    expect(rounds[1].prompt[0]).toContain('missing implementation');
    expect(progress).toContain('timed out; checking partial work');
    expect(progress).toContain('Verification round 2/3: passed');
  });

  test('bounds individual rounds and cannot accept failed or cancelled verification', async () => {
    for (const [checkCode, expected] of [
      [7, 124],
      [124, 124],
      [130, 130],
      [143, 143],
    ]) {
      let checks = 0;
      const code = await runVerifiedOpenCode(
        buildConfig(),
        {
          project: testProject,
          prompt: ['task'],
          verify: 'check',
          attempts: 1,
          timeoutMs: 10000,
          roundTimeoutMs: 300,
        },
        {
          stderr: quiet,
          runModel: async (_config, opts, deps) => {
            expect(opts.timeoutMs).toBe(300);
            deps.onEvent({ sessionID: 'ses_timed_round' });
            return 124;
          },
          runCheck: async () => {
            checks++;
            return { code: checkCode, output: 'failed' };
          },
        },
      );
      expect(code).toBe(expected);
      expect(checks).toBe(1);
    }
    for (const code of [1, 130, 143]) {
      expect(
        await runVerifiedOpenCode(
          buildConfig(),
          { project: testProject, prompt: ['task'], verify: 'check' },
          {
            stderr: quiet,
            runModel: async () => code,
            runCheck: async () => {
              throw new Error('must not verify an API error or cancelled run');
            },
          },
        ),
      ).toBe(code);
    }
  });

  test('uses a five-minute default and rejects invalid round deadlines', async () => {
    expect(
      parseArgs(['run', '--round-timeout-ms', '120000', '--verify', 'check']).roundTimeoutMs,
    ).toBe(120000);
    const code = await runVerifiedOpenCode(
      buildConfig(),
      { project: testProject, prompt: ['task'], verify: 'check', timeoutMs: 900000 },
      {
        stderr: quiet,
        runModel: async (_config, opts) => {
          expect(opts.timeoutMs).toBe(300000);
          return 1;
        },
      },
    );
    expect(code).toBe(1);
    for (const roundTimeoutMs of [0, -1, NaN, 1.5, Infinity])
      await expect(
        runVerifiedOpenCode(buildConfig(), {
          project: testProject,
          prompt: ['task'],
          verify: 'check',
          roundTimeoutMs,
        }),
      ).rejects.toThrow('Round timeout');
  });

  test('CLI rejects round deadlines without verification before writing any files', async () => {
    const dir = await temp();
    for (const args of [['setup'], ['check'], ['run']]) {
      const child = Bun.spawnSync([
        process.execPath,
        path.join(import.meta.dir, 'setup_opencode.mjs'),
        ...args,
        '--project',
        path.join(dir, 'uncreated'),
        '--round-timeout-ms',
        '100',
      ]);
      expect(child.exitCode).toBe(1);
      expect(child.stderr.toString()).toMatch(
        /Verification options|Round timeout requires --verify/,
      );
      expect(await readdir(dir)).toEqual([]);
    }
  });

  test.skipIf(process.platform === 'win32')(
    'checks a real partial edit after timeout and stops background tools',
    async () => {
      const dir = await temp();
      const fixture = path.join(dir, 'opencode-fixture');
      await writeFile(
        fixture,
        `#!${process.execPath}
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
if (process.argv.includes('--help')) { console.log('--model --format'); process.exit(0); }
const child = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000);'], { stdio: 'ignore' });
writeFileSync('tool.pid', String(child.pid));
writeFileSync('feature.mjs', 'export const answer = 42;\\n');
console.log(JSON.stringify({ type: 'step_start', sessionID: 'ses_process_fixture' }));
process.on('SIGTERM', () => process.exit(0));
setInterval(() => {}, 1000);
`,
        { mode: 0o700 },
      );
      let toolPid: number | undefined;
      try {
        const code = await runVerifiedOpenCode(
          buildConfig(),
          {
            project: dir,
            prompt: ['task'],
            timeoutMs: 10000,
            roundTimeoutMs: 1000,
            verify: `'${process.execPath.replaceAll("'", "'\\''")}' -e 'import { answer } from "./feature.mjs"; if (answer !== 42) process.exit(7);'`,
          },
          { env: { ...process.env, OPENCODE_BIN: fixture }, stdout: quiet, stderr: quiet },
        );
        expect(code).toBe(0);
        toolPid = Number(await readFile(path.join(dir, 'tool.pid'), 'utf8'));
        expect(toolPid).toBeGreaterThan(0);
        // A reparented Linux zombie is already stopped and may await init's reap.
        const status = Bun.spawnSync(['ps', '-p', String(toolPid), '-o', 'stat=']);
        const state = status.stdout.toString().trim();
        expect(state === '' || state.startsWith('Z')).toBe(true);
      } finally {
        if (toolPid) {
          try {
            process.kill(toolPid, 'SIGKILL');
          } catch {}
        }
      }
    },
  );

  test('cancellation and API errors take priority over an editing timeout', async () => {
    for (const signal of ['SIGINT', 'SIGTERM'] as const) {
      const spawnImpl = (() => {
        const child: any = new EventEmitter();
        let interrupted = false;
        child.kill = () => {
          if (!interrupted) {
            interrupted = true;
            process.emit(signal);
          }
          queueMicrotask(() => child.emit('close', 0, null));
        };
        return child;
      }) as any;
      expect(
        await runOpenCode(
          buildConfig(),
          { project: testProject, prompt: ['task'], timeoutMs: 10 },
          { runFlags: [], spawnImpl, env: {}, stdout: quiet },
        ),
      ).toBe(signal === 'SIGINT' ? 130 : 143);
      expect(
        (
          await runVerification(
            'check',
            { project: testProject, timeoutMs: 10 },
            { runFlags: [], spawnImpl, stderr: quiet },
          )
        ).code,
      ).toBe(signal === 'SIGINT' ? 130 : 143);
    }
    const code = await runOpenCode(
      buildConfig(),
      { project: testProject, prompt: ['task'], timeoutMs: 10 },
      {
        env: {},
        stdout: quiet,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.kill = () => {
            child.stdout.emit('data', Buffer.from('{"type":"error"}\n'));
            queueMicrotask(() => child.emit('close', 0, null));
          };
          return child;
        }) as any,
      },
    );
    expect(code).toBe(1);
  });

  test.skipIf(process.platform === 'win32')(
    'a successful verifier cannot leave background processes running',
    async () => {
      const dir = await temp();
      await writeFile(
        path.join(dir, 'check.mjs'),
        `
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const child = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000);'], { stdio: 'ignore' });
writeFileSync('tool.pid', String(child.pid));
process.exit(0);
`,
      );
      let toolPid: number | undefined;
      try {
        const checked = await runVerification(
          `'${process.execPath.replaceAll("'", "'\\''")}' check.mjs`,
          { project: dir, timeoutMs: 3000 },
          { stderr: quiet },
        );
        toolPid = Number(await readFile(path.join(dir, 'tool.pid'), 'utf8'));
        expect(checked.code).toBe(0);
        const status = Bun.spawnSync(['ps', '-p', String(toolPid), '-o', 'stat=']);
        const state = status.stdout.toString().trim();
        expect(state === '' || state.startsWith('Z')).toBe(true);
      } finally {
        if (toolPid) {
          try {
            process.kill(toolPid, 'SIGKILL');
          } catch {}
        }
      }
    },
  );

  test('captures real command failure output without running OpenCode', async () => {
    const dir = await temp();
    const checked = await runVerification(
      'printf "compiler diagnostic"; exit 7',
      { project: dir, timeoutMs: 5000 },
      { stderr: quiet },
    );
    expect(checked.code).toBe(7);
    expect(checked.output).toBe('compiler diagnostic');
  });

  test('bounds feedback while retaining the first diagnostic and final summary', async () => {
    const checked = await runVerification(
      'literal command',
      { project: testProject, timeoutMs: 1000 },
      {
        stderr: quiet,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stdout = new EventEmitter();
          child.stderr = new EventEmitter();
          queueMicrotask(() => {
            child.stderr.emit(
              'data',
              Buffer.from('FIRST ERROR\n' + 'x'.repeat(20000) + '\nFINAL SUMMARY'),
            );
            child.emit('close', 1, null);
          });
          return child;
        }) as any,
      },
    );
    expect(checked.output).toStartWith('FIRST ERROR');
    expect(checked.output).toEndWith('FINAL SUMMARY');
    expect(checked.output.length).toBeLessThan(8100);
  });

  test('terminates a timed-out verifier and preserves timeout exit status', async () => {
    const signals: string[] = [];
    const checked = await runVerification(
      'checks',
      { project: testProject, timeoutMs: 10 },
      {
        stderr: quiet,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.kill = (signal: string) => {
            signals.push(signal);
            queueMicrotask(() => child.emit('close', null, signal));
          };
          return child;
        }) as any,
      },
    );
    expect(checked.code).toBe(124);
    expect(signals).toEqual(['SIGTERM']);
    expect(
      parseArgs(['run', '--verify', 'bun test', '--attempts', '2', '--', 'task']).attempts,
    ).toBe(2);
  });

  test('retains a failure in the middle of a long passing-test log', async () => {
    const checked = await runVerification(
      'project-tests',
      { project: testProject, timeoutMs: 1000 },
      {
        stderr: quiet,
        runFlags: [],
        fetchImpl: noSessionCacheFetch,
        spawnImpl: (() => {
          const child: any = new EventEmitter();
          child.stderr = new EventEmitter();
          queueMicrotask(() => {
            child.stderr.emit('data', Buffer.from('(pass) unrelated test\n'.repeat(1000)));
            child.stderr.emit(
              'data',
              Buffer.from('module.zig:9: error: missing module documentation\n'),
            );
            child.stderr.emit('data', Buffer.from('expected: a module description\n'));
            child.stderr.emit('data', Buffer.from('(pass) later unrelated test\n'.repeat(1000)));
            child.emit('close', 1, null);
          });
          return child;
        }) as any,
      },
    );
    expect(checked.code).toBe(1);
    expect(checked.output).toContain('module.zig:9: error: missing module documentation');
    expect(checked.output).toContain('expected: a module description');
    expect(checked.output.length).toBeLessThan(8100);
  });

  test('an interrupted child exiting zero cannot produce success', async () => {
    for (const [signal, expected] of [
      ['SIGINT', 130],
      ['SIGTERM', 143],
    ] as const) {
      const spawnImpl = (() => {
        const child: any = new EventEmitter();
        child.kill = () => queueMicrotask(() => child.emit('close', 0, null));
        queueMicrotask(() => process.emit(signal));
        return child;
      }) as any;
      const checked = await runVerification(
        'checks',
        { project: testProject, timeoutMs: 1000 },
        {
          runFlags: [],
          spawnImpl,
          stderr: quiet,
        },
      );
      expect(checked.code).toBe(expected);
      const code = await runOpenCode(
        buildConfig(),
        {
          project: testProject,
          prompt: ['task'],
          timeoutMs: 1000,
        },
        { runFlags: [], spawnImpl, stdout: quiet, env: {} },
      );
      expect(code).toBe(expected);
    }
  });
});

describe('connection checks', () => {
  test('checks JSON and streaming tool completions rather than health alone', async () => {
    const seen: any[] = [];
    const tool = {
      id: 'call_probe',
      type: 'function',
      function: { name: 'zinc_probe', arguments: '{"message":"OK"}' },
    };
    const result = await checkServer(buildConfig(), {
      env: {},
      fetchImpl: (async (_url: string, opts: any) => {
        const body = opts.body ? JSON.parse(opts.body) : null;
        seen.push(body);
        if (!body) return Response.json({ status: 'ok' });
        if (!body.tools)
          return Response.json({
            choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
          });
        if (!body.stream)
          return Response.json({
            choices: [{ message: { tool_calls: [tool] }, finish_reason: 'tool_calls' }],
          });
        return new Response(
          `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, ...tool }] } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })}\n\ndata: [DONE]\n\n`,
        );
      }) as any,
    });
    expect(result.streaming_tools).toBe('pass');
    expect(seen.length).toBe(4);
    expect(seen.at(-1).stream).toBe(true);
  });
  test('fails clearly when the server answers chat but cannot return tool calls', async () => {
    await expect(
      checkServer(buildConfig(), {
        env: {},
        fetchImpl: (async (_url: string, opts: any) =>
          opts.body
            ? Response.json({
                choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
              })
            : Response.json({ status: 'ok' })) as any,
      }),
    ).rejects.toThrow('invalid tool completion');
  });
  test('reports HTTP status without reflecting an endpoint or response secret', async () => {
    await expect(
      checkServer(buildConfig(), {
        env: {},
        fetchImpl: (async () => new Response('private response', { status: 503 })) as any,
      }),
    ).rejects.toThrow('ZINC health check returned HTTP 503');
  });
});
